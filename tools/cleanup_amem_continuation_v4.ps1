param([Parameter(Mandatory=$true)][string]$CleanupReview)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath 'D:\jev mem').Path
$runRoot = Join-Path $workspace 'runs\five_baselines_gemini25_v1_20261008'
$names = @('amem', 'amem_format_v2', 'amem_native_v3', 'amem_native_v4')
$aliases = @('v1', 'v2', 'v3', 'v4')
$finalRoot = Join-Path $runRoot 'amem_native_v4'
$planPath = Join-Path $finalRoot 'cleanup_plan_native_v4.json'
$archiveRoot = Join-Path $finalRoot 'results\continuation_archive'
$intentPath = Join-Path $archiveRoot 'cleanup_execution_intent.json'
$python = Join-Path $workspace 'runtime\baseline_env_v1\Scripts\python.exe'
$helper = Join-Path $workspace 'tools\archive_amem_continuation_v4.py'

function FileHash([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function AssertStopped {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(?:w|[0-9]+(?:\.[0-9]+)*)?\.exe$' -and
        $_.CommandLine -match 'run_five_baselines[\w]*(?:\.py)?(?:["\s]|$)' -and
        $_.CommandLine -match '(?:^|["\s])run(?:["\s]|$)'
    }
    if ($active) { throw 'A baseline controller is still active' }
}
function WriteExclusive([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12))
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
function AssertUnderWorkspace([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escaped workspace' }
    $current = Get-Item -LiteralPath $resolved -Force
    while ($true) {
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in target/ancestor' }
        if ($current.FullName -eq $workspace) { break }
        $current = if ($current -is [IO.FileInfo]) { $current.Directory } else { $current.Parent }
        if ($null -eq $current) { throw 'Workspace ancestor not reached' }
    }
    return $resolved
}

AssertStopped
if ((Test-Path -LiteralPath $intentPath) -or (Test-Path -LiteralPath (Join-Path $finalRoot 'results\cleanup_receipt.json'))) {
    throw 'Cleanup previously started/completed; no implicit reentry'
}
$planPath = AssertUnderWorkspace $planPath
$CleanupReview = AssertUnderWorkspace $CleanupReview
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$review = Get-Content -LiteralPath $CleanupReview -Raw -Encoding UTF8 | ConvertFrom-Json
$planHash = FileHash $planPath
$reviewHash = FileHash $CleanupReview
if ($plan.schema -ne 'amem-four-root-cleanup-plan-v1' -or $plan.workspace -ne $workspace -or -not $plan.archive_crc_verified) { throw 'Invalid exact cleanup plan' }
if ($review.status -ne 'passed' -or $review.cleanup_plan_sha256 -ne $planHash -or $review.archive_sha256 -ne $plan.archive.sha256) { throw 'Independent archive/cleanup review differs' }
if ($plan.allowed_delete_roots.Count -ne 4) { throw 'Exactly four approved work roots required' }

# This independently verifies every ZIP CRC/blob SHA, keep SHA, source inventory,
# result review and all 1540 archived firstpass/effective exact-raw QA replays.
& $python $helper verify --plan $planPath
if ($LASTEXITCODE -ne 0) { throw 'Archive resolver verification failed; nothing deleted' }

$verifiedTargets = @()
for ($i=0; $i -lt 4; $i++) {
    $expected = Join-Path (Join-Path $runRoot $names[$i]) 'work'
    $target = AssertUnderWorkspace $plan.allowed_delete_roots[$i]
    if ($target -ne $expected) { throw 'Delete target differs from exact named work root' }
    $items = @(Get-Item -LiteralPath $target -Force) + @(Get-ChildItem -LiteralPath $target -Force -Recurse)
    if ($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw 'Reparse point inside work root' }
    $files = @($items | Where-Object { -not $_.PSIsContainer })
    $wanted = $plan.work_inventories.($aliases[$i])
    if ($files.Count -ne @($wanted.PSObject.Properties).Count) { throw 'Work file inventory count changed' }
    foreach ($item in $files) {
        $hash = $wanted.PSObject.Properties[$item.FullName].Value
        if (-not $hash -or (FileHash $item.FullName) -ne $hash) { throw 'Work file changed before deletion' }
    }
    $verifiedTargets += @{path=$target; bytes=($files | Measure-Object Length -Sum).Sum; alias=$aliases[$i]}
}
foreach ($prop in $plan.keep_sha256.PSObject.Properties) {
    $keep = AssertUnderWorkspace $prop.Name
    if ((FileHash $keep) -ne $prop.Value) { throw "Retained artifact changed: $keep" }
}
if ((FileHash $planPath) -ne $planHash -or (FileHash $CleanupReview) -ne $reviewHash) { throw 'Plan/review changed during verification' }
AssertStopped
WriteExclusive $intentPath @{schema='amem-four-root-cleanup-intent-v1'; created_utc=[DateTime]::UtcNow.ToString('o'); cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash; targets=$verifiedTargets}
foreach ($item in $verifiedTargets) {
    AssertStopped
    # A single native PowerShell deletion; the absolute literal target was
    # checked against its exact allowed work root before this loop.
    Remove-Item -LiteralPath $item.path -Recurse -Force
    if (Test-Path -LiteralPath $item.path) { throw 'Work deletion did not complete' }
    WriteExclusive (Join-Path $archiveRoot ("cleanup_" + $item.alias + '.json')) @{
        schema='amem-four-root-removal-receipt-v1'; completed_utc=[DateTime]::UtcNow.ToString('o');
        removed_path=$item.path; removed_bytes=$item.bytes; cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash
    }
}
& $python $helper complete --plan $planPath --cleanup-review $CleanupReview
if ($LASTEXITCODE -ne 0) { throw 'Post-delete archived replay/completion failed; next-method gate remains closed' }
