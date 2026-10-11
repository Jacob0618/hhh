param(
    [Parameter(Mandatory=$true)][ValidateSet('simplemem','hypermem','vmem','prism')][string]$Method,
    [Parameter(Mandatory=$true)][string]$CleanupReview
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath 'D:\jev mem').Path
$methodRoot = Join-Path $workspace "runs\five_baselines_gemini25_v1_20261008\$Method"
$planPath = Join-Path $methodRoot 'cleanup_plan_chain_v3.json'
$out = Join-Path $methodRoot 'results\chain_lifecycle'
$intentPath = Join-Path $out 'cleanup_execution_intent.json'
$receiptPath = Join-Path $out 'removal_receipt.json'
$python = Join-Path $workspace 'runtime\baseline_env_v1\Scripts\python.exe'
$helper = Join-Path $workspace 'tools\baseline_lifecycle_v3.py'

function FileHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}
function AssertStopped {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?(\d+(\.\d+)*)?\.exe$' -and
        $_.CommandLine -match 'run_five_baselines[\w]*(\.py)?(["\s]|$)' -and
        $_.CommandLine -match '(^|["\s])run(["\s]|$)'
    }
    if ($active) { throw 'A baseline controller is still active' }
}
function AssertUnderWorkspace([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escaped workspace' }
    $current = Get-Item -LiteralPath $resolved -Force
    while ($true) {
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in target or ancestor' }
        if ($current.FullName -eq $workspace) { break }
        $current = if ($current -is [IO.DirectoryInfo]) { $current.Parent } else { $current.Directory }
        if ($null -eq $current) { throw 'Workspace ancestor not reached' }
    }
    return $resolved
}
function WriteExclusive([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12))
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}

AssertStopped
$planPath = AssertUnderWorkspace $planPath
$CleanupReview = AssertUnderWorkspace $CleanupReview
if ((Test-Path -LiteralPath $intentPath) -or (Test-Path -LiteralPath $receiptPath) -or
    (Test-Path -LiteralPath (Join-Path $methodRoot 'results\cleanup_receipt.json'))) { throw 'Cleanup already started; no implicit retry' }
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$review = Get-Content -LiteralPath $CleanupReview -Raw -Encoding UTF8 | ConvertFrom-Json
$planHash = FileHash $planPath
$reviewHash = FileHash $CleanupReview
if ($plan.schema -ne 'baseline-single-root-v3-cleanup-plan-v1' -or $plan.method -ne $Method -or
    $plan.workspace -ne $workspace -or -not $plan.archive_crc_verified) { throw 'Wrong cleanup scope' }
if ($review.status -ne 'passed' -or $review.method -ne $Method -or $review.cleanup_plan_sha256 -ne $planHash -or
    $review.archive_sha256 -ne $plan.archive.sha256) { throw 'Independent archive/cleanup review differs' }
if ($plan.allowed_delete_roots.Count -ne 1) { throw 'Exactly one method work root required' }
$expected = Join-Path $methodRoot 'work'
$target = AssertUnderWorkspace $plan.allowed_delete_roots[0]
if ($target -ne $expected) { throw 'Target is not this exact new method work root' }

# Pure verification checks terminal status, manifest/source/review, every ZIP
# CRC/blob SHA, keep SHA, work seal, and all 3080 raw QA stage pairs.
& $python $helper verify --method $Method
if ($LASTEXITCODE -ne 0) { throw 'Offline evidence verification failed; nothing deleted' }
$manifestPath = AssertUnderWorkspace $plan.archive_manifest.path
if ((FileHash $manifestPath) -ne $plan.archive_manifest.sha256) { throw 'Archive manifest changed' }
$archive = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$items = @(Get-Item -LiteralPath $target -Force) + @(Get-ChildItem -LiteralPath $target -Force -Recurse)
if ($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw 'Reparse point inside work root' }
$files = @($items | Where-Object { -not $_.PSIsContainer })
if ($files.Count -ne @($archive.work_inventory.PSObject.Properties).Count) { throw 'Work inventory count changed' }
foreach ($item in $files) {
    $record = $archive.work_inventory.PSObject.Properties[$item.FullName].Value
    if (-not $record -or (FileHash $item.FullName) -ne $record.sha256 -or $item.Length -ne $record.size) { throw 'Work file changed' }
}
foreach ($prop in $plan.keep_sha256.PSObject.Properties) {
    $keep = AssertUnderWorkspace $prop.Name
    if ((FileHash $keep) -ne $prop.Value) { throw 'Retained artifact changed' }
}
foreach ($prop in $plan.code_sha256.PSObject.Properties) {
    $source = AssertUnderWorkspace (Join-Path $workspace $prop.Name)
    if ((FileHash $source) -ne $prop.Value) { throw 'Lifecycle source changed before deletion' }
}
if ((FileHash $planPath) -ne $planHash -or (FileHash $CleanupReview) -ne $reviewHash) { throw 'Plan/review changed' }
AssertStopped
$removedBytes = ($files | Measure-Object Length -Sum).Sum
WriteExclusive $intentPath @{schema='baseline-single-root-v3-cleanup-intent-v1'; method=$Method; created_utc=[DateTime]::UtcNow.ToString('o'); target=$target; cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash}
# Native PowerShell only; the resolved absolute literal target was checked above.
Remove-Item -LiteralPath $target -Recurse -Force
if (Test-Path -LiteralPath $target) { throw 'Work deletion did not complete' }
WriteExclusive $receiptPath @{schema='baseline-single-root-v3-removal-receipt-v1'; method=$Method; completed_utc=[DateTime]::UtcNow.ToString('o'); removed_path=$target; removed_bytes=$removedBytes; cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash}
& $python $helper complete --method $Method --cleanup-review $CleanupReview
if ($LASTEXITCODE -ne 0) { throw 'Post-delete evidence verification failed; launch gate remains closed' }
