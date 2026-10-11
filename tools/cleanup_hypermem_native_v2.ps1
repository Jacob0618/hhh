param([Parameter(Mandatory=$true)][string]$CleanupReview)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath 'D:\jev mem').Path
$out = Join-Path $workspace 'research\baseline_execution_2026-10-08\hypermem_native_v2_lifecycle'
$planPath = Join-Path $out 'cleanup_plan.json'
$intentPath = Join-Path $out 'cleanup_execution_intent.json'
$python = Join-Path $workspace 'runtime\baseline_env_v1\Scripts\python.exe'
$helper = Join-Path $workspace 'tools\archive_hypermem_native_v2.py'
$expected = @(
    (Join-Path $workspace 'runs\five_baselines_gemini25_v1_20261008\hypermem\work'),
    (Join-Path $workspace 'runs\five_baselines_gemini25_v1_20261008\hypermem_native_v2\work')
)
function FileHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path); $algorithm = [Security.Cryptography.SHA256]::Create()
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
function AssertLiteralChain([string]$Path) {
    if (-not $Path.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escaped workspace' }
    $current = Get-Item -LiteralPath $Path -Force
    while ($true) {
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in path or ancestor' }
        if ($current.FullName -eq $workspace) { break }
        $current = if ($current -is [IO.DirectoryInfo]) { $current.Parent } else { $current.Directory }
        if ($null -eq $current) { throw 'Workspace ancestor not reached' }
    }
}
function AssertUnderWorkspace([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path) -or $Path -match '(^|[\\/])\.\.?([\\/]|$)') { throw 'Absolute literal path required' }
    $literal = [IO.Path]::GetFullPath($Path)
    if ($literal -ne $Path) { throw 'Noncanonical input path' }
    # Check the caller's literal ancestor chain BEFORE following any alias.
    AssertLiteralChain $literal
    $resolved = (Resolve-Path -LiteralPath $literal).Path
    AssertLiteralChain $resolved
    return $resolved
}
function WriteExclusive([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12))
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
AssertStopped
$planPath = AssertUnderWorkspace $planPath; $CleanupReview = AssertUnderWorkspace $CleanupReview
foreach ($name in @('cleanup_execution_intent.json','removal_original.json','removal_continuation.json','cleanup_receipt.json','predecessor_completion.json')) {
    if (Test-Path -LiteralPath (Join-Path $out $name)) { throw 'Cleanup already started; no implicit retry' }
}
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$review = Get-Content -LiteralPath $CleanupReview -Raw -Encoding UTF8 | ConvertFrom-Json
$planHash = FileHash $planPath; $reviewHash = FileHash $CleanupReview
if ($plan.schema -ne 'hypermem-native-v2-two-root-cleanup-plan-v1' -or $plan.method -ne 'hypermem' -or
    $plan.workspace -ne $workspace -or $plan.archive_crc_verified -ne $true) { throw 'Wrong cleanup contract' }
if ($review.schema -ne 'hypermem-native-v2-cleanup-review-v1' -or $review.status -ne 'passed' -or
    $review.cleanup_plan_sha256 -ne $planHash -or $review.archive_sha256 -ne $plan.archive.sha256 -or
    $review.scope -ne $plan.scope) { throw 'Independent archive review differs' }
if ($plan.allowed_delete_roots.Count -ne 2 -or $review.allowed_delete_roots.Count -ne 2) { throw 'Exactly two reviewed roots required' }
$targets = @()
for ($i=0; $i -lt 2; $i++) {
    $target = AssertUnderWorkspace $plan.allowed_delete_roots[$i]
    if ($target -ne $expected[$i] -or $review.allowed_delete_roots[$i] -ne $target) { throw 'Not the exact original and continuation work roots' }
    $targets += $target
}
& $python $helper verify
if ($LASTEXITCODE -ne 0) { throw 'Offline archive verification failed; nothing deleted' }
$archivePath = AssertUnderWorkspace $plan.archive_manifest.path
if ((FileHash $archivePath) -ne $plan.archive_manifest.sha256) { throw 'Manifest changed' }
$archive = Get-Content -LiteralPath $archivePath -Raw -Encoding UTF8 | ConvertFrom-Json
$names = @('original','continuation'); $sizes = @()
for ($i=0; $i -lt 2; $i++) {
    $items = @(Get-Item -LiteralPath $targets[$i] -Force) + @(Get-ChildItem -LiteralPath $targets[$i] -Recurse -Force)
    if ($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw 'Reparse inside work' }
    $files = @($items | Where-Object { -not $_.PSIsContainer })
    $sealed = $archive.work_inventories.PSObject.Properties[$names[$i]].Value
    if ($files.Count -ne @($sealed.PSObject.Properties).Count) { throw 'Work file set changed' }
    foreach ($file in $files) {
        $record = $sealed.PSObject.Properties[$file.FullName].Value
        if (-not $record -or (FileHash $file.FullName) -ne $record.sha256 -or $file.Length -ne $record.size) { throw 'Work bytes changed' }
    }
    $sizes += ($files | Measure-Object Length -Sum).Sum
}
foreach ($prop in $plan.keep_sha256.PSObject.Properties) {
    $keep = AssertUnderWorkspace $prop.Name
    foreach ($target in $targets) { if ($keep.StartsWith($target + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Keep within deletion target' } }
    if ((FileHash $keep) -ne $prop.Value) { throw 'Permanent evidence changed' }
}
foreach ($prop in $plan.code_sha256.PSObject.Properties) {
    $source = AssertUnderWorkspace (Join-Path $workspace $prop.Name)
    if ((FileHash $source) -ne $prop.Value) { throw 'Tool source changed' }
}
if ((FileHash $planPath) -ne $planHash -or (FileHash $CleanupReview) -ne $reviewHash) { throw 'Plan or review changed' }
AssertStopped
WriteExclusive $intentPath @{schema='hypermem-native-v2-cleanup-intent-v1'; targets=$targets; created_utc=[DateTime]::UtcNow.ToString('o'); cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash}
for ($i=0; $i -lt 2; $i++) {
    # Each literal absolute target and its ancestor chain was validated above.
    Remove-Item -LiteralPath $targets[$i] -Recurse -Force
    if (Test-Path -LiteralPath $targets[$i]) { throw 'Work root remains' }
    WriteExclusive (Join-Path $out ('removal_' + $names[$i] + '.json')) @{schema='hypermem-native-v2-removal-receipt-v1'; removed_path=$targets[$i]; removed_bytes=$sizes[$i]; completed_utc=[DateTime]::UtcNow.ToString('o'); cleanup_plan_sha256=$planHash; cleanup_review_sha256=$reviewHash}
}
& $python $helper complete --cleanup-review $CleanupReview
if ($LASTEXITCODE -ne 0) { throw 'Postdelete proof failed; no launch completion' }
