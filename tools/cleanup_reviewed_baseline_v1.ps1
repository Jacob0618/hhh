param([Parameter(Mandatory=$true)][ValidateSet('amem','simplemem','hypermem','vmem','prism')][string]$Method)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath 'D:\jev mem').Path
$methodRoot = (Resolve-Path -LiteralPath (Join-Path $workspace "runs\five_baselines_gemini25_v1_20261008\$Method")).Path
$planPath = Join-Path $methodRoot 'cleanup_plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.method -ne $Method -or $plan.schema -ne 'baseline-reviewed-cleanup-plan-v1' -or -not $plan.archive_crc_verified) { throw 'Invalid reviewed cleanup plan' }
$receipt = Join-Path $methodRoot 'results\cleanup_receipt.json'
if (Test-Path -LiteralPath $receipt) { throw 'Cleanup receipt already exists; inspect it before any action' }
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match "run_five_baselines_gemini25_v1\.py.*\brun\b.*--method[ =]+$Method\b" }
if ($active) { throw 'Experiment process still active' }
foreach ($prop in $plan.keep_sha256.PSObject.Properties) {
    $keep = (Resolve-Path -LiteralPath (Join-Path $methodRoot $prop.Name)).Path
    if (-not $keep.StartsWith($methodRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Retained path escaped method root' }
    if ((Get-FileHash -LiteralPath $keep -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prop.Value) { throw "Retained artifact changed: $keep" }
}
if ((Get-FileHash -LiteralPath $plan.result_review.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.result_review.sha256) { throw 'Independent review changed' }
$targets = @($plan.allowed_delete_roots)
if ($targets.Count -ne 1) { throw 'Exactly one new method work directory is allowed' }
$target = (Resolve-Path -LiteralPath $targets[0]).Path
$expected = Join-Path $methodRoot 'work'
if ($target -ne $expected -or -not $target.StartsWith($workspace + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion target outside exact new method work directory' }
$items = @(Get-Item -LiteralPath $target) + @(Get-ChildItem -LiteralPath $target -Force -Recurse)
if ($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw 'Reparse point in cleanup target' }
$bytes = ($items | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum
$planHash = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
Remove-Item -LiteralPath $target -Recurse -Force
if (Test-Path -LiteralPath $target) { throw 'Deletion did not complete' }
$record = @{schema='baseline-cleanup-receipt-v1'; method=$Method; completed_utc=[DateTime]::UtcNow.ToString('o'); removed_path=$target; removed_bytes=$bytes; cleanup_plan_sha256=$planHash; kept_results_verified=$true}
$json = $record | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($receipt, $json, [System.Text.UTF8Encoding]::new($false))
Write-Output $json
