$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath('D:\jev mem')
$auditRoot = Join-Path $taskRoot 'research\cleanup_2026-10-05'
$manifestPath = Join-Path $auditRoot 'delete_manifest.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$preserved = Get-Content -Raw -LiteralPath (Join-Path $auditRoot 'preserved_files.json') | ConvertFrom-Json
$archiveProof = Get-Content -Raw -LiteralPath (Join-Path $taskRoot $manifest.archive_required) | ConvertFrom-Json
if ($archiveProof.status -ne 'verified') { throw 'Archive is not verified' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archiveProof.archive).Hash.ToLowerInvariant() -ne $archiveProof.sha256) { throw 'Archive hash mismatch' }
$protected = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $preserved) {
    if (-not (Test-Path -LiteralPath $entry.path -PathType Leaf)) { throw "Preserved file missing: $($entry.path)" }
    $null = $protected.Add([IO.Path]::GetFullPath($entry.path))
    $parentPath = [IO.Path]::GetDirectoryName($entry.path)
    while ($parentPath -and $parentPath.StartsWith($taskRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $null = $protected.Add($parentPath)
        $parentPath = [IO.Path]::GetDirectoryName($parentPath)
    }
}
$allowed = @((Join-Path $taskRoot 'runs'), (Join-Path $taskRoot 'research'))
$obsoleteEnv = Join-Path $taskRoot '.venv-anaconda-failed'
foreach ($entry in $manifest.deletions) {
    $absolute = [IO.Path]::GetFullPath($entry.path)
    $inAllowed = $absolute -eq $obsoleteEnv
    foreach ($prefix in $allowed) { if ($absolute.StartsWith($prefix + '\', [StringComparison]::OrdinalIgnoreCase)) { $inAllowed = $true } }
    if (-not $inAllowed -or $protected.Contains($absolute) -or $absolute.StartsWith($auditRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe deletion target: $absolute" }
    if ((Resolve-Path -LiteralPath $absolute).ProviderPath -ne $absolute) { throw "Resolved path differs: $absolute" }
    $item = Get-Item -Force -LiteralPath $absolute
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse target rejected: $absolute" }
}
$intentPath = Join-Path $auditRoot 'delete_execution_intent.json'
if (Test-Path -LiteralPath $intentPath) { throw 'Cleanup already attempted; review ledger before resuming' }
$startFree = (Get-PSDrive D).Free
@{started=(Get-Date).ToString('o');pid=$PID;manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash;disk_free_before=$startFree;target_count=$manifest.deletions.Count} | ConvertTo-Json | Set-Content -LiteralPath $intentPath -Encoding UTF8
$log = New-Object IO.StreamWriter((Join-Path $auditRoot 'deletion_log.jsonl'), $false, (New-Object Text.UTF8Encoding($false)))
$count=0; $removedBytes=[long]0; $removedFiles=[long]0; $errors=New-Object 'System.Collections.Generic.List[object]'
try {
    foreach ($entry in ($manifest.deletions | Sort-Object -Property bytes -Descending)) {
        $record=@{path=$entry.path;started=(Get-Date).ToString('o');bytes=$entry.bytes;files=$entry.files}
        try {
            Remove-Item -LiteralPath $entry.path -Recurse -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $entry.path) { throw 'Target still exists after removal' }
            $record.status='removed'; $removedBytes += $entry.bytes; $removedFiles += $entry.files
        } catch {
            $record.status='error';$record.error=$_.Exception.Message;$errors.Add($record)
        }
        $log.WriteLine(($record | ConvertTo-Json -Compress));$log.Flush();$count++
        if ($count % 500 -eq 0 -or $entry.bytes -gt 500000000) { Write-Output "Removed targets $count/$($manifest.deletions.Count); logical bytes $removedBytes; errors $($errors.Count)" }
    }
} finally { $log.Dispose() }
$finishFree=(Get-PSDrive D).Free
@{status=$(if($errors.Count){'completed_with_errors'}else{'completed'});finished=(Get-Date).ToString('o');targets_processed=$count;logical_bytes_removed=$removedBytes;files_removed=$removedFiles;disk_free_before=$startFree;disk_free_after=$finishFree;disk_free_delta=$finishFree-$startFree;errors=$errors.ToArray()} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $auditRoot 'deletion_result.json') -Encoding UTF8
Write-Output "Finished. Disk free increased by $($finishFree-$startFree) bytes; errors $($errors.Count)."
