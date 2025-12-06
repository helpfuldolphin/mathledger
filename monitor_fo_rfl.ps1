# Monitor 1000-cycle RFL run progress
# Run this in a separate PowerShell tab while the main run is executing

Write-Host "Monitoring fo_rfl.jsonl file growth..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
Write-Host ""

$file = "results/fo_rfl.jsonl"
$lastSize = 0
$lastTime = Get-Date

while ($true) {
    if (Test-Path $file) {
        $item = Get-Item $file
        $currentSize = $item.Length
        $currentTime = $item.LastWriteTime
        $sizeKB = [math]::Round($currentSize / 1KB, 2)
        $sizeMB = [math]::Round($currentSize / 1MB, 2)
        
        $sizeChange = $currentSize - $lastSize
        $timeChange = ($currentTime - $lastTime).TotalSeconds
        
        if ($sizeChange -gt 0) {
            $rateKB = [math]::Round($sizeChange / 1KB, 2)
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Size: $sizeKB KB ($sizeMB MB) | Change: +$rateKB KB | Last update: $currentTime" -ForegroundColor Green
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Size: $sizeKB KB ($sizeMB MB) | No change (may be processing...) | Last update: $currentTime" -ForegroundColor Yellow
        }
        
        $lastSize = $currentSize
        $lastTime = $currentTime
    } else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] File not found yet..." -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 5
}

