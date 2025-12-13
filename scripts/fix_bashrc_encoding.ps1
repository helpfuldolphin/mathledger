<#
.SYNOPSIS
    Fix Git Bash .bashrc UTF-16 BOM encoding issue.

.DESCRIPTION
    Git Bash expects ~/.bashrc to be UTF-8 without BOM. Windows Notepad
    sometimes saves files as UTF-16 LE with BOM (0xFFFE), causing the error:
        /c/Users/.../.bashrc: line 1: $'\377\376export': command not found

    This script converts the file to UTF-8 without BOM.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fix_bashrc_encoding.ps1
#>

$bashrcPath = "$env:USERPROFILE\.bashrc"
$backupPath = "$env:USERPROFILE\.bashrc.bak"

if (-not (Test-Path $bashrcPath)) {
    Write-Host "No .bashrc found at $bashrcPath" -ForegroundColor Yellow
    exit 0
}

# Read raw bytes to detect encoding
$bytes = [System.IO.File]::ReadAllBytes($bashrcPath)

# Check for UTF-16 LE BOM (0xFF 0xFE)
$isUtf16LE = ($bytes.Length -ge 2) -and ($bytes[0] -eq 0xFF) -and ($bytes[1] -eq 0xFE)

# Check for UTF-8 BOM (0xEF 0xBB 0xBF)
$isUtf8BOM = ($bytes.Length -ge 3) -and ($bytes[0] -eq 0xEF) -and ($bytes[1] -eq 0xBB) -and ($bytes[2] -eq 0xBF)

if (-not $isUtf16LE -and -not $isUtf8BOM) {
    Write-Host ".bashrc encoding is already correct (UTF-8 without BOM)" -ForegroundColor Green
    exit 0
}

# Backup original
Copy-Item $bashrcPath $backupPath -Force
Write-Host "Backed up to $backupPath" -ForegroundColor Cyan

# Read content with detected encoding
if ($isUtf16LE) {
    Write-Host "Detected UTF-16 LE BOM - converting to UTF-8 without BOM" -ForegroundColor Yellow
    $content = [System.IO.File]::ReadAllText($bashrcPath, [System.Text.Encoding]::Unicode)
} else {
    Write-Host "Detected UTF-8 BOM - removing BOM" -ForegroundColor Yellow
    $content = [System.IO.File]::ReadAllText($bashrcPath, [System.Text.Encoding]::UTF8)
}

# Write as UTF-8 without BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($bashrcPath, $content, $utf8NoBom)

Write-Host ".bashrc converted successfully" -ForegroundColor Green

# Verify
$newBytes = [System.IO.File]::ReadAllBytes($bashrcPath)
$stillHasBOM = ($newBytes.Length -ge 2) -and (($newBytes[0] -eq 0xFF) -or ($newBytes[0] -eq 0xEF))
if ($stillHasBOM) {
    Write-Host "WARNING: BOM still present after conversion" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Verified: No BOM present" -ForegroundColor Green
}
