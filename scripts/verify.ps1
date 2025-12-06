# One-liner verification runner for MathLedger (PowerShell)
# Usage: .\scripts\verify.ps1 [-Offline] [-Check <name>]

param(
    [switch]$Offline,
    [string]$Check = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

$args = @()
if ($Offline) {
    $args += "--offline"
}
if ($Check) {
    $args += "--check", $Check
}

python tools/verify_all.py @args
exit $LASTEXITCODE
