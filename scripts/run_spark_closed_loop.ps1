#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run SPARK closed-loop integration test for First Organism.

.DESCRIPTION
    This script runs the SPARK closed-loop test that validates the complete
    First Organism pipeline: UI Event → Curriculum Gate → Derivation →
    Lean Verify → Dual-Attest seal H_t → RFL runner metabolism.

    Prerequisites:
    1. First Organism infrastructure must be running (use start_first_organism_infra.ps1)
    2. Database migrations should be run (optional, test handles setup)

.EXAMPLE
    .\scripts\run_spark_closed_loop.ps1

.NOTES
    This script is part of OPERATION SPARK - First Organism integration.
    Output is logged to ops/logs/SPARK_run_log.txt
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SPARK Closed-Loop Test Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure logs directory exists
$logsDir = "ops/logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

$logFile = Join-Path $logsDir "SPARK_run_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "[INFO] Logging to: $logFile" -ForegroundColor Yellow
Write-Host "[INFO] Starting SPARK test at $timestamp" -ForegroundColor Yellow
Write-Host ""

# Set environment variables
$env:FIRST_ORGANISM_TESTS = "true"
$env:SPARK_RUN = "1"

Write-Host "[INFO] Environment variables set:" -ForegroundColor Yellow
Write-Host "  FIRST_ORGANISM_TESTS=$env:FIRST_ORGANISM_TESTS" -ForegroundColor White
Write-Host "  SPARK_RUN=$env:SPARK_RUN" -ForegroundColor White
Write-Host ""

# Run the test
Write-Host "[INFO] Running SPARK closed-loop test..." -ForegroundColor Yellow
Write-Host ""

$testCommand = "uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s"

try {
    # Capture both stdout and stderr
    $output = & pwsh -Command $testCommand 2>&1 | Tee-Object -Variable allOutput
    
    # Write to log file with timestamp
    $logContent = @"
========================================
SPARK Run Log - $timestamp
========================================

Command: $testCommand
Environment:
  FIRST_ORGANISM_TESTS=$env:FIRST_ORGANISM_TESTS
  SPARK_RUN=$env:SPARK_RUN

Output:
$($allOutput | Out-String)

========================================
"@
    
    $logContent | Out-File -FilePath $logFile -Encoding UTF8
    
    # Check for PASS line
    $passLine = $allOutput | Select-String -Pattern "\[PASS\] FIRST ORGANISM ALIVE H_t="
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    if ($passLine) {
        Write-Host "✅ SPARK: PASS" -ForegroundColor Green
        Write-Host "   Found: $($passLine.Line.Trim())" -ForegroundColor White
    } else {
        Write-Host "❌ SPARK: NO PASS LINE FOUND" -ForegroundColor Red
        Write-Host ""
        Write-Host "   The test may have failed or the PASS line was not emitted." -ForegroundColor Yellow
        Write-Host "   Check the log file for details: $logFile" -ForegroundColor Yellow
    }
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[INFO] Full output saved to: $logFile" -ForegroundColor Yellow
    
    # Exit with test exit code
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ SPARK: Test execution failed" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Check the log file for details: $logFile" -ForegroundColor Yellow
    
    # Log the error
    $errorLog = @"
========================================
SPARK Run Log - $timestamp
========================================

Command: $testCommand
Error: $_

========================================
"@
    $errorLog | Out-File -FilePath $logFile -Encoding UTF8
    
    exit 1
}

