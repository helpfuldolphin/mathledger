#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run Wide Slice experiments for First Organism cycles.

.DESCRIPTION
    This script runs the Wide Slice experiments for First Organism (FO) cycles.
    It executes both baseline (RFL OFF) and RFL (RFL ON) modes, writing results
    to JSONL files for analysis.

    Prerequisites:
    1. First Organism infrastructure should be running (optional, but recommended)
    2. Python environment with uv available

.PARAMETER Cycles
    Number of FO cycles to run (default: 1000)

.PARAMETER SliceName
    Optional slice name identifier for the experiment (default: "wide")

.EXAMPLE
    .\scripts\run_wide_slice_experiments.ps1

.EXAMPLE
    .\scripts\run_wide_slice_experiments.ps1 -Cycles 500 -SliceName "test"

.NOTES
    This script is part of OPERATION SPARK - First Organism integration.
    Outputs:
    - results/fo_baseline_wide.jsonl (baseline mode)
    - results/fo_rfl_wide.jsonl (RFL mode)
    Logs:
    - ops/logs/wide_slice_baseline.log
    - ops/logs/wide_slice_rfl.log
#>

param(
    [int]$Cycles = 1000,
    [string]$SliceName = "wide"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Wide Slice Experiments Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure directories exist
$resultsDir = "results"
$logsDir = "ops/logs"

if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
    Write-Host "[INFO] Created results directory: $resultsDir" -ForegroundColor Yellow
}

if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    Write-Host "[INFO] Created logs directory: $logsDir" -ForegroundColor Yellow
}

$baselineOut = Join-Path $resultsDir "fo_baseline_wide.jsonl"
$rflOut = Join-Path $resultsDir "fo_rfl_wide.jsonl"
$baselineLog = Join-Path $logsDir "wide_slice_baseline.log"
$rflLog = Join-Path $logsDir "wide_slice_rfl.log"

Write-Host "[INFO] Configuration:" -ForegroundColor Yellow
Write-Host "  Cycles: $Cycles" -ForegroundColor White
Write-Host "  Slice Name: $SliceName" -ForegroundColor White
Write-Host "  Baseline Output: $baselineOut" -ForegroundColor White
Write-Host "  RFL Output: $rflOut" -ForegroundColor White
Write-Host ""

# Function to run a single experiment
function Run-Experiment {
    param(
        [string]$Mode,
        [string]$OutputFile,
        [string]$LogFile
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running $Mode experiment..." -ForegroundColor Yellow
    Write-Host "  Started: $timestamp" -ForegroundColor White
    Write-Host "  Output: $OutputFile" -ForegroundColor White
    Write-Host ""
    
    $command = "uv run python experiments/run_fo_cycles.py --mode=$Mode --cycles=$Cycles --out=$OutputFile"
    
    try {
        $startTime = Get-Date
        
        # Run command and capture output
        $output = & pwsh -Command $command 2>&1 | Tee-Object -Variable allOutput
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        # Write to log file
        $logContent = @"
========================================
Wide Slice $Mode Experiment - $timestamp
========================================

Command: $command
Cycles: $Cycles
Slice Name: $SliceName
Duration: $($duration.TotalSeconds) seconds

Output:
$($allOutput | Out-String)

========================================
"@
        
        $logContent | Out-File -FilePath $LogFile -Encoding UTF8
        
        Write-Host "  ✅ Completed in $([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor Green
        Write-Host "  ✅ Results written to: $OutputFile" -ForegroundColor Green
        Write-Host "  ✅ Log written to: $LogFile" -ForegroundColor Green
        
        # Check if output file exists and has content
        if (Test-Path $OutputFile) {
            $lineCount = (Get-Content $OutputFile | Measure-Object -Line).Lines
            Write-Host "  ✅ Output file contains $lineCount lines" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Warning: Output file not found after completion" -ForegroundColor Yellow
        }
        
        Write-Host ""
        return $true
        
    } catch {
        Write-Host "  ❌ Failed: $_" -ForegroundColor Red
        Write-Host ""
        
        # Log the error
        $errorLog = @"
========================================
Wide Slice $Mode Experiment - $timestamp
========================================

Command: $command
Error: $_

========================================
"@
        $errorLog | Out-File -FilePath $LogFile -Encoding UTF8
        
        return $false
    }
}

# Run baseline experiment
$baselineSuccess = Run-Experiment -Mode "baseline" -OutputFile $baselineOut -LogFile $baselineLog

Write-Host ""
Start-Sleep -Seconds 2

# Run RFL experiment
$rflSuccess = Run-Experiment -Mode "rfl" -OutputFile $rflOut -LogFile $rflLog

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($baselineSuccess) {
    Write-Host "  ✅ Baseline experiment: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "  ❌ Baseline experiment: FAILED" -ForegroundColor Red
}

if ($rflSuccess) {
    Write-Host "  ✅ RFL experiment: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "  ❌ RFL experiment: FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Output files:" -ForegroundColor Yellow
Write-Host "  Baseline: $baselineOut" -ForegroundColor White
Write-Host "  RFL:      $rflOut" -ForegroundColor White
Write-Host ""
Write-Host "Log files:" -ForegroundColor Yellow
Write-Host "  Baseline: $baselineLog" -ForegroundColor White
Write-Host "  RFL:      $rflLog" -ForegroundColor White
Write-Host ""

if ($baselineSuccess -and $rflSuccess) {
    Write-Host "✅ All experiments completed successfully" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ One or more experiments failed" -ForegroundColor Red
    exit 1
}

