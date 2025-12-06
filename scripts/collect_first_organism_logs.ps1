#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Collect First Organism diagnostic logs.

.DESCRIPTION
    Gathers logs from First Organism Docker containers and SPARK run logs
    into a single diagnostic bundle for troubleshooting.

    Collects:
    - Docker logs from first_organism_postgres container
    - Docker logs from first_organism_redis container
    - Latest SPARK_run_log.txt from ops/logs/

    Outputs to: ops/logs/SPARK_diag_bundle.txt

.EXAMPLE
    .\scripts\collect_first_organism_logs.ps1

.NOTES
    Part of OPERATION SPARK - First Organism integration.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "First Organism Log Collection" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir = Join-Path $ProjectRoot "ops\logs"
$OutputFile = Join-Path $LogDir "SPARK_diag_bundle.txt"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    Write-Host "Created log directory: $LogDir" -ForegroundColor Green
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$diagBundle = @"
========================================
First Organism Diagnostic Bundle
Generated: $timestamp
========================================

"@

# Collect PostgreSQL logs
Write-Host "Collecting PostgreSQL logs..." -ForegroundColor Yellow
$postgresContainer = "first_organism_postgres"
try {
    $postgresLogs = docker logs $postgresContainer 2>&1
    if ($LASTEXITCODE -eq 0) {
        $diagBundle += @"

========================================
PostgreSQL Container Logs
Container: $postgresContainer
========================================

$postgresLogs

"@
        Write-Host "  ✓ Collected PostgreSQL logs" -ForegroundColor Green
    } else {
        $diagBundle += @"

========================================
PostgreSQL Container Logs
Container: $postgresContainer
Status: Container not found or inaccessible
========================================

Error: Could not retrieve logs from container '$postgresContainer'
Exit code: $LASTEXITCODE

"@
        Write-Host "  ⚠ PostgreSQL container not found" -ForegroundColor Yellow
    }
} catch {
    $diagBundle += @"

========================================
PostgreSQL Container Logs
Container: $postgresContainer
Status: Error collecting logs
========================================

Error: $_

"@
    Write-Host "  ⚠ Error collecting PostgreSQL logs: $_" -ForegroundColor Yellow
}

# Collect Redis logs
Write-Host "Collecting Redis logs..." -ForegroundColor Yellow
$redisContainer = "first_organism_redis"
try {
    $redisLogs = docker logs $redisContainer 2>&1
    if ($LASTEXITCODE -eq 0) {
        $diagBundle += @"

========================================
Redis Container Logs
Container: $redisContainer
========================================

$redisLogs

"@
        Write-Host "  ✓ Collected Redis logs" -ForegroundColor Green
    } else {
        $diagBundle += @"

========================================
Redis Container Logs
Container: $redisContainer
Status: Container not found or inaccessible
========================================

Error: Could not retrieve logs from container '$redisContainer'
Exit code: $LASTEXITCODE

"@
        Write-Host "  ⚠ Redis container not found" -ForegroundColor Yellow
    }
} catch {
    $diagBundle += @"

========================================
Redis Container Logs
Container: $redisContainer
Status: Error collecting logs
========================================

Error: $_

"@
    Write-Host "  ⚠ Error collecting Redis logs: $_" -ForegroundColor Yellow
}

# Collect SPARK run log
Write-Host "Collecting SPARK run log..." -ForegroundColor Yellow
$sparkLogFile = Join-Path $LogDir "SPARK_run_log.txt"
if (Test-Path $sparkLogFile) {
    try {
        $sparkLogs = Get-Content $sparkLogFile -Raw -ErrorAction Stop
        $diagBundle += @"

========================================
SPARK Run Log
File: $sparkLogFile
========================================

$sparkLogs

"@
        Write-Host "  ✓ Collected SPARK run log" -ForegroundColor Green
    } catch {
        $diagBundle += @"

========================================
SPARK Run Log
File: $sparkLogFile
Status: Error reading file
========================================

Error: $_

"@
        Write-Host "  ⚠ Error reading SPARK run log: $_" -ForegroundColor Yellow
    }
} else {
    $diagBundle += @"

========================================
SPARK Run Log
File: $sparkLogFile
Status: File not found
========================================

SPARK_run_log.txt not found at expected location.
The SPARK test may not have been run yet, or logs were written elsewhere.

"@
    Write-Host "  ⚠ SPARK run log not found" -ForegroundColor Yellow
}

# Add container status summary
Write-Host "Collecting container status..." -ForegroundColor Yellow
$composeFile = Join-Path $ProjectRoot "ops\first_organism\docker-compose.yml"
try {
    $containerStatus = docker compose -f $composeFile ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        $diagBundle += @"

========================================
Container Status Summary
========================================

$containerStatus

"@
        Write-Host "  ✓ Collected container status" -ForegroundColor Green
    }
} catch {
    # Container status is optional, just log it
    $diagBundle += @"

========================================
Container Status Summary
Status: Could not retrieve container status
========================================

Error: $_

"@
}

# Finalize bundle
$diagBundle += @"
========================================
End of Diagnostic Bundle
========================================

"@

# Write output file
try {
    $diagBundle | Out-File -FilePath $OutputFile -Encoding UTF8 -NoNewline
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "✅ Diagnostic bundle created" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Output file: $OutputFile" -ForegroundColor White
    Write-Host ""
    Write-Host "You can now share this file for troubleshooting." -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "  ❌ Failed to write diagnostic bundle: $_" -ForegroundColor Red
    exit 1
}

