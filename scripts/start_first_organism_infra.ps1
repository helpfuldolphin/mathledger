#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start Docker services for First Organism integration tests.

.DESCRIPTION
    This script starts PostgreSQL and Redis containers required for
    First Organism closed-loop integration tests.

    Prerequisites:
    1. Docker Desktop must be running
    2. .env.first_organism file must exist with secure credentials
    3. See ops/first_organism/first_organism.env.template for setup

.EXAMPLE
    .\scripts\start_first_organism_infra.ps1

.NOTES
    This script is part of OPERATION SPARK - First Organism integration.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "First Organism Infrastructure Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker is running
Write-Host "[1/4] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "  ❌ Docker not running" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Remediation:" -ForegroundColor Yellow
    Write-Host "    1. Open Docker Desktop application" -ForegroundColor White
    Write-Host "    2. Wait for it to fully start (whale icon in system tray)" -ForegroundColor White
    Write-Host "    3. Run this script again" -ForegroundColor White
    exit 1
}

# Check .env.first_organism exists
Write-Host "[2/4] Checking environment file..." -ForegroundColor Yellow
$envFile = ".env.first_organism"
if (-not (Test-Path $envFile)) {
    Write-Host "  ✗ $envFile not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "  To create it:" -ForegroundColor Yellow
    Write-Host "    1. Copy template: Copy-Item ops/first_organism/first_organism.env.template .env.first_organism" -ForegroundColor Yellow
    Write-Host "    2. Generate secure credentials (see template for commands)" -ForegroundColor Yellow
    Write-Host "    3. Replace all <REPLACE_...> placeholders" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✓ $envFile found" -ForegroundColor Green

# Start services
Write-Host "[3/4] Starting Docker services..." -ForegroundColor Yellow
$composeFile = "ops/first_organism/docker-compose.yml"
if (-not (Test-Path $composeFile)) {
    Write-Host "  ✗ $composeFile not found" -ForegroundColor Red
    exit 1
}

try {
    docker compose -f $composeFile --env-file $envFile up -d
    Write-Host "  ✓ Services started" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to start services: $_" -ForegroundColor Red
    exit 1
}

# Wait for health checks
Write-Host "[4/4] Waiting for services to be healthy..." -ForegroundColor Yellow
$maxWait = 60
$waited = 0
$interval = 2

while ($waited -lt $maxWait) {
    $status = docker compose -f $composeFile ps --format json | ConvertFrom-Json
    $postgresHealthy = ($status | Where-Object { $_.Service -eq "postgres" -and $_.Health -eq "healthy" }) -ne $null
    $redisHealthy = ($status | Where-Object { $_.Service -eq "redis" -and $_.Health -eq "healthy" }) -ne $null
    
    if ($postgresHealthy -and $redisHealthy) {
        Write-Host "  ✓ All services healthy" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $interval
    $waited += $interval
    Write-Host "  ... waiting ($waited/$maxWait seconds)" -ForegroundColor Gray
}

if (-not ($postgresHealthy -and $redisHealthy)) {
    Write-Host ""
    Write-Host "  ❌ Health checks failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Remediation:" -ForegroundColor Yellow
    Write-Host "    1. Check logs: docker compose -f $composeFile logs postgres" -ForegroundColor White
    Write-Host "    2. Check logs: docker compose -f $composeFile logs redis" -ForegroundColor White
    Write-Host "    3. Verify .env.first_organism has correct credentials" -ForegroundColor White
    Write-Host "    4. Try restarting: docker compose -f $composeFile --env-file $envFile restart" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ First Organism infra is up (Postgres/Redis healthy)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "  Redis:      localhost:6380" -ForegroundColor White
Write-Host ""
Write-Host "Run tests:" -ForegroundColor Yellow
Write-Host "  `$env:FIRST_ORGANISM_TESTS='true'" -ForegroundColor White
Write-Host "  uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s" -ForegroundColor White
Write-Host ""
Write-Host "Stop services:" -ForegroundColor Yellow
Write-Host "  docker compose -f $composeFile --env-file $envFile down" -ForegroundColor White
Write-Host ""

