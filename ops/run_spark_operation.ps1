#!/usr/bin/env pwsh
<#
.SYNOPSIS
    OPERATION SPARK - First Organism Integration Test Runner
    
.DESCRIPTION
    Executes the First Organism closed-loop integration test with live Docker services.
    This is the SPARK operation - the sole objective until it passes.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OPERATION SPARK - First Organism Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker
Write-Host "[1/4] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    Write-Host ""
    Write-Host "  OPERATION SPARK requires live Dockerized Postgres + Redis." -ForegroundColor Yellow
    Write-Host "  Please start Docker Desktop and run this script again." -ForegroundColor Yellow
    exit 1
}

# Start Docker services
Write-Host "[2/4] Starting Docker services (Postgres + Redis)..." -ForegroundColor Yellow
try {
    docker compose -f docker-compose.yml up -d postgres redis
    Write-Host "  ✓ Services starting..." -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to start services: $_" -ForegroundColor Red
    exit 1
}

# Wait for services to be ready
Write-Host "[3/4] Waiting for services to be healthy..." -ForegroundColor Yellow
$maxWait = 30
$waited = 0
$interval = 2

while ($waited -lt $maxWait) {
    $postgresHealthy = $false
    $redisHealthy = $false
    
    try {
        docker exec $(docker ps -q -f name=postgres) pg_isready -U ${env:POSTGRES_USER} | Out-Null
        $postgresHealthy = $true
    } catch { }
    
    try {
        docker exec $(docker ps -q -f name=redis) redis-cli ping | Out-Null
        $redisHealthy = $true
    } catch { }
    
    if ($postgresHealthy -and $redisHealthy) {
        Write-Host "  ✓ All services healthy" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $interval
    $waited += $interval
    Write-Host "  ... waiting ($waited/$maxWait seconds)" -ForegroundColor Gray
}

if (-not ($postgresHealthy -and $redisHealthy)) {
    Write-Host "  ✗ Services did not become healthy" -ForegroundColor Red
    Write-Host "  Check logs: docker compose logs postgres redis" -ForegroundColor Yellow
    exit 1
}

# Set environment variables
Write-Host "[4/4] Setting environment and running test..." -ForegroundColor Yellow
# Default DB URL uses port 5432 to align with First Organism docker-compose.yml
$env:DATABASE_URL = $env:DATABASE_URL ?? "postgresql://ml:mlpass@127.0.0.1:5432/mathledger"
$env:REDIS_URL = $env:REDIS_URL ?? "redis://127.0.0.1:6379/0"
$env:FIRST_ORGANISM_TESTS = "true"
$env:SPARK_RUN = "1"

Write-Host "  DATABASE_URL=$($env:DATABASE_URL)" -ForegroundColor Gray
Write-Host "  REDIS_URL=$($env:REDIS_URL)" -ForegroundColor Gray
Write-Host ""

# Run the test
Write-Host "Running First Organism closed-loop test..." -ForegroundColor Cyan
Write-Host ""

$testName = "test_first_organism_closed_loop_happy_path"
$result = uv run pytest "tests/integration/test_first_organism.py::$testName" -v -s

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "OPERATION SPARK: SUCCESS" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "First Organism closed-loop test PASSED" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "OPERATION SPARK: FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "First Organism closed-loop test FAILED" -ForegroundColor Red
    Write-Host ""
    exit 1
}

