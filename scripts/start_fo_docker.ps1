#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start First Organism Docker infrastructure.

.DESCRIPTION
    Helper script to start First Organism Docker Compose stack.
    Checks Docker availability and starts PostgreSQL and Redis containers.

    Prerequisites:
    1. Docker Desktop must be running
    2. .env.first_organism file must exist with secure credentials

.EXAMPLE
    .\scripts\start_fo_docker.ps1

.NOTES
    Part of OPERATION SPARK - First Organism integration.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "First Organism Docker Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker is available
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker not available or not running" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

# Check .env.first_organism exists
$envFile = ".env.first_organism"
if (-not (Test-Path $envFile)) {
    Write-Host "  ❌ $envFile not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "  To create it:" -ForegroundColor Yellow
    Write-Host "    1. Copy template: Copy-Item ops/first_organism/first_organism.env.template .env.first_organism" -ForegroundColor White
    Write-Host "    2. Generate secure credentials (see template for commands)" -ForegroundColor White
    Write-Host "    3. Replace all <REPLACE_...> placeholders" -ForegroundColor White
    exit 1
}

Write-Host "  ✓ $envFile found" -ForegroundColor Green
Write-Host ""

# Check docker-compose.yml exists
$composeFile = "ops/first_organism/docker-compose.yml"
if (-not (Test-Path $composeFile)) {
    Write-Host "  ❌ $composeFile not found" -ForegroundColor Red
    exit 1
}

# Start Docker Compose services
Write-Host "Starting Docker Compose services..." -ForegroundColor Yellow

try {
    docker compose -f $composeFile --env-file $envFile up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE"
    }
    Write-Host "  ✓ Services started" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Failed to start services: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Container Status:" -ForegroundColor Cyan
Write-Host ""

# Show container names and status
try {
    $containers = docker compose -f $composeFile ps --format json | ConvertFrom-Json
    foreach ($container in $containers) {
        $status = $container.State
        $name = $container.Name
        $service = $container.Service
        
        if ($status -eq "running") {
            Write-Host "  ✓ $name ($service) - Running" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ $name ($service) - $status" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ⚠ Could not retrieve container status" -ForegroundColor Yellow
    Write-Host "  Run manually: docker compose -f $composeFile ps" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ First Organism Docker stack is up" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

