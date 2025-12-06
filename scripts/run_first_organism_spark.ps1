# First Organism SPARK Test Launcher
# ===================================
#
# This script loads environment variables from .env.first_organism and runs
# the First Organism closed-loop happy path test.
#
# Prerequisites:
#   1. Docker Desktop must be running
#   2. Containers 'postgres' and 'redis' must be up (check with: docker ps)
#   3. .env.first_organism file must exist in the project root
#
# Usage:
#   .\scripts\run_first_organism_spark.ps1
#
# Output:
#   - Test results printed to console
#   - Full log saved to ops/logs/SPARK_run_log.txt

$ErrorActionPreference = "Stop"

# Get the project root (parent of scripts/)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$EnvFile = Join-Path $ProjectRoot ".env.first_organism"
$LogDir = Join-Path $ProjectRoot "ops\logs"
$LogFile = Join-Path $LogDir "SPARK_run_log.txt"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "First Organism SPARK Test Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env.first_organism exists
if (-not (Test-Path $EnvFile)) {
    Write-Host "ERROR: .env.first_organism not found at: $EnvFile" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please create it by copying the template:" -ForegroundColor Yellow
    Write-Host "  cp config/first_organism.env.template .env.first_organism" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then customize the values in .env.first_organism" -ForegroundColor Yellow
    exit 1
}

Write-Host "Loading environment from: $EnvFile" -ForegroundColor Green

# Load environment variables from .env.first_organism
# PowerShell doesn't natively support .env files, so we parse it manually
$envVars = @{}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    # Skip empty lines and comments
    if ($line -and -not $line.StartsWith("#")) {
        if ($line -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Remove quotes if present
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $envVars[$key] = $value
        }
    }
}

# Set required environment variables
$requiredVars = @("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "REDIS_PASSWORD", "LEDGER_API_KEY", "CORS_ALLOWED_ORIGINS")
$missingVars = @()

foreach ($var in $requiredVars) {
    if (-not $envVars.ContainsKey($var)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "ERROR: Missing required environment variables:" -ForegroundColor Red
    foreach ($var in $missingVars) {
        Write-Host "  - $var" -ForegroundColor Red
    }
    exit 1
}

# Set environment variables
$env:POSTGRES_USER = $envVars["POSTGRES_USER"]
$env:POSTGRES_PASSWORD = $envVars["POSTGRES_PASSWORD"]
$env:POSTGRES_DB = $envVars["POSTGRES_DB"]
$env:REDIS_PASSWORD = $envVars["REDIS_PASSWORD"]
$env:LEDGER_API_KEY = $envVars["LEDGER_API_KEY"]
$env:CORS_ALLOWED_ORIGINS = $envVars["CORS_ALLOWED_ORIGINS"]

# Get Redis port (default 6379, but ops/first_organism/docker-compose.yml uses 6380)
$redisPort = if ($envVars.ContainsKey("REDIS_PORT")) { $envVars["REDIS_PORT"] } else { "6379" }

# Construct connection URLs
# SSL mode: disable for local Docker (matches config/first_organism.env line 10)
$env:DATABASE_URL = "postgresql://$($env:POSTGRES_USER):$($env:POSTGRES_PASSWORD)@localhost:5432/$($env:POSTGRES_DB)?sslmode=disable"
$env:REDIS_URL = "redis://:$($env:REDIS_PASSWORD)@localhost:$redisPort/0"

# Set test flags
$env:FIRST_ORGANISM_TESTS = "true"

Write-Host "Environment variables loaded successfully" -ForegroundColor Green
Write-Host "  POSTGRES_USER: $($env:POSTGRES_USER)" -ForegroundColor Gray
Write-Host "  POSTGRES_DB: $($env:POSTGRES_DB)" -ForegroundColor Gray
Write-Host "  REDIS_URL: redis://:***@localhost:$redisPort/0" -ForegroundColor Gray
Write-Host ""

# Check Docker Desktop
Write-Host "Checking Docker Desktop..." -ForegroundColor Cyan
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not found"
    }
    Write-Host "  Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Docker not found or not running" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

# Check if containers are running
# Explicit check for first_organism containers (from ops/first_organism/docker-compose.yml)
# Also checks for generic postgres/redis containers
Write-Host "Checking containers..." -ForegroundColor Cyan
$allContainers = docker ps --format "{{.Names}}" 2>&1
$postgresRunning = ($allContainers | Select-String -Pattern "first_organism_postgres|postgres") -ne $null
$redisRunning = ($allContainers | Select-String -Pattern "first_organism_redis|redis") -ne $null

if (-not $postgresRunning) {
    Write-Host "  WARNING: PostgreSQL container not found" -ForegroundColor Yellow
    Write-Host "  Expected: first_organism_postgres (from ops/first_organism/docker-compose.yml) or container with 'postgres' in name" -ForegroundColor Yellow
    Write-Host "  Run: docker ps" -ForegroundColor Yellow
    Write-Host ""
}

if (-not $redisRunning) {
    Write-Host "  WARNING: Redis container not found" -ForegroundColor Yellow
    Write-Host "  Expected: first_organism_redis (from ops/first_organism/docker-compose.yml) or container with 'redis' in name" -ForegroundColor Yellow
    Write-Host "  Run: docker ps" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Running SPARK test..." -ForegroundColor Cyan
Write-Host "  Test: tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path" -ForegroundColor Gray
Write-Host "  Log file: $LogFile" -ForegroundColor Gray
Write-Host ""

# Change to project root
Push-Location $ProjectRoot

try {
    # Run the test and capture output
    $testOutput = uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s 2>&1
    
    # Display output to console
    $testOutput | Write-Host
    
    # Save to log file with timestamp
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    @"
========================================
SPARK Test Run - $timestamp
========================================
Environment:
  POSTGRES_USER: $($env:POSTGRES_USER)
  POSTGRES_DB: $($env:POSTGRES_DB)
  DATABASE_URL: postgresql://$($env:POSTGRES_USER):***@localhost:5432/$($env:POSTGRES_DB)
  REDIS_URL: redis://:***@localhost:$redisPort/0

Test Command:
  uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s

Output:
$($testOutput -join "`n")

========================================
"@ | Out-File -FilePath $LogFile -Encoding UTF8
    
    # Check exit code
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "SPARK test PASSED" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Full log saved to: $LogFile" -ForegroundColor Gray
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "SPARK test FAILED (exit code: $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "Full log saved to: $LogFile" -ForegroundColor Gray
        Write-Host ""
        Write-Host "See ops/SPARK_INFRA_CHECKLIST.md for troubleshooting" -ForegroundColor Yellow
    }
    
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

