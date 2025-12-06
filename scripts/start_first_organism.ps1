#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start First Organism infrastructure with security validation.

.DESCRIPTION
    This script performs security validation before starting Docker services
    for the First Organism integration test environment.

    Security checks performed (Phase I enforcer thresholds):
    1. Validates .env.first_organism exists
    2. Checks for <REPLACE_ME_...> placeholders
    3. Invokes First Organism security enforcer:
       - PostgreSQL password >= 12 chars, not in banned list
       - Redis password >= 12 chars, not in banned list
       - API key >= 16 chars, >= 6 unique characters
       - CORS origins must not contain wildcard (*)
    4. Verifies Docker availability
    5. Starts Docker Compose services
    6. Waits for health checks

.EXAMPLE
    .\scripts\start_first_organism.ps1

.EXAMPLE
    .\scripts\start_first_organism.ps1 -SkipHealthWait

.NOTES
    Part of First Organism Security Hardening (Phase I)
    Author: CLAUDE N - Security Officer
    Date: 2025-11-30

    Enforcer source: backend/security/first_organism_enforcer.py
    MIN_PASSWORD_LENGTH = 12
    MIN_API_KEY_LENGTH = 16

    RFL Evidence & Environment (Phase I - Current):
    - RFL logs do not hit infrastructure (in-memory only)
    - RFL Phase I operates hermetically without DB/Redis
    - Environment hardening does not affect RFL runs
    - This script is for First Organism integration tests, not RFL experiments

    Phase II RFL Uplift (Future - Not Implemented):
    When RFL transitions to DB-backed experiments, this script will need to:
    - Validate RFL_ENV_MODE=phase2-uplift is set
    - Validate RFL_DB_URL with dedicated RFL credentials
    - Validate RFL_REDIS_URL or isolated database index
    - Verify RFL schema migrations are applied
    - Check RFL_AUDIT_ENABLED for transaction logging

    These checks are NOT implemented. Phase II stubs exist in .env.first_organism.template
    as comments only. See docs/first_organism_env_hardening_plan.md for requirements.
#>

[CmdletBinding()]
param(
    [switch]$SkipHealthWait,
    [int]$HealthWaitTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================

$EnvFile = ".env.first_organism"
$EnvTemplate = "config\.env.first_organism.template"
$ComposeFile = "ops\first_organism\docker-compose.yml"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Change to project root
Set-Location $ProjectRoot

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "  [FAIL] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "  [INFO] $Message" -ForegroundColor Gray
}

function Exit-WithError {
    param([string]$Message, [int]$ExitCode = 1)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Red
    Write-Host " STARTUP ABORTED" -ForegroundColor Red
    Write-Host "=" * 60 -ForegroundColor Red
    Write-Host ""
    Write-Failure $Message
    Write-Host ""
    exit $ExitCode
}

# ============================================================================
# Step 1: Check Environment File Exists
# ============================================================================

Write-Section "Step 1/5: Checking Environment File"

if (-not (Test-Path $EnvFile)) {
    Write-Failure "$EnvFile not found"
    Write-Host ""
    Write-Host "  To create it:" -ForegroundColor Yellow
    Write-Host "    1. Copy template:" -ForegroundColor White
    Write-Host "       Copy-Item $EnvTemplate $EnvFile" -ForegroundColor Gray
    Write-Host ""
    Write-Host "    2. Generate secure credentials:" -ForegroundColor White
    Write-Host '       $pg = -join ((65..90)+(97..122)+(48..57)+(33,35,37,38,42,64) | Get-Random -Count 32 | %{[char]$_})' -ForegroundColor Gray
    Write-Host '       $rd = -join ((65..90)+(97..122)+(48..57) | Get-Random -Count 24 | %{[char]$_})' -ForegroundColor Gray
    Write-Host '       $ak = -join ((48..57)+(97..102) | Get-Random -Count 64 | %{[char]$_})' -ForegroundColor Gray
    Write-Host '       Write-Host "`nPOSTGRES_PASSWORD=$pg`nREDIS_PASSWORD=$rd`nLEDGER_API_KEY=$ak"' -ForegroundColor Gray
    Write-Host ""
    Write-Host "    3. Replace all <REPLACE_ME_...> placeholders" -ForegroundColor White
    Write-Host ""
    Exit-WithError "Environment file not found. See instructions above."
}

Write-Success "Environment file found: $EnvFile"

# ============================================================================
# Step 2: Check for Unreplaced Placeholders
# ============================================================================

Write-Section "Step 2/5: Checking for Placeholders"

$envContent = Get-Content $EnvFile -Raw
$placeholders = [regex]::Matches($envContent, '<REPLACE_ME_[^>]+>')

if ($placeholders.Count -gt 0) {
    Write-Failure "Found $($placeholders.Count) unreplaced placeholder(s):"
    foreach ($match in $placeholders) {
        Write-Host "    - $($match.Value)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Generate credentials with:" -ForegroundColor Yellow
    Write-Host '    $pg = -join ((65..90)+(97..122)+(48..57)+(33,35,37,38,42,64) | Get-Random -Count 32 | %{[char]$_})' -ForegroundColor Gray
    Write-Host '    $rd = -join ((65..90)+(97..122)+(48..57) | Get-Random -Count 24 | %{[char]$_})' -ForegroundColor Gray
    Write-Host '    $ak = -join ((48..57)+(97..102) | Get-Random -Count 64 | %{[char]$_})' -ForegroundColor Gray
    Write-Host ""
    Exit-WithError "Unreplaced placeholders found. Replace them with generated credentials."
}

Write-Success "No unreplaced placeholders found"

# ============================================================================
# Step 3: Load Environment and Run Security Enforcer
# ============================================================================

Write-Section "Step 3/5: Running Security Enforcer"

# Load environment variables from file
$envVars = @{}
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        # Remove quotes if present
        $value = $value -replace '^["'']|["'']$', ''
        $envVars[$key] = $value
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

Write-Info "Loaded $($envVars.Count) environment variables"

# Run the Python security enforcer
Write-Info "Invoking First Organism security enforcer..."

$enforcerScript = @"
import sys
try:
    from backend.security.first_organism_enforcer import enforce_first_organism_env
    config = enforce_first_organism_env()
    print("ENFORCER_OK")
    print(f"  Database: {config.postgres_user}@***")
    print(f"  Redis: password set ({len(config.redis_password or '')} chars)")
    print(f"  API Key: {len(config.api_key)} chars, {len(set(config.api_key))} unique")
    print(f"  CORS Origins: {len(config.cors_origins)} allowed")
except Exception as e:
    print(f"ENFORCER_FAIL: {e}")
    sys.exit(1)
"@

try {
    $result = $enforcerScript | uv run python - 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0 -or $result -match "ENFORCER_FAIL") {
        Write-Failure "Security enforcer rejected configuration"
        Write-Host ""
        Write-Host "  Enforcer output:" -ForegroundColor Yellow
        $result | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        Write-Host ""
        Exit-WithError "Security validation failed. Fix the issues above."
    }

    Write-Success "Security enforcer validated configuration"
    $result | Where-Object { $_ -notmatch "ENFORCER_OK" } | ForEach-Object {
        Write-Info $_
    }
}
catch {
    Write-Failure "Failed to run security enforcer: $_"
    Exit-WithError "Could not execute Python enforcer. Ensure 'uv' is installed."
}

# ============================================================================
# Step 4: Check Docker Availability
# ============================================================================

Write-Section "Step 4/5: Checking Docker"

try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not responding"
    }
    Write-Success "Docker is running (version $dockerVersion)"
}
catch {
    Write-Failure "Docker is not available"
    Write-Host ""
    Write-Host "  Remediation:" -ForegroundColor Yellow
    Write-Host "    1. Open Docker Desktop" -ForegroundColor White
    Write-Host "    2. Wait for the whale icon to be solid (not animating)" -ForegroundColor White
    Write-Host "    3. Run this script again" -ForegroundColor White
    Write-Host ""
    Exit-WithError "Docker Desktop is not running."
}

# Check docker-compose file exists
if (-not (Test-Path $ComposeFile)) {
    Write-Failure "Docker Compose file not found: $ComposeFile"
    Exit-WithError "Missing docker-compose.yml"
}

Write-Success "Docker Compose file found: $ComposeFile"

# ============================================================================
# Step 5: Start Docker Services
# ============================================================================

Write-Section "Step 5/5: Starting Docker Services"

Write-Info "Starting containers..."

try {
    docker compose -f $ComposeFile --env-file $EnvFile up -d 2>&1 | ForEach-Object {
        if ($_ -match "error|Error|ERROR") {
            Write-Host "    $_" -ForegroundColor Red
        } else {
            Write-Host "    $_" -ForegroundColor Gray
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE"
    }

    Write-Success "Docker Compose services started"
}
catch {
    Write-Failure "Failed to start services: $_"
    Write-Host ""
    Write-Host "  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "    docker compose -f $ComposeFile logs" -ForegroundColor Gray
    Write-Host ""
    Exit-WithError "Docker Compose startup failed."
}

# ============================================================================
# Step 6: Wait for Health Checks (Optional)
# ============================================================================

if (-not $SkipHealthWait) {
    Write-Host ""
    Write-Info "Waiting for health checks (timeout: ${HealthWaitTimeoutSeconds}s)..."

    $waited = 0
    $interval = 2
    $postgresHealthy = $false
    $redisHealthy = $false

    while ($waited -lt $HealthWaitTimeoutSeconds) {
        try {
            $status = docker compose -f $ComposeFile ps --format json 2>$null | ConvertFrom-Json
            $postgresHealthy = ($status | Where-Object { $_.Service -eq "postgres" -and $_.Health -eq "healthy" }) -ne $null
            $redisHealthy = ($status | Where-Object { $_.Service -eq "redis" -and $_.Health -eq "healthy" }) -ne $null

            if ($postgresHealthy -and $redisHealthy) {
                break
            }
        }
        catch {
            # Ignore JSON parse errors during startup
        }

        Start-Sleep -Seconds $interval
        $waited += $interval
        Write-Host "    Waiting... ($waited/${HealthWaitTimeoutSeconds}s)" -ForegroundColor Gray
    }

    if ($postgresHealthy -and $redisHealthy) {
        Write-Success "All services are healthy"
    }
    else {
        Write-Warning "Health check timeout - services may still be starting"
        Write-Host "    Check status: docker compose -f $ComposeFile ps" -ForegroundColor Gray
    }
}

# ============================================================================
# Success Summary
# ============================================================================

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host " FIRST ORGANISM INFRASTRUCTURE READY" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "  Services:" -ForegroundColor Cyan
Write-Host "    PostgreSQL: 127.0.0.1:5432" -ForegroundColor White
Write-Host "    Redis:      127.0.0.1:6380" -ForegroundColor White
Write-Host ""
Write-Host "  Run tests:" -ForegroundColor Cyan
Write-Host '    $env:FIRST_ORGANISM_TESTS="true"' -ForegroundColor Gray
Write-Host "    uv run pytest -m first_organism -v" -ForegroundColor Gray
Write-Host ""
Write-Host "  Stop services:" -ForegroundColor Cyan
Write-Host "    docker compose -f $ComposeFile --env-file $EnvFile down" -ForegroundColor Gray
Write-Host ""
Write-Host "  Stop and remove volumes (clean slate):" -ForegroundColor Cyan
Write-Host "    docker compose -f $ComposeFile --env-file $EnvFile down -v" -ForegroundColor Gray
Write-Host ""

exit 0
