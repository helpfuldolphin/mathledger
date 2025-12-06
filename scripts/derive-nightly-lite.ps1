# Nightly derivation script (lite version - no ratchet)
# Runs derive with seal, appends progress (skips ratchet evaluation)
#
# SECURITY NOTICE: Requires DATABASE_URL in environment.

$ErrorActionPreference = "Stop"

# Validate required environment variable
if (-not $env:DATABASE_URL) {
    Write-Host "[FATAL] DATABASE_URL environment variable is not set." -ForegroundColor Red
    exit 1
}

Write-Host "Starting nightly derivation run (lite)..." -ForegroundColor Cyan
Write-Host "Using DATABASE_URL from environment" -ForegroundColor Gray

# Step 1: Run derive with seal flag
Write-Host "Running axiom engine derive with seal..." -ForegroundColor Yellow
uv run python -m backend.axiom_engine.derive --system pl --seal
if ($LASTEXITCODE -ne 0) {
    Write-Host "Derive failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 2: Append progress (skip ratchet)
Write-Host "Appending progress..." -ForegroundColor Yellow
try {
    $progressResult = & uv run python -m backend.tools.progress --append-latest 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Progress appended successfully" -ForegroundColor Green
    } else {
        Write-Host "Progress append failed: $progressResult" -ForegroundColor Red
    }
} catch {
    Write-Host "Progress append failed: $_" -ForegroundColor Red
}

Write-Host "nightly=ok" -ForegroundColor Green
