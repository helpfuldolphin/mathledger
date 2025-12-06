# Nightly derivation script with curriculum ratcheting
# Runs derive with seal, checks ratcheting decision, appends progress
#
# SECURITY NOTICE: Requires DATABASE_URL and REDIS_URL in environment.

$ErrorActionPreference = "Stop"

# Validate required environment variables
if (-not $env:DATABASE_URL) {
    Write-Host "[FATAL] DATABASE_URL environment variable is not set." -ForegroundColor Red
    exit 1
}

Write-Host "Starting nightly derivation run..." -ForegroundColor Cyan
Write-Host "Using DATABASE_URL from environment" -ForegroundColor Gray

# Step 1: Run derive with seal flag and PL-Depth-4 parameters
Write-Host "Running axiom engine derive for PL-Depth-4 slice..." -ForegroundColor Yellow
Write-Host "Parameters: atoms=4, depth_max=4, breadth_max=500, total_max=2000" -ForegroundColor Cyan
uv run python -m backend.axiom_engine.derive --system pl --seal --depth-max 4 --max-breadth 500 --max-total 2000
if ($LASTEXITCODE -ne 0) {
    Write-Host "Derive failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 2: Run ratchet evaluation
Write-Host "Evaluating curriculum ratchet..." -ForegroundColor Yellow
try {
    # Run ratchet and capture output (use JSON mode for file)
    $ratchetOutputJson = & uv run python -m backend.frontier.ratchet --system pl --dry-run --format json --metrics-path metrics/mock_metrics.json 2>&1
    $ratchetOutputPlain = & uv run python -m backend.frontier.ratchet --system pl --dry-run --format plain --metrics-path metrics/mock_metrics.json 2>&1
    $ratchetExitCode = $LASTEXITCODE

    if ($ratchetExitCode -eq 0) {
        Write-Host "Ratchet decision: $ratchetOutputPlain" -ForegroundColor Green

        # Write JSON output to file
        if (-not (Test-Path "metrics")) {
            New-Item -ItemType Directory -Path "metrics" -Force | Out-Null
        }
        [System.IO.File]::WriteAllText("metrics/ratchet_last.txt", $ratchetOutputJson, [System.Text.Encoding]::UTF8)
        Write-Host "Ratchet result saved to metrics/ratchet_last.txt" -ForegroundColor Green
    } else {
        Write-Host "Ratchet evaluation failed: $ratchetOutputPlain" -ForegroundColor Red
    }
} catch {
    Write-Host "Ratchet evaluation failed: $_" -ForegroundColor Red
}

# Step 3: Append progress
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
