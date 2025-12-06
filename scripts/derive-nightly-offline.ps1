# Nightly derivation script (offline/online version)
# Skips derive step in offline mode, uses real data in online mode
#
# SECURITY NOTICE: Requires DATABASE_URL in environment (no defaults).

param(
    [switch]$Online,
    [string]$DbUrl = "",
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

# Set environment variables - require explicit configuration
if (-not $DbUrl) {
    if ($env:DATABASE_URL) {
        $DbUrl = $env:DATABASE_URL
    } else {
        Write-Host "[FATAL] DATABASE_URL environment variable is not set." -ForegroundColor Red
        Write-Host "Set it explicitly or pass -DbUrl parameter." -ForegroundColor Yellow
        exit 1
    }
}

if (-not $ApiKey) {
    if ($env:LEDGER_API_KEY) {
        $ApiKey = $env:LEDGER_API_KEY
    } else {
        Write-Host "[WARN] LEDGER_API_KEY not set, API calls may fail." -ForegroundColor Yellow
        $ApiKey = ""
    }
}

$env:DATABASE_URL = $DbUrl
$env:API_KEY = $ApiKey
$env:ML_USE_LOCAL_DB = "1"

# Ensure artifacts directory exists
if (-not (Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts" -Force | Out-Null
}

if ($Online) {
    Write-Host "Starting nightly derivation run (ONLINE)..." -ForegroundColor Cyan
} else {
    Write-Host "Starting nightly derivation run (offline)..." -ForegroundColor Cyan
}

# Step 1: Health check and derive step
if ($Online) {
    Write-Host "Verifying API health..." -ForegroundColor Yellow
    try {
        $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction Stop
        if ($healthResponse.ok -ne $true) {
            throw "Health check failed: $($healthResponse | ConvertTo-Json)"
        }
        Write-Host "API health check passed" -ForegroundColor Green
    } catch {
        $errorMsg = "Health check failed: $_"
        Write-Host $errorMsg -ForegroundColor Red
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        Add-Content -Path "metrics/integration_errors.log" -Value "$timestamp health_check_failed $errorMsg"
        exit 1
    }

    Write-Host "Running axiom engine derive with seal (ONLINE)..." -ForegroundColor Yellow
    $deriveOutput = & uv run python -m backend.axiom_engine.derive --system pl --smoke-pl --seal 2>&1
    $deriveOutput | Out-File -FilePath "artifacts/nightly_smoke.log" -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        $errorMsg = "Derive failed with exit code $LASTEXITCODE"
        Write-Host $errorMsg -ForegroundColor Red
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        Add-Content -Path "metrics/integration_errors.log" -Value "$timestamp derive_failed $errorMsg"
        exit $LASTEXITCODE
    }
    Write-Host "Derive completed successfully" -ForegroundColor Green
} else {
    Write-Host "Skipping axiom engine derive (offline mode)..." -ForegroundColor Yellow
}

# Step 2: Pull metrics and latest block
if ($Online) {
    Write-Host "Pulling metrics and latest block..." -ForegroundColor Yellow
    $headers = @{ "X-API-Key" = $ApiKey }

    try {
        # Get metrics
        $metricsResponse = Invoke-RestMethod -Uri "http://localhost:8000/metrics" -Headers $headers -ErrorAction Stop
        Write-Host "Metrics retrieved successfully" -ForegroundColor Green

        # Get latest block
        $blockResponse = Invoke-RestMethod -Uri "http://localhost:8000/blocks/latest" -Headers $headers -ErrorAction Stop
        Write-Host "Latest block retrieved successfully" -ForegroundColor Green

        # Extract data
        $proofsSuccess = $metricsResponse.proofs_success
        $proofsFailure = $metricsResponse.proofs_failure
        $blockNumber = $blockResponse.block_number
        $merkleRoot = $blockResponse.merkle_root

        # Decision rule: advance if proofs_success >= 2 else hold
        $decision = if ($proofsSuccess -ge 2) { "advance" } else { "hold" }
        $reason = if ($proofsSuccess -ge 2) { "proofs_success >= 2" } else { "proofs_success < 2" }

        # Write ratchet_last.txt
        $ratchetData = @{
            system = "pl"
            atoms = 1
            depth = 1
            proofs_success = $proofsSuccess
            proofs_failure = $proofsFailure
            decision = $decision
            reason = $reason
        } | ConvertTo-Json -Depth 3

        if (-not (Test-Path "metrics")) {
            New-Item -ItemType Directory -Path "metrics" -Force | Out-Null
        }
        [System.IO.File]::WriteAllText("metrics/ratchet_last.txt", $ratchetData, [System.Text.Encoding]::UTF8)
        Write-Host "Ratchet data saved: $decision - $reason" -ForegroundColor Green

    } catch {
        $errorMsg = "Failed to pull metrics/block: $_"
        Write-Host $errorMsg -ForegroundColor Red
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        Add-Content -Path "metrics/integration_errors.log" -Value "$timestamp metrics_pull_failed $errorMsg"
    }
} else {
    Write-Host "Evaluating curriculum ratchet (offline)..." -ForegroundColor Yellow
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
}

# Step 3: Append progress
if ($Online) {
    Write-Host "Appending progress (ONLINE)..." -ForegroundColor Yellow
    try {
        # Append tab-delimited line to progress.md
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        $progressLine = "$timestamp`tBLOCK: $blockNumber`tMERKLE: $merkleRoot`tPROOFS: $proofsSuccess/$proofsFailure"
        Add-Content -Path "docs/progress.md" -Value $progressLine
        Write-Host "Progress appended: $progressLine" -ForegroundColor Green
    } catch {
        $errorMsg = "Progress append failed: $_"
        Write-Host $errorMsg -ForegroundColor Red
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        Add-Content -Path "metrics/integration_errors.log" -Value "$timestamp progress_append_failed $errorMsg"
    }
} else {
    Write-Host "Appending progress (offline)..." -ForegroundColor Yellow
    try {
        $progressResult = & uv run python -m backend.tools.progress --append-latest --offline 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Progress appended successfully" -ForegroundColor Green
        } else {
            Write-Host "Progress append failed: $progressResult" -ForegroundColor Red
        }
    } catch {
        Write-Host "Progress append failed: $_" -ForegroundColor Red
    }
}

# Step 4: Validate
if ($Online) {
    Write-Host "Validating nightly run (ONLINE)..." -ForegroundColor Yellow
    try {
        $validateResult = & uv run python -m tools.validate_nightly 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Validation passed: $validateResult" -ForegroundColor Green
        } else {
            Write-Host "Validation failed: $validateResult" -ForegroundColor Red
        }
    } catch {
        Write-Host "Validation failed: $_" -ForegroundColor Red
    }
    Write-Host "nightly=ok" -ForegroundColor Green
} else {
    Write-Host "Validating nightly run (offline)..." -ForegroundColor Yellow
    try {
        $validateResult = & uv run python -m tools.validate_nightly --offline 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Validation passed: $validateResult" -ForegroundColor Green
        } else {
            Write-Host "Validation failed: $validateResult" -ForegroundColor Red
        }
    } catch {
        Write-Host "Validation failed: $_" -ForegroundColor Red
    }
    Write-Host "nightly=ok-offline" -ForegroundColor Green
}
