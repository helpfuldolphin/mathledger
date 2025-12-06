# MathLedger Golden Run Script
# Reproducible PL-1 derivation with progress logging

param(
    [string]$SystemSlug = "pl"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write timestamped log message
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    if ($script:logFile) {
        Add-Content -Path $script:logFile -Value $logMessage
    }
}

# Main execution
try {
    # Require DATABASE_URL and REDIS_URL - no fallbacks
    if (-not $env:DATABASE_URL) {
        Write-Host "[FATAL] DATABASE_URL environment variable is not set." -ForegroundColor Red
        exit 1
    }
    if (-not $env:REDIS_URL) {
        Write-Host "[FATAL] REDIS_URL environment variable is not set." -ForegroundColor Red
        exit 1
    }
    $databaseUrl = $env:DATABASE_URL
    $redisUrl = $env:REDIS_URL

    # Set up logging
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $script:logFile = "logs\golden-$timestamp.log"

    # Create logs directory if it doesn't exist
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
        Write-Log "Created logs directory"
    }

    Write-Log "Starting MathLedger Golden Run"
    Write-Log "System: $SystemSlug"
    Write-Log "Database: (from environment)"
    Write-Log "Redis: (from environment)"

    # Fast database precheck with 5s timeout
    Write-Log "Testing database connectivity..."
    try {
        $dbTestScript = @"
import os, psycopg
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('DB_FAILED: DATABASE_URL not set')
    exit(1)
try:
    conn = psycopg.connect(db_url + '?connect_timeout=5' if '?' not in db_url else db_url)
    conn.close()
    print('DB_OK')
except Exception as e:
    print(f'DB_FAILED: {e}')
    exit(1)
"@
        $dbTestResult = & uv run python -c $dbTestScript 2>&1
        if ($LASTEXITCODE -eq 0 -and $dbTestResult -match "DB_OK") {
            Write-Log "Database connection successful"
        } else {
            Write-Host "DB UNAVAILABLE" -ForegroundColor Red
            Write-Log "Database connection failed: $dbTestResult" "ERROR"
            exit 1
        }
    } catch {
        Write-Host "DB UNAVAILABLE" -ForegroundColor Red
        Write-Log "Database connection test failed: $_" "ERROR"
        exit 1
    }

    # Set environment variables for the derive command
    $env:DATABASE_URL = $databaseUrl
    $env:REDIS_URL = $redisUrl

    # Run derivation with sealing
    Write-Log "Running derivation with sealing..."
    $deriveArgs = @(
        "-m", "backend.axiom_engine.derive"
        "--system", $SystemSlug
        "--steps", "200"
        "--max-breadth", "1000"
        "--max-total", "5000"
        "--seal"
    )

    Write-Log "Command: uv run python $($deriveArgs -join ' ')"

    # Capture output and tee to log file
    $deriveResult = & uv run python @deriveArgs 2>&1
    $exitCode = $LASTEXITCODE

    # Write raw output to log file
    Add-Content -Path $script:logFile -Value "=== DERIVE OUTPUT ==="
    Add-Content -Path $script:logFile -Value $deriveResult
    Add-Content -Path $script:logFile -Value "=================="

    if ($exitCode -eq 0) {
        Write-Log "Derivation completed successfully"

        # Extract and print BLOCK line from derive output
        $blockLine = $deriveResult | Where-Object { $_ -match "Block sealed.*number=.*merkle" }
        if ($blockLine) {
            Write-Host "BLOCK $blockLine" -ForegroundColor Green
        }

        # Update progress documentation
        Write-Log "Updating progress documentation..."
        try {
            $progressResult = & uv run python -m backend.tools.progress --append-latest 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Progress updated successfully"
            } else {
                Write-Log "Progress update failed: $progressResult" "WARN"
            }
        } catch {
            Write-Log "Progress update failed: $_" "WARN"
        }

        Write-Log "Golden run completed successfully"
        Write-Log "Log file: $script:logFile"

    } else {
        Write-Log "Golden run failed with exit code $exitCode" "ERROR"
        Write-Log "Output: $deriveResult" "ERROR"
        exit 1
    }

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
