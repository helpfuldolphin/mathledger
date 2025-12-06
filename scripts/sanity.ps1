# MathLedger Sanity Check Script
# This script runs a comprehensive sanity check of the MathLedger system

param(
    [string]$ConfigPath = "config/nightly.env",
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to load configuration from .env file
function Load-Config {
    param([string]$ConfigPath)

    $config = @{}
    if (Test-Path $ConfigPath) {
        Get-Content $ConfigPath | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                $config[$key] = $value
            }
        }
    } else {
        Write-Error "Configuration file not found: $ConfigPath"
        exit 1
    }
    return $config
}

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

# Function to get Python path
function Get-PythonPath {
    Write-Log "Getting Python path..."

    try {
        $pythonPath = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Python version: $pythonPath"
        } else {
            Write-Log "Python not found in PATH" "WARN"
        }

        # Try uv run python
        $uvPythonPath = & uv run python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "UV Python version: $uvPythonPath"
        } else {
            Write-Log "UV Python not accessible" "WARN"
        }

        return @{ success = $true; python = $pythonPath; uv_python = $uvPythonPath }
    } catch {
        Write-Log "Failed to get Python path: $_" "ERROR"
        return @{ success = $false; error = $_ }
    }
}

# Function to run pytest
function Invoke-Pytest {
    Write-Log "Running pytest suite..."

    try {
        $pytestResult = & uv run pytest -q 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log "Pytest: PASSED"
            if ($Verbose) {
                Write-Log "Pytest output: $pytestResult"
            }
            return @{ success = $true; output = $pytestResult }
        } else {
            Write-Log "Pytest: FAILED (Exit code: $exitCode)" "ERROR"
            Write-Log "Pytest output: $pytestResult" "ERROR"
            return @{ success = $false; output = $pytestResult; exitCode = $exitCode }
        }
    } catch {
        Write-Log "Pytest command failed: $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    }
}

# Function to run small derivation
function Invoke-SmallDerivation {
    param([hashtable]$Config)

    Write-Log "Running small derivation test..."

    $deriveArgs = @(
        "-m", "backend.axiom_engine.derive"
        "--steps", "50"
        "--depth-max", "4"
        "--max-breadth", "200"
        "--max-total", "600"
        "--db-url", $Config.DATABASE_URL
        "--redis-url", $Config.REDIS_URL
        "--queue-key", $Config.QUEUE_KEY
        "--progress-path", $Config.PROGRESS_PATH
    )

    try {
        $deriveResult = & uv run python @deriveArgs 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log "Small derivation: SUCCESS"
            if ($Verbose) {
                Write-Log "Derivation output: $deriveResult"
            }
            return @{ success = $true; output = $deriveResult }
        } else {
            Write-Log "Small derivation: FAILED (Exit code: $exitCode)" "ERROR"
            Write-Log "Derivation output: $deriveResult" "ERROR"
            return @{ success = $false; output = $deriveResult; exitCode = $exitCode }
        }
    } catch {
        Write-Log "Derivation command failed: $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    }
}

# Function to check metrics endpoint
function Test-MetricsEndpoint {
    param([hashtable]$Config)

    Write-Log "Checking metrics endpoint..."

    try {
        $metricsResponse = Invoke-RestMethod -Uri $Config.METRICS_URL -Method GET -TimeoutSec 10

        if ($metricsResponse) {
            $totalStatements = $metricsResponse.statements.total
            $successRate = $metricsResponse.proofs.success_rate
            $queueLength = $metricsResponse.queue.length

            Write-Log "Metrics endpoint: OK"
            Write-Log "Total statements: $totalStatements"
            Write-Log "Success rate: $successRate%"
            Write-Log "Queue length: $queueLength"

            return @{
                success = $true;
                data = $metricsResponse;
                total_statements = $totalStatements;
                success_rate = $successRate;
                queue_length = $queueLength
            }
        } else {
            Write-Log "Metrics endpoint returned empty response" "WARN"
            return @{ success = $false; error = "Empty response" }
        }
    } catch {
        Write-Log "Metrics endpoint failed: $_" "ERROR"
        return @{ success = $false; error = $_.Exception.Message }
    }
}

# Function to create sanity summary
function New-SanitySummary {
    param(
        [hashtable]$Results,
        [string]$LogFile
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $summary = @{
        timestamp = $timestamp
        overall_success = $true
        checks = @{}
        log_file = $LogFile
    }

    foreach ($check in $Results.Keys) {
        $result = $Results[$check]
        $summary.checks[$check] = @{
            success = $result.success
            exit_code = if ($result.exitCode) { $result.exitCode } else { 0 }
        }

        if (-not $result.success) {
            $summary.overall_success = $false
        }
    }

    # Add metrics data if available
    if ($Results.metrics.success) {
        $summary.metrics = @{
            total_statements = $Results.metrics.total_statements
            success_rate = $Results.metrics.success_rate
            queue_length = $Results.metrics.queue_length
        }
    }

    return $summary
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    # Set up logging
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $script:logFile = Join-Path $config.LOGS_DIR "sanity-$timestamp.log"
    Write-Log "Logging to: $script:logFile"

    # Create logs directory if it doesn't exist
    if (-not (Test-Path $config.LOGS_DIR)) {
        New-Item -ItemType Directory -Path $config.LOGS_DIR -Force | Out-Null
    }

    Write-Log "Starting MathLedger sanity check"
    Write-Log "Configuration: MetricsURL=$($config.METRICS_URL), ProgressPath=$($config.PROGRESS_PATH)"

    $results = @{}
    $overallSuccess = $true

    # Check 1: Python path
    Write-Log "=== Check 1: Python Environment ==="
    $pythonResult = Get-PythonPath
    $results["python_path"] = $pythonResult
    if (-not $pythonResult.success) {
        $overallSuccess = $false
    }

    # Check 2: Run pytest
    Write-Log "=== Check 2: Test Suite ==="
    $pytestResult = Invoke-Pytest
    $results["pytest"] = $pytestResult
    if (-not $pytestResult.success) {
        $overallSuccess = $false
    }

    # Check 3: Small derivation
    Write-Log "=== Check 3: Small Derivation ==="
    $deriveResult = Invoke-SmallDerivation $config
    $results["small_derivation"] = $deriveResult
    if (-not $deriveResult.success) {
        $overallSuccess = $false
    }

    # Check 4: Metrics endpoint
    Write-Log "=== Check 4: Metrics Endpoint ==="
    $metricsResult = Test-MetricsEndpoint $config
    $results["metrics"] = $metricsResult
    if (-not $metricsResult.success) {
        $overallSuccess = $false
    }

    # Generate summary
    $summary = New-SanitySummary $results $script:logFile

    # Report results
    Write-Log "=== Sanity Check Summary ==="
    Write-Log "Overall Success: $(if ($summary.overall_success) { 'PASSED' } else { 'FAILED' })"
    Write-Log ""

    foreach ($check in $summary.checks.Keys) {
        $checkInfo = $summary.checks[$check]
        $status = if ($checkInfo.success) { "PASSED" } else { "FAILED" }
        $icon = if ($checkInfo.success) { "✅" } else { "❌" }
        Write-Log "$icon $check`: $status"
    }

    if ($summary.metrics) {
        Write-Log ""
        Write-Log "=== Current System State ==="
        Write-Log "Total statements: $($summary.metrics.total_statements)"
        Write-Log "Success rate: $($summary.metrics.success_rate)%"
        Write-Log "Queue length: $($summary.metrics.queue_length)"
    }

    Write-Log ""
    Write-Log "Log file: $script:logFile"

    if ($summary.overall_success) {
        Write-Log "Sanity check PASSED - All systems operational" "INFO"
        exit 0
    } else {
        Write-Log "Sanity check FAILED - Some checks failed" "ERROR"
        exit 1
    }

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
