# MathLedger Nightly Operations Master Script
# This script orchestrates all nightly operations in the correct order

param(
    [string]$ConfigPath = "config/nightly.env",
    [switch]$SkipHealthCheck = $false,
    [switch]$SkipDerivation = $false,
    [switch]$SkipSnapshot = $false,
    [switch]$SkipMaintenance = $false,
    [switch]$DryRun = $false
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
}

# Function to run a script and capture results
function Invoke-Script {
    param(
        [string]$ScriptPath,
        [string]$Description,
        [hashtable]$Config,
        [string[]]$Arguments = @()
    )

    Write-Log "Starting: $Description"
    Write-Log "Script: $ScriptPath"

    if ($DryRun) {
        Write-Log "DRY RUN: Would execute $ScriptPath with arguments: $($Arguments -join ' ')" "INFO"
        return @{ success = $true; output = "Dry run mode" }
    }

    try {
        $scriptArgs = @($ScriptPath) + $Arguments
        $result = & powershell -File @scriptArgs 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log "Completed: $Description"
            if ($Verbose -and $result) {
                Write-Log "Output: $result"
            }
            return @{ success = $true; output = $result; exitCode = $exitCode }
        } else {
            Write-Log "Failed: $Description (Exit code: $exitCode)" "ERROR"
            if ($result) {
                Write-Log "Output: $result" "ERROR"
            }
            return @{ success = $false; output = $result; exitCode = $exitCode }
        }
    } catch {
        Write-Log "Exception in $Description : $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    }
}

# Function to create run summary
function New-RunSummary {
    param(
        [hashtable]$Results,
        [string]$LogDir
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $summaryFile = Join-Path $LogDir "nightly-run-summary-$((Get-Date).ToString('yyyyMMdd-HHmmss')).json"

    $summary = @{
        timestamp = $timestamp
        overall_success = $true
        steps = @{}
        duration_minutes = 0
        log_dir = $LogDir
    }

    $startTime = $null
    $endTime = Get-Date

    foreach ($step in $Results.Keys) {
        $result = $Results[$step]
        $summary.steps[$step] = @{
            success = $result.success
            exit_code = $result.exitCode
            duration_seconds = if ($result.duration) { $result.duration.TotalSeconds } else { 0 }
        }

        if (-not $result.success) {
            $summary.overall_success = $false
        }

        if ($result.startTime -and (-not $startTime -or $result.startTime -lt $startTime)) {
            $startTime = $result.startTime
        }
    }

    if ($startTime) {
        $summary.duration_minutes = [math]::Round(($endTime - $startTime).TotalMinutes, 2)
    }

    $summary | ConvertTo-Json -Depth 3 | Out-File -FilePath $summaryFile -Encoding UTF8
    Write-Log "Run summary written to: $summaryFile"

    return $summary
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    # Set up logging
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $logDir = Join-Path $config.LOGS_DIR "nightly-$timestamp"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    Write-Log "Starting MathLedger nightly operations"
    Write-Log "Log directory: $logDir"
    Write-Log "Configuration: Steps=$($config.DERIVE_STEPS), Depth=$($config.DERIVE_DEPTH_MAX)"

    if ($DryRun) {
        Write-Log "Running in DRY RUN mode - no changes will be made" "INFO"
    }

    $results = @{}
    $overallSuccess = $true

    # Step 1: Health Check
    if (-not $SkipHealthCheck) {
        Write-Log "=== Step 1: Health Check ==="
        $startTime = Get-Date
        $healthResult = Invoke-Script "scripts/healthcheck.ps1" "Health Check" $config @("-ConfigPath", $ConfigPath, "-Verbose")
        $healthResult.startTime = $startTime
        $healthResult.duration = (Get-Date) - $startTime
        $results["health_check"] = $healthResult

        if (-not $healthResult.success) {
            Write-Log "Health check failed. Aborting nightly operations." "ERROR"
            $overallSuccess = $false
        }
    } else {
        Write-Log "Skipping health check" "INFO"
    }

    # Step 1.5: First Organism Vital Signs
    if ($overallSuccess) {
        Write-Log "=== Step 1.5: First Organism Vital Signs ==="
        $startTime = Get-Date
        try {
            $foResult = & uv run python scripts/run_first_organism.py --standalone --verbose 2>&1
            $foExitCode = $LASTEXITCODE
            $foSuccess = $foExitCode -eq 0

            $results["first_organism"] = @{
                success = $foSuccess
                exitCode = $foExitCode
                startTime = $startTime
                duration = (Get-Date) - $startTime
                output = $foResult
            }

            if ($foSuccess) {
                Write-Log "First Organism test passed"
            } else {
                Write-Log "First Organism test failed (exit code: $foExitCode)" "WARN"
                # Don't fail nightly for FO failure - just log warning
            }
        } catch {
            Write-Log "First Organism test exception: $_" "WARN"
            $results["first_organism"] = @{
                success = $false
                exitCode = 1
                startTime = $startTime
                duration = (Get-Date) - $startTime
                output = $_
            }
        }
    }

    # Step 2: Derivation
    if (-not $SkipDerivation -and $overallSuccess) {
        Write-Log "=== Step 2: Derivation ==="
        $startTime = Get-Date
        $deriveArgs = @("-ConfigPath", $ConfigPath)
        if ($DryRun) { $deriveArgs += "-DryRun" }

        $deriveResult = Invoke-Script "scripts/derive-nightly.ps1" "Nightly Derivation" $config $deriveArgs
        $deriveResult.startTime = $startTime
        $deriveResult.duration = (Get-Date) - $startTime
        $results["derivation"] = $deriveResult

        if (-not $deriveResult.success) {
            Write-Log "Derivation failed. Continuing with remaining steps." "WARN"
            $overallSuccess = $false
        }
    } else {
        Write-Log "Skipping derivation" "INFO"
    }

    # Step 3: Snapshot Export
    if (-not $SkipSnapshot -and $overallSuccess) {
        Write-Log "=== Step 3: Snapshot Export ==="
        $startTime = Get-Date
        $exportArgs = @("-ConfigPath", $ConfigPath, "-OutputDir", $config.EXPORTS_DIR, "-IncludeDependencies")

        $exportResult = Invoke-Script "scripts/export-snapshot.ps1" "Snapshot Export" $config $exportArgs
        $exportResult.startTime = $startTime
        $exportResult.duration = (Get-Date) - $startTime
        $results["snapshot_export"] = $exportResult

        if (-not $exportResult.success) {
            Write-Log "Snapshot export failed. Continuing with remaining steps." "WARN"
        }
    } else {
        Write-Log "Skipping snapshot export" "INFO"
    }

    # Step 4: Database Maintenance
    if (-not $SkipMaintenance) {
        Write-Log "=== Step 4: Database Maintenance ==="
        $startTime = Get-Date
        $maintenanceArgs = @("-ConfigPath", $ConfigPath)
        if ($DryRun) { $maintenanceArgs += "-DryRun" }

        $maintenanceResult = Invoke-Script "scripts/db-maintenance.ps1" "Database Maintenance" $config $maintenanceArgs
        $maintenanceResult.startTime = $startTime
        $maintenanceResult.duration = (Get-Date) - $startTime
        $results["db_maintenance"] = $maintenanceResult

        if (-not $maintenanceResult.success) {
            Write-Log "Database maintenance failed." "WARN"
        }
    } else {
        Write-Log "Skipping database maintenance" "INFO"
    }

    # Step 5: Metrics Collection (Cursor K - Metrics Oracle)
    Write-Log "=== Step 5: Metrics Collection ==="
    $startTime = Get-Date
    try {
        # Run metrics cartographer
        $metricsResult = & uv run python backend/metrics_cartographer.py 2>&1
        $metricsExitCode = $LASTEXITCODE

        # Generate ASCII report
        $reportResult = & uv run python backend/metrics_reporter.py 2>&1
        $reportExitCode = $LASTEXITCODE

        # Generate Markdown report
        $mdResult = & uv run python backend/metrics_md_report.py 2>&1
        $mdExitCode = $LASTEXITCODE

        $metricsSuccess = ($metricsExitCode -eq 0) -and ($reportExitCode -eq 0) -and ($mdExitCode -eq 0)

        $results["metrics_collection"] = @{
            success = $metricsSuccess
            exitCode = $metricsExitCode
            startTime = $startTime
            duration = (Get-Date) - $startTime
            output = @{
                cartographer = $metricsResult
                ascii_report = $reportResult
                markdown_report = $mdResult
            }
        }

        if ($metricsSuccess) {
            Write-Log "Metrics collection completed successfully"
            Write-Log "  - artifacts/metrics/latest.json updated"
            Write-Log "  - artifacts/metrics/latest_report.txt updated"
            Write-Log "  - reports/metrics_*.md updated"
        } else {
            Write-Log "Metrics collection had errors (exit codes: cartographer=$metricsExitCode, report=$reportExitCode, md=$mdExitCode)" "WARN"
        }
    } catch {
        Write-Log "Metrics collection exception: $_" "WARN"
        $results["metrics_collection"] = @{
            success = $false
            exitCode = 1
            startTime = $startTime
            duration = (Get-Date) - $startTime
            output = $_
        }
    }

    # Generate run summary
    $summary = New-RunSummary $results $logDir

    # Report final results
    Write-Log "=== Nightly Operations Summary ==="
    Write-Log "Overall Success: $(if ($summary.overall_success) { 'SUCCESS' } else { 'FAILED' })"
    Write-Log "Duration: $($summary.duration_minutes) minutes"
    Write-Log ""

    foreach ($step in $summary.steps.Keys) {
        $stepInfo = $summary.steps[$step]
        $status = if ($stepInfo.success) { "SUCCESS" } else { "FAILED" }
        $duration = [math]::Round($stepInfo.duration_seconds, 1)
        $icon = if ($stepInfo.success) { "✅" } else { "❌" }
        Write-Log "$icon $step`: $status (${duration}s)"
    }

    Write-Log ""
    Write-Log "Log directory: $logDir"
    Write-Log "Summary file: $($summary.log_dir)/nightly-run-summary-*.json"

    if ($summary.overall_success) {
        Write-Log "Nightly operations completed successfully" "INFO"
        exit 0
    } else {
        Write-Log "Nightly operations completed with errors" "WARN"
        exit 1
    }

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
