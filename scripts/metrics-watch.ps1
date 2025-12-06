# MathLedger Metrics Watchdog Script
# This script monitors system metrics and sends alerts when thresholds are exceeded

param(
    [string]$ConfigPath = "config/nightly.env",
    [int]$DurationMinutes = 0,
    [switch]$Continuous = $false
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

# Function to get metrics from the API
function Get-Metrics {
    param([hashtable]$Config)

    try {
        $response = Invoke-RestMethod -Uri $Config.METRICS_URL -Method GET -TimeoutSec 10
        return @{ success = $true; data = $response }
    } catch {
        return @{ success = $false; error = $_.Exception.Message }
    }
}

# Function to check if metrics indicate problems
function Test-MetricsThresholds {
    param(
        [hashtable]$Config,
        [hashtable]$Metrics
    )

    $issues = @()

    # Check success rate
    $successRate = $Metrics.data.proofs.success_rate
    $threshold = [int]$Config.SUCCESS_RATE_THRESHOLD

    if ($successRate -lt $threshold) {
        $issues += "Success rate $successRate% is below threshold $threshold%"
    }

    # Check for high error rates
    $proofStatus = $Metrics.data.proofs.by_status
    $totalProofs = ($proofStatus | Get-Member -MemberType NoteProperty | Measure-Object).Count

    if ($proofStatus.error -and $proofStatus.error -gt 0) {
        $errorRate = ($proofStatus.error / $totalProofs) * 100
        if ($errorRate -gt 10) {  # More than 10% errors
            $issues += "Error rate $errorRate% is above 10%"
        }
    }

    # Check queue length (if it's growing too fast)
    $queueLength = $Metrics.data.queue.length
    if ($queueLength -gt 1000) {
        $issues += "Queue length $queueLength is above 1000"
    }

    # Check recent activity
    $recentProofs = $Metrics.data.proofs.recent_hour
    if ($recentProofs -eq 0) {
        $issues += "No recent proof activity in the last hour"
    }

    return $issues
}

# Function to send alert
function Send-Alert {
    param(
        [hashtable]$Config,
        [string[]]$Issues
    )

    if ($Config.ALERT_ENABLED -ne "true" -or [string]::IsNullOrEmpty($Config.ALERT_WEBHOOK)) {
        Write-Log "Alerting disabled or webhook not configured" "WARN"
        return
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $issuesText = $Issues -join "; "

    # Create Discord/Slack compatible payload
    $payload = @{
        text = "🚨 MathLedger Alert - $timestamp"
        attachments = @(
            @{
                color = "danger"
                fields = @(
                    @{
                        title = "Issues Detected"
                        value = $issuesText
                        short = $false
                    }
                )
                timestamp = [int][double]::Parse((Get-Date -UFormat %s))
            }
        )
    } | ConvertTo-Json -Depth 3

    try {
        $response = Invoke-RestMethod -Uri $Config.ALERT_WEBHOOK -Method POST -Body $payload -ContentType "application/json" -TimeoutSec 10
        Write-Log "Alert sent successfully"
    } catch {
        Write-Log "Failed to send alert: $($_.Exception.Message)" "ERROR"
    }
}

# Function to run one monitoring cycle
function Invoke-MonitoringCycle {
    param([hashtable]$Config)

    Write-Log "Checking metrics..."

    $metricsResult = Get-Metrics $Config
    if (-not $metricsResult.success) {
        Write-Log "Failed to get metrics: $($metricsResult.error)" "ERROR"
        return $false
    }

    $issues = Test-MetricsThresholds $Config $metricsResult
    if ($issues.Count -gt 0) {
        Write-Log "Issues detected: $($issues -join '; ')" "WARN"
        Send-Alert $Config $issues
        return $false
    } else {
        Write-Log "All metrics within normal ranges"
        return $true
    }
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    Write-Log "Starting MathLedger metrics monitoring"
    Write-Log "Metrics URL: $($config.METRICS_URL)"
    Write-Log "Success rate threshold: $($config.SUCCESS_RATE_THRESHOLD)%"
    Write-Log "Poll interval: $($config.POLL_INTERVAL_MINUTES) minutes"

    if ($Continuous) {
        Write-Log "Running in continuous mode (Ctrl+C to stop)"
        $startTime = Get-Date
        $cycleCount = 0

        while ($true) {
            $cycleCount++
            Write-Log "=== Monitoring Cycle $cycleCount ==="

            $healthy = Invoke-MonitoringCycle $config

            if ($DurationMinutes -gt 0) {
                $elapsed = (Get-Date) - $startTime
                if ($elapsed.TotalMinutes -ge $DurationMinutes) {
                    Write-Log "Duration limit reached. Stopping monitoring."
                    break
                }
            }

            Write-Log "Waiting $($config.POLL_INTERVAL_MINUTES) minutes until next check..."
            Start-Sleep -Seconds ([int]$config.POLL_INTERVAL_MINUTES * 60)
        }
    } else {
        # Single check
        $healthy = Invoke-MonitoringCycle $config
        if ($healthy) {
            Write-Log "System is healthy"
            exit 0
        } else {
            Write-Log "System issues detected"
            exit 1
        }
    }

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
