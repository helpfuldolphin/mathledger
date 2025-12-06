# MathLedger Health Check Script
# This script checks the health of all MathLedger services

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
}

# Function to check PostgreSQL health
function Test-PostgreSQLHealth {
    param([hashtable]$Config)

    Write-Log "Checking PostgreSQL health..."

    try {
        # Test basic connection
        $connectionTest = docker exec infra-postgres-1 psql -U ml -d mathledger -c "SELECT 1;" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "PostgreSQL connection failed"
        }

        # Check if container is running
        $containerStatus = docker ps --filter "name=infra-postgres-1" --format "{{.Status}}" 2>$null
        if (-not $containerStatus) {
            throw "PostgreSQL container not running"
        }

        # Check database size
        $sizeQuery = "SELECT pg_size_pretty(pg_database_size('mathledger'));"
        $sizeResult = docker exec infra-postgres-1 psql -U ml -d mathledger -t -c $sizeQuery 2>$null
        if ($LASTEXITCODE -eq 0) {
            $size = $sizeResult.Trim()
            Write-Log "PostgreSQL: OK (Database size: $size)"
        } else {
            Write-Log "PostgreSQL: OK (Could not get size)" "WARN"
        }

        # Check for long-running queries
        $longQueriesQuery = "SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';"
        $longQueriesResult = docker exec infra-postgres-1 psql -U ml -d mathledger -t -c $longQueriesQuery 2>$null
        if ($LASTEXITCODE -eq 0) {
            $longQueryCount = [int]$longQueriesResult.Trim()
            if ($longQueryCount -gt 0) {
                Write-Log "PostgreSQL: WARNING - $longQueryCount long-running queries" "WARN"
            }
        }

        return @{ success = $true; status = "healthy" }
    } catch {
        Write-Log "PostgreSQL: FAILED - $_" "ERROR"
        return @{ success = $false; status = "unhealthy"; error = $_ }
    }
}

# Function to check Redis health
function Test-RedisHealth {
    param([hashtable]$Config)

    Write-Log "Checking Redis health..."

    try {
        # Test basic connection
        $pingResult = docker exec infra-redis-1 redis-cli ping 2>$null
        if ($pingResult -ne "PONG") {
            throw "Redis ping failed"
        }

        # Check if container is running
        $containerStatus = docker ps --filter "name=infra-redis-1" --format "{{.Status}}" 2>$null
        if (-not $containerStatus) {
            throw "Redis container not running"
        }

        # Check memory usage
        $memoryInfo = docker exec infra-redis-1 redis-cli info memory 2>$null
        if ($LASTEXITCODE -eq 0) {
            $usedMemory = ($memoryInfo | Select-String "used_memory_human:" | ForEach-Object { $_.Line -split ":" | Select-Object -Last 1 }).Trim()
            Write-Log "Redis: OK (Memory used: $usedMemory)"
        } else {
            Write-Log "Redis: OK (Could not get memory info)" "WARN"
        }

        # Check queue length
        $queueLength = docker exec infra-redis-1 redis-cli llen $Config.QUEUE_KEY 2>$null
        if ($LASTEXITCODE -eq 0) {
            $queueLen = [int]$queueLength
            Write-Log "Redis: OK (Queue length: $queueLen)"
            if ($queueLen -gt 1000) {
                Write-Log "Redis: WARNING - High queue length: $queueLen" "WARN"
            }
        }

        return @{ success = $true; status = "healthy" }
    } catch {
        Write-Log "Redis: FAILED - $_" "ERROR"
        return @{ success = $false; status = "unhealthy"; error = $_ }
    }
}

# Function to check FastAPI health
function Test-FastAPIHealth {
    param([hashtable]$Config)

    Write-Log "Checking FastAPI health..."

    try {
        # Test health endpoint
        $healthResponse = Invoke-RestMethod -Uri $Config.HEALTH_URL -Method GET -TimeoutSec 10
        if ($healthResponse.ok -ne $true) {
            throw "Health endpoint returned unhealthy status"
        }

        # Test metrics endpoint
        $metricsResponse = Invoke-RestMethod -Uri $Config.METRICS_URL -Method GET -TimeoutSec 10
        if (-not $metricsResponse) {
            throw "Metrics endpoint not accessible"
        }

        # Check if we can get basic metrics
        if ($metricsResponse.statements -and $metricsResponse.proofs) {
            $totalStatements = $metricsResponse.statements.total
            $successRate = $metricsResponse.proofs.success_rate
            Write-Log "FastAPI: OK (Statements: $totalStatements, Success rate: $successRate%)"
        } else {
            Write-Log "FastAPI: OK (Could not parse metrics)" "WARN"
        }

        return @{ success = $true; status = "healthy"; metrics = $metricsResponse }
    } catch {
        Write-Log "FastAPI: FAILED - $_" "ERROR"
        return @{ success = $false; status = "unhealthy"; error = $_ }
    }
}

# Function to check Lean project health
function Test-LeanHealth {
    param([hashtable]$Config)

    Write-Log "Checking Lean project health..."

    try {
        # Check if Lean project directory exists
        if (-not (Test-Path $Config.LEAN_PROJECT_DIR)) {
            throw "Lean project directory not found: $($Config.LEAN_PROJECT_DIR)"
        }

        # Check if lakefile.lean exists
        $lakefile = Join-Path $Config.LEAN_PROJECT_DIR "lakefile.lean"
        if (-not (Test-Path $lakefile)) {
            throw "lakefile.lean not found"
        }

        # Try to run lake build (quick check)
        Push-Location $Config.LEAN_PROJECT_DIR
        try {
            $buildResult = & lake build --quiet 2>&1
            $exitCode = $LASTEXITCODE

            if ($exitCode -eq 0) {
                Write-Log "Lean: OK (Project builds successfully)"
                return @{ success = $true; status = "healthy" }
            } else {
                Write-Log "Lean: WARNING - Build issues detected" "WARN"
                if ($Verbose) {
                    Write-Log "Build output: $buildResult" "WARN"
                }
                return @{ success = $true; status = "degraded"; warning = "Build issues" }
            }
        } finally {
            Pop-Location
        }
    } catch {
        Write-Log "Lean: FAILED - $_" "ERROR"
        return @{ success = $false; status = "unhealthy"; error = $_ }
    }
}

# Function to check Docker containers
function Test-DockerHealth {
    param([hashtable]$Config)

    Write-Log "Checking Docker containers..."

    try {
        # Check if Docker is running
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker not accessible"
        }

        # Check required containers
        $requiredContainers = @("infra-postgres-1", "infra-redis-1")
        $containerStatus = @{}

        foreach ($container in $requiredContainers) {
            $status = docker ps --filter "name=$container" --format "{{.Status}}" 2>$null
            if ($status) {
                $containerStatus[$container] = $status
                Write-Log "Container $container`: $status"
            } else {
                $containerStatus[$container] = "Not running"
                Write-Log "Container $container`: Not running" "ERROR"
            }
        }

        $allRunning = $containerStatus.Values | Where-Object { $_ -notlike "*Not running*" } | Measure-Object | Select-Object -ExpandProperty Count
        if ($allRunning -eq $requiredContainers.Count) {
            Write-Log "Docker: OK (All required containers running)"
            return @{ success = $true; status = "healthy"; containers = $containerStatus }
        } else {
            Write-Log "Docker: WARNING - Some containers not running" "WARN"
            return @{ success = $true; status = "degraded"; containers = $containerStatus }
        }
    } catch {
        Write-Log "Docker: FAILED - $_" "ERROR"
        return @{ success = $false; status = "unhealthy"; error = $_ }
    }
}

# Function to generate health summary
function New-HealthSummary {
    param([hashtable]$Results)

    $summary = @{
        timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        overall_status = "healthy"
        services = @{}
    }

    $unhealthyServices = 0
    $degradedServices = 0

    foreach ($service in $Results.Keys) {
        $result = $Results[$service]
        $summary.services[$service] = @{
            status = $result.status
            success = $result.success
        }

        if (-not $result.success) {
            $unhealthyServices++
        } elseif ($result.status -eq "degraded") {
            $degradedServices++
        }
    }

    if ($unhealthyServices -gt 0) {
        $summary.overall_status = "unhealthy"
    } elseif ($degradedServices -gt 0) {
        $summary.overall_status = "degraded"
    }

    return $summary
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    Write-Log "Starting MathLedger health check"
    Write-Log "Configuration: HealthURL=$($config.HEALTH_URL), MetricsURL=$($config.METRICS_URL)"

    $results = @{}

    # Check Docker containers
    $results["docker"] = Test-DockerHealth $config

    # Check PostgreSQL
    $results["postgresql"] = Test-PostgreSQLHealth $config

    # Check Redis
    $results["redis"] = Test-RedisHealth $config

    # Check FastAPI
    $results["fastapi"] = Test-FastAPIHealth $config

    # Check Lean project
    $results["lean"] = Test-LeanHealth $config

    # Generate summary
    $summary = New-HealthSummary $results

    # Report results
    Write-Log "=== Health Check Summary ==="
    Write-Log "Overall Status: $($summary.overall_status.ToUpper())"
    Write-Log ""

    foreach ($service in $summary.services.Keys) {
        $serviceInfo = $summary.services[$service]
        $status = $serviceInfo.status.ToUpper()
        $icon = if ($serviceInfo.success) { "✅" } else { "❌" }
        Write-Log "$icon $service`: $status"
    }

    Write-Log ""

    if ($summary.overall_status -eq "unhealthy") {
        Write-Log "Health check FAILED - Some services are unhealthy" "ERROR"
        exit 1
    } elseif ($summary.overall_status -eq "degraded") {
        Write-Log "Health check WARNING - Some services are degraded" "WARN"
        exit 2
    } else {
        Write-Log "Health check PASSED - All services are healthy" "INFO"
        exit 0
    }

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
