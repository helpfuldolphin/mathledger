# MathLedger Database Maintenance Script
# This script performs database maintenance tasks like VACUUM and cleanup

param(
    [string]$ConfigPath = "config/nightly.env",
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

# Function to execute SQL command
function Invoke-SqlCommand {
    param(
        [string]$Query,
        [string]$Description,
        [hashtable]$Config
    )

    Write-Log "Executing: $Description"

    if ($DryRun) {
        Write-Log "DRY RUN: Would execute SQL: $Query" "INFO"
        return @{ success = $true; output = "Dry run mode" }
    }

    try {
        $result = docker exec infra-postgres-1 psql -U ml -d mathledger -c $Query 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log "Success: $Description"
            Write-Log "Output: $result"
            return @{ success = $true; output = $result }
        } else {
            Write-Log "Failed: $Description" "ERROR"
            Write-Log "Output: $result" "ERROR"
            return @{ success = $false; output = $result; exitCode = $exitCode }
        }
    } catch {
        Write-Log "SQL command failed: $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    }
}

# Function to get table statistics
function Get-TableStats {
    param([hashtable]$Config)

    Write-Log "Gathering table statistics..."

    $tables = @("statements", "proofs", "dependencies", "blocks", "lemma_cache")
    $stats = @{}

    foreach ($table in $tables) {
        $countQuery = "SELECT COUNT(*) FROM $table;"
        $result = Invoke-SqlCommand $countQuery "Count records in $table" $Config

        if ($result.success) {
            $count = ($result.output -split "`n" | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1).Trim()
            $stats[$table] = [int]$count
            Write-Log "$table`: $count records"
        } else {
            Write-Log "Failed to get count for $table" "WARN"
            $stats[$table] = -1
        }
    }

    return $stats
}

# Function to run VACUUM ANALYZE
function Invoke-VacuumAnalyze {
    param([hashtable]$Config)

    if ($Config.VACUUM_ANALYZE -ne "true") {
        Write-Log "VACUUM ANALYZE disabled in configuration" "INFO"
        return @{ success = $true; output = "Skipped" }
    }

    Write-Log "Running VACUUM ANALYZE on all tables..."

    $tables = @("statements", "proofs", "dependencies", "blocks", "lemma_cache")
    $results = @{}

    foreach ($table in $tables) {
        $query = "VACUUM ANALYZE $table;"
        $result = Invoke-SqlCommand $query "VACUUM ANALYZE $table" $Config
        $results[$table] = $result
    }

    $allSuccess = $results.Values | Where-Object { -not $_.success } | Measure-Object | Select-Object -ExpandProperty Count
    if ($allSuccess -eq 0) {
        Write-Log "VACUUM ANALYZE completed successfully on all tables"
        return @{ success = $true; results = $results }
    } else {
        Write-Log "VACUUM ANALYZE failed on some tables" "WARN"
        return @{ success = $false; results = $results }
    }
}

# Function to prune failed proofs
function Invoke-PruneFailedProofs {
    param([hashtable]$Config)

    $days = [int]$Config.PRUNE_FAILED_PROOFS_DAYS
    if ($days -le 0) {
        Write-Log "Failed proof pruning disabled (days = $days)" "INFO"
        return @{ success = $true; output = "Skipped" }
    }

    Write-Log "Pruning failed proofs older than $days days..."

    # First, get count of proofs to be deleted
    $countQuery = "SELECT COUNT(*) FROM proofs WHERE success = false AND created_at < NOW() - INTERVAL '$days days';"
    $countResult = Invoke-SqlCommand $countQuery "Count failed proofs to prune" $Config

    if ($countResult.success) {
        $count = ($countResult.output -split "`n" | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1).Trim()
        Write-Log "Found $count failed proofs to prune"

        if ([int]$count -gt 0) {
            $deleteQuery = "DELETE FROM proofs WHERE success = false AND created_at < NOW() - INTERVAL '$days days';"
            $deleteResult = Invoke-SqlCommand $deleteQuery "Delete failed proofs older than $days days" $Config

            if ($deleteResult.success) {
                Write-Log "Successfully pruned $count failed proofs"
                return @{ success = $true; prunedCount = [int]$count }
            } else {
                Write-Log "Failed to prune proofs: $($deleteResult.output)" "ERROR"
                return @{ success = $false; error = $deleteResult.output }
            }
        } else {
            Write-Log "No failed proofs to prune"
            return @{ success = $true; prunedCount = 0 }
        }
    } else {
        Write-Log "Failed to count proofs for pruning: $($countResult.output)" "ERROR"
        return @{ success = $false; error = $countResult.output }
    }
}

# Function to update table statistics
function Invoke-UpdateStatistics {
    param([hashtable]$Config)

    Write-Log "Updating table statistics..."

    $tables = @("statements", "proofs", "dependencies", "blocks", "lemma_cache")
    $results = @{}

    foreach ($table in $tables) {
        $query = "ANALYZE $table;"
        $result = Invoke-SqlCommand $query "ANALYZE $table" $Config
        $results[$table] = $result
    }

    $allSuccess = $results.Values | Where-Object { -not $_.success } | Measure-Object | Select-Object -ExpandProperty Count
    if ($allSuccess -eq 0) {
        Write-Log "Statistics updated successfully for all tables"
        return @{ success = $true; results = $results }
    } else {
        Write-Log "Statistics update failed for some tables" "WARN"
        return @{ success = $false; results = $results }
    }
}

# Function to check database health
function Test-DatabaseHealth {
    param([hashtable]$Config)

    Write-Log "Checking database health..."

    # Check connection
    $connectionQuery = "SELECT 1;"
    $connectionResult = Invoke-SqlCommand $connectionQuery "Test database connection" $Config

    if (-not $connectionResult.success) {
        Write-Log "Database connection failed" "ERROR"
        return $false
    }

    # Check for long-running queries
    $longQueriesQuery = "SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';"
    $longQueriesResult = Invoke-SqlCommand $longQueriesQuery "Check for long-running queries" $Config

    if ($longQueriesResult.success) {
        $longQueryCount = ($longQueriesResult.output -split "`n" | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1).Trim()
        if ([int]$longQueryCount -gt 0) {
            Write-Log "Warning: $longQueryCount long-running queries detected" "WARN"
        }
    }

    # Check database size
    $sizeQuery = "SELECT pg_size_pretty(pg_database_size('mathledger'));"
    $sizeResult = Invoke-SqlCommand $sizeQuery "Get database size" $Config

    if ($sizeResult.success) {
        $size = ($sizeResult.output -split "`n" | Where-Object { $_ -match '\d+\s+\w+' } | Select-Object -First 1).Trim()
        Write-Log "Database size: $size"
    }

    Write-Log "Database health check completed"
    return $true
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    Write-Log "Starting MathLedger database maintenance"
    Write-Log "Configuration: VACUUM=$($config.VACUUM_ANALYZE), PruneDays=$($config.PRUNE_FAILED_PROOFS_DAYS)"

    if ($DryRun) {
        Write-Log "Running in DRY RUN mode - no changes will be made" "INFO"
    }

    # Check database health
    if (-not (Test-DatabaseHealth $config)) {
        Write-Log "Database health check failed. Aborting maintenance." "ERROR"
        exit 1
    }

    # Get initial table statistics
    $initialStats = Get-TableStats $config

    # Run VACUUM ANALYZE
    $vacuumResult = Invoke-VacuumAnalyze $config
    if (-not $vacuumResult.success) {
        Write-Log "VACUUM ANALYZE failed on some tables" "WARN"
    }

    # Prune failed proofs
    $pruneResult = Invoke-PruneFailedProofs $config
    if (-not $pruneResult.success) {
        Write-Log "Failed proof pruning failed: $($pruneResult.error)" "WARN"
    } elseif ($pruneResult.prunedCount -gt 0) {
        Write-Log "Pruned $($pruneResult.prunedCount) failed proofs"
    }

    # Update statistics
    $statsResult = Invoke-UpdateStatistics $config
    if (-not $statsResult.success) {
        Write-Log "Statistics update failed on some tables" "WARN"
    }

    # Get final table statistics
    $finalStats = Get-TableStats $config

    # Report maintenance summary
    Write-Log "=== Database Maintenance Summary ==="
    Write-Log "VACUUM ANALYZE: $(if ($vacuumResult.success) { 'SUCCESS' } else { 'PARTIAL/FAILED' })"
    Write-Log "Failed proof pruning: $(if ($pruneResult.success) { "SUCCESS ($($pruneResult.prunedCount) pruned)" } else { 'FAILED' })"
    Write-Log "Statistics update: $(if ($statsResult.success) { 'SUCCESS' } else { 'PARTIAL/FAILED' })"

    Write-Log "=== Table Statistics ==="
    foreach ($table in $initialStats.Keys) {
        $initial = $initialStats[$table]
        $final = $finalStats[$table]
        if ($initial -ge 0 -and $final -ge 0) {
            $change = $final - $initial
            Write-Log "$table`: $initial -> $final ($(if ($change -gt 0) { '+' } else { '' })$change)"
        }
    }

    Write-Log "Database maintenance completed"

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
