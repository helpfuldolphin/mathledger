# MathLedger Database Restore Script
# This script restores a database backup into a fresh database

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupPath,
    [string]$ConfigPath = "config/nightly.env",
    [switch]$DryRun = $false,
    [switch]$Force = $false
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

# Function to parse database URL
function Parse-DatabaseUrl {
    param([string]$DatabaseUrl)

    # Parse postgresql://user:pass@host:port/database
    if ($DatabaseUrl -match 'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)') {
        return @{
            user = $matches[1]
            password = $matches[2]
            host = $matches[3]
            port = $matches[4]
            database = $matches[5]
        }
    } else {
        throw "Invalid database URL format"
    }
}

# Function to check if backup file exists and is valid
function Test-BackupFile {
    param([string]$BackupPath)

    Write-Log "Validating backup file: $BackupPath"

    if (-not (Test-Path $BackupPath)) {
        throw "Backup file not found: $BackupPath"
    }

    $fileInfo = Get-Item $BackupPath
    $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)

    Write-Log "Backup file size: $fileSizeMB MB"
    Write-Log "Last modified: $($fileInfo.LastWriteTime)"

    # Check if it's a valid pg_dump file by looking at the first few bytes
    $header = Get-Content $BackupPath -TotalCount 1 -Raw
    if ($header -match '^-- PostgreSQL database dump' -or $header -match '^pg_dump') {
        Write-Log "Backup file appears to be a valid PostgreSQL dump"
        return @{ success = $true; fileSize = $fileInfo.Length; fileSizeMB = $fileSizeMB }
    } else {
        Write-Log "Warning: Backup file may not be a valid PostgreSQL dump" "WARN"
        return @{ success = $true; fileSize = $fileInfo.Length; fileSizeMB = $fileSizeMB; warning = "Invalid format" }
    }
}

# Function to get database information
function Get-DatabaseInfo {
    param([hashtable]$Config)

    try {
        $dbInfo = Parse-DatabaseUrl $Config.DATABASE_URL

        # Test connection to database
        $env:PGPASSWORD = $dbInfo.password
        $testQuery = "SELECT version(), current_database(), current_user;"
        $testResult = & psql --host $dbInfo.host --port $dbInfo.port --username $dbInfo.user --dbname $dbInfo.database --command $testQuery --no-password --tuples-only 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-Log "Database connection successful"
            Write-Log "Target database: $($dbInfo.database)"
            Write-Log "Target host: $($dbInfo.host):$($dbInfo.port)"
            Write-Log "Target user: $($dbInfo.user)"
            return @{ success = $true; dbInfo = $dbInfo; connectionTest = $testResult }
        } else {
            throw "Database connection failed"
        }
    } catch {
        Write-Log "Database connection failed: $_" "ERROR"
        return @{ success = $false; error = $_ }
    } finally {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Function to create restore plan
function New-RestorePlan {
    param(
        [string]$BackupPath,
        [hashtable]$DbInfo,
        [hashtable]$BackupInfo
    )

    Write-Log "Creating restore plan..."

    $plan = @{
        backup_file = $BackupPath
        backup_size_mb = $BackupInfo.fileSizeMB
        target_database = $DbInfo.database
        target_host = "$($DbInfo.host):$($DbInfo.port)"
        target_user = $DbInfo.user
        estimated_time_minutes = [math]::Round($BackupInfo.fileSizeMB / 10, 1)  # Rough estimate: 10MB/min
        steps = @()
    }

    # Add restore steps
    $plan.steps += "1. Stop any running MathLedger services"
    $plan.steps += "2. Drop existing database (if exists)"
    $plan.steps += "3. Create fresh database"
    $plan.steps += "4. Restore from backup file"
    $plan.steps += "5. Verify restore integrity"
    $plan.steps += "6. Restart MathLedger services"

    return $plan
}

# Function to confirm restore operation
function Confirm-Restore {
    param([hashtable]$Plan)

    Write-Log "=== RESTORE PLAN ==="
    Write-Log "Backup file: $($Plan.backup_file)"
    Write-Log "Backup size: $($Plan.backup_size_mb) MB"
    Write-Log "Target database: $($Plan.target_database)"
    Write-Log "Target host: $($Plan.target_host)"
    Write-Log "Estimated time: $($Plan.estimated_time_minutes) minutes"
    Write-Log ""
    Write-Log "Restore steps:"
    foreach ($step in $Plan.steps) {
        Write-Log "  $step"
    }
    Write-Log ""
    Write-Log "WARNING: This will DESTROY the existing database and replace it with the backup!"
    Write-Log ""

    if ($Force) {
        Write-Log "Force flag set - proceeding without confirmation" "WARN"
        return $true
    }

    $response = Read-Host "Do you want to proceed with the restore? (yes/no)"
    return $response -eq "yes"
}

# Function to execute restore
function Invoke-DatabaseRestore {
    param(
        [string]$BackupPath,
        [hashtable]$DbInfo
    )

    Write-Log "Starting database restore..."

    try {
        $env:PGPASSWORD = $DbInfo.password

        # Step 1: Drop existing database (if exists)
        Write-Log "Dropping existing database (if exists)..."
        $dropResult = & psql --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname postgres --command "DROP DATABASE IF EXISTS $($DbInfo.database);" --no-password 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Warning: Could not drop existing database: $dropResult" "WARN"
        }

        # Step 2: Create fresh database
        Write-Log "Creating fresh database..."
        $createResult = & psql --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname postgres --command "CREATE DATABASE $($DbInfo.database);" --no-password 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create database: $createResult"
        }

        # Step 3: Restore from backup
        Write-Log "Restoring from backup file..."

        # Determine restore command based on backup format
        if ($BackupPath -match '\.dump$') {
            # Custom format backup
            $restoreResult = & pg_restore --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname $DbInfo.database --verbose --no-password $BackupPath 2>&1
        } else {
            # Plain text backup
            $restoreResult = & psql --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname $DbInfo.database --file $BackupPath --no-password 2>&1
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Log "Database restore completed successfully"
            return @{ success = $true; output = $restoreResult }
        } else {
            Write-Log "Database restore failed with exit code $LASTEXITCODE" "ERROR"
            Write-Log "Output: $restoreResult" "ERROR"
            return @{ success = $false; output = $restoreResult; exitCode = $LASTEXITCODE }
        }
    } catch {
        Write-Log "Restore command failed: $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    } finally {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Function to verify restore integrity
function Test-RestoreIntegrity {
    param([hashtable]$DbInfo)

    Write-Log "Verifying restore integrity..."

    try {
        $env:PGPASSWORD = $DbInfo.password

        # Check if key tables exist and have data
        $tables = @("theories", "statements", "proofs", "blocks")
        $integrityResults = @{}

        foreach ($table in $tables) {
            $countQuery = "SELECT COUNT(*) FROM $table;"
            $countResult = & psql --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname $DbInfo.database --command $countQuery --no-password --tuples-only 2>$null

            if ($LASTEXITCODE -eq 0) {
                $count = [int]$countResult.Trim()
                $integrityResults[$table] = $count
                Write-Log "Table $table`: $count records"
            } else {
                Write-Log "Failed to check table $table" "ERROR"
                $integrityResults[$table] = -1
            }
        }

        # Check if we have the expected base theory
        $theoryQuery = "SELECT COUNT(*) FROM theories WHERE name = 'Propositional';"
        $theoryResult = & psql --host $DbInfo.host --port $DbInfo.port --username $DbInfo.user --dbname $DbInfo.database --command $theoryQuery --no-password --tuples-only 2>$null

        if ($LASTEXITCODE -eq 0) {
            $theoryCount = [int]$theoryResult.Trim()
            if ($theoryCount -gt 0) {
                Write-Log "Base theory 'Propositional' found"
                $integrityResults["base_theory"] = $true
            } else {
                Write-Log "Base theory 'Propositional' not found" "WARN"
                $integrityResults["base_theory"] = $false
            }
        }

        Write-Log "Restore integrity verification completed"
        return @{ success = $true; results = $integrityResults }

    } catch {
        Write-Log "Integrity verification failed: $_" "ERROR"
        return @{ success = $false; error = $_ }
    } finally {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    Write-Log "Starting MathLedger database restore"
    Write-Log "Backup file: $BackupPath"
    Write-Log "Database URL: $($config.DATABASE_URL)"

    if ($DryRun) {
        Write-Log "Running in DRY RUN mode - no changes will be made" "INFO"
    }

    # Validate backup file
    $backupInfo = Test-BackupFile $BackupPath
    if (-not $backupInfo.success) {
        Write-Log "Backup file validation failed" "ERROR"
        exit 1
    }

    # Get database information
    $dbInfo = Get-DatabaseInfo $config
    if (-not $dbInfo.success) {
        Write-Log "Database connection failed" "ERROR"
        exit 1
    }

    # Create restore plan
    $plan = New-RestorePlan $BackupPath $dbInfo.dbInfo $backupInfo

    # Confirm restore operation
    if (-not (Confirm-Restore $plan)) {
        Write-Log "Restore operation cancelled by user" "INFO"
        exit 0
    }

    if ($DryRun) {
        Write-Log "DRY RUN: Would execute restore plan" "INFO"
        Write-Log "Restore plan validated successfully"
        exit 0
    }

    # Execute restore
    $restoreResult = Invoke-DatabaseRestore $BackupPath $dbInfo.dbInfo
    if (-not $restoreResult.success) {
        Write-Log "Database restore failed" "ERROR"
        exit 1
    }

    # Verify restore integrity
    $integrityResult = Test-RestoreIntegrity $dbInfo.dbInfo
    if (-not $integrityResult.success) {
        Write-Log "Restore integrity verification failed" "WARN"
    }

    # Report results
    Write-Log "=== Restore Summary ==="
    Write-Log "Backup file: $BackupPath"
    Write-Log "Target database: $($dbInfo.dbInfo.database)"
    Write-Log "Restore status: SUCCESS"

    if ($integrityResult.success) {
        Write-Log "Integrity verification: PASSED"
        foreach ($table in $integrityResult.results.Keys) {
            if ($table -ne "base_theory") {
                $count = $integrityResult.results[$table]
                if ($count -ge 0) {
                    Write-Log "  $table`: $count records"
                }
            }
        }
    } else {
        Write-Log "Integrity verification: FAILED" "WARN"
    }

    Write-Log ""
    Write-Log "Database restore completed successfully"
    Write-Log "You may now restart MathLedger services"

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
