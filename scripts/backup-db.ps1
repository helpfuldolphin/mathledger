# MathLedger Database Backup Script
# This script creates a compressed backup of the MathLedger database using pg_dump

param(
    [string]$ConfigPath = "config/nightly.env",
    [string]$OutputPath = "",
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

# Function to create backup
function New-DatabaseBackup {
    param(
        [hashtable]$Config,
        [string]$OutputPath
    )

    Write-Log "Creating database backup..."

    try {
        # Parse database URL
        $dbInfo = Parse-DatabaseUrl $Config.DATABASE_URL

        # Set up environment variables for pg_dump
        $env:PGPASSWORD = $dbInfo.password

        # Build pg_dump command
        $pgDumpArgs = @(
            "--host", $dbInfo.host
            "--port", $dbInfo.port
            "--username", $dbInfo.user
            "--dbname", $dbInfo.database
            "--verbose"
            "--no-password"
        )

        # Add compression if enabled
        if ($Config.BACKUP_COMPRESSION -eq "true") {
            $pgDumpArgs += "--format=custom"
            $pgDumpArgs += "--compress=9"
        } else {
            $pgDumpArgs += "--format=plain"
        }

        # Add output file
        $pgDumpArgs += "--file", $OutputPath

        Write-Log "Running pg_dump with args: $($pgDumpArgs -join ' ')"

        if ($DryRun) {
            Write-Log "DRY RUN: Would execute pg_dump with output to $OutputPath" "INFO"
            return @{ success = $true; output = "Dry run mode" }
        }

        # Execute pg_dump
        $backupResult = & pg_dump @pgDumpArgs 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            # Get file size
            $fileSize = (Get-Item $OutputPath).Length
            $fileSizeMB = [math]::Round($fileSize / 1MB, 2)

            Write-Log "Backup completed successfully"
            Write-Log "Output file: $OutputPath"
            Write-Log "File size: $fileSizeMB MB"

            return @{
                success = $true;
                output = $backupResult;
                filePath = $OutputPath;
                fileSize = $fileSize;
                fileSizeMB = $fileSizeMB
            }
        } else {
            Write-Log "Backup failed with exit code $exitCode" "ERROR"
            Write-Log "Output: $backupResult" "ERROR"
            return @{ success = $false; output = $backupResult; exitCode = $exitCode }
        }
    } catch {
        Write-Log "Backup command failed: $_" "ERROR"
        return @{ success = $false; output = $_; exitCode = 1 }
    } finally {
        # Clear password from environment
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Function to clean up old backups
function Remove-OldBackups {
    param(
        [string]$BackupDir,
        [int]$RetentionDays
    )

    Write-Log "Cleaning up backups older than $RetentionDays days..."

    try {
        $cutoffDate = (Get-Date).AddDays(-$RetentionDays)
        $oldBackups = Get-ChildItem -Path $BackupDir -Filter "*.dump" | Where-Object { $_.LastWriteTime -lt $cutoffDate }

        if ($oldBackups.Count -gt 0) {
            foreach ($backup in $oldBackups) {
                if ($DryRun) {
                    Write-Log "DRY RUN: Would delete $($backup.FullName)" "INFO"
                } else {
                    Remove-Item $backup.FullName -Force
                    Write-Log "Deleted old backup: $($backup.Name)"
                }
            }
            Write-Log "Cleaned up $($oldBackups.Count) old backup(s)"
        } else {
            Write-Log "No old backups to clean up"
        }

        return @{ success = $true; deletedCount = $oldBackups.Count }
    } catch {
        Write-Log "Failed to clean up old backups: $_" "ERROR"
        return @{ success = $false; error = $_ }
    }
}

# Function to create backup manifest
function New-BackupManifest {
    param(
        [string]$BackupDir,
        [string]$BackupFile,
        [hashtable]$BackupInfo
    )

    $manifestFile = Join-Path $BackupDir "backup-manifest.json"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $manifest = @{
        timestamp = $timestamp
        backup_file = $BackupFile
        file_size_bytes = $BackupInfo.fileSize
        file_size_mb = $BackupInfo.fileSizeMB
        database_url = $Config.DATABASE_URL
        compression = $Config.BACKUP_COMPRESSION
        created_by = "backup-db.ps1"
    }

    # Read existing manifest or create new one
    $existingManifests = @()
    if (Test-Path $manifestFile) {
        try {
            $existingManifests = Get-Content $manifestFile | ConvertFrom-Json
            if (-not ($existingManifests -is [array])) {
                $existingManifests = @($existingManifests)
            }
        } catch {
            Write-Log "Could not read existing manifest, creating new one" "WARN"
            $existingManifests = @()
        }
    }

    # Add new backup to manifest
    $existingManifests += $manifest

    # Keep only last 10 backups in manifest
    if ($existingManifests.Count -gt 10) {
        $existingManifests = $existingManifests | Select-Object -Last 10
    }

    # Write manifest
    $existingManifests | ConvertTo-Json -Depth 3 | Out-File -FilePath $manifestFile -Encoding UTF8
    Write-Log "Backup manifest updated: $manifestFile"

    return $manifestFile
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    # Determine output path
    if ([string]::IsNullOrEmpty($OutputPath)) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupDir = $config.BACKUP_DIR
        $OutputPath = Join-Path $backupDir "mathledger-$timestamp.dump"
    }

    # Create backup directory if it doesn't exist
    $backupDir = Split-Path $OutputPath -Parent
    if (-not (Test-Path $backupDir)) {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        Write-Log "Created backup directory: $backupDir"
    }

    Write-Log "Starting MathLedger database backup"
    Write-Log "Database URL: $($config.DATABASE_URL)"
    Write-Log "Output path: $OutputPath"
    Write-Log "Compression: $($config.BACKUP_COMPRESSION)"

    if ($DryRun) {
        Write-Log "Running in DRY RUN mode - no changes will be made" "INFO"
    }

    # Create backup
    $backupResult = New-DatabaseBackup $config $OutputPath

    if (-not $backupResult.success) {
        Write-Log "Backup failed. Exiting." "ERROR"
        exit 1
    }

    # Create backup manifest
    $manifestFile = New-BackupManifest $backupDir $OutputPath $backupResult

    # Clean up old backups
    $retentionDays = [int]$config.BACKUP_RETENTION_DAYS
    $cleanupResult = Remove-OldBackups $backupDir $retentionDays

    # Report results
    Write-Log "=== Backup Summary ==="
    Write-Log "Backup file: $OutputPath"
    Write-Log "File size: $($backupResult.fileSizeMB) MB"
    Write-Log "Manifest: $manifestFile"
    if ($cleanupResult.success) {
        Write-Log "Old backups cleaned: $($cleanupResult.deletedCount)"
    }

    Write-Log "Database backup completed successfully"

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
