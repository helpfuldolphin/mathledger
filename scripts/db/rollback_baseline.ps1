# scripts/db/rollback_baseline.ps1
# 
# Rollback script for baseline_20251019 migration (COLD PATH) - PowerShell version
# 
# WARNING: This script performs destructive operations.
# Only use when baseline migration has failed or needs to be reverted.
# 
# Prerequisites:
# - Database backup exists (created before migration)
# - PostgreSQL client tools installed (psql, pg_dump)
# - $env:DATABASE_URL environment variable set
#
# Usage:
#   .\scripts\db\rollback_baseline.ps1 [-VerifyOnly]
#
# Parameters:
#   -VerifyOnly    Only verify rollback prerequisites, don't execute
#
# Author: Manus G - Systems Mechanic
# Date: 2025-10-31

param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

# Configuration
$BASELINE_VERSION = "baseline_20251019"
$BACKUP_DIR = if ($env:BACKUP_DIR) { $env:BACKUP_DIR } else { ".\backups" }

# Functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Prerequisites {
    Write-Info "Checking rollback prerequisites..."
    
    # Check DATABASE_URL
    if (-not $env:DATABASE_URL) {
        Write-Error-Custom "DATABASE_URL environment variable not set"
        return $false
    }
    
    # Check psql
    if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "psql command not found. Install PostgreSQL client tools."
        return $false
    }
    
    # Check pg_dump
    if (-not (Get-Command pg_dump -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "pg_dump command not found. Install PostgreSQL client tools."
        return $false
    }
    
    # Check database connection
    $testQuery = "SELECT 1"
    $result = psql $env:DATABASE_URL -c $testQuery 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Cannot connect to database at $env:DATABASE_URL"
        return $false
    }
    
    Write-Info "✓ All prerequisites met"
    return $true
}

function Test-BaselineApplied {
    Write-Info "Checking if baseline migration is applied..."
    
    # Check if schema_migrations table exists
    $checkTable = "SELECT 1 FROM schema_migrations LIMIT 1"
    $result = psql $env:DATABASE_URL -c $checkTable 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "schema_migrations table does not exist"
        Write-Warn "Baseline migration may not have been applied"
        return $false
    }
    
    # Check if baseline_20251019 is recorded
    $checkBaseline = "SELECT COUNT(*) FROM schema_migrations WHERE version='$BASELINE_VERSION'"
    $count = psql $env:DATABASE_URL -t -c $checkBaseline 2>&1
    $count = $count.Trim()
    
    if ($count -eq "0") {
        Write-Warn "Baseline migration $BASELINE_VERSION not found in schema_migrations"
        return $false
    }
    
    Write-Info "✓ Baseline migration $BASELINE_VERSION is applied"
    return $true
}

function Find-LatestBackup {
    Write-Info "Searching for latest database backup..."
    
    if (-not (Test-Path $BACKUP_DIR)) {
        Write-Error-Custom "Backup directory $BACKUP_DIR does not exist"
        return $null
    }
    
    # Find most recent .dump or .sql file
    $backupFiles = Get-ChildItem -Path $BACKUP_DIR -Include *.dump,*.sql -File | Sort-Object LastWriteTime -Descending
    
    if ($backupFiles.Count -eq 0) {
        Write-Error-Custom "No backup files found in $BACKUP_DIR"
        return $null
    }
    
    $latestBackup = $backupFiles[0].FullName
    Write-Info "Found backup: $latestBackup"
    return $latestBackup
}

function Test-BackupIntegrity {
    param([string]$BackupFile)
    
    Write-Info "Verifying backup integrity..."
    
    # Check file exists and is readable
    if (-not (Test-Path $BackupFile -PathType Leaf)) {
        Write-Error-Custom "Backup file $BackupFile does not exist or is not readable"
        return $false
    }
    
    # Check file size
    $fileInfo = Get-Item $BackupFile
    if ($fileInfo.Length -lt 1024) {
        Write-Error-Custom "Backup file is suspiciously small ($($fileInfo.Length) bytes)"
        return $false
    }
    
    Write-Info "✓ Backup integrity check passed"
    return $true
}

function New-PreRollbackSnapshot {
    Write-Info "Creating pre-rollback snapshot..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $snapshotFile = Join-Path $BACKUP_DIR "pre_rollback_$timestamp.sql"
    
    pg_dump $env:DATABASE_URL --schema-only | Out-File -FilePath $snapshotFile -Encoding UTF8
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "✓ Pre-rollback snapshot saved to $snapshotFile"
        return $snapshotFile
    } else {
        Write-Error-Custom "Failed to create pre-rollback snapshot"
        return $null
    }
}

function Invoke-Rollback {
    param([string]$BackupFile)
    
    Write-Warn "=========================================="
    Write-Warn "  EXECUTING ROLLBACK (DESTRUCTIVE)"
    Write-Warn "=========================================="
    Write-Warn ""
    Write-Warn "This will:"
    Write-Warn "  1. Drop all tables in the database"
    Write-Warn "  2. Restore schema from backup: $BackupFile"
    Write-Warn ""
    
    $confirmation = Read-Host "Are you sure you want to proceed? (type 'ROLLBACK' to confirm)"
    
    if ($confirmation -ne "ROLLBACK") {
        Write-Info "Rollback cancelled by user"
        return $false
    }
    
    Write-Info "Step 1: Dropping all tables..."
    
    # Drop all tables in public schema
    $dropTablesSQL = @"
DO `$`$ 
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END `$`$;
"@
    
    $dropTablesSQL | psql $env:DATABASE_URL
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to drop tables"
        return $false
    }
    
    Write-Info "✓ All tables dropped"
    
    Write-Info "Step 2: Restoring from backup..."
    
    # Restore from backup
    if ($BackupFile -like "*.dump") {
        # Custom format dump
        pg_restore -d $env:DATABASE_URL $BackupFile
    } else {
        # Plain SQL dump
        Get-Content $BackupFile | psql $env:DATABASE_URL
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to restore from backup"
        Write-Error-Custom "Database may be in inconsistent state!"
        return $false
    }
    
    Write-Info "✓ Backup restored successfully"
    
    Write-Info "Step 3: Verifying restoration..."
    
    # Check if tables exist
    $tableCountQuery = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
    $tableCount = (psql $env:DATABASE_URL -t -c $tableCountQuery).Trim()
    
    if ($tableCount -eq "0") {
        Write-Error-Custom "No tables found after restoration!"
        return $false
    }
    
    Write-Info "✓ Found $tableCount tables after restoration"
    
    Write-Info "Step 4: Removing baseline migration record..."
    
    # Remove baseline migration record if schema_migrations exists
    $checkTable = "SELECT 1 FROM schema_migrations LIMIT 1"
    $result = psql $env:DATABASE_URL -c $checkTable 2>&1
    if ($LASTEXITCODE -eq 0) {
        $deleteBaseline = "DELETE FROM schema_migrations WHERE version='$BASELINE_VERSION'"
        psql $env:DATABASE_URL -c $deleteBaseline
        Write-Info "✓ Baseline migration record removed"
    } else {
        Write-Info "schema_migrations table not found (expected for pre-baseline backups)"
    }
    
    Write-Info "=========================================="
    Write-Info "  ROLLBACK COMPLETE"
    Write-Info "=========================================="
    
    return $true
}

function Test-RollbackSuccess {
    Write-Info "Verifying rollback success..."
    
    # Check database connection
    $testQuery = "SELECT 1"
    $result = psql $env:DATABASE_URL -c $testQuery 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Cannot connect to database after rollback"
        return $false
    }
    
    # Check if baseline migration is no longer recorded
    if (Test-BaselineApplied) {
        Write-Error-Custom "Baseline migration still appears to be applied"
        return $false
    }
    
    # Check if core tables exist (depends on backup)
    $tableCountQuery = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
    $tableCount = (psql $env:DATABASE_URL -t -c $tableCountQuery).Trim()
    
    if ($tableCount -eq "0") {
        Write-Warn "No tables found (empty database)"
    } else {
        Write-Info "✓ Found $tableCount tables"
    }
    
    Write-Info "✓ Rollback verification passed"
    return $true
}

# Main execution
function Main {
    Write-Host "========================================"
    Write-Host "  Baseline Migration Rollback Script"
    Write-Host "  Version: $BASELINE_VERSION"
    Write-Host "========================================"
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        Write-Error-Custom "Prerequisites check failed"
        exit 1
    }
    
    # Check if baseline is applied
    if (-not (Test-BaselineApplied)) {
        Write-Warn "Baseline migration does not appear to be applied"
        $continueAnyway = Read-Host "Continue anyway? (y/N)"
        if ($continueAnyway -notmatch "^[Yy]$") {
            Write-Info "Rollback cancelled"
            exit 0
        }
    }
    
    # Find latest backup
    $backupFile = Find-LatestBackup
    if (-not $backupFile) {
        Write-Error-Custom "Cannot proceed without a backup file"
        exit 1
    }
    
    # Verify backup integrity
    if (-not (Test-BackupIntegrity -BackupFile $backupFile)) {
        Write-Error-Custom "Backup integrity check failed"
        exit 1
    }
    
    # If verify-only mode, stop here
    if ($VerifyOnly) {
        Write-Info "=========================================="
        Write-Info "  VERIFICATION COMPLETE (-VerifyOnly)"
        Write-Info "=========================================="
        Write-Info "Rollback prerequisites are satisfied"
        Write-Info "Backup file: $backupFile"
        Write-Info ""
        Write-Info "To execute rollback, run:"
        Write-Info "  .\scripts\db\rollback_baseline.ps1"
        exit 0
    }
    
    # Create pre-rollback snapshot
    $snapshotFile = New-PreRollbackSnapshot
    if (-not $snapshotFile) {
        Write-Error-Custom "Failed to create pre-rollback snapshot"
        Write-Error-Custom "Aborting rollback for safety"
        exit 1
    }
    
    # Execute rollback
    if (-not (Invoke-Rollback -BackupFile $backupFile)) {
        Write-Error-Custom "Rollback failed"
        Write-Error-Custom "Pre-rollback snapshot saved at: $snapshotFile"
        exit 1
    }
    
    # Verify rollback success
    if (-not (Test-RollbackSuccess)) {
        Write-Error-Custom "Rollback verification failed"
        exit 1
    }
    
    Write-Info ""
    Write-Info "=========================================="
    Write-Info "  ROLLBACK SUCCESSFUL"
    Write-Info "=========================================="
    Write-Info "Database restored from: $backupFile"
    Write-Info "Pre-rollback snapshot: $snapshotFile"
    Write-Info ""
    Write-Info "Next steps:"
    Write-Info "  1. Verify application functionality"
    Write-Info "  2. Review rollback logs"
    Write-Info "  3. Investigate root cause of migration failure"
    Write-Info ""
}

# Run main function
Main

