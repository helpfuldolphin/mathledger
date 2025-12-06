#!/bin/bash
# scripts/db/rollback_baseline.sh
# 
# Rollback script for baseline_20251019 migration (COLD PATH)
# 
# WARNING: This script performs destructive operations.
# Only use when baseline migration has failed or needs to be reverted.
# 
# Prerequisites:
# - Database backup exists (created before migration)
# - PostgreSQL client tools installed
# - DATABASE_URL environment variable set
#
# Usage:
#   ./scripts/db/rollback_baseline.sh [--verify-only]
#
# Options:
#   --verify-only    Only verify rollback prerequisites, don't execute
#
# Author: Manus G - Systems Mechanic
# Date: 2025-10-31

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASELINE_VERSION="baseline_20251019"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
VERIFY_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--verify-only]"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking rollback prerequisites..."
    
    # Check DATABASE_URL
    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL environment variable not set"
        return 1
    fi
    
    # Check psql
    if ! command -v psql &> /dev/null; then
        log_error "psql command not found. Install PostgreSQL client tools."
        return 1
    fi
    
    # Check pg_dump
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump command not found. Install PostgreSQL client tools."
        return 1
    fi
    
    # Check database connection
    if ! psql "$DATABASE_URL" -c "SELECT 1" &> /dev/null; then
        log_error "Cannot connect to database at $DATABASE_URL"
        return 1
    fi
    
    log_info "✓ All prerequisites met"
    return 0
}

check_baseline_applied() {
    log_info "Checking if baseline migration is applied..."
    
    # Check if schema_migrations table exists
    if ! psql "$DATABASE_URL" -c "SELECT 1 FROM schema_migrations LIMIT 1" &> /dev/null; then
        log_warn "schema_migrations table does not exist"
        log_warn "Baseline migration may not have been applied"
        return 1
    fi
    
    # Check if baseline_20251019 is recorded
    local count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM schema_migrations WHERE version='$BASELINE_VERSION'")
    count=$(echo "$count" | tr -d ' ')
    
    if [ "$count" -eq 0 ]; then
        log_warn "Baseline migration $BASELINE_VERSION not found in schema_migrations"
        return 1
    fi
    
    log_info "✓ Baseline migration $BASELINE_VERSION is applied"
    return 0
}

find_latest_backup() {
    log_info "Searching for latest database backup..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory $BACKUP_DIR does not exist"
        return 1
    fi
    
    # Find most recent .dump or .sql file
    local latest_backup=$(find "$BACKUP_DIR" -type f \( -name "*.dump" -o -name "*.sql" \) -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_backup" ]; then
        log_error "No backup files found in $BACKUP_DIR"
        return 1
    fi
    
    log_info "Found backup: $latest_backup"
    echo "$latest_backup"
    return 0
}

verify_backup_integrity() {
    local backup_file="$1"
    
    log_info "Verifying backup integrity..."
    
    # Check file exists and is readable
    if [ ! -r "$backup_file" ]; then
        log_error "Backup file $backup_file is not readable"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null)
    if [ "$file_size" -lt 1024 ]; then
        log_error "Backup file is suspiciously small ($file_size bytes)"
        return 1
    fi
    
    # Check if it's a valid SQL dump (basic check)
    if file "$backup_file" | grep -q "text"; then
        log_info "✓ Backup file appears to be valid SQL"
    else
        log_warn "Backup file may not be a valid SQL dump"
    fi
    
    log_info "✓ Backup integrity check passed"
    return 0
}

create_pre_rollback_snapshot() {
    log_info "Creating pre-rollback snapshot..."
    
    local snapshot_file="$BACKUP_DIR/pre_rollback_$(date +%Y%m%d_%H%M%S).sql"
    
    if pg_dump "$DATABASE_URL" --schema-only > "$snapshot_file"; then
        log_info "✓ Pre-rollback snapshot saved to $snapshot_file"
        echo "$snapshot_file"
        return 0
    else
        log_error "Failed to create pre-rollback snapshot"
        return 1
    fi
}

execute_rollback() {
    local backup_file="$1"
    
    log_warn "=========================================="
    log_warn "  EXECUTING ROLLBACK (DESTRUCTIVE)"
    log_warn "=========================================="
    log_warn ""
    log_warn "This will:"
    log_warn "  1. Drop all tables in the database"
    log_warn "  2. Restore schema from backup: $backup_file"
    log_warn ""
    
    read -p "Are you sure you want to proceed? (type 'ROLLBACK' to confirm): " confirmation
    
    if [ "$confirmation" != "ROLLBACK" ]; then
        log_info "Rollback cancelled by user"
        return 1
    fi
    
    log_info "Step 1: Dropping all tables..."
    
    # Drop all tables in public schema
    psql "$DATABASE_URL" << 'EOF'
DO $$ 
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
EOF
    
    if [ $? -ne 0 ]; then
        log_error "Failed to drop tables"
        return 1
    fi
    
    log_info "✓ All tables dropped"
    
    log_info "Step 2: Restoring from backup..."
    
    # Restore from backup
    if [[ "$backup_file" == *.dump ]]; then
        # Custom format dump
        pg_restore -d "$DATABASE_URL" "$backup_file"
    else
        # Plain SQL dump
        psql "$DATABASE_URL" < "$backup_file"
    fi
    
    if [ $? -ne 0 ]; then
        log_error "Failed to restore from backup"
        log_error "Database may be in inconsistent state!"
        return 1
    fi
    
    log_info "✓ Backup restored successfully"
    
    log_info "Step 3: Verifying restoration..."
    
    # Check if tables exist
    local table_count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
    table_count=$(echo "$table_count" | tr -d ' ')
    
    if [ "$table_count" -eq 0 ]; then
        log_error "No tables found after restoration!"
        return 1
    fi
    
    log_info "✓ Found $table_count tables after restoration"
    
    log_info "Step 4: Removing baseline migration record..."
    
    # Remove baseline migration record if schema_migrations exists
    if psql "$DATABASE_URL" -c "SELECT 1 FROM schema_migrations LIMIT 1" &> /dev/null; then
        psql "$DATABASE_URL" -c "DELETE FROM schema_migrations WHERE version='$BASELINE_VERSION'"
        log_info "✓ Baseline migration record removed"
    else
        log_info "schema_migrations table not found (expected for pre-baseline backups)"
    fi
    
    log_info "=========================================="
    log_info "  ROLLBACK COMPLETE"
    log_info "=========================================="
    
    return 0
}

verify_rollback_success() {
    log_info "Verifying rollback success..."
    
    # Check database connection
    if ! psql "$DATABASE_URL" -c "SELECT 1" &> /dev/null; then
        log_error "Cannot connect to database after rollback"
        return 1
    fi
    
    # Check if baseline migration is no longer recorded
    if check_baseline_applied; then
        log_error "Baseline migration still appears to be applied"
        return 1
    fi
    
    # Check if core tables exist (depends on backup)
    local table_count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
    table_count=$(echo "$table_count" | tr -d ' ')
    
    if [ "$table_count" -eq 0 ]; then
        log_warn "No tables found (empty database)"
    else
        log_info "✓ Found $table_count tables"
    fi
    
    log_info "✓ Rollback verification passed"
    return 0
}

# Main execution
main() {
    echo "========================================"
    echo "  Baseline Migration Rollback Script"
    echo "  Version: $BASELINE_VERSION"
    echo "========================================"
    echo ""
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # Check if baseline is applied
    if ! check_baseline_applied; then
        log_warn "Baseline migration does not appear to be applied"
        read -p "Continue anyway? (y/N): " continue_anyway
        if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Find latest backup
    local backup_file
    backup_file=$(find_latest_backup)
    if [ $? -ne 0 ]; then
        log_error "Cannot proceed without a backup file"
        exit 1
    fi
    
    # Verify backup integrity
    if ! verify_backup_integrity "$backup_file"; then
        log_error "Backup integrity check failed"
        exit 1
    fi
    
    # If verify-only mode, stop here
    if [ "$VERIFY_ONLY" = true ]; then
        log_info "=========================================="
        log_info "  VERIFICATION COMPLETE (--verify-only)"
        log_info "=========================================="
        log_info "Rollback prerequisites are satisfied"
        log_info "Backup file: $backup_file"
        log_info ""
        log_info "To execute rollback, run:"
        log_info "  $0"
        exit 0
    fi
    
    # Create pre-rollback snapshot
    local snapshot_file
    snapshot_file=$(create_pre_rollback_snapshot)
    if [ $? -ne 0 ]; then
        log_error "Failed to create pre-rollback snapshot"
        log_error "Aborting rollback for safety"
        exit 1
    fi
    
    # Execute rollback
    if ! execute_rollback "$backup_file"; then
        log_error "Rollback failed"
        log_error "Pre-rollback snapshot saved at: $snapshot_file"
        exit 1
    fi
    
    # Verify rollback success
    if ! verify_rollback_success; then
        log_error "Rollback verification failed"
        exit 1
    fi
    
    log_info ""
    log_info "=========================================="
    log_info "  ROLLBACK SUCCESSFUL"
    log_info "=========================================="
    log_info "Database restored from: $backup_file"
    log_info "Pre-rollback snapshot: $snapshot_file"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Verify application functionality"
    log_info "  2. Review rollback logs"
    log_info "  3. Investigate root cause of migration failure"
    log_info ""
}

# Run main function
main "$@"

