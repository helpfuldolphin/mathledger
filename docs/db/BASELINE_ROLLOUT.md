# Baseline Migration Rollout Guide

**Migration**: `baseline_20251019`  
**PR**: #60  
**Status**: Ready for deployment  
**Author**: Manus G - Systems Mechanic  
**Date**: 2025-10-31

---

## Executive Summary

This document provides comprehensive guidance for deploying the baseline migration (`baseline_20251019.sql`) to production environments. The baseline migration consolidates 17 legacy migrations (001-014) into a single authoritative schema, resolving 9 critical migration failures and establishing a clean foundation for future database evolution.

The rollout has been designed with **fail-closed verification**, **2-pass idempotency testing**, and **comprehensive rollback procedures** to ensure safe deployment with minimal risk.

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Scenarios](#deployment-scenarios)
3. [Step-by-Step Rollout Procedures](#step-by-step-rollout-procedures)
4. [Verification Procedures](#verification-procedures)
5. [Rollback Procedures](#rollback-procedures)
6. [Risk Assessment](#risk-assessment)
7. [Troubleshooting](#troubleshooting)
8. [Post-Deployment](#post-deployment)

---

## Pre-Deployment Checklist

Before deploying the baseline migration, ensure all prerequisites are met:

### Environment Prerequisites

- [ ] **PostgreSQL 15** or later installed
- [ ] **Database backup** created and verified (see [Backup Procedures](#backup-procedures))
- [ ] **Maintenance window** scheduled (estimated downtime: 5-15 minutes)
- [ ] **Rollback plan** reviewed and understood
- [ ] **Monitoring** in place to detect issues
- [ ] **Team notification** sent (deployment window, expected impact)

### Technical Prerequisites

- [ ] **DATABASE_URL** environment variable configured
- [ ] **PostgreSQL client tools** installed (`psql`, `pg_dump`, `pg_restore`)
- [ ] **Python 3.11+** with `uv` package manager
- [ ] **Git repository** up to date with PR #60 branch
- [ ] **CI tests passing** (2-pass idempotency verification)
- [ ] **Schema checksums** verified from CI artifacts

### Access Prerequisites

- [ ] **Database credentials** with DDL permissions (CREATE, ALTER, DROP)
- [ ] **Backup storage** accessible and has sufficient space
- [ ] **Application downtime** coordinated (if required)
- [ ] **Rollback authority** confirmed (who can authorize rollback)

---

## Deployment Scenarios

The baseline migration supports three deployment scenarios:

### Scenario A: Fresh Database (New Installation)

**When to use**: New deployments, development environments, testing

**Characteristics**:
- No existing schema
- No data to preserve
- Fastest deployment path
- Lowest risk

**Procedure**: [Fresh Database Deployment](#fresh-database-deployment)

### Scenario B: Existing Database (Upgrade from Legacy Migrations)

**When to use**: Production databases with existing schema from migrations 001-014

**Characteristics**:
- Existing tables and data
- Schema may be inconsistent due to failed migrations
- Baseline migration uses `IF NOT EXISTS` (idempotent)
- Medium risk (requires backup)

**Procedure**: [Existing Database Upgrade](#existing-database-upgrade)

### Scenario C: Staging Validation (Pre-Production Testing)

**When to use**: Before production deployment

**Characteristics**:
- Clone of production database
- Full validation of migration path
- Rollback testing
- Recommended for all production deployments

**Procedure**: [Staging Validation](#staging-validation)

---

## Step-by-Step Rollout Procedures

### Fresh Database Deployment

**Estimated time**: 2-5 minutes

#### Step 1: Verify Environment

```powershell
# Windows (PowerShell)
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/mathledger"
psql $env:DATABASE_URL -c "SELECT version();"
```

```bash
# Linux/macOS (Bash)
export DATABASE_URL="postgresql://user:pass@localhost:5432/mathledger"
psql "$DATABASE_URL" -c "SELECT version();"
```

**Expected output**: PostgreSQL version 15.x or later

#### Step 2: Run Baseline Migration

```powershell
# Windows
cd C:\dev\mathledger
uv run python scripts/run-migrations.py
```

```bash
# Linux/macOS
cd /path/to/mathledger
uv run python scripts/run-migrations.py
```

**Expected output**:
```
Running migration: migrations/baseline_20251019.sql
  Executing migration file...
  Migration migrations/baseline_20251019.sql completed successfully!
```

#### Step 3: Verify Schema

```powershell
# Check table count
psql $env:DATABASE_URL -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';"
# Expected: 13 tables

# Check migration tracking
psql $env:DATABASE_URL -c "SELECT * FROM schema_migrations;"
# Expected: version | baseline_20251019
```

#### Step 4: Validate Idempotency

```powershell
# Run migrations again (should be no-op)
uv run python scripts/run-migrations.py
```

**Expected output**: No errors, same table count

---

### Existing Database Upgrade

**Estimated time**: 10-15 minutes (including backup)

#### Step 1: Create Backup

```powershell
# Windows
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupFile = "backups/mathledger_pre_baseline_$timestamp.dump"
New-Item -ItemType Directory -Force -Path backups
pg_dump $env:DATABASE_URL --format=custom --file=$backupFile
Write-Host "Backup created: $backupFile"
```

```bash
# Linux/macOS
timestamp=$(date +%Y%m%d_%H%M%S)
backup_file="backups/mathledger_pre_baseline_${timestamp}.dump"
mkdir -p backups
pg_dump "$DATABASE_URL" --format=custom --file="$backup_file"
echo "Backup created: $backup_file"
```

#### Step 2: Verify Backup Integrity

```powershell
# Windows
$backupInfo = Get-Item $backupFile
Write-Host "Backup size: $($backupInfo.Length / 1MB) MB"
# Verify backup is restorable (dry run)
pg_restore --list $backupFile | Select-Object -First 10
```

```bash
# Linux/macOS
ls -lh "$backup_file"
# Verify backup is restorable (dry run)
pg_restore --list "$backup_file" | head -10
```

**Expected**: Backup file size > 1 KB, list of database objects displayed

#### Step 3: Capture Pre-Migration Schema Checksum

```powershell
# Windows
pg_dump $env:DATABASE_URL --schema-only | Out-File -FilePath schema_pre_baseline.sql
$preChecksum = (Get-FileHash schema_pre_baseline.sql -Algorithm SHA256).Hash
Write-Host "Pre-migration checksum: $preChecksum"
```

```bash
# Linux/macOS
pg_dump "$DATABASE_URL" --schema-only > schema_pre_baseline.sql
pre_checksum=$(sha256sum schema_pre_baseline.sql | cut -d' ' -f1)
echo "Pre-migration checksum: $pre_checksum"
```

#### Step 4: Run Baseline Migration

```powershell
# Windows
uv run python scripts/run-migrations.py
```

```bash
# Linux/macOS
uv run python scripts/run-migrations.py
```

**Expected output**:
```
=== NO BASELINE DETECTED ===
Running all migrations in sequence
...
Migration migrations/baseline_20251019.sql completed successfully!
```

**Note**: Baseline migration uses `IF NOT EXISTS` extensively, so existing tables will not be recreated.

#### Step 5: Verify Migration Success

```powershell
# Check migration tracking
psql $env:DATABASE_URL -c "SELECT * FROM schema_migrations WHERE version='baseline_20251019';"
# Expected: 1 row

# Check table count
psql $env:DATABASE_URL -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';"
# Expected: 13 tables (or more if additional tables exist)

# Check core tables
$coreTables = @('theories', 'symbols', 'statements', 'proofs', 'dependencies', 'runs', 'blocks', 'lemma_cache', 'axioms')
foreach ($table in $coreTables) {
    psql $env:DATABASE_URL -c "SELECT COUNT(*) FROM $table LIMIT 1;"
}
```

#### Step 6: Capture Post-Migration Schema Checksum

```powershell
# Windows
pg_dump $env:DATABASE_URL --schema-only | Out-File -FilePath schema_post_baseline.sql
$postChecksum = (Get-FileHash schema_post_baseline.sql -Algorithm SHA256).Hash
Write-Host "Post-migration checksum: $postChecksum"

# Compare with CI checksum (from artifacts)
# Download schema_seal.json from CI run and compare
```

```bash
# Linux/macOS
pg_dump "$DATABASE_URL" --schema-only > schema_post_baseline.sql
post_checksum=$(sha256sum schema_post_baseline.sql | cut -d' ' -f1)
echo "Post-migration checksum: $post_checksum"
```

#### Step 7: Test Idempotency

```powershell
# Run migrations again (should be no-op)
uv run python scripts/run-migrations.py

# Capture schema again
pg_dump $env:DATABASE_URL --schema-only | Out-File -FilePath schema_post_baseline_2.sql
$postChecksum2 = (Get-FileHash schema_post_baseline_2.sql -Algorithm SHA256).Hash

# Compare checksums
if ($postChecksum -eq $postChecksum2) {
    Write-Host "[PASS] Migrations: 2-pass idempotent" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Schema changed on second pass" -ForegroundColor Red
}
```

```bash
# Linux/macOS
uv run python scripts/run-migrations.py

pg_dump "$DATABASE_URL" --schema-only > schema_post_baseline_2.sql
post_checksum_2=$(sha256sum schema_post_baseline_2.sql | cut -d' ' -f1)

if [ "$post_checksum" = "$post_checksum_2" ]; then
    echo "[PASS] Migrations: 2-pass idempotent"
else
    echo "[FAIL] Schema changed on second pass"
    exit 1
fi
```

**Expected**: `[PASS] Migrations: 2-pass idempotent`

---

### Staging Validation

**Estimated time**: 30-60 minutes

Staging validation provides the highest confidence before production deployment.

#### Step 1: Clone Production Database

```powershell
# Windows
$prodBackup = "backups/production_clone_$(Get-Date -Format 'yyyyMMdd_HHmmss').dump"
pg_dump $env:PROD_DATABASE_URL --format=custom --file=$prodBackup

# Restore to staging
$env:DATABASE_URL = $env:STAGING_DATABASE_URL
pg_restore -d $env:DATABASE_URL --clean --if-exists $prodBackup
```

```bash
# Linux/macOS
prod_backup="backups/production_clone_$(date +%Y%m%d_%H%M%S).dump"
pg_dump "$PROD_DATABASE_URL" --format=custom --file="$prod_backup"

# Restore to staging
export DATABASE_URL="$STAGING_DATABASE_URL"
pg_restore -d "$DATABASE_URL" --clean --if-exists "$prod_backup"
```

#### Step 2: Execute Full Rollout Procedure

Follow [Existing Database Upgrade](#existing-database-upgrade) steps 1-7 on staging database.

#### Step 3: Validate Application Functionality

```powershell
# Start application against staging database
# Run smoke tests
# Verify core functionality
```

#### Step 4: Test Rollback Procedure

```powershell
# Windows
.\scripts\db\rollback_baseline.ps1 -VerifyOnly
# Expected: Verification passes

# Execute rollback
.\scripts\db\rollback_baseline.ps1
# Confirm with "ROLLBACK"
```

```bash
# Linux/macOS
./scripts/db/rollback_baseline.sh --verify-only
# Expected: Verification passes

# Execute rollback
./scripts/db/rollback_baseline.sh
# Confirm with "ROLLBACK"
```

#### Step 5: Re-apply Migration

After successful rollback test, re-apply the migration to confirm repeatability:

```powershell
uv run python scripts/run-migrations.py
```

#### Step 6: Document Results

Create staging validation report:
- Pre-migration schema checksum
- Post-migration schema checksum
- Idempotency test result
- Rollback test result
- Application functionality test results
- Any issues encountered and resolutions

---

## Verification Procedures

### Schema Verification

**Verify all core tables exist**:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

**Expected tables**:
- axioms
- blocks
- dependencies
- derived_statements
- lemma_cache
- policy_settings
- proof_parents
- proofs
- runs
- schema_migrations
- statements
- symbols
- theories

### Index Verification

**Verify indexes created**:

```sql
SELECT schemaname, tablename, indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
ORDER BY tablename, indexname;
```

**Expected**: At least 35 indexes across all tables

### Constraint Verification

**Verify foreign key constraints**:

```sql
SELECT 
    tc.table_name, 
    tc.constraint_name, 
    tc.constraint_type,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
LEFT JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name, tc.constraint_name;
```

**Expected**: At least 18 foreign key relationships

### Data Verification

**Verify seed data**:

```sql
SELECT * FROM theories;
```

**Expected**: At least 2 theories (Propositional, First Order Logic)

### Migration Tracking Verification

**Verify baseline recorded**:

```sql
SELECT * FROM schema_migrations WHERE version = 'baseline_20251019';
```

**Expected**: 1 row with description "Consolidated baseline schema from migrations 001-014"

---

## Rollback Procedures

### When to Rollback

Initiate rollback if:
- Migration fails with errors
- Post-migration verification fails
- Application functionality is broken
- Schema checksum doesn't match expected value
- Idempotency test fails

### Rollback Prerequisites

Before executing rollback:
- [ ] Valid backup exists and is verified
- [ ] Rollback authority obtained
- [ ] Application stopped (if running)
- [ ] Team notified of rollback decision

### Rollback Execution

#### Windows (PowerShell)

```powershell
# Verify rollback prerequisites
.\scripts\db\rollback_baseline.ps1 -VerifyOnly

# Execute rollback
.\scripts\db\rollback_baseline.ps1
# Type "ROLLBACK" when prompted
```

#### Linux/macOS (Bash)

```bash
# Verify rollback prerequisites
./scripts/db/rollback_baseline.sh --verify-only

# Execute rollback
./scripts/db/rollback_baseline.sh
# Type "ROLLBACK" when prompted
```

### Post-Rollback Verification

After rollback:

1. **Verify database connection**:
   ```powershell
   psql $env:DATABASE_URL -c "SELECT version();"
   ```

2. **Verify baseline not recorded**:
   ```sql
   SELECT * FROM schema_migrations WHERE version = 'baseline_20251019';
   ```
   Expected: 0 rows

3. **Verify table count**:
   ```sql
   SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';
   ```
   Expected: Same as pre-migration count

4. **Test application functionality**

5. **Review rollback logs** for any warnings or errors

### Rollback Risks

**Data Loss**: If backup is stale or incomplete, data created after backup will be lost.

**Mitigation**: Always create fresh backup immediately before migration.

**Downtime**: Rollback requires dropping all tables and restoring from backup.

**Mitigation**: Schedule rollback during maintenance window.

**Incomplete Restoration**: Backup may not include all database objects.

**Mitigation**: Verify backup integrity before rollback, test rollback on staging first.

---

## Risk Assessment

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration fails on fresh database | Low | Medium | 2-pass CI testing, pre-deployment validation |
| Migration fails on existing database | Low | High | Backup before migration, staging validation |
| Schema inconsistency after migration | Very Low | High | Checksum verification, idempotency testing |
| Data loss during migration | Very Low | Critical | Baseline uses IF NOT EXISTS (no DROP), backup required |
| Rollback fails | Very Low | Critical | Rollback script tested on staging, pre-rollback snapshot |
| Application incompatibility | Low | High | Staging validation with application tests |
| Performance degradation | Very Low | Medium | Indexes created by baseline, monitoring in place |

### Risk Mitigation Summary

**Pre-Deployment**:
- Comprehensive CI testing (2-pass idempotency)
- Staging validation with production clone
- Backup verification
- Rollback procedure testing

**During Deployment**:
- Maintenance window (minimize user impact)
- Real-time monitoring
- Incremental verification steps
- Rollback authority on standby

**Post-Deployment**:
- Schema checksum verification
- Application functionality testing
- Performance monitoring
- Incident response plan ready

---

## Troubleshooting

### Issue: Migration Fails with "relation already exists"

**Symptom**: Error message like `ERROR: relation "theories" already exists`

**Cause**: Baseline migration expects `IF NOT EXISTS` to handle existing tables, but error still occurs

**Resolution**:
1. Check if table was created outside migration system
2. Manually add `IF NOT EXISTS` to failing statement (if not present)
3. Report issue to Manus G for baseline migration update

### Issue: Idempotency Test Fails (Schema Changes on Second Pass)

**Symptom**: Schema checksums differ between first and second migration run

**Cause**: Non-idempotent operation in migration (e.g., timestamp defaults, sequence resets)

**Resolution**:
1. Capture schema diff: `diff -u schema_pass1.sql schema_pass2.sql`
2. Identify non-idempotent operation
3. Do NOT proceed to production
4. Report issue to Manus G for baseline migration fix

### Issue: Backup Restoration Fails During Rollback

**Symptom**: `pg_restore` or `psql` errors during rollback

**Cause**: Corrupted backup, incompatible backup format, insufficient permissions

**Resolution**:
1. Verify backup file integrity: `pg_restore --list backup_file.dump`
2. Check PostgreSQL version compatibility
3. Verify database user has CREATE permissions
4. Try alternative backup if available
5. Escalate to DBA if restoration continues to fail

### Issue: Application Errors After Migration

**Symptom**: Application fails to start or queries fail after migration

**Cause**: Schema incompatibility, missing columns, type mismatches

**Resolution**:
1. Check application logs for specific error
2. Verify schema matches expected structure
3. Check if application code is compatible with baseline schema
4. If critical: initiate rollback
5. If non-critical: document issue and create hotfix

### Issue: Performance Degradation After Migration

**Symptom**: Slow queries, high CPU usage, increased latency

**Cause**: Missing indexes, query plan changes, statistics not updated

**Resolution**:
1. Run `ANALYZE` on all tables: `ANALYZE;`
2. Verify indexes exist: Check [Index Verification](#index-verification)
3. Review slow query log
4. If severe: consider rollback
5. If moderate: optimize queries or add indexes

---

## Post-Deployment

### Immediate Post-Deployment (T+0 to T+1 hour)

- [ ] Verify application is running normally
- [ ] Monitor error logs for anomalies
- [ ] Check database performance metrics
- [ ] Verify user-facing functionality
- [ ] Document any issues encountered
- [ ] Notify team of successful deployment

### Short-Term Post-Deployment (T+1 hour to T+24 hours)

- [ ] Monitor database performance trends
- [ ] Review application logs for migration-related errors
- [ ] Verify batch jobs and scheduled tasks run successfully
- [ ] Check data integrity (row counts, foreign key violations)
- [ ] Gather feedback from users
- [ ] Update runbook with lessons learned

### Long-Term Post-Deployment (T+24 hours to T+1 week)

- [ ] Analyze performance metrics vs. baseline
- [ ] Review incident reports related to migration
- [ ] Plan for removal of legacy migrations (001-014)
- [ ] Document schema evolution strategy
- [ ] Update migration procedures based on experience
- [ ] Archive backup files (retain for 30 days minimum)

### Cleanup (After 1 Week)

Once baseline migration is proven stable in production:

1. **Remove legacy migrations** (001-014):
   ```bash
   git rm migrations/001_init.sql migrations/002_*.sql migrations/003_*.sql ...
   ```

2. **Update migration runner** to always use baseline path

3. **Archive pre-baseline backups** to long-term storage

4. **Update documentation** to reflect baseline as canonical schema

---

## Pass-Lines

Successful deployment is confirmed when all pass-lines are achieved:

### Pre-Deployment Pass-Lines

- `[PASS] Prerequisites: All checks satisfied`
- `[PASS] Backup: Created and verified`
- `[PASS] Staging: Validation complete`

### Deployment Pass-Lines

- `[PASS] Migrations: 2-pass idempotent`
- `[PASS] Rollback Playbook: checksum sealed`
- `[PASS] Schema: Verified against CI baseline`
- `[PASS] Application: Functionality confirmed`

### Post-Deployment Pass-Lines

- `[PASS] Monitoring: No anomalies detected`
- `[PASS] Performance: Within acceptable thresholds`
- `[PASS] Rollback: Tested and verified`

---

## Appendix

### A. Backup Procedures

**Create Full Backup (Custom Format)**:
```powershell
# Windows
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
pg_dump $env:DATABASE_URL --format=custom --file="backups/mathledger_$timestamp.dump"
```

```bash
# Linux/macOS
timestamp=$(date +%Y%m%d_%H%M%S)
pg_dump "$DATABASE_URL" --format=custom --file="backups/mathledger_${timestamp}.dump"
```

**Create Schema-Only Backup**:
```powershell
pg_dump $env:DATABASE_URL --schema-only --file="backups/schema_$timestamp.sql"
```

**Verify Backup**:
```powershell
pg_restore --list backups/mathledger_$timestamp.dump | Select-Object -First 20
```

### B. Schema Checksum Verification

**Generate Schema Checksum**:
```powershell
# Windows
pg_dump $env:DATABASE_URL --schema-only --no-owner --no-privileges | Out-File -FilePath schema.sql
$checksum = (Get-FileHash schema.sql -Algorithm SHA256).Hash
Write-Host "Schema checksum: $checksum"
```

```bash
# Linux/macOS
pg_dump "$DATABASE_URL" --schema-only --no-owner --no-privileges > schema.sql
checksum=$(sha256sum schema.sql | cut -d' ' -f1)
echo "Schema checksum: $checksum"
```

**Compare with CI Checksum**:
1. Download `schema_seal.json` from CI artifacts
2. Extract `pass1_checksum` or `pass2_checksum`
3. Compare with local checksum
4. If match: `[PASS] Rollback Playbook: checksum sealed`

### C. Emergency Contacts

**Database Issues**:
- DBA: [Contact Information]
- On-Call Engineer: [Contact Information]

**Application Issues**:
- Backend Team: [Contact Information]
- DevOps: [Contact Information]

**Escalation**:
- Engineering Manager: [Contact Information]
- CTO: [Contact Information]

### D. References

- **PR #60**: https://github.com/helpfuldolphin/mathledger/pull/60
- **Migration Diagnostic**: `scripts/db/repair_migrations.md`
- **Baseline Migration**: `migrations/baseline_20251019.sql`
- **Rollback Script (Bash)**: `scripts/db/rollback_baseline.sh`
- **Rollback Script (PowerShell)**: `scripts/db/rollback_baseline.ps1`
- **CI Workflow**: `.github/workflows/db-migration-check.yml`

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-31  
**Author**: Manus G - Systems Mechanic  
**Status**: Ready for Production Deployment

