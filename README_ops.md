# MathLedger Operations Guide

This document provides operational guidance for running MathLedger in production, including nightly operations, monitoring, backup/restore procedures, and troubleshooting.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Nightly Operations](#nightly-operations)
- [Monitoring & Alerting](#monitoring--alerting)
- [Backup & Restore](#backup--restore)
- [Sanity Checks](#sanity-checks)
- [Troubleshooting](#troubleshooting)
- [Task Scheduler Setup](#task-scheduler-setup)

## Quick Start

### Prerequisites

- Windows 10/11 with PowerShell 5.1+
- Docker Desktop with PostgreSQL and Redis containers running
- Python 3.11+ with `uv` package manager
- MathLedger backend services running (FastAPI orchestrator)

### Basic Health Check

```powershell
# Run a comprehensive sanity check
powershell -File .\scripts\sanity.ps1

# Check system health
powershell -File .\scripts\healthcheck.ps1
```

### Manual Nightly Run

```powershell
# Run complete nightly operations
powershell -File .\scripts\run-nightly.ps1

# Or run individual components
powershell -File .\scripts\derive-nightly.ps1
powershell -File .\scripts\export-snapshot.ps1
powershell -File .\scripts\db-maintenance.ps1
```

## Configuration

### Environment Configuration

All operational settings are configured in `config/nightly.env`:

```bash
# Database Configuration
DATABASE_URL=postgresql://ml:mlpass@localhost:5432/mathledger
REDIS_URL=redis://localhost:6379/0
QUEUE_KEY=ml:jobs

# Derivation Limits
DERIVE_STEPS=300
DERIVE_DEPTH_MAX=4
DERIVE_MAX_BREADTH=500
DERIVE_MAX_TOTAL=2000

# Monitoring Configuration
METRICS_URL=http://localhost:8000/metrics
HEALTH_URL=http://localhost:8000/health
SUCCESS_RATE_THRESHOLD=85
TIMEOUT_THRESHOLD_MS=5000
POLL_INTERVAL_MINUTES=5

# Alerting Configuration
ALERT_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_ENABLED=true

# File Paths
PROGRESS_PATH=docs/progress.md
EXPORTS_DIR=exports
LOGS_DIR=logs
BACKUP_DIR=backups

# Database Maintenance
VACUUM_ANALYZE=true
PRUNE_FAILED_PROOFS_DAYS=14

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
```

### Key Configuration Parameters

- **DERIVE_STEPS**: Number of derivation steps per nightly run
- **DERIVE_DEPTH_MAX**: Maximum derivation depth to prevent infinite loops
- **DERIVE_MAX_BREADTH**: Maximum new statements per step
- **DERIVE_MAX_TOTAL**: Maximum total new statements per run
- **SUCCESS_RATE_THRESHOLD**: Alert threshold for proof success rate (%)
- **ALERT_WEBHOOK**: Discord/Slack webhook URL for notifications
- **BACKUP_RETENTION_DAYS**: How long to keep database backups

## Nightly Operations

### Automated Nightly Run

The nightly operations script (`scripts/run-nightly.ps1`) orchestrates all automated tasks:

1. **Health Check**: Verifies all services are running
2. **Derivation**: Runs the axiom engine to derive new statements
3. **Snapshot Export**: Exports current state to JSONL files
4. **Database Maintenance**: Runs VACUUM ANALYZE and cleanup
5. **Progress Update**: Appends run summary to progress.md

```powershell
# Full nightly run
powershell -File .\scripts\run-nightly.ps1

# Skip specific steps
powershell -File .\scripts\run-nightly.ps1 -SkipHealthCheck
powershell -File .\scripts\run-nightly.ps1 -SkipDerivation
powershell -File .\scripts\run-nightly.ps1 -SkipSnapshot
powershell -File .\scripts\run-nightly.ps1 -SkipMaintenance

# Dry run (no changes)
powershell -File .\scripts\run-nightly.ps1 -DryRun
```

### Individual Components

#### Derivation Engine

```powershell
# Run derivation with custom parameters
powershell -File .\scripts\derive-nightly.ps1 -ConfigPath config/nightly.env

# Dry run
powershell -File .\scripts\derive-nightly.ps1 -DryRun
```

#### Snapshot Export

```powershell
# Export all data
powershell -File .\scripts\export-snapshot.ps1

# Export with dependencies
powershell -File .\scripts\export-snapshot.ps1 -IncludeDependencies

# Export to custom directory
powershell -File .\scripts\export-snapshot.ps1 -OutputDir custom/exports
```

#### Database Maintenance

```powershell
# Run maintenance tasks
powershell -File .\scripts\db-maintenance.ps1

# Dry run
powershell -File .\scripts\db-maintenance.ps1 -DryRun
```

## Monitoring & Alerting

### Metrics Monitoring

The metrics watchdog script monitors system health and sends alerts:

```powershell
# Single check
powershell -File .\scripts\metrics-watch.ps1

# Continuous monitoring (5 minutes)
powershell -File .\scripts\metrics-watch.ps1 -Continuous

# Custom duration
powershell -File .\scripts\metrics-watch.ps1 -Continuous -DurationMinutes 60
```

### Alert Conditions

Alerts are sent when:
- Success rate drops below 85%
- Error rate exceeds 10%
- Queue length exceeds 1000
- No recent proof activity (last hour)

### Manual Metrics Check

```powershell
# Check current metrics
curl http://localhost:8000/metrics

# Check health endpoint
curl http://localhost:8000/health
```

## Backup & Restore

### Database Backup

```powershell
# Create backup with default settings
powershell -File .\scripts\backup-db.ps1

# Create backup to specific location
powershell -File .\scripts\backup-db.ps1 -OutputPath backups/custom-backup.dump

# Dry run
powershell -File .\scripts\backup-db.ps1 -DryRun
```

Backup files are stored in `backups/` directory with format:
- `mathledger-YYYYMMDD-HHMMSS.dump`
- Compressed with pg_dump custom format
- Automatic cleanup of old backups (configurable retention)

### Database Restore

```powershell
# Restore from backup (with confirmation)
powershell -File .\scripts\restore-db.ps1 -BackupPath backups/mathledger-20250108-020000.dump

# Force restore without confirmation
powershell -File .\scripts\restore-db.ps1 -BackupPath backups/backup.dump -Force

# Dry run (shows restore plan)
powershell -File .\scripts\restore-db.ps1 -BackupPath backups/backup.dump -DryRun
```

**⚠️ WARNING**: Restore operations will destroy the existing database and replace it with the backup data.

### Restore Process

1. **Validation**: Checks backup file integrity
2. **Planning**: Shows restore plan and estimated time
3. **Confirmation**: Requires user confirmation (unless -Force)
4. **Execution**: Drops existing DB, creates fresh DB, restores data
5. **Verification**: Checks table counts and base theory existence

## Sanity Checks

### Comprehensive Sanity Check

```powershell
# Run full sanity check
powershell -File .\scripts\sanity.ps1

# Verbose output
powershell -File .\scripts\sanity.ps1 -Verbose
```

The sanity check verifies:
1. **Python Environment**: Python and uv availability
2. **Test Suite**: Runs `pytest -q` to verify code integrity
3. **Small Derivation**: Runs derivation with limited parameters
4. **Metrics Endpoint**: Checks API accessibility and data

### Health Check

```powershell
# Check all services
powershell -File .\scripts\healthcheck.ps1

# Verbose output
powershell -File .\scripts\healthcheck.ps1 -Verbose
```

Health check verifies:
- Docker containers (PostgreSQL, Redis)
- Database connectivity and size
- Redis connectivity and memory usage
- FastAPI health and metrics endpoints
- Lean project build status

## Troubleshooting

### Common Issues

#### Service Not Running

```powershell
# Check Docker containers
docker ps

# Check FastAPI service
curl http://localhost:8000/health

# Check database connection
docker exec infra-postgres-1 psql -U ml -d mathledger -c "SELECT 1;"
```

#### Derivation Failures

```powershell
# Check recent logs
Get-Content logs/derive-*.log | Select-Object -Last 50

# Run with verbose output
powershell -File .\scripts\derive-nightly.ps1 -Verbose

# Check queue status
docker exec infra-redis-1 redis-cli llen ml:jobs
```

#### Database Issues

```powershell
# Check database size
docker exec infra-postgres-1 psql -U ml -d mathledger -c "SELECT pg_size_pretty(pg_database_size('mathledger'));"

# Check table counts
docker exec infra-postgres-1 psql -U ml -d mathledger -c "SELECT COUNT(*) FROM statements; SELECT COUNT(*) FROM proofs;"

# Run database maintenance
powershell -File .\scripts\db-maintenance.ps1
```

#### Export Failures

```powershell
# Check export directory permissions
Test-Path exports
Get-ChildItem exports

# Run export with verbose output
powershell -File .\scripts\export-snapshot.ps1 -Verbose
```

### Log Files

All operations create timestamped log files in `logs/`:

- `sanity-YYYYMMDD-HHMMSS.log`: Sanity check results
- `derive-YYYYMMDD-HHMMSS.log`: Derivation run logs
- `nightly-YYYYMMDD-HHMMSS/`: Nightly run directory with all logs
- `healthcheck-YYYYMMDD-HHMMSS.log`: Health check results

### Performance Monitoring

```powershell
# Check system metrics
curl http://localhost:8000/metrics | ConvertFrom-Json

# Monitor queue length
docker exec infra-redis-1 redis-cli llen ml:jobs

# Check database performance
docker exec infra-postgres-1 psql -U ml -d mathledger -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

## Task Scheduler Setup

### Windows Task Scheduler

Create scheduled tasks for automated operations:

#### Nightly Derivation (Daily at 2 AM)

```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-01-08T02:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-File "C:\dev\mathledger\scripts\run-nightly.ps1"</Arguments>
      <WorkingDirectory>C:\dev\mathledger</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

#### Metrics Monitoring (Every 5 Minutes)

```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-01-08T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT5M</Interval>
        <Duration>P1D</Duration>
      </Repetition>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-File "C:\dev\mathledger\scripts\metrics-watch.ps1"</Arguments>
      <WorkingDirectory>C:\dev\mathledger</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

### PowerShell Registration

```powershell
# Register nightly task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File `"C:\dev\mathledger\scripts\run-nightly.ps1`"" -WorkingDirectory "C:\dev\mathledger"
$trigger = New-ScheduledTaskTrigger -Daily -At 2AM
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -TaskName "MathLedger Nightly" -Description "MathLedger nightly derivation and maintenance"

# Register metrics monitoring task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File `"C:\dev\mathledger\scripts\metrics-watch.ps1`"" -WorkingDirectory "C:\dev\mathledger"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -TaskName "MathLedger Metrics Watch" -Description "MathLedger metrics monitoring and alerting"
```

## Best Practices

### Operational Guidelines

1. **Always run sanity checks** before major operations
2. **Monitor logs regularly** for early issue detection
3. **Keep backups current** and test restore procedures
4. **Set up alerting** for production environments
5. **Use dry-run mode** for testing configuration changes
6. **Document any custom configurations** or modifications

### Performance Optimization

1. **Adjust derivation limits** based on system capacity
2. **Monitor database growth** and plan for maintenance windows
3. **Use incremental exports** for large datasets
4. **Schedule maintenance** during low-usage periods
5. **Monitor resource usage** (CPU, memory, disk)

### Security Considerations

1. **Secure webhook URLs** and rotate them regularly
2. **Limit database access** to necessary services only
3. **Monitor for unusual activity** in logs and metrics
4. **Keep backup files secure** and encrypted if needed
5. **Use least-privilege principles** for service accounts

## Support

For issues or questions:

1. Check the logs in `logs/` directory
2. Run sanity checks to identify problems
3. Review this documentation for troubleshooting steps
4. Check the main MathLedger documentation for system architecture details

---

*Last updated: January 8, 2025*
