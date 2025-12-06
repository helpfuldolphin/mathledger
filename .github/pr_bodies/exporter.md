# Exporter Lane PR

## Scope
This PR addresses data export functionality and related infrastructure. Changes may include:
- Database export utilities and scripts
- Data serialization improvements
- Export format enhancements
- Migration script updates
- Backup and restore functionality
- Data validation and integrity checks
- Export performance optimizations

## How to Run

### PowerShell (Windows)
```powershell
# Install dependencies
pip install -r requirements.txt

# Run export functionality tests
$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py

# Test export scripts
python scripts/export-snapshot.ps1
python scripts/backup-db.ps1

# Run migration tests
python run_migration.py
python test_migration.py

# Validate export data integrity
python -m pytest tests/qa/ -v
```

### Ubuntu/Linux
```bash
# Install dependencies
pip install -r requirements.txt

# Run export functionality tests
NO_NETWORK=true PYTHONPATH=$(pwd) pytest -q tests/test_canon.py tests/test_mp.py tests/test_subst.py tests/test_taut.py tests/devxp/test_branch_guard.py tests/qa/test_metrics_lint_v1.py

# Test export scripts (if available)
python scripts/export-snapshot.sh
python scripts/backup-db.sh

# Run migration tests
python run_migration.py
python test_migration.py

# Validate export data integrity
python -m pytest tests/qa/ -v
```

## Acceptance Criteria
- [ ] All export-related tests pass
- [ ] Test subset passes: `$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py`
- [ ] Export scripts execute without errors
- [ ] Data integrity is maintained during export/import cycles
- [ ] Migration scripts run successfully
- [ ] Export formats are valid and consistent
- [ ] Performance meets requirements for expected data volumes
- [ ] Error handling is robust for edge cases

## Repro Commands
```bash
# Core test validation
$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py

# Export functionality validation
python -m pytest tests/qa/ -v

# Migration validation
python run_migration.py
python test_migration.py

# Data integrity check
python -c "from backend.tools.db_stats import main; main()"
```

## Notes
- Ensure backward compatibility with existing export formats
- Test with various data sizes and edge cases
- Verify data integrity throughout export/import processes
- Document any changes to export schemas or formats
- Consider performance impact on large datasets
