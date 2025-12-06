# Linter Lane PR

## Scope
This PR addresses linting issues and code quality improvements across the codebase. Changes may include:
- Code style fixes (PEP 8, formatting)
- Import organization and cleanup
- Type hint improvements
- Documentation string updates
- Unused variable/import removal
- Deprecation warning fixes

## How to Run

### PowerShell (Windows)
```powershell
# Install dependencies
pip install -r requirements.txt

# Run linter checks
python -m flake8 backend/ tests/ --max-line-length=88 --extend-ignore=E203,W503
python -m black --check backend/ tests/
python -m isort --check-only backend/ tests/

# Run type checking
python -m mypy backend/ --ignore-missing-imports

# Run specific test subset
$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py
```

### Ubuntu/Linux
```bash
# Install dependencies
pip install -r requirements.txt

# Run linter checks
python -m flake8 backend/ tests/ --max-line-length=88 --extend-ignore=E203,W503
python -m black --check backend/ tests/
python -m isort --check-only backend/ tests/

# Run type checking
python -m mypy backend/ --ignore-missing-imports

# Run specific test subset
NO_NETWORK=true PYTHONPATH=$(pwd) pytest -q tests/test_canon.py tests/test_mp.py tests/test_subst.py tests/test_taut.py tests/devxp/test_branch_guard.py tests/qa/test_metrics_lint_v1.py
```

## Acceptance Criteria
- [ ] All linter checks pass without errors
- [ ] Code formatting is consistent (Black, isort)
- [ ] No new type checking errors introduced
- [ ] Test subset passes: `$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py`
- [ ] No new deprecation warnings
- [ ] Import statements are properly organized
- [ ] Code follows project style guidelines

## Repro Commands
```bash
# Quick validation
$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_canon.py tests\test_mp.py tests\test_subst.py tests\test_taut.py tests\devxp\test_branch_guard.py tests\qa\test_metrics_lint_v1.py

# Full linter suite
python -m flake8 backend/ tests/ --max-line-length=88 --extend-ignore=E203,W503
python -m black --check backend/ tests/
python -m isort --check-only backend/ tests/
python -m mypy backend/ --ignore-missing-imports
```

## Notes
- Focus on code quality improvements without changing functionality
- Ensure all changes maintain backward compatibility
- Update documentation if new patterns or conventions are introduced
