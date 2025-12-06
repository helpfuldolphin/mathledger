# Local Development Guardrails

This document describes the local development tools and guardrails available in the mathledger project.

**Note:** These hooks are not wired into CI - they are purely local development tools.

## Pre-commit Hooks

Pre-commit hooks provide automated checks that run before each commit to ensure code quality and consistency.

### Quick Start

```bash
pip install pre-commit
pre-commit install
make check-local
```

### Installation

1. Install pre-commit (if not already installed):
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks:
   ```bash
   pre-commit install
   ```

### Usage

- **Automatic**: Hooks run automatically on `git commit`
- **Manual**: Run on all files with:
  ```bash
  pre-commit run --all-files
  ```
- **Specific hook**: Run a specific hook with:
  ```bash
  pre-commit run <hook-id>
  ```

### What It Checks

The pre-commit configuration includes:

- **Trailing whitespace removal**: Removes trailing spaces and ensures files end with newlines
- **File format validation**: Checks YAML, JSON, and TOML files for validity
- **ASCII-only enforcement**: Ensures documentation and script files contain only ASCII characters
- **Large file detection**: Prevents accidentally committing large files (>1MB)
- **Merge conflict detection**: Identifies unresolved merge conflict markers
- **Case conflict detection**: Prevents case-sensitive filesystem issues

### ASCII-Only Check

The ASCII-only check applies to files in:
- `docs/` - Documentation files
- `scripts/` - Shell scripts and automation
- `qa/` - Quality assurance files
- `tools/qa/` - QA tools

Supported file extensions: `.md`, `.py`, `.sh`, `.ps1`, `.yaml`, `.yml`, `.json`, `.txt`

## Branch Naming Guard

A local script validates branch names to ensure they follow the project's naming convention.

### Branch Naming Convention

Branches must match the pattern: `^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$`

Examples of valid branch names:
- `feature/user-authentication`
- `perf/database-optimization`
- `ops/deployment-scripts`
- `qa/test-coverage`
- `devxp/local-guardrails`
- `docs/api-documentation`

Examples of invalid branch names:
- `main` (should not commit directly to main)
- `feature/User-Auth` (uppercase not allowed)
- `feature/user_auth` (underscores not allowed)
- `user-auth` (missing prefix)
- `feature/` (empty suffix)

### Running the Branch Guard

The branch guard is integrated into the Makefile:

```bash
make check-local
```

This will:
1. Validate the current branch name
2. Display "OK: local guardrails passed" if valid
3. Exit with error code if invalid

## Make Targets

- `make check-local`: Run local development guardrails (branch naming check)

## Troubleshooting

### Pre-commit Issues

If pre-commit hooks fail:

1. **Fix the issues** shown in the output
2. **Stage the fixes**: `git add <fixed-files>`
3. **Commit again**: `git commit`

To skip hooks temporarily (not recommended):
```bash
git commit --no-verify
```

### Branch Naming Issues

If the branch guard fails:

1. **Rename your branch** to follow the convention
2. **Push the renamed branch**: `git push origin <new-branch-name>`
3. **Update any open PRs** to point to the new branch name

## Integration with CI

These local guardrails are designed to run locally and do not interfere with the CI/CD pipeline. They provide immediate feedback during development without requiring CI resources.

The pre-commit hooks and branch naming checks are purely local tools that help maintain code quality and project standards before code reaches the CI system.
