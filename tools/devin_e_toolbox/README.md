# Devin E Toolbox

> "A tool is done only when no one ever has to ask how to use it."

The Devin E Toolbox transforms repetitive MathLedger operations into single-command utilities. Built by Devin E, the forge-master of developer tools, to enable teammates by replacing toil with automation.

## Philosophy

Every repetitive action in the fleet becomes a single-command utility. No more hunting through documentation, no more copy-pasting commands, no more "how do I...?" questions. Just tools that work.

## Quick Start

```bash
# Add toolbox to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$PATH:/path/to/mathledger/tools/devin_e_toolbox"

# Check environment
ml env check

# Run smoke test
ml derive smoke

# Run tests
ml test unit

# Create PR
python pr-helper.py check
python pr-helper.py create --title "feat: add feature" --tag POA
```

## Tools

### 1. ml - MathLedger CLI

The main CLI tool for all MathLedger operations. Single command for everything.

#### Environment Management

```bash
ml env check        # Check environment setup
ml env setup        # Set default environment variables
```

#### Database Operations

```bash
ml db stats         # Show database statistics
ml db backup        # Create database backup
ml db maintenance   # Run VACUUM ANALYZE
ml db connect       # Open psql shell
```

#### Derivation

```bash
ml derive smoke         # Quick PL smoke test (10 steps)
ml derive smoke-fol     # Quick FOL smoke test (50 steps)
ml derive quick         # Fast PL derivation (100 steps)
ml derive nightly       # Standard nightly run (300 steps)
ml derive guided        # FOL guided derivation (3600 steps)
```

#### Testing

```bash
ml test unit            # Run unit tests only
ml test integration     # Run integration tests only
ml test all             # Run all tests
ml test coverage        # Run with coverage report
ml test fast            # Run fast tests only
ml test smoke           # Run smoke tests
```

#### Sanity Checks

```bash
ml check ascii      # Check ASCII-only compliance
ml check branch     # Validate branch naming
ml check metrics    # Lint metrics JSONL files
ml check sanity     # Run comprehensive sanity check
ml check all        # Run all checks
```

#### Code Scaffolding

```bash
ml scaffold test my_module      # Create tests/test_my_module.py
ml scaffold script my_script    # Create scripts/my_script.py
```

#### Metrics

```bash
ml metrics show     # Show current metrics from API
ml metrics lint     # Lint metrics JSONL file
ml metrics export   # Export metrics to CSV
```

#### Data Export

```bash
ml export statements                    # Export statements
ml export statements --format jsonl     # Export as JSONL
ml export metrics --output my.csv       # Export metrics to CSV
```

#### Help

```bash
ml help             # Show all commands
ml help env         # Show detailed help for env command
ml help db          # Show detailed help for db command
```

### 2. quick-start.sh - Zero to Green in 30 Minutes

Bootstrap script that gets you from clone to first green run.

```bash
./quick-start.sh
```

What it does:
1. Checks prerequisites (Python, Docker)
2. Creates virtual environment
3. Installs dependencies
4. Starts infrastructure (PostgreSQL + Redis)
5. Runs database migrations
6. Executes smoke test
7. Shows next steps

### 3. pr-helper.py - PR Automation

Automates PR creation with proper formatting and strategic differentiator tags.

#### Check Before PR

```bash
python pr-helper.py check
```

Runs:
- Branch naming validation
- ASCII compliance check
- Artifact detection
- Unit tests

#### Generate PR Template

```bash
python pr-helper.py template --tag POA
python pr-helper.py template --tag ASD --output pr_body.md
```

#### Create PR

```bash
python pr-helper.py create --title "feat: add guided derivation" --tag POA
python pr-helper.py create --title "perf: optimize modus ponens" --tag ASD --draft
```

Strategic Differentiator Tags:
- **POA** - Proof of Automation
- **ASD** - Algorithmic Superiority Demonstration
- **RC** - Reliability & Correctness
- **ME** - Metrics & Evidence
- **IVL** - Integration & Validation Layer
- **NSF** - Network Security & Forensics
- **FM** - Formal Methods

### 4. workflow-validator.py - Development Workflow Compliance

Validates development workflow compliance before commits and PRs.

```bash
# Run all checks
python workflow-validator.py --check all

# Run specific checks
python workflow-validator.py --check branch
python workflow-validator.py --check commits
python workflow-validator.py --check ascii
python workflow-validator.py --check artifacts
python workflow-validator.py --check coverage

# Fix issues
python workflow-validator.py --fix ascii
```

## Installation

### Option 1: Add to PATH (Recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/mathledger/tools/devin_e_toolbox"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Option 2: Create Symlinks

```bash
sudo ln -s /path/to/mathledger/tools/devin_e_toolbox/ml /usr/local/bin/ml
```

### Option 3: Use Directly

```bash
cd /path/to/mathledger
./tools/devin_e_toolbox/ml env check
```

## Common Workflows

### Starting a New Feature

```bash
# Check environment
ml env check

# Create feature branch
git checkout main
git pull --ff-only
git checkout -b feature/my-feature

# Run tests to ensure clean baseline
ml test unit

# Make changes...

# Validate before commit
python workflow-validator.py --check all

# Create PR
python pr-helper.py create --title "feat: my feature" --tag POA
```

### Daily Development

```bash
# Morning: Check environment
ml env check
ml db stats

# Run tests frequently
ml test fast

# Before committing
ml check all

# Before PR
python pr-helper.py check
```

### Debugging Issues

```bash
# Check environment
ml env check

# Check database
ml db stats
ml db connect

# Run smoke test
ml derive smoke

# Check metrics
ml metrics show

# Run sanity checks
ml check sanity
```

### Performance Testing

```bash
# Run baseline derivation
ml derive quick

# Run guided derivation
ml derive guided --seed 42

# Export and analyze metrics
ml metrics export
```

## Tool Design Principles

1. **Single Command**: Every operation should be a single command
2. **No Documentation Required**: Tools should be self-explanatory
3. **Fail Fast**: Clear error messages, no silent failures
4. **Idempotent**: Safe to run multiple times
5. **ASCII-Only**: All output is ASCII-only
6. **Network-Free**: No external dependencies where possible
7. **Versioned**: Tools are versioned with the repository

## Troubleshooting

### ml command not found

```bash
# Check if toolbox is in PATH
echo $PATH | grep devin_e_toolbox

# If not, add to PATH
export PATH="$PATH:/path/to/mathledger/tools/devin_e_toolbox"
```

### Permission denied

```bash
# Make tools executable
chmod +x tools/devin_e_toolbox/ml
chmod +x tools/devin_e_toolbox/*.sh
chmod +x tools/devin_e_toolbox/*.py
```

### Environment variables not set

```bash
# Set default environment
ml env setup

# Or manually
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
export REDIS_URL="redis://localhost:6379/0"
export PYTHONPATH="$(pwd)"
```

### Docker containers not running

```bash
# Start infrastructure
docker compose up -d postgres redis

# Check status
docker ps
```

## Contributing

When adding new tools to the toolbox:

1. **Follow naming convention**: Use lowercase with hyphens
2. **Make executable**: `chmod +x tool-name`
3. **Add help text**: Include usage documentation
4. **Test thoroughly**: Ensure tool works in clean environment
5. **Update README**: Document the new tool
6. **ASCII-only**: All code and documentation must be ASCII-only

## Version History

- **v1.0.0** (2025-10-19): Initial release
  - ml CLI tool with 8 command categories
  - quick-start.sh bootstrap script
  - pr-helper.py for PR automation
  - workflow-validator.py for compliance checks

## Support

For issues or questions:

1. Check this README
2. Run `ml help <command>` for detailed help
3. Check the main MathLedger documentation
4. Create a GitHub issue

---

**Built by Devin E - The Toolsmith**

*"No drift, no delay, no defeat. Build fast, verify twice, and leave every process cleaner than you found it."*
<!-- BEGIN AUTO DOCS -->

ASCII Auto-Generated Docs
==========================

ml (MathLedger CLI)
--------------------
MathLedger Toolbox (ml) - Single-command utilities for MathLedger operations

Usage:
    ml env check                    # Check environment setup
    ml db stats                     # Show database statistics
    ml derive smoke                 # Run smoke test derivation
    ml test unit                    # Run unit tests
    ml check all                    # Run all sanity checks
    ml scaffold test my_module      # Generate test boilerplate
    ml metrics show                 # Show current metrics
    ml metrics diff --last 10       # Compare last 10 runs
    ml export statements            # Export statements
    ml build deterministic          # Deterministic build with hash verification
    ml replay seed 42               # Replay derivation with specific seed
    ml verify artifacts             # Verify artifact integrity
    ml audit sync                   # Synchronize audit trail
    ml audit verify                 # Verify audit chain integrity
    ml velocity seal                # Measure CI velocity and seal artifact
    ml audit chain                  # Run audit sync+verify and seal chain
    ml allblue freeze               # Freeze all-blue CI state with signature
    ml flightdeck                   # Run all operator flows and generate consolidated report
    ml flightdeck --preflight       # Run preflight checks only
    ml flightdeck --dry-run         # Dry-run mode with NO_NETWORK
    ml flightdeck --parallel        # Run operations in parallel
    ml flightdeck --sign            # Sign evidence bundle
    ml flightdeck --force-preflight # Force preflight checks (bypass cache)
    ml flightdeck --verify          # Verify sealed report
    ml flightdeck --verify-signature # Verify bundle signature

For detailed help:
    ml --help

deterministic_build.py
----------------------
Deterministic Build - Reproducible builds with hash verification

Ensures that builds are deterministic and verifiable by:
1. Setting SOURCE_DATE_EPOCH for reproducible timestamps
2. Computing hashes of all source files
3. Building with fixed parameters
4. Verifying output hashes match expected values

Usage:
    python deterministic_build.py
    python deterministic_build.py --verify-only
    python deterministic_build.py --save-baseline

seed_replay.py
--------------
Seed Replay - Reproduce derivations with specific seeds

Enables deterministic replay of derivations by:
1. Loading seed and parameters from metrics/blocks
2. Re-running derivation with exact same configuration
3. Verifying output matches original run
4. Comparing Merkle roots for integrity

Usage:
    python seed_replay.py --seed 42
    python seed_replay.py --block 1592
    python seed_replay.py --seed 101 --system fol --mode guided
    python seed_replay.py --verify-only --seed 42

artifact_verifier.py
--------------------
Artifact Verifier - Verify integrity of artifacts and proofs

Verifies:
1. Artifact file integrity (checksums, format validation)
2. Merkle root consistency across blocks
3. Proof parent-child relationships
4. JSONL schema compliance

Usage:
    python artifact_verifier.py --all
    python artifact_verifier.py --merkle
    python artifact_verifier.py --proofs
    python artifact_verifier.py --artifacts
    python artifact_verifier.py --file artifacts/wpv5/run_metrics.jsonl

pr-helper.py
------------
PR Helper - Automate PR creation with proper formatting and checks

Integrates with Cursor P (GitOps Conductor) for coordinated PR workflows.

Usage:
    python pr-helper.py create --title "feat: add feature" --tag POA
    python pr-helper.py check                    # Run pre-PR checks
    python pr-helper.py template --tag ASD       # Generate PR template
    python pr-helper.py cursor-sync              # Sync with Cursor P conductor
    python pr-helper.py cursor-status            # Check Cursor P status

workflow-validator.py
---------------------
Workflow Validator - Validate development workflow compliance

Checks:
- Branch naming convention
- Commit message format
- No direct pushes to main
- ASCII-only content
- No artifacts in commits
- Test coverage maintained

Usage:
    python workflow-validator.py --check all
    python workflow-validator.py --check branch
    python workflow-validator.py --check commits
    python workflow-validator.py --fix ascii


<!-- END AUTO DOCS -->
