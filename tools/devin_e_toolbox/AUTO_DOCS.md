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
