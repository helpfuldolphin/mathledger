"""
Regression test: Logging timestamps are ALLOWED in non-attestation paths.

This test documents the correct interpretation of the nondeterminism audit:

ATTESTATION-CRITICAL PATHS (timestamps banned):
- backend/repro/  - Deterministic harness code
- backend/bridge/ - Bridge layer
- ledger/         - Ledger core
- attestation/    - Attestation logic
- derivation/     - Derivation logic
- curriculum/     - Curriculum logic

NON-CRITICAL PATHS (timestamps allowed):
- rfl/metrics_logger.py      - Telemetry logging (metadata only)
- rfl/experiment_logging.py  - Experiment logging (metadata only)
- rfl/prng/governance.py     - PRNG governance logs (audit trail timestamps)
- rfl/runner.py              - RFL runner summary timestamps

The nondeterminism audit MUST NOT fail on these files because:
1. Timestamps in these files are for LOGGING, not ATTESTATION
2. They do not affect H_t = SHA256(R_t || U_t) computation
3. They are used for human-readable metadata in shadow logs

Reference: docs/FIRST_ORGANISM_DETERMINISM.md
"""

import pytest
from pathlib import Path


# Files that are ALLOWED to use datetime.now/utcnow (logging only)
ALLOWED_TIMESTAMP_FILES = [
    "rfl/metrics_logger.py",
    "rfl/experiment_logging.py",
    "rfl/prng/governance.py",
    "rfl/runner.py",
]

# Files that must NEVER use datetime.now/utcnow (attestation-critical)
BANNED_TIMESTAMP_PATHS = [
    "backend/repro",
    "backend/bridge",
    "ledger",
    "attestation",
    "derivation",
    "curriculum",
]


@pytest.mark.unit
def test_allowed_timestamp_files_exist():
    """Verify the allowed files exist and we're testing the right thing."""
    repo_root = Path(__file__).parent.parent.parent

    for file_path in ALLOWED_TIMESTAMP_FILES:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Expected allowed file {file_path} to exist"


@pytest.mark.unit
def test_allowed_files_contain_timestamps():
    """Verify the allowed files actually contain timestamp calls (regression test)."""
    import re

    repo_root = Path(__file__).parent.parent.parent
    timestamp_pattern = re.compile(r"datetime\.(now|utcnow)")

    for file_path in ALLOWED_TIMESTAMP_FILES:
        full_path = repo_root / file_path
        if not full_path.exists():
            continue

        content = full_path.read_text(encoding="utf-8")
        matches = timestamp_pattern.findall(content)

        # These files SHOULD contain timestamps (that's their purpose)
        # If this test fails, someone removed timestamps - which is fine,
        # but we should update the ALLOWED list
        assert len(matches) > 0 or "# NO TIMESTAMPS" in content, (
            f"Expected {file_path} to contain datetime.now/utcnow calls. "
            f"If timestamps were intentionally removed, update ALLOWED_TIMESTAMP_FILES."
        )


@pytest.mark.unit
def test_banned_paths_no_timestamps():
    """Verify attestation-critical paths don't contain bare timestamps."""
    import re

    repo_root = Path(__file__).parent.parent.parent
    timestamp_pattern = re.compile(r"\bdatetime\.(now|utcnow)\s*\(")

    violations = []

    for banned_path in BANNED_TIMESTAMP_PATHS:
        path = repo_root / banned_path
        if not path.exists():
            continue

        for py_file in path.rglob("*.py"):
            # Skip test files
            if "test_" in py_file.name or "_test.py" in py_file.name:
                continue

            content = py_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Skip documentation comments that mention banned patterns
                if "NO datetime" in line or "- NO datetime" in line:
                    continue
                # Skip lines that are just comments (e.g., "# datetime.now banned")
                if "#" in line and timestamp_pattern.search(line):
                    # Check if the match is before the #
                    comment_pos = line.index("#")
                    match = timestamp_pattern.search(line)
                    if match and match.start() > comment_pos:
                        continue  # Pattern is in comment

                if timestamp_pattern.search(line):
                    violations.append(f"{py_file.relative_to(repo_root)}:{i}")

    assert len(violations) == 0, (
        f"Found {len(violations)} timestamp call(s) in attestation-critical paths:\n"
        + "\n".join(violations[:10])
        + "\n\nSee docs/FIRST_ORGANISM_DETERMINISM.md for migration guide."
    )


@pytest.mark.unit
def test_rfl_timestamps_are_metadata_only():
    """Document that rfl/ timestamps are for metadata, not attestation hashes."""
    # This is a documentation test - it exists to explain WHY timestamps
    # are allowed in rfl/ and to make the policy explicit.

    # The rfl/ module provides:
    # 1. metrics_logger.py - Logging telemetry to shadow logs
    # 2. experiment_logging.py - Experiment audit trails
    # 3. prng/governance.py - PRNG governance audit records
    # 4. runner.py - Run summary metadata

    # None of these affect the dual-root attestation:
    #   H_t = SHA256(R_t || U_t)
    #
    # R_t (Lean proof tree root) is computed from Lean outputs
    # U_t (USLA state root) is computed from deterministic state snapshots
    #
    # Logging timestamps are AFTER attestation, for human-readable records only.

    assert True, "This test documents the timestamp policy"
