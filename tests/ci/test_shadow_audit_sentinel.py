"""
SENTINEL TEST: Shadow Audit Contract Gatekeeper

This test guards against non-compliant implementations landing.

Behavior:
- PASS if scripts/run_shadow_audit.py does NOT exist (pre-implementation)
- FAIL if script exists but violates contract (catches bad PRs)
- PASS if script exists and is fully compliant

Contract requirements (CANONICAL - flags-only CLI):
1. schema_version="1.0.0" (immutable for v0.1)
2. mode="SHADOW" in output
3. CLI flags: --input, --output, --seed, --verbose, --dry-run (NO subcommands)
4. Exit codes: 0=OK, 1=FATAL, 2=RESERVED

Owner: CLAUDE V (Gatekeeper)
Source of Truth: docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md
Implementer: Claude S (PR-1)
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def find_script() -> Path:
    """Find run_shadow_audit.py from repo root."""
    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent.parent
    return repo_root / "scripts" / "run_shadow_audit.py"


SCRIPT_PATH = find_script()


# =============================================================================
# SENTINEL TEST: Contract Compliance Gate
# =============================================================================

@pytest.mark.unit
def test_sentinel_shadow_audit_contract():
    """
    SENTINEL: Fail if script exists but violates contract.

    This test is the gatekeeper that prevents non-compliant code from merging.

    States:
    - Script doesn't exist -> PASS (expected pre-implementation)
    - Script exists, compliant -> PASS
    - Script exists, non-compliant -> FAIL (blocks merge)
    """
    if not SCRIPT_PATH.exists():
        # Pre-implementation state: script doesn't exist yet
        # This is expected and acceptable
        pytest.skip(
            f"Script not deployed yet: {SCRIPT_PATH}\n"
            "Waiting for Claude S PR-1. This is expected."
        )

    # =========================================================================
    # SCRIPT EXISTS: Now we MUST verify contract compliance
    # =========================================================================

    # -------------------------------------------------------------------------
    # Check 1: Source contains required constants
    # -------------------------------------------------------------------------
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    # schema_version="1.0.0" must be defined
    assert 'schema_version' in source.lower() or 'SCHEMA_VERSION' in source, \
        "SENTINEL FAIL: Script missing schema_version constant"

    assert '"1.0.0"' in source or "'1.0.0'" in source, \
        "SENTINEL FAIL: schema_version must be '1.0.0' for v0.1"

    # mode="SHADOW" must be in source
    assert '"SHADOW"' in source or "'SHADOW'" in source, \
        "SENTINEL FAIL: Script missing mode='SHADOW' constant"

    # -------------------------------------------------------------------------
    # Check 2: Canonical CLI flags present (--input, --output required)
    # -------------------------------------------------------------------------
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    help_text = result.stdout.lower()

    # Required canonical flags
    assert "--input" in help_text, \
        "SENTINEL FAIL: Required flag --input missing from help"
    assert "--output" in help_text, \
        "SENTINEL FAIL: Required flag --output missing from help"

    # Optional canonical flags (should also be present)
    canonical_optional = ["--seed", "--verbose", "--dry-run"]
    for flag in canonical_optional:
        assert flag in help_text, \
            f"SENTINEL FAIL: Optional flag {flag} missing from help"

    # Forbidden patterns (non-canonical)
    forbidden = ["--p3-dir", "--p4-dir", "--output-dir", "--deterministic"]
    for flag in forbidden:
        assert flag not in help_text, \
            f"SENTINEL FAIL: Non-canonical flag {flag} found (violates frozen CLI)"

    # -------------------------------------------------------------------------
    # Check 3: --dry-run exits 0 with mock input
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--input", str(input_dir),
             "--output", str(output_dir),
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, \
            f"SENTINEL FAIL: --dry-run must exit 0, got {result.returncode}\n" \
            f"stderr: {result.stderr[:500]}"

    # -------------------------------------------------------------------------
    # Check 4: Full run produces output with required markers
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        # Create minimal shadow log
        log_file = input_dir / "shadow_log_test.jsonl"
        log_file.write_text('{"_header":true,"mode":"SHADOW","schema_version":"1.0.0"}\n', encoding="utf-8")

        output_dir = Path(tmpdir) / "output"

        subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--input", str(input_dir),
             "--output", str(output_dir),
             "--seed", "42"],
            capture_output=True,
            timeout=60,
        )

        # Find run_summary.json in output
        summaries = list(output_dir.glob("**/run_summary.json"))

        if summaries:
            summary_content = summaries[0].read_text(encoding="utf-8")

            assert '"mode": "SHADOW"' in summary_content or '"mode":"SHADOW"' in summary_content, \
                "SENTINEL FAIL: run_summary.json missing mode='SHADOW'"

            assert '"1.0.0"' in summary_content, \
                "SENTINEL FAIL: run_summary.json missing schema_version='1.0.0'"

    # All checks passed - script is compliant
    print("SENTINEL PASS: Script exists and is contract-compliant")


@pytest.mark.unit
def test_sentinel_no_enforcement_true():
    """
    SENTINEL: Fail if script outputs enforcement=true.

    This is a SHADOW MODE violation that must never occur.
    """
    if not SCRIPT_PATH.exists():
        pytest.skip("Script not deployed yet")

    source = SCRIPT_PATH.read_text(encoding="utf-8")

    # enforcement=True must NEVER appear in source
    # (except in test assertions or comments)
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        # Skip comments and test assertions
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'assert' in line.lower():
            continue
        if 'test' in line.lower():
            continue

        # Check for violation
        if 'enforcement' in line.lower() and 'true' in line.lower():
            if '"enforcement": true' in line.lower() or "'enforcement': true" in line.lower():
                pytest.fail(
                    f"SENTINEL FAIL: enforcement=True found at line {i}\n"
                    f"Line: {line}\n"
                    "This violates SHADOW MODE contract."
                )


@pytest.mark.unit
def test_sentinel_utf8_encoding():
    """
    SENTINEL: Fail if script uses wrong encoding patterns.

    Windows-safe encoding requires explicit encoding="utf-8" on file writes.
    """
    if not SCRIPT_PATH.exists():
        pytest.skip("Script not deployed yet")

    source = SCRIPT_PATH.read_text(encoding="utf-8")

    # Check for open() calls without encoding
    # This is a heuristic - not perfect but catches common issues
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        if 'open(' in line and '"w"' in line or "'w'" in line:
            if 'encoding=' not in line:
                # Could be a false positive if it's in a comment
                stripped = line.strip()
                if not stripped.startswith('#'):
                    pytest.fail(
                        f"SENTINEL FAIL: File write without explicit encoding at line {i}\n"
                        f"Line: {line}\n"
                        "Must use encoding='utf-8' for Windows safety."
                    )
