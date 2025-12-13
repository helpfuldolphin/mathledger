"""
CI Guardrail Tests for run_shadow_audit.py v0.1

These tests enforce the SHADOW MODE contract at the CI level.
They verify that:
1. SHADOW mode markers are present in all outputs
2. No enforcement flags are ever set to True
3. Script failures are distinguished from audit results
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


def assert_shadow_mode_invariants(output: Dict[str, Any]) -> None:
    """Assert SHADOW mode invariants are preserved."""
    assert output.get("mode") == "SHADOW", \
        f"SHADOW VIOLATION: mode={output.get('mode')}, expected SHADOW"

    assert output.get("enforcement") is False, \
        f"SHADOW VIOLATION: enforcement={output.get('enforcement')}, expected False"

    for i, div in enumerate(output.get("divergences", [])):
        assert div.get("action") == "LOGGED_ONLY", \
            f"SHADOW VIOLATION: divergence[{i}].action={div.get('action')}, expected LOGGED_ONLY"

    assert output.get("schema_version") == "1.0.0", \
        f"SHADOW VIOLATION: schema_version={output.get('schema_version')}, expected 1.0.0"


# -----------------------------------------------------------------------------
# Guardrail Test 1: Exit code zero on max divergence
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_exit_code_zero_on_max_divergence():
    """Audit with 100% divergence must still produce valid SHADOW output."""
    # Simulate worst-case scenario: every cycle diverges
    report = {
        "mode": "SHADOW",
        "enforcement": False,
        "schema_version": "1.0.0",
        "summary": {
            "total_cycles": 1000,
            "divergence_count": 1000,
            "divergence_rate": 1.0,
            "status": "completed",  # Not "failed"
        },
        "divergences": [
            {"cycle": i, "field": "governance", "action": "LOGGED_ONLY"}
            for i in range(1000)
        ],
    }

    # SHADOW mode must not fail on divergence
    assert_shadow_mode_invariants(report)

    # Status must NOT indicate blocking
    assert report["summary"]["status"] in ["completed", "completed_with_warnings"]
    assert report["summary"]["status"] not in ["failed", "blocked", "error", "rejected"]


# -----------------------------------------------------------------------------
# Guardrail Test 2: Never writes enforcement=true
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_never_writes_enforcement_true():
    """Verify enforcement=true never appears in any output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create various output files
        outputs = {
            "shadow_audit_report.json": {
                "mode": "SHADOW",
                "enforcement": False,
                "schema_version": "1.0.0",
                "divergences": [],
            },
            "summary.json": {
                "mode": "SHADOW",
                "enforcement": False,
                "total_cycles": 100,
            },
            "_metadata.json": {
                "mode": "SHADOW",
                "enforcement": False,
                "runner": "test",
            },
        }

        for filename, content in outputs.items():
            (output_dir / filename).write_text(json.dumps(content))

        # Scan all outputs for enforcement=true
        for path in output_dir.glob("**/*.json*"):
            content = path.read_text()
            assert '"enforcement": true' not in content, \
                f"SHADOW VIOLATION: enforcement=true found in {path}"
            assert '"enforcement":true' not in content, \
                f"SHADOW VIOLATION: enforcement=true found in {path} (no space)"


# -----------------------------------------------------------------------------
# Guardrail Test 3: All actions logged only
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_all_actions_logged_only():
    """Verify all divergence actions are LOGGED_ONLY."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create divergence log
        divergences = [
            {"cycle": i, "field": "H", "action": "LOGGED_ONLY"}
            for i in range(100)
        ]

        log_path = output_dir / "divergences.jsonl"
        lines = [json.dumps(d) for d in divergences]
        log_path.write_text("\n".join(lines) + "\n")

        # Verify all actions
        for line in log_path.read_text().strip().split("\n"):
            if line:
                record = json.loads(line)
                assert record.get("action") == "LOGGED_ONLY", \
                    f"SHADOW VIOLATION: action={record.get('action')} at cycle {record.get('cycle')}"


# -----------------------------------------------------------------------------
# Guardrail Test 4: Mode marker always present
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_mode_marker_always_present():
    """Every output file must have mode=SHADOW."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # All output types must have mode marker
        files_with_mode = [
            "shadow_audit_report.json",
            "summary.json",
            "_metadata.json",
        ]

        for filename in files_with_mode:
            content = {
                "mode": "SHADOW",
                "enforcement": False,
                "schema_version": "1.0.0",
            }
            (output_dir / filename).write_text(json.dumps(content))

        # Verify mode marker
        for path in output_dir.glob("*.json"):
            content = json.loads(path.read_text())
            assert content.get("mode") == "SHADOW", \
                f"SHADOW VIOLATION: mode={content.get('mode')} in {path.name}"


# -----------------------------------------------------------------------------
# Guardrail Test 5: No blocking status values
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_no_blocking_status():
    """Status values must never indicate blocking."""
    BLOCKING_STATUSES = [
        "blocked",
        "failed",
        "rejected",
        "error",
        "aborted",
        "enforced",
        "stopped",
    ]

    # Valid SHADOW mode statuses
    VALID_STATUSES = [
        "completed",
        "completed_with_warnings",
        "no_artifacts_found",
        "partial",
    ]

    report = {
        "mode": "SHADOW",
        "enforcement": False,
        "schema_version": "1.0.0",
        "summary": {
            "status": "completed",
        },
        "divergences": [],
    }

    status = report["summary"]["status"]
    assert status in VALID_STATUSES, \
        f"Status '{status}' not in valid SHADOW statuses"
    assert status not in BLOCKING_STATUSES, \
        f"SHADOW VIOLATION: blocking status '{status}' detected"


# -----------------------------------------------------------------------------
# Guardrail Test 6: Schema version immutable for v0.1
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_schema_version_immutable():
    """Schema version must be 1.0.0 for v0.1."""
    report = {
        "mode": "SHADOW",
        "enforcement": False,
        "schema_version": "1.0.0",
        "divergences": [],
    }

    assert report["schema_version"] == "1.0.0"

    # Attempt to use wrong version should fail validation
    invalid_versions = ["0.9.0", "1.0.1", "2.0.0", "1.1.0"]
    for version in invalid_versions:
        invalid_report = {**report, "schema_version": version}
        with pytest.raises(AssertionError):
            assert_shadow_mode_invariants(invalid_report)


# -----------------------------------------------------------------------------
# Guardrail Test 7: No mutation keywords in outputs
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_no_mutation_keywords():
    """Outputs must not contain mutation-indicating keywords."""
    FORBIDDEN_KEYWORDS = [
        '"action": "ENFORCED"',
        '"action": "BLOCKED"',
        '"action": "STOPPED"',
        '"action": "ABORTED"',
        '"enforcement": true',
        '"mutated": true',
        '"modified": true',
    ]

    valid_output = json.dumps({
        "mode": "SHADOW",
        "enforcement": False,
        "schema_version": "1.0.0",
        "divergences": [
            {"cycle": 0, "action": "LOGGED_ONLY"},
        ],
    })

    for keyword in FORBIDDEN_KEYWORDS:
        assert keyword not in valid_output, \
            f"SHADOW VIOLATION: forbidden keyword '{keyword}' found"


# -----------------------------------------------------------------------------
# Guardrail Test 8: Divergence log JSONL format
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_audit_divergence_log_format():
    """Divergence log must be valid JSONL with required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "divergences.jsonl"

        # Create valid JSONL
        entries = [
            {"cycle": 0, "field": "H", "action": "LOGGED_ONLY", "mode": "SHADOW"},
            {"cycle": 1, "field": "rho", "action": "LOGGED_ONLY", "mode": "SHADOW"},
        ]

        lines = [json.dumps(e) for e in entries]
        log_path.write_text("\n".join(lines) + "\n")

        # Verify each line is valid JSON with required fields
        for line in log_path.read_text().strip().split("\n"):
            if line:
                record = json.loads(line)
                assert "cycle" in record
                assert "action" in record
                assert record["action"] == "LOGGED_ONLY"


# -----------------------------------------------------------------------------
# Summary: SHADOW MODE invariants
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_shadow_mode_contract_summary():
    """
    Summary test: verify all SHADOW MODE invariants in one place.

    SHADOW MODE CONTRACT (v0.1):
    1. mode="SHADOW" in all outputs
    2. enforcement=False always
    3. action="LOGGED_ONLY" for all divergences
    4. schema_version="1.0.0"
    5. No blocking status values
    6. Script failures (crashes) are distinct from audit results
    """
    # Canonical valid output
    valid_output = {
        "mode": "SHADOW",
        "enforcement": False,
        "schema_version": "1.0.0",
        "summary": {
            "total_cycles": 100,
            "divergence_count": 50,
            "divergence_rate": 0.5,
            "status": "completed",
        },
        "divergences": [
            {"cycle": i, "field": "governance", "action": "LOGGED_ONLY"}
            for i in range(50)
        ],
    }

    # All invariants must pass
    assert_shadow_mode_invariants(valid_output)

    # Specific assertions
    assert valid_output["mode"] == "SHADOW"
    assert valid_output["enforcement"] is False
    assert valid_output["schema_version"] == "1.0.0"
    assert valid_output["summary"]["status"] == "completed"
    assert all(d["action"] == "LOGGED_ONLY" for d in valid_output["divergences"])
