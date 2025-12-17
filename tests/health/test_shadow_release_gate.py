"""
Tests for Shadow Release Gate — SHADOW MODE contract enforcement.

Test Categories:
1. Prohibited phrase detection in SHADOW-GATED contexts
2. SHADOW-OBSERVE language validation
3. Gate registry requirement checks
4. Unqualified SHADOW MODE detection
5. Edge cases and allowed technical phrases

SHADOW MODE: SHADOW-GATED
Gate Registry: SRG-001
Contract Reference: docs/system_law/SHADOW_MODE_CONTRACT.md v1.0.0
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from backend.health.shadow_release_gate import (
    ShadowReleaseGate,
    GateViolation,
    GateReport,
    scan_file,
    scan_directory,
    load_gate_registry,
    PROHIBITED_PHRASES_IN_GATED,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def gate() -> ShadowReleaseGate:
    """Create a ShadowReleaseGate instance."""
    return ShadowReleaseGate()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def create_test_file(temp_dir: Path, name: str, content: str) -> Path:
    """Helper to create a test file."""
    path = temp_dir / name
    path.write_text(content, encoding="utf-8")
    return path


# =============================================================================
# TEST CASE 1: Prohibited phrase "observational only" in SHADOW-GATED
# =============================================================================

@pytest.mark.unit
def test_prohibited_phrase_observational_only_in_gated(gate: ShadowReleaseGate, temp_dir: Path):
    """
    'observational only' is REJECTED if present in SHADOW-GATED contexts.

    Contract §4.1: "observational only" implies no blocking capability,
    which contradicts SHADOW-GATED semantics.
    """
    content = """
# Test Document

**SHADOW MODE**: SHADOW-GATED

This system operates in observational only mode.
"""
    file_path = create_test_file(temp_dir, "gated_with_observational.md", content)
    violations = gate.scan_file(file_path)

    assert len(violations) >= 1
    assert any(v.violation_type == "PROHIBITED_PHRASE_IN_GATED" for v in violations)
    assert any("observational only" in v.message.lower() for v in violations)


# =============================================================================
# TEST CASE 2: Prohibited phrase "advisory" in SHADOW-GATED
# =============================================================================

@pytest.mark.unit
def test_prohibited_phrase_advisory_in_gated(gate: ShadowReleaseGate, temp_dir: Path):
    """
    'advisory' is REJECTED in SHADOW-GATED contexts.

    Contract §4.1: "advisory" implies non-binding.
    """
    content = """
# Advisory System

**Mode**: SHADOW-GATED

The gate provides advisory notifications only.
gate_id: SRG-001
"""
    file_path = create_test_file(temp_dir, "gated_with_advisory.md", content)
    violations = gate.scan_file(file_path)

    assert len(violations) >= 1
    assert any(v.violation_type == "PROHIBITED_PHRASE_IN_GATED" for v in violations)
    assert any("advisory" in v.message.lower() for v in violations)


# =============================================================================
# TEST CASE 3: Prohibited phrase "non-blocking" in SHADOW-GATED
# =============================================================================

@pytest.mark.unit
def test_prohibited_phrase_nonblocking_in_gated(gate: ShadowReleaseGate, temp_dir: Path):
    """
    'non-blocking' is REJECTED in SHADOW-GATED contexts.

    Contract §4.1: "non-blocking" contradicts SHADOW-GATED semantics.
    """
    content = """
# Non-blocking Gate

**Mode**: SHADOW-GATED

This system operates in SHADOW-GATED mode with non-blocking behavior.

gate_id: SRG-001
"""
    file_path = create_test_file(temp_dir, "gated_with_nonblocking.md", content)
    violations = gate.scan_file(file_path)

    assert len(violations) >= 1
    assert any(v.violation_type == "PROHIBITED_PHRASE_IN_GATED" for v in violations)
    assert any("non-blocking" in v.message.lower() for v in violations)


# =============================================================================
# TEST CASE 4: SHADOW-OBSERVE allows "observational only"
# =============================================================================

@pytest.mark.unit
def test_observe_allows_observational_language(gate: ShadowReleaseGate, temp_dir: Path):
    """
    SHADOW-OBSERVE documents MAY use 'observational only' — this is correct usage.

    Contract §2.1: SHADOW-OBSERVE executes for observational purposes only.
    """
    content = """
# Observation System

**SHADOW MODE**: SHADOW-OBSERVE

This system operates in observational only mode.
It does not affect the primary control flow.
"""
    file_path = create_test_file(temp_dir, "observe_with_observational.md", content)
    violations = gate.scan_file(file_path)

    # Should NOT have PROHIBITED_PHRASE_IN_GATED violations
    # (only SHADOW-GATED triggers that check)
    prohibited_violations = [
        v for v in violations
        if v.violation_type == "PROHIBITED_PHRASE_IN_GATED"
    ]
    assert len(prohibited_violations) == 0


# =============================================================================
# TEST CASE 5: SHADOW-GATED without gate registry reference
# =============================================================================

@pytest.mark.unit
def test_gated_missing_gate_registry(gate: ShadowReleaseGate, temp_dir: Path):
    """
    SHADOW-GATED documents MUST reference gate registry.

    Contract §3.4: Operations without a gate registry entry MUST NOT be blocked.
    """
    content = """
# Missing Registry

**Mode**: SHADOW-GATED

This document declares SHADOW-GATED but has no gate_id reference.
"""
    file_path = create_test_file(temp_dir, "gated_no_registry.md", content)
    violations = gate.scan_file(file_path)

    assert len(violations) >= 1
    assert any(v.violation_type == "MISSING_GATE_REGISTRY" for v in violations)


# =============================================================================
# TEST CASE 6: SHADOW-GATED with valid gate registry reference
# =============================================================================

@pytest.mark.unit
def test_gated_with_valid_registry(gate: ShadowReleaseGate, temp_dir: Path):
    """
    SHADOW-GATED with gate_id reference should pass registry check.
    """
    content = """
# Valid Gated Document

**Mode**: SHADOW-GATED
**Gate ID**: SRG-001

This document correctly references the gate registry.
"""
    file_path = create_test_file(temp_dir, "gated_with_registry.md", content)
    violations = gate.scan_file(file_path)

    # Should NOT have MISSING_GATE_REGISTRY violation
    registry_violations = [
        v for v in violations
        if v.violation_type == "MISSING_GATE_REGISTRY"
    ]
    assert len(registry_violations) == 0


# =============================================================================
# TEST CASE 7: Unqualified "SHADOW MODE" usage
# =============================================================================

@pytest.mark.unit
def test_unqualified_shadow_mode_warned(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Unqualified 'SHADOW MODE' usage should trigger warning.

    Contract §1.1: Must specify SHADOW-OBSERVE or SHADOW-GATED.
    """
    content = """
# Unqualified Mode

This system operates in SHADOW MODE without further qualification.
"""
    file_path = create_test_file(temp_dir, "unqualified_shadow.md", content)
    violations = gate.scan_file(file_path)

    assert any(v.violation_type == "UNQUALIFIED_SHADOW_MODE" for v in violations)
    assert any(v.severity == "WARN" for v in violations)


# =============================================================================
# TEST CASE 8: Qualified SHADOW MODE passes
# =============================================================================

@pytest.mark.unit
def test_qualified_shadow_mode_passes(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Properly qualified 'SHADOW-OBSERVE' or 'SHADOW-GATED' should pass.
    """
    content = """
# Qualified Mode

This system operates in SHADOW-OBSERVE mode.

Another component uses SHADOW-GATED with gate_id: SRG-001.
"""
    file_path = create_test_file(temp_dir, "qualified_shadow.md", content)
    violations = gate.scan_file(file_path)

    # Should NOT have UNQUALIFIED_SHADOW_MODE violations
    unqualified = [v for v in violations if v.violation_type == "UNQUALIFIED_SHADOW_MODE"]
    assert len(unqualified) == 0


# =============================================================================
# TEST CASE 9: "fail-close" / "fail-safe" allowed in technical contexts
# =============================================================================

@pytest.mark.unit
def test_failclose_failsafe_allowed(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Technical phrases 'fail-close' and 'fail-safe' are allowed.

    Contract §3.7: "The operation MUST fail-safe (block the operation)"
    """
    content = """
# Failure Handling

**Mode**: SHADOW-GATED
**gate_id**: SRG-001

On error, the gate uses fail-close semantics.
For gated operations: The operation MUST fail-safe.
"""
    file_path = create_test_file(temp_dir, "failsafe_allowed.md", content)
    violations = gate.scan_file(file_path)

    # Should NOT flag fail-close or fail-safe as prohibited
    # (They are technical terms, not prohibited phrases)
    assert all("fail-close" not in v.message.lower() for v in violations)
    assert all("fail-safe" not in v.message.lower() for v in violations)


# =============================================================================
# TEST CASE 10: Multiple prohibited phrases
# =============================================================================

@pytest.mark.unit
def test_multiple_prohibited_phrases(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Multiple prohibited phrases should each be flagged.
    """
    content = """
# Multiple Violations

**Mode**: SHADOW-GATED
**gate_id**: SRG-001

This is observational only.
It provides advisory guidance.
The system is passive.
All checks are informational.
"""
    file_path = create_test_file(temp_dir, "multiple_prohibited.md", content)
    violations = gate.scan_file(file_path)

    # Should find at least 4 violations (one per prohibited phrase)
    prohibited = [v for v in violations if v.violation_type == "PROHIBITED_PHRASE_IN_GATED"]
    assert len(prohibited) >= 4


# =============================================================================
# TEST CASE 11: Clean document passes completely
# =============================================================================

@pytest.mark.unit
def test_clean_document_passes(gate: ShadowReleaseGate, temp_dir: Path):
    """
    A properly written SHADOW-GATED document should have no violations.
    """
    content = """
# Clean SHADOW-GATED Document

**Mode**: SHADOW-GATED
**Gate ID**: SRG-001
**Contract Reference**: SHADOW_MODE_CONTRACT.md v1.0.0

## Gate Definition

This gate blocks release of artifacts that violate SHADOW MODE semantics.

## Operations Gated

| Operation | Condition | Enforcement |
|-----------|-----------|-------------|
| release | Prohibited phrases detected | BLOCK |

## What Happens When Gate Triggers

When a violation is detected:
1. The CI pipeline fails
2. A detailed report is generated
3. The artifact is not released

This is not a simulation or dry-run. It enforces contract compliance.
"""
    file_path = create_test_file(temp_dir, "clean_gated.md", content)
    violations = gate.scan_file(file_path)

    # Filter to only ERROR severity violations
    errors = [v for v in violations if v.severity == "ERROR"]
    assert len(errors) == 0, f"Clean document has errors: {[v.to_dict() for v in errors]}"


# =============================================================================
# TEST CASE 12: Directory scan aggregates violations
# =============================================================================

@pytest.mark.unit
def test_directory_scan_aggregates(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Directory scan should aggregate violations from multiple files.
    """
    # Create clean file
    create_test_file(temp_dir, "clean.md", """
# Clean Document
**Mode**: SHADOW-OBSERVE
Observational only.
""")

    # Create file with violation
    create_test_file(temp_dir, "violation.md", """
# Bad Document
**Mode**: SHADOW-GATED
This is observational only.
gate_id: SRG-001
""")

    report = gate.scan_directory(temp_dir)

    assert report.files_scanned >= 2
    assert len(report.violations) >= 1
    assert not report.passed  # Should fail due to violation


# =============================================================================
# TEST CASE 13: Gate report structure validation
# =============================================================================

@pytest.mark.unit
def test_gate_report_structure(gate: ShadowReleaseGate, temp_dir: Path):
    """
    Gate report should conform to SHADOW MODE audit artifact schema.

    Contract §8.3: Audit artifacts must contain shadow_mode, gate_ids,
    verification_result, system_impact, timestamp, contract_version.
    """
    content = """
# Test Document
**Mode**: SHADOW-GATED
This is observational only.
"""
    create_test_file(temp_dir, "test.md", content)
    report = gate.scan_directory(temp_dir)

    report_dict = report.to_dict()

    # Verify required fields per §8.3
    assert "shadow_mode" in report_dict
    assert report_dict["shadow_mode"] in ["SHADOW-OBSERVE", "SHADOW-GATED"]
    assert "gate_id" in report_dict or "gate_ids" in report_dict
    assert "timestamp" in report_dict
    assert "contract_version" in report_dict
    assert "system_impact" in report_dict
    assert report_dict["system_impact"] in ["NONE", "BLOCKED", "ALLOWED"]


# =============================================================================
# TEST CASE 14: Reference to SHADOW_MODE_CONTRACT.md is not flagged
# =============================================================================

@pytest.mark.unit
def test_contract_reference_not_flagged(gate: ShadowReleaseGate, temp_dir: Path):
    """
    References to SHADOW_MODE_CONTRACT.md should not be flagged as unqualified.
    """
    content = """
# Contract Reference

See docs/system_law/SHADOW_MODE_CONTRACT.md for the authoritative definition.

The SHADOW MODE CONTRACT defines two sub-modes.
"""
    file_path = create_test_file(temp_dir, "contract_ref.md", content)
    violations = gate.scan_file(file_path)

    # Should NOT flag contract references as unqualified
    unqualified = [v for v in violations if v.violation_type == "UNQUALIFIED_SHADOW_MODE"]
    assert len(unqualified) == 0


# =============================================================================
# TEST CASE 15: Gate registry loading
# =============================================================================

@pytest.mark.unit
def test_gate_registry_loads():
    """
    Gate registry should load default entries even if file doesn't exist.
    """
    registry = load_gate_registry()

    assert len(registry) >= 1
    assert any(e.gate_id == "SRG-001" for e in registry)


# =============================================================================
# INTEGRATION TEST: Full scan of docs/system_law/
# =============================================================================

@pytest.mark.integration
def test_system_law_scan():
    """
    Integration test: scan actual docs/system_law/ directory.

    This test verifies the gate can run on real documentation.
    Note: May find violations if docs have issues — that's expected.
    """
    system_law_path = Path(__file__).parent.parent.parent / "docs" / "system_law"

    if not system_law_path.exists():
        pytest.skip("docs/system_law/ not found")

    gate = ShadowReleaseGate()
    report = gate.scan_directory(system_law_path)

    # Should complete without error
    assert report.files_scanned > 0

    # Report should be valid JSON
    json_output = report.to_json()
    parsed = json.loads(json_output)
    assert "violations" in parsed
    assert "passed" in parsed
