# REAL-READY
"""
Test Suite for Curriculum Drift Enforcement

Tests drift detection, governance signal emission (BLOCK/WARN), drift report generation,
and RunLedgerEntry stamping.

Author: MANUS-E, Curriculum Integrity Engineer
Status: OPERATIONAL
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.frontier.curriculum import (
    CurriculumSystem,
    CurriculumSlice,
    SliceGates,
    CoverageGateSpec,
    AbstentionGateSpec,
    VelocityGateSpec,
    CapsGateSpec,
    CurriculumDriftSentinel,
    CurriculumDriftError
)
from backend.frontier.curriculum_drift_enforcement import (
    DriftReport,
    emit_drift_signal,
    check_curriculum_drift,
    stamp_run_ledger_with_fingerprint
)
from rfl.curriculum_provenance import (
    initialize_curriculum_baseline,
    check_drift_before_run,
    extend_run_ledger_entry
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_curriculum_system():
    """Create a sample curriculum system for testing."""
    gates = SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.85, sample_min=30),
        abstention=AbstentionGateSpec(max_rate_pct=20.0),
        velocity=VelocityGateSpec(min_pph=50.0, max_cv=0.30),
        caps=CapsGateSpec(min_attempts=100, min_runtime_minutes=10.0)
    )
    
    slice1 = CurriculumSlice(
        name="slice_easy",
        params={"atoms": 3, "depth_max": 6},
        gates=gates,
        metadata={"wave": 1}
    )
    
    slice2 = CurriculumSlice(
        name="slice_medium",
        params={"atoms": 5, "depth_max": 8},
        gates=gates,
        metadata={"wave": 2}
    )
    
    system = CurriculumSystem(
        slug="pl",
        description="Test curriculum",
        slices=[slice1, slice2],
        active_index=0,
        invariants={"monotonic_axes": ["atoms", "depth_max"]},
        monotonic_axes=("atoms", "depth_max"),
        version=2
    )
    
    return system


@pytest.fixture
def temp_artifact_dir():
    """Create a temporary directory for artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test 1: Drift Sentinel Detects ContentDrift
# ============================================================================

def test_drift_sentinel_detects_content_drift(sample_curriculum_system):
    """Test that CurriculumDriftSentinel detects content drift (fingerprint mismatch)."""
    baseline_fingerprint = sample_curriculum_system.fingerprint()
    
    sentinel = CurriculumDriftSentinel(
        baseline_fingerprint=baseline_fingerprint,
        baseline_version=sample_curriculum_system.version,
        baseline_slice_count=len(sample_curriculum_system.slices)
    )
    
    # Modify curriculum (change parameter)
    sample_curriculum_system.slices[0].params["atoms"] = 4
    
    violations = sentinel.check(sample_curriculum_system)
    
    assert len(violations) > 0
    assert any("ContentDrift" in v or "Fingerprint mismatch" in v for v in violations)


# ============================================================================
# Test 2: Drift Sentinel Detects SchemaDrift
# ============================================================================

def test_drift_sentinel_detects_schema_drift(sample_curriculum_system):
    """Test that CurriculumDriftSentinel detects schema drift (version change)."""
    baseline_fingerprint = sample_curriculum_system.fingerprint()
    
    sentinel = CurriculumDriftSentinel(
        baseline_fingerprint=baseline_fingerprint,
        baseline_version=2,
        baseline_slice_count=len(sample_curriculum_system.slices)
    )
    
    # Change version
    sample_curriculum_system.version = 3
    
    violations = sentinel.check(sample_curriculum_system)
    
    assert len(violations) > 0
    assert any("SchemaDrift" in v or "Version changed" in v for v in violations)


# ============================================================================
# Test 3: Drift Sentinel Detects SliceCountDrift
# ============================================================================

def test_drift_sentinel_detects_slice_count_drift(sample_curriculum_system):
    """Test that CurriculumDriftSentinel detects slice count drift."""
    baseline_fingerprint = sample_curriculum_system.fingerprint()
    
    sentinel = CurriculumDriftSentinel(
        baseline_fingerprint=baseline_fingerprint,
        baseline_version=sample_curriculum_system.version,
        baseline_slice_count=2
    )
    
    # Add a slice
    gates = SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.90, sample_min=40),
        abstention=AbstentionGateSpec(max_rate_pct=15.0),
        velocity=VelocityGateSpec(min_pph=60.0, max_cv=0.25),
        caps=CapsGateSpec(min_attempts=150, min_runtime_minutes=15.0)
    )
    
    new_slice = CurriculumSlice(
        name="slice_hard",
        params={"atoms": 7, "depth_max": 10},
        gates=gates,
        metadata={"wave": 3}
    )
    
    sample_curriculum_system.slices.append(new_slice)
    
    violations = sentinel.check(sample_curriculum_system)
    
    assert len(violations) > 0
    assert any("SliceCountDrift" in v or "Slice count changed" in v for v in violations)


# ============================================================================
# Test 4: emit_drift_signal BLOCK Mode
# ============================================================================

def test_emit_drift_signal_block_mode(temp_artifact_dir):
    """Test that emit_drift_signal raises CurriculumDriftError in BLOCK mode."""
    violations = ["ContentDrift: Fingerprint mismatch"]
    
    with pytest.raises(CurriculumDriftError) as exc_info:
        emit_drift_signal(
            signal="BLOCK",
            violations=violations,
            curriculum_slug="pl",
            expected_fingerprint="abc123",
            observed_fingerprint="def456",
            artifact_dir=temp_artifact_dir
        )
    
    assert "ContentDrift" in str(exc_info.value)
    
    # Verify drift_report.json was created
    report_path = temp_artifact_dir / "drift_report.json"
    assert report_path.exists()
    
    with open(report_path) as f:
        report = json.load(f)
    
    assert report["governance_signal"] == "BLOCK"
    assert report["curriculum_slug"] == "pl"
    assert report["expected_fingerprint"] == "abc123"
    assert report["observed_fingerprint"] == "def456"
    assert len(report["violations"]) == 1


# ============================================================================
# Test 5: emit_drift_signal WARN Mode
# ============================================================================

def test_emit_drift_signal_warn_mode(temp_artifact_dir, capsys):
    """Test that emit_drift_signal logs warning in WARN mode without raising."""
    violations = ["ContentDrift: Fingerprint mismatch"]
    
    # Should not raise
    emit_drift_signal(
        signal="WARN",
        violations=violations,
        curriculum_slug="pl",
        expected_fingerprint="abc123",
        observed_fingerprint="def456",
        artifact_dir=temp_artifact_dir
    )
    
    # Verify warning was logged to stderr
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "ContentDrift" in captured.err
    
    # Verify drift_report.json was created
    report_path = temp_artifact_dir / "drift_report.json"
    assert report_path.exists()
    
    with open(report_path) as f:
        report = json.load(f)
    
    assert report["governance_signal"] == "WARN"


# ============================================================================
# Test 6: stamp_run_ledger_with_fingerprint
# ============================================================================

@patch('backend.frontier.curriculum_drift_enforcement.load_curriculum')
def test_stamp_run_ledger_with_fingerprint(mock_load, sample_curriculum_system):
    """Test that stamp_run_ledger_with_fingerprint adds provenance fields."""
    mock_load.return_value = sample_curriculum_system
    
    ledger_entry = {
        "run_id": "test_run_001",
        "slice_name": "slice_easy",
        "status": "completed"
    }
    
    stamped_entry = stamp_run_ledger_with_fingerprint(ledger_entry, "pl")
    
    assert "curriculum_slug" in stamped_entry
    assert stamped_entry["curriculum_slug"] == "pl"
    
    assert "curriculum_fingerprint" in stamped_entry
    assert len(stamped_entry["curriculum_fingerprint"]) == 64  # SHA-256 hex


# ============================================================================
# Integration Test: End-to-End Drift Detection
# ============================================================================

@patch('backend.frontier.curriculum_drift_enforcement.load_curriculum')
def test_end_to_end_drift_detection_block_mode(mock_load, sample_curriculum_system, temp_artifact_dir):
    """Integration test: Full drift detection flow in BLOCK mode."""
    # Setup: baseline curriculum
    baseline_fingerprint = sample_curriculum_system.fingerprint()
    
    # Modify curriculum
    sample_curriculum_system.slices[0].params["atoms"] = 10
    
    mock_load.return_value = sample_curriculum_system
    
    # Run drift check
    with pytest.raises(CurriculumDriftError):
        check_curriculum_drift(
            curriculum_slug="pl",
            baseline_fingerprint=baseline_fingerprint,
            artifact_dir=temp_artifact_dir,
            signal_mode="BLOCK"
        )
    
    # Verify drift report exists
    report_path = temp_artifact_dir / "drift_report.json"
    assert report_path.exists()
    
    with open(report_path) as f:
        report = json.load(f)
    
    assert report["governance_signal"] == "BLOCK"
    assert len(report["violations"]) > 0


# ============================================================================
# Provenance Integration Tests
# ============================================================================

@patch('rfl.curriculum_provenance.load_curriculum')
def test_initialize_curriculum_baseline(mock_load, sample_curriculum_system):
    """Test initialize_curriculum_baseline returns correct baseline info."""
    mock_load.return_value = sample_curriculum_system
    
    baseline = initialize_curriculum_baseline("pl")
    
    assert baseline["curriculum_slug"] == "pl"
    assert "baseline_fingerprint" in baseline
    assert baseline["baseline_version"] == 2
    assert baseline["baseline_slice_count"] == 2


@patch('rfl.curriculum_provenance.load_curriculum')
def test_extend_run_ledger_entry(mock_load, sample_curriculum_system):
    """Test extend_run_ledger_entry adds provenance fields."""
    mock_load.return_value = sample_curriculum_system
    
    ledger_entry = {
        "run_id": "test_run_002",
        "slice_name": "slice_medium"
    }
    
    extended = extend_run_ledger_entry(ledger_entry, "pl")
    
    assert "curriculum_slug" in extended
    assert "curriculum_fingerprint" in extended
    assert extended["curriculum_slug"] == "pl"
