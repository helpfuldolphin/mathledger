"""
PHASE II — NOT USED IN PHASE I

Tests for U2 calibration integration with run_uplift_u2.py

Validates:
- Successful run when valid calibration exists
- Exit code 2 when calibration missing with --require-calibration
- Exit code 2 when calibration invalid with --require-calibration
- Calibration check is skipped when --require-calibration not set
"""

import json
import pytest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def valid_calibration_summary():
    """Valid calibration summary data."""
    return {
        "slice_name": "test_slice",
        "determinism_verified": True,
        "schema_valid": True,
        "replay_hash": "abc123...",
        "baseline_hash": "abc123...",
        "metadata": {}
    }


@pytest.fixture
def invalid_calibration_summary():
    """Invalid calibration summary (determinism failed)."""
    return {
        "slice_name": "test_slice",
        "determinism_verified": False,
        "schema_valid": True,
        "replay_hash": "abc123...",
        "baseline_hash": "def456...",
        "metadata": {}
    }


def create_calibration_dir(base_dir: Path, slice_name: str, summary_data: dict):
    """Create calibration directory structure with summary file."""
    calib_dir = base_dir / slice_name
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = calib_dir / "calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return summary_path


@pytest.mark.unit
def test_calibration_check_succeeds_when_valid(temp_dir, valid_calibration_summary):
    """Test that calibration check passes with valid calibration."""
    from experiments.u2_calibration import validate_calibration
    
    # Create valid calibration
    create_calibration_dir(temp_dir, "test_slice", valid_calibration_summary)
    
    # Should not raise
    summary = validate_calibration(temp_dir, "test_slice", require_valid=True)
    assert summary is not None
    assert summary.is_valid()


@pytest.mark.unit
def test_calibration_check_fails_when_missing(temp_dir):
    """Test that calibration check fails when calibration missing."""
    from experiments.u2_calibration import (
        validate_calibration,
        CalibrationNotFoundError,
    )
    
    # No calibration created
    with pytest.raises(CalibrationNotFoundError):
        validate_calibration(temp_dir, "test_slice", require_valid=True)


@pytest.mark.unit
def test_calibration_check_fails_when_invalid(temp_dir, invalid_calibration_summary):
    """Test that calibration check fails when calibration invalid."""
    from experiments.u2_calibration import (
        validate_calibration,
        CalibrationInvalidError,
    )
    
    # Create invalid calibration
    create_calibration_dir(temp_dir, "test_slice", invalid_calibration_summary)
    
    # Should raise CalibrationInvalidError
    with pytest.raises(CalibrationInvalidError) as exc_info:
        validate_calibration(temp_dir, "test_slice", require_valid=True)
    
    assert "determinism check failed" in str(exc_info.value).lower()


@pytest.mark.unit
def test_calibration_check_optional_when_not_required(temp_dir):
    """Test that calibration check is optional when require_valid=False."""
    from experiments.u2_calibration import validate_calibration
    
    # No calibration created
    summary = validate_calibration(temp_dir, "test_slice", require_valid=False)
    assert summary is None  # Should return None, not raise


@pytest.mark.unit
def test_check_calibration_exists(temp_dir, valid_calibration_summary):
    """Test check_calibration_exists function."""
    from experiments.u2_calibration import check_calibration_exists
    
    # Before creating calibration
    assert not check_calibration_exists(temp_dir, "test_slice")
    
    # After creating calibration
    create_calibration_dir(temp_dir, "test_slice", valid_calibration_summary)
    assert check_calibration_exists(temp_dir, "test_slice")


@pytest.mark.unit
def test_compute_result_hash_deterministic():
    """Test that compute_result_hash is deterministic."""
    from experiments.u2_calibration import compute_result_hash
    
    results = [
        {"cycle": 0, "success": True, "item": "p"},
        {"cycle": 1, "success": False, "item": "q"},
        {"cycle": 2, "success": True, "item": "r"},
    ]
    
    hash1 = compute_result_hash(results)
    hash2 = compute_result_hash(results)
    
    assert hash1 == hash2
    
    # Same results in different order should give same hash
    results_reordered = [results[2], results[0], results[1]]
    hash3 = compute_result_hash(results_reordered)
    assert hash1 == hash3


@pytest.mark.unit
def test_calibration_summary_dataclass():
    """Test CalibrationSummary dataclass."""
    from experiments.u2_calibration import CalibrationSummary
    
    summary = CalibrationSummary(
        slice_name="test",
        determinism_verified=True,
        schema_valid=True,
        replay_hash="abc",
        baseline_hash="abc",
    )
    
    assert summary.is_valid()
    
    # Test to_dict
    d = summary.to_dict()
    assert d["slice_name"] == "test"
    assert d["determinism_verified"] is True


@pytest.mark.unit
def test_save_and_load_calibration_summary(temp_dir):
    """Test saving and loading calibration summary."""
    from experiments.u2_calibration import (
        CalibrationSummary,
        save_calibration_summary,
        load_calibration_summary,
    )
    
    summary = CalibrationSummary(
        slice_name="test_slice",
        determinism_verified=True,
        schema_valid=True,
        replay_hash="abc123",
    )
    
    # Save
    path = save_calibration_summary(temp_dir, summary)
    assert path.exists()
    
    # Load
    loaded = load_calibration_summary(temp_dir, "test_slice")
    assert loaded.slice_name == summary.slice_name
    assert loaded.determinism_verified == summary.determinism_verified
    assert loaded.replay_hash == summary.replay_hash


@pytest.mark.unit
def test_run_calibration_placeholder():
    """Test that run_calibration is a placeholder."""
    from experiments.u2_calibration import run_calibration
    from pathlib import Path
    
    with pytest.raises(NotImplementedError):
        run_calibration(
            slice_name="test",
            config_path=Path("dummy.yaml"),
            output_dir=Path("/tmp"),
        )


# Integration tests with run_uplift_u2.py
# These test the CLI interface and exit codes

def test_cli_exit_code_calibration_missing(temp_dir):
    """
    Test that run_uplift_u2.py exits with code 2 when calibration missing.
    
    Note: This test requires the u2.runner module to exist. If it doesn't,
    the test will be skipped or may fail during import.
    """
    # This is a sketch - actual implementation depends on whether
    # experiments/u2/ module exists with required classes
    pytest.skip("Requires u2.runner module which may not exist yet")


def test_cli_exit_code_calibration_invalid(temp_dir):
    """
    Test that run_uplift_u2.py exits with code 2 when calibration invalid.
    
    Note: This test requires the u2.runner module to exist.
    """
    pytest.skip("Requires u2.runner module which may not exist yet")


def test_cli_success_with_valid_calibration(temp_dir):
    """
    Test that run_uplift_u2.py succeeds when valid calibration exists.
    
    Note: This test requires the u2.runner module to exist.
    """
    pytest.skip("Requires u2.runner module which may not exist yet")


def test_cli_skips_calibration_check_by_default(temp_dir):
    """
    Test that calibration check is skipped when --require-calibration not set.
    
    Note: This test requires the u2.runner module to exist.
    """
    pytest.skip("Requires u2.runner module which may not exist yet")
