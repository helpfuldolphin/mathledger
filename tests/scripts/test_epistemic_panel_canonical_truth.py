"""
Tests for epistemic panel canonical truth hardening.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify extraction provenance, top_reason_code selection, and GGFL invariants
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from scripts.generate_first_light_status import generate_status
from backend.health.epistemic_p3p4_integration import epistemic_panel_for_alignment_view


@pytest.fixture
def evidence_pack_dir(tmp_path: Path) -> Path:
    """Create a minimal evidence pack directory structure."""
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()
    
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {},
    }
    
    manifest_path = pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    return pack_dir


@pytest.fixture
def p3_dir(tmp_path: Path) -> Path:
    """Create a minimal P3 directory structure."""
    p3_dir = tmp_path / "p3"
    p3_dir.mkdir()
    p3_run = p3_dir / "fl_test"
    p3_run.mkdir()
    (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
    (p3_run / "stability_report.json").write_text(
        json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
    )
    (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
    (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
    (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
    return p3_dir


@pytest.fixture
def p4_dir(tmp_path: Path) -> Path:
    """Create a minimal P4 directory structure."""
    p4_dir = tmp_path / "p4"
    p4_dir.mkdir()
    p4_run = p4_dir / "p4_test"
    p4_run.mkdir()
    (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
    (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
    (p4_run / "calibration_report.json").write_text("{}", encoding="utf-8")
    (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
    (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
    (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    return p4_dir


def test_extraction_source_manifest(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that extraction_source is MANIFEST when panel comes from manifest."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 2,
                "num_consistent": 1,
                "num_inconsistent": 1,
                "num_unknown": 0,
                "experiments_inconsistent": [],
                "reason_code_histogram": {},
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["extraction_source"] == "MANIFEST"
    assert panel_signal["panel_schema_version"] == "1.0.0"
    
    # No fallback warning
    warnings = status.get("warnings", [])
    fallback_warnings = [w for w in warnings if "fallback" in w.lower()]
    assert len(fallback_warnings) == 0


def test_extraction_source_evidence_json_with_fallback_warning(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that extraction_source is EVIDENCE_JSON and fallback warning is generated."""
    # Manifest without epistemic panel
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {},
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    # Evidence.json with epistemic panel
    evidence = {
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 2,
                "num_consistent": 1,
                "num_inconsistent": 1,
                "num_unknown": 0,
                "experiments_inconsistent": [],
                "reason_code_histogram": {},
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["extraction_source"] == "EVIDENCE_JSON"
    assert panel_signal["panel_schema_version"] == "1.0.0"
    
    # Exactly one fallback warning
    warnings = status.get("warnings", [])
    fallback_warnings = [w for w in warnings if "fallback" in w.lower() and "manifest missing" in w.lower()]
    assert len(fallback_warnings) == 1
    assert "advisory only" in fallback_warnings[0].lower()


def test_top_reason_code_deterministic_tie_breaking(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that top_reason_code selection is deterministic with tie-breaking."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 4,
                "num_consistent": 0,
                "num_inconsistent": 4,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_STABLE",  # Count 2
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_STABLE",  # Count 2
                    },
                    {
                        "cal_id": "CAL-EXP-3",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",  # Count 2 (tie)
                    },
                    {
                        "cal_id": "CAL-EXP-4",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",  # Count 2 (tie)
                    },
                ],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    top_reason_code = panel_signal["top_reason_code"]
    
    # With tie (both have count 2), should select alphabetically first
    # EPI_DEGRADED_EVID_IMPROVING < EPI_DEGRADED_EVID_STABLE (alphabetically)
    assert top_reason_code == "EPI_DEGRADED_EVID_IMPROVING"
    
    # Verify deterministic (run again)
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    panel_signal2 = status2["signals"]["epistemic_panel"]
    assert panel_signal2["top_reason_code"] == top_reason_code


def test_top3_truncation_and_ordering(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that top_inconsistent_cal_ids_top3 is truncated to 3 and sorted."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 5,
                "num_consistent": 0,
                "num_inconsistent": 5,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-5",  # Out of order to test sorting
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-3",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-4",
                        "reason": "Test",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                ],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    top3 = panel_signal["top_inconsistent_cal_ids_top3"]
    
    # Should be truncated to 3
    assert len(top3) == 3
    
    # Should be sorted
    assert top3 == sorted(top3)
    
    # Should be first 3 alphabetically
    assert top3 == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]


def test_ggfl_reason_code_drivers(
) -> None:
    """Test that GGFL adapter uses reason-code drivers."""
    signal = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "extraction_source": "MANIFEST",
        "num_experiments": 3,
        "num_consistent": 1,
        "num_inconsistent": 2,
        "num_unknown": 0,
        "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2"],
        "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
        "reason_code_histogram": {
            "EPI_DEGRADED_EVID_IMPROVING": 2,
        },
    }
    
    result = epistemic_panel_for_alignment_view(signal)
    
    # Check drivers are reason-code only
    drivers = result["drivers"]
    assert len(drivers) <= 3
    
    # Verify reason-code driver format
    assert "DRIVER_EPI_INCONSISTENT_PRESENT" in drivers
    assert "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING" in drivers
    assert "DRIVER_TOP_CAL_IDS_PRESENT" in drivers
    
    # Verify deterministic ordering
    assert drivers[0] == "DRIVER_EPI_INCONSISTENT_PRESENT"
    assert drivers[1] == "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING"
    assert drivers[2] == "DRIVER_TOP_CAL_IDS_PRESENT"


def test_ggfl_shadow_mode_invariants(
) -> None:
    """Test that GGFL adapter includes shadow_mode_invariants with fixed values."""
    signal = {
        "num_experiments": 3,
        "num_consistent": 3,
        "num_inconsistent": 0,
        "num_unknown": 0,
        "top_inconsistent_cal_ids_top3": [],
        "top_reason_code": "UNKNOWN",
        "reason_code_histogram": {},
    }
    
    result = epistemic_panel_for_alignment_view(signal)
    
    # Check invariants are present
    assert "shadow_mode_invariants" in result
    invariants = result["shadow_mode_invariants"]
    
    # Check fixed values
    assert invariants["advisory_only"] is True
    assert invariants["no_enforcement"] is True
    assert invariants["conflict_invariant"] is True
    
    # Check other invariants
    assert result["conflict"] is False
    assert result["weight_hint"] == "LOW"


def test_ggfl_deterministic_byte_identical(
) -> None:
    """Test that GGFL adapter produces byte-identical output for identical inputs."""
    signal = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "num_experiments": 3,
        "num_consistent": 1,
        "num_inconsistent": 2,
        "num_unknown": 0,
        "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2"],
        "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
        "reason_code_histogram": {
            "EPI_DEGRADED_EVID_IMPROVING": 2,
        },
    }
    
    result1 = epistemic_panel_for_alignment_view(signal)
    result2 = epistemic_panel_for_alignment_view(signal)
    
    # Byte-identical JSON serialization
    json1 = json.dumps(result1, sort_keys=True)
    json2 = json.dumps(result2, sort_keys=True)
    
    assert json1 == json2
    assert result1 == result2

