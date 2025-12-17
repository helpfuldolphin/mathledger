"""
Tests for evidence failure shortlist extraction in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings (capped to 1 line)
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from scripts.generate_first_light_status import generate_status


@pytest.fixture
def evidence_pack_dir(tmp_path: Path) -> Path:
    """Create a minimal evidence pack directory structure."""
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()
    
    # Create minimal manifest.json (required for check_evidence_pack)
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "files": [
            {
                "path": "manifest.json",
                "sha256": "abc123",
            }
        ],
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
    return p4_dir


def test_evidence_failure_shortlist_extracted_from_evidence(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that evidence failure shortlist is extracted from evidence.json when present."""
    # Create evidence.json with failure shortlist
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "cal_exp1_episode_1",
                        "trajectory_class": "DEGRADING",
                        "predicted_band": "LOW",
                        "cycles_until_risk": 0,
                        "flags": ["regression detected"],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                    },
                    {
                        "rank": 2,
                        "cal_id": "cal_exp2",
                        "episode_id": "cal_exp2_episode_1",
                        "trajectory_class": "STABLE",
                        "predicted_band": "MEDIUM",
                        "cycles_until_risk": 2,
                        "flags": ["attention points"],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp2.json",
                    },
                ],
                "total_shelves": 2,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "evidence_failure_shortlist" in status["signals"]
    
    shortlist_signal = status["signals"]["evidence_failure_shortlist"]
    assert shortlist_signal["extraction_source"] == "EVIDENCE_JSON"
    assert shortlist_signal["total_items"] == 2
    assert len(shortlist_signal["top5"]) == 2
    
    # Check top5 structure
    top5 = shortlist_signal["top5"]
    assert top5[0]["cal_id"] == "cal_exp1"
    assert top5[0]["episode_id"] == "cal_exp1_episode_1"
    assert top5[0]["predicted_band"] == "LOW"
    assert top5[0]["cycles_until_risk"] == 0
    assert top5[1]["cal_id"] == "cal_exp2"
    assert top5[1]["predicted_band"] == "MEDIUM"


def test_evidence_failure_shortlist_top5_truncation(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that top5 is truncated to 5 items even if shortlist has more."""
    # Create evidence.json with 10 items
    items = []
    for i in range(10):
        items.append({
            "rank": i + 1,
            "cal_id": f"cal_exp{i+1}",
            "episode_id": f"cal_exp{i+1}_episode_1",
            "trajectory_class": "DEGRADING",
            "predicted_band": "LOW",
            "cycles_until_risk": i,
            "flags": [],
            "evidence_path_hint": f"calibration/evidence_failure_shelf_cal_exp{i+1}.json",
        })
    
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": items,
                "total_shelves": 10,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    shortlist_signal = status["signals"]["evidence_failure_shortlist"]
    assert shortlist_signal["extraction_source"] == "EVIDENCE_JSON"
    assert shortlist_signal["total_items"] == 10
    assert len(shortlist_signal["top5"]) == 5


def test_evidence_failure_shortlist_missing_not_error(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that missing evidence failure shortlist does not cause an error."""
    # Create evidence.json without failure shortlist
    evidence = {
        "governance": {},
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Should not have evidence_failure_shortlist signal
    assert "signals" in status
    assert "evidence_failure_shortlist" not in status["signals"]


def test_evidence_failure_shortlist_warning_high_band(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that warning is generated when HIGH predicted_band items in top5."""
    # Create evidence.json with HIGH band item (unexpected in failure shortlist)
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "cal_exp1_episode_1",
                        "trajectory_class": "IMPROVING",
                        "predicted_band": "HIGH",
                        "cycles_until_risk": 5,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                    },
                ],
                "total_shelves": 1,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Should have warning about HIGH band
    assert "warnings" in status
    warnings = status["warnings"]
    high_band_warnings = [w for w in warnings if "HIGH" in w and ("evidence failure shortlist" in w.lower() or "Evidence failure shortlist" in w)]
    assert len(high_band_warnings) == 1
    assert "cal_exp1" in high_band_warnings[0]
    assert "high_band_count_in_top5" in high_band_warnings[0] or "1 item(s)" in high_band_warnings[0]


def test_evidence_failure_shortlist_warning_capped_to_one_line(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that warnings are capped to 1 line total even if multiple HIGH items."""
    # Create evidence.json with multiple HIGH band items
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "cal_exp1_episode_1",
                        "trajectory_class": "IMPROVING",
                        "predicted_band": "HIGH",
                        "cycles_until_risk": 5,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                    },
                    {
                        "rank": 2,
                        "cal_id": "cal_exp2",
                        "episode_id": "cal_exp2_episode_1",
                        "trajectory_class": "IMPROVING",
                        "predicted_band": "HIGH",
                        "cycles_until_risk": 6,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp2.json",
                    },
                    {
                        "rank": 3,
                        "cal_id": "cal_exp3",
                        "episode_id": "cal_exp3_episode_1",
                        "trajectory_class": "IMPROVING",
                        "predicted_band": "HIGH",
                        "cycles_until_risk": 7,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp3.json",
                    },
                ],
                "total_shelves": 3,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Should have exactly 1 warning (capped)
    assert "warnings" in status
    warnings = status["warnings"]
    high_band_warnings = [w for w in warnings if "HIGH" in w and ("evidence failure shortlist" in w.lower() or "Evidence failure shortlist" in w)]
    assert len(high_band_warnings) == 1, f"Expected 1 warning, got {len(high_band_warnings)}: {high_band_warnings}"
    # Check that warning includes high_band_count_in_top5 and top_cal_ids
    warning_text = high_band_warnings[0]
    assert "3 item(s)" in warning_text or "high_band_count_in_top5" in warning_text
    assert "cal_exp1" in warning_text and "cal_exp2" in warning_text and "cal_exp3" in warning_text


def test_evidence_failure_shortlist_no_warning_when_all_low_medium(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that no warning is generated when all items have LOW/MEDIUM bands."""
    # Create evidence.json with LOW/MEDIUM band items only
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "cal_exp1_episode_1",
                        "trajectory_class": "DEGRADING",
                        "predicted_band": "LOW",
                        "cycles_until_risk": 0,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                    },
                    {
                        "rank": 2,
                        "cal_id": "cal_exp2",
                        "episode_id": "cal_exp2_episode_1",
                        "trajectory_class": "STABLE",
                        "predicted_band": "MEDIUM",
                        "cycles_until_risk": 2,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp2.json",
                    },
                ],
                "total_shelves": 2,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Should not have HIGH band warnings
    assert "warnings" in status
    warnings = status["warnings"]
    high_band_warnings = [w for w in warnings if "HIGH" in w and ("evidence failure shortlist" in w.lower() or "Evidence failure shortlist" in w)]
    assert len(high_band_warnings) == 0


def test_evidence_failure_shortlist_deterministic(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that evidence failure shortlist extraction is deterministic."""
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "cal_exp1_episode_1",
                        "trajectory_class": "DEGRADING",
                        "predicted_band": "LOW",
                        "cycles_until_risk": 0,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                    },
                ],
                "total_shelves": 1,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Signals should be identical
    signal1 = status1.get("signals", {}).get("evidence_failure_shortlist")
    signal2 = status2.get("signals", {}).get("evidence_failure_shortlist")
    
    assert signal1 == signal2
    assert signal1["extraction_source"] == "EVIDENCE_JSON"
    
    # JSON round-trip should be identical
    json1 = json.dumps(signal1, sort_keys=True) if signal1 else None
    json2 = json.dumps(signal2, sort_keys=True) if signal2 else None
    assert json1 == json2


def test_evidence_failure_shortlist_manifest_first_preference(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that manifest is preferred over evidence.json for shortlist extraction."""
    # Create manifest.json with shortlist
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    manifest["governance"] = {
        "evidence_failure_shortlist": {
            "schema_version": "1.0.0",
            "items": [
                {
                    "rank": 1,
                    "cal_id": "cal_exp1",
                    "episode_id": "cal_exp1_episode_1",
                    "trajectory_class": "DEGRADING",
                    "predicted_band": "LOW",
                    "cycles_until_risk": 0,
                    "flags": [],
                    "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp1.json",
                },
            ],
            "total_shelves": 1,
        },
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    # Create evidence.json with different shortlist (should be ignored)
    evidence = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp2",  # Different cal_id
                        "episode_id": "cal_exp2_episode_1",
                        "trajectory_class": "STABLE",
                        "predicted_band": "MEDIUM",
                        "cycles_until_risk": 2,
                        "flags": [],
                        "evidence_path_hint": "calibration/evidence_failure_shelf_cal_exp2.json",
                    },
                ],
                "total_shelves": 1,
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Should use manifest (cal_exp1), not evidence.json (cal_exp2)
    shortlist_signal = status["signals"]["evidence_failure_shortlist"]
    assert shortlist_signal["extraction_source"] == "MANIFEST"
    assert shortlist_signal["top5"][0]["cal_id"] == "cal_exp1"


def test_evidence_failure_shortlist_ggfl_adapter_deterministic() -> None:
    """Test 236: GGFL adapter is deterministic."""
    from backend.health.evidence_quality_adapter import (
        evidence_failure_shortlist_for_alignment_view,
    )
    import json
    
    signal = {
        "extraction_source": "MANIFEST",
        "total_items": 2,
        "top5": [
            {"cal_id": "cal_exp1", "episode_id": "ep1", "predicted_band": "HIGH", "cycles_until_risk": 5},
            {"cal_id": "cal_exp2", "episode_id": "ep2", "predicted_band": "LOW", "cycles_until_risk": 0},
        ],
    }
    
    ggfl1 = evidence_failure_shortlist_for_alignment_view(signal)
    ggfl2 = evidence_failure_shortlist_for_alignment_view(signal)
    
    json1 = json.dumps(ggfl1, sort_keys=True)
    json2 = json.dumps(ggfl2, sort_keys=True)
    
    assert json1 == json2, "GGFL adapter should be deterministic"
    assert ggfl1["signal_type"] == "SIG-EVID"
    assert ggfl1["status"] == "warn"  # HIGH in top5
    assert ggfl1["conflict"] is False
    assert ggfl1["weight_hint"] == "LOW"
    assert "DRIVER_HIGH_BAND_PRESENT" in ggfl1["drivers"]


def test_evidence_failure_shortlist_ggfl_adapter_ok_status() -> None:
    """Test 237: GGFL adapter returns ok status when no HIGH in top5."""
    from backend.health.evidence_quality_adapter import (
        evidence_failure_shortlist_for_alignment_view,
    )
    
    signal = {
        "extraction_source": "MANIFEST",
        "total_items": 2,
        "top5": [
            {"cal_id": "cal_exp1", "episode_id": "ep1", "predicted_band": "LOW", "cycles_until_risk": 0},
            {"cal_id": "cal_exp2", "episode_id": "ep2", "predicted_band": "MEDIUM", "cycles_until_risk": 2},
        ],
    }
    
    ggfl = evidence_failure_shortlist_for_alignment_view(signal)
    
    assert ggfl["signal_type"] == "SIG-EVID"
    assert ggfl["status"] == "ok"
    assert ggfl["conflict"] is False
    assert ggfl["weight_hint"] == "LOW"
    assert len(ggfl["drivers"]) == 0
    assert "no unexpected HIGH" in ggfl["summary"]


def test_evidence_failure_shortlist_ggfl_adapter_empty_signal() -> None:
    """Test 238: GGFL adapter handles None/empty signal."""
    from backend.health.evidence_quality_adapter import (
        evidence_failure_shortlist_for_alignment_view,
    )
    
    ggfl = evidence_failure_shortlist_for_alignment_view(None)
    
    assert ggfl["signal_type"] == "SIG-EVID"
    assert ggfl["status"] == "ok"
    assert ggfl["conflict"] is False
    assert ggfl["weight_hint"] == "LOW"
    assert len(ggfl["drivers"]) == 0
    assert "not available" in ggfl["summary"]


def test_evidence_failure_shortlist_extraction_source_enum_normalization() -> None:
    """Test 239: extraction_source enum values are normalized to uppercase."""
    from backend.health.evidence_quality_adapter import (
        extract_evidence_failure_shortlist_signal_for_status,
    )
    
    # Test MANIFEST source
    manifest_data = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp1",
                        "episode_id": "ep1",
                        "predicted_band": "LOW",
                        "cycles_until_risk": 0,
                    },
                ],
            },
        },
    }
    
    signal_manifest = extract_evidence_failure_shortlist_signal_for_status(
        pack_manifest=manifest_data,
        evidence_data=None,
    )
    assert signal_manifest is not None
    assert signal_manifest["extraction_source"] == "MANIFEST"
    
    # Test EVIDENCE_JSON source
    evidence_data = {
        "governance": {
            "evidence_failure_shortlist": {
                "schema_version": "1.0.0",
                "items": [
                    {
                        "rank": 1,
                        "cal_id": "cal_exp2",
                        "episode_id": "ep2",
                        "predicted_band": "MEDIUM",
                        "cycles_until_risk": 2,
                    },
                ],
            },
        },
    }
    
    signal_evidence = extract_evidence_failure_shortlist_signal_for_status(
        pack_manifest=None,
        evidence_data=evidence_data,
    )
    assert signal_evidence is not None
    assert signal_evidence["extraction_source"] == "EVIDENCE_JSON"
    
    # Test MISSING (no data)
    signal_missing = extract_evidence_failure_shortlist_signal_for_status(
        pack_manifest=None,
        evidence_data=None,
    )
    assert signal_missing is None  # Returns None when missing, not "MISSING"
    
    # Verify enum values are uppercase only
    valid_enum_values = {"MANIFEST", "EVIDENCE_JSON", "MISSING"}
    assert signal_manifest["extraction_source"] in valid_enum_values
    assert signal_evidence["extraction_source"] in valid_enum_values


def test_evidence_failure_shortlist_ggfl_adapter_invariants_present() -> None:
    """Test 240: GGFL adapter output includes shadow_mode_invariants block."""
    from backend.health.evidence_quality_adapter import (
        evidence_failure_shortlist_for_alignment_view,
    )
    
    signal = {
        "extraction_source": "MANIFEST",
        "total_items": 2,
        "top5": [
            {"cal_id": "cal_exp1", "episode_id": "ep1", "predicted_band": "HIGH", "cycles_until_risk": 5},
        ],
    }
    
    ggfl = evidence_failure_shortlist_for_alignment_view(signal)
    
    # Verify shadow_mode_invariants block is present
    assert "shadow_mode_invariants" in ggfl
    invariants = ggfl["shadow_mode_invariants"]
    
    # Verify all 3 required booleans are present
    assert "all_divergence_logged_only" in invariants
    assert "no_governance_modification" in invariants
    assert "no_abort_enforcement" in invariants
    
    # Verify all invariants are True (SHADOW MODE contract)
    assert invariants["all_divergence_logged_only"] is True
    assert invariants["no_governance_modification"] is True
    assert invariants["no_abort_enforcement"] is True
    
    # Test with None signal (should also have invariants)
    ggfl_empty = evidence_failure_shortlist_for_alignment_view(None)
    assert "shadow_mode_invariants" in ggfl_empty
    invariants_empty = ggfl_empty["shadow_mode_invariants"]
    assert invariants_empty["all_divergence_logged_only"] is True
    assert invariants_empty["no_governance_modification"] is True
    assert invariants_empty["no_abort_enforcement"] is True

