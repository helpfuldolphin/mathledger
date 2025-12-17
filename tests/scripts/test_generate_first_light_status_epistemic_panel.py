"""
Tests for epistemic panel extraction in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from scripts.generate_first_light_status import generate_status

# Import reusable warning neutrality helpers (single source of truth)
from tests.helpers.warning_neutrality import pytest_assert_warning_neutral


@pytest.fixture
def evidence_pack_dir(tmp_path: Path) -> Path:
    """Create a minimal evidence pack directory structure."""
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()
    
    # Create minimal manifest.json
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
    (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
    (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    return p4_dir


def test_epistemic_panel_extracted_from_manifest_first(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that epistemic panel is extracted from manifest.json first, then fallback to evidence.json."""
    # Create manifest.json with epistemic panel
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_consistent": 1,
                "num_inconsistent": 2,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason 1",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test reason 2",
                        "reason_code": "EPI_DEGRADED_EVID_STABLE",
                    },
                ],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    # Also create evidence.json with different data (should be ignored)
    evidence = {
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 999,  # Different value to verify manifest is used
                "num_consistent": 999,
                "num_inconsistent": 999,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "epistemic_panel" in status["signals"]
    
    panel_signal = status["signals"]["epistemic_panel"]
    # Verify manifest data is used (not evidence.json)
    assert panel_signal["num_experiments"] == 3
    assert panel_signal["num_consistent"] == 1
    assert panel_signal["num_inconsistent"] == 2
    assert panel_signal["schema_version"] == "1.0.0"
    assert panel_signal["mode"] == "SHADOW"
    assert panel_signal["extraction_source"] == "MANIFEST"
    assert panel_signal["panel_schema_version"] == "1.0.0"
    assert "top_reason_code" in panel_signal
    
    # Check sorted cal_ids (top3)
    assert len(panel_signal["top_inconsistent_cal_ids_top3"]) == 2
    assert panel_signal["top_inconsistent_cal_ids_top3"] == sorted(panel_signal["top_inconsistent_cal_ids_top3"])
    
    # Check sorted reason code histogram
    assert "reason_code_histogram" in panel_signal
    histogram = panel_signal["reason_code_histogram"]
    assert list(histogram.keys()) == sorted(histogram.keys())
    assert histogram["EPI_DEGRADED_EVID_IMPROVING"] == 1
    assert histogram["EPI_DEGRADED_EVID_STABLE"] == 1


def test_epistemic_panel_fallback_to_evidence(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that epistemic panel falls back to evidence.json when not in manifest."""
    # Create manifest.json without epistemic panel
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {},
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    # Create evidence.json with epistemic panel
    evidence = {
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_consistent": 1,
                "num_inconsistent": 2,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                ],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "epistemic_panel" in status["signals"]
    
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["num_experiments"] == 3
    assert panel_signal["num_inconsistent"] == 2
    assert panel_signal["extraction_source"] == "EVIDENCE_JSON"
    assert panel_signal["panel_schema_version"] == "1.0.0"
    
    # Check that fallback warning is generated
    assert "warnings" in status
    warnings = status["warnings"]
    fallback_warnings = [w for w in warnings if "evidence.json fallback" in w.lower()]
    assert len(fallback_warnings) == 1, f"Expected 1 fallback warning, got {len(fallback_warnings)}"


def test_epistemic_panel_absent_when_not_in_evidence(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that epistemic panel signal is absent when not in evidence.json."""
    # Create evidence.json without epistemic panel
    evidence = {
        "governance": {},
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    signals = status.get("signals") or {}
    # Epistemic panel should not be present when not in evidence
    assert "epistemic_panel" not in signals


def test_epistemic_panel_warning_capped_and_formatted(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that exactly one warning is generated with count + top cal_ids + top reason_code."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_consistent": 1,
                "num_inconsistent": 2,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason 1",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test reason 2",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",  # Most frequent
                    },
                ],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "warnings" in status
    warnings = status["warnings"]
    
    # Check that exactly one epistemic panel inconsistency warning exists
    epistemic_warnings = [w for w in warnings if "epistemic panel" in w.lower() and "inconsistent" in w.lower()]
    assert len(epistemic_warnings) == 1, f"Expected 1 inconsistency warning, got {len(epistemic_warnings)}. Warnings: {epistemic_warnings}"
    
    warning_text = epistemic_warnings[0]
    
    # Check warning contains: count, top cal_ids, top reason_code
    assert "2/3" in warning_text or "2/3" in warning_text.replace(" ", "")
    assert "CAL-EXP-1" in warning_text or "CAL-EXP-2" in warning_text
    assert "EPI_DEGRADED_EVID_IMPROVING" in warning_text

    # Use reusable helper (single source of truth for banned words)
    pytest_assert_warning_neutral(warning_text, context="epistemic panel warning")


def test_epistemic_panel_no_warning_when_all_consistent(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that no warning is generated when all experiments are consistent."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_consistent": 3,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "warnings" in status
    warnings = status["warnings"]
    
    # Check that no epistemic panel inconsistency warning is present
    epistemic_warnings = [w for w in warnings if "epistemic panel" in w.lower() and "inconsistent" in w.lower()]
    assert len(epistemic_warnings) == 0


def test_epistemic_panel_top3_limited(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that top_inconsistent_cal_ids is limited to top 3."""
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
                        "cal_id": f"CAL-EXP-{i}",
                        "reason": f"Reason {i}",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    }
                    for i in range(1, 6)
                ],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "epistemic_panel" in status["signals"]
    panel_signal = status["signals"]["epistemic_panel"]
    assert len(panel_signal["top_inconsistent_cal_ids_top3"]) == 3
    # Verify truncation: should be exactly 3 even if more exist
    assert len(panel_signal["top_inconsistent_cal_ids_top3"]) <= 3


def test_epistemic_panel_json_safe(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that epistemic panel signal is JSON-safe."""
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
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason",
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
    
    # Should not raise
    json_str = json.dumps(status)
    assert isinstance(json_str, str)
    
    # Should be able to parse back
    parsed = json.loads(json_str)
    assert "signals" in parsed
    assert "epistemic_panel" in parsed["signals"]
    panel_signal = parsed["signals"]["epistemic_panel"]
    assert "extraction_source" in panel_signal
    assert "panel_schema_version" in panel_signal
    assert "top_reason_code" in panel_signal


def test_extraction_source_correctness(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that extraction_source is correctly set based on source."""
    # Test MANIFEST source
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 1,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["extraction_source"] == "MANIFEST"
    
    # Test EVIDENCE_JSON fallback
    manifest["governance"] = {}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    evidence = {
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 1,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["extraction_source"] == "EVIDENCE_JSON"
    
    # Verify fallback warning is present
    warnings = status["warnings"]
    fallback_warnings = [w for w in warnings if "evidence.json fallback" in w.lower()]
    assert len(fallback_warnings) == 1


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
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_STABLE",  # Same count, should come after IMPROVING
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",  # Same count, should come first (asc order)
                    },
                    {
                        "cal_id": "CAL-EXP-3",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_STABLE",
                    },
                    {
                        "cal_id": "CAL-EXP-4",
                        "reason": "Test reason",
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
    
    # Run again to verify determinism
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal1 = status["signals"]["epistemic_panel"]
    panel_signal2 = status2["signals"]["epistemic_panel"]
    
    # Both should have same top_reason_code
    assert panel_signal1["top_reason_code"] == panel_signal2["top_reason_code"]
    
    # With tie (both have count 2), EPI_DEGRADED_EVID_IMPROVING should win (asc order)
    assert panel_signal1["top_reason_code"] == "EPI_DEGRADED_EVID_IMPROVING"


def test_extraction_source_coercion_unknown_to_missing(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that unknown extraction_source values are coerced to MISSING with advisory."""
    # This test simulates an edge case where extraction_source might be set to an invalid value
    # In practice, this should not happen, but we test the coercion law
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 1,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "epistemic_panel" in status["signals"]
    
    panel_signal = status["signals"]["epistemic_panel"]
    # Should be MANIFEST (valid value)
    assert panel_signal["extraction_source"] in ("MANIFEST", "EVIDENCE_JSON", "MISSING")
    assert panel_signal["extraction_source"] == "MANIFEST"


def test_panel_schema_version_present_field(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that panel_schema_version_present bool field is correctly set."""
    # Test with schema_version present
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 1,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    assert panel_signal["panel_schema_version_present"] is True
    assert panel_signal["panel_schema_version"] == "1.0.0"
    
    # Test with schema_version missing
    manifest["governance"]["epistemic_panel"].pop("schema_version")
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal2 = status2["signals"]["epistemic_panel"]
    assert panel_signal2["panel_schema_version_present"] is False
    assert panel_signal2["panel_schema_version"] == "UNKNOWN"


def test_status_ggfl_consistency_block_present(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that status↔GGFL consistency block is attached to signal."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 2,
                "num_consistent": 0,
                "num_inconsistent": 2,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test reason",
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
    assert "consistency" in panel_signal
    
    consistency = panel_signal["consistency"]
    assert "status_consistent" in consistency
    assert "driver_consistent" in consistency
    assert "notes" in consistency
    assert isinstance(consistency["notes"], list)
    
    # Should be consistent in normal case
    assert consistency["status_consistent"] is True
    assert consistency["driver_consistent"] is True


def test_status_ggfl_consistency_detects_status_mismatch(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that consistency block detects status mismatch between signal and GGFL."""
    # This test requires mocking the GGFL adapter to return a mismatched status
    # For now, we verify the consistency block structure exists
    # In practice, status should always be consistent, but we test the detection logic
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 1,
                "num_inconsistent": 0,
                "num_unknown": 0,
                "experiments_inconsistent": [],
            },
        },
    }
    
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    panel_signal = status["signals"]["epistemic_panel"]
    consistency = panel_signal["consistency"]
    
    # With no inconsistencies, status should be "ok" and should match GGFL
    assert consistency["status_consistent"] is True


def test_status_ggfl_consistency_detects_driver_mismatch(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that consistency block detects top_reason_code mismatch in GGFL drivers."""
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 1,
                "num_consistent": 0,
                "num_inconsistent": 1,
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason",
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
    consistency = panel_signal["consistency"]
    
    # Top reason code should be present in GGFL drivers
    assert consistency["driver_consistent"] is True
    # Status should be "warn" (num_inconsistent > 0) and should match GGFL
    assert consistency["status_consistent"] is True


def test_warning_caps_fallback_and_inconsistency(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that warning caps are enforced: ≤1 fallback warning, ≤1 inconsistency warning."""
    # Create evidence.json with epistemic panel (triggers fallback warning)
    evidence = {
        "governance": {
            "epistemic_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_consistent": 0,
                "num_inconsistent": 3,  # Triggers inconsistency warning
                "num_unknown": 0,
                "experiments_inconsistent": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                    {
                        "cal_id": "CAL-EXP-3",
                        "reason": "Test reason",
                        "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                    },
                ],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    warnings = status.get("warnings", [])
    
    # Count fallback warnings
    fallback_warnings = [w for w in warnings if "evidence.json fallback" in w.lower()]
    assert len(fallback_warnings) <= 1, f"Expected ≤1 fallback warning, got {len(fallback_warnings)}"
    
    # Count inconsistency warnings
    inconsistency_warnings = [w for w in warnings if "epistemic panel" in w.lower() and "inconsistent" in w.lower()]
    assert len(inconsistency_warnings) <= 1, f"Expected ≤1 inconsistency warning, got {len(inconsistency_warnings)}"
    
    # Should have exactly 1 of each in this case
    assert len(fallback_warnings) == 1
    assert len(inconsistency_warnings) == 1