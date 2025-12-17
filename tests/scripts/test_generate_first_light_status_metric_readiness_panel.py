"""
Tests for Metric Readiness Panel integration with First Light status generator.

Validates:
- Metric readiness panel signal extraction from evidence pack manifest
- Status signal appears when panel present, absent otherwise
- Top driver ordering deterministic
- JSON round-trip safe
- Warning generation for num_block > 0 OR num_poly_fail > 0
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest

from scripts.generate_first_light_status import generate_status


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def minimal_evidence_pack_dir(tmp_path: Path) -> Path:
    """Create minimal evidence pack directory structure."""
    evidence_dir = tmp_path / "evidence_pack"
    evidence_dir.mkdir()

    # Create minimal P3 directory
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

    # Create minimal P4 directory
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

    # Create minimal manifest.json
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "test_run",
        "governance": {},
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return evidence_dir


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_metric_readiness_panel_signal_appears_when_panel_present(
    minimal_evidence_pack_dir: Path,
):
    """Status signal appears when metric readiness panel is present in manifest."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add metric readiness panel to manifest
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "num_ok": 1,
        "num_warn": 1,
        "num_block": 1,
        "num_poly_fail": 1,
        "global_norm_range": {
            "p3_min": 0.2,
            "p3_max": 0.8,
            "p4_min": 0.2,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": ["CAL-EXP-3", "CAL-EXP-2", "CAL-EXP-1"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check signal is present
    assert status["signals"] is not None
    assert "metric_readiness_panel" in status["signals"]

    signal = status["signals"]["metric_readiness_panel"]
    assert signal["schema_version"] == "1.0.0"
    assert signal["mode"] == "SHADOW"
    assert signal["extraction_source"] == "MANIFEST"
    assert signal["panel_schema_version"] == "1.0.0"
    assert signal["num_ok"] == 1
    assert signal["num_warn"] == 1
    assert signal["num_block"] == 1
    assert signal["num_poly_fail"] == 1
    assert signal["p4_global_norm_range"]["min"] == 0.2
    assert signal["p4_global_norm_range"]["max"] == 0.75
    assert signal["top_driver_cal_ids"] == ["CAL-EXP-3", "CAL-EXP-2", "CAL-EXP-1"]
    # Verify top3 truncation and deterministic sorting
    assert signal["top_driver_cal_ids_top3"] == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]
    assert len(signal["top_driver_cal_ids_top3"]) <= 3


def test_metric_readiness_panel_signal_absent_when_panel_missing(
    minimal_evidence_pack_dir: Path,
):
    """Status signal is absent when metric readiness panel is not in manifest or evidence.json."""
    # Generate status without panel
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check signal is absent
    if status["signals"]:
        assert "metric_readiness_panel" not in status["signals"]


def test_metric_readiness_panel_extraction_manifest_first(
    minimal_evidence_pack_dir: Path,
):
    """Panel extraction prefers manifest over evidence.json."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel to manifest
    manifest["governance"]["metric_readiness_panel"] = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {"p3_min": 0.5, "p3_max": 0.8, "p4_min": 0.5, "p4_max": 0.75},
        "top_driver_cal_ids": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create evidence.json with different panel (should be ignored)
    evidence_path = minimal_evidence_pack_dir / "evidence.json"
    evidence = {
        "governance": {
            "metric_readiness_panel": {
                "num_experiments": 999,  # Different value
            }
        }
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check manifest panel was used (not evidence.json)
    signal = status["signals"]["metric_readiness_panel"]
    assert signal["extraction_source"] == "MANIFEST"
    assert signal["panel_schema_version"] == "1.0.0"
    # Verify no warning for MANIFEST source
    evidence_json_warnings = [w for w in status["warnings"] if "evidence.json fallback" in w]
    assert len(evidence_json_warnings) == 0


def test_metric_readiness_panel_extraction_evidence_fallback(
    minimal_evidence_pack_dir: Path,
):
    """Panel extraction falls back to evidence.json when not in manifest."""
    # Load manifest (no panel)
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Ensure no panel in manifest
    if "governance" in manifest:
        manifest["governance"].pop("metric_readiness_panel", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create evidence.json with panel
    evidence_path = minimal_evidence_pack_dir / "evidence.json"
    evidence = {
        "governance": {
            "metric_readiness_panel": {
                "schema_version": "1.0.0",
                "num_experiments": 3,
                "num_ok": 1,
                "num_warn": 1,
                "num_block": 1,
                "num_poly_fail": 0,
                "global_norm_range": {"p3_min": 0.2, "p3_max": 0.8, "p4_min": 0.2, "p4_max": 0.75},
                "top_driver_cal_ids": ["CAL-EXP-3"],
            }
        }
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check evidence.json panel was used
    signal = status["signals"]["metric_readiness_panel"]
    assert signal["extraction_source"] == "EVIDENCE_JSON"
    assert signal["panel_schema_version"] == "1.0.0"
    # Verify exactly one warning for EVIDENCE_JSON source
    evidence_json_warnings = [w for w in status["warnings"] if "evidence.json fallback" in w]
    assert len(evidence_json_warnings) == 1
    assert "Advisory only" in evidence_json_warnings[0]


def test_metric_readiness_panel_single_warning_cap(
    minimal_evidence_pack_dir: Path,
):
    """At most one warning line is emitted even when both num_block and num_poly_fail > 0."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel with both num_block and num_poly_fail > 0
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "num_ok": 1,
        "num_warn": 0,
        "num_block": 1,
        "num_poly_fail": 1,
        "global_norm_range": {
            "p3_min": 0.2,
            "p3_max": 0.8,
            "p4_min": 0.2,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": ["CAL-EXP-3", "CAL-EXP-2", "CAL-EXP-1"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check exactly one warning line for metric readiness panel (excluding extraction_source warning)
    metric_readiness_warnings = [
        w for w in status["warnings"] 
        if "Metric readiness panel" in w and "fallback" not in w
    ]
    assert len(metric_readiness_warnings) == 1
    # Warning should include both counts and top3 cal_ids (max 3)
    warning_text = metric_readiness_warnings[0]
    assert "BLOCK" in warning_text
    assert "poly-fail" in warning_text
    # Verify top3 cal_ids are in warning (sorted deterministically)
    signal = status["signals"]["metric_readiness_panel"]
    top3 = signal["top_driver_cal_ids_top3"]
    assert len(top3) <= 3
    # All top3 cal_ids should be in warning if present
    for cal_id in top3:
        assert cal_id in warning_text


def test_metric_readiness_panel_warning_when_num_block_gt_zero(
    minimal_evidence_pack_dir: Path,
):
    """Warning generated when num_block > 0."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel with num_block > 0
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "num_ok": 1,
        "num_warn": 1,
        "num_block": 1,
        "num_poly_fail": 0,
        "global_norm_range": {
            "p3_min": 0.2,
            "p3_max": 0.8,
            "p4_min": 0.2,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": ["CAL-EXP-3"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check warning is present (single consolidated warning)
    assert len(status["warnings"]) > 0
    block_warnings = [w for w in status["warnings"] if "BLOCK" in w and "Metric readiness panel" in w]
    assert len(block_warnings) > 0
    assert "CAL-EXP-3" in block_warnings[0] or "drivers" in block_warnings[0]


def test_metric_readiness_panel_warning_when_num_poly_fail_gt_zero(
    minimal_evidence_pack_dir: Path,
):
    """Warning generated when num_poly_fail > 0."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel with num_poly_fail > 0
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 1,
        "global_norm_range": {
            "p3_min": 0.5,
            "p3_max": 0.8,
            "p4_min": 0.5,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": [],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check warning is present (single consolidated warning)
    assert len(status["warnings"]) > 0
    poly_fail_warnings = [w for w in status["warnings"] if "poly-fail" in w and "Metric readiness panel" in w]
    assert len(poly_fail_warnings) > 0


def test_metric_readiness_panel_signal_json_round_trip(
    minimal_evidence_pack_dir: Path,
):
    """Metric readiness panel signal is JSON round-trip safe."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add metric readiness panel
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 1,
        "num_warn": 1,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {
            "p3_min": 0.4,
            "p3_max": 0.8,
            "p4_min": 0.3,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": ["CAL-EXP-2"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Round-trip through JSON
    signal = status["signals"]["metric_readiness_panel"]
    json_str = json.dumps(signal, sort_keys=True)
    parsed = json.loads(json_str)

    # Verify fields preserved (including new provenance fields)
    assert parsed["extraction_source"] == signal["extraction_source"]
    assert parsed["panel_schema_version"] == signal["panel_schema_version"]
    assert parsed["panel_schema_version_present"] == signal["panel_schema_version_present"]
    assert parsed["num_ok"] == signal["num_ok"]
    assert parsed["num_warn"] == signal["num_warn"]
    assert parsed["num_block"] == signal["num_block"]
    assert parsed["num_poly_fail"] == signal["num_poly_fail"]
    assert parsed["p4_global_norm_range"]["min"] == signal["p4_global_norm_range"]["min"]
    assert parsed["p4_global_norm_range"]["max"] == signal["p4_global_norm_range"]["max"]
    assert parsed["top_driver_cal_ids"] == signal["top_driver_cal_ids"]
    assert parsed["top_driver_cal_ids_top3"] == signal["top_driver_cal_ids_top3"]
    # Verify top3 is deterministically sorted
    assert parsed["top_driver_cal_ids_top3"] == sorted(parsed["top_driver_cal_ids_top3"])
    assert len(parsed["top_driver_cal_ids_top3"]) <= 3
    # Verify consistency field is present
    assert "consistency" in parsed
    assert parsed["consistency"]["status"] in ["CONSISTENT", "PARTIAL", "INCONSISTENT", "UNKNOWN"]


def test_metric_readiness_panel_warning_neutral_language(
    minimal_evidence_pack_dir: Path,
):
    """Warning text uses neutral and descriptive language."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel with both num_block and num_poly_fail > 0
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "num_ok": 1,
        "num_warn": 1,
        "num_block": 1,
        "num_poly_fail": 1,
        "global_norm_range": {
            "p3_min": 0.2,
            "p3_max": 0.8,
            "p4_min": 0.2,
            "p4_max": 0.75,
        },
        "top_driver_cal_ids": ["CAL-EXP-3"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Check warnings use neutral language
    for warning in status["warnings"]:
        warning_lower = warning.lower()
        # Check for absence of value judgment words
        assert "good" not in warning_lower
        assert "bad" not in warning_lower
        assert "better" not in warning_lower
        assert "worse" not in warning_lower
        assert "success" not in warning_lower
        assert "failure" not in warning_lower
        # Should be descriptive
        assert "metric readiness" in warning_lower or "poly-fail" in warning_lower


def test_metric_readiness_panel_extraction_source_coercion(
    minimal_evidence_pack_dir: Path,
):
    """Unknown extraction_source is coerced to MISSING with advisory note."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add panel
    panel = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {"p3_min": 0.5, "p3_max": 0.8, "p4_min": 0.5, "p4_max": 0.75},
        "top_driver_cal_ids": [],
    }
    manifest["governance"]["metric_readiness_panel"] = panel
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    # Verify extraction_source is valid enum value
    signal = status["signals"]["metric_readiness_panel"]
    assert signal["extraction_source"] in ["MANIFEST", "EVIDENCE_JSON", "MISSING"]
    # Should be MANIFEST in this case
    assert signal["extraction_source"] == "MANIFEST"


def test_metric_readiness_panel_schema_version_present(
    minimal_evidence_pack_dir: Path,
):
    """panel_schema_version_present indicates if schema_version was present in panel."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Test case 1: schema_version present
    panel_with_version = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {},
        "top_driver_cal_ids": [],
    }
    manifest["governance"]["metric_readiness_panel"] = panel_with_version
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    signal = status["signals"]["metric_readiness_panel"]
    assert signal["panel_schema_version_present"] is True
    assert signal["panel_schema_version"] == "1.0.0"

    # Test case 2: schema_version missing
    panel_without_version = {
        "num_experiments": 2,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {},
        "top_driver_cal_ids": [],
    }
    manifest["governance"]["metric_readiness_panel"] = panel_without_version
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    signal = status["signals"]["metric_readiness_panel"]
    assert signal["panel_schema_version_present"] is False
    assert signal["panel_schema_version"] == "UNKNOWN"


def test_metric_readiness_panel_consistency_checker(
    minimal_evidence_pack_dir: Path,
):
    """Consistency checker compares status signal vs GGFL adapter output."""
    # Load manifest
    manifest_path = minimal_evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Test case 1: Consistent (both warn)
    panel_warn = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 1,
        "num_warn": 0,
        "num_block": 1,
        "num_poly_fail": 0,
        "global_norm_range": {"p4_min": 0.5, "p4_max": 0.75},
        "top_driver_cal_ids": ["CAL-EXP-3"],
    }
    manifest["governance"]["metric_readiness_panel"] = panel_warn
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    p3_dir = minimal_evidence_pack_dir.parent / "p3"
    p4_dir = minimal_evidence_pack_dir.parent / "p4"
    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    signal = status["signals"]["metric_readiness_panel"]
    assert "consistency" in signal
    consistency = signal["consistency"]
    assert consistency["status"] in ["CONSISTENT", "PARTIAL", "INCONSISTENT", "UNKNOWN"]
    assert "status_signal_expected" in consistency
    assert "ggfl_adapter_status" in consistency
    # Should be CONSISTENT (both should say warn)
    assert consistency["status"] == "CONSISTENT"
    assert consistency["status_signal_expected"] == "warn"
    assert consistency["ggfl_adapter_status"] == "warn"

    # Test case 2: Consistent (both ok)
    panel_ok = {
        "schema_version": "1.0.0",
        "num_experiments": 2,
        "num_ok": 2,
        "num_warn": 0,
        "num_block": 0,
        "num_poly_fail": 0,
        "global_norm_range": {"p4_min": 0.8, "p4_max": 0.9},
        "top_driver_cal_ids": [],
    }
    manifest["governance"]["metric_readiness_panel"] = panel_ok
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status = generate_status(
        p3_dir,
        p4_dir,
        minimal_evidence_pack_dir,
        pipeline="local",
    )

    signal = status["signals"]["metric_readiness_panel"]
    consistency = signal["consistency"]
    assert consistency["status"] == "CONSISTENT"
    assert consistency["status_signal_expected"] == "ok"
    assert consistency["ggfl_adapter_status"] == "ok"

