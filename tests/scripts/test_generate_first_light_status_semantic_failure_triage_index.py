"""
Tests for semantic failure triage index extraction in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings
- Manifest-first extraction pattern validated
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
    return p4_dir


def test_triage_index_signal_extracted_from_manifest(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify triage index signal is extracted from manifest.json (manifest-first)."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 2.0,
            "p4_tensor_norm": 2.5,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        },
        {
            "cal_id": "CAL-EXP-2",
            "p3_tensor_norm": 1.0,
            "p4_tensor_norm": 1.5,
            "semantic_hotspots": ["slice_b"],
            "regression_status": "ATTENTION",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal is present
    assert "signals" in status
    assert "semantic_failure_triage_index" in status["signals"]

    signal = status["signals"]["semantic_failure_triage_index"]
    assert signal["schema_version"] == "1.0.0"
    assert signal["mode"] == "SHADOW"
    assert signal["extraction_source"] == "MANIFEST"
    assert signal["total_items"] == 2
    assert len(signal["top5"]) == 2

    # Verify top5 structure
    top5 = signal["top5"]
    assert top5[0]["cal_id"] == "CAL-EXP-1"
    assert top5[0]["regression_status"] == "REGRESSED"
    assert top5[0]["combined_tensor_norm"] == 4.5
    assert top5[0]["hotspots_count"] == 1

    assert top5[1]["cal_id"] == "CAL-EXP-2"
    assert top5[1]["regression_status"] == "ATTENTION"
    assert top5[1]["combined_tensor_norm"] == 2.5
    assert top5[1]["hotspots_count"] == 1


def test_triage_index_signal_fallback_to_evidence_json(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify triage index signal falls back to evidence.json if not in manifest."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 1.5,
            "p4_tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "STABLE",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to evidence.json (not manifest)
    evidence_path = evidence_pack_dir / "evidence.json"
    evidence = {
        "governance": {
            "semantic_failure_triage_index": triage_index,
        },
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal is present (extracted from evidence.json fallback)
    assert "signals" in status
    assert "semantic_failure_triage_index" in status["signals"]

    signal = status["signals"]["semantic_failure_triage_index"]
    assert signal["extraction_source"] == "EVIDENCE_JSON"
    assert signal["total_items"] == 1
    assert len(signal["top5"]) == 1


def test_triage_index_signal_missing_index_safe(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify status generation is safe when triage index is missing."""
    # Generate status without triage index
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal is not present (missing index is safe)
    signals = status.get("signals") or {}
    assert "semantic_failure_triage_index" not in signals


def test_triage_index_signal_advisory_warning_on_regressed(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify advisory warning is emitted when REGRESSED status detected."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index with REGRESSED status
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 2.0,
            "p4_tensor_norm": 2.5,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal has advisory_warning
    signal = status["signals"]["semantic_failure_triage_index"]
    assert "advisory_warning" in signal
    assert "REGRESSED" in signal["advisory_warning"]

    # Verify warning is emitted exactly once
    warnings = status.get("warnings", [])
    regressed_warnings = [w for w in warnings if "REGRESSED" in w]
    assert len(regressed_warnings) == 1, "Advisory warning should be emitted exactly once"
    
    # Verify warning includes cal_id
    warning_text = regressed_warnings[0]
    assert "CAL-EXP-1" in warning_text, "Warning should include top cal_id"


def test_triage_index_signal_no_warning_when_stable(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify no advisory warning when all items are STABLE."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index with STABLE status
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 0.5,
            "p4_tensor_norm": 0.5,
            "semantic_hotspots": [],
            "regression_status": "STABLE",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal does not have advisory_warning
    signal = status["signals"]["semantic_failure_triage_index"]
    assert "advisory_warning" not in signal

    # Verify no warnings about REGRESSED
    warnings = status.get("warnings", [])
    regressed_warnings = [w for w in warnings if "REGRESSED" in w]
    assert len(regressed_warnings) == 0


def test_triage_index_signal_deterministic(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify triage index signal extraction is deterministic."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 2.0,
            "p4_tensor_norm": 2.5,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        },
        {
            "cal_id": "CAL-EXP-2",
            "p3_tensor_norm": 1.0,
            "p4_tensor_norm": 1.5,
            "semantic_hotspots": ["slice_b"],
            "regression_status": "ATTENTION",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status twice
    status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signals are identical
    signal1 = status1["signals"]["semantic_failure_triage_index"]
    signal2 = status2["signals"]["semantic_failure_triage_index"]

    assert signal1 == signal2, "Signal extraction should be deterministic"

    # Verify JSON serialization is also deterministic
    json1 = json.dumps(signal1, sort_keys=True)
    json2 = json.dumps(signal2, sort_keys=True)
    assert json1 == json2


def test_ggfl_adapter_stub_sig_sdrift(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify GGFL adapter stub SIG-SDRIFT produces correct signal."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
        extract_semantic_failure_triage_index_signal,
        extract_semantic_drift_triage_signal_for_ggfl,
    )

    # Build a triage index with REGRESSED status
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 2.0,
            "p4_tensor_norm": 2.5,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        },
        {
            "cal_id": "CAL-EXP-2",
            "p3_tensor_norm": 1.0,
            "p4_tensor_norm": 1.5,
            "semantic_hotspots": ["slice_b"],
            "regression_status": "STABLE",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)
    triage_signal = extract_semantic_failure_triage_index_signal(triage_index)
    ggfl_signal = extract_semantic_drift_triage_signal_for_ggfl(triage_signal)

    # Verify GGFL signal structure (GGFL conventions)
    assert ggfl_signal["signal_type"] == "SIG-SDRIFT"
    assert ggfl_signal["status"] == "warn"  # Lowercase, warn because REGRESSED exists
    assert ggfl_signal["conflict"] is False
    assert ggfl_signal["weight_hint"] == "LOW"
    
    # Verify drivers (reason codes only, no prose)
    assert "drivers" in ggfl_signal
    assert isinstance(ggfl_signal["drivers"], list)
    assert "DRIVER_REGRESSED_PRESENT" in ggfl_signal["drivers"]
    assert "DRIVER_TOP_CAL_IDS_PRESENT" in ggfl_signal["drivers"]
    
    # Verify shadow_mode_invariants (3-boolean block)
    assert "shadow_mode_invariants" in ggfl_signal
    invariants = ggfl_signal["shadow_mode_invariants"]
    assert invariants["observational_only"] is True
    assert invariants["no_control_flow"] is True
    assert invariants["advisory_weight"] == "LOW"
    
    # Verify summary (neutral sentence)
    assert "summary" in ggfl_signal
    assert isinstance(ggfl_signal["summary"], str)
    
    # Verify data fields
    assert ggfl_signal["total_items"] == 2
    assert ggfl_signal["regressed_count"] == 1


def test_ggfl_adapter_stub_status_ok_when_stable(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify GGFL adapter stub returns ok status when all items are STABLE."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
        extract_semantic_failure_triage_index_signal,
        extract_semantic_drift_triage_signal_for_ggfl,
    )

    # Build a triage index with STABLE status only
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 0.5,
            "p4_tensor_norm": 0.5,
            "semantic_hotspots": [],
            "regression_status": "STABLE",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)
    triage_signal = extract_semantic_failure_triage_index_signal(triage_index)
    ggfl_signal = extract_semantic_drift_triage_signal_for_ggfl(triage_signal)

    # Verify GGFL signal structure (GGFL conventions)
    assert ggfl_signal["signal_type"] == "SIG-SDRIFT"
    assert ggfl_signal["status"] == "ok"  # Lowercase, ok when no REGRESSED
    assert ggfl_signal["conflict"] is False
    assert ggfl_signal["weight_hint"] == "LOW"
    
    # Verify drivers (empty when no regressed)
    assert "drivers" in ggfl_signal
    assert isinstance(ggfl_signal["drivers"], list)
    assert len(ggfl_signal["drivers"]) == 0  # No drivers when all stable
    
    # Verify shadow_mode_invariants (3-boolean block)
    assert "shadow_mode_invariants" in ggfl_signal
    invariants = ggfl_signal["shadow_mode_invariants"]
    assert invariants["observational_only"] is True
    assert invariants["no_control_flow"] is True
    assert invariants["advisory_weight"] == "LOW"
    
    # Verify summary present
    assert "summary" in ggfl_signal
    assert isinstance(ggfl_signal["summary"], str)
    
    # Verify data fields
    assert ggfl_signal["regressed_count"] == 0


def test_triage_index_signal_top5_truncates_to_5(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify top5 truncates to 5 items even if more exist."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index with 10 items
    shelves = [
        {
            "cal_id": f"CAL-EXP-{i}",
            "p3_tensor_norm": float(i),
            "p4_tensor_norm": float(i),
            "semantic_hotspots": [],
            "regression_status": "REGRESSED",
        }
        for i in range(10)
    ]

    triage_index = build_semantic_failure_triage_index(shelves, max_items=10)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify signal has total_items=10 but top5 has only 5 items
    signal = status["signals"]["semantic_failure_triage_index"]
    assert signal["total_items"] == 10
    assert len(signal["top5"]) == 5


def test_triage_index_signal_warning_cap(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
) -> None:
    """Verify advisory warning is emitted at most once (warning cap)."""
    from backend.health.semantic_drift_adapter import (
        build_semantic_failure_triage_index,
    )

    # Build a triage index with multiple REGRESSED items
    shelves = [
        {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 2.0,
            "p4_tensor_norm": 2.5,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        },
        {
            "cal_id": "CAL-EXP-2",
            "p3_tensor_norm": 1.5,
            "p4_tensor_norm": 2.0,
            "semantic_hotspots": ["slice_b"],
            "regression_status": "REGRESSED",
        },
        {
            "cal_id": "CAL-EXP-3",
            "p3_tensor_norm": 1.0,
            "p4_tensor_norm": 1.5,
            "semantic_hotspots": ["slice_c"],
            "regression_status": "REGRESSED",
        },
    ]

    triage_index = build_semantic_failure_triage_index(shelves)

    # Add triage index to manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["governance"]["semantic_failure_triage_index"] = triage_index
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    # Verify warning is emitted exactly once (not once per REGRESSED item)
    warnings = status.get("warnings", [])
    regressed_warnings = [w for w in warnings if "REGRESSED" in w]
    assert len(regressed_warnings) == 1, "Advisory warning should be emitted exactly once, not per item"

    # Verify warning mentions count of regressed experiments and includes cal_ids
    warning_text = regressed_warnings[0]
    assert "3" in warning_text or "experiment" in warning_text.lower()
    # Verify top cal_ids are included (≤3)
    assert "CAL-EXP-1" in warning_text
    assert "CAL-EXP-2" in warning_text
    assert "CAL-EXP-3" in warning_text

