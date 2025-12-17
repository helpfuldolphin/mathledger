"""
Tests for P3 pathology signal surfacing in first_light_status.json.

SHADOW MODE CONTRACT:
- Purely observational
- Pathology is test-only metadata
- Missing pathology must be safe
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from scripts.generate_first_light_status import generate_status


pytestmark = pytest.mark.unit


def _make_evidence_pack_dir(
    tmp_path: Path,
    name: str,
    manifest_pathology: Optional[Dict[str, Any]] = None,
    evidence_json_pathology: Optional[Dict[str, Any]] = None,
) -> Path:
    pack_dir = tmp_path / name
    pack_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "files": [{"path": "manifest.json", "sha256": "abc123"}],
        "governance": {},
    }
    if manifest_pathology is not None:
        manifest["evidence"] = {"data": {"p3_pathology": manifest_pathology}}
    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if evidence_json_pathology is not None:
        evidence = {"evidence": {"data": {"p3_pathology": evidence_json_pathology}}}
        (pack_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    return pack_dir


def _make_p3_dir(
    tmp_path: Path,
    name: str,
    pathology: Optional[str] = "none",
    magnitude: Optional[float] = None,
) -> Path:
    p3_dir = tmp_path / name
    p3_dir.mkdir(parents=True)
    fl_dir = p3_dir / "fl_run_001"
    fl_dir.mkdir()

    stability_report: Dict[str, Any] = {
        "metrics": {
            "success_rate": 0.95,
            "omega": {"occupancy_rate": 0.92},
            "rsi": {"mean": 0.88},
            "hard_mode": {"ok_rate": 0.90},
        },
        "criteria_evaluation": {"all_passed": True},
        "red_flag_summary": {"total_flags": 0, "hypothetical_abort": False},
    }
    if pathology is not None:
        stability_report["pathology"] = pathology
    if magnitude is not None:
        stability_report["pathology_params"] = {"magnitude": magnitude, "at": 3}

    with open(fl_dir / "stability_report.json", "w", encoding="utf-8") as f:
        json.dump(stability_report, f)

    for artifact in [
        "synthetic_raw.jsonl",
        "red_flag_matrix.json",
        "metrics_windows.json",
        "tda_metrics.json",
        "run_config.json",
    ]:
        (fl_dir / artifact).touch()

    return p3_dir


def _make_p4_dir(tmp_path: Path, name: str) -> Path:
    p4_dir = tmp_path / name
    p4_dir.mkdir(parents=True)
    run_dir = p4_dir / "p4_run_001"
    run_dir.mkdir()

    p4_summary = {
        "mode": "SHADOW",
        "uplift_metrics": {"u2_success_rate_final": 0.93},
        "divergence_analysis": {"divergence_rate": 0.05, "max_divergence_streak": 2},
        "twin_accuracy": {
            "success_prediction_accuracy": 0.85,
            "omega_prediction_accuracy": 0.90,
        },
    }
    (run_dir / "p4_summary.json").write_text(json.dumps(p4_summary), encoding="utf-8")

    run_config = {"run_id": "p4_run_001", "telemetry_source": "mock"}
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    for artifact in [
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
        "divergence_log.jsonl",
        "twin_accuracy.json",
    ]:
        (run_dir / artifact).touch()

    return p4_dir


def test_p3_pathology_signal_provenance(tmp_path: Path) -> None:
    p3_dir = _make_p3_dir(tmp_path, name="p3_none", pathology="none")
    p4_dir = _make_p4_dir(tmp_path, name="p4")

    manifest_pathology = {
        "pathology": "spike",
        "pathology_params": {"magnitude": 0.75, "at": 3},
    }
    evidence_pathology = {
        "pathology": "drift",
        "pathology_params": {"magnitude": 0.11, "slope": 0.02},
    }

    pack_manifest = _make_evidence_pack_dir(
        tmp_path,
        name="pack_manifest",
        manifest_pathology=manifest_pathology,
        evidence_json_pathology=evidence_pathology,
    )
    status_manifest = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_manifest,
    )
    signal_manifest = status_manifest.get("signals", {}).get("p3_pathology")
    assert signal_manifest is not None
    assert signal_manifest["extraction_source"] == "MANIFEST"
    assert signal_manifest["used"] is True
    assert signal_manifest["type"] == "spike"
    assert signal_manifest["magnitude"] == pytest.approx(0.75)

    pack_evidence = _make_evidence_pack_dir(
        tmp_path,
        name="pack_evidence",
        evidence_json_pathology=manifest_pathology,
    )
    status_evidence = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_evidence,
    )
    signal_evidence = status_evidence.get("signals", {}).get("p3_pathology")
    assert signal_evidence is not None
    assert signal_evidence["extraction_source"] == "EVIDENCE_JSON"
    assert signal_evidence["used"] is True
    assert signal_evidence["type"] == "spike"
    assert signal_evidence["magnitude"] == pytest.approx(0.75)

    pack_missing = _make_evidence_pack_dir(tmp_path, name="pack_missing")
    status_missing = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_missing,
    )
    signal_missing = status_missing.get("signals", {}).get("p3_pathology")
    assert signal_missing is not None
    assert signal_missing["extraction_source"] == "MISSING"
    assert signal_missing["used"] is False
    assert signal_missing["type"] is None
    assert signal_missing["magnitude"] is None


def test_p3_pathology_no_warning_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p3_dir = _make_p3_dir(tmp_path, name="p3_spike", pathology="spike", magnitude=0.75)
    p4_dir = _make_p4_dir(tmp_path, name="p4")

    manifest_pathology = {
        "pathology": "spike",
        "pathology_params": {"magnitude": 0.75, "at": 3},
    }
    evidence_pack_dir = _make_evidence_pack_dir(
        tmp_path,
        name="pack_pathology",
        manifest_pathology=manifest_pathology,
    )

    status = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=evidence_pack_dir,
    )

    warnings = status.get("warnings") or []
    assert all(not str(w).startswith("P3-PATH-001:") for w in warnings)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    status_non_test = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=evidence_pack_dir,
        pipeline="manual",
    )
    warnings_non_test = status_non_test.get("warnings") or []
    pathology_warnings = [w for w in warnings_non_test if str(w).startswith("P3-PATH-001:")]
    assert len(pathology_warnings) == 1
    assert "--pathology" in pathology_warnings[0]
