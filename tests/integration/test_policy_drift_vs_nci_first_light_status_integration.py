"""
Integration tests for POLICY DRIFT × NCI status surfacing in first_light_status.json.

SHADOW MODE CONTRACT:
- Observational only (no gating)
- Tests validate deterministic, non-mutating extraction behavior
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.health.policy_drift_tile import policy_drift_vs_nci_for_alignment_view
from scripts.generate_first_light_status import generate_status


@pytest.fixture
def evidence_pack_dir(tmp_path: Path) -> Path:
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()

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

    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack_dir


@pytest.fixture
def p3_dir(tmp_path: Path) -> Path:
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
    p4_dir = tmp_path / "p4"
    p4_dir.mkdir()
    p4_run = p4_dir / "p4_test"
    p4_run.mkdir()
    (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
    (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
    (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
    (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}), encoding="utf-8")
    (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
    (p4_run / "run_config.json").write_text("{}", encoding="utf-8")
    return p4_dir


@pytest.mark.integration
def test_policy_drift_vs_nci_source_precedence_manifest_over_evidence_json(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    evidence = {
        "governance": {
            "policy_drift": {"status": "BLOCK", "num_rules": 2, "blocked_rules": 1},
            "nci": {"health_contribution": {"status": "BREACH", "global_nci": 0.50}},
        }
    }
    (evidence_pack_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    manifest = json.loads((evidence_pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["governance"] = {
        "policy_drift": {"status": "OK", "num_rules": 0, "blocked_rules": 0},
    }
    manifest["signals"] = {
        "nci_p5": {"slo_status": "OK", "global_nci": 0.85, "confidence": 0.7},
    }
    (evidence_pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    signal = (status.get("signals") or {}).get("policy_drift_vs_nci")
    assert signal is not None

    assert signal["extraction_source_policy"] == "MANIFEST"
    assert signal["extraction_source_nci"] == "MANIFEST"
    assert signal["consistency_status"] == "CONSISTENT"


@pytest.mark.integration
@pytest.mark.determinism
def test_policy_drift_vs_nci_determinism_and_sources_evidence_json(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    evidence = {
        "governance": {
            "policy_drift": {"status": "WARN", "num_rules": 1, "blocked_rules": 0},
            "nci": {"health_contribution": {"status": "WARN", "global_nci": 0.72}},
        }
    }
    (evidence_pack_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    status_a = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    status_b = generate_status(p3_dir, p4_dir, evidence_pack_dir)

    signals_a = status_a.get("signals") or {}
    signals_b = status_b.get("signals") or {}

    assert "policy_drift_vs_nci" in signals_a
    assert signals_a["policy_drift_vs_nci"] == signals_b["policy_drift_vs_nci"]
    assert signals_a["policy_drift_vs_nci"] == {
        "consistency_status": "CONSISTENT",
        "policy_status_light": "YELLOW",
        "nci_status_light": "YELLOW",
        "advisory_notes": [
            "Policy drift status WARN with 0 blocking rule(s).",
            "NCI signal status WARN with global NCI 0.72.",
        ],
        "extraction_source_policy": "EVIDENCE_JSON",
        "extraction_source_nci": "EVIDENCE_JSON",
    }


@pytest.mark.integration
def test_policy_drift_vs_nci_alignment_view_adapter_shape(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    evidence = {
        "governance": {
            "policy_drift": {"status": "WARN", "num_rules": 1, "blocked_rules": 0},
            "nci": {"health_contribution": {"status": "WARN", "global_nci": 0.72}},
        }
    }
    (evidence_pack_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    signal = (status.get("signals") or {}).get("policy_drift_vs_nci")
    assert signal is not None

    view = policy_drift_vs_nci_for_alignment_view(signal)
    assert view == {
        "signal_type": "SIG-PD",
        "status": "ok",
        "conflict": False,
        "drivers": [],
        "summary": "Policy drift and NCI are CONSISTENT (policy YELLOW, NCI YELLOW).",
    }


@pytest.mark.integration
def test_policy_drift_vs_nci_alignment_view_driver_ordering_and_cap() -> None:
    view = policy_drift_vs_nci_for_alignment_view(
        {
            "consistency_status": "INCONSISTENT",
            "policy_status_light": "RED",
            "nci_status_light": "RED",
            "advisory_notes": [
                "Policy drift status BLOCK with 1 blocking rule(s).",
                "NCI signal status BREACH with global NCI 0.55.",
            ],
            "extraction_source_policy": "MANIFEST",
            "extraction_source_nci": "EVIDENCE_JSON",
        }
    )
    assert view["status"] == "warn"
    assert view["drivers"] == [
        "DRIVER_STATUS_INCONSISTENT",
        "DRIVER_POLICY_BLOCK",
        "DRIVER_NCI_BREACH",
    ]
    assert len(view["drivers"]) <= 3


@pytest.mark.integration
def test_policy_drift_vs_nci_alignment_view_json_safety() -> None:
    view = policy_drift_vs_nci_for_alignment_view(
        {
            "consistency_status": None,
            "policy_status_light": 123,
            "nci_status_light": object(),
            "advisory_notes": {"not": "a list"},
            "extraction_source_policy": None,
            "extraction_source_nci": ["bad"],
        }
    )
    encoded = json.dumps(view, sort_keys=True)
    assert isinstance(encoded, str) and len(encoded) > 20


@pytest.mark.integration
def test_policy_drift_vs_nci_inconsistent_adds_single_warning_line(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    evidence = {
        "governance": {
            "policy_drift": {"status": "OK", "num_rules": 0, "blocked_rules": 0},
        }
    }
    (evidence_pack_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    manifest = json.loads((evidence_pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["signals"] = {
        "nci_p5": {"slo_status": "BREACH", "global_nci": 0.55, "confidence": 0.7},
    }
    (evidence_pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    warnings = status.get("warnings") or []
    pdn_lines = [
        line
        for line in warnings
        if "Policy drift vs NCI INCONSISTENT:" in line
    ]
    assert len(pdn_lines) == 1
    assert "DRIVER_STATUS_INCONSISTENT" in pdn_lines[0]
    assert "DRIVER_POLICY_BLOCK" not in pdn_lines[0]
    assert "DRIVER_NCI_BREACH" not in pdn_lines[0]
