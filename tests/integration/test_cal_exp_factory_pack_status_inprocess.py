from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.topology.first_light.calibration_annex import load_cal_exp1_annex
from scripts.generate_first_light_status import generate_status
from tests.factories.first_light_factories import (
    emit_cal_exp_reports_for_evidence_pack,
    make_p4_divergence_log_record,
    make_stability_report_payload,
    make_synthetic_raw_record,
)

pytestmark = pytest.mark.integration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _create_minimal_p3_run(p3_dir: Path, *, seed: int) -> None:
    run_dir = p3_dir / "fl_factory_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "synthetic_raw.jsonl",
        [make_synthetic_raw_record(cycle, seed=seed) for cycle in range(1, 4)],
    )
    _write_json(run_dir / "stability_report.json", make_stability_report_payload(total_cycles=50, seed=seed))
    _write_json(run_dir / "red_flag_matrix.json", {"schema_version": "1.0.0", "flags": []})
    _write_json(
        run_dir / "metrics_windows.json",
        {
            "schema_version": "1.0.0",
            "run_id": run_dir.name,
            "window_size": 50,
            "total_windows": 1,
            "windows": [
                {
                    "window_index": 0,
                    "start_cycle": 1,
                    "end_cycle": 50,
                    "metrics": {"mean_rsi": 0.7, "omega_occupancy": 0.9},
                    "delta_p": {"success": None, "abstention": None},
                    "red_flags_in_window": 0,
                }
            ],
            "trajectories": {"mean_rsi": [0.7], "omega_occupancy": [0.9]},
            "mode": "SHADOW",
        },
    )
    _write_json(run_dir / "tda_metrics.json", {"schema_version": "1.0.0", "metrics": []})
    _write_json(run_dir / "run_config.json", {"seed": seed})


def _create_minimal_p4_run(p4_dir: Path, *, seed: int) -> None:
    run_dir = p4_dir / "p4_factory_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(run_dir / "real_cycles.jsonl", [{"cycle": 1, "success": True}])
    _write_jsonl(run_dir / "twin_predictions.jsonl", [{"real_cycle": 1, "predictions": {"success": True}}])
    _write_jsonl(
        run_dir / "divergence_log.jsonl",
        [make_p4_divergence_log_record(1, seed=seed)],
    )
    _write_json(
        run_dir / "p4_summary.json",
        {
            "mode": "SHADOW",
            "uplift_metrics": {"u2_success_rate_final": 0.85},
            "divergence_analysis": {"divergence_rate": 0.2, "max_divergence_streak": 3},
            "twin_accuracy": {"success_prediction_accuracy": 0.81, "omega_prediction_accuracy": 0.9},
        },
    )
    _write_json(run_dir / "twin_accuracy.json", {"success_prediction_accuracy": 0.81})
    _write_json(run_dir / "run_config.json", {"seed": seed, "telemetry_source": "mock"})


def _write_minimal_manifest(pack_dir: Path, *, file_paths: list[Path]) -> None:
    files = [{"path": path.relative_to(pack_dir).as_posix()} for path in file_paths]
    _write_json(
        pack_dir / "manifest.json",
        {
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "files": files,
            "file_count": len(files),
        },
    )


def test_factory_pack_generates_deterministic_p5_calibration(tmp_path: Path) -> None:
    seed = 4242
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    _create_minimal_p3_run(p3_dir, seed=seed)
    _create_minimal_p4_run(p4_dir, seed=seed)

    pack_dir_1 = tmp_path / "pack_1"
    cal_paths_1 = emit_cal_exp_reports_for_evidence_pack(pack_dir_1, seed=seed)
    _write_minimal_manifest(pack_dir_1, file_paths=list(cal_paths_1.values()))

    status_1 = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=pack_dir_1)
    assert status_1["p5_calibration"] is not None
    assert "cal_exp1" in status_1["p5_calibration"]

    annex_1 = load_cal_exp1_annex(cal_paths_1["cal_exp1_report"])
    assert status_1["p5_calibration"]["cal_exp1"] == annex_1

    pack_dir_2 = tmp_path / "pack_2"
    cal_paths_2 = emit_cal_exp_reports_for_evidence_pack(pack_dir_2, seed=seed)
    _write_minimal_manifest(pack_dir_2, file_paths=list(cal_paths_2.values()))

    status_2 = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=pack_dir_2)
    assert status_1["p5_calibration"] == status_2["p5_calibration"]


def test_factory_reports_build_evidence_pack_manifest_paths_posix_and_deterministic(tmp_path: Path) -> None:
    from scripts.build_first_light_evidence_pack import build_evidence_pack

    seed = 777
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    _create_minimal_p3_run(p3_dir, seed=seed)
    _create_minimal_p4_run(p4_dir, seed=seed)

    cal_source_dir = tmp_path / "cal_source"
    cal_paths = emit_cal_exp_reports_for_evidence_pack(cal_source_dir, seed=seed)

    pack_dir_1 = tmp_path / "pack_1"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir_1,
        cal_exp1_report=cal_paths["cal_exp1_report"],
        cal_exp2_report=cal_paths["cal_exp2_report"],
        cal_exp3_report=cal_paths["cal_exp3_report"],
    )
    manifest_1 = json.loads((pack_dir_1 / "manifest.json").read_text(encoding="utf-8"))
    paths_1 = [entry["path"] for entry in manifest_1.get("files", [])]
    cal_exp_reports_1 = (
        manifest_1.get("governance", {})
        .get("schema_versioned", {})
        .get("cal_exp_reports", {})
    )

    assert "calibration/cal_exp1_report.json" in paths_1
    assert "calibration/cal_exp2_report.json" in paths_1
    assert "calibration/cal_exp3_report.json" in paths_1
    assert all("\\" not in path for path in paths_1)

    assert cal_exp_reports_1["cal_exp2"]["extraction_source"] == "DIRECT_FLAG"
    assert cal_exp_reports_1["cal_exp2"]["mirrored_from"] == "GOVERNANCE_PATH"
    assert cal_exp_reports_1["cal_exp3"]["extraction_source"] == "DIRECT_FLAG"
    assert cal_exp_reports_1["cal_exp3"]["mirrored_from"] == "GOVERNANCE_PATH"

    pack_dir_2 = tmp_path / "pack_2"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir_2,
        cal_exp1_report=cal_paths["cal_exp1_report"],
        cal_exp2_report=cal_paths["cal_exp2_report"],
        cal_exp3_report=cal_paths["cal_exp3_report"],
    )
    manifest_2 = json.loads((pack_dir_2 / "manifest.json").read_text(encoding="utf-8"))
    paths_2 = [entry["path"] for entry in manifest_2.get("files", [])]

    paths_blob_1 = ("\n".join(paths_1)).encode("utf-8")
    paths_blob_2 = ("\n".join(paths_2)).encode("utf-8")
    assert paths_blob_1 == paths_blob_2


def test_cal_exp2_cal_exp3_attachment_provenance_manifest(tmp_path: Path) -> None:
    from scripts.build_first_light_evidence_pack import build_evidence_pack

    seed = 2024
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    _create_minimal_p3_run(p3_dir, seed=seed)
    _create_minimal_p4_run(p4_dir, seed=seed)

    pack_dir = tmp_path / "pack_manifest"
    emit_cal_exp_reports_for_evidence_pack(pack_dir, seed=seed)

    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir,
        cal_exp2_report=None,
        cal_exp3_report=None,
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    cal_exp_reports = (
        manifest.get("governance", {})
        .get("schema_versioned", {})
        .get("cal_exp_reports", {})
    )

    assert cal_exp_reports["cal_exp2"]["extraction_source"] == "MANIFEST"
    assert cal_exp_reports["cal_exp2"]["mirrored_from"] == "GOVERNANCE_PATH"
    assert cal_exp_reports["cal_exp3"]["extraction_source"] == "MANIFEST"
    assert cal_exp_reports["cal_exp3"]["mirrored_from"] == "GOVERNANCE_PATH"


def test_cal_exp2_cal_exp3_attachment_provenance_fallback(tmp_path: Path) -> None:
    from scripts.build_first_light_evidence_pack import build_evidence_pack

    seed = 9001
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    _create_minimal_p3_run(p3_dir, seed=seed)
    _create_minimal_p4_run(p4_dir, seed=seed)

    cal_source_dir = tmp_path / "cal_source"
    cal_paths = emit_cal_exp_reports_for_evidence_pack(cal_source_dir, seed=seed)

    p4_run_dir = p4_dir / "p4_factory_run"
    payload_2 = json.loads(cal_paths["cal_exp2_report"].read_text(encoding="utf-8"))
    payload_3 = json.loads(cal_paths["cal_exp3_report"].read_text(encoding="utf-8"))
    _write_json(p4_run_dir / "cal_exp2_report.json", payload_2)
    _write_json(p4_run_dir / "cal_exp3_report.json", payload_3)

    pack_dir = tmp_path / "pack_fallback"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir,
        cal_exp2_report=None,
        cal_exp3_report=None,
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    cal_exp_reports = (
        manifest.get("governance", {})
        .get("schema_versioned", {})
        .get("cal_exp_reports", {})
    )

    assert cal_exp_reports["cal_exp2"]["extraction_source"] == "FALLBACK"
    assert cal_exp_reports["cal_exp2"]["mirrored_from"] == "LEGACY_PATH"
    assert cal_exp_reports["cal_exp3"]["extraction_source"] == "FALLBACK"
    assert cal_exp_reports["cal_exp3"]["mirrored_from"] == "LEGACY_PATH"
