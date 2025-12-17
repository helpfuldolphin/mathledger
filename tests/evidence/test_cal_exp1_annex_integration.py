import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_first_light_evidence_pack import build_evidence_pack
from scripts.first_light_cal_exp1_warm_start import run_cal_exp1
from scripts.generate_first_light_status import generate_status


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows))


def _create_minimal_runs(base_dir: Path) -> tuple[Path, Path]:
    p3_dir = base_dir / "p3"
    p4_dir = base_dir / "p4"
    p3_run = p3_dir / "fl_dummy_run"
    p4_run = p4_dir / "p4_dummy_run"
    p3_run.mkdir(parents=True, exist_ok=True)
    p4_run.mkdir(parents=True, exist_ok=True)

    _write_jsonl(p3_run / "synthetic_raw.jsonl", [{"cycle": 1, "success": True}])
    _write_json(
        p3_run / "stability_report.json",
        {
            "metrics": {
                "success_rate": 0.8,
                "rsi": {"mean": 0.7},
                "omega_occupancy": 0.9,
            },
            "red_flag_summary": {"total": 0},
        },
    )
    _write_json(p3_run / "red_flag_matrix.json", {"flags": []})
    _write_json(
        p3_run / "metrics_windows.json",
        {"windows": [{"window_index": 0, "metrics": {"mean_rsi": 0.7}}]},
    )
    _write_json(p3_run / "tda_metrics.json", {"metrics": []})
    _write_json(p3_run / "run_config.json", {"seed": 1})

    _write_jsonl(p4_run / "real_cycles.jsonl", [{"cycle": 1, "success": True}])
    _write_jsonl(
        p4_run / "twin_predictions.jsonl",
        [{"real_cycle": 1, "predictions": {"success": True}}],
    )
    _write_jsonl(
        p4_run / "divergence_log.jsonl",
        [{"cycle": 1, "delta_p": 0.02, "delta_p_percent": 2.0}],
    )
    _write_json(
        p4_run / "p4_summary.json",
        {
            "mode": "SHADOW",
            "uplift_metrics": {"u2_success_rate_final": 0.85},
            "divergence_analysis": {"divergence_rate": 0.2, "max_divergence_streak": 3},
            "twin_accuracy": {"success_prediction_accuracy": 0.7},
        },
    )
    _write_json(p4_run / "twin_accuracy.json", {"success_prediction_accuracy": 0.7})
    _write_json(p4_run / "run_config.json", {"seed": 2})
    return p3_dir, p4_dir


def _run_cal_exp1_report(tmp_path: Path) -> Path:
    out_dir = tmp_path / "cal_run"
    args = SimpleNamespace(
        adapter="mock",
        cycles=40,
        learning_rate=0.1,
        seed=5,
        output_dir=out_dir,
        adapter_config=None,
        runner_type="u2",
        slice="arithmetic_simple",
    )
    report_path = run_cal_exp1(args)
    report = json.loads(report_path.read_text())
    if report.get("windows"):
        report["windows"][0]["divergence_rate"] = 0.1
        report["windows"][-1]["divergence_rate"] = 0.9
        report_path.write_text(json.dumps(report))
    return report_path


def _write_stalled_cal_exp1_report(report_path: Path) -> None:
    _write_json(
        report_path,
        {
            "schema_version": "1.0.0",
            "windows": [
                {
                    "divergence_rate": 0.1,
                    "delta_bias": 0.01,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
                {
                    "divergence_rate": 0.9,
                    "delta_bias": 0.02,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
            ],
            "summary": {
                "final_divergence_rate": 0.9,
            },
        },
    )


def _load_manifest(pack_dir: Path) -> dict:
    return json.loads((pack_dir / "manifest.json").read_text())


def test_cal_exp1_annex_in_manifest_and_status(tmp_path: Path) -> None:
    p3_dir, p4_dir = _create_minimal_runs(tmp_path)
    report_path = _run_cal_exp1_report(tmp_path)

    pack_dir_1 = tmp_path / "pack_one"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir_1,
        cal_exp1_report=report_path,
    )
    manifest_1 = _load_manifest(pack_dir_1)
    annex_1 = (
        manifest_1.get("governance", {})
        .get("p5_calibration", {})
        .get("cal_exp1")
    )
    assert annex_1 is not None
    assert "final_divergence_rate" in annex_1
    assert "learning_occurring" in annex_1
    assert "learning_occurring_legacy" in annex_1
    assert annex_1["learning_occurring_basis"] == "INSUFFICIENT_WINDOWS"
    assert annex_1["metric_definitions_ref"].startswith(
        "docs/system_law/calibration/METRIC_DEFINITIONS.md@v"
    )
    cal_path = pack_dir_1 / "calibration" / "cal_exp1_report.json"
    assert cal_path.exists()

    status_1 = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_dir_1,
    )
    assert status_1["p5_calibration"]["cal_exp1"] == annex_1
    assert any(
        "CAL-EXP-1 learning check inconclusive" in warning
        for warning in status_1.get("warnings", [])
    )
    assert any(
        "state_delta_p_mean=" in warning for warning in status_1.get("warnings", [])
    )

    pack_dir_2 = tmp_path / "pack_two"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir_2,
        cal_exp1_report=report_path,
    )
    manifest_2 = _load_manifest(pack_dir_2)
    annex_2 = (
        manifest_2.get("governance", {})
        .get("p5_calibration", {})
        .get("cal_exp1")
    )
    assert annex_1 == annex_2

    status_2 = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_dir_2,
    )
    assert status_1["p5_calibration"] == status_2["p5_calibration"]


@pytest.mark.integration
def test_cal_exp1_autodiscovery_without_flag(tmp_path: Path) -> None:
    p3_dir, p4_dir = _create_minimal_runs(tmp_path)

    # Place report where real runs commonly write it:
    # <run_dir>/calibration/cal_exp1_report.json
    report_path = p3_dir / "fl_dummy_run" / "calibration" / "cal_exp1_report.json"
    _write_stalled_cal_exp1_report(report_path)

    pack_dir = tmp_path / "pack_autodiscover"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir,
        cal_exp1_report=None,
    )

    manifest = _load_manifest(pack_dir)
    p5_calibration = manifest.get("governance", {}).get("p5_calibration", {})
    annex = p5_calibration.get("cal_exp1")
    assert annex is not None
    assert p5_calibration.get("cal_exp1_report_path") == "calibration/cal_exp1_report.json"

    status = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_dir,
    )
    assert status.get("p5_calibration", {}).get("cal_exp1") == annex
    assert status.get("p5_calibration", {}).get("cal_exp1_report_path") == "calibration/cal_exp1_report.json"
    assert any(
        "CAL-EXP-1 indicates no learning" in warning
        for warning in status.get("warnings", [])
    )
    assert any(
        "basis=LEGACY_DIVERGENCE_HEURISTIC" in warning
        for warning in status.get("warnings", [])
    )


@pytest.mark.integration
def test_cal_exp1_divergence_flat_but_state_delta_p_improves_no_warning(tmp_path: Path) -> None:
    p3_dir, p4_dir = _create_minimal_runs(tmp_path)

    report_path = tmp_path / "cal_run" / "cal_exp1_report.json"
    _write_json(
        report_path,
        {
            "schema_version": "1.0.0",
            "windows": [
                {
                    "divergence_rate": 0.9,
                    "mean_delta_p": 0.20,
                    "delta_bias": 0.01,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
                {
                    "divergence_rate": 0.9,
                    "mean_delta_p": 0.10,
                    "delta_bias": 0.01,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
            ],
            "summary": {
                "final_divergence_rate": 0.9,
            },
        },
    )

    pack_dir = tmp_path / "pack_state_improves"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=pack_dir,
        cal_exp1_report=report_path,
    )

    status = generate_status(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=pack_dir,
    )

    assert status.get("p5_calibration", {}).get("cal_exp1") is not None
    assert not any(
        "CAL-EXP-1 indicates no learning" in warning for warning in status.get("warnings", [])
    )


@pytest.mark.integration
def test_cal_exp1_warning_basis_phrasing_never_conflates_metrics(tmp_path: Path) -> None:
    p3_dir, p4_dir = _create_minimal_runs(tmp_path)

    report_mean_trend = tmp_path / "cal_mean_trend" / "cal_exp1_report.json"
    _write_json(
        report_mean_trend,
        {
            "schema_version": "1.0.0",
            "windows": [
                {
                    "divergence_rate": 0.9,
                    "mean_delta_p": 0.10,
                    "delta_bias": 0.01,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
                {
                    "divergence_rate": 0.9,
                    "mean_delta_p": 0.20,
                    "delta_bias": 0.01,
                    "delta_variance": 0.0,
                    "phase_lag_xcorr": 0.0,
                    "pattern_tag": "DRIFT",
                },
            ],
            "summary": {
                "final_divergence_rate": 0.9,
            },
        },
    )
    pack_mean_trend = tmp_path / "pack_mean_trend"
    build_evidence_pack(p3_dir=p3_dir, p4_dir=p4_dir, out_dir=pack_mean_trend, cal_exp1_report=report_mean_trend)
    status_mean_trend = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=pack_mean_trend)
    warning_mean_trend = next(
        w for w in status_mean_trend.get("warnings", []) if "CAL-EXP-1" in w
    )
    assert "basis=MEAN_DELTA_P_TREND" in warning_mean_trend
    assert "state_delta_p_mean=" in warning_mean_trend
    assert "final_divergence_rate" not in warning_mean_trend
    assert "divergence_rate" not in warning_mean_trend

    report_legacy = tmp_path / "cal_legacy" / "cal_exp1_report.json"
    _write_stalled_cal_exp1_report(report_legacy)
    pack_legacy = tmp_path / "pack_legacy"
    build_evidence_pack(p3_dir=p3_dir, p4_dir=p4_dir, out_dir=pack_legacy, cal_exp1_report=report_legacy)
    status_legacy = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=pack_legacy)
    warning_legacy = next(
        w for w in status_legacy.get("warnings", []) if "CAL-EXP-1" in w
    )
    assert "basis=LEGACY_DIVERGENCE_HEURISTIC" in warning_legacy
    assert "final_divergence_rate" in warning_legacy
    assert "state_delta_p_mean" not in warning_legacy
    assert "mean_delta_p" not in warning_legacy
