import json
import runpy
from pathlib import Path

from scripts.generate_first_light_status import generate_status

build_module = runpy.run_path("scripts/build_first_light_evidence_pack.py")
build_evidence_pack = build_module["build_evidence_pack"]


def test_build_pack_generates_plots_and_manifest_entries(tmp_path):
    p3_dir, p4_dir, output_dir = _build_pack(tmp_path)

    viz_dir = output_dir / "visualizations"
    expected_files = {
        viz_dir / "delta_p_vs_cycles.svg",
        viz_dir / "rsi_vs_cycles.svg",
        viz_dir / "omega_occupancy_vs_cycles.svg",
    }
    for path in expected_files:
        assert path.exists(), f"{path} was not generated"
        assert path.stat().st_size > 0

    manifest = json.loads((output_dir / "manifest.json").read_text())
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    expected_manifest_paths = {
        "visualizations/delta_p_vs_cycles.svg",
        "visualizations/rsi_vs_cycles.svg",
        "visualizations/omega_occupancy_vs_cycles.svg",
    }
    assert expected_manifest_paths <= manifest_paths
    for rel_path in expected_manifest_paths:
        assert (output_dir / Path(rel_path)).exists()


def test_status_reports_plots_present(tmp_path):
    p3_dir, p4_dir, output_dir = _build_pack(tmp_path)

    status = generate_status(p3_dir, p4_dir, output_dir)
    artifacts = status["artifacts"]
    assert artifacts["plots_present"] is True
    assert artifacts["plots_dir"] == str(output_dir / "visualizations")


def _build_pack(tmp_path: Path):
    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    p3_run = _create_p3_run(p3_dir)
    p4_run = _create_p4_run(p4_dir)
    assert p3_run.exists()
    assert p4_run.exists()

    output_dir = tmp_path / "evidence_pack"
    build_evidence_pack(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        out_dir=output_dir,
    )
    return p3_dir, p4_dir, output_dir


def _create_p3_run(base_dir: Path) -> Path:
    run_dir = base_dir / "fl_test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "synthetic_raw.jsonl",
        [
            {"cycle": 1, "runner": {"success": True}, "usla_state": {"in_omega": True}},
            {"cycle": 2, "runner": {"success": False}, "usla_state": {"in_omega": False}},
        ],
    )
    _write_json(
        run_dir / "stability_report.json",
        {
            "metrics": {
                "success_rate": 0.9,
                "rsi": {"mean": 0.8},
                "omega_occupancy": 0.7,
            },
            "red_flag_summary": {"total": 1},
        },
    )
    _write_json(run_dir / "red_flag_matrix.json", {"flags": []})
    _write_json(
        run_dir / "metrics_windows.json",
        {
            "windows": [
                {"window_index": 0, "metrics": {"mean_rsi": 0.6, "omega_occupancy": 0.5}},
                {"window_index": 1, "metrics": {"mean_rsi": 0.7, "omega_occupancy": 0.6}},
            ]
        },
    )
    _write_json(run_dir / "tda_metrics.json", {"metrics": []})
    _write_json(run_dir / "run_config.json", {"seed": 42})
    return run_dir


def _create_p4_run(base_dir: Path) -> Path:
    run_dir = base_dir / "p4_test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "real_cycles.jsonl",
        [
            {"cycle": 1, "success": True},
            {"cycle": 2, "success": False},
        ],
    )
    _write_jsonl(
        run_dir / "twin_predictions.jsonl",
        [{"real_cycle": 1, "predictions": {"success": True}}],
    )
    _write_jsonl(
        run_dir / "divergence_log.jsonl",
        [
            {"cycle": 1, "delta_p": 0.02},
            {"cycle": 2, "delta_p": 0.01},
        ],
    )
    _write_json(
        run_dir / "p4_summary.json",
        {
            "uplift_metrics": {"u2_success_rate_final": 0.9},
            "divergence_analysis": {"divergence_rate": 0.5, "max_divergence_streak": 1},
            "twin_accuracy": {"success_prediction_accuracy": 0.8},
        },
    )
    _write_json(run_dir / "twin_accuracy.json", {"success_prediction_accuracy": 0.8})
    _write_json(run_dir / "run_config.json", {"seed": 42})
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows))
