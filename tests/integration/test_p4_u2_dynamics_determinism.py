import argparse
import json
from pathlib import Path

import pytest

from scripts.generate_first_light_status import generate_status
from scripts.usla_first_light_p4_harness import run_harness


@pytest.mark.integration
@pytest.mark.determinism
def test_p4_harness_u2_dynamics_windows_deterministic_across_runs(tmp_path):
    """Run P4 harness twice and ensure U2 dynamics windows are cross-run stable."""

    def run_once(base_dir: Path) -> Path:
        args = argparse.Namespace(
            cycles=60,
            seed=123,
            output_dir=str(base_dir),
            slice="arithmetic_simple",
            runner_type="u2",
            tau_0=0.20,
            dry_run=False,
            telemetry_adapter="mock",
            adapter_config=None,
            synthetic_config=None,
            prod_config=None,
            p4_evidence_pack=None,
            emit_p5_diagnostic=False,
            emit_rtts_validation=False,
            twin_lr_H=None,
            twin_lr_rho=None,
            twin_lr_tau=None,
            twin_lr_beta=None,
        )
        exit_code = run_harness(args)
        assert exit_code == 0
        run_dirs = sorted(base_dir.glob("p4_*"))
        assert run_dirs
        return run_dirs[-1]

    run_dir1 = run_once(tmp_path / "p4_run1")
    run_dir2 = run_once(tmp_path / "p4_run2")

    summary1 = json.loads((run_dir1 / "p4_summary.json").read_text(encoding="utf-8"))
    summary2 = json.loads((run_dir2 / "p4_summary.json").read_text(encoding="utf-8"))

    assert summary1["u2_dynamics"]["window_size"] == 50
    assert summary2["u2_dynamics"]["window_size"] == 50

    windows1 = summary1["u2_dynamics"]["windows"]
    windows2 = summary2["u2_dynamics"]["windows"]

    def strip_timestamps(obj: object) -> object:
        if isinstance(obj, dict):
            return {
                k: strip_timestamps(v)
                for k, v in obj.items()
                if k not in {"timestamp", "start_time", "end_time"}
            }
        if isinstance(obj, list):
            return [strip_timestamps(v) for v in obj]
        return obj

    assert strip_timestamps(windows1) == strip_timestamps(windows2)


@pytest.mark.integration
@pytest.mark.determinism
def test_p4_harness_status_u2_dynamics_decomposition_present_and_deterministic(tmp_path):
    """Run P4 harness, generate status twice, and ensure decomposition is stable."""

    p4_dir = tmp_path / "p4"
    p3_dir = tmp_path / "p3"
    evidence_pack_dir = tmp_path / "pack"
    p4_dir.mkdir()
    p3_dir.mkdir()
    evidence_pack_dir.mkdir()

    # Minimal evidence pack manifest (SHADOW-only; no plots, no snapshot).
    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "file_count": 0,
        "files": [],
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "governance": {},
    }
    (evidence_pack_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # Minimal P3 harness stub (enough to satisfy check_p3_harness()).
    p3_run = p3_dir / "fl_test"
    p3_run.mkdir()
    (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
    (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
    (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
    (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
    (p3_run / "stability_report.json").write_text(
        json.dumps({"metrics": {"success_rate": 0.85}, "pathology": "none"}),
        encoding="utf-8",
    )

    # Run P4 harness (short).
    args = argparse.Namespace(
        cycles=60,
        seed=123,
        output_dir=str(p4_dir),
        slice="arithmetic_simple",
        runner_type="u2",
        tau_0=0.20,
        dry_run=False,
        telemetry_adapter="mock",
        adapter_config=None,
        synthetic_config=None,
        prod_config=None,
        p4_evidence_pack=None,
        emit_p5_diagnostic=False,
        emit_rtts_validation=False,
        twin_lr_H=None,
        twin_lr_rho=None,
        twin_lr_tau=None,
        twin_lr_beta=None,
    )
    assert run_harness(args) == 0

    status1 = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=evidence_pack_dir)
    status2 = generate_status(p3_dir=p3_dir, p4_dir=p4_dir, evidence_pack_dir=evidence_pack_dir)

    sig1 = (status1.get("signals") or {}).get("u2_dynamics")
    sig2 = (status2.get("signals") or {}).get("u2_dynamics")
    assert isinstance(sig1, dict)
    assert isinstance(sig2, dict)
    assert sig1["extraction_source"] == "P4_SUMMARY"
    assert sig2["extraction_source"] == "P4_SUMMARY"
    assert "decomposition_summary" in sig1
    assert sig1["decomposition_summary"] == sig2["decomposition_summary"]

    if "warning" in sig1:
        assert "\n" not in sig1["warning"]
        assert "window_count=" in sig1["warning"]
        assert "driver=" in sig1["warning"]
        driver = sig1["warning"].split("driver=", 1)[-1].strip()
        assert driver.startswith("DRIVER_U2_")
        assert " " not in driver
