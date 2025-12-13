import json
from pathlib import Path

import scripts.build_first_light_evidence_pack as evidence_builder
from scripts.build_ledger_guard_summary import (
    build_summary_from_chain,
    write_summary,
)
from scripts.generate_first_light_status import generate_status


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_p3_run(base: Path) -> Path:
    run_dir = base / "fl_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "synthetic_raw.jsonl", "{}\n")
    _write_json(
        run_dir / "stability_report.json",
        {
            "metrics": {
                "success_rate": 1.0,
                "omega": {"occupancy_rate": 0.99},
                "rsi": {"mean": 0.5},
                "hard_mode": {"ok_rate": 1.0},
            },
            "criteria_evaluation": {"all_passed": True},
            "red_flag_summary": {"total_flags": 0, "hypothetical_abort": False},
        },
    )
    _write_json(run_dir / "red_flag_matrix.json", {})
    _write_json(run_dir / "metrics_windows.json", {"windows": []})
    _write_json(run_dir / "tda_metrics.json", {})
    _write_json(run_dir / "run_config.json", {"mode": "SHADOW"})
    return run_dir


def _make_p4_run(base: Path) -> Path:
    run_dir = base / "p4_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "real_cycles.jsonl", "{}\n")
    _write_text(run_dir / "twin_predictions.jsonl", "{}\n")
    _write_text(run_dir / "divergence_log.jsonl", "{}\n")
    _write_json(
        run_dir / "p4_summary.json",
        {
            "mode": "SHADOW",
            "uplift_metrics": {"u2_success_rate_final": 0.5},
            "divergence_analysis": {"divergence_rate": 0.2, "max_divergence_streak": 2},
            "twin_accuracy": {"success_prediction_accuracy": 0.8, "omega_prediction_accuracy": 0.75},
        },
    )
    _write_json(run_dir / "twin_accuracy.json", {})
    _write_json(run_dir / "run_config.json", {"mode": "SHADOW"})
    return run_dir


def test_ledger_guard_summary_integrates_with_status(tmp_path, monkeypatch):
    # Patch plotting helpers to avoid heavy dependencies
    for attr in ("plot_delta_p", "plot_rsi", "plot_omega_occupancy"):
        monkeypatch.setattr(evidence_builder, attr, lambda source, dest: _write_text(Path(dest), "<svg></svg>"))

    p3_root = tmp_path / "p3"
    p4_root = tmp_path / "p4"
    _make_p3_run(p3_root)
    _make_p4_run(p4_root)

    headers = [
        {
            "height": 0,
            "prev_hash": "0" * 64,
            "root_hash": "1" * 64,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "height": 0,
            "prev_hash": "1" * 64,
            "root_hash": "2" * 64,
            "timestamp": "2024-01-01T00:01:00Z",
        },
    ]
    summary = build_summary_from_chain(headers)
    summary.pop("schema_version", None)
    summary_path = tmp_path / "ledger_guard_summary.json"
    write_summary(summary, summary_path)

    pack_dir = tmp_path / "pack"
    evidence_builder.build_evidence_pack(
        p3_dir=p3_root,
        p4_dir=p4_root,
        out_dir=pack_dir,
        plots_dir=tmp_path / "plots",
        ledger_guard_summary=summary_path,
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text())
    ledger_guard_manifest = manifest["governance"]["schema_versioned"]["ledger_guard_summary"]
    assert ledger_guard_manifest["mode"] == "SHADOW"
    assert ledger_guard_manifest["schema_version"] == "1.0.0"
    assert ledger_guard_manifest["status_light"] == summary["status_light"]
    copied_summary = json.loads((pack_dir / "governance" / "ledger_guard_summary.json").read_text())
    assert copied_summary["schema_version"] == "1.0.0"

    status = generate_status(
        p3_dir=p3_root,
        p4_dir=p4_root,
        evidence_pack_dir=pack_dir,
        pipeline="local",
    )

    assert status["signals"]["ledger_guard"]["schema_version"] == ledger_guard_manifest["schema_version"]
    assert status["signals"]["ledger_guard"]["sha256"] == ledger_guard_manifest["sha256"]
    assert status["signals"]["ledger_guard"]["status_light"] == summary["status_light"]
    assert status["signals"]["ledger_guard"]["violation_counts"] == summary["violation_counts"]
    assert status["signals"]["ledger_guard"]["headline"] == (
        f"Ledger guard: status {summary['status_light']}, violation_counts {summary['violation_counts']}"
    )
    assert status["signals"]["ledger_guard"]["extraction_source"] == "MANIFEST_REFERENCE"
    assert not any("Ledger guard summary integrity" in warning for warning in status["warnings"])


def test_ledger_guard_sha_mismatch_emits_single_reason_code_warning(tmp_path, monkeypatch):
    for attr in ("plot_delta_p", "plot_rsi", "plot_omega_occupancy"):
        monkeypatch.setattr(
            evidence_builder,
            attr,
            lambda source, dest: _write_text(Path(dest), "<svg></svg>"),
        )

    p3_root = tmp_path / "p3"
    p4_root = tmp_path / "p4"
    _make_p3_run(p3_root)
    _make_p4_run(p4_root)

    summary = build_summary_from_chain(
        [
            {
                "height": 0,
                "prev_hash": "0" * 64,
                "root_hash": "1" * 64,
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "height": 0,
                "prev_hash": "1" * 64,
                "root_hash": "2" * 64,
                "timestamp": "2024-01-01T00:01:00Z",
            },
        ]
    )
    summary_path = tmp_path / "ledger_guard_summary.json"
    write_summary(summary, summary_path)

    pack_dir = tmp_path / "pack"
    evidence_builder.build_evidence_pack(
        p3_dir=p3_root,
        p4_dir=p4_root,
        out_dir=pack_dir,
        plots_dir=tmp_path / "plots",
        ledger_guard_summary=summary_path,
    )

    tampered_path = pack_dir / "governance" / "ledger_guard_summary.json"
    tampered_payload = json.loads(tampered_path.read_text())
    tampered_payload["headline"] = "tampered"
    tampered_path.write_text(json.dumps(tampered_payload, indent=2))

    status = generate_status(
        p3_dir=p3_root,
        p4_dir=p4_root,
        evidence_pack_dir=pack_dir,
        pipeline="local",
    )

    ledger_guard_warnings = [
        warning
        for warning in status["warnings"]
        if warning.startswith("Ledger guard summary integrity")
    ]
    assert ledger_guard_warnings == ["Ledger guard summary integrity [SHA256_MISMATCH]"]
    assert "\n" not in ledger_guard_warnings[0]
