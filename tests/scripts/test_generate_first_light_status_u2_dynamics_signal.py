"""
Tests for U2 dynamics signal extraction in generate_first_light_status.py.

Validates:
- U2 dynamics block is surfaced under signals.u2_dynamics when present in p4_summary.json.
- Shape is compact and SHADOW-only.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.generate_first_light_status import generate_status


def test_generate_status_includes_u2_dynamics_signal_when_present():
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        evidence_pack_dir = base_dir / "pack"
        evidence_pack_dir.mkdir()

        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
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

        p3_dir = base_dir / "p3"
        p4_dir = base_dir / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        p3_run_dir = p3_dir / "fl_test"
        p3_run_dir.mkdir()
        (p3_run_dir / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )

        p4_run_dir = p4_dir / "p4_test"
        p4_run_dir.mkdir()
        u2_dynamics = {
            "success_rate": 0.9,
            "max_depth": 80,
            "status_light": "GREEN",
            "headline": "stub",
            "window_size": 50,
            "windows": [
                {
                    "window_index": 0,
                    "start_cycle": 1,
                    "end_cycle": 50,
                    "tile": {
                        "status_light": "GREEN",
                        "metrics": {"success_rate": 0.9, "max_depth": 80, "mean_depth": 40.0, "runs": 50},
                    },
                }
            ],
        }
        (p4_run_dir / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "u2_dynamics": u2_dynamics, "metrics": {}}),
            encoding="utf-8",
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        signals = status.get("signals") or {}
        assert "u2_dynamics" in signals
        signal = signals["u2_dynamics"]
        assert signal["mode"] == "SHADOW"
        assert signal["extraction_source"] == "P4_SUMMARY"
        assert signal["success_rate"] == 0.9
        assert signal["max_depth"] == 80
        assert signal["window_size"] == 50
        assert signal["windows"][0]["start_cycle"] == 1
        assert "decomposition_summary" in signal
        assert signal["decomposition_summary"]["state_components"]["window_count"] == 1
        assert "warning" not in signal


def test_generate_status_u2_dynamics_extraction_source_manifest_fallback():
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        evidence_pack_dir = base_dir / "pack"
        evidence_pack_dir.mkdir()

        u2_dynamics_manifest = {
            "status_light": "YELLOW",
            "success_rate": 0.8,
            "max_depth": 120,
            "window_size": 50,
            "windows": [
                {
                    "window_index": 0,
                    "start_cycle": 1,
                    "end_cycle": 50,
                    "tile": {
                        "status_light": "YELLOW",
                        "metrics": {"success_rate": 0.8, "max_depth": 120, "mean_depth": 40.0, "runs": 50},
                    },
                }
            ],
        }

        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {"u2_dynamics": u2_dynamics_manifest},
        }
        (evidence_pack_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        p3_dir = base_dir / "p3"
        p4_dir = base_dir / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        (p3_dir / "fl_test").mkdir()
        (p3_dir / "fl_test" / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )

        (p4_dir / "p4_test").mkdir()
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        signal = (status.get("signals") or {})["u2_dynamics"]
        assert signal["extraction_source"] == "MANIFEST"
        assert signal["status_light"] == "YELLOW"
        assert signal["warning"] == "window_count=1 driver=DRIVER_U2_SUCCESS_RATE_LOW"


def test_generate_status_u2_dynamics_extraction_source_evidence_json_fallback():
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        evidence_pack_dir = base_dir / "pack"
        evidence_pack_dir.mkdir()

        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
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

        u2_dynamics_evidence = {
            "status_light": "GREEN",
            "success_rate": 0.95,
            "max_depth": 90,
            "window_size": 50,
            "windows": [],
        }
        (evidence_pack_dir / "evidence.json").write_text(
            json.dumps({"governance": {"u2_dynamics": u2_dynamics_evidence}}),
            encoding="utf-8",
        )

        p3_dir = base_dir / "p3"
        p4_dir = base_dir / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        (p3_dir / "fl_test").mkdir()
        (p3_dir / "fl_test" / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )

        (p4_dir / "p4_test").mkdir()
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        signal = (status.get("signals") or {})["u2_dynamics"]
        assert signal["extraction_source"] == "EVIDENCE_JSON"
        assert signal["status_light"] == "GREEN"
        assert "warning" not in signal


def test_generate_status_u2_dynamics_extraction_source_missing_emits_stub():
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        evidence_pack_dir = base_dir / "pack"
        evidence_pack_dir.mkdir()

        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
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

        p3_dir = base_dir / "p3"
        p4_dir = base_dir / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        signal = (status.get("signals") or {})["u2_dynamics"]
        assert signal["extraction_source"] == "MISSING"
