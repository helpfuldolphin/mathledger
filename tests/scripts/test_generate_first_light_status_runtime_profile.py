"""
Tests for runtime profile snapshot ingestion in generate_first_light_status.py.

Validates:
- Runtime profile snapshot extraction from manifest
- Signal structure and fields
- SHADOW-mode behavior (no gating)
- Missing snapshot handling
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.generate_first_light_status import generate_status


def test_runtime_profile_snapshot_extracted_from_manifest():
    """Test that runtime profile snapshot is extracted from manifest and added to signals."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with runtime profile snapshot
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "runtime_profile": {
                    "schema_version": "1.0.0",
                    "profile": "prod-hardened",
                    "status_light": "GREEN",
                    "profile_stability": 0.95,
                    "no_run_rate": 0.05,
                }
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Create minimal P3 and P4 directories
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        # Create minimal P3 run directory
        p3_run_dir = p3_dir / "fl_test"
        p3_run_dir.mkdir()
        (p3_run_dir / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )

        # Create minimal P4 run directory
        p4_run_dir = p4_dir / "p4_test"
        p4_run_dir.mkdir()
        (p4_run_dir / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}}), encoding="utf-8"
        )
        (p4_run_dir / "run_config.json").write_text(
            json.dumps({"telemetry_source": "mock"}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify runtime profile signal is present
        assert status["signals"] is not None
        assert "runtime_profile" in status["signals"]
        runtime_profile = status["signals"]["runtime_profile"]
        assert runtime_profile["profile"] == "prod-hardened"
        assert runtime_profile["status_light"] == "GREEN"
        assert runtime_profile["profile_stability"] == 0.95
        assert runtime_profile["no_run_rate"] == 0.05


def test_runtime_profile_red_status_generates_warning():
    """Test that RED runtime profile status generates an advisory warning."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with RED runtime profile snapshot
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "runtime_profile": {
                    "schema_version": "1.0.0",
                    "profile": "prod-hardened",
                    "status_light": "RED",
                    "profile_stability": 0.65,
                    "no_run_rate": 0.25,
                }
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Create minimal P3 and P4 directories
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        # Create minimal P3 run directory
        p3_run_dir = p3_dir / "fl_test"
        p3_run_dir.mkdir()
        (p3_run_dir / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )

        # Create minimal P4 run directory
        p4_run_dir = p4_dir / "p4_test"
        p4_run_dir.mkdir()
        (p4_run_dir / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}}), encoding="utf-8"
        )
        (p4_run_dir / "run_config.json").write_text(
            json.dumps({"telemetry_source": "mock"}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify warning is generated
        assert status["warnings"] is not None
        assert any("Runtime profile" in w and "RED" in w for w in status["warnings"])

        # Verify signal is still present (no blocking)
        assert status["signals"] is not None
        assert "runtime_profile" in status["signals"]


def test_missing_runtime_profile_snapshot_not_an_error():
    """Test that missing runtime profile snapshot does not cause errors."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest without runtime profile snapshot
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "file_count": 0,
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Create minimal P3 and P4 directories
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        p3_dir.mkdir()
        p4_dir.mkdir()

        # Create minimal P3 run directory with all required artifacts
        p3_run_dir = p3_dir / "fl_test"
        p3_run_dir.mkdir()
        (p3_run_dir / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
        (p3_run_dir / "stability_report.json").write_text(
            json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
        )
        (p3_run_dir / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
        (p3_run_dir / "metrics_windows.json").write_text("{}", encoding="utf-8")
        (p3_run_dir / "tda_metrics.json").write_text("{}", encoding="utf-8")
        (p3_run_dir / "run_config.json").write_text("{}", encoding="utf-8")

        # Create minimal P4 run directory with all required artifacts
        p4_run_dir = p4_dir / "p4_test"
        p4_run_dir.mkdir()
        (p4_run_dir / "real_cycles.jsonl").write_text("", encoding="utf-8")
        (p4_run_dir / "twin_predictions.jsonl").write_text("", encoding="utf-8")
        (p4_run_dir / "divergence_log.jsonl").write_text("", encoding="utf-8")
        (p4_run_dir / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}}), encoding="utf-8"
        )
        (p4_run_dir / "twin_accuracy.json").write_text("{}", encoding="utf-8")
        (p4_run_dir / "run_config.json").write_text(
            json.dumps({"telemetry_source": "mock"}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify status generation succeeds (evidence pack should be OK)
        assert status["evidence_pack_ok"] is True

        # Verify runtime profile signal is not present (missing snapshot is not an error)
        if status["signals"]:
            assert "runtime_profile" not in status["signals"]
        else:
            # signals can be None if empty, which is fine
            assert status["signals"] is None or "runtime_profile" not in status["signals"]

