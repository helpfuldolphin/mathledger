"""
Tests for release attitude strip extraction in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings
- Warning hygiene: one warning line only when advisory_warning present
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
    p3_base = tmp_path / "p3"
    p3_base.mkdir()
    p3_run = p3_base / "fl_test"
    p3_run.mkdir()
    
    (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
    (p3_run / "stability_report.json").write_text(
        json.dumps({"metrics": {"success_rate": 0.85}}), encoding="utf-8"
    )
    (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
    (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
    (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
    (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
    
    return p3_base


@pytest.fixture
def p4_dir(tmp_path: Path) -> Path:
    """Create a minimal P4 directory structure."""
    p4_base = tmp_path / "p4"
    p4_base.mkdir()
    p4_run = p4_base / "p4_test"
    p4_run.mkdir()
    
    (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
    (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
    (p4_run / "calibration_report.json").write_text("{}", encoding="utf-8")
    (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
    (p4_run / "p4_summary.json").write_text(
        json.dumps({"mode": "SHADOW"}), encoding="utf-8"
    )
    (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
    (p4_run / "run_config.json").write_text("{}", encoding="utf-8")
    
    return p4_base


def test_release_attitude_strip_extracted_from_manifest(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that release attitude strip is extracted from manifest (manifest-first)."""
    # Create release attitude strip in manifest
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    release_attitude_strip = {
        "schema_version": "1.0.0",
        "experiments": [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
            {
                "cal_id": "CAL-EXP-2",
                "global_band": "YELLOW",
                "system_alignment": "PARTIAL",
                "release_ready": False,
                "status_light": "🟡",
            },
        ],
        "summary": {
            "total_count": 2,
            "release_ready_count": 1,
            "release_ready_ratio": 0.5,
            "trend": "DEGRADING",
        },
    }
    
    manifest["governance"]["release_attitude_strip"] = release_attitude_strip
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signal is present
    assert "signals" in status
    assert status["signals"] is not None
    assert "release_attitude_strip" in status["signals"]
    
    signal = status["signals"]["release_attitude_strip"]
    assert signal["mode"] == "SHADOW"
    assert signal["schema_version"] == "1.0.0"
    assert signal["total_count"] == 2
    assert signal["release_ready_ratio"] == 0.5
    assert signal["trend"] == "DEGRADING"
    assert signal["first_status_light"] == "🟢"
    assert signal["last_status_light"] == "🟡"
    assert signal["advisory_warning"] is not None
    # Warning should be ASCII-safe (emoji replaced)
    assert "DEGRADING" in signal["advisory_warning"]


def test_release_attitude_strip_fallback_to_evidence_json(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that release attitude strip falls back to evidence.json if not in manifest."""
    # Create evidence.json with release attitude strip
    evidence_json_path = evidence_pack_dir / "evidence.json"
    evidence_data = {
        "governance": {
            "release_attitude_strip": {
                "schema_version": "1.0.0",
                "experiments": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "global_band": "GREEN",
                        "system_alignment": "ALIGNED",
                        "release_ready": True,
                        "status_light": "🟢",
                    },
                ],
                "summary": {
                    "total_count": 1,
                    "release_ready_count": 1,
                    "release_ready_ratio": 1.0,
                    "trend": "STABLE",
                },
            },
        },
    }
    evidence_json_path.write_text(json.dumps(evidence_data, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signal is present
    assert "signals" in status
    assert status["signals"] is not None
    assert "release_attitude_strip" in status["signals"]
    
    signal = status["signals"]["release_attitude_strip"]
    assert signal["total_count"] == 1
    assert signal["trend"] == "STABLE"


def test_release_attitude_strip_missing_safe(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that missing release attitude strip does not cause errors."""
    # Generate status without release attitude strip
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signal is not present (but status generation succeeds)
    assert "signals" in status
    if status["signals"]:
        assert "release_attitude_strip" not in status["signals"]


def test_release_attitude_strip_warning_hygiene_degrading(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that one warning line is emitted when trend is DEGRADING."""
    # Create release attitude strip with DEGRADING trend
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    release_attitude_strip = {
        "schema_version": "1.0.0",
        "experiments": [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
            {
                "cal_id": "CAL-EXP-2",
                "global_band": "RED",
                "system_alignment": "MISALIGNED",
                "release_ready": False,
                "status_light": "🔴",
            },
        ],
        "summary": {
            "total_count": 2,
            "release_ready_count": 1,
            "release_ready_ratio": 0.5,
            "trend": "DEGRADING",
        },
    }
    
    manifest["governance"]["release_attitude_strip"] = release_attitude_strip
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify one warning line is present
    assert "warnings" in status
    assert status["warnings"] is not None
    
    # Count warnings related to release attitude strip
    strip_warnings = [w for w in status["warnings"] if "DEGRADING" in str(w)]
    assert len(strip_warnings) == 1
    
    # Verify warning contains the advisory message (ASCII-safe, no emoji)
    warning_text = strip_warnings[0]
    assert "Release attitude trend is DEGRADING" in warning_text
    # Ensure no emoji in warning (should be replaced with [GREEN]/[YELLOW]/[RED])
    assert "🟢" not in warning_text
    assert "🟡" not in warning_text
    assert "🔴" not in warning_text


def test_release_attitude_strip_no_warning_stable(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that no warning is emitted when trend is STABLE."""
    # Create release attitude strip with STABLE trend
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    release_attitude_strip = {
        "schema_version": "1.0.0",
        "experiments": [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
            {
                "cal_id": "CAL-EXP-2",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
        ],
        "summary": {
            "total_count": 2,
            "release_ready_count": 2,
            "release_ready_ratio": 1.0,
            "trend": "STABLE",
        },
    }
    
    manifest["governance"]["release_attitude_strip"] = release_attitude_strip
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify no warning related to release attitude strip
    assert "warnings" in status
    if status["warnings"]:
        strip_warnings = [w for w in status["warnings"] if "DEGRADING" in str(w)]
        assert len(strip_warnings) == 0


def test_release_attitude_strip_deterministic(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that signal extraction is deterministic."""
    # Create release attitude strip
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    release_attitude_strip = {
        "schema_version": "1.0.0",
        "experiments": [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
        ],
        "summary": {
            "total_count": 1,
            "release_ready_count": 1,
            "release_ready_ratio": 1.0,
            "trend": "STABLE",
        },
    }
    
    manifest["governance"]["release_attitude_strip"] = release_attitude_strip
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Generate status twice
    status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signals are identical
    signals1 = status1.get("signals") or {}
    signals2 = status2.get("signals") or {}
    signal1 = signals1.get("release_attitude_strip")
    signal2 = signals2.get("release_attitude_strip")
    
    assert signal1 is not None
    assert signal2 is not None
    assert signal1 == signal2


def test_release_attitude_strip_signal_shape(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that signal has correct shape with all required fields."""
    # Create release attitude strip
    manifest_path = evidence_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    release_attitude_strip = {
        "schema_version": "1.0.0",
        "experiments": [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
            {
                "cal_id": "CAL-EXP-2",
                "global_band": "YELLOW",
                "system_alignment": "PARTIAL",
                "release_ready": False,
                "status_light": "🟡",
            },
        ],
        "summary": {
            "total_count": 2,
            "release_ready_count": 1,
            "release_ready_ratio": 0.5,
            "trend": "IMPROVING",
        },
    }
    
    manifest["governance"]["release_attitude_strip"] = release_attitude_strip
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signal shape
    signals = status.get("signals")
    assert signals is not None
    signal = signals["release_attitude_strip"]
    assert "mode" in signal
    assert "schema_version" in signal
    assert "total_count" in signal
    assert "release_ready_ratio" in signal
    assert "trend" in signal
    assert "first_status_light" in signal
    assert "last_status_light" in signal
    assert "advisory_warning" in signal
    
    assert signal["mode"] == "SHADOW"
    assert signal["schema_version"] == "1.0.0"
    assert isinstance(signal["total_count"], int)
    assert isinstance(signal["release_ready_ratio"], (int, float))
    assert signal["trend"] in ["IMPROVING", "STABLE", "DEGRADING"]
    assert signal["first_status_light"] in ["🟢", "🟡", "🔴"]
    assert signal["last_status_light"] in ["🟢", "🟡", "🔴"]


def test_release_attitude_strip_fallback_to_signals_section(
    evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
):
    """Test that signal is extracted from evidence.json signals section if strip not in governance."""
    # Create evidence.json with signal already extracted in signals section
    evidence_json_path = evidence_pack_dir / "evidence.json"
    evidence_data = {
        "signals": {
            "release_attitude_strip": {
                "mode": "SHADOW",
                "total_count": 1,
                "release_ready_ratio": 1.0,
                "trend": "STABLE",
                "first_status_light": "🟢",
                "last_status_light": "🟢",
                "advisory_warning": None,
            },
        },
    }
    evidence_json_path.write_text(json.dumps(evidence_data, indent=2), encoding="utf-8")
    
    # Generate status
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Verify signal is present
    assert "signals" in status
    assert status["signals"] is not None
    assert "release_attitude_strip" in status["signals"]
    
    signal = status["signals"]["release_attitude_strip"]
    assert signal["total_count"] == 1
    assert signal["trend"] == "STABLE"

