"""
Tests for semantic safety panel extraction in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from scripts.generate_first_light_status import generate_status

# Import reusable warning neutrality helpers (single source of truth)
from tests.helpers.warning_neutrality import pytest_assert_warning_neutral


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
    """Create a minimal P4 directory structure."""
    p4_dir = tmp_path / "p4"
    p4_dir.mkdir()
    p4_run = p4_dir / "p4_test"
    p4_run.mkdir()
    (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
    (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
    (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
    (p4_run / "p4_summary.json").write_text(
        json.dumps({"mode": "SHADOW"}), encoding="utf-8"
    )
    (p4_run / "twin_accuracy.json").write_text("{}", encoding="utf-8")
    (p4_run / "run_config.json").write_text("{}", encoding="utf-8")
    return p4_dir


def test_semantic_safety_panel_extracted_from_manifest_first(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that semantic safety panel signal is extracted from manifest.json first (manifest-first precedence)."""
    # Read existing manifest and add semantic safety panel
    manifest_path = evidence_pack_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    manifest["governance"] = {
        "semantic_safety_panel": {
            "schema_version": "1.0.0",
            "total_experiments": 3,
            "grid_counts": {
                "ok_ok": 2,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-3"],
            "experiments": [],
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    # Also create evidence.json with different panel (to verify manifest takes precedence)
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 5,
                "grid_counts": {
                    "ok_ok": 4,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 1,
                },
                "top_drivers": ["CAL-EXP-5"],
                "experiments": [],
            },
        },
    }
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    signals = status.get("signals") or {}
    assert "semantic_safety_panel" in signals
    
    panel_signal = signals["semantic_safety_panel"]
    # Should use manifest data (2 ok_ok, not 4)
    assert panel_signal["ok_ok"] == 2
    assert panel_signal["not_ok_not_ok"] == 1
    assert panel_signal["top_drivers"] == ["CAL-EXP-3"]
    assert panel_signal["extraction_source"] == "MANIFEST"


def test_semantic_safety_panel_extracted_from_governance(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that semantic safety panel signal is extracted from governance when present."""
    # Create evidence.json with semantic safety panel in governance
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 3,
                "grid_counts": {
                    "ok_ok": 2,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 1,
                },
                "top_drivers": ["CAL-EXP-3"],
                "experiments": [
                    {
                        "cal_id": "CAL-EXP-1",
                        "p3_status": "OK",
                        "p4_status": "OK",
                        "broken_invariant_count": 0,
                        "grid_bucket": "OK×OK",
                    },
                    {
                        "cal_id": "CAL-EXP-2",
                        "p3_status": "OK",
                        "p4_status": "OK",
                        "broken_invariant_count": 0,
                        "grid_bucket": "OK×OK",
                    },
                    {
                        "cal_id": "CAL-EXP-3",
                        "p3_status": "BLOCK",
                        "p4_status": "BROKEN",
                        "broken_invariant_count": 5,
                        "grid_bucket": "Not-OK×Not-OK",
                    },
                ],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "semantic_safety_panel" in status["signals"]
    
    signals = status.get("signals") or {}
    panel_signal = signals["semantic_safety_panel"]
    assert panel_signal["ok_ok"] == 2
    assert panel_signal["ok_not_ok"] == 0
    assert panel_signal["not_ok_ok"] == 0
    assert panel_signal["not_ok_not_ok"] == 1
    assert panel_signal["top_drivers"] == ["CAL-EXP-3"]
    assert panel_signal["extraction_source"] == "EVIDENCE_JSON_GOVERNANCE"


def test_semantic_safety_panel_extracted_from_signals_fallback(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that semantic safety panel signal is extracted from signals section when not in governance."""
    # Create evidence.json with signal already extracted in signals section
    evidence = {
        "governance": {},
        "signals": {
            "semantic_safety_panel": {
                "ok_ok": 1,
                "ok_not_ok": 1,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
                "top_drivers": ["CAL-EXP-2"],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    assert "semantic_safety_panel" in status["signals"]
    
    signals = status.get("signals") or {}
    panel_signal = signals["semantic_safety_panel"]
    assert panel_signal["ok_ok"] == 1
    assert panel_signal["ok_not_ok"] == 1
    assert panel_signal["not_ok_not_ok"] == 1
    assert panel_signal["top_drivers"] == ["CAL-EXP-2"]
    assert panel_signal["extraction_source"] == "EVIDENCE_JSON_SIGNALS"


def test_semantic_safety_panel_absent_when_not_in_evidence(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that semantic safety panel signal is absent when not in evidence.json."""
    # Create evidence.json without semantic safety panel
    evidence = {
        "governance": {},
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "signals" in status
    signals = status.get("signals") or {}
    # Semantic safety panel should not be present when not in evidence
    assert "semantic_safety_panel" not in signals


def test_semantic_safety_panel_warning_generated_when_not_ok_not_ok(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that advisory warning is generated when not_ok_not_ok > 0 and includes only top 3 drivers."""
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 5,
                "grid_counts": {
                    "ok_ok": 1,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 4,
                },
                "top_drivers": ["CAL-EXP-2", "CAL-EXP-3", "CAL-EXP-1", "CAL-EXP-4", "CAL-EXP-5"],
                "experiments": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "warnings" in status
    warnings = status["warnings"]
    
    # Check that warning contains neutral wording
    semantic_warnings = [w for w in warnings if "semantic safety panel" in w.lower()]
    assert len(semantic_warnings) > 0
    
    warning_text = semantic_warnings[0]
    # Check warning is one line (no newlines)
    assert "\n" not in warning_text
    assert warning_text.count("\n") == 0
    
    # Check warning includes top drivers (limited to top 3)
    assert "top drivers" in warning_text.lower()
    # Should include at least one of the top 3 drivers
    assert any(driver in warning_text for driver in ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"])
    # Should NOT include drivers beyond top 3
    assert "CAL-EXP-4" not in warning_text
    assert "CAL-EXP-5" not in warning_text

    # Use reusable helper (single source of truth for banned words)
    pytest_assert_warning_neutral(warning_text, context="semantic safety panel warning")


def test_semantic_safety_panel_no_warning_when_all_ok(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that no warning is generated when all experiments are OK×OK."""
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 3,
                "grid_counts": {
                    "ok_ok": 3,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 0,
                },
                "top_drivers": [],
                "experiments": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "warnings" in status
    warnings = status["warnings"]
    
    # Check that no semantic safety panel warning is present
    semantic_warnings = [w for w in warnings if "semantic safety panel" in w.lower()]
    assert len(semantic_warnings) == 0


def test_semantic_safety_panel_deterministic(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that semantic safety panel extraction is deterministic."""
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 2,
                "grid_counts": {
                    "ok_ok": 1,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 1,
                },
                "top_drivers": ["CAL-EXP-2"],
                "experiments": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Signals should be identical
    assert status1["signals"].get("semantic_safety_panel") == status2["signals"].get("semantic_safety_panel")


def test_semantic_safety_panel_missing_panel_not_error(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that missing semantic safety panel does not cause errors."""
    # Create evidence.json without semantic safety panel
    evidence = {
        "governance": {},
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    # Should not raise
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    # Status should be generated successfully
    assert "signals" in status
    signals = status.get("signals") or {}
    # Panel should not be present but should not cause errors
    assert "semantic_safety_panel" not in signals


def test_semantic_safety_panel_warning_includes_top_drivers(
    evidence_pack_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
) -> None:
    """Test that warning includes top drivers when present (limited to top 3)."""
    evidence = {
        "governance": {
            "semantic_safety_panel": {
                "schema_version": "1.0.0",
                "total_experiments": 5,
                "grid_counts": {
                    "ok_ok": 0,
                    "ok_not_ok": 0,
                    "not_ok_ok": 0,
                    "not_ok_not_ok": 4,
                },
                "top_drivers": ["CAL-EXP-3", "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-4", "CAL-EXP-5"],
                "experiments": [],
            },
        },
    }
    
    evidence_path = evidence_pack_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f)
    
    status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
    
    assert "warnings" in status
    warnings = status["warnings"]
    
    semantic_warnings = [w for w in warnings if "semantic safety panel" in w.lower()]
    assert len(semantic_warnings) > 0
    
    warning_text = semantic_warnings[0]
    # Check warning is one line
    assert "\n" not in warning_text
    assert warning_text.count("\n") == 0
    # Check that top drivers are mentioned (limited to top 3)
    assert "CAL-EXP-3" in warning_text
    assert "CAL-EXP-1" in warning_text
    # Verify only top 3 drivers are included (not CAL-EXP-4)
    assert "CAL-EXP-4" not in warning_text

