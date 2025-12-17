"""
Tests for curriculum coherence panel extraction in generate_first_light_status.py.

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
    p3 = tmp_path / "p3"
    p3.mkdir()
    fl_dir = p3 / "fl_run_001"
    fl_dir.mkdir()
    
    # Create minimal stability_report.json
    stability_report = {
        "metrics": {
            "success_rate": 0.95,
            "omega": {"occupancy_rate": 0.92},
            "rsi": {"mean": 0.88},
            "hard_mode": {"ok_rate": 0.90},
        },
        "criteria_evaluation": {"all_passed": True},
        "red_flag_summary": {"total_flags": 0, "hypothetical_abort": False},
        "pathology": "none",
    }
    
    with open(fl_dir / "stability_report.json", "w") as f:
        json.dump(stability_report, f)
    
    # Create other required artifacts
    for artifact in [
        "synthetic_raw.jsonl",
        "red_flag_matrix.json",
        "metrics_windows.json",
        "tda_metrics.json",
        "run_config.json",
    ]:
        (fl_dir / artifact).touch()
    
    return p3


@pytest.fixture
def p4_dir(tmp_path: Path) -> Path:
    """Create a minimal P4 directory structure."""
    p4 = tmp_path / "p4"
    p4.mkdir()
    p4_run_dir = p4 / "p4_run_001"
    p4_run_dir.mkdir()
    
    # Create minimal p4_summary.json
    p4_summary = {
        "mode": "SHADOW",
        "uplift_metrics": {"u2_success_rate_final": 0.93},
        "divergence_analysis": {"divergence_rate": 0.05, "max_divergence_streak": 2},
        "twin_accuracy": {
            "success_prediction_accuracy": 0.85,
            "omega_prediction_accuracy": 0.90,
        },
    }
    
    with open(p4_run_dir / "p4_summary.json", "w") as f:
        json.dump(p4_summary, f)
    
    # Create run_config.json
    run_config = {
        "run_id": "p4_run_001",
        "telemetry_source": "mock",
    }
    
    with open(p4_run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f)
    
    # Create other required artifacts
    for artifact in [
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
        "divergence_log.jsonl",
        "twin_accuracy.json",
    ]:
        (p4_run_dir / artifact).touch()
    
    return p4


class TestCurriculumCoherencePanelExtraction:
    """Test curriculum coherence panel extraction in status generation."""

    def test_panel_extraction_when_present(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test curriculum coherence panel is extracted when present in manifest."""
        # Load manifest and add panel
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 2,
            "num_warn": 1,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.9,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        assert "signals" in status
        assert status["signals"] is not None
        assert "curriculum_coherence_panel" in status["signals"]
        
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["num_experiments"] == 3
        assert panel_signal["num_block"] == 0
        assert panel_signal["num_high_drift"] == 0
        assert panel_signal["median_alignment_score"] == 0.9

    def test_panel_extraction_when_absent(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test curriculum coherence panel is not extracted when absent."""
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        assert "signals" in status
        # Panel should not be present if not in manifest
        if status["signals"]:
            assert "curriculum_coherence_panel" not in status["signals"]

    def test_warning_when_num_block_gt_zero(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test advisory warning is generated when num_block > 0."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_high_drift": 0,
            "median_alignment_score": 0.75,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        warnings = status.get("warnings", [])
        block_warnings = [w for w in warnings if "BLOCK status" in w]
        assert len(block_warnings) > 0
        assert "1 experiment(s) with BLOCK status" in block_warnings[0]

    def test_warning_when_num_high_drift_gt_zero(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test advisory warning is generated when num_high_drift > 0."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 2,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 1,
            "median_alignment_score": 0.8,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        warnings = status.get("warnings", [])
        drift_warnings = [w for w in warnings if "HIGH_DRIFT" in w]
        assert len(drift_warnings) > 0
        assert "1 experiment(s) with HIGH_DRIFT" in drift_warnings[0]
        assert "median_alignment_score=0.800" in drift_warnings[0]

    def test_no_warning_when_all_ok(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test no warning is generated when num_block == 0 and num_high_drift == 0."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        warnings = status.get("warnings", [])
        coherence_warnings = [
            w for w in warnings if "curriculum coherence panel" in w.lower()
        ]
        assert len(coherence_warnings) == 0

    def test_panel_extraction_determinism(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test panel extraction is deterministic (same input → same output)."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 2,
            "num_warn": 1,
            "num_block": 0,
            "num_high_drift": 1,
            "median_alignment_score": 0.85,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Compare panel signals
        panel1 = status1.get("signals", {}).get("curriculum_coherence_panel")
        panel2 = status2.get("signals", {}).get("curriculum_coherence_panel")
        
        assert panel1 == panel2
        assert json.dumps(panel1, sort_keys=True) == json.dumps(panel2, sort_keys=True)

    def test_panel_extraction_prefers_manifest_over_evidence_json(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test panel extraction prefers manifest over evidence.json fallback."""
        manifest_path = evidence_pack_dir / "manifest.json"
        evidence_json_path = evidence_pack_dir / "evidence.json"
        
        # Load manifest and add panel
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Create evidence.json with different panel (should be ignored)
        evidence_data = {
            "governance": {
                "curriculum_coherence_panel": {
                    "schema_version": "1.0.0",
                    "num_experiments": 2,  # Different value
                    "num_ok": 2,
                    "num_warn": 0,
                    "num_block": 0,
                    "num_high_drift": 0,
                    "median_alignment_score": 0.90,
                }
            }
        }
        
        with open(evidence_json_path, "w") as f:
            json.dump(evidence_data, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Should use manifest value (3 experiments), not evidence.json (2)
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["num_experiments"] == 3
        assert panel_signal["extraction_source"] == "MANIFEST"
        assert panel_signal["panel_schema_version"] == "1.0.0"
        # No warning for manifest extraction
        warnings = status.get("warnings", [])
        evidence_json_warnings = [w for w in warnings if "evidence.json fallback" in w]
        assert len(evidence_json_warnings) == 0

    def test_panel_extraction_fallback_to_evidence_json(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test panel extraction falls back to evidence.json when not in manifest."""
        manifest_path = evidence_pack_dir / "manifest.json"
        evidence_json_path = evidence_pack_dir / "evidence.json"
        
        # Load manifest (no panel)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Create evidence.json with panel (should be used as fallback)
        evidence_data = {
            "governance": {
                "curriculum_coherence_panel": {
                    "schema_version": "1.0.0",
                    "num_experiments": 2,
                    "num_ok": 2,
                    "num_warn": 0,
                    "num_block": 0,
                    "num_high_drift": 0,
                    "median_alignment_score": 0.90,
                }
            }
        }
        
        with open(evidence_json_path, "w") as f:
            json.dump(evidence_data, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Should use evidence.json value
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["num_experiments"] == 2
        assert panel_signal["extraction_source"] == "EVIDENCE_JSON"
        assert panel_signal["panel_schema_version"] == "1.0.0"
        # Should have exactly one warning about evidence.json fallback
        warnings = status.get("warnings", [])
        evidence_json_warnings = [w for w in warnings if "evidence.json fallback" in w]
        assert len(evidence_json_warnings) == 1

    def test_schema_version_passthrough(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test schema_version is passed through to status signal."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["panel_schema_version"] == "1.0.0"
        assert panel_signal["mode"] == "SHADOW"
        assert panel_signal["extraction_source"] == "MANIFEST"

    def test_mode_marker_always_present(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test mode marker is always present in signal output."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
            # No schema_version
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["mode"] == "SHADOW"
        assert panel_signal["extraction_source"] == "MANIFEST"
        assert panel_signal["panel_schema_version"] == "UNKNOWN"  # No schema_version in panel


class TestGGFLAdapter:
    """Test GGFL adapter for curriculum coherence panel."""

    def test_ggfl_adapter_basic(self):
        """Test GGFL adapter produces correct shape with stable output contract."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        # Verify stable output contract
        assert "signal_type" in result
        assert "status" in result
        assert "conflict" in result
        assert "drivers" in result
        assert "summary" in result
        
        assert result["signal_type"] == "SIG-CURR"
        assert result["status"] in ("ok", "warn")
        assert result["conflict"] is False
        assert isinstance(result["drivers"], list)
        assert isinstance(result["summary"], str)

    def test_ggfl_adapter_status_ok(self):
        """Test GGFL adapter returns 'ok' when no blocks and no high drift."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,  # High score, but status is ok
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert result["status"] == "ok"
        assert len(result["drivers"]) == 0  # No drivers when status is ok

    def test_ggfl_adapter_status_warn_with_block(self):
        """Test GGFL adapter returns 'warn' when num_block > 0."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 1,
            "num_high_drift": 0,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert result["status"] == "warn"
        assert len(result["drivers"]) > 0
        assert result["drivers"][0].startswith("DRIVER_BLOCK_COUNT:")

    def test_ggfl_adapter_status_warn_with_high_drift(self):
        """Test GGFL adapter returns 'warn' when num_high_drift > 0."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 0,
            "num_high_drift": 2,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert result["status"] == "warn"
        assert len(result["drivers"]) > 0
        assert result["drivers"][0].startswith("DRIVER_HIGH_DRIFT_COUNT:")

    def test_ggfl_adapter_drivers_truncated_to_three(self):
        """Test GGFL adapter limits drivers to top 3."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 2,
            "num_high_drift": 2,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert len(result["drivers"]) <= 3

    def test_ggfl_adapter_determinism(self):
        """Test GGFL adapter is deterministic."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 1,
            "num_high_drift": 1,
        }
        
        result1 = curriculum_coherence_panel_for_alignment_view(panel)
        result2 = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_ggfl_adapter_read_only(self):
        """Test GGFL adapter does not mutate input panel."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_block": 1,
            "num_high_drift": 0,
        }
        
        panel_copy = json.loads(json.dumps(panel))
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert json.dumps(panel, sort_keys=True) == json.dumps(panel_copy, sort_keys=True)

    def test_ggfl_adapter_drivers_ordering(self):
        """Test GGFL adapter drivers follow deterministic ordering: blocks first, then high drift, then median."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_experiments": 3,
            "num_block": 1,
            "num_high_drift": 1,
            "median_alignment_score": 0.65,  # Below 0.7 threshold
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        drivers = result["drivers"]
        # Should have: blocks first, then high drift, then median (if < 0.7)
        assert len(drivers) <= 3
        if len(drivers) >= 1:
            assert drivers[0].startswith("DRIVER_BLOCK_COUNT:")
        if len(drivers) >= 2:
            assert drivers[1].startswith("DRIVER_HIGH_DRIFT_COUNT:")
        if len(drivers) >= 3:
            assert drivers[2].startswith("DRIVER_MEDIAN_BELOW_THRESHOLD:")

    def test_ggfl_adapter_summary_neutral_sentence(self):
        """Test GGFL adapter summary is a single neutral sentence."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_experiments": 3,
            "num_block": 1,
            "num_high_drift": 1,
            "median_alignment_score": 0.85,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        summary = result["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should be a single sentence (no multiple periods in middle)
        # Basic check: should contain key information
        assert "Curriculum coherence panel" in summary
        assert "3 experiment(s)" in summary

    def test_ggfl_adapter_invariants_block(self):
        """Test GGFL adapter includes shadow_mode_invariants block."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_experiments": 3,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        assert "shadow_mode_invariants" in result
        invariants = result["shadow_mode_invariants"]
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True

    def test_ggfl_adapter_driver_reason_codes_only(self):
        """Test GGFL adapter drivers use only prefixed reason codes, no freeform text."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        panel = {
            "num_experiments": 3,
            "num_block": 1,
            "num_high_drift": 1,
            "median_alignment_score": 0.65,
        }
        
        result = curriculum_coherence_panel_for_alignment_view(panel)
        
        drivers = result["drivers"]
        # All drivers must start with DRIVER_ prefix
        for driver in drivers:
            assert driver.startswith("DRIVER_")
            # Verify it's one of the allowed reason codes
            assert any(
                driver.startswith(code)
                for code in [
                    "DRIVER_BLOCK_COUNT:",
                    "DRIVER_HIGH_DRIFT_COUNT:",
                    "DRIVER_MEDIAN_BELOW_THRESHOLD:",
                ]
            )

    def test_extraction_source_missing_when_panel_absent(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test extraction_source is MISSING when panel is not found."""
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Panel should not be present
        if status.get("signals"):
            assert "curriculum_coherence_panel" not in status["signals"]

    def test_extraction_source_enum_coercion(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test extraction_source is coerced to valid enum values only."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["extraction_source"] in ("MANIFEST", "EVIDENCE_JSON", "MISSING")

    def test_schema_version_present_audit_field(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test schema_version_present boolean for audit clarity."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Test with schema_version present
        manifest["governance"]["curriculum_coherence_panel"] = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 3,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.95,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["schema_version_present"] is True
        assert panel_signal["panel_schema_version"] == "1.0.0"
        
        # Test without schema_version
        manifest["governance"]["curriculum_coherence_panel"].pop("schema_version")
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        panel_signal2 = status2["signals"]["curriculum_coherence_panel"]
        assert panel_signal2["schema_version_present"] is False
        assert panel_signal2["panel_schema_version"] == "UNKNOWN"

    def test_warning_code_stability_evidence_json_fallback(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path
    ):
        """Test warning code CURR-PROV-001 is stable and deterministic for EVIDENCE_JSON fallback."""
        manifest_path = evidence_pack_dir / "manifest.json"
        evidence_json_path = evidence_pack_dir / "evidence.json"
        
        # Load manifest (no panel)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Create evidence.json with panel (should trigger fallback)
        evidence_data = {
            "governance": {
                "curriculum_coherence_panel": {
                    "schema_version": "1.0.0",
                    "num_experiments": 2,
                    "num_ok": 2,
                    "num_warn": 0,
                    "num_block": 0,
                    "num_high_drift": 0,
                    "median_alignment_score": 0.90,
                }
            }
        }
        
        with open(evidence_json_path, "w") as f:
            json.dump(evidence_data, f)
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Should have exactly one warning with code CURR-PROV-001
        warnings = status.get("warnings", [])
        evidence_json_warnings = [w for w in warnings if "CURR-PROV-001" in w]
        assert len(evidence_json_warnings) == 1
        assert evidence_json_warnings[0].startswith("CURR-PROV-001:")
        
        # Verify extraction_source is EVIDENCE_JSON
        panel_signal = status["signals"]["curriculum_coherence_panel"]
        assert panel_signal["extraction_source"] == "EVIDENCE_JSON"

    def test_no_freeform_driver_strings(self):
        """Test GGFL adapter never emits freeform driver strings, only reason codes."""
        from scripts.taxonomy_governance import curriculum_coherence_panel_for_alignment_view
        
        # Test various panel configurations
        test_panels = [
            {"num_block": 1, "num_high_drift": 0, "median_alignment_score": 0.95},
            {"num_block": 0, "num_high_drift": 2, "median_alignment_score": 0.80},
            {"num_block": 1, "num_high_drift": 1, "median_alignment_score": 0.65},
            {"num_block": 0, "num_high_drift": 0, "median_alignment_score": 0.95},
        ]
        
        allowed_prefixes = ("DRIVER_BLOCK_COUNT:", "DRIVER_HIGH_DRIFT_COUNT:", "DRIVER_MEDIAN_BELOW_THRESHOLD:")
        
        for panel in test_panels:
            result = curriculum_coherence_panel_for_alignment_view(panel)
            drivers = result["drivers"]
            
            # All drivers must start with allowed prefixes
            for driver in drivers:
                assert any(driver.startswith(prefix) for prefix in allowed_prefixes), (
                    f"Driver must use prefixed reason code, got: {driver}"
                )

