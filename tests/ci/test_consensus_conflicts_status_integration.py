"""
Tests for consensus conflicts signal integration in First Light status generation.

SHADOW MODE: These tests verify manifest-first extraction, signal emission, and warning hygiene.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.generate_first_light_status import generate_status


def test_consensus_conflicts_signal_manifest_first_extraction() -> None:
    """Verify consensus conflicts signal is extracted from manifest (manifest-first)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        # Create minimal directory structure
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        # Create minimal P3/P4 run directories
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        # Create manifest with consensus conflict register
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 3,
                    "experiments_high_conflict": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-5"],
                    "high_risk_experiments_count": 2,
                    "fusion_crosscheck": {
                        "schema_version": "1.0.0",
                        "consistency_status": "CONFLICT",
                        "examples": [],
                        "advisory_notes": [],
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Create minimal P3/P4 artifacts
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        assert "signals" in status
        assert status["signals"] is not None, "Signals should not be None when consensus_conflicts is present"
        assert "consensus_conflicts" in status["signals"]
        
        signal = status["signals"]["consensus_conflicts"]
        assert signal["schema_version"] == "1.0.0"
        assert signal["mode"] == "SHADOW"
        assert signal["experiments_high_conflict_count"] == 3
        assert signal["high_risk_band_count"] == 2
        assert signal["fusion_consistency_status"] == "CONFLICT"
        assert signal["top_high_conflict_cal_ids_top3"] == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-5"]
        assert signal["extraction_source"] == "MANIFEST"
        assert signal["top_reason_code"] == "CONFLICT"


def test_consensus_conflicts_signal_missing_crosscheck_safe() -> None:
    """Verify signal extraction is safe when fusion_crosscheck is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        # Create manifest without fusion_crosscheck
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 2,
                    "experiments_high_conflict": ["CAL-EXP-1"],
                    "high_risk_experiments_count": 1,
                    # No fusion_crosscheck
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        assert "signals" in status
        assert status["signals"] is not None, "Signals should not be None when consensus_conflicts is present"
        assert "consensus_conflicts" in status["signals"]
        
        signal = status["signals"]["consensus_conflicts"]
        assert signal["fusion_consistency_status"] == "UNKNOWN"  # Default when crosscheck missing
        assert signal["extraction_source"] == "MANIFEST"


def test_consensus_conflicts_signal_deterministic() -> None:
    """Verify consensus conflicts signal output is deterministic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        # Create manifest with unsorted cal_ids
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 3,
                    "experiments_high_conflict": ["CAL-EXP-3", "CAL-EXP-1", "CAL-EXP-2"],
                    "high_risk_experiments_count": 1,
                    "fusion_crosscheck": {
                        "consistency_status": "TENSION",
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status1 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        status2 = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        signal1 = status1["signals"]["consensus_conflicts"]
        signal2 = status2["signals"]["consensus_conflicts"]
        
        assert signal1 == signal2, "Signal output should be deterministic"
        
        # Verify cal_ids are sorted
        assert signal1["top_high_conflict_cal_ids_top3"] == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]


def test_consensus_conflicts_warning_hygiene_tension() -> None:
    """Verify single warning is generated for TENSION status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 2,
                    "experiments_high_conflict": ["CAL-EXP-1", "CAL-EXP-2"],
                    "high_risk_experiments_count": 1,
                    "fusion_crosscheck": {
                        "consistency_status": "TENSION",
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Count warnings related to consensus conflicts
        consensus_warnings = [
            w for w in status.get("warnings", [])
            if "Consensus conflict register" in w or "consensus" in w.lower()
        ]
        
        assert len(consensus_warnings) == 1, "Should generate exactly one warning for TENSION"
        assert "TENSION" in consensus_warnings[0]
        assert "CAL-EXP-1" in consensus_warnings[0] or "CAL-EXP-2" in consensus_warnings[0]


def test_consensus_conflicts_warning_hygiene_conflict() -> None:
    """Verify single warning is generated for CONFLICT status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 1,
                    "experiments_high_conflict": ["CAL-EXP-1"],
                    "high_risk_experiments_count": 1,
                    "fusion_crosscheck": {
                        "consistency_status": "CONFLICT",
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Count warnings related to consensus conflicts
        consensus_warnings = [
            w for w in status.get("warnings", [])
            if "Consensus conflict register" in w
        ]
        
        assert len(consensus_warnings) == 1, "Should generate exactly one warning for CONFLICT"
        assert "CONFLICT" in consensus_warnings[0]
        assert "CAL-EXP-1" in consensus_warnings[0]


def test_consensus_conflicts_no_warning_for_consistent() -> None:
    """Verify no warning is generated for CONSISTENT status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 1,
                    "experiments_high_conflict": [],
                    "high_risk_experiments_count": 0,
                    "fusion_crosscheck": {
                        "consistency_status": "CONSISTENT",
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        # Count warnings related to consensus conflicts
        consensus_warnings = [
            w for w in status.get("warnings", [])
            if "Consensus conflict register" in w
        ]
        
        assert len(consensus_warnings) == 0, "Should not generate warning for CONSISTENT status"


def test_consensus_conflicts_top3_limit() -> None:
    """Verify top_high_conflict_cal_ids_top3 is limited to 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p3_dir = Path(tmpdir) / "p3"
        p4_dir = Path(tmpdir) / "p4"
        evidence_pack_dir = Path(tmpdir) / "evidence"
        
        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)
        
        (p3_dir / "fl_test").mkdir()
        (p4_dir / "p4_test").mkdir()
        
        manifest: Dict[str, Any] = {
            "file_count": 0,
            "mode": "SHADOW",
            "shadow_mode_compliance": {
                "all_divergence_logged_only": True,
                "no_governance_modification": True,
                "no_abort_enforcement": True,
            },
            "governance": {
                "consensus_conflict_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 5,
                    "experiments_high_conflict": ["CAL-EXP-5", "CAL-EXP-1", "CAL-EXP-3", "CAL-EXP-2", "CAL-EXP-4"],
                    "high_risk_experiments_count": 0,
                    "fusion_crosscheck": {
                        "consistency_status": "CONSISTENT",
                    },
                },
            },
            "files": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        (p3_dir / "fl_test" / "stability_report.json").write_text(json.dumps({"metrics": {}}))
        (p4_dir / "p4_test" / "p4_summary.json").write_text(
            json.dumps({"mode": "SHADOW", "uplift_metrics": {}, "divergence_analysis": {}, "twin_accuracy": {}})
        )
        (p4_dir / "p4_test" / "run_config.json").write_text(json.dumps({}))
        
        status = generate_status(p3_dir, p4_dir, evidence_pack_dir)
        
        signal = status["signals"]["consensus_conflicts"]
        top3 = signal["top_high_conflict_cal_ids_top3"]
        
        assert len(top3) == 3, "Should limit to top 3 cal_ids"
        assert top3 == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"], "Should be sorted deterministically"


class TestConsensusConflictsForAlignmentView:
    """
    Tests for consensus conflicts GGFL alignment view adapter.

    SHADOW MODE: These tests verify the adapter structure, determinism, and invariants.
    """

    def test_consensus_conflicts_for_alignment_view_structure(self) -> None:
        """Verify alignment view has required structure."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 2,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "TENSION",
            "extraction_source": "MANIFEST",
            "top_reason_code": "TENSION",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        required_fields = [
            "signal_type",
            "status",
            "conflict",
            "weight_hint",
            "drivers",
            "summary",
            "extraction_source",
        ]

        for field in required_fields:
            assert field in view, f"Missing required field: {field}"

        assert view["signal_type"] == "SIG-CON"
        assert view["status"] in ("ok", "warn")
        assert view["conflict"] is False  # Invariant
        assert view["weight_hint"] == "LOW"  # Invariant

    def test_consensus_conflicts_for_alignment_view_deterministic(self) -> None:
        """Verify alignment view output is deterministic."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 3,
            "high_risk_band_count": 2,
            "fusion_consistency_status": "CONFLICT",
            "extraction_source": "MANIFEST",
        }

        view1 = consensus_conflicts_for_alignment_view(signal)
        view2 = consensus_conflicts_for_alignment_view(signal)

        assert view1 == view2, "Alignment view output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(view1, sort_keys=True)
        json2 = json.dumps(view2, sort_keys=True)
        assert json1 == json2, "JSON serialization should be deterministic"

    def test_consensus_conflicts_for_alignment_view_reason_code_drivers(self) -> None:
        """Verify drivers use reason codes only."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 1,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "TENSION",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        # All drivers should be reason codes
        expected_drivers = [
            "DRIVER_FUSION_TENSION_OR_CONFLICT",
            "DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT",
            "DRIVER_HIGH_RISK_BAND_PRESENT",
        ]

        assert len(view["drivers"]) == 3
        assert view["drivers"] == expected_drivers

    def test_consensus_conflicts_for_alignment_view_status_warn(self) -> None:
        """Verify status is warn when conditions are met."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        # Test TENSION
        signal1: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "TENSION",
            "extraction_source": "MANIFEST",
        }
        view1 = consensus_conflicts_for_alignment_view(signal1)
        assert view1["status"] == "warn"

        # Test CONFLICT
        signal2: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "CONFLICT",
            "extraction_source": "MANIFEST",
        }
        view2 = consensus_conflicts_for_alignment_view(signal2)
        assert view2["status"] == "warn"

        # Test high_conflict_count > 0
        signal3: Dict[str, Any] = {
            "experiments_high_conflict_count": 1,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "CONSISTENT",
            "extraction_source": "MANIFEST",
        }
        view3 = consensus_conflicts_for_alignment_view(signal3)
        assert view3["status"] == "warn"

        # Test high_risk_band_count > 0
        signal4: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "CONSISTENT",
            "extraction_source": "MANIFEST",
        }
        view4 = consensus_conflicts_for_alignment_view(signal4)
        assert view4["status"] == "warn"

    def test_consensus_conflicts_for_alignment_view_status_ok(self) -> None:
        """Verify status is ok when no warning conditions."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "CONSISTENT",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        assert view["status"] == "ok"
        assert len(view["drivers"]) == 0

    def test_consensus_conflicts_for_alignment_view_invariants(self) -> None:
        """Verify shadow mode invariants are always enforced."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        # Test with various signal states
        test_signals = [
            {
                "experiments_high_conflict_count": 0,
                "high_risk_band_count": 0,
                "fusion_consistency_status": "CONSISTENT",
                "extraction_source": "MANIFEST",
            },
            {
                "experiments_high_conflict_count": 5,
                "high_risk_band_count": 3,
                "fusion_consistency_status": "CONFLICT",
                "extraction_source": "EVIDENCE_JSON",
            },
            {
                "experiments_high_conflict_count": 1,
                "high_risk_band_count": 0,
                "fusion_consistency_status": "TENSION",
                "extraction_source": "MISSING",
            },
        ]

        for signal in test_signals:
            view = consensus_conflicts_for_alignment_view(signal)

            # Invariants: conflict always False, weight_hint always LOW
            assert view["conflict"] is False, "conflict must always be False (invariant)"
            assert view["weight_hint"] == "LOW", "weight_hint must always be LOW (invariant)"
            assert view["signal_type"] == "SIG-CON", "signal_type must always be SIG-CON (invariant)"

    def test_consensus_conflicts_for_alignment_view_extraction_source(self) -> None:
        """Verify extraction_source is passed through."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "CONSISTENT",
            "extraction_source": "EVIDENCE_JSON",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        assert view["extraction_source"] == "EVIDENCE_JSON"

    def test_consensus_conflicts_for_alignment_view_drivers_limit(self) -> None:
        """Verify drivers are limited to 3."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 1,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "CONFLICT",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        # Should have all 3 drivers, but limited to 3
        assert len(view["drivers"]) <= 3
        assert "DRIVER_FUSION_TENSION_OR_CONFLICT" in view["drivers"]
        assert "DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT" in view["drivers"]
        assert "DRIVER_HIGH_RISK_BAND_PRESENT" in view["drivers"]

    def test_consensus_conflicts_for_alignment_view_json_safe(self) -> None:
        """Verify alignment view is JSON-safe."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 2,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "TENSION",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        # Should serialize without error
        json_str = json.dumps(view)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["signal_type"] == "SIG-CON"

    def test_consensus_conflicts_for_alignment_view_invariants_block(self) -> None:
        """Verify shadow_mode_invariants block is present and correct."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 0,
            "high_risk_band_count": 0,
            "fusion_consistency_status": "CONSISTENT",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        assert "shadow_mode_invariants" in view
        invariants = view["shadow_mode_invariants"]
        assert isinstance(invariants, dict)
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True

    def test_consensus_conflicts_for_alignment_view_drivers_no_prose(self) -> None:
        """Verify drivers contain only reason codes (no prose)."""
        from backend.health.consensus_polygraph_adapter import (
            consensus_conflicts_for_alignment_view,
        )

        # Test with all driver conditions met
        signal: Dict[str, Any] = {
            "experiments_high_conflict_count": 1,
            "high_risk_band_count": 1,
            "fusion_consistency_status": "CONFLICT",
            "extraction_source": "MANIFEST",
        }

        view = consensus_conflicts_for_alignment_view(signal)

        drivers = view["drivers"]
        assert isinstance(drivers, list)

        # All drivers must be reason codes (start with DRIVER_)
        for driver in drivers:
            assert isinstance(driver, str)
            assert driver.startswith("DRIVER_"), f"Driver '{driver}' must be a reason code (start with DRIVER_)"
            # No prose: should not contain spaces, colons, or descriptive text
            # Reason codes are uppercase with underscores only
            assert driver.isupper() or driver.replace("_", "").isupper(), f"Driver '{driver}' must be uppercase reason code"
            # No descriptive text (no lowercase words, no colons, no parentheses)
            assert ":" not in driver, f"Driver '{driver}' must not contain prose (no colons)"
            assert "(" not in driver, f"Driver '{driver}' must not contain prose (no parentheses)"
            assert " " not in driver, f"Driver '{driver}' must not contain prose (no spaces)"

        # Verify expected reason codes are present
        expected_codes = [
            "DRIVER_FUSION_TENSION_OR_CONFLICT",
            "DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT",
            "DRIVER_HIGH_RISK_BAND_PRESENT",
        ]
        for code in expected_codes:
            if code in drivers:
                assert code == drivers[drivers.index(code)], f"Reason code '{code}' must match exactly"

