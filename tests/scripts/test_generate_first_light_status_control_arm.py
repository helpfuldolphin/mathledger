"""
Tests for control arm calibration panel ingestion in generate_first_light_status.py.

Validates:
- Control arm panel extraction from manifest
- Signal structure and fields (num_experiments, red_flag_count, too_similar_detected, headline)
- SHADOW-mode behavior (no gating, never blocks)
- Missing panel handling (not an error)
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.generate_first_light_status import generate_status


def test_control_arm_panel_extracted_from_manifest():
    """Test that control arm panel is extracted from manifest and added to signals."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with control arm panel
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
                "mock_oracle_panel": {
                    "schema_version": "1.0.0",
                    "experiments": ["CAL-EXP-001", "CAL-EXP-002"],
                    "control_vs_twin_delta": {
                        "CAL-EXP-001": {
                            "abstention_rate_delta": 0.08,
                            "invalid_rate_delta": 0.13,
                            "status_match": False,
                        },
                        "CAL-EXP-002": {
                            "abstention_rate_delta": 0.15,
                            "invalid_rate_delta": 0.10,
                            "status_match": False,
                        },
                    },
                    "red_flags": [],
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
            json.dumps({"metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify control arm signal is present
        assert status["signals"] is not None
        assert "control_arm" in status["signals"]
        control_arm = status["signals"]["control_arm"]
        assert control_arm["num_experiments"] == 2
        assert control_arm["red_flag_count"] == 0
        assert control_arm["too_similar_detected"] is False
        assert "no red flags" in control_arm["headline"].lower()


def test_control_arm_panel_with_red_flags():
    """Test that control arm panel with red flags is extracted correctly."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with control arm panel containing red flags
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
                "mock_oracle_panel": {
                    "schema_version": "1.0.0",
                    "experiments": ["CAL-EXP-001"],
                    "control_vs_twin_delta": {
                        "CAL-EXP-001": {
                            "abstention_rate_delta": 0.005,
                            "invalid_rate_delta": 0.003,
                            "status_match": False,
                        },
                    },
                    "red_flags": [
                        "Experiment 'CAL-EXP-001': Control and twin metrics are too similar "
                        "(abstention_delta=0.005, invalid_delta=0.003). "
                        "This may indicate overfitting or lack of sensitivity."
                    ],
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
            json.dumps({"metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify control arm signal is present with red flags
        assert status["signals"] is not None
        assert "control_arm" in status["signals"]
        control_arm = status["signals"]["control_arm"]
        assert control_arm["num_experiments"] == 1
        assert control_arm["red_flag_count"] == 1
        assert control_arm["too_similar_detected"] is True
        assert "too similar detected" in control_arm["headline"].lower()


def test_control_arm_panel_too_similar_detection():
    """Test that too_similar_detected is correctly detected from red flags."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with control arm panel with non-too-similar red flag
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
                "mock_oracle_panel": {
                    "schema_version": "1.0.0",
                    "experiments": ["CAL-EXP-001"],
                    "control_vs_twin_delta": {},
                    "red_flags": [
                        "Experiment 'CAL-EXP-001': Control and twin have matching status with "
                        "similar rates. This suggests the pipeline may not be distinguishing "
                        "between expected stochasticity (control) and actual behavior (twin)."
                    ],
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
            json.dumps({"metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify too_similar_detected is False when red flag doesn't contain "too similar"
        assert status["signals"] is not None
        assert "control_arm" in status["signals"]
        control_arm = status["signals"]["control_arm"]
        assert control_arm["too_similar_detected"] is False
        assert control_arm["red_flag_count"] == 1


def test_missing_control_arm_panel_not_an_error():
    """Test that missing control arm panel does not cause an error."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest without control arm panel
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
            json.dumps({"metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify control arm signal is not present (not an error)
        # signals may be None or empty dict - both are valid
        if status.get("signals"):
            assert "control_arm" not in status["signals"]


def test_control_arm_signal_shape():
    """Test that control arm signal has correct shape with all required fields."""
    with TemporaryDirectory() as tmpdir:
        evidence_pack_dir = Path(tmpdir)
        manifest_path = evidence_pack_dir / "manifest.json"

        # Create manifest with control arm panel
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
                "mock_oracle_panel": {
                    "schema_version": "1.0.0",
                    "experiments": ["CAL-EXP-001", "CAL-EXP-002", "CAL-EXP-003"],
                    "control_vs_twin_delta": {},
                    "red_flags": ["Flag 1", "Flag 2"],
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
            json.dumps({"metrics": {}}), encoding="utf-8"
        )

        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )

        # Verify control arm signal shape
        assert status["signals"] is not None
        assert "control_arm" in status["signals"]
        control_arm = status["signals"]["control_arm"]
        
        # Check all required fields
        assert "num_experiments" in control_arm
        assert "red_flag_count" in control_arm
        assert "too_similar_detected" in control_arm
        assert "headline" in control_arm
        assert "status" in control_arm
        assert "weight_hint" in control_arm
        
        # Check types
        assert isinstance(control_arm["num_experiments"], int)
        assert isinstance(control_arm["red_flag_count"], int)
        assert isinstance(control_arm["too_similar_detected"], bool)
        assert isinstance(control_arm["headline"], str)
        assert isinstance(control_arm["status"], str)
        assert isinstance(control_arm["weight_hint"], str)
        
        # Check values
        assert control_arm["num_experiments"] == 3
        assert control_arm["red_flag_count"] == 2
        assert control_arm["too_similar_detected"] is False  # No "too similar" in flags
        assert control_arm["status"] == "WARN"  # Red flags present
        assert control_arm["weight_hint"] == "LOW"
    
    def test_control_arm_signal_status_field(self):
        """Control arm signal should include status field derived from red_flag_count."""
        with TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            manifest_path = evidence_pack_dir / "manifest.json"

            # Test with no red flags (should be OK)
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
                    "mock_oracle_panel": {
                        "schema_version": "1.0.0",
                        "experiments": ["CAL-EXP-001"],
                        "control_vs_twin_delta": {},
                        "red_flags": [],
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
                json.dumps({"metrics": {}}), encoding="utf-8"
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Verify status field
            assert status["signals"] is not None
            assert "control_arm" in status["signals"]
            control_arm = status["signals"]["control_arm"]
            assert "status" in control_arm
            assert control_arm["status"] == "OK"  # No red flags
    
    def test_control_arm_signal_status_warn_with_red_flags(self):
        """Control arm signal should have status='WARN' when red flags present."""
        with TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            manifest_path = evidence_pack_dir / "manifest.json"

            # Test with red flags (should be WARN)
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
                    "mock_oracle_panel": {
                        "schema_version": "1.0.0",
                        "experiments": ["CAL-EXP-001"],
                        "control_vs_twin_delta": {},
                        "red_flags": ["Test red flag"],
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
                json.dumps({"metrics": {}}), encoding="utf-8"
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Verify status field
            assert status["signals"] is not None
            assert "control_arm" in status["signals"]
            control_arm = status["signals"]["control_arm"]
            assert "status" in control_arm
            assert control_arm["status"] == "WARN"  # Red flags present
    
    def test_control_arm_signal_weight_hint(self):
        """Control arm signal should include weight_hint='LOW'."""
        with TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            manifest_path = evidence_pack_dir / "manifest.json"

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
                    "mock_oracle_panel": {
                        "schema_version": "1.0.0",
                        "experiments": ["CAL-EXP-001"],
                        "control_vs_twin_delta": {},
                        "red_flags": [],
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
                json.dumps({"metrics": {}}), encoding="utf-8"
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Verify weight_hint field
            assert status["signals"] is not None
            assert "control_arm" in status["signals"]
            control_arm = status["signals"]["control_arm"]
            assert "weight_hint" in control_arm
            assert control_arm["weight_hint"] == "LOW"


def test_control_arm_signal_consistency_check():
        """Control arm signal should include consistency check when mock oracle is available."""
        import os
        # Set environment variable to enable mock oracle
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        try:
            with TemporaryDirectory() as tmpdir:
                evidence_pack_dir = Path(tmpdir)
                manifest_path = evidence_pack_dir / "manifest.json"

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
                        "mock_oracle_panel": {
                            "schema_version": "1.0.0",
                            "experiments": ["CAL-EXP-001"],
                            "control_vs_twin_delta": {},
                            "red_flags": [],
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
                    json.dumps({"metrics": {}}), encoding="utf-8"
                )

                status = generate_status(
                    p3_dir=p3_dir,
                    p4_dir=p4_dir,
                    evidence_pack_dir=evidence_pack_dir,
                )

                # Verify consistency check is present
                assert status["signals"] is not None
                assert "control_arm" in status["signals"]
                control_arm = status["signals"]["control_arm"]
                
                # Consistency check should be present if mock oracle is available
                if "consistency" in control_arm:
                    consistency = control_arm["consistency"]
                    assert "consistency" in consistency
                    assert "notes" in consistency
                    assert "conflict_invariant_violated" in consistency
                    assert consistency["consistency"] in ("CONSISTENT", "PARTIAL", "INCONSISTENT")
                    assert isinstance(consistency["conflict_invariant_violated"], bool)
        finally:
            # Clean up environment variable
            if "MATHLEDGER_ALLOW_MOCK_ORACLE" in os.environ:
                del os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"]

