"""
Tests for chronicle risk register status extraction in status generator.

SHADOW MODE: These tests verify that chronicle risk register signals are
correctly extracted from evidence packs and included in status JSON.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from scripts.generate_first_light_status import generate_status


class TestChronicleRiskRegisterStatusExtraction(unittest.TestCase):
    """Tests for chronicle risk register status extraction."""

    def test_chronicle_risk_signal_extracted_from_manifest(self) -> None:
        """Verify chronicle risk signal is extracted from manifest (preferred source)."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Create test evidence pack with chronicle risk register in manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            # Create minimal P3/P4 check artifacts
            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Create snapshots and risk register
            snapshots = [
                build_cal_exp_recurrence_snapshot("cal_001", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_002", {
                    "recurrence_likelihood": 0.3,
                    "band": "LOW",
                    "invariants_ok": True,
                }),
            ]

            risk_register = build_chronicle_risk_register(snapshots)

            # Create manifest.json with chronicle risk register (preferred source)
            manifest = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "chronicle_risk_register": risk_register,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            # Also create evidence.json (should be ignored when manifest has it)
            evidence = {
                "evidence_type": "test",
                "governance": {
                    "chronicle_risk_register": {
                        "total_calibrations": 999,  # Wrong value to verify manifest is used
                        "high_risk_calibrations": [],
                        "high_risk_details": [],
                    },
                },
            }
            (evidence_pack_dir / "evidence.json").write_text(
                json.dumps(evidence, indent=2)
            )

            # Generate status
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Verify chronicle_risk signal is present
            signals = status.get("signals")
            assert signals is not None
            assert "chronicle_risk" in signals

            chronicle_risk = signals["chronicle_risk"]
            assert "total_calibrations" in chronicle_risk
            assert "high_risk_count" in chronicle_risk
            assert "high_risk_cal_ids_top3" in chronicle_risk
            assert "has_any_invariants_violated" in chronicle_risk

            # Verify values from manifest (not evidence.json)
            assert chronicle_risk["total_calibrations"] == 2  # From manifest, not 999
            assert chronicle_risk["high_risk_count"] == 1
            assert "cal_001" in chronicle_risk["high_risk_cal_ids_top3"]
            assert len(chronicle_risk["high_risk_cal_ids_top3"]) == 1
            assert chronicle_risk["has_any_invariants_violated"] is True
            assert chronicle_risk["extraction_source"] == "MANIFEST"

    def test_chronicle_risk_signal_extracted_from_evidence_fallback(self) -> None:
        """Verify chronicle risk signal falls back to evidence.json when not in manifest."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Create test evidence pack with chronicle risk register only in evidence.json
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            # Create minimal P3/P4 check artifacts
            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Create evidence.json with chronicle risk register
            snapshots = [
                build_cal_exp_recurrence_snapshot("cal_001", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_002", {
                    "recurrence_likelihood": 0.3,
                    "band": "LOW",
                    "invariants_ok": True,
                }),
            ]

            risk_register = build_chronicle_risk_register(snapshots)
            evidence = {
                "evidence_type": "test",
                "governance": {
                    "chronicle_risk_register": risk_register,
                },
            }

            evidence_json_path = evidence_pack_dir / "evidence.json"
            with open(evidence_json_path, "w", encoding="utf-8") as f:
                json.dump(evidence, f, indent=2)

            # Create manifest.json WITHOUT chronicle_risk_register
            manifest = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {},  # Empty governance, should fallback to evidence.json
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            # Generate status
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Verify chronicle_risk signal is present (from evidence.json fallback)
            signals = status.get("signals")
            assert signals is not None
            assert "chronicle_risk" in signals

            chronicle_risk = signals["chronicle_risk"]
            assert "total_calibrations" in chronicle_risk
            assert "high_risk_count" in chronicle_risk
            assert "high_risk_cal_ids_top3" in chronicle_risk
            assert "has_any_invariants_violated" in chronicle_risk

            # Verify values
            assert chronicle_risk["total_calibrations"] == 2
            assert chronicle_risk["high_risk_count"] == 1
            assert "cal_001" in chronicle_risk["high_risk_cal_ids_top3"]
            assert len(chronicle_risk["high_risk_cal_ids_top3"]) == 1
            assert chronicle_risk["has_any_invariants_violated"] is True
            assert chronicle_risk["extraction_source"] == "EVIDENCE_JSON"

    def test_chronicle_risk_signal_top3_limit(self) -> None:
        """Verify high_risk_cal_ids_top3 is limited to top 3."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Create 5 high-risk calibrations
            snapshots = [
                build_cal_exp_recurrence_snapshot("cal_001", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_002", {
                    "recurrence_likelihood": 0.85,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_003", {
                    "recurrence_likelihood": 0.9,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_004", {
                    "recurrence_likelihood": 0.75,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_005", {
                    "recurrence_likelihood": 0.7,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
            ]

            risk_register = build_chronicle_risk_register(snapshots)
            evidence = {
                "evidence_type": "test",
                "governance": {
                    "chronicle_risk_register": risk_register,
                },
            }

            evidence_json_path = evidence_pack_dir / "evidence.json"
            with open(evidence_json_path, "w", encoding="utf-8") as f:
                json.dump(evidence, f, indent=2)

            manifest = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            signals = status.get("signals")
            assert signals is not None
            assert "chronicle_risk" in signals

            chronicle_risk = signals["chronicle_risk"]
            assert chronicle_risk["high_risk_count"] == 5
            assert len(chronicle_risk["high_risk_cal_ids_top3"]) == 3
            # Should be sorted (alphabetically)
            assert chronicle_risk["high_risk_cal_ids_top3"] == ["cal_001", "cal_002", "cal_003"]
            assert chronicle_risk["has_any_invariants_violated"] is True

    def test_chronicle_risk_signal_missing_evidence(self) -> None:
        """Verify status generation works when evidence.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            manifest = {
                "mode": "SHADOW",
                "file_count": 0,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            # Should not raise error when evidence.json is missing
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # chronicle_risk signal should not be present
            signals = status.get("signals")
            if signals:
                assert "chronicle_risk" not in signals

    def test_chronicle_risk_signal_missing_register(self) -> None:
        """Verify status generation works when chronicle_risk_register is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Create evidence.json without chronicle_risk_register
            evidence = {
                "evidence_type": "test",
                "governance": {},
            }

            evidence_json_path = evidence_pack_dir / "evidence.json"
            with open(evidence_json_path, "w", encoding="utf-8") as f:
                json.dump(evidence, f, indent=2)

            manifest = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # chronicle_risk signal should not be present
            signals = status.get("signals")
            if signals:
                assert "chronicle_risk" not in signals

    def test_chronicle_risk_signal_deterministic_ordering(self) -> None:
        """Verify high_risk_cal_ids_top3 ordering is deterministic."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Create snapshots in different orders
            snapshots1 = [
                build_cal_exp_recurrence_snapshot("cal_z", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_a", {
                    "recurrence_likelihood": 0.9,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_m", {
                    "recurrence_likelihood": 0.85,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
            ]

            snapshots2 = [
                build_cal_exp_recurrence_snapshot("cal_m", {
                    "recurrence_likelihood": 0.85,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_a", {
                    "recurrence_likelihood": 0.9,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
                build_cal_exp_recurrence_snapshot("cal_z", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,
                }),
            ]

            # Test with manifest
            risk_register1 = build_chronicle_risk_register(snapshots1)
            manifest1 = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "chronicle_risk_register": risk_register1,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest1, indent=2)
            )

            status1 = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Test with different order
            risk_register2 = build_chronicle_risk_register(snapshots2)
            manifest2 = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "chronicle_risk_register": risk_register2,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest2, indent=2)
            )

            status2 = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Both should produce same ordered top3
            signals1 = status1.get("signals", {})
            signals2 = status2.get("signals", {})
            
            assert "chronicle_risk" in signals1
            assert "chronicle_risk" in signals2
            
            top3_1 = signals1["chronicle_risk"]["high_risk_cal_ids_top3"]
            top3_2 = signals2["chronicle_risk"]["high_risk_cal_ids_top3"]
            
            # Should be deterministically sorted alphabetically
            assert top3_1 == top3_2 == ["cal_a", "cal_m", "cal_z"]

    def test_chronicle_risk_signal_has_any_invariants_violated(self) -> None:
        """Verify has_any_invariants_violated field is correctly set."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p3_dir = evidence_pack_dir / "p3"
            p4_dir = evidence_pack_dir / "p4"
            p3_dir.mkdir(parents=True)
            p4_dir.mkdir(parents=True)

            (p3_dir / "fl_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "cycles": 100})
            )
            (p4_dir / "p4_summary.json").write_text(
                json.dumps({"mode": "SHADOW", "uplift_metrics": {}})
            )

            # Test with invariants violated
            snapshots_with_violations = [
                build_cal_exp_recurrence_snapshot("cal_001", {
                    "recurrence_likelihood": 0.8,
                    "band": "HIGH",
                    "invariants_ok": False,  # Violated
                }),
            ]

            risk_register = build_chronicle_risk_register(snapshots_with_violations)
            manifest = {
                "mode": "SHADOW",
                "file_count": 1,
                "files": [],
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "chronicle_risk_register": risk_register,
                },
            }
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            signals = status.get("signals")
            assert signals is not None
            assert "chronicle_risk" in signals
            assert signals["chronicle_risk"]["has_any_invariants_violated"] is True

            # Test with no violations (only LOW/MEDIUM bands)
            snapshots_no_violations = [
                build_cal_exp_recurrence_snapshot("cal_002", {
                    "recurrence_likelihood": 0.3,
                    "band": "LOW",
                    "invariants_ok": True,
                }),
            ]

            risk_register_no_violations = build_chronicle_risk_register(snapshots_no_violations)
            manifest["governance"]["chronicle_risk_register"] = risk_register_no_violations
            (evidence_pack_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            status2 = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Should still have chronicle_risk signal (register exists, just no high-risk)
            # But has_any_invariants_violated should be False
            signals2 = status2.get("signals")
            assert signals2 is not None
            assert "chronicle_risk" in signals2
            assert signals2["chronicle_risk"]["has_any_invariants_violated"] is False
            assert signals2["chronicle_risk"]["high_risk_count"] == 0

