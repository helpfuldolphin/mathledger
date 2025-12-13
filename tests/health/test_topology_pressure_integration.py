"""
Tests for topology pressure integration functions.

Tests cover:
- Evidence attachment
- P3 stability report integration
- P4 calibration report integration
- Uplift council adapter
"""

import json
import unittest
from typing import Any, Dict, List

from backend.health.topology_pressure_adapter import (
    attach_topology_pressure_to_evidence,
    summarize_topology_pressure_for_uplift_council,
    add_topology_pressure_to_p3_stability_report,
    add_topology_pressure_to_p4_calibration_report,
    build_topology_pressure_governance_tile,
    build_first_light_topology_stress_summary,
    build_cal_exp_topology_stress_snapshot,
    persist_cal_exp_topology_stress_snapshot,
    build_topology_stress_panel,
    build_topology_hotspot_ledger,
    extract_topology_hotspot_ledger_signal,
    extract_topology_hotspot_ledger_signal_from_evidence,
    extract_topology_hotspot_ledger_warnings,
    attach_topology_stress_panel_to_evidence,
    attach_topology_hotspot_ledger_signal_to_evidence,
)


class TestTopologyPressureEvidenceIntegration(unittest.TestCase):
    """Tests for attach_topology_pressure_to_evidence()."""

    def _make_tile(self) -> Dict[str, Any]:
        """Helper to create mock topology pressure tile."""
        return {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "pressure_band": "LOW",
            "promotion_gate_status": "OK",
            "pressure_hotspots": [],
            "headline": "Topology pressure status: stable",
        }

    def test_01_attaches_to_evidence(self):
        """Test tile is attached to evidence under governance key."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        tile = self._make_tile()

        result = attach_topology_pressure_to_evidence(evidence, tile)

        self.assertIn("governance", result)
        self.assertIn("topology_pressure", result["governance"])

    def test_02_extracts_required_fields(self):
        """Test required fields are extracted correctly."""
        evidence = {"timestamp": "2024-01-01"}
        tile = self._make_tile()
        tile["pressure_band"] = "HIGH"
        tile["pressure_hotspots"] = ["Depth trend contributing to pressure"]
        tile["headline"] = "Topology pressure: elevated"

        result = attach_topology_pressure_to_evidence(evidence, tile)

        summary = result["governance"]["topology_pressure"]
        self.assertEqual(summary["pressure_band"], "HIGH")
        self.assertEqual(summary["pressure_hotspots"], ["Depth trend contributing to pressure"])
        self.assertEqual(summary["headline"], "Topology pressure: elevated")

    def test_03_non_mutating(self):
        """Test function does not modify input evidence."""
        evidence = {"timestamp": "2024-01-01", "governance": {"existing": "data"}}
        tile = self._make_tile()

        result = attach_topology_pressure_to_evidence(evidence, tile)

        # Original evidence should be unchanged
        self.assertNotIn("topology_pressure", evidence.get("governance", {}))
        # Result should have new field
        self.assertIn("topology_pressure", result["governance"])

    def test_04_handles_missing_governance(self):
        """Test function creates governance section if missing."""
        evidence = {"timestamp": "2024-01-01"}
        tile = self._make_tile()

        result = attach_topology_pressure_to_evidence(evidence, tile)

        self.assertIn("governance", result)
        self.assertIn("topology_pressure", result["governance"])


class TestTopologyPressureCouncilAdapter(unittest.TestCase):
    """Tests for summarize_topology_pressure_for_uplift_council()."""

    def test_01_maps_high_to_block(self):
        """Test HIGH pressure band maps to BLOCK status."""
        tile = {
            "pressure_band": "HIGH",
            "pressure_hotspots": ["test"],
            "headline": "High pressure",
        }

        result = summarize_topology_pressure_for_uplift_council(tile)

        self.assertEqual(result["status"], "BLOCK")
        self.assertEqual(result["pressure_band"], "HIGH")

    def test_02_maps_medium_to_warn(self):
        """Test MEDIUM pressure band maps to WARN status."""
        tile = {
            "pressure_band": "MEDIUM",
            "pressure_hotspots": [],
            "headline": "Medium pressure",
        }

        result = summarize_topology_pressure_for_uplift_council(tile)

        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["pressure_band"], "MEDIUM")

    def test_03_maps_low_to_ok(self):
        """Test LOW pressure band maps to OK status."""
        tile = {
            "pressure_band": "LOW",
            "pressure_hotspots": [],
            "headline": "Low pressure",
        }

        result = summarize_topology_pressure_for_uplift_council(tile)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["pressure_band"], "LOW")

    def test_04_includes_all_fields(self):
        """Test summary includes all required fields."""
        tile = {
            "pressure_band": "MEDIUM",
            "pressure_hotspots": ["hotspot1", "hotspot2"],
            "headline": "Test headline",
        }

        result = summarize_topology_pressure_for_uplift_council(tile)

        self.assertIn("status", result)
        self.assertIn("pressure_band", result)
        self.assertIn("pressure_hotspots", result)
        self.assertIn("headline", result)
        self.assertEqual(result["pressure_hotspots"], ["hotspot1", "hotspot2"])


class TestTopologyPressureP3Integration(unittest.TestCase):
    """Tests for add_topology_pressure_to_p3_stability_report()."""

    def _make_stability_report(self) -> Dict[str, Any]:
        """Helper to create mock P3 stability report."""
        return {
            "schema_version": "1.0.0",
            "run_id": "test_run",
            "config": {"slice_name": "test_slice"},
            "timing": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
            "metrics": {},
            "criteria_evaluation": {"all_passed": True},
        }

    def _make_tile(self) -> Dict[str, Any]:
        """Helper to create mock topology pressure tile."""
        return {
            "pressure_band": "MEDIUM",
            "status_light": "YELLOW",
            "pressure_hotspots": ["Depth trend contributing"],
            "headline": "Topology pressure: moderate",
        }

    def test_01_adds_topology_pressure_summary(self):
        """Test topology_pressure_summary is added to report."""
        report = self._make_stability_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p3_stability_report(report, tile)

        self.assertIn("topology_pressure_summary", result)

    def test_02_includes_required_fields(self):
        """Test summary includes all required fields."""
        report = self._make_stability_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p3_stability_report(report, tile)

        summary = result["topology_pressure_summary"]
        self.assertEqual(summary["pressure_band"], "MEDIUM")
        self.assertEqual(summary["status_light"], "YELLOW")
        self.assertEqual(summary["pressure_hotspots"], ["Depth trend contributing"])
        self.assertEqual(summary["headline"], "Topology pressure: moderate")

    def test_03_non_mutating(self):
        """Test function does not modify input report."""
        report = self._make_stability_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p3_stability_report(report, tile)

        # Original report should be unchanged
        self.assertNotIn("topology_pressure_summary", report)
        # Result should have new field
        self.assertIn("topology_pressure_summary", result)

    def test_04_preserves_existing_fields(self):
        """Test function preserves existing report fields."""
        report = self._make_stability_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p3_stability_report(report, tile)

        self.assertEqual(result["run_id"], "test_run")
        self.assertEqual(result["config"]["slice_name"], "test_slice")


class TestTopologyPressureP4Integration(unittest.TestCase):
    """Tests for add_topology_pressure_to_p4_calibration_report()."""

    def _make_calibration_report(self) -> Dict[str, Any]:
        """Helper to create mock P4 calibration report."""
        return {
            "schema_version": "1.0.0",
            "run_id": "test_run_p4",
            "timing": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
            "divergence_statistics": {},
            "accuracy_metrics": {},
            "calibration_assessment": {"twin_validity": "VALID"},
        }

    def _make_tile(self) -> Dict[str, Any]:
        """Helper to create mock topology pressure tile."""
        return {
            "pressure_band": "HIGH",
            "status_light": "RED",
            "pressure_hotspots": ["Risk envelope contributing"],
            "headline": "Topology pressure: elevated",
            "pressure_components": {
                "depth": 0.8,
                "branching": 0.6,
                "risk": 0.7,
            },
        }

    def test_01_adds_topology_pressure_calibration(self):
        """Test topology_pressure_calibration is added to report."""
        report = self._make_calibration_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p4_calibration_report(report, tile)

        self.assertIn("topology_pressure_calibration", result)

    def test_02_includes_required_fields(self):
        """Test calibration includes all required fields."""
        report = self._make_calibration_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p4_calibration_report(report, tile)

        calibration = result["topology_pressure_calibration"]
        self.assertEqual(calibration["pressure_band"], "HIGH")
        self.assertEqual(calibration["status_light"], "RED")
        self.assertEqual(calibration["pressure_hotspots"], ["Risk envelope contributing"])
        self.assertIn("structural_notes", calibration)

    def test_03_includes_structural_notes(self):
        """Test structural notes are included from pressure components."""
        report = self._make_calibration_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p4_calibration_report(report, tile)

        calibration = result["topology_pressure_calibration"]
        self.assertIsInstance(calibration["structural_notes"], list)
        self.assertGreater(len(calibration["structural_notes"]), 0)

    def test_04_non_mutating(self):
        """Test function does not modify input report."""
        report = self._make_calibration_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p4_calibration_report(report, tile)

        # Original report should be unchanged
        self.assertNotIn("topology_pressure_calibration", report)
        # Result should have new field
        self.assertIn("topology_pressure_calibration", result)

    def test_05_preserves_existing_fields(self):
        """Test function preserves existing report fields."""
        report = self._make_calibration_report()
        tile = self._make_tile()

        result = add_topology_pressure_to_p4_calibration_report(report, tile)

        self.assertEqual(result["run_id"], "test_run_p4")
        self.assertEqual(result["calibration_assessment"]["twin_validity"], "VALID")


class TestFirstLightTopologyStressSummary(unittest.TestCase):
    """Tests for build_first_light_topology_stress_summary()."""

    def _make_p3_summary(self) -> Dict[str, Any]:
        """Helper to create mock P3 topology summary."""
        return {
            "pressure_band": "MEDIUM",
            "status_light": "YELLOW",
            "pressure_hotspots": ["Depth trend contributing"],
            "headline": "Topology pressure: moderate",
        }

    def _make_p4_calibration(self) -> Dict[str, Any]:
        """Helper to create mock P4 topology calibration."""
        return {
            "pressure_band": "HIGH",
            "status_light": "RED",
            "pressure_hotspots": ["Risk envelope contributing"],
            "structural_notes": [],
        }

    def test_01_builds_stress_summary(self):
        """Test stress summary is built with required fields."""
        p3_summary = self._make_p3_summary()
        p4_calibration = self._make_p4_calibration()

        result = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        self.assertIn("schema_version", result)
        self.assertIn("p3_pressure_band", result)
        self.assertIn("p4_pressure_band", result)
        self.assertIn("pressure_hotspots", result)

    def test_02_extracts_pressure_bands(self):
        """Test pressure bands are extracted correctly."""
        p3_summary = self._make_p3_summary()
        p4_calibration = self._make_p4_calibration()

        result = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        self.assertEqual(result["p3_pressure_band"], "MEDIUM")
        self.assertEqual(result["p4_pressure_band"], "HIGH")

    def test_03_merges_hotspots(self):
        """Test hotspots are merged and deduplicated."""
        p3_summary = self._make_p3_summary()
        p3_summary["pressure_hotspots"] = ["hotspot1", "hotspot2"]
        p4_calibration = self._make_p4_calibration()
        p4_calibration["pressure_hotspots"] = ["hotspot2", "hotspot3"]

        result = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        # Should have 3 unique hotspots (hotspot2 deduplicated)
        self.assertEqual(len(result["pressure_hotspots"]), 3)
        self.assertIn("hotspot1", result["pressure_hotspots"])
        self.assertIn("hotspot2", result["pressure_hotspots"])
        self.assertIn("hotspot3", result["pressure_hotspots"])

    def test_04_limits_hotspots_to_five(self):
        """Test hotspots are limited to maximum of 5.
        
        Note: The 5-hotspot limit is for human legibility and presentation
        constraints, not because only 5 hotspots matter. This limit keeps the
        summary concise for external reviewers while acknowledging that topology
        stress may manifest through many component-level signals. The limit is
        a presentation constraint, not a semantic constraint on topology analysis.
        """
        p3_summary = self._make_p3_summary()
        p3_summary["pressure_hotspots"] = ["hotspot1", "hotspot2", "hotspot3"]
        p4_calibration = self._make_p4_calibration()
        p4_calibration["pressure_hotspots"] = ["hotspot4", "hotspot5", "hotspot6", "hotspot7"]

        result = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        # Should be limited to 5 (for legibility, not semantic significance)
        self.assertLessEqual(len(result["pressure_hotspots"]), 5)

    def test_05_deterministic_output(self):
        """Test output is deterministic for same inputs."""
        p3_summary = self._make_p3_summary()
        p4_calibration = self._make_p4_calibration()

        result1 = build_first_light_topology_stress_summary(p3_summary, p4_calibration)
        result2 = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        # Compare as JSON strings for exact match
        import json
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)

        self.assertEqual(json1, json2, "Stress summary output is not deterministic")

    def test_06_handles_missing_hotspots(self):
        """Test function handles missing hotspots gracefully."""
        p3_summary = {"pressure_band": "LOW"}
        p4_calibration = {"pressure_band": "LOW"}

        result = build_first_light_topology_stress_summary(p3_summary, p4_calibration)

        self.assertEqual(result["pressure_hotspots"], [])

    def test_07_evidence_attachment_includes_stress_summary(self):
        """Test attach_topology_pressure_to_evidence includes stress summary when P3/P4 provided."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "pressure_band": "MEDIUM",
            "pressure_hotspots": [],
            "headline": "Test",
        }
        p3_summary = self._make_p3_summary()
        p4_calibration = self._make_p4_calibration()

        result = attach_topology_pressure_to_evidence(
            evidence, tile, p3_summary, p4_calibration
        )

        self.assertIn("governance", result)
        self.assertIn("first_light_topology_stress", result["governance"])
        stress = result["governance"]["first_light_topology_stress"]
        self.assertEqual(stress["p3_pressure_band"], "MEDIUM")
        self.assertEqual(stress["p4_pressure_band"], "HIGH")

    def test_08_evidence_attachment_without_p3_p4(self):
        """Test attach_topology_pressure_to_evidence works without P3/P4 summaries."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "pressure_band": "MEDIUM",
            "pressure_hotspots": [],
            "headline": "Test",
        }

        result = attach_topology_pressure_to_evidence(evidence, tile)

        self.assertIn("governance", result)
        self.assertIn("topology_pressure", result["governance"])
        # Should not have stress summary if P3/P4 not provided
        self.assertNotIn("first_light_topology_stress", result.get("governance", {}))

    def test_09_evidence_attachment_non_mutating(self):
        """Test evidence attachment does not modify input evidence."""
        evidence = {"timestamp": "2024-01-01", "governance": {"existing": "data"}}
        tile = {"pressure_band": "LOW", "pressure_hotspots": [], "headline": "Test"}
        p3_summary = self._make_p3_summary()
        p4_calibration = self._make_p4_calibration()

        result = attach_topology_pressure_to_evidence(
            evidence, tile, p3_summary, p4_calibration
        )

        # Original evidence should be unchanged
        self.assertNotIn("first_light_topology_stress", evidence.get("governance", {}))
        # Result should have new field
        self.assertIn("first_light_topology_stress", result["governance"])


class TestCalExpTopologyStressSnapshot(unittest.TestCase):
    """Tests for build_cal_exp_topology_stress_snapshot()."""

    def _make_stress_summary(self) -> Dict[str, Any]:
        """Helper to create mock stress summary."""
        return {
            "schema_version": "1.0.0",
            "p3_pressure_band": "MEDIUM",
            "p4_pressure_band": "HIGH",
            "pressure_hotspots": ["Depth trend contributing", "Risk envelope contributing"],
        }

    def test_01_builds_snapshot_with_required_fields(self):
        """Test snapshot has all required fields."""
        summary = self._make_stress_summary()
        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        required_fields = [
            "schema_version",
            "cal_id",
            "p3_pressure_band",
            "p4_pressure_band",
            "pressure_hotspots",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_02_includes_cal_id(self):
        """Test cal_id is included correctly."""
        summary = self._make_stress_summary()
        result = build_cal_exp_topology_stress_snapshot("cal_exp2", summary)

        self.assertEqual(result["cal_id"], "cal_exp2")

    def test_03_preserves_pressure_bands(self):
        """Test pressure bands are preserved from summary."""
        summary = self._make_stress_summary()
        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        self.assertEqual(result["p3_pressure_band"], "MEDIUM")
        self.assertEqual(result["p4_pressure_band"], "HIGH")

    def test_04_preserves_hotspots(self):
        """Test hotspots are preserved from summary."""
        summary = self._make_stress_summary()
        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        self.assertEqual(result["pressure_hotspots"], ["Depth trend contributing", "Risk envelope contributing"])

    def test_05_limits_hotspots_to_five(self):
        """Test hotspots are limited to maximum of 5."""
        summary = self._make_stress_summary()
        summary["pressure_hotspots"] = ["hotspot1", "hotspot2", "hotspot3", "hotspot4", "hotspot5", "hotspot6"]

        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        self.assertLessEqual(len(result["pressure_hotspots"]), 5)

    def test_06_json_serializable(self):
        """Test snapshot is JSON serializable."""
        import json
        summary = self._make_stress_summary()
        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        # Should not raise
        json_str = json.dumps(result, sort_keys=True)
        self.assertIsInstance(json_str, str)

        # Should deserialize correctly
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized, result)

    def test_07_deterministic_output(self):
        """Test snapshot output is deterministic."""
        import json
        summary = self._make_stress_summary()

        result1 = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)
        result2 = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        # Compare as JSON strings for exact match
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)

        self.assertEqual(json1, json2, "Snapshot output is not deterministic")

    def test_08_handles_missing_hotspots(self):
        """Test function handles missing hotspots gracefully."""
        summary = {
            "p3_pressure_band": "LOW",
            "p4_pressure_band": "LOW",
        }

        result = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        self.assertEqual(result["pressure_hotspots"], [])

    def test_09_persist_snapshot_writes_file(self):
        """Test persistence function writes snapshot to disk."""
        import tempfile
        from pathlib import Path

        summary = self._make_stress_summary()
        snapshot = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            path = persist_cal_exp_topology_stress_snapshot(snapshot, output_dir)

            self.assertTrue(path.exists())
            self.assertEqual(path.name, "topology_stress_cal_exp1.json")

    def test_10_persist_snapshot_json_safe(self):
        """Test persisted snapshot is valid JSON."""
        import json
        import tempfile
        from pathlib import Path

        summary = self._make_stress_summary()
        snapshot = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            path = persist_cal_exp_topology_stress_snapshot(snapshot, output_dir)

            # Should be readable as JSON
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["cal_id"], "cal_exp1")
            self.assertEqual(loaded["p3_pressure_band"], "MEDIUM")

    def test_11_persist_snapshot_deterministic(self):
        """Test persisted snapshot is deterministic."""
        import json
        import tempfile
        from pathlib import Path

        summary = self._make_stress_summary()
        snapshot = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            path1 = persist_cal_exp_topology_stress_snapshot(snapshot, output_dir)

            # Write again
            output_dir2 = Path(tmpdir) / "calibration2"
            path2 = persist_cal_exp_topology_stress_snapshot(snapshot, output_dir2)

            # Both files should have identical content
            with open(path1, "r", encoding="utf-8") as f1:
                content1 = f1.read()
            with open(path2, "r", encoding="utf-8") as f2:
                content2 = f2.read()

            self.assertEqual(content1, content2)


class TestTopologyStressPanel(unittest.TestCase):
    """Tests for build_topology_stress_panel()."""

    def _make_snapshot(self, cal_id: str, p3_band: str = "LOW", p4_band: str = "LOW") -> Dict[str, Any]:
        """Helper to create mock snapshot."""
        return {
            "schema_version": "1.0.0",
            "cal_id": cal_id,
            "p3_pressure_band": p3_band,
            "p4_pressure_band": p4_band,
            "pressure_hotspots": [],
        }

    def test_01_builds_panel_with_required_fields(self):
        """Test panel has all required fields."""
        snapshots = [
            self._make_snapshot("cal_exp1", "MEDIUM", "HIGH"),
            self._make_snapshot("cal_exp2", "LOW", "MEDIUM"),
        ]

        result = build_topology_stress_panel(snapshots)

        required_fields = ["schema_version", "panel_type", "experiments", "summary"]
        for field in required_fields:
            self.assertIn(field, result)

    def test_02_panel_type_is_heatmap(self):
        """Test panel_type is set correctly."""
        snapshots = [self._make_snapshot("cal_exp1")]

        result = build_topology_stress_panel(snapshots)

        self.assertEqual(result["panel_type"], "topology_stress_heatmap")

    def test_03_includes_all_experiments(self):
        """Test all snapshots are included in experiments list."""
        snapshots = [
            self._make_snapshot("cal_exp1", "MEDIUM", "HIGH"),
            self._make_snapshot("cal_exp2", "LOW", "MEDIUM"),
            self._make_snapshot("cal_exp3", "HIGH", "HIGH"),
        ]

        result = build_topology_stress_panel(snapshots)

        self.assertEqual(len(result["experiments"]), 3)
        self.assertEqual(result["experiments"][0]["cal_id"], "cal_exp1")
        self.assertEqual(result["experiments"][1]["cal_id"], "cal_exp2")
        self.assertEqual(result["experiments"][2]["cal_id"], "cal_exp3")

    def test_04_summary_includes_band_distributions(self):
        """Test summary includes band distribution counts."""
        snapshots = [
            self._make_snapshot("cal_exp1", "LOW", "LOW"),
            self._make_snapshot("cal_exp2", "MEDIUM", "MEDIUM"),
            self._make_snapshot("cal_exp3", "HIGH", "HIGH"),
        ]

        result = build_topology_stress_panel(snapshots)

        summary = result["summary"]
        self.assertIn("p3_band_distribution", summary)
        self.assertIn("p4_band_distribution", summary)
        self.assertEqual(summary["p3_band_distribution"]["LOW"], 1)
        self.assertEqual(summary["p3_band_distribution"]["MEDIUM"], 1)
        self.assertEqual(summary["p3_band_distribution"]["HIGH"], 1)

    def test_05_summary_includes_total_experiments(self):
        """Test summary includes total experiment count."""
        snapshots = [
            self._make_snapshot("cal_exp1"),
            self._make_snapshot("cal_exp2"),
        ]

        result = build_topology_stress_panel(snapshots)

        self.assertEqual(result["summary"]["total_experiments"], 2)

    def test_06_collects_common_hotspots(self):
        """Test panel collects common hotspots across experiments."""
        snapshot1 = self._make_snapshot("cal_exp1", "MEDIUM", "HIGH")
        snapshot1["pressure_hotspots"] = ["Depth trend", "Branching volatility"]
        snapshot2 = self._make_snapshot("cal_exp2", "LOW", "MEDIUM")
        snapshot2["pressure_hotspots"] = ["Depth trend", "Risk envelope"]

        result = build_topology_stress_panel([snapshot1, snapshot2])

        common_hotspots = result["summary"]["common_hotspots"]
        self.assertIn("Depth trend", common_hotspots)
        self.assertIn("Branching volatility", common_hotspots)
        self.assertIn("Risk envelope", common_hotspots)

    def test_07_non_mutating(self):
        """Test function does not modify input snapshots."""
        snapshot = self._make_snapshot("cal_exp1", "MEDIUM", "HIGH")
        original_cal_id = snapshot["cal_id"]

        result = build_topology_stress_panel([snapshot])

        # Original snapshot should be unchanged
        self.assertEqual(snapshot["cal_id"], original_cal_id)
        # Result should have panel structure
        self.assertIn("experiments", result)

    def test_08_handles_empty_snapshots(self):
        """Test function handles empty snapshot list."""
        result = build_topology_stress_panel([])

        self.assertEqual(len(result["experiments"]), 0)
        self.assertEqual(result["summary"]["total_experiments"], 0)

    def test_09_evidence_attachment_includes_panel(self):
        """Test attach_topology_stress_panel_to_evidence includes panel."""
        evidence = {"timestamp": "2024-01-01"}
        snapshots = [self._make_snapshot("cal_exp1", "MEDIUM", "HIGH")]
        panel = build_topology_stress_panel(snapshots)

        result = attach_topology_stress_panel_to_evidence(evidence, panel)

        self.assertIn("governance", result)
        self.assertIn("topology_stress_panel", result["governance"])
        self.assertEqual(result["governance"]["topology_stress_panel"]["panel_type"], "topology_stress_heatmap")

    def test_10_evidence_attachment_non_mutating(self):
        """Test evidence attachment does not modify input evidence."""
        evidence = {"timestamp": "2024-01-01", "governance": {"existing": "data"}}
        snapshots = [self._make_snapshot("cal_exp1")]
        panel = build_topology_stress_panel(snapshots)

        result = attach_topology_stress_panel_to_evidence(evidence, panel)

        # Original evidence should be unchanged
        self.assertNotIn("topology_stress_panel", evidence.get("governance", {}))
        # Result should have new field
        self.assertIn("topology_stress_panel", result["governance"])


class TestTopologyHotspotLedger(unittest.TestCase):
    """Tests for build_topology_hotspot_ledger()."""

    def _make_panel(self, experiments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Helper to create mock panel from experiment data."""
        experiments = []
        for exp_data in experiments_data:
            experiments.append({
                "cal_id": exp_data.get("cal_id", "unknown"),
                "p3_pressure_band": exp_data.get("p3_band", "LOW"),
                "p4_pressure_band": exp_data.get("p4_band", "LOW"),
                "pressure_hotspots": exp_data.get("hotspots", []),
            })
        return {
            "schema_version": "1.0.0",
            "panel_type": "topology_stress_heatmap",
            "experiments": experiments,
            "summary": {"total_experiments": len(experiments)},
        }

    def test_01_builds_ledger_with_required_fields(self):
        """Test ledger has all required fields."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["slice_uplift_tree"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        required_fields = ["schema_version", "hotspot_counts", "top_hotspots", "num_experiments"]
        for field in required_fields:
            self.assertIn(field, result)

    def test_02_counts_hotspots_correctly(self):
        """Test hotspot counts are computed correctly."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["slice_uplift_tree", "Depth trend"]},
            {"cal_id": "cal_exp2", "hotspots": ["slice_uplift_tree", "Risk envelope"]},
            {"cal_id": "cal_exp3", "hotspots": ["Depth trend"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        counts = result["hotspot_counts"]
        self.assertEqual(counts.get("slice_uplift_tree"), 2)
        self.assertEqual(counts.get("Depth trend"), 2)
        self.assertEqual(counts.get("Risk envelope"), 1)

    def test_03_sorted_keys_deterministic(self):
        """Test hotspot_counts keys are sorted for determinism."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["z_slice", "a_slice", "m_slice"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        keys = list(result["hotspot_counts"].keys())
        self.assertEqual(keys, sorted(keys), "Keys should be sorted for determinism")

    def test_04_top_hotspots_limited_to_ten(self):
        """Test top_hotspots list is limited to 10."""
        # Create panel with more than 10 unique hotspots
        hotspots_list = [f"hotspot_{i}" for i in range(15)]
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": hotspots_list},
        ])

        result = build_topology_hotspot_ledger(panel)

        self.assertLessEqual(len(result["top_hotspots"]), 10)

    def test_05_top_hotspots_sorted_by_count(self):
        """Test top_hotspots are sorted by count (descending)."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["hotspot_a", "hotspot_b", "hotspot_c"]},
            {"cal_id": "cal_exp2", "hotspots": ["hotspot_a", "hotspot_b"]},
            {"cal_id": "cal_exp3", "hotspots": ["hotspot_a"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        top_hotspots = result["top_hotspots"]
        # hotspot_a should be first (count=3), then hotspot_b (count=2), then hotspot_c (count=1)
        self.assertEqual(top_hotspots[0], "hotspot_a")
        self.assertEqual(top_hotspots[1], "hotspot_b")
        self.assertEqual(top_hotspots[2], "hotspot_c")

    def test_06_top_hotspots_tie_breaking_deterministic(self):
        """Test top_hotspots tie-breaking by name is deterministic."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["hotspot_z", "hotspot_a"]},
            {"cal_id": "cal_exp2", "hotspots": ["hotspot_z", "hotspot_a"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        top_hotspots = result["top_hotspots"]
        # Both have count=2, so should be sorted by name (a before z)
        self.assertEqual(top_hotspots[0], "hotspot_a")
        self.assertEqual(top_hotspots[1], "hotspot_z")

    def test_07_num_experiments_correct(self):
        """Test num_experiments matches panel experiments count."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": []},
            {"cal_id": "cal_exp2", "hotspots": []},
            {"cal_id": "cal_exp3", "hotspots": []},
        ])

        result = build_topology_hotspot_ledger(panel)

        self.assertEqual(result["num_experiments"], 3)

    def test_08_handles_empty_hotspots(self):
        """Test function handles experiments with no hotspots."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": []},
            {"cal_id": "cal_exp2", "hotspots": ["slice_uplift_tree"]},
        ])

        result = build_topology_hotspot_ledger(panel)

        self.assertEqual(result["num_experiments"], 2)
        self.assertEqual(result["hotspot_counts"].get("slice_uplift_tree"), 1)

    def test_09_non_mutating(self):
        """Test function does not modify input panel."""
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["slice_uplift_tree"]},
        ])
        original_experiments_count = len(panel["experiments"])

        result = build_topology_hotspot_ledger(panel)

        # Original panel should be unchanged
        self.assertEqual(len(panel["experiments"]), original_experiments_count)
        # Result should have ledger structure
        self.assertIn("hotspot_counts", result)

    def test_10_deterministic_output(self):
        """Test ledger output is deterministic."""
        import json
        panel = self._make_panel([
            {"cal_id": "cal_exp1", "hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "hotspots": ["slice_b", "slice_c"]},
        ])

        result1 = build_topology_hotspot_ledger(panel)
        result2 = build_topology_hotspot_ledger(panel)

        # Compare as JSON strings for exact match
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)

        self.assertEqual(json1, json2, "Ledger output is not deterministic")

    def test_11_evidence_attachment_includes_ledger(self):
        """Test evidence attachment includes hotspot ledger in panel."""
        evidence = {"timestamp": "2024-01-01"}
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_uplift_tree"]},
        ]
        panel = build_topology_stress_panel(snapshots)

        result = attach_topology_stress_panel_to_evidence(evidence, panel)

        self.assertIn("hotspot_ledger", result["governance"]["topology_stress_panel"])
        ledger = result["governance"]["topology_stress_panel"]["hotspot_ledger"]
        self.assertIn("hotspot_counts", ledger)
        self.assertIn("top_hotspots", ledger)


class TestTopologyHotspotLedgerSignal(unittest.TestCase):
    """Tests for topology hotspot ledger signal extraction and integration."""

    def _make_ledger(self, hotspot_counts: Dict[str, int], num_experiments: int = 3) -> Dict[str, Any]:
        """Helper to create mock ledger."""
        # Build top_hotspots from counts (sorted by count desc, then name asc)
        sorted_by_count = sorted(
            hotspot_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
        top_hotspots = [hotspot for hotspot, _ in sorted_by_count[:10]]
        sorted_counts = dict(sorted(hotspot_counts.items()))

        return {
            "schema_version": "1.0.0",
            "hotspot_counts": sorted_counts,
            "top_hotspots": top_hotspots,
            "num_experiments": num_experiments,
        }

    def test_01_extracts_signal_with_required_fields(self):
        """Test signal extraction includes all required fields."""
        ledger = self._make_ledger({"slice_a": 2, "slice_b": 1})

        signal = extract_topology_hotspot_ledger_signal(ledger)

        required_fields = [
            "schema_version",
            "num_experiments",
            "unique_hotspot_count",
            "top_hotspots_top3",
            "top_hotspot_counts_top3",
        ]
        for field in required_fields:
            self.assertIn(field, signal)

    def test_02_signal_num_experiments_correct(self):
        """Test signal num_experiments matches ledger."""
        ledger = self._make_ledger({"slice_a": 1}, num_experiments=5)

        signal = extract_topology_hotspot_ledger_signal(ledger)

        self.assertEqual(signal["num_experiments"], 5)

    def test_03_signal_unique_hotspot_count_correct(self):
        """Test signal unique_hotspot_count matches ledger."""
        ledger = self._make_ledger({"slice_a": 2, "slice_b": 1, "slice_c": 3})

        signal = extract_topology_hotspot_ledger_signal(ledger)

        self.assertEqual(signal["unique_hotspot_count"], 3)

    def test_04_signal_top_hotspots_top3_correct(self):
        """Test signal top_hotspots_top3 contains top 3 hotspots."""
        ledger = self._make_ledger({
            "slice_a": 3,
            "slice_b": 2,
            "slice_c": 1,
            "slice_d": 0,
        })

        signal = extract_topology_hotspot_ledger_signal(ledger)

        self.assertEqual(len(signal["top_hotspots_top3"]), 3)
        self.assertEqual(signal["top_hotspots_top3"][0], "slice_a")
        self.assertEqual(signal["top_hotspots_top3"][1], "slice_b")
        self.assertEqual(signal["top_hotspots_top3"][2], "slice_c")

    def test_05_signal_top_hotspot_counts_top3_correct(self):
        """Test signal top_hotspot_counts_top3 matches top 3 counts."""
        ledger = self._make_ledger({
            "slice_a": 3,
            "slice_b": 2,
            "slice_c": 1,
        })

        signal = extract_topology_hotspot_ledger_signal(ledger)

        self.assertEqual(signal["top_hotspot_counts_top3"], [3, 2, 1])

    def test_06_signal_handles_fewer_than_three_hotspots(self):
        """Test signal handles cases with fewer than 3 hotspots."""
        ledger = self._make_ledger({"slice_a": 2})

        signal = extract_topology_hotspot_ledger_signal(ledger)

        self.assertEqual(len(signal["top_hotspots_top3"]), 1)
        self.assertEqual(len(signal["top_hotspot_counts_top3"]), 1)

    def test_07_extract_from_evidence_with_ledger(self):
        """Test extraction from evidence when ledger is present."""
        ledger = self._make_ledger({"slice_a": 2})
        evidence = {
            "governance": {
                "topology_stress_panel": {
                    "hotspot_ledger": ledger,
                }
            }
        }

        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["num_experiments"], 3)

    def test_08_extract_from_evidence_missing_panel(self):
        """Test extraction from evidence when panel is missing."""
        evidence = {"governance": {}}

        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence)

        self.assertIsNone(signal)

    def test_09_extract_from_evidence_missing_ledger(self):
        """Test extraction from evidence when ledger is missing."""
        evidence = {
            "governance": {
                "topology_stress_panel": {},
            }
        }

        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence)

        self.assertIsNone(signal)

    def test_10_warnings_extracted_for_recurring_hotspots(self):
        """Test warnings are generated for hotspots with count >= 2."""
        ledger = self._make_ledger({
            "slice_a": 3,
            "slice_b": 2,
            "slice_c": 1,
        })

        warnings = extract_topology_hotspot_ledger_warnings(ledger)

        self.assertEqual(len(warnings), 2)  # slice_a and slice_b
        self.assertTrue(any("slice_a" in w for w in warnings))
        self.assertTrue(any("slice_b" in w for w in warnings))
        self.assertFalse(any("slice_c" in w for w in warnings))

    def test_11_warnings_include_count(self):
        """Test warnings include the count of occurrences."""
        ledger = self._make_ledger({"slice_a": 3})

        warnings = extract_topology_hotspot_ledger_warnings(ledger)

        self.assertEqual(len(warnings), 1)
        self.assertIn("3", warnings[0])

    def test_12_warnings_empty_when_no_recurring_hotspots(self):
        """Test warnings are empty when no hotspots recur (all count < 2)."""
        ledger = self._make_ledger({
            "slice_a": 1,
            "slice_b": 1,
        })

        warnings = extract_topology_hotspot_ledger_warnings(ledger)

        self.assertEqual(len(warnings), 0)

    def test_13_warnings_sorted_deterministic(self):
        """Test warnings are sorted for determinism."""
        ledger = self._make_ledger({
            "slice_z": 2,
            "slice_a": 2,
            "slice_m": 2,
        })

        warnings1 = extract_topology_hotspot_ledger_warnings(ledger)
        warnings2 = extract_topology_hotspot_ledger_warnings(ledger)

        self.assertEqual(warnings1, warnings2)
        # Should be sorted alphabetically
        self.assertTrue(warnings1[0] < warnings1[1] < warnings1[2])

    def test_14_attach_signal_to_evidence_includes_signal(self):
        """Test signal attachment includes signal in evidence."""
        ledger = self._make_ledger({"slice_a": 2})
        evidence = {
            "governance": {
                "topology_stress_panel": {
                    "hotspot_ledger": ledger,
                }
            }
        }

        result = attach_topology_hotspot_ledger_signal_to_evidence(evidence)

        self.assertIn("signals", result)
        self.assertIn("topology_hotspot_ledger", result["signals"])
        signal = result["signals"]["topology_hotspot_ledger"]
        self.assertEqual(signal["num_experiments"], 3)

    def test_15_attach_signal_to_evidence_includes_warnings_when_recurring(self):
        """Test signal attachment includes warnings when hotspots recur."""
        ledger = self._make_ledger({"slice_a": 2})
        evidence = {
            "governance": {
                "topology_stress_panel": {
                    "hotspot_ledger": ledger,
                }
            }
        }

        result = attach_topology_hotspot_ledger_signal_to_evidence(evidence)

        signal = result["signals"]["topology_hotspot_ledger"]
        self.assertIn("warnings", signal)
        self.assertEqual(len(signal["warnings"]), 1)

    def test_16_attach_signal_to_evidence_no_warnings_when_no_recurring(self):
        """Test signal attachment has no warnings when no hotspots recur."""
        ledger = self._make_ledger({"slice_a": 1})
        evidence = {
            "governance": {
                "topology_stress_panel": {
                    "hotspot_ledger": ledger,
                }
            }
        }

        result = attach_topology_hotspot_ledger_signal_to_evidence(evidence)

        signal = result["signals"]["topology_hotspot_ledger"]
        self.assertNotIn("warnings", signal)

    def test_17_attach_signal_to_evidence_missing_ledger_safe(self):
        """Test signal attachment handles missing ledger gracefully."""
        evidence = {"governance": {}}

        result = attach_topology_hotspot_ledger_signal_to_evidence(evidence)

        # Should not have signals key or should not have topology_hotspot_ledger
        if "signals" in result:
            self.assertNotIn("topology_hotspot_ledger", result["signals"])

    def test_18_attach_signal_to_evidence_non_mutating(self):
        """Test signal attachment does not modify input evidence."""
        ledger = self._make_ledger({"slice_a": 2})
        evidence = {
            "governance": {
                "topology_stress_panel": {
                    "hotspot_ledger": ledger,
                }
            }
        }

        result = attach_topology_hotspot_ledger_signal_to_evidence(evidence)

        # Original evidence should not have signals key
        self.assertNotIn("signals", evidence)
        # Result should have signals
        self.assertIn("signals", result)

    def test_19_signal_deterministic_output(self):
        """Test signal extraction is deterministic."""
        import json
        ledger = self._make_ledger({
            "slice_a": 3,
            "slice_b": 2,
            "slice_c": 1,
        })

        signal1 = extract_topology_hotspot_ledger_signal(ledger)
        signal2 = extract_topology_hotspot_ledger_signal(ledger)

        json1 = json.dumps(signal1, sort_keys=True)
        json2 = json.dumps(signal2, sort_keys=True)

        self.assertEqual(json1, json2, "Signal output is not deterministic")


if __name__ == "__main__":
    unittest.main()

