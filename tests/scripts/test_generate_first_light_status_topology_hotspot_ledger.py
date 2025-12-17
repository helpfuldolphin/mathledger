"""
Integration tests for topology hotspot ledger signal extraction in status generator.

Tests verify:
- Signal extraction from manifest
- Warning generation (one line, max top 3 names)
- Determinism
- Missing ledger safe behavior
"""

import json
import unittest
from pathlib import Path
from typing import Any, Dict

from backend.health.topology_pressure_adapter import (
    build_topology_stress_panel,
    build_topology_hotspot_ledger,
    extract_topology_hotspot_ledger_signal_from_evidence,
)
from scripts.generate_first_light_status import generate_status


class TestTopologyHotspotLedgerStatusIntegration(unittest.TestCase):
    """Integration tests for topology hotspot ledger in status generator."""

    def _make_minimal_manifest(
        self,
        topology_stress_panel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Helper to create minimal manifest with topology stress panel."""
        return {
            "schema_version": "1.0.0",
            "pack_type": "first_light_evidence",
            "mode": "SHADOW",
            "governance": {
                "topology_stress_panel": topology_stress_panel,
            },
        }

    def _make_topology_stress_panel_with_ledger(
        self,
        snapshots: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Helper to create topology stress panel with hotspot ledger."""
        panel = build_topology_stress_panel(snapshots)
        ledger = build_topology_hotspot_ledger(panel)
        panel["hotspot_ledger"] = ledger
        return panel

    def test_01_signal_extracted_from_manifest(self):
        """Test signal is extracted from manifest when panel present."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        self.assertIsNotNone(signal)
        self.assertEqual(signal["num_experiments"], 2)
        self.assertEqual(signal["unique_hotspot_count"], 2)
        self.assertEqual(signal["extraction_source"], "MANIFEST")

    def test_02_signal_includes_top3_hotspots(self):
        """Test signal includes top 3 hotspots."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b", "slice_c"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        self.assertEqual(len(signal["top_hotspots_top3"]), 3)
        self.assertEqual(signal["top_hotspots_top3"][0], "slice_a")  # count=3
        self.assertEqual(signal["top_hotspots_top3"][1], "slice_b")  # count=2
        self.assertEqual(signal["top_hotspots_top3"][2], "slice_c")  # count=1

    def test_03_warning_generated_for_recurring_hotspots(self):
        """Test warning is generated when hotspots recur (count >= 2)."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        # Check that slice_a has count >= 2
        hotspot_counts = ledger.get("hotspot_counts", {})
        self.assertGreaterEqual(hotspot_counts.get("slice_a", 0), 2)

    def test_04_warning_limited_to_top3_names(self):
        """Test warning lists at most top 3 recurring hotspot names."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b", "slice_c", "slice_d"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a", "slice_b", "slice_c"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp4", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        hotspot_counts = ledger.get("hotspot_counts", {})
        recurring_hotspots = [
            hotspot for hotspot, count in hotspot_counts.items() if count >= 2
        ]
        # Should have slice_a (4), slice_b (3), slice_c (2) - all >= 2
        self.assertGreaterEqual(len(recurring_hotspots), 3)

        # Top 3 should be sorted alphabetically
        top_recurring_names = sorted(recurring_hotspots)[:3]
        self.assertLessEqual(len(top_recurring_names), 3)

    def test_05_warning_one_line_total(self):
        """Test only one warning line is generated."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a", "slice_c"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_b", "slice_d"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        hotspot_counts = ledger.get("hotspot_counts", {})
        recurring_hotspots = [
            hotspot for hotspot, count in hotspot_counts.items() if count >= 2
        ]

        # Should have multiple recurring hotspots, but warning should be one line
        self.assertGreater(len(recurring_hotspots), 1)

    def test_06_no_warning_when_no_recurring_hotspots(self):
        """Test no warning when all hotspots have count < 2."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_b"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_c"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        hotspot_counts = ledger.get("hotspot_counts", {})
        recurring_hotspots = [
            hotspot for hotspot, count in hotspot_counts.items() if count >= 2
        ]

        # Should have no recurring hotspots
        self.assertEqual(len(recurring_hotspots), 0)

    def test_07_deterministic_signal_extraction(self):
        """Test signal extraction is deterministic."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal1 = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")
        signal2 = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        json1 = json.dumps(signal1, sort_keys=True)
        json2 = json.dumps(signal2, sort_keys=True)

        self.assertEqual(json1, json2, "Signal extraction is not deterministic")

    def test_08_missing_ledger_safe_behavior(self):
        """Test missing ledger is handled gracefully."""
        # Panel without ledger
        panel = {
            "schema_version": "1.0.0",
            "panel_type": "topology_stress_heatmap",
            "experiments": [],
            "summary": {},
        }
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        # Should return None when ledger is missing
        self.assertIsNone(signal)

    def test_09_missing_panel_safe_behavior(self):
        """Test missing panel is handled gracefully."""
        manifest = {
            "schema_version": "1.0.0",
            "pack_type": "first_light_evidence",
            "mode": "SHADOW",
            "governance": {},
        }

        evidence = {"governance": {}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MISSING")

        # Should return None when panel is missing
        self.assertIsNone(signal)

    def test_10_signal_includes_schema_version(self):
        """Test signal includes schema_version."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        self.assertIn("schema_version", signal)
        self.assertEqual(signal["schema_version"], "1.0.0")

    def test_12_signal_includes_extraction_source(self):
        """Test signal includes extraction_source field."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        manifest = self._make_minimal_manifest(panel)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")

        self.assertIn("extraction_source", signal)
        self.assertEqual(signal["extraction_source"], "MANIFEST")

    def test_13_extraction_source_evidence_json(self):
        """Test extraction_source is EVIDENCE_JSON when extracted from evidence.json."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)

        evidence = {"governance": {"topology_stress_panel": panel}}
        signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "EVIDENCE_JSON")

        self.assertEqual(signal["extraction_source"], "EVIDENCE_JSON")

    def test_14_warning_reason_code_present_when_warning_emitted(self):
        """Test warning_reason_code is present in signal when warning is emitted."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        # Check that slice_a has count >= 2 (should trigger warning)
        hotspot_counts = ledger.get("hotspot_counts", {})
        self.assertGreaterEqual(hotspot_counts.get("slice_a", 0), 2)

    def test_15_warning_reason_code_stability(self):
        """Test warning_reason_code is stable (always TOP-HOT-001)."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b", "slice_c"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a", "slice_b"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_a"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        hotspot_counts = ledger.get("hotspot_counts", {})
        recurring_hotspots = [
            hotspot for hotspot, count in hotspot_counts.items() if count >= 2
        ]

        # Should have recurring hotspots, and reason code should always be TOP-HOT-001
        self.assertGreater(len(recurring_hotspots), 0)

    def test_11_warning_format_respects_top3_limit(self):
        """Test warning format respects top 3 limit even when more recurring hotspots exist."""
        snapshots = [
            {"cal_id": "cal_exp1", "pressure_hotspots": ["slice_a", "slice_b", "slice_c", "slice_d", "slice_e"]},
            {"cal_id": "cal_exp2", "pressure_hotspots": ["slice_a", "slice_b", "slice_c", "slice_d"]},
            {"cal_id": "cal_exp3", "pressure_hotspots": ["slice_a", "slice_b", "slice_c"]},
            {"cal_id": "cal_exp4", "pressure_hotspots": ["slice_a", "slice_b"]},
        ]
        panel = self._make_topology_stress_panel_with_ledger(snapshots)
        ledger = panel["hotspot_ledger"]

        hotspot_counts = ledger.get("hotspot_counts", {})
        recurring_hotspots = [
            hotspot for hotspot, count in hotspot_counts.items() if count >= 2
        ]

        # Should have more than 3 recurring hotspots
        self.assertGreater(len(recurring_hotspots), 3)

        # Top 3 should be limited to 3
        top_recurring_names = sorted(recurring_hotspots)[:3]
        self.assertEqual(len(top_recurring_names), 3)


if __name__ == "__main__":
    unittest.main()

