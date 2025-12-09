"""
Tests for Phase V: Topological Pressure Field & Curriculum Promotion Gate.

Tests cover:
- Topological pressure field computation
- Curriculum promotion gate logic
- Console tile with pressure hotspots
- Integration with existing Phase IV components
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List


class TestTopologicalPressureField(unittest.TestCase):
    """Tests for build_topological_pressure_field()."""

    def _make_risk_envelope(
        self,
        risk_band: str = "OK",
        volatility: float = 0.5,
    ) -> Dict[str, Any]:
        """Helper to create mock risk envelope."""
        return {
            "risk_band": risk_band,
            "branching_volatility": volatility,
            "max_depth_band": [3, 5],
            "envelope_summary": "test",
        }

    def test_01_pressure_field_has_required_fields(self):
        """Test pressure field has all required fields."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope()
        result = build_topological_pressure_field(
            depth_trend="STABLE",
            branching_volatility=0.5,
            risk_envelope=risk_envelope,
        )
        
        required_fields = [
            "slice_pressure",
            "pressure_components",
            "pressure_band",
            "neutral_notes",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_02_pressure_components_normalized(self):
        """Test pressure components are normalized to [0, 1]."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope()
        result = build_topological_pressure_field(
            depth_trend="DEEPENING",
            branching_volatility=2.0,
            risk_envelope=risk_envelope,
        )
        
        components = result["pressure_components"]
        for component_name, value in components.items():
            self.assertGreaterEqual(value, 0.0, f"{component_name} below 0")
            self.assertLessEqual(value, 1.0, f"{component_name} above 1")

    def test_03_pressure_band_low(self):
        """Test LOW pressure band for low pressure."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope(risk_band="OK", volatility=0.1)
        result = build_topological_pressure_field(
            depth_trend="SHALLOWING",
            branching_volatility=0.1,
            risk_envelope=risk_envelope,
        )
        
        self.assertEqual(result["pressure_band"], "LOW")
        self.assertLess(result["slice_pressure"], 0.4)

    def test_04_pressure_band_medium(self):
        """Test MEDIUM pressure band for medium pressure."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope(risk_band="ATTENTION", volatility=1.0)
        result = build_topological_pressure_field(
            depth_trend="STABLE",
            branching_volatility=1.0,
            risk_envelope=risk_envelope,
        )
        
        self.assertEqual(result["pressure_band"], "MEDIUM")
        self.assertGreaterEqual(result["slice_pressure"], 0.4)
        self.assertLess(result["slice_pressure"], 0.7)

    def test_05_pressure_band_high(self):
        """Test HIGH pressure band for high pressure."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope(risk_band="STRESSED", volatility=2.5)
        result = build_topological_pressure_field(
            depth_trend="DEEPENING",
            branching_volatility=2.5,
            risk_envelope=risk_envelope,
        )
        
        self.assertEqual(result["pressure_band"], "HIGH")
        self.assertGreaterEqual(result["slice_pressure"], 0.7)

    def test_06_weighted_composite_pressure(self):
        """Test composite pressure uses correct weights."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope(risk_band="OK", volatility=0.0)
        result = build_topological_pressure_field(
            depth_trend="DEEPENING",  # = 1.0
            branching_volatility=0.0,  # = 0.0
            risk_envelope=risk_envelope,  # OK = 0.0
        )
        
        # Expected: 1.0 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3 = 0.4
        expected_pressure = 1.0 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3
        self.assertAlmostEqual(result["slice_pressure"], expected_pressure, places=3)

    def test_07_depth_trend_normalization(self):
        """Test depth trend normalization."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope()
        
        # DEEPENING should give highest depth pressure
        result_deep = build_topological_pressure_field(
            depth_trend="DEEPENING",
            branching_volatility=0.0,
            risk_envelope=risk_envelope,
        )
        
        # SHALLOWING should give lowest depth pressure
        result_shallow = build_topological_pressure_field(
            depth_trend="SHALLOWING",
            branching_volatility=0.0,
            risk_envelope=risk_envelope,
        )
        
        self.assertGreater(
            result_deep["pressure_components"]["depth"],
            result_shallow["pressure_components"]["depth"],
        )

    def test_08_risk_band_normalization(self):
        """Test risk band normalization."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        # STRESSED should give highest risk pressure
        risk_stressed = self._make_risk_envelope(risk_band="STRESSED")
        result_stressed = build_topological_pressure_field(
            depth_trend="STABLE",
            branching_volatility=0.0,
            risk_envelope=risk_stressed,
        )
        
        # OK should give lowest risk pressure
        risk_ok = self._make_risk_envelope(risk_band="OK")
        result_ok = build_topological_pressure_field(
            depth_trend="STABLE",
            branching_volatility=0.0,
            risk_envelope=risk_ok,
        )
        
        self.assertGreater(
            result_stressed["pressure_components"]["risk"],
            result_ok["pressure_components"]["risk"],
        )

    def test_09_neutral_notes_present(self):
        """Test neutral notes are generated."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope()
        result = build_topological_pressure_field(
            depth_trend="DEEPENING",
            branching_volatility=2.0,
            risk_envelope=risk_envelope,
        )
        
        self.assertGreater(len(result["neutral_notes"]), 0)
        notes_text = " ".join(result["neutral_notes"]).lower()
        
        # Check for neutral language (no value judgments)
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, notes_text, f"Found forbidden word: {word}")

    def test_10_weights_included(self):
        """Test weights are included in result."""
        from experiments.visualize_dag_topology import build_topological_pressure_field
        
        risk_envelope = self._make_risk_envelope()
        result = build_topological_pressure_field(
            depth_trend="STABLE",
            branching_volatility=0.5,
            risk_envelope=risk_envelope,
        )
        
        self.assertIn("weights", result)
        self.assertEqual(result["weights"]["depth"], 0.4)
        self.assertEqual(result["weights"]["branching"], 0.3)
        self.assertEqual(result["weights"]["risk"], 0.3)


class TestTopologyCurriculumPromotionGate(unittest.TestCase):
    """Tests for topology_curriculum_promotion_gate()."""

    def _make_slice_view(
        self,
        slice_status: str = "OK",
    ) -> Dict[str, Any]:
        """Helper to create mock slice view."""
        return {
            "slice_view": {
                "slice_topology_status": slice_status,
                "depth_trend": "STABLE",
                "typical_max_depth": 5,
            },
            "entry_count": 3,
        }

    def _make_pressure_field(
        self,
        pressure_band: str = "LOW",
        slice_pressure: float = 0.3,
    ) -> Dict[str, Any]:
        """Helper to create mock pressure field."""
        return {
            "pressure_band": pressure_band,
            "slice_pressure": slice_pressure,
            "pressure_components": {
                "depth": 0.3,
                "branching": 0.3,
                "risk": 0.3,
            },
            "neutral_notes": [],
        }

    def _make_progression_predictor(
        self,
        readiness_status: str = "READY",
    ) -> Dict[str, Any]:
        """Helper to create mock progression predictor."""
        return {
            "readiness_status": readiness_status,
            "slices_ready_for_next_depth": [],
            "slices_needing_stabilization": [],
            "notes": [],
        }

    def test_11_promotion_gate_has_required_fields(self):
        """Test promotion gate has all required fields."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view()
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        required_fields = [
            "promotion_status",
            "explanations",
            "slices_at_risk",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_12_promotion_status_ok(self):
        """Test OK promotion status for healthy conditions."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view(slice_status="OK")
        pressure_field = self._make_pressure_field(pressure_band="LOW")
        progression = self._make_progression_predictor(readiness_status="READY")
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "OK")

    def test_13_promotion_status_block_stressed_slice(self):
        """Test BLOCK status for stressed slice."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "BLOCK")
        self.assertIn("current_slice", result["slices_at_risk"])

    def test_14_promotion_status_block_high_pressure(self):
        """Test BLOCK status for high pressure."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view()
        pressure_field = self._make_pressure_field(
            pressure_band="HIGH", slice_pressure=0.85
        )
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "BLOCK")

    def test_15_promotion_status_block_stabilize(self):
        """Test BLOCK status when progression predictor indicates stabilize."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view()
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor(readiness_status="STABILIZE")
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "BLOCK")

    def test_16_promotion_status_attention(self):
        """Test ATTENTION status for moderate issues."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view(slice_status="ATTENTION")
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "ATTENTION")

    def test_17_promotion_status_attention_medium_pressure(self):
        """Test ATTENTION status for medium pressure."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view()
        pressure_field = self._make_pressure_field(
            pressure_band="MEDIUM", slice_pressure=0.5
        )
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "ATTENTION")

    def test_18_promotion_status_attention_hold(self):
        """Test ATTENTION status when progression predictor indicates HOLD."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view()
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor(readiness_status="HOLD")
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertEqual(result["promotion_status"], "ATTENTION")

    def test_19_explanations_use_neutral_language(self):
        """Test explanations use neutral language."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        pressure_field = self._make_pressure_field()
        progression = self._make_progression_predictor()
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        explanations_text = " ".join(result["explanations"]).lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, explanations_text, f"Found forbidden word: {word}")

    def test_20_gate_components_included(self):
        """Test gate_components are included in result."""
        from experiments.visualize_dag_topology import topology_curriculum_promotion_gate
        
        slice_view = self._make_slice_view(slice_status="OK")
        pressure_field = self._make_pressure_field(pressure_band="LOW")
        progression = self._make_progression_predictor(readiness_status="READY")
        
        result = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        self.assertIn("gate_components", result)
        self.assertEqual(result["gate_components"]["slice_status"], "OK")
        self.assertEqual(result["gate_components"]["pressure_band"], "LOW")
        self.assertEqual(result["gate_components"]["readiness_status"], "READY")


class TestTopologyConsoleTile(unittest.TestCase):
    """Tests for build_topology_console_tile()."""

    def _make_analytics(
        self,
        entry_count: int = 3,
        avg_stability: float = 0.7,
    ) -> Dict[str, Any]:
        """Helper to create mock analytics."""
        return {
            "entry_count": entry_count,
            "average_stability_score": avg_stability,
            "max_depth_over_time": [3, 4, 5],
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }

    def _make_pressure_field(
        self,
        pressure_band: str = "LOW",
        slice_pressure: float = 0.3,
    ) -> Dict[str, Any]:
        """Helper to create mock pressure field."""
        return {
            "pressure_band": pressure_band,
            "slice_pressure": slice_pressure,
            "pressure_components": {
                "depth": 0.3,
                "branching": 0.3,
                "risk": 0.3,
            },
            "neutral_notes": [],
        }

    def _make_promotion_gate(
        self,
        promotion_status: str = "OK",
    ) -> Dict[str, Any]:
        """Helper to create mock promotion gate."""
        return {
            "promotion_status": promotion_status,
            "explanations": [],
            "slices_at_risk": [],
        }

    def test_21_console_tile_has_required_fields(self):
        """Test console tile has all required fields."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        result = build_topology_console_tile(analytics)
        
        required_fields = [
            "status_light",
            "promotion_gate",
            "pressure_hotspots",
            "headline",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_22_status_light_present(self):
        """Test status light is present and valid."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        result = build_topology_console_tile(analytics)
        
        self.assertIn(result["status_light"], ["GREEN", "YELLOW", "RED"])

    def test_23_promotion_gate_status_included(self):
        """Test promotion gate status is included."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate(promotion_status="BLOCK")
        
        result = build_topology_console_tile(
            analytics, pressure_field=pressure_field, promotion_gate=promotion_gate
        )
        
        self.assertEqual(result["promotion_gate"], "BLOCK")

    def test_24_pressure_hotspots_for_high_pressure(self):
        """Test pressure hotspots include high pressure observations."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        pressure_field = self._make_pressure_field(
            pressure_band="HIGH", slice_pressure=0.85
        )
        
        result = build_topology_console_tile(analytics, pressure_field=pressure_field)
        
        self.assertGreater(len(result["pressure_hotspots"]), 0)
        hotspots_text = " ".join(result["pressure_hotspots"]).lower()
        self.assertIn("high", hotspots_text)

    def test_25_pressure_hotspots_for_component_pressure(self):
        """Test pressure hotspots include component observations."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        pressure_field = self._make_pressure_field()
        pressure_field["pressure_components"]["depth"] = 0.85  # High depth pressure
        
        result = build_topology_console_tile(analytics, pressure_field=pressure_field)
        
        hotspots_text = " ".join(result["pressure_hotspots"]).lower()
        self.assertIn("depth", hotspots_text)

    def test_26_headline_is_neutral(self):
        """Test headline uses neutral language."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        result = build_topology_console_tile(analytics)
        
        headline = result["headline"].lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, headline, f"Found forbidden word: {word}")

    def test_27_headline_includes_promotion_gate(self):
        """Test headline includes promotion gate status when provided."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        promotion_gate = self._make_promotion_gate(promotion_status="BLOCK")
        
        result = build_topology_console_tile(
            analytics, promotion_gate=promotion_gate
        )
        
        headline = result["headline"].lower()
        self.assertIn("block", headline)

    def test_28_metrics_included(self):
        """Test metrics are included in console tile."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = self._make_analytics()
        result = build_topology_console_tile(analytics)
        
        self.assertIn("metrics", result)
        self.assertIn("entry_count", result["metrics"])
        self.assertIn("average_stability", result["metrics"])

    def test_29_empty_analytics_handled(self):
        """Test empty analytics handled gracefully."""
        from experiments.visualize_dag_topology import build_topology_console_tile
        
        analytics = {
            "entry_count": 0,
            "average_stability_score": 0.0,
            "max_depth_over_time": [],
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = build_topology_console_tile(analytics)
        
        self.assertIn("status_light", result)
        self.assertEqual(result["promotion_gate"], "OK")


class TestPhaseVIntegration(unittest.TestCase):
    """Integration tests for Phase V components."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and generate ledger entries."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_phase_v_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(30):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "phase_v_test",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "depth": (i % 4) + 1,
                    },
                })
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_30_full_phase_v_pipeline(self):
        """Test full Phase V pipeline from ledger to console tile."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
            build_slice_topology_curriculum_view,
            build_topology_risk_envelope,
            build_topological_pressure_field,
            predict_curriculum_progression_from_topology,
            topology_curriculum_promotion_gate,
            build_topology_console_tile,
        )
        
        # Generate ledger entry
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="phase_v_test"
        )
        
        # Build risk envelope
        risk_envelope = build_topology_risk_envelope([entry])
        
        # Run analytics and build slice view
        analytics = analyze_topology_ledger_entries([entry])
        slice_view = build_slice_topology_curriculum_view(analytics)
        
        # Build pressure field
        depth_trend = slice_view["slice_view"]["depth_trend"]
        volatility = risk_envelope["branching_volatility"]
        pressure_field = build_topological_pressure_field(
            depth_trend, volatility, risk_envelope
        )
        
        # Predict progression
        progression = predict_curriculum_progression_from_topology(
            slice_view, risk_envelope
        )
        
        # Promotion gate
        promotion_gate = topology_curriculum_promotion_gate(
            slice_view, pressure_field, progression
        )
        
        # Console tile
        console_tile = build_topology_console_tile(
            analytics, pressure_field=pressure_field, promotion_gate=promotion_gate
        )
        
        # Verify all outputs are valid
        self.assertIn("status_light", console_tile)
        self.assertIn("promotion_gate", console_tile)
        self.assertIn("pressure_hotspots", console_tile)
        self.assertIn("promotion_status", promotion_gate)
        self.assertIn(promotion_gate["promotion_status"], ["OK", "ATTENTION", "BLOCK"])
        self.assertIn("pressure_band", pressure_field)
        self.assertIn(pressure_field["pressure_band"], ["LOW", "MEDIUM", "HIGH"])


if __name__ == "__main__":
    unittest.main()

