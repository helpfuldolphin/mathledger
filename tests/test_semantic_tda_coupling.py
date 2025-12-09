"""
PHASE V — SEMANTIC/TDA CROSS-CORRELATION TESTS

Unit tests for semantic/TDA cross-correlation and governance tile building.
"""

import unittest
from typing import Any, Dict

from experiments.semantic_consistency_audit import (
    correlate_semantic_and_tda_signals,
    build_semantic_tda_governance_tile,
    SEMANTIC_TDA_COUPLING_SCHEMA_VERSION,
)


class TestCorrelateSemanticAndTDASignals(unittest.TestCase):
    """Tests for correlate_semantic_and_tda_signals."""
    
    def test_both_green_ok(self):
        """Both semantic and TDA are OK → high positive correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 0, "status": "OK"}],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertEqual(result["schema_version"], SEMANTIC_TDA_COUPLING_SCHEMA_VERSION)
        self.assertAlmostEqual(result["correlation_coefficient"], 1.0, places=2)
        self.assertEqual(result["slices_where_both_signal"], [])
        self.assertIn("alignment_note", result)
        self.assertIn("Strong positive correlation", result["alignment_note"])
    
    def test_both_red_alert(self):
        """Both semantic and TDA are critical → high positive correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 2, "status": "CRITICAL"}],
            "runs_with_critical_signals": ["run1"],
            "node_disappearance_events": [
                {"run_id": "run1", "term": "slice_uplift_goal"},
                {"run_id": "run1", "term": "slice_uplift_sparse"},
            ],
            "trend": "DRIFTING",
        }
        tda_health = {
            "tda_status": "ALERT",
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertAlmostEqual(result["correlation_coefficient"], 1.0, places=2)
        self.assertIn("slice_uplift_goal", result["slices_where_both_signal"])
        self.assertIn("slice_uplift_sparse", result["slices_where_both_signal"])
        self.assertIn("Strong positive correlation", result["alignment_note"])
    
    def test_semantic_red_tda_green(self):
        """Semantic RED, TDA OK → negative correlation (mismatch)."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 1, "status": "CRITICAL"}],
            "runs_with_critical_signals": ["run1"],
            "node_disappearance_events": [{"run_id": "run1", "term": "slice_uplift_goal"}],
            "trend": "DRIFTING",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertAlmostEqual(result["correlation_coefficient"], -1.0, places=2)
        self.assertIn("slice_uplift_goal", result["semantic_only_slices"])
        self.assertIn("Strong negative correlation", result["alignment_note"])
    
    def test_semantic_green_tda_alert(self):
        """Semantic GREEN, TDA ALERT → negative correlation (mismatch)."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 0, "status": "OK"}],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
        }
        tda_health = {
            "tda_status": "ALERT",
            "block_rate": 0.3,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertAlmostEqual(result["correlation_coefficient"], -1.0, places=2)
        self.assertIn("tda_topology_drift", result["tda_only_slices"])
        self.assertIn("Strong negative correlation", result["alignment_note"])
    
    def test_both_yellow_attention(self):
        """Both semantic and TDA are ATTENTION → moderate positive correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 0, "status": "OK"}],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "DRIFTING",  # YELLOW
        }
        tda_health = {
            "tda_status": "ATTENTION",
            "block_rate": 0.05,
            "hss_trend": "DEGRADING",
            "governance_signal": "WARN",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertAlmostEqual(result["correlation_coefficient"], 0.5, places=2)
        self.assertIn("Moderate positive correlation", result["alignment_note"])
    
    def test_empty_timeline(self):
        """Empty semantic timeline should handle gracefully."""
        semantic_timeline = {
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertEqual(result["schema_version"], SEMANTIC_TDA_COUPLING_SCHEMA_VERSION)
        self.assertAlmostEqual(result["correlation_coefficient"], 1.0, places=2)
    
    def test_partial_mismatch(self):
        """Semantic YELLOW, TDA OK → weak correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run1", "term_count": 100, "critical_signal_count": 0, "status": "OK"}],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "VOLATILE",  # YELLOW
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        
        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)
        
        self.assertAlmostEqual(result["correlation_coefficient"], 0.0, places=2)
        self.assertIn("Weak correlation", result["alignment_note"])


class TestBuildSemanticTDAGoverningTile(unittest.TestCase):
    """Tests for build_semantic_tda_governance_tile."""
    
    def test_both_red_block_tile(self):
        """Both panels RED → BLOCK tile."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run1"],
            "headline": "Semantic graph shows critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        correlation = {
            "correlation_coefficient": 1.0,
            "slices_where_both_signal": ["slice_uplift_goal", "slice_uplift_sparse"],
            "semantic_only_slices": [],
            "tda_only_slices": [],
            "alignment_note": "Strong positive correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["schema_version"], SEMANTIC_TDA_COUPLING_SCHEMA_VERSION)
        self.assertEqual(result["status"], "BLOCK")
        self.assertEqual(result["status_light"], "RED")
        self.assertIn("both indicate critical", result["headline"])
        self.assertIn("slice_uplift_goal", result["key_slices"])
        self.assertIn("slice_uplift_sparse", result["key_slices"])
    
    def test_mismatch_attention_tile(self):
        """Semantic RED, TDA GREEN → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run1"],
            "headline": "Semantic graph shows critical drift",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        correlation = {
            "correlation_coefficient": -1.0,
            "slices_where_both_signal": [],
            "semantic_only_slices": ["slice_uplift_goal"],
            "tda_only_slices": [],
            "alignment_note": "Strong negative correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["status"], "ATTENTION")
        self.assertEqual(result["status_light"], "YELLOW")
        # Should have specific mismatch message
        self.assertIn("Semantic signals indicate drift", result["headline"])
        self.assertIn("TDA topology appears stable", result["headline"])
        self.assertIn("slice_uplift_goal", result["key_slices"])
    
    def test_both_green_ok_tile(self):
        """Both panels GREEN → OK tile."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Semantic graph is stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        correlation = {
            "correlation_coefficient": 1.0,
            "slices_where_both_signal": [],
            "semantic_only_slices": [],
            "tda_only_slices": [],
            "alignment_note": "Strong positive correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["status_light"], "GREEN")
        self.assertIn("stable structural health", result["headline"])
        self.assertEqual(result["key_slices"], [])
    
    def test_tda_alert_semantic_green_attention(self):
        """TDA ALERT, Semantic GREEN → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Semantic graph is stable",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.3,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        correlation = {
            "correlation_coefficient": -1.0,
            "slices_where_both_signal": [],
            "semantic_only_slices": [],
            "tda_only_slices": ["tda_topology_drift"],
            "alignment_note": "Strong negative correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["status"], "ATTENTION")
        self.assertEqual(result["status_light"], "YELLOW")
        self.assertIn("TDA topology indicates drift", result["headline"])
        self.assertIn("tda_topology_drift", result["key_slices"])
    
    def test_high_correlation_both_signals_block(self):
        """High correlation with both signals → BLOCK even if not both RED."""
        semantic_panel = {
            "semantic_status_light": "YELLOW",
            "alignment_status": "PARTIAL",
            "critical_run_ids": [],
            "headline": "Semantic graph shows drift",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.1,
            "hss_trend": "DEGRADING",
            "governance_signal": "WARN",
        }
        correlation = {
            "correlation_coefficient": 0.9,  # High correlation
            "slices_where_both_signal": ["slice_uplift_goal"],
            "semantic_only_slices": [],
            "tda_only_slices": [],
            "alignment_note": "Strong positive correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        # Should be BLOCK due to high correlation with both signals
        self.assertEqual(result["status"], "BLOCK")
        self.assertEqual(result["status_light"], "RED")
        self.assertIn("slice_uplift_goal", result["key_slices"])
    
    def test_yellow_attention_tile(self):
        """Both YELLOW/ATTENTION → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "YELLOW",
            "alignment_status": "PARTIAL",
            "critical_run_ids": [],
            "headline": "Semantic graph shows drift",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.05,
            "hss_trend": "DEGRADING",
            "governance_signal": "WARN",
        }
        correlation = {
            "correlation_coefficient": 0.5,
            "slices_where_both_signal": [],
            "semantic_only_slices": [],
            "tda_only_slices": [],
            "alignment_note": "Moderate positive correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["status"], "ATTENTION")
        self.assertEqual(result["status_light"], "YELLOW")
        self.assertIn("partial alignment", result["headline"])
    
    def test_negative_correlation_attention(self):
        """Negative correlation → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Semantic graph is stable",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.05,
            "hss_trend": "DEGRADING",
            "governance_signal": "WARN",
        }
        correlation = {
            "correlation_coefficient": -0.5,  # Negative correlation
            "slices_where_both_signal": [],
            "semantic_only_slices": [],
            "tda_only_slices": [],
            "alignment_note": "Moderate negative correlation",
        }
        
        result = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
        
        self.assertEqual(result["status"], "ATTENTION")
        self.assertEqual(result["status_light"], "YELLOW")
        self.assertIn("disagreement", result["headline"])


if __name__ == "__main__":
    unittest.main()

