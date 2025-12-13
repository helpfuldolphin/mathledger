"""
Tests for Metric Readiness Panel GGFL adapter.

Validates:
- GGFL adapter shape and fields
- Deterministic output
- Driver ordering (blocks, poly_fail, low p4 range)
- Status derivation (ok vs warn)
- Summary generation
"""

import json
import unittest
from typing import Dict, Any

from backend.health.readiness_tensor_adapter import (
    metric_readiness_panel_for_alignment_view,
    build_metric_readiness_panel,
)


class TestMetricReadinessPanelForAlignmentView(unittest.TestCase):
    """Tests for metric_readiness_panel_for_alignment_view function."""

    def test_ggfl_adapter_has_required_fields(self):
        """GGFL adapter has all required fields."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_poly_fail": 0,
            "global_norm_range": {
                "p3_min": 0.2,
                "p3_max": 0.8,
                "p4_min": 0.2,
                "p4_max": 0.75,
            },
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("signal_type", result)
        self.assertIn("status", result)
        self.assertIn("conflict", result)
        self.assertIn("drivers", result)
        self.assertIn("summary", result)

    def test_ggfl_adapter_signal_type(self):
        """GGFL adapter has correct signal_type."""
        panel = {
            "num_experiments": 1,
            "num_ok": 1,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertEqual(result["signal_type"], "SIG-READY")

    def test_ggfl_adapter_status_ok_when_no_issues(self):
        """Status is 'ok' when no blocks or poly_fail."""
        panel = {
            "num_experiments": 2,
            "num_ok": 2,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {"p4_min": 0.8, "p4_max": 0.9},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertEqual(result["status"], "ok")

    def test_ggfl_adapter_status_warn_when_block(self):
        """Status is 'warn' when num_block > 0."""
        panel = {
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 0,
            "num_block": 1,
            "num_poly_fail": 0,
            "global_norm_range": {},
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertEqual(result["status"], "warn")

    def test_ggfl_adapter_status_warn_when_poly_fail(self):
        """Status is 'warn' when num_poly_fail > 0."""
        panel = {
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 0,
            "num_poly_fail": 1,
            "global_norm_range": {},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertEqual(result["status"], "warn")

    def test_ggfl_adapter_conflict_always_false(self):
        """Conflict is always False (invariant)."""
        panel = {
            "num_experiments": 1,
            "num_ok": 0,
            "num_warn": 0,
            "num_block": 1,
            "num_poly_fail": 1,
            "global_norm_range": {},
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertFalse(result["conflict"])
    
    def test_ggfl_adapter_has_shadow_mode_invariants(self):
        """GGFL adapter includes shadow_mode_invariants block."""
        panel = {
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {"p4_min": 0.8, "p4_max": 0.9},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("shadow_mode_invariants", result)
        invariants = result["shadow_mode_invariants"]
        self.assertTrue(invariants["advisory_only"])
        self.assertTrue(invariants["no_enforcement"])
        self.assertTrue(invariants["conflict_invariant"])
    
    def test_ggfl_adapter_invariants_stable_values(self):
        """Shadow mode invariants have stable values (always True)."""
        panels = [
            {
                "num_experiments": 1,
                "num_ok": 1,
                "num_warn": 0,
                "num_block": 0,
                "num_poly_fail": 0,
                "global_norm_range": {},
                "top_driver_cal_ids": [],
            },
            {
                "num_experiments": 3,
                "num_ok": 0,
                "num_warn": 0,
                "num_block": 3,
                "num_poly_fail": 3,
                "global_norm_range": {"p4_min": 0.1, "p4_max": 0.2},
                "top_driver_cal_ids": ["CAL-EXP-3"],
            },
        ]

        for panel in panels:
            result = metric_readiness_panel_for_alignment_view(panel)
            invariants = result["shadow_mode_invariants"]
            # All invariants must be True (stable values)
            self.assertTrue(invariants["advisory_only"])
            self.assertTrue(invariants["no_enforcement"])
            self.assertTrue(invariants["conflict_invariant"])
    
    def test_ggfl_adapter_driver_low_p4_norm_range(self):
        """DRIVER_LOW_P4_NORM_RANGE is added when p4_min < 0.35."""
        panel = {
            "num_experiments": 2,
            "num_ok": 2,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {"p4_min": 0.3, "p4_max": 0.5},  # p4_min < 0.35
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("DRIVER_LOW_P4_NORM_RANGE", result["drivers"])
    
    def test_ggfl_adapter_driver_low_p4_norm_range_not_added_when_above_threshold(self):
        """DRIVER_LOW_P4_NORM_RANGE is NOT added when p4_min >= 0.35."""
        panel = {
            "num_experiments": 2,
            "num_ok": 2,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {"p4_min": 0.4, "p4_max": 0.5},  # p4_min >= 0.35
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertNotIn("DRIVER_LOW_P4_NORM_RANGE", result["drivers"])

    def test_ggfl_adapter_drivers_use_reason_codes(self):
        """Drivers list uses reason codes, not freeform strings."""
        panel = {
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_poly_fail": 1,
            "global_norm_range": {"p4_min": 0.2, "p4_max": 0.75},
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        # Verify drivers are reason codes only
        for driver in result["drivers"]:
            self.assertTrue(
                driver.startswith("DRIVER_"),
                f"Driver '{driver}' is not a reason code"
            )
            self.assertIn(
                driver,
                ["DRIVER_BLOCK_PRESENT", "DRIVER_POLY_FAIL_PRESENT", "DRIVER_LOW_P4_NORM_RANGE"],
                f"Unknown driver reason code: {driver}"
            )
        
        # Verify ordering: BLOCK → POLY_FAIL → LOW_NORM
        if "DRIVER_BLOCK_PRESENT" in result["drivers"]:
            block_idx = result["drivers"].index("DRIVER_BLOCK_PRESENT")
            if "DRIVER_POLY_FAIL_PRESENT" in result["drivers"]:
                poly_idx = result["drivers"].index("DRIVER_POLY_FAIL_PRESENT")
                self.assertLess(block_idx, poly_idx)
            if "DRIVER_LOW_P4_NORM_RANGE" in result["drivers"]:
                norm_idx = result["drivers"].index("DRIVER_LOW_P4_NORM_RANGE")
                self.assertLess(block_idx, norm_idx)
        
        if "DRIVER_POLY_FAIL_PRESENT" in result["drivers"] and "DRIVER_LOW_P4_NORM_RANGE" in result["drivers"]:
            poly_idx = result["drivers"].index("DRIVER_POLY_FAIL_PRESENT")
            norm_idx = result["drivers"].index("DRIVER_LOW_P4_NORM_RANGE")
            self.assertLess(poly_idx, norm_idx)

    def test_ggfl_adapter_drivers_max_3(self):
        """Drivers list is limited to maximum 3."""
        panel = {
            "num_experiments": 5,
            "num_ok": 2,
            "num_warn": 1,
            "num_block": 2,
            "num_poly_fail": 1,
            "global_norm_range": {"p4_min": 0.2, "p4_max": 0.75},
            "top_driver_cal_ids": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3", "CAL-EXP-4"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        # Should have all 3 reason codes (BLOCK, POLY_FAIL, LOW_NORM)
        self.assertEqual(len(result["drivers"]), 3)
        self.assertIn("DRIVER_BLOCK_PRESENT", result["drivers"])
        self.assertIn("DRIVER_POLY_FAIL_PRESENT", result["drivers"])
        self.assertIn("DRIVER_LOW_P4_NORM_RANGE", result["drivers"])

    def test_ggfl_adapter_summary_includes_block_count(self):
        """Summary includes block count when num_block > 0."""
        panel = {
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_poly_fail": 0,
            "global_norm_range": {},
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("BLOCK", result["summary"])
        self.assertIn("1", result["summary"])

    def test_ggfl_adapter_summary_includes_poly_fail_count(self):
        """Summary includes poly-fail count when num_poly_fail > 0."""
        panel = {
            "num_experiments": 3,
            "num_ok": 2,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 1,
            "global_norm_range": {},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("poly-fail", result["summary"])
        self.assertIn("1", result["summary"])

    def test_ggfl_adapter_summary_includes_both_block_and_poly_fail(self):
        """Summary includes both block and poly-fail when both > 0."""
        panel = {
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 0,
            "num_block": 1,
            "num_poly_fail": 1,
            "global_norm_range": {},
            "top_driver_cal_ids": ["CAL-EXP-3"],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertIn("BLOCK", result["summary"])
        self.assertIn("poly-fail", result["summary"])

    def test_ggfl_adapter_summary_neutral_language(self):
        """Summary uses neutral language (no value judgments)."""
        panel = {
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {"p4_min": 0.8, "p4_max": 0.9},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        summary_lower = result["summary"].lower()
        # Check for absence of value judgment words
        self.assertNotIn("good", summary_lower)
        self.assertNotIn("bad", summary_lower)
        self.assertNotIn("better", summary_lower)
        self.assertNotIn("worse", summary_lower)
        self.assertNotIn("success", summary_lower)
        self.assertNotIn("failure", summary_lower)

    def test_ggfl_adapter_deterministic(self):
        """GGFL adapter output is deterministic."""
        panel = {
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_poly_fail": 0,
            "global_norm_range": {},
            "top_driver_cal_ids": ["CAL-EXP-3", "CAL-EXP-2"],
        }

        results = [
            metric_readiness_panel_for_alignment_view(panel) for _ in range(5)
        ]

        json_outputs = [json.dumps(r, sort_keys=True) for r in results]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))

    def test_ggfl_adapter_json_round_trip(self):
        """GGFL adapter output is JSON round-trip safe."""
        panel = {
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {},
            "top_driver_cal_ids": [],
        }

        result = metric_readiness_panel_for_alignment_view(panel)

        json_str = json.dumps(result, sort_keys=True)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["signal_type"], result["signal_type"])
        self.assertEqual(parsed["status"], result["status"])
        self.assertEqual(parsed["conflict"], result["conflict"])
        self.assertEqual(parsed["drivers"], result["drivers"])
        self.assertEqual(parsed["summary"], result["summary"])

    def test_ggfl_adapter_with_built_panel(self):
        """GGFL adapter works with output from build_metric_readiness_panel."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,
                "p4_global_norm": 0.3,
                "poly_fail_detected": False,
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_global_norm": 0.2,
                "p4_global_norm": 0.2,
                "poly_fail_detected": True,
            },
        ]

        panel = build_metric_readiness_panel(annexes)
        result = metric_readiness_panel_for_alignment_view(panel)

        self.assertEqual(result["signal_type"], "SIG-READY")
        self.assertEqual(result["status"], "warn")  # Has block and poly_fail
        self.assertFalse(result["conflict"])
        # Drivers are now reason codes, not cal_ids
        self.assertIn("DRIVER_BLOCK_PRESENT", result["drivers"])
        self.assertIn("DRIVER_POLY_FAIL_PRESENT", result["drivers"])
        self.assertIn("DRIVER_LOW_P4_NORM_RANGE", result["drivers"])  # p4_min=0.2 < 0.35


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

