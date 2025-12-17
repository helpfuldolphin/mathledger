"""
PHASE X — Telemetry Governance Evidence & Council Tests

Tests for telemetry governance evidence attachment and council summary functions.

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

import pytest

from backend.health.telemetry_fusion_adapter import (
    attach_telemetry_governance_to_evidence,
    attach_telemetry_governance_to_p3_stability_report,
    attach_telemetry_governance_to_p4_calibration_report,
    summarize_telemetry_behavior_consistency,
    summarize_telemetry_for_uplift_council,
)


# -----------------------------------------------------------------------------
# Test: Evidence Attachment
# -----------------------------------------------------------------------------

class TestAttachTelemetryGovernanceToEvidence(unittest.TestCase):
    """Tests for attach_telemetry_governance_to_evidence()."""

    def test_attaches_telemetry_governance_to_evidence(self):
        """Should attach telemetry governance under evidence['governance']['telemetry']."""
        evidence = {
            "timestamp": "2024-01-01",
            "data": {"some": "data"},
        }
        governance_tile = {
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "status_light": "YELLOW",
        }
        first_light_signal = {
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "num_incoherence_vectors": 3,
        }

        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal
        )

        self.assertIn("governance", enriched)
        self.assertIn("telemetry", enriched["governance"])
        self.assertEqual(enriched["governance"]["telemetry"]["fusion_band"], "MEDIUM")
        self.assertEqual(enriched["governance"]["telemetry"]["uplift_gate_status"], "ATTENTION")
        self.assertEqual(enriched["governance"]["telemetry"]["num_incoherence_vectors"], 3)
        self.assertEqual(enriched["governance"]["telemetry"]["status_light"], "YELLOW")

    def test_evidence_is_non_mutating(self):
        """Should not modify input evidence dict."""
        evidence = {
            "timestamp": "2024-01-01",
            "data": {"some": "data"},
        }
        governance_tile = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "status_light": "GREEN",
        }
        first_light_signal = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "num_incoherence_vectors": 0,
        }

        evidence_copy = evidence.copy()
        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal
        )

        # Original evidence should be unchanged
        self.assertEqual(evidence, evidence_copy)
        self.assertNotIn("governance", evidence)

        # Enriched should be a new dict
        self.assertIsNot(enriched, evidence)

    def test_evidence_is_json_serializable(self):
        """Enriched evidence should be JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01",
            "data": {"some": "data"},
        }
        governance_tile = {
            "fusion_band": "HIGH",
            "uplift_gate_status": "BLOCK",
            "status_light": "RED",
        }
        first_light_signal = {
            "fusion_band": "HIGH",
            "uplift_gate_status": "BLOCK",
            "num_incoherence_vectors": 5,
        }

        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal
        )

        # Should not raise exception
        json_str = json.dumps(enriched)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertIn("governance", deserialized)
        self.assertIn("telemetry", deserialized["governance"])

    def test_evidence_handles_existing_governance(self):
        """Should preserve existing governance keys."""
        evidence = {
            "timestamp": "2024-01-01",
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }
        governance_tile = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "status_light": "GREEN",
        }
        first_light_signal = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "num_incoherence_vectors": 0,
        }

        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal
        )

        # Should preserve existing governance
        self.assertIn("other_tile", enriched["governance"])
        self.assertEqual(enriched["governance"]["other_tile"]["status"], "OK")

        # Should add telemetry
        self.assertIn("telemetry", enriched["governance"])


# -----------------------------------------------------------------------------
# Test: P3/P4 Report Integration
# -----------------------------------------------------------------------------

class TestAttachTelemetryGovernanceToP3StabilityReport(unittest.TestCase):
    """Tests for attach_telemetry_governance_to_p3_stability_report()."""

    def test_attaches_telemetry_governance_to_p3_report(self):
        """Should attach telemetry_governance field to P3 stability report."""
        stability_report = {
            "run_id": "fl_20240101_120000",
            "metrics": {"mean_rsi": 0.85},
        }
        governance_tile = {
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "status_light": "YELLOW",
        }
        first_light_signal = {
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "num_incoherence_vectors": 2,
        }

        updated = attach_telemetry_governance_to_p3_stability_report(
            stability_report, governance_tile, first_light_signal
        )

        self.assertIn("telemetry_governance", updated)
        self.assertEqual(updated["telemetry_governance"]["fusion_band"], "MEDIUM")
        self.assertEqual(updated["telemetry_governance"]["uplift_gate_status"], "ATTENTION")
        self.assertEqual(updated["telemetry_governance"]["num_incoherence_vectors"], 2)
        self.assertEqual(updated["telemetry_governance"]["status_light"], "YELLOW")

    def test_p3_report_is_non_mutating(self):
        """Should not modify input report dict."""
        stability_report = {
            "run_id": "fl_20240101_120000",
            "metrics": {"mean_rsi": 0.85},
        }
        governance_tile = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "status_light": "GREEN",
        }
        first_light_signal = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "num_incoherence_vectors": 0,
        }

        report_copy = stability_report.copy()
        updated = attach_telemetry_governance_to_p3_stability_report(
            stability_report, governance_tile, first_light_signal
        )

        # Original report should be unchanged
        self.assertEqual(stability_report, report_copy)
        self.assertNotIn("telemetry_governance", stability_report)

        # Updated should be a new dict
        self.assertIsNot(updated, stability_report)


class TestAttachTelemetryGovernanceToP4CalibrationReport(unittest.TestCase):
    """Tests for attach_telemetry_governance_to_p4_calibration_report()."""

    def test_attaches_telemetry_governance_to_p4_report(self):
        """Should attach telemetry_governance field to P4 calibration report."""
        calibration_report = {
            "run_id": "p4_20240101_120000",
            "calibration": {"divergence": 0.05},
        }
        governance_tile = {
            "fusion_band": "HIGH",
            "uplift_gate_status": "BLOCK",
            "status_light": "RED",
        }
        first_light_signal = {
            "fusion_band": "HIGH",
            "uplift_gate_status": "BLOCK",
            "num_incoherence_vectors": 7,
        }

        updated = attach_telemetry_governance_to_p4_calibration_report(
            calibration_report, governance_tile, first_light_signal
        )

        self.assertIn("telemetry_governance", updated)
        self.assertEqual(updated["telemetry_governance"]["fusion_band"], "HIGH")
        self.assertEqual(updated["telemetry_governance"]["uplift_gate_status"], "BLOCK")
        self.assertEqual(updated["telemetry_governance"]["num_incoherence_vectors"], 7)
        self.assertEqual(updated["telemetry_governance"]["status_light"], "RED")

    def test_p4_report_is_non_mutating(self):
        """Should not modify input report dict."""
        calibration_report = {
            "run_id": "p4_20240101_120000",
            "calibration": {"divergence": 0.05},
        }
        governance_tile = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "status_light": "GREEN",
        }
        first_light_signal = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "num_incoherence_vectors": 0,
        }

        report_copy = calibration_report.copy()
        updated = attach_telemetry_governance_to_p4_calibration_report(
            calibration_report, governance_tile, first_light_signal
        )

        # Original report should be unchanged
        self.assertEqual(calibration_report, report_copy)
        self.assertNotIn("telemetry_governance", calibration_report)

        # Updated should be a new dict
        self.assertIsNot(updated, calibration_report)


# -----------------------------------------------------------------------------
# Test: Council Summary
# -----------------------------------------------------------------------------

class TestSummarizeTelemetryForUpliftCouncil(unittest.TestCase):
    """Tests for summarize_telemetry_for_uplift_council()."""

    def test_block_status_when_gate_blocked(self):
        """Should return BLOCK when uplift_gate_status is BLOCK."""
        tile = {
            "uplift_gate_status": "BLOCK",
            "fusion_band": "LOW",
            "incoherence_vectors": [],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["status"], "BLOCK")
        self.assertEqual(summary["uplift_gate_status"], "BLOCK")

    def test_warn_status_when_gate_attention(self):
        """Should return WARN when uplift_gate_status is ATTENTION."""
        tile = {
            "uplift_gate_status": "ATTENTION",
            "fusion_band": "LOW",
            "incoherence_vectors": [],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["status"], "WARN")
        self.assertEqual(summary["uplift_gate_status"], "ATTENTION")

    def test_warn_status_when_fusion_high(self):
        """Should return WARN when fusion_band is HIGH."""
        tile = {
            "uplift_gate_status": "OK",
            "fusion_band": "HIGH",
            "incoherence_vectors": [],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["status"], "WARN")
        self.assertEqual(summary["fusion_band"], "HIGH")

    def test_ok_status_when_all_ok(self):
        """Should return OK when gate is OK and fusion is not HIGH."""
        tile = {
            "uplift_gate_status": "OK",
            "fusion_band": "LOW",
            "incoherence_vectors": [],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["uplift_gate_status"], "OK")
        self.assertEqual(summary["fusion_band"], "LOW")

    def test_includes_num_incoherence_vectors(self):
        """Should include num_incoherence_vectors in summary."""
        tile = {
            "uplift_gate_status": "OK",
            "fusion_band": "LOW",
            "incoherence_vectors": ["vector1", "vector2", "vector3"],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["num_incoherence_vectors"], 3)

    def test_council_summary_is_deterministic(self):
        """Council summary should be deterministic for fixed inputs."""
        tile = {
            "uplift_gate_status": "ATTENTION",
            "fusion_band": "MEDIUM",
            "incoherence_vectors": ["vector1"],
        }

        summary1 = summarize_telemetry_for_uplift_council(tile)
        summary2 = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary1, summary2)

    def test_council_summary_is_json_serializable(self):
        """Council summary should be JSON serializable."""
        tile = {
            "uplift_gate_status": "BLOCK",
            "fusion_band": "HIGH",
            "incoherence_vectors": ["vector1", "vector2"],
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        # Should not raise exception
        json_str = json.dumps(summary)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertIn("status", deserialized)
        self.assertIn("num_incoherence_vectors", deserialized)

    def test_council_summary_handles_missing_incoherence_vectors(self):
        """Should handle missing incoherence_vectors gracefully."""
        tile = {
            "uplift_gate_status": "OK",
            "fusion_band": "LOW",
            # Missing incoherence_vectors
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        self.assertEqual(summary["num_incoherence_vectors"], 0)

    def test_council_summary_handles_non_list_incoherence_vectors(self):
        """Should handle non-list incoherence_vectors gracefully."""
        tile = {
            "uplift_gate_status": "OK",
            "fusion_band": "LOW",
            "incoherence_vectors": "not_a_list",  # Wrong type
        }

        summary = summarize_telemetry_for_uplift_council(tile)

        # Should default to 0
        self.assertEqual(summary["num_incoherence_vectors"], 0)


# -----------------------------------------------------------------------------
# Test: Telemetry-Behavior Consistency Cross-Check
# -----------------------------------------------------------------------------

class TestSummarizeTelemetryBehaviorConsistency(unittest.TestCase):
    """Tests for summarize_telemetry_behavior_consistency()."""

    def test_detects_telemetry_warn_while_readiness_ok(self):
        """Should detect INCONSISTENT when telemetry is WARN/RED while readiness is GREEN."""
        telemetry_tile = {
            "status_light": "YELLOW",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        self.assertEqual(consistency["consistency_status"], "INCONSISTENT")
        self.assertEqual(consistency["telemetry_status"], "YELLOW")
        self.assertEqual(consistency["readiness_status"], "GREEN")
        self.assertGreater(len(consistency["advisory_notes"]), 0)
        self.assertIn("YELLOW", consistency["advisory_notes"][0])
        self.assertIn("GREEN", consistency["advisory_notes"][0])

    def test_detects_telemetry_red_while_readiness_ok(self):
        """Should detect INCONSISTENT when telemetry is RED while readiness is GREEN."""
        telemetry_tile = {
            "status_light": "RED",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        self.assertEqual(consistency["consistency_status"], "INCONSISTENT")
        self.assertEqual(consistency["telemetry_status"], "RED")
        self.assertEqual(consistency["readiness_status"], "GREEN")

    def test_detects_telemetry_warn_while_perf_ok(self):
        """Should detect INCONSISTENT when telemetry is WARN/RED while perf is GREEN."""
        telemetry_tile = {
            "status_light": "YELLOW",
        }
        perf_tile = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, perf_tile=perf_tile
        )

        self.assertEqual(consistency["consistency_status"], "INCONSISTENT")
        self.assertEqual(consistency["telemetry_status"], "YELLOW")
        self.assertEqual(consistency["perf_status"], "GREEN")
        self.assertGreater(len(consistency["advisory_notes"]), 0)

    def test_detects_telemetry_red_while_both_ok(self):
        """Should detect INCONSISTENT when telemetry is RED while both readiness and perf are GREEN."""
        telemetry_tile = {
            "status_light": "RED",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }
        perf_tile = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex, perf_tile=perf_tile
        )

        self.assertEqual(consistency["consistency_status"], "INCONSISTENT")
        self.assertGreater(len(consistency["advisory_notes"]), 0)
        self.assertIn("both readiness and performance", consistency["advisory_notes"][0])

    def test_detects_alignment_when_all_show_issues(self):
        """Should detect PARTIAL/CONSISTENT when telemetry and readiness both show issues."""
        telemetry_tile = {
            "status_light": "YELLOW",
        }
        readiness_annex = {
            "status_light": "YELLOW",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        self.assertIn(consistency["consistency_status"], ("CONSISTENT", "PARTIAL"))
        self.assertGreater(len(consistency["advisory_notes"]), 0)
        self.assertIn("aligns", consistency["advisory_notes"][0].lower())

    def test_detects_all_green_alignment(self):
        """Should detect CONSISTENT when all systems are GREEN."""
        telemetry_tile = {
            "status_light": "GREEN",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }
        perf_tile = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex, perf_tile=perf_tile
        )

        self.assertEqual(consistency["consistency_status"], "CONSISTENT")
        self.assertGreater(len(consistency["advisory_notes"]), 0)
        self.assertIn("all GREEN", consistency["advisory_notes"][0])

    def test_handles_missing_readiness_annex(self):
        """Should handle missing readiness_annex gracefully."""
        telemetry_tile = {
            "status_light": "YELLOW",
        }

        consistency = summarize_telemetry_behavior_consistency(telemetry_tile)

        self.assertEqual(consistency["readiness_status"], "UNKNOWN")
        self.assertEqual(consistency["perf_status"], "UNKNOWN")
        self.assertIsInstance(consistency["advisory_notes"], list)

    def test_handles_missing_perf_tile(self):
        """Should handle missing perf_tile gracefully."""
        telemetry_tile = {
            "status_light": "GREEN",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        self.assertEqual(consistency["perf_status"], "UNKNOWN")
        self.assertIsInstance(consistency["advisory_notes"], list)

    def test_consistency_is_deterministic(self):
        """Consistency summary should be deterministic for fixed inputs."""
        telemetry_tile = {
            "status_light": "YELLOW",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        consistency1 = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )
        consistency2 = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        self.assertEqual(consistency1, consistency2)

    def test_consistency_is_json_serializable(self):
        """Consistency summary should be JSON serializable."""
        telemetry_tile = {
            "status_light": "RED",
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile, readiness_annex=readiness_annex
        )

        import json
        json_str = json.dumps(consistency)
        self.assertIsInstance(json_str, str)

        deserialized = json.loads(json_str)
        self.assertIn("consistency_status", deserialized)
        self.assertIn("advisory_notes", deserialized)

    def test_evidence_includes_behavior_consistency(self):
        """attach_telemetry_governance_to_evidence should include behavior_consistency when provided."""
        evidence = {
            "timestamp": "2024-01-01",
        }
        governance_tile = {
            "status_light": "YELLOW",
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
        }
        first_light_signal = {
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "num_incoherence_vectors": 2,
        }
        readiness_annex = {
            "status_light": "GREEN",
        }

        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal, readiness_annex=readiness_annex
        )

        self.assertIn("behavior_consistency", enriched["governance"]["telemetry"])
        consistency = enriched["governance"]["telemetry"]["behavior_consistency"]
        self.assertEqual(consistency["consistency_status"], "INCONSISTENT")
        self.assertEqual(consistency["telemetry_status"], "YELLOW")
        self.assertEqual(consistency["readiness_status"], "GREEN")

    def test_evidence_without_consistency_when_not_provided(self):
        """attach_telemetry_governance_to_evidence should not include behavior_consistency when not provided."""
        evidence = {
            "timestamp": "2024-01-01",
        }
        governance_tile = {
            "status_light": "GREEN",
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
        }
        first_light_signal = {
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "num_incoherence_vectors": 0,
        }

        enriched = attach_telemetry_governance_to_evidence(
            evidence, governance_tile, first_light_signal
        )

        # Should not include behavior_consistency when readiness/perf not provided
        self.assertNotIn("behavior_consistency", enriched["governance"]["telemetry"])


if __name__ == "__main__":
    unittest.main()

