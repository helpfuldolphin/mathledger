"""
PHASE X — Telemetry × Behavior Consistency Matrix Tests

Tests for CAL-EXP consistency snapshot and matrix aggregation.

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.health.telemetry_fusion_adapter import (
    _map_inconsistency_to_reason_code,
    attach_consistency_matrix_to_evidence,
    build_consistency_matrix,
    emit_cal_exp_telemetry_behavior_consistency,
    summarize_telemetry_behavior_consistency,
)


# -----------------------------------------------------------------------------
# Test: Per-Experiment Consistency Snapshot
# -----------------------------------------------------------------------------

class TestEmitCalExpTelemetryBehaviorConsistency(unittest.TestCase):
    """Tests for emit_cal_exp_telemetry_behavior_consistency()."""

    def test_snapshot_contains_required_fields(self):
        """Snapshot should contain all required fields."""
        consistency = {
            "consistency_status": "INCONSISTENT",
            "telemetry_status": "YELLOW",
            "readiness_status": "GREEN",
            "perf_status": "UNKNOWN",
        }

        snapshot = emit_cal_exp_telemetry_behavior_consistency("cal_exp1", consistency)

        self.assertEqual(snapshot["schema_version"], "1.0.0")
        self.assertEqual(snapshot["cal_id"], "cal_exp1")
        self.assertEqual(snapshot["consistency_status"], "INCONSISTENT")
        self.assertEqual(snapshot["telemetry_status"], "YELLOW")
        self.assertEqual(snapshot["readiness_status"], "GREEN")
        self.assertEqual(snapshot["perf_status"], "UNKNOWN")

    def test_snapshot_is_json_serializable(self):
        """Snapshot should be JSON serializable."""
        consistency = {
            "consistency_status": "CONSISTENT",
            "telemetry_status": "GREEN",
            "readiness_status": "GREEN",
            "perf_status": "GREEN",
        }

        snapshot = emit_cal_exp_telemetry_behavior_consistency("cal_exp2", consistency)

        # Should not raise exception
        json_str = json.dumps(snapshot)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["cal_id"], "cal_exp2")

    def test_snapshot_is_deterministic(self):
        """Snapshot should be deterministic for fixed inputs."""
        consistency = {
            "consistency_status": "PARTIAL",
            "telemetry_status": "YELLOW",
            "readiness_status": "YELLOW",
            "perf_status": "GREEN",
        }

        snapshot1 = emit_cal_exp_telemetry_behavior_consistency("cal_exp3", consistency)
        snapshot2 = emit_cal_exp_telemetry_behavior_consistency("cal_exp3", consistency)

        self.assertEqual(snapshot1, snapshot2)

    def test_snapshot_persists_to_file(self):
        """Snapshot should persist to file when output_dir provided."""
        consistency = {
            "consistency_status": "INCONSISTENT",
            "telemetry_status": "RED",
            "readiness_status": "GREEN",
            "perf_status": "GREEN",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            snapshot = emit_cal_exp_telemetry_behavior_consistency(
                "cal_exp1", consistency, output_dir=output_dir
            )

            # File should exist
            expected_path = output_dir / "telemetry_behavior_consistency_cal_exp1.json"
            self.assertTrue(expected_path.exists())

            # File should contain valid JSON
            with open(expected_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded["cal_id"], "cal_exp1")
            self.assertEqual(loaded["consistency_status"], "INCONSISTENT")

    def test_snapshot_does_not_persist_when_no_output_dir(self):
        """Snapshot should not persist when output_dir is None."""
        consistency = {
            "consistency_status": "CONSISTENT",
            "telemetry_status": "GREEN",
            "readiness_status": "GREEN",
            "perf_status": "GREEN",
        }

        snapshot = emit_cal_exp_telemetry_behavior_consistency("cal_exp1", consistency)

        # Should still return snapshot
        self.assertEqual(snapshot["cal_id"], "cal_exp1")
        # But no file should be created (we can't easily test this without checking filesystem)

    def test_snapshot_handles_missing_consistency_fields(self):
        """Snapshot should handle missing consistency fields gracefully."""
        consistency = {
            "consistency_status": "CONSISTENT",
            # Missing other fields
        }

        snapshot = emit_cal_exp_telemetry_behavior_consistency("cal_exp1", consistency)

        # Should use defaults
        self.assertEqual(snapshot["telemetry_status"], "GREEN")
        self.assertEqual(snapshot["readiness_status"], "UNKNOWN")
        self.assertEqual(snapshot["perf_status"], "UNKNOWN")


# -----------------------------------------------------------------------------
# Test: Consistency Matrix
# -----------------------------------------------------------------------------

class TestMapInconsistencyToReasonCode(unittest.TestCase):
    """Tests for _map_inconsistency_to_reason_code()."""

    def test_tel_warn_behav_ok(self):
        """Should return TEL_WARN_BEHAV_OK when telemetry YELLOW and both behavior GREEN."""
        code = _map_inconsistency_to_reason_code("YELLOW", "GREEN", "GREEN")
        self.assertEqual(code, "TEL_WARN_BEHAV_OK")

    def test_tel_red_behav_ok(self):
        """Should return TEL_RED_BEHAV_OK when telemetry RED and both behavior GREEN."""
        code = _map_inconsistency_to_reason_code("RED", "GREEN", "GREEN")
        self.assertEqual(code, "TEL_RED_BEHAV_OK")

    def test_tel_red_behav_ok_with_unknown(self):
        """Should return TEL_RED_BEHAV_OK when telemetry RED and behavior UNKNOWN (treated as OK)."""
        code = _map_inconsistency_to_reason_code("RED", "UNKNOWN", "GREEN")
        self.assertEqual(code, "TEL_RED_BEHAV_OK")

    def test_tel_warn_readiness_ok(self):
        """Should return TEL_WARN_READINESS_OK when telemetry YELLOW and readiness GREEN."""
        code = _map_inconsistency_to_reason_code("YELLOW", "GREEN", "YELLOW")
        self.assertEqual(code, "TEL_WARN_READINESS_OK")

    def test_tel_warn_perf_ok(self):
        """Should return TEL_WARN_PERF_OK when telemetry YELLOW and perf GREEN."""
        code = _map_inconsistency_to_reason_code("YELLOW", "YELLOW", "GREEN")
        self.assertEqual(code, "TEL_WARN_PERF_OK")

    def test_tel_ok_behav_warn(self):
        """Should return TEL_OK_BEHAV_WARN when telemetry GREEN and behavior YELLOW."""
        code = _map_inconsistency_to_reason_code("GREEN", "YELLOW", "GREEN")
        self.assertEqual(code, "TEL_OK_BEHAV_WARN")
        
        code = _map_inconsistency_to_reason_code("GREEN", "GREEN", "YELLOW")
        self.assertEqual(code, "TEL_OK_BEHAV_WARN")

    def test_tel_ok_behav_red(self):
        """Should return TEL_OK_BEHAV_RED when telemetry GREEN and behavior RED."""
        code = _map_inconsistency_to_reason_code("GREEN", "RED", "GREEN")
        self.assertEqual(code, "TEL_OK_BEHAV_RED")
        
        code = _map_inconsistency_to_reason_code("GREEN", "GREEN", "RED")
        self.assertEqual(code, "TEL_OK_BEHAV_RED")
        
        code = _map_inconsistency_to_reason_code("GREEN", "RED", "RED")
        self.assertEqual(code, "TEL_OK_BEHAV_RED")

    def test_returns_none_for_consistent(self):
        """Should return None when all statuses are consistent (all GREEN)."""
        code = _map_inconsistency_to_reason_code("GREEN", "GREEN", "GREEN")
        self.assertIsNone(code)

    def test_returns_none_for_unknown_telemetry(self):
        """Should return None when telemetry status is not recognized."""
        code = _map_inconsistency_to_reason_code("UNKNOWN", "GREEN", "GREEN")
        self.assertIsNone(code)

    def test_mapping_is_deterministic(self):
        """Reason code mapping should be deterministic for same inputs."""
        code1 = _map_inconsistency_to_reason_code("YELLOW", "GREEN", "GREEN")
        code2 = _map_inconsistency_to_reason_code("YELLOW", "GREEN", "GREEN")
        self.assertEqual(code1, code2)


class TestBuildConsistencyMatrix(unittest.TestCase):
    """Tests for build_consistency_matrix()."""

    def test_matrix_counts_consistency_statuses(self):
        """Matrix should correctly count CONSISTENT, INCONSISTENT, PARTIAL."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "UNKNOWN",
            },
            {
                "cal_id": "cal_exp3",
                "consistency_status": "PARTIAL",
                "telemetry_status": "YELLOW",
                "readiness_status": "YELLOW",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertEqual(matrix["total_experiments"], 3)
        self.assertEqual(matrix["consistency_counts"]["CONSISTENT"], 1)
        self.assertEqual(matrix["consistency_counts"]["INCONSISTENT"], 1)
        self.assertEqual(matrix["consistency_counts"]["PARTIAL"], 1)

    def test_matrix_identifies_inconsistent_experiments(self):
        """Matrix should identify inconsistent experiments with reasons and reason codes."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "UNKNOWN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertEqual(len(matrix["inconsistent_experiments"]), 2)
        self.assertEqual(matrix["inconsistent_experiments"][0]["cal_id"], "cal_exp1")
        self.assertIn("telemetry RED", matrix["inconsistent_experiments"][0]["reason"])
        self.assertIn("readiness GREEN", matrix["inconsistent_experiments"][0]["reason"])
        # Check reason code
        self.assertEqual(matrix["inconsistent_experiments"][0]["reason_code"], "TEL_RED_BEHAV_OK")
        self.assertEqual(matrix["inconsistent_experiments"][1]["reason_code"], "TEL_WARN_READINESS_OK")

    def test_matrix_handles_empty_snapshots(self):
        """Matrix should handle empty snapshot list gracefully."""
        matrix = build_consistency_matrix([])

        self.assertEqual(matrix["total_experiments"], 0)
        self.assertEqual(matrix["consistency_counts"]["CONSISTENT"], 0)
        self.assertEqual(matrix["consistency_counts"]["INCONSISTENT"], 0)
        self.assertEqual(matrix["consistency_counts"]["PARTIAL"], 0)
        self.assertEqual(len(matrix["inconsistent_experiments"]), 0)
        self.assertIn("No consistency snapshots", matrix["summary"])

    def test_matrix_summary_all_consistent(self):
        """Matrix summary should reflect all consistent experiments."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertIn("All 2 experiment(s) show consistent", matrix["summary"])

    def test_matrix_summary_all_inconsistent(self):
        """Matrix summary should reflect all inconsistent experiments."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "UNKNOWN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertIn("All 2 experiment(s) show inconsistent", matrix["summary"])

    def test_matrix_summary_mixed_pattern(self):
        """Matrix summary should reflect mixed consistency pattern."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "UNKNOWN",
            },
            {
                "cal_id": "cal_exp3",
                "consistency_status": "PARTIAL",
                "telemetry_status": "YELLOW",
                "readiness_status": "YELLOW",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertIn("Mixed alignment pattern", matrix["summary"])
        self.assertIn("1 consistent", matrix["summary"])
        self.assertIn("1 inconsistent", matrix["summary"])
        self.assertIn("1 partial", matrix["summary"])

    def test_matrix_is_json_serializable(self):
        """Matrix should be JSON serializable."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        # Should not raise exception
        json_str = json.dumps(matrix)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertIn("total_experiments", deserialized)
        self.assertIn("consistency_counts", deserialized)
        self.assertIn("signals", deserialized)

    def test_matrix_includes_signals_section(self):
        """Matrix should include signals section with counts, top_inconsistent_cal_ids, and histogram."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp3",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        self.assertIn("signals", matrix)
        signals = matrix["signals"]
        
        # Check counts
        self.assertIn("counts", signals)
        self.assertEqual(signals["counts"]["CONSISTENT"], 1)
        self.assertEqual(signals["counts"]["INCONSISTENT"], 2)
        
        # Check top_inconsistent_cal_ids (should be top 3, sorted)
        self.assertIn("top_inconsistent_cal_ids", signals)
        self.assertEqual(len(signals["top_inconsistent_cal_ids"]), 2)  # Only 2 inconsistent
        self.assertIn("cal_exp1", signals["top_inconsistent_cal_ids"])
        self.assertIn("cal_exp2", signals["top_inconsistent_cal_ids"])
        
        # Check reason_code_histogram
        self.assertIn("reason_code_histogram", signals)
        histogram = signals["reason_code_histogram"]
        self.assertEqual(histogram.get("TEL_WARN_BEHAV_OK", 0), 1)
        self.assertEqual(histogram.get("TEL_RED_BEHAV_OK", 0), 1)

    def test_matrix_reason_code_histogram_correctness(self):
        """Reason code histogram should correctly count reason codes."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp3",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        histogram = matrix["signals"]["reason_code_histogram"]
        self.assertEqual(histogram.get("TEL_WARN_BEHAV_OK", 0), 2)
        self.assertEqual(histogram.get("TEL_RED_BEHAV_OK", 0), 1)

    def test_matrix_top_inconsistent_cal_ids_limit(self):
        """top_inconsistent_cal_ids should be limited to top 3."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp3",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "YELLOW",
            },
            {
                "cal_id": "cal_exp4",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "RED",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp5",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "YELLOW",
                "perf_status": "GREEN",
            },
        ]

        matrix = build_consistency_matrix(snapshots)

        top_ids = matrix["signals"]["top_inconsistent_cal_ids"]
        self.assertLessEqual(len(top_ids), 3)

    def test_matrix_is_deterministic(self):
        """Matrix should be deterministic for same inputs."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "CONSISTENT",
                "telemetry_status": "GREEN",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]

        matrix1 = build_consistency_matrix(snapshots)
        matrix2 = build_consistency_matrix(snapshots)

        # Compare all fields
        self.assertEqual(matrix1["total_experiments"], matrix2["total_experiments"])
        self.assertEqual(matrix1["consistency_counts"], matrix2["consistency_counts"])
        self.assertEqual(matrix1["inconsistent_experiments"], matrix2["inconsistent_experiments"])
        self.assertEqual(matrix1["signals"], matrix2["signals"])
        self.assertEqual(matrix1["summary"], matrix2["summary"])
        
        # Verify signals are deterministic
        self.assertEqual(matrix1["signals"]["counts"], matrix2["signals"]["counts"])
        self.assertEqual(matrix1["signals"]["top_inconsistent_cal_ids"], matrix2["signals"]["top_inconsistent_cal_ids"])
        self.assertEqual(matrix1["signals"]["reason_code_histogram"], matrix2["signals"]["reason_code_histogram"])


# -----------------------------------------------------------------------------
# Test: Evidence Attachment
# -----------------------------------------------------------------------------

class TestAttachConsistencyMatrixToEvidence(unittest.TestCase):
    """Tests for attach_consistency_matrix_to_evidence()."""

    def test_attaches_matrix_to_evidence(self):
        """Should attach consistency matrix under evidence['governance']['telemetry_behavior_panel']."""
        evidence = {
            "timestamp": "2024-01-01",
            "data": {"some": "data"},
        }
        matrix = {
            "schema_version": "1.0.0",
            "total_experiments": 3,
            "consistency_counts": {
                "CONSISTENT": 2,
                "INCONSISTENT": 1,
                "PARTIAL": 0,
            },
            "inconsistent_experiments": [
                {"cal_id": "cal_exp2", "reason": "telemetry YELLOW vs readiness GREEN"},
            ],
            "summary": "Mixed alignment pattern detected.",
        }

        enriched = attach_consistency_matrix_to_evidence(evidence, matrix)

        self.assertIn("governance", enriched)
        self.assertIn("telemetry_behavior_panel", enriched["governance"])
        self.assertEqual(enriched["governance"]["telemetry_behavior_panel"]["total_experiments"], 3)
        self.assertEqual(len(enriched["governance"]["telemetry_behavior_panel"]["inconsistent_experiments"]), 1)

    def test_evidence_is_non_mutating(self):
        """Should not modify input evidence dict."""
        evidence = {
            "timestamp": "2024-01-01",
        }
        matrix = {
            "total_experiments": 1,
            "consistency_counts": {"CONSISTENT": 1, "INCONSISTENT": 0, "PARTIAL": 0},
            "inconsistent_experiments": [],
            "summary": "All consistent.",
        }

        evidence_copy = evidence.copy()
        enriched = attach_consistency_matrix_to_evidence(evidence, matrix)

        # Original evidence should be unchanged
        self.assertEqual(evidence, evidence_copy)
        self.assertNotIn("governance", evidence)

        # Enriched should be a new dict
        self.assertIsNot(enriched, evidence)

    def test_evidence_is_json_serializable(self):
        """Enriched evidence should be JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01",
        }
        matrix = {
            "total_experiments": 2,
            "consistency_counts": {"CONSISTENT": 1, "INCONSISTENT": 1, "PARTIAL": 0},
            "inconsistent_experiments": [{"cal_id": "cal_exp1", "reason": "test"}],
            "summary": "Mixed pattern.",
        }

        enriched = attach_consistency_matrix_to_evidence(evidence, matrix)

        # Should not raise exception
        json_str = json.dumps(enriched)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertIn("governance", deserialized)
        self.assertIn("telemetry_behavior_panel", deserialized["governance"])

    def test_evidence_handles_existing_governance(self):
        """Should preserve existing governance keys."""
        evidence = {
            "timestamp": "2024-01-01",
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }
        matrix = {
            "total_experiments": 1,
            "consistency_counts": {"CONSISTENT": 1, "INCONSISTENT": 0, "PARTIAL": 0},
            "inconsistent_experiments": [],
            "summary": "All consistent.",
        }

        enriched = attach_consistency_matrix_to_evidence(evidence, matrix)

        # Should preserve existing governance
        self.assertIn("other_tile", enriched["governance"])
        self.assertEqual(enriched["governance"]["other_tile"]["status"], "OK")

        # Should add telemetry_behavior_panel
        self.assertIn("telemetry_behavior_panel", enriched["governance"])


# -----------------------------------------------------------------------------
# Test: End-to-End Workflow
# -----------------------------------------------------------------------------

class TestConsistencyWorkflow(unittest.TestCase):
    """Tests for end-to-end consistency workflow."""

    def test_full_workflow_from_consistency_to_matrix(self):
        """Test full workflow: consistency → snapshot → matrix → evidence."""
        # Step 1: Build consistency summaries
        telemetry_tile1 = {"status_light": "YELLOW"}
        readiness_annex1 = {"status_light": "GREEN"}
        consistency1 = summarize_telemetry_behavior_consistency(
            telemetry_tile1, readiness_annex=readiness_annex1
        )

        telemetry_tile2 = {"status_light": "GREEN"}
        readiness_annex2 = {"status_light": "GREEN"}
        consistency2 = summarize_telemetry_behavior_consistency(
            telemetry_tile2, readiness_annex=readiness_annex2
        )

        # Step 2: Emit snapshots
        snapshot1 = emit_cal_exp_telemetry_behavior_consistency("cal_exp1", consistency1)
        snapshot2 = emit_cal_exp_telemetry_behavior_consistency("cal_exp2", consistency2)

        # Step 3: Build matrix
        matrix = build_consistency_matrix([snapshot1, snapshot2])

        # Step 4: Attach to evidence
        evidence = {"timestamp": "2024-01-01"}
        enriched = attach_consistency_matrix_to_evidence(evidence, matrix)

        # Verify results
        self.assertEqual(matrix["total_experiments"], 2)
        self.assertEqual(matrix["consistency_counts"]["INCONSISTENT"], 1)
        self.assertEqual(matrix["consistency_counts"]["CONSISTENT"], 1)
        self.assertIn("telemetry_behavior_panel", enriched["governance"])


if __name__ == "__main__":
    unittest.main()



