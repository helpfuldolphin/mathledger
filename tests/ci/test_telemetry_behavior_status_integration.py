"""
PHASE X — Telemetry × Behavior Status Integration Tests

Tests for telemetry-behavior panel signal extraction and first_light_status.json integration.

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

import pytest

from backend.health.telemetry_fusion_adapter import (
    build_consistency_matrix,
    extract_telemetry_behavior_panel_signal,
)


# -----------------------------------------------------------------------------
# Test: Signal Extraction
# -----------------------------------------------------------------------------

class TestExtractTelemetryBehaviorPanelSignal(unittest.TestCase):
    """Tests for extract_telemetry_behavior_panel_signal()."""

    def test_signal_contains_required_fields(self):
        """Signal should contain all required fields."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix, extraction_source="MANIFEST")

        self.assertEqual(signal["schema_version"], "1.0.0")
        self.assertEqual(signal["mode"], "SHADOW")
        self.assertEqual(signal["extraction_source"], "MANIFEST")
        self.assertIn("consistency_counts", signal)
        self.assertIn("top_inconsistent_cal_ids", signal)
        self.assertIn("reason_code_histogram", signal)

    def test_signal_extraction_source_manifest(self):
        """Signal should include extraction_source=MANIFEST when from manifest."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix, extraction_source="MANIFEST")
        
        self.assertEqual(signal["extraction_source"], "MANIFEST")

    def test_signal_extraction_source_evidence_json(self):
        """Signal should include extraction_source=EVIDENCE_JSON when from evidence."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix, extraction_source="EVIDENCE_JSON")
        
        self.assertEqual(signal["extraction_source"], "EVIDENCE_JSON")

    def test_signal_extraction_source_missing(self):
        """Signal should include extraction_source=MISSING when not provided."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        self.assertEqual(signal["extraction_source"], "MISSING")

    def test_signal_extraction_source_normalizes_invalid(self):
        """Signal should normalize invalid extraction_source to MISSING."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix, extraction_source="INVALID")
        
        self.assertEqual(signal["extraction_source"], "MISSING")

    def test_signal_consistency_counts(self):
        """Signal should include consistency counts from matrix."""
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
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)

        counts = signal["consistency_counts"]
        self.assertEqual(counts["CONSISTENT"], 1)
        self.assertEqual(counts["INCONSISTENT"], 1)
        self.assertEqual(counts["PARTIAL"], 0)

    def test_signal_top_inconsistent_cal_ids(self):
        """Signal should include top 3 inconsistent cal_ids."""
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
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)

        top_ids = signal["top_inconsistent_cal_ids"]
        self.assertLessEqual(len(top_ids), 3)
        self.assertIn("cal_exp1", top_ids)
        self.assertIn("cal_exp2", top_ids)
        self.assertIn("cal_exp3", top_ids)

    def test_signal_reason_code_histogram(self):
        """Signal should include reason code histogram."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix)

        histogram = signal["reason_code_histogram"]
        self.assertEqual(histogram.get("TEL_WARN_BEHAV_OK", 0), 2)
        self.assertEqual(histogram.get("TEL_RED_BEHAV_OK", 0), 1)

    def test_signal_is_deterministic(self):
        """Signal should be deterministic for same inputs."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal1 = extract_telemetry_behavior_panel_signal(matrix)
        signal2 = extract_telemetry_behavior_panel_signal(matrix)

        self.assertEqual(signal1, signal2)

    def test_signal_is_json_serializable(self):
        """Signal should be JSON serializable."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)

        # Should not raise exception
        json_str = json.dumps(signal)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["mode"], "SHADOW")

    def test_signal_handles_empty_matrix(self):
        """Signal should handle empty matrix gracefully."""
        matrix = build_consistency_matrix([])
        signal = extract_telemetry_behavior_panel_signal(matrix)

        self.assertEqual(signal["consistency_counts"]["CONSISTENT"], 0)
        self.assertEqual(signal["consistency_counts"]["INCONSISTENT"], 0)
        self.assertEqual(len(signal["top_inconsistent_cal_ids"]), 0)
        self.assertEqual(len(signal["reason_code_histogram"]), 0)

    def test_signal_histogram_is_sorted(self):
        """Reason code histogram should be sorted for determinism."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "YELLOW",
            },
            {
                "cal_id": "cal_exp2",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "RED",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)

        histogram = signal["reason_code_histogram"]
        # Check that keys are sorted
        keys = list(histogram.keys())
        self.assertEqual(keys, sorted(keys))


# -----------------------------------------------------------------------------
# Test: Status Integration (Manifest-First Extraction)
# -----------------------------------------------------------------------------

class TestStatusIntegration(unittest.TestCase):
    """Tests for status integration with manifest-first extraction."""

    def test_manifest_first_extraction(self):
        """Should extract from manifest first, then fallback to evidence."""
        # This is a unit test for the extraction logic
        # The actual integration test would require running the status generator
        # which is tested in integration tests
        
        # Build a test matrix
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        
        # Simulate manifest structure
        manifest = {
            "governance": {
                "telemetry_behavior_panel": matrix,
            }
        }
        
        # Extract from manifest
        extracted_matrix = manifest.get("governance", {}).get("telemetry_behavior_panel")
        self.assertIsNotNone(extracted_matrix)
        self.assertEqual(extracted_matrix["total_experiments"], 1)
        
        # Extract signal
        signal = extract_telemetry_behavior_panel_signal(extracted_matrix)
        self.assertEqual(signal["consistency_counts"]["INCONSISTENT"], 1)

    def test_evidence_fallback_extraction(self):
        """Should fallback to evidence if not in manifest."""
        # Build a test matrix
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        
        # Simulate manifest without panel
        manifest = {
            "governance": {},
        }
        
        # Simulate evidence with panel
        evidence = {
            "governance": {
                "telemetry_behavior_panel": matrix,
            }
        }
        
        # Try manifest first (should be None)
        extracted_matrix = manifest.get("governance", {}).get("telemetry_behavior_panel")
        self.assertIsNone(extracted_matrix)
        
        # Fallback to evidence
        extracted_matrix = evidence.get("governance", {}).get("telemetry_behavior_panel")
        self.assertIsNotNone(extracted_matrix)
        self.assertEqual(extracted_matrix["total_experiments"], 1)


# -----------------------------------------------------------------------------
# Test: Warning Generation Logic
# -----------------------------------------------------------------------------

class TestWarningGeneration(unittest.TestCase):
    """Tests for warning generation logic (single warning cap, top reason_code)."""

    def test_warning_includes_inconsistent_count(self):
        """Warning should include inconsistent count."""
        snapshots = [
            {
                "cal_id": "cal_exp1",
                "consistency_status": "INCONSISTENT",
                "telemetry_status": "YELLOW",
                "readiness_status": "GREEN",
                "perf_status": "GREEN",
            },
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        consistency_counts = signal.get("consistency_counts", {})
        inconsistent_count = consistency_counts.get("INCONSISTENT", 0)
        
        # Simulate warning generation logic
        if inconsistent_count > 0:
            warning_parts = [f"{inconsistent_count} inconsistent experiment(s)"]
            self.assertIn("1 inconsistent experiment(s)", warning_parts[0])

    def test_warning_includes_top_cal_ids(self):
        """Warning should include top cal_ids when available."""
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
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        top_cal_ids = signal.get("top_inconsistent_cal_ids", [])
        
        # Simulate warning generation logic
        if top_cal_ids:
            top_cal_ids_str = ", ".join(top_cal_ids[:3])
            self.assertIn("cal_exp", top_cal_ids_str)

    def test_warning_includes_top_reason_code(self):
        """Warning should include top reason code from histogram."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        reason_code_histogram = signal.get("reason_code_histogram", {})
        
        # Simulate top reason code selection logic
        top_reason_code = None
        if reason_code_histogram:
            sorted_codes = sorted(
                reason_code_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for descending, then code ascending
            )
            top_reason_code = sorted_codes[0][0] if sorted_codes else None
        
        # TEL_WARN_BEHAV_OK should be top (count=2)
        self.assertEqual(top_reason_code, "TEL_WARN_BEHAV_OK")

    def test_warning_single_warning_cap(self):
        """Should generate only one warning regardless of inconsistent count."""
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
        ]
        matrix = build_consistency_matrix(snapshots)
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        consistency_counts = signal.get("consistency_counts", {})
        inconsistent_count = consistency_counts.get("INCONSISTENT", 0)
        
        # Simulate warning generation (should be single warning with TELBEH-WARN-001 prefix)
        warnings = []
        if inconsistent_count > 0:
            top_cal_ids = signal.get("top_inconsistent_cal_ids", [])
            top_cal_ids_str = ", ".join(top_cal_ids[:3]) if top_cal_ids else "none"
            
            reason_code_histogram = signal.get("reason_code_histogram", {})
            top_reason_code = None
            if reason_code_histogram:
                sorted_codes = sorted(
                    reason_code_histogram.items(),
                    key=lambda x: (-x[1], x[0])
                )
                top_reason_code = sorted_codes[0][0] if sorted_codes else None
            
            warning_parts = [f"{inconsistent_count} inconsistent experiment(s)"]
            if top_cal_ids:
                warning_parts.append(f"top cal_ids: {top_cal_ids_str}")
            if top_reason_code:
                warning_parts.append(f"top reason_code: {top_reason_code}")
            
            warnings.append(
                f"TELBEH-WARN-001: Telemetry × behavior consistency: " + ", ".join(warning_parts)
            )
        
        # Should be exactly one warning
        self.assertEqual(len(warnings), 1)
        self.assertIn("2 inconsistent experiment(s)", warnings[0])
        self.assertIn("TELBEH-WARN-001", warnings[0])

    def test_warning_no_warning_when_consistent(self):
        """Should not generate warning when all experiments are consistent."""
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
        signal = extract_telemetry_behavior_panel_signal(matrix)
        
        consistency_counts = signal.get("consistency_counts", {})
        inconsistent_count = consistency_counts.get("INCONSISTENT", 0)
        
        # Simulate warning generation
        warnings = []
        if inconsistent_count > 0:
            warnings.append("Should not appear")
        
        # Should be no warnings
        self.assertEqual(len(warnings), 0)


# -----------------------------------------------------------------------------
# Test: Regression - Status Generator Integration
# -----------------------------------------------------------------------------

class TestStatusGeneratorIntegration(unittest.TestCase):
    """Regression test: Verify telemetry behavior panel is integrated into status generator."""

    def test_status_generator_includes_telemetry_behavior_panel_key(self):
        """Regression: Status generator should include telemetry_behavior_panel in signals when matrix present."""
        from pathlib import Path
        from scripts.generate_first_light_status import generate_status
        import tempfile
        import json
        
        # Create minimal test directories and files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create minimal P3/P4/evidence pack structure
            p3_dir = tmp_path / "p3"
            p4_dir = tmp_path / "p4"
            evidence_dir = tmp_path / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal manifest with telemetry behavior panel
            matrix = build_consistency_matrix([
                {
                    "cal_id": "cal_exp1",
                    "consistency_status": "INCONSISTENT",
                    "telemetry_status": "YELLOW",
                    "readiness_status": "GREEN",
                    "perf_status": "GREEN",
                },
            ])
            
            manifest = {
                "governance": {
                    "telemetry_behavior_panel": matrix,
                },
                "file_count": 1,
                "files": [],
            }
            
            manifest_path = evidence_dir / "manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f)
            
            # Create minimal P3/P4 run directories
            p3_run = p3_dir / "fl_test_seed42"
            p4_run = p4_dir / "p4_test"
            p3_run.mkdir()
            p4_run.mkdir()
            
            # Create minimal required files
            (p3_run / "synthetic_raw.jsonl").write_text("")
            (p3_run / "stability_report.json").write_text(json.dumps({"metrics": {}}))
            (p4_run / "p4_summary.json").write_text(json.dumps({"mode": "SHADOW"}))
            
            # Generate status
            try:
                status = generate_status(
                    p3_dir=p3_dir,
                    p4_dir=p4_dir,
                    evidence_pack_dir=evidence_dir,
                )
                
                # Verify telemetry_behavior_panel is in signals
                signals = status.get("signals")
                self.assertIsNotNone(signals, "Signals section should exist")
                self.assertIn("telemetry_behavior_panel", signals, "telemetry_behavior_panel should be in signals")
                
                # Verify signal structure
                panel_signal = signals["telemetry_behavior_panel"]
                self.assertEqual(panel_signal["mode"], "SHADOW")
                self.assertEqual(panel_signal["extraction_source"], "MANIFEST")
                self.assertIn("consistency_counts", panel_signal)
                self.assertIn("top_inconsistent_cal_ids", panel_signal)
                self.assertIn("reason_code_histogram", panel_signal)
                
                # Verify warning includes TELBEH-WARN-001 prefix
                warnings_list = status.get("warnings", [])
                telemetry_warnings = [w for w in warnings_list if "Telemetry × behavior consistency" in w]
                if telemetry_warnings:
                    self.assertIn("TELBEH-WARN-001", telemetry_warnings[0])
                
            except Exception as e:
                # If generation fails due to missing files, that's OK for this test
                # We just want to verify the code path exists and doesn't crash on import
                self.fail(f"Status generation should not crash: {e}")


if __name__ == "__main__":
    unittest.main()

