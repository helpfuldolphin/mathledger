"""
Integration tests for Budget Calibration CLI and Evidence Pack Hook.

Tests cover:
- CLI output file generation (JSONL + summary)
- Output determinism with same seed
- Evidence pack hook attachment
- Calibration summary loading from file

Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.topology.first_light.budget_calibration import (
    CalibrationHarness,
    CalibrationPhase,
)
from backend.topology.first_light.budget_binding import (
    attach_calibration_summary_to_evidence,
    load_calibration_summary_from_file,
    maybe_attach_calibration_to_evidence,
)

# Import CLI functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_budget_calibration import (
    write_jsonl,
    write_summary,
    build_compact_summary,
    run_phase_1_only,
)


# =============================================================================
# CLI Output Tests
# =============================================================================

class TestCLIOutputGeneration:
    """Tests for CLI output file generation."""

    def test_write_jsonl_creates_file(self):
        """JSONL file is created with correct format."""
        harness = CalibrationHarness(seed=42)
        harness.run_phase_1(cycles=10)
        entries = harness.get_all_entries()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            count = write_jsonl(entries, output_path)

            assert output_path.exists()
            assert count == 10

            # Verify JSONL format
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            assert len(lines) == 10
            for line in lines:
                # Each line should be valid JSON
                data = json.loads(line)
                assert "calibration_log" in data

    def test_write_summary_creates_file(self):
        """Summary JSON file is created with correct structure."""
        harness = CalibrationHarness(seed=42)
        result = harness.run_full_experiment()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.json"
            metadata = {"seed": 42, "test": True}
            write_summary(result, output_path, metadata)

            assert output_path.exists()

            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert "schema_version" in data
            assert "generated_at" in data
            assert "shadow_mode" in data
            assert data["shadow_mode"] is True
            assert "metadata" in data
            assert "experiment" in data
            assert "compact_summary" in data

    def test_build_compact_summary_structure(self):
        """Compact summary has required fields for evidence pack."""
        harness = CalibrationHarness(seed=42)
        result = harness.run_full_experiment()

        compact = build_compact_summary(result)

        assert "schema_version" in compact
        assert "experiment_id" in compact
        assert "overall_pass" in compact
        assert "enablement_recommendation" in compact
        assert "phases" in compact

        # Check phase structure
        assert "phase_1" in compact["phases"]
        assert "phase_2" in compact["phases"]
        assert "phase_3" in compact["phases"]

        for phase_key in ["phase_1", "phase_2", "phase_3"]:
            phase = compact["phases"][phase_key]
            assert "cycles" in phase
            assert "fp_rate" in phase
            assert "fn_rate" in phase
            assert "meets_criteria" in phase


class TestCLIDeterminism:
    """Tests for output determinism."""

    def test_jsonl_deterministic_with_same_seed(self):
        """Same seed produces identical JSONL output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run
            harness1 = CalibrationHarness(seed=42)
            harness1.run_phase_1(cycles=50)
            path1 = Path(tmpdir) / "run1.jsonl"
            write_jsonl(harness1.get_all_entries(), path1)

            # Second run with same seed
            harness2 = CalibrationHarness(seed=42)
            harness2.run_phase_1(cycles=50)
            path2 = Path(tmpdir) / "run2.jsonl"
            write_jsonl(harness2.get_all_entries(), path2)

            # Compare content (excluding timestamps)
            with open(path1, 'r') as f1, open(path2, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

            assert len(lines1) == len(lines2)

            for l1, l2 in zip(lines1, lines2):
                d1 = json.loads(l1)
                d2 = json.loads(l2)

                # Remove timestamp for comparison
                d1["calibration_log"]["timestamp"] = "REMOVED"
                d2["calibration_log"]["timestamp"] = "REMOVED"

                assert d1 == d2

    def test_summary_deterministic_with_same_seed(self):
        """Same seed produces identical summary (except timestamps)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run
            harness1 = CalibrationHarness(seed=42)
            result1 = run_phase_1_only(harness1, 50, Path(tmpdir), verbose=False)
            compact1 = build_compact_summary(result1)

            # Second run with same seed
            harness2 = CalibrationHarness(seed=42)
            result2 = run_phase_1_only(harness2, 50, Path(tmpdir), verbose=False)
            compact2 = build_compact_summary(result2)

            # Core metrics should be identical
            assert compact1["overall_pass"] == compact2["overall_pass"]
            assert compact1["enablement_recommendation"] == compact2["enablement_recommendation"]

            if "phase_1" in compact1["phases"] and "phase_1" in compact2["phases"]:
                assert compact1["phases"]["phase_1"]["fp_rate"] == compact2["phases"]["phase_1"]["fp_rate"]
                assert compact1["phases"]["phase_1"]["fn_rate"] == compact2["phases"]["phase_1"]["fn_rate"]

    def test_different_seeds_produce_different_output(self):
        """Different seeds produce different outputs."""
        harness1 = CalibrationHarness(seed=42)
        harness1.run_phase_1(cycles=100)

        harness2 = CalibrationHarness(seed=999)
        harness2.run_phase_1(cycles=100)

        entries1 = harness1.get_all_entries()
        entries2 = harness2.get_all_entries()

        # At least some drift values should differ
        diffs = sum(
            1 for e1, e2 in zip(entries1, entries2)
            if e1.drift_value != e2.drift_value
        )
        assert diffs > 0


# =============================================================================
# Evidence Pack Hook Tests
# =============================================================================

class TestEvidencePackHook:
    """Tests for evidence pack calibration attachment."""

    def test_attach_calibration_to_evidence(self):
        """Calibration summary attaches under correct path."""
        evidence = {
            "proof_hash": "abc123",
            "governance": {"aligned": True},
        }

        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test123",
            "overall_pass": True,
            "enablement_recommendation": "PROCEED_TO_STAGE_2",
            "phases": {
                "phase_1": {"cycles": 500, "fp_rate": 0.01, "fn_rate": 0.005, "meets_criteria": True},
                "phase_2": {"cycles": 1000, "fp_rate": 0.03, "fn_rate": 0.01, "meets_criteria": True},
            },
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        # Original unchanged
        assert "budget_risk" not in evidence.get("governance", {})

        # New evidence has calibration
        assert "budget_risk" in new_evidence["governance"]
        assert "calibration" in new_evidence["governance"]["budget_risk"]

        cal = new_evidence["governance"]["budget_risk"]["calibration"]
        assert cal["schema_version"] == "1.0.0"
        assert cal["experiment_id"] == "test123"
        assert cal["overall_pass"] is True
        assert cal["fp_rate_p1"] == 0.01
        assert cal["fn_rate_p1"] == 0.005
        assert cal["fp_rate_p2"] == 0.03
        assert cal["fn_rate_p2"] == 0.01

    def test_attach_calibration_preserves_existing_budget_risk(self):
        """Calibration attachment preserves existing budget_risk fields."""
        evidence = {
            "proof_hash": "abc123",
            "governance": {
                "aligned": True,
                "budget_risk": {
                    "drift_class": "STABLE",
                    "noise_multiplier": 1.0,
                },
            },
        }

        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test123",
            "overall_pass": True,
            "enablement_recommendation": "PROCEED_TO_STAGE_2",
            "phases": {},
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        # Existing fields preserved
        assert new_evidence["governance"]["budget_risk"]["drift_class"] == "STABLE"
        assert new_evidence["governance"]["budget_risk"]["noise_multiplier"] == 1.0

        # Calibration added
        assert "calibration" in new_evidence["governance"]["budget_risk"]

    def test_attach_calibration_creates_governance_if_missing(self):
        """Calibration attachment creates governance section if missing."""
        evidence = {"proof_hash": "abc123"}

        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test123",
            "overall_pass": False,
            "enablement_recommendation": "NOT_RECOMMENDED",
            "phases": {},
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        assert "governance" in new_evidence
        assert "budget_risk" in new_evidence["governance"]
        assert "calibration" in new_evidence["governance"]["budget_risk"]

    def test_attach_calibration_non_mutating(self):
        """Original evidence dict is not mutated."""
        evidence = {
            "proof_hash": "abc123",
            "governance": {"aligned": True},
        }
        original_str = json.dumps(evidence, sort_keys=True)

        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test123",
            "overall_pass": True,
            "enablement_recommendation": "PROCEED_TO_STAGE_2",
            "phases": {"phase_1": {"cycles": 500, "fp_rate": 0.01, "fn_rate": 0.005, "meets_criteria": True}},
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        # Original unchanged
        assert json.dumps(evidence, sort_keys=True) == original_str

        # New evidence is different
        assert new_evidence != evidence


class TestCalibrationFileLoading:
    """Tests for loading calibration from file."""

    def test_load_calibration_from_valid_file(self):
        """Valid summary file loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "budget_calibration_summary.json"

            summary_data = {
                "schema_version": "1.0.0",
                "compact_summary": {
                    "schema_version": "1.0.0",
                    "experiment_id": "abc123",
                    "overall_pass": True,
                    "enablement_recommendation": "PROCEED_TO_STAGE_2",
                    "phases": {
                        "phase_1": {"cycles": 500, "fp_rate": 0.02, "fn_rate": 0.01, "meets_criteria": True},
                    },
                },
            }

            with open(summary_path, 'w') as f:
                json.dump(summary_data, f)

            loaded = load_calibration_summary_from_file(str(summary_path))

            assert loaded is not None
            assert loaded["experiment_id"] == "abc123"
            assert loaded["overall_pass"] is True

    def test_load_calibration_from_missing_file(self):
        """Missing file returns None."""
        loaded = load_calibration_summary_from_file("/nonexistent/path/summary.json")
        assert loaded is None

    def test_load_calibration_from_invalid_json(self):
        """Invalid JSON returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.json"
            with open(invalid_path, 'w') as f:
                f.write("not valid json {{{")

            loaded = load_calibration_summary_from_file(str(invalid_path))
            assert loaded is None

    def test_maybe_attach_when_file_exists(self):
        """maybe_attach attaches calibration when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "budget_calibration_summary.json"

            summary_data = {
                "compact_summary": {
                    "schema_version": "1.0.0",
                    "experiment_id": "test456",
                    "overall_pass": True,
                    "enablement_recommendation": "PROCEED_TO_STAGE_2",
                    "phases": {},
                },
            }

            with open(summary_path, 'w') as f:
                json.dump(summary_data, f)

            evidence = {"proof_hash": "xyz"}
            new_evidence = maybe_attach_calibration_to_evidence(evidence, tmpdir)

            assert "calibration" in new_evidence["governance"]["budget_risk"]
            assert new_evidence["governance"]["budget_risk"]["calibration"]["experiment_id"] == "test456"

    def test_maybe_attach_when_file_missing(self):
        """maybe_attach returns original evidence when file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence = {"proof_hash": "xyz"}
            result = maybe_attach_calibration_to_evidence(evidence, tmpdir)

            # Should be the same dict (or at least equal)
            assert result == evidence


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_phase_1_generates_valid_outputs(self):
        """Phase 1 run generates valid JSONL and summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            harness = CalibrationHarness(seed=42)
            result = run_phase_1_only(harness, cycles=100, output_dir=tmpdir, verbose=False)

            # Write outputs
            log_path = tmpdir / "budget_calibration_log.jsonl"
            summary_path = tmpdir / "budget_calibration_summary.json"

            write_jsonl(harness.get_all_entries(), log_path)
            write_summary(result, summary_path, {"seed": 42})

            # Verify log file
            assert log_path.exists()
            with open(log_path, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 100

            # Verify each line is valid
            for line in lines:
                data = json.loads(line)
                log = data["calibration_log"]
                assert log["phase"] == "PHASE_1_BASELINE"
                assert "budget_metrics" in log
                assert "classification" in log
                assert "derived" in log

            # Verify summary file
            assert summary_path.exists()
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            assert summary["shadow_mode"] is True
            assert "compact_summary" in summary
            assert summary["compact_summary"]["phases"]["phase_1"]["cycles"] == 100

    def test_full_pipeline_evidence_attachment(self):
        """Full pipeline: generate outputs, load, attach to evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Run calibration and write outputs
            harness = CalibrationHarness(seed=42)
            result = run_phase_1_only(harness, cycles=50, output_dir=tmpdir, verbose=False)

            log_path = tmpdir / "budget_calibration_log.jsonl"
            summary_path = tmpdir / "budget_calibration_summary.json"

            write_jsonl(harness.get_all_entries(), log_path)
            write_summary(result, summary_path, {"seed": 42})

            # Step 2: Load and attach to evidence
            evidence = {
                "proof_hash": "test_proof_123",
                "statement": "P → P",
                "governance": {"aligned": True},
            }

            new_evidence = maybe_attach_calibration_to_evidence(evidence, str(tmpdir))

            # Step 3: Verify attachment
            assert "calibration" in new_evidence["governance"]["budget_risk"]
            cal = new_evidence["governance"]["budget_risk"]["calibration"]

            assert cal["schema_version"] == "1.0.0"
            assert "experiment_id" in cal
            assert "overall_pass" in cal
            assert "enablement_recommendation" in cal
            assert "fp_rate_p1" in cal
            assert "fn_rate_p1" in cal

    def test_deterministic_full_pipeline(self):
        """Full pipeline is deterministic with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir1 = Path(tmpdir1)
            tmpdir2 = Path(tmpdir2)

            # Run 1
            harness1 = CalibrationHarness(seed=42)
            result1 = run_phase_1_only(harness1, cycles=50, output_dir=tmpdir1, verbose=False)
            write_jsonl(harness1.get_all_entries(), tmpdir1 / "log.jsonl")
            write_summary(result1, tmpdir1 / "summary.json", {"seed": 42})

            # Run 2
            harness2 = CalibrationHarness(seed=42)
            result2 = run_phase_1_only(harness2, cycles=50, output_dir=tmpdir2, verbose=False)
            write_jsonl(harness2.get_all_entries(), tmpdir2 / "log.jsonl")
            write_summary(result2, tmpdir2 / "summary.json", {"seed": 42})

            # Compare JSONL (excluding timestamps)
            with open(tmpdir1 / "log.jsonl", 'r') as f1, open(tmpdir2 / "log.jsonl", 'r') as f2:
                for l1, l2 in zip(f1, f2):
                    d1 = json.loads(l1)
                    d2 = json.loads(l2)
                    d1["calibration_log"]["timestamp"] = "X"
                    d2["calibration_log"]["timestamp"] = "X"
                    assert d1 == d2

            # Compare summaries (excluding timestamps and experiment_id which includes timestamp)
            with open(tmpdir1 / "summary.json", 'r') as f1, open(tmpdir2 / "summary.json", 'r') as f2:
                s1 = json.load(f1)
                s2 = json.load(f2)

            # Compare core metrics (experiment_id differs due to timestamp-based generation)
            c1 = s1["compact_summary"]
            c2 = s2["compact_summary"]

            assert c1["schema_version"] == c2["schema_version"]
            assert c1["overall_pass"] == c2["overall_pass"]
            assert c1["enablement_recommendation"] == c2["enablement_recommendation"]
            assert c1["phases"] == c2["phases"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_phases_dict(self):
        """Handle empty phases dict in calibration summary."""
        evidence = {"proof_hash": "test"}
        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test",
            "overall_pass": False,
            "enablement_recommendation": "NOT_RECOMMENDED",
            "phases": {},
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        cal = new_evidence["governance"]["budget_risk"]["calibration"]
        assert "fp_rate_p1" not in cal
        assert "fp_rate_p2" not in cal
        assert "fp_rate_p3" not in cal

    def test_partial_phases(self):
        """Handle partial phases (only phase_1)."""
        evidence = {"proof_hash": "test"}
        calibration = {
            "schema_version": "1.0.0",
            "experiment_id": "test",
            "overall_pass": True,
            "enablement_recommendation": "PHASE_1_ONLY",
            "phases": {
                "phase_1": {"cycles": 100, "fp_rate": 0.02, "fn_rate": 0.01, "meets_criteria": True},
            },
        }

        new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)

        cal = new_evidence["governance"]["budget_risk"]["calibration"]
        assert cal["fp_rate_p1"] == 0.02
        assert "fp_rate_p2" not in cal
        assert "fp_rate_p3" not in cal

    def test_zero_cycles(self):
        """Handle zero cycle calibration."""
        harness = CalibrationHarness(seed=42)
        harness.run_phase_1(cycles=0)

        entries = harness.get_all_entries()
        assert len(entries) == 0

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "empty.jsonl"
            count = write_jsonl(entries, log_path)
            assert count == 0

            # File should exist but be empty
            assert log_path.exists()
            with open(log_path, 'r') as f:
                content = f.read()
            assert content == ""
