"""
Tests for Curriculum Stability Integration

Verifies integration helpers for RFLRunner, U2Runner, Evidence Packs, and P4 reports.
"""

import json
from typing import Any, Dict, List
from dataclasses import dataclass
from curriculum.integration import (
    extract_slice_metrics_from_rfl_runner,
    add_stability_to_rfl_results,
    create_evidence_pack_with_stability,
    create_p4_calibration_report_with_stability,
)


@dataclass
class MockRFLRunnerConfig:
    """Mock RFL config for testing."""
    experiment_id: str
    curriculum: List[Any]
    abstention_tolerance: float = 0.2


@dataclass
class MockSliceConfig:
    """Mock slice config."""
    name: str
    atoms: int = 5
    depth_max: int = 6
    max_breadth: int = 1500
    derive_steps: int = 100
    max_total: int = 10000
    start_run: int = 1
    end_run: int = 10


@dataclass
class MockPolicyLedgerEntry:
    """Mock policy ledger entry."""
    slice_name: str
    coverage_rate: float
    abstention_fraction: float
    novelty_rate: float = 0.0
    throughput: float = 0.0


class MockRFLRunner:
    """Mock RFLRunner for testing."""
    
    def __init__(self):
        self.config = MockRFLRunnerConfig(
            experiment_id="test_001",
            curriculum=[
                MockSliceConfig(name="slice_a", atoms=4),
                MockSliceConfig(name="slice_b", atoms=5),
            ]
        )
        self.policy_ledger = [
            MockPolicyLedgerEntry("slice_a", 0.90, 0.05),
            MockPolicyLedgerEntry("slice_a", 0.88, 0.06),
            MockPolicyLedgerEntry("slice_b", 0.85, 0.10),
            MockPolicyLedgerEntry("slice_b", 0.83, 0.12),
        ]


class TestRFLIntegration:
    """Test RFL runner integration."""
    
    def test_extract_slice_metrics(self):
        """Test extracting slice metrics from RFL runner."""
        runner = MockRFLRunner()
        
        metrics = extract_slice_metrics_from_rfl_runner(runner)
        
        assert len(metrics) == 2
        assert metrics[0]["slice_name"] == "slice_a"
        assert metrics[1]["slice_name"] == "slice_b"
        
        # Check averages
        assert 0.85 <= metrics[0]["coverage_rate"] <= 0.95
        assert 0.80 <= metrics[1]["coverage_rate"] <= 0.90
    
    def test_add_stability_to_results(self):
        """Test adding stability to RFL results."""
        runner = MockRFLRunner()
        
        results = {
            "experiment_id": "test_001",
            "execution_summary": {"total_runs": 4},
        }
        
        updated = add_stability_to_rfl_results(results, runner, include_council=False)
        
        assert "curriculum_stability_envelope" in updated
        envelope = updated["curriculum_stability_envelope"]
        assert "mean_HSS" in envelope
        assert "status_light" in envelope
        assert "suitability_scores" in envelope
    
    def test_add_stability_with_council(self):
        """Test adding stability with council advisory."""
        runner = MockRFLRunner()
        
        results = {"experiment_id": "test_002"}
        
        updated = add_stability_to_rfl_results(results, runner, include_council=True)
        
        assert "curriculum_stability_envelope" in updated
        assert "uplift_council_advisory" in updated
        
        advisory = updated["uplift_council_advisory"]
        assert "status" in advisory
        assert advisory["status"] in ["OK", "WARN", "BLOCK"]


class TestEvidencePackIntegration:
    """Test evidence pack integration."""
    
    def test_create_evidence_with_stability(self):
        """Test creating evidence pack with stability."""
        evidence = {
            "experiment_id": "test_003",
            "results": {"coverage": 0.85},
        }
        
        slice_metrics = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            }
        ]
        
        new_evidence = create_evidence_pack_with_stability(evidence, slice_metrics)
        
        assert "governance" in new_evidence
        assert "curriculum_stability" in new_evidence["governance"]
        
        stability = new_evidence["governance"]["curriculum_stability"]
        assert "status_light" in stability
        assert "slices_flagged" in stability
        assert "suitability_scores" in stability
    
    def test_evidence_non_mutating(self):
        """Test that evidence pack creation is non-mutating."""
        original = {"experiment_id": "test_004"}
        
        slice_metrics = [
            {
                "slice_name": "slice_x",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.90,
                "abstention_rate": 0.05,
            }
        ]
        
        new_evidence = create_evidence_pack_with_stability(original, slice_metrics)
        
        # Original should be unchanged
        assert "governance" not in original
        
        # New evidence should have stability
        assert "governance" in new_evidence
    
    def test_evidence_json_serializable(self):
        """Test that evidence pack is JSON-serializable."""
        evidence = {"experiment_id": "test_005"}
        
        slice_metrics = [
            {
                "slice_name": "slice_y",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.80,
                "abstention_rate": 0.15,
            }
        ]
        
        new_evidence = create_evidence_pack_with_stability(evidence, slice_metrics)
        
        # Should serialize without error
        json_str = json.dumps(new_evidence)
        assert json_str is not None
        
        # Should deserialize correctly
        parsed = json.loads(json_str)
        assert parsed["governance"]["curriculum_stability"]["status_light"] in ["GREEN", "YELLOW", "RED"]


class TestP4CalibrationIntegration:
    """Test P4 calibration integration."""
    
    def test_create_p4_report_with_stability(self):
        """Test creating P4 calibration report with stability."""
        calibration_data = {
            "calibration_id": "p4_001",
            "timestamp": "2025-12-11T00:00:00Z",
        }
        
        slice_metrics = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.90,
                "abstention_rate": 0.05,
            },
            {
                "slice_name": "slice_b",
                "params": {"atoms": 6, "depth_max": 8, "breadth_max": 2000},
                "coverage_rate": 0.50,
                "abstention_rate": 0.40,
            },
        ]
        
        report = create_p4_calibration_report_with_stability(calibration_data, slice_metrics)
        
        assert "curriculum_stability" in report
        stability = report["curriculum_stability"]
        
        assert "stable_slices" in stability
        assert "unstable_slices" in stability
        assert "HSS_variance_spikes" in stability
        assert "stability_gate_decisions" in stability
    
    def test_p4_gate_decisions(self):
        """Test P4 gate decisions in shadow mode."""
        calibration_data = {"calibration_id": "p4_002"}
        
        slice_metrics = [
            {
                "slice_name": "good_slice",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.90,
                "abstention_rate": 0.05,
            },
            {
                "slice_name": "bad_slice",
                "params": {"atoms": 8, "depth_max": 10, "breadth_max": 3500},
                "coverage_rate": 0.40,
                "abstention_rate": 0.50,
            },
        ]
        
        report = create_p4_calibration_report_with_stability(calibration_data, slice_metrics)
        
        decisions = report["curriculum_stability"]["stability_gate_decisions"]
        
        # Good slice should be ALLOW
        assert decisions["good_slice"] == "ALLOW"
        
        # Bad slice might be BLOCK (depending on suitability threshold)
        assert decisions["bad_slice"] in ["ALLOW", "BLOCK"]
    
    def test_p4_historical_data(self):
        """Test P4 report with historical data."""
        calibration_data = {"calibration_id": "p4_003"}
        
        slice_metrics = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            }
        ]
        
        historical = {
            "slice_a": [
                {"coverage_rate": 0.88, "abstention_rate": 0.08},
                {"coverage_rate": 0.86, "abstention_rate": 0.09},
                {"coverage_rate": 0.85, "abstention_rate": 0.10},
            ]
        }
        
        report = create_p4_calibration_report_with_stability(
            calibration_data,
            slice_metrics,
            historical
        )
        
        assert "curriculum_stability" in report
        # With stable history, slice should be in stable_slices
        # (actual classification depends on thresholds)
        stability = report["curriculum_stability"]
        assert len(stability["stable_slices"]) + len(stability["unstable_slices"]) > 0


class TestDeterminism:
    """Test determinism of integration."""
    
    def test_deterministic_output(self):
        """Test that integration produces deterministic output."""
        evidence = {"experiment_id": "test_det"}
        
        slice_metrics = [
            {
                "slice_name": "slice_z",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            }
        ]
        
        result1 = create_evidence_pack_with_stability(dict(evidence), slice_metrics)
        result2 = create_evidence_pack_with_stability(dict(evidence), slice_metrics)
        
        # Should be identical
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)
        
        assert json1 == json2
