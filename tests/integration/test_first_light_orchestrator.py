"""
First Light Orchestrator Integration Tests
===========================================

Tests for the First Light orchestrator including:
1. Basic run execution
2. Evidence package generation
3. Deterministic harness (same seed = identical trajectories)
4. Verification mode
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.first_light_orchestrator import (
    FirstLightConfig,
    FirstLightRunner,
    build_first_light_evidence_package,
    verify_evidence_package,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestFirstLightOrchestrator:
    """Test suite for First Light orchestrator."""
    
    def test_basic_run_execution(self, temp_output_dir):
        """Test basic orchestrator run."""
        config = FirstLightConfig(
            seed=42,
            cycles=10,
            slice_name="arithmetic_simple",
            mode="baseline",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        # Verify result structure
        assert result.run_id.startswith("fl_baseline_42_")
        assert result.config.seed == 42
        assert result.config.cycles == 10
        assert len(result.delta_p_trajectory) == 10
        assert len(result.hss_trajectory) == 10
        assert len(result.governance_envelopes) == 10
        
        # Verify trajectories are non-empty
        assert result.delta_p_trajectory[0] is not None
        assert result.hss_trajectory[0] >= 0.0
        
        # Verify final statistics
        assert result.total_proofs_verified > 0
        assert result.total_candidates_processed > 0
        
        # Verify stability report
        assert "hss_mean" in result.stability_report
        assert "policy_stable" in result.stability_report
        
        # Verify artifacts were written
        run_dir = runner.run_dir
        assert (run_dir / "result.json").exists()
        assert (run_dir / "trajectories.json").exists()
        assert (run_dir / "governance.json").exists()
        assert (run_dir / "cycles.jsonl").exists()
    
    def test_integrated_mode_policy_updates(self, temp_output_dir):
        """Test that integrated mode applies policy updates."""
        config = FirstLightConfig(
            seed=42,
            cycles=20,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        # In integrated mode, policy weights should change
        initial_weights = result.delta_p_trajectory[0]
        final_weights = result.delta_p_trajectory[-1]
        
        # At least one weight should have changed
        weights_changed = (
            initial_weights["len"] != final_weights["len"] or
            initial_weights["depth"] != final_weights["depth"] or
            initial_weights["success"] != final_weights["success"]
        )
        assert weights_changed, "Policy weights should change in integrated mode"
        
        # Success weight should never go negative
        for weights in result.delta_p_trajectory:
            assert weights["success"] >= 0.0, "Success weight should be non-negative"
    
    def test_baseline_mode_no_policy_updates(self, temp_output_dir):
        """Test that baseline mode keeps policy weights at zero."""
        config = FirstLightConfig(
            seed=42,
            cycles=20,
            slice_name="arithmetic_simple",
            mode="baseline",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        # In baseline mode, policy weights should stay at zero
        for weights in result.delta_p_trajectory:
            assert weights["len"] == 0.0
            assert weights["depth"] == 0.0
            assert weights["success"] == 0.0
    
    def test_evidence_package_generation(self, temp_output_dir):
        """Test evidence package building."""
        config = FirstLightConfig(
            seed=42,
            cycles=10,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        # Build evidence package
        evidence = build_first_light_evidence_package(runner.run_dir)
        
        # Verify structure
        assert evidence["version"] == "1.0.0"
        assert "run_metadata" in evidence
        assert "stability_report" in evidence
        assert "trajectories" in evidence
        assert "governance" in evidence
        assert "summary" in evidence
        
        # Verify trajectories
        assert len(evidence["trajectories"]["delta_p"]) == 10
        assert len(evidence["trajectories"]["hss"]) == 10
        
        # Verify governance
        gov = evidence["governance"]
        assert len(gov["curriculum_stability"]) == 10
        assert "safety_summary" in gov
        assert "cortex_summary" in gov
        assert "tda_metrics" in gov
        assert len(gov["epistemic_tile"]) == 10
        assert len(gov["harmonic_tile"]) == 10
        assert len(gov["drift_tile"]) == 10
        assert len(gov["semantic_tile"]) == 10
        
        # Verify summary
        summary = evidence["summary"]
        assert summary["total_proofs_verified"] > 0
        assert summary["total_candidates_processed"] > 0
        assert "final_policy_weights" in summary
        assert "convergence_achieved" in summary
    
    def test_deterministic_harness(self, temp_output_dir):
        """Test that same seed produces identical trajectories."""
        seed = 123
        cycles = 50
        
        # Run 1
        config1 = FirstLightConfig(
            seed=seed,
            cycles=cycles,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir / "run1",
        )
        runner1 = FirstLightRunner(config1)
        result1 = runner1.run()
        
        # Run 2 (same seed)
        config2 = FirstLightConfig(
            seed=seed,
            cycles=cycles,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir / "run2",
        )
        runner2 = FirstLightRunner(config2)
        result2 = runner2.run()
        
        # Verify Δp trajectory identical
        assert len(result1.delta_p_trajectory) == len(result2.delta_p_trajectory)
        for i, (dp1, dp2) in enumerate(zip(result1.delta_p_trajectory, result2.delta_p_trajectory)):
            assert dp1["len"] == dp2["len"], f"Cycle {i}: len weight mismatch"
            assert dp1["depth"] == dp2["depth"], f"Cycle {i}: depth weight mismatch"
            assert dp1["success"] == dp2["success"], f"Cycle {i}: success weight mismatch"
        
        # Verify HSS trajectory identical
        assert len(result1.hss_trajectory) == len(result2.hss_trajectory)
        for i, (hss1, hss2) in enumerate(zip(result1.hss_trajectory, result2.hss_trajectory)):
            assert hss1 == hss2, f"Cycle {i}: HSS mismatch"
        
        # Verify governance tiles identical (after sorting where necessary)
        assert len(result1.governance_envelopes) == len(result2.governance_envelopes)
        for i, (env1, env2) in enumerate(zip(result1.governance_envelopes, result2.governance_envelopes)):
            # Curriculum stability should be identical
            assert env1.curriculum_stability == env2.curriculum_stability, f"Cycle {i}: curriculum mismatch"
            # Safety metrics should be identical
            assert env1.safety_metrics == env2.safety_metrics, f"Cycle {i}: safety mismatch"
            # TDA metrics should be identical
            assert env1.tda_metrics == env2.tda_metrics, f"Cycle {i}: TDA mismatch"
            # Epistemic tile should be identical
            assert env1.epistemic_tile == env2.epistemic_tile, f"Cycle {i}: epistemic mismatch"
    
    def test_verify_evidence_package_valid(self, temp_output_dir):
        """Test evidence package verification with valid package."""
        config = FirstLightConfig(
            seed=42,
            cycles=10,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        runner.run()
        
        # Verify evidence package
        valid, message = verify_evidence_package(runner.run_dir)
        
        assert valid, f"Evidence package should be valid: {message}"
        assert "valid" in message.lower()
    
    def test_verify_evidence_package_missing_file(self, temp_output_dir):
        """Test evidence package verification with missing file."""
        config = FirstLightConfig(
            seed=42,
            cycles=10,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        runner.run()
        
        # Remove a required file
        (runner.run_dir / "trajectories.json").unlink()
        
        # Verify should fail
        valid, message = verify_evidence_package(runner.run_dir)
        
        assert not valid
        assert "not found" in message.lower() or "missing" in message.lower()
    
    def test_verify_evidence_package_invalid_structure(self, temp_output_dir):
        """Test evidence package verification with invalid structure."""
        config = FirstLightConfig(
            seed=42,
            cycles=10,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        runner.run()
        
        # Corrupt the result file
        result_path = runner.run_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump({"invalid": "structure"}, f)
        
        # Verify should fail
        valid, message = verify_evidence_package(runner.run_dir)
        
        assert not valid
        assert "missing" in message.lower() or "mismatch" in message.lower()
    
    def test_governance_envelopes_content(self, temp_output_dir):
        """Test that governance envelopes contain expected content."""
        config = FirstLightConfig(
            seed=42,
            cycles=5,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
            enable_safety_gate=True,
            enable_curriculum_gate=True,
            enable_tda_gate=True,
            enable_telemetry=True,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        # Check each envelope has expected content
        for i, envelope in enumerate(result.governance_envelopes):
            # Curriculum
            assert "active_slice" in envelope.curriculum_stability
            assert envelope.curriculum_stability["active_slice"] == "arithmetic_simple"
            
            # Safety
            assert "abstention_rate" in envelope.safety_metrics
            assert "safety_threshold_met" in envelope.safety_metrics
            assert envelope.safety_metrics["abstention_rate"] >= 0.0
            
            # TDA
            assert "betti_numbers" in envelope.tda_metrics
            assert len(envelope.tda_metrics["betti_numbers"]) == 3
            
            # Telemetry
            assert "throughput_proofs_per_hour" in envelope.telemetry_metrics
            assert envelope.telemetry_metrics["throughput_proofs_per_hour"] >= 0
            
            # Governance tiles
            assert "uncertainty_mass" in envelope.epistemic_tile
            assert "oscillation_amplitude" in envelope.harmonic_tile
            assert "concept_drift_score" in envelope.drift_tile
            assert "vocabulary_coverage" in envelope.semantic_tile
    
    def test_stability_report_metrics(self, temp_output_dir):
        """Test that stability report contains correct metrics."""
        config = FirstLightConfig(
            seed=42,
            cycles=100,
            slice_name="arithmetic_simple",
            mode="integrated",
            output_dir=temp_output_dir,
        )
        
        runner = FirstLightRunner(config)
        result = runner.run()
        
        report = result.stability_report
        
        # Check all required metrics
        assert "hss_mean" in report
        assert "hss_std" in report
        assert "hss_cv" in report
        assert "hss_stable" in report
        assert "policy_len_std" in report
        assert "policy_depth_std" in report
        assert "policy_success_std" in report
        assert "policy_stable" in report
        assert "num_cycles" in report
        assert "convergence_achieved" in report
        
        # Check values are reasonable
        assert 0.0 <= report["hss_mean"] <= 1.0
        assert report["hss_std"] >= 0.0
        assert report["hss_cv"] >= 0.0
        assert report["num_cycles"] == 100
        assert isinstance(report["hss_stable"], bool)
        assert isinstance(report["policy_stable"], bool)
        assert isinstance(report["convergence_achieved"], bool)
