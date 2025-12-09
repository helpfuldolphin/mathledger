#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Synthetic Uplift Scenarios
------------------------------------------

This test module validates:
1. Synthetic log generation produces valid U2-compatible JSONL
2. Uplift direction matches expected sign (positive, zero, negative)
3. Analysis pipeline can ingest and process synthetic logs
4. Deterministic reproducibility (same seed → same output)

NOT derived from real derivations; NOT part of Evidence Pack.

Usage:
    pytest experiments/synthetic_uplift/tests/test_synthetic_uplift_scenarios.py -v
    
    # Run with integration marker if available
    pytest -m synthetic experiments/synthetic_uplift/tests/

==============================================================================
"""

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import pytest
import yaml

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the generator module
from experiments.synthetic_uplift.generate_synthetic_logs import (
    SAFETY_LABEL,
    SyntheticOutcomeGenerator,
    compute_sha256,
    generate_seed_schedule,
    generate_synthetic_logs,
    load_synthetic_config,
    get_slice_config,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_config():
    """Load the synthetic slices configuration."""
    return load_synthetic_config()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="synthetic_test_") as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# UNIT TESTS - Configuration Loading
# ==============================================================================

class TestConfigLoading:
    """Tests for configuration loading and validation."""
    
    def test_load_synthetic_config(self, synthetic_config):
        """Config file should load successfully with required fields."""
        assert synthetic_config is not None
        assert "version" in synthetic_config
        assert "slices" in synthetic_config
        assert synthetic_config.get("label") == SAFETY_LABEL
    
    def test_all_expected_slices_present(self, synthetic_config):
        """All three expected synthetic slices should be defined."""
        slices = synthetic_config.get("slices", {})
        expected_slices = ["synthetic_easy", "synthetic_shifted", "synthetic_regression"]
        
        for slice_name in expected_slices:
            assert slice_name in slices, f"Missing expected slice: {slice_name}"
    
    def test_slice_has_required_fields(self, synthetic_config):
        """Each slice should have required fields."""
        required_fields = ["description", "probabilities", "items", "prereg_hash"]
        
        for slice_name, slice_config in synthetic_config.get("slices", {}).items():
            for field in required_fields:
                assert field in slice_config, f"Slice {slice_name} missing field: {field}"
    
    def test_probabilities_structure(self, synthetic_config):
        """Probabilities should have baseline and rfl modes with valid ranges."""
        for slice_name, slice_config in synthetic_config.get("slices", {}).items():
            probs = slice_config.get("probabilities", {})
            
            assert "baseline" in probs, f"{slice_name}: missing baseline probabilities"
            assert "rfl" in probs, f"{slice_name}: missing rfl probabilities"
            
            for mode in ["baseline", "rfl"]:
                for class_name, prob in probs[mode].items():
                    assert 0.0 <= prob <= 1.0, (
                        f"{slice_name}/{mode}/{class_name}: probability {prob} out of range"
                    )


# ==============================================================================
# UNIT TESTS - Deterministic Generation
# ==============================================================================

class TestDeterministicGeneration:
    """Tests for deterministic reproducibility."""
    
    def test_seed_schedule_deterministic(self):
        """Seed schedule should be deterministic."""
        schedule1 = generate_seed_schedule(42, 100)
        schedule2 = generate_seed_schedule(42, 100)
        
        assert schedule1 == schedule2
        assert len(schedule1) == 100
    
    def test_seed_schedule_different_seeds(self):
        """Different seeds should produce different schedules."""
        schedule1 = generate_seed_schedule(42, 100)
        schedule2 = generate_seed_schedule(123, 100)
        
        assert schedule1 != schedule2
    
    def test_outcome_generator_deterministic(self, synthetic_config):
        """Outcome generator should be deterministic."""
        slice_config = get_slice_config(synthetic_config, "synthetic_easy")
        
        gen1 = SyntheticOutcomeGenerator(slice_config, "baseline", 42)
        gen2 = SyntheticOutcomeGenerator(slice_config, "baseline", 42)
        
        items = gen1.get_item_ids()
        
        for item in items:
            outcome1 = gen1.generate_outcome(item, cycle_seed=12345)
            outcome2 = gen2.generate_outcome(item, cycle_seed=12345)
            
            assert outcome1["success"] == outcome2["success"]
            assert outcome1["outcome"] == outcome2["outcome"]
    
    def test_full_generation_deterministic(self, synthetic_config, temp_output_dir):
        """Full log generation should be deterministic."""
        # Generate logs twice with same seed
        path1 = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=50,
            out_dir=temp_output_dir / "run1",
            seed=42,
            config=synthetic_config,
        )
        
        path2 = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=50,
            out_dir=temp_output_dir / "run2",
            seed=42,
            config=synthetic_config,
        )
        
        # Read and compare
        with open(path1) as f:
            lines1 = f.readlines()
        with open(path2) as f:
            lines2 = f.readlines()
        
        assert len(lines1) == len(lines2)
        
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            r1 = json.loads(l1)
            r2 = json.loads(l2)
            assert r1["success"] == r2["success"], f"Cycle {i} mismatch"
            assert r1["item"] == r2["item"], f"Cycle {i} item mismatch"


# ==============================================================================
# UNIT TESTS - Log Format Validation
# ==============================================================================

class TestLogFormatValidation:
    """Tests for U2-compatible log format."""
    
    def test_log_has_required_fields(self, synthetic_config, temp_output_dir):
        """Generated logs should have all required U2 schema fields."""
        required_fields = [
            "cycle", "slice", "mode", "seed", "item",
            "result", "success", "label", "synthetic"
        ]
        
        path = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=10,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                for field in required_fields:
                    assert field in record, f"Missing field: {field}"
    
    def test_log_safety_label(self, synthetic_config, temp_output_dir):
        """All log records should have safety label."""
        path = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=10,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                assert record.get("label") == SAFETY_LABEL
                assert record.get("synthetic") is True
    
    def test_manifest_generated(self, synthetic_config, temp_output_dir):
        """Manifest file should be generated with metadata."""
        generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=10,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        manifest_path = temp_output_dir / "synthetic_synthetic_easy_baseline_manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest.get("label") == SAFETY_LABEL
        assert manifest.get("synthetic") is True
        assert "statistics" in manifest
        assert "telemetry_hash" in manifest


# ==============================================================================
# SCENARIO TESTS - Uplift Direction Validation
# ==============================================================================

class TestUpliftDirectionScenarios:
    """
    Tests that validate synthetic slices produce expected uplift directions.
    
    These are the core stress-tests for the synthetic noise infrastructure.
    """
    
    @pytest.fixture
    def generate_pair(self, synthetic_config, temp_output_dir):
        """Helper to generate baseline and RFL logs for a slice."""
        def _generate(slice_name: str, cycles: int = 200, seed: int = 42):
            baseline_path = generate_synthetic_logs(
                slice_name=slice_name,
                mode="baseline",
                cycles=cycles,
                out_dir=temp_output_dir / f"{slice_name}_baseline",
                seed=seed,
                config=synthetic_config,
            )
            
            rfl_path = generate_synthetic_logs(
                slice_name=slice_name,
                mode="rfl",
                cycles=cycles,
                out_dir=temp_output_dir / f"{slice_name}_rfl",
                seed=seed,
                config=synthetic_config,
            )
            
            return baseline_path, rfl_path
        
        return _generate
    
    @staticmethod
    def compute_success_rate(log_path: Path) -> float:
        """Compute success rate from JSONL log."""
        successes = 0
        total = 0
        
        with open(log_path) as f:
            for line in f:
                record = json.loads(line)
                total += 1
                if record.get("success"):
                    successes += 1
        
        return successes / total if total > 0 else 0.0
    
    def test_synthetic_easy_no_uplift(self, generate_pair, synthetic_config):
        """
        synthetic_easy: Both modes should have similar success rates.
        Expected: |uplift| < 8% (no meaningful difference)
        
        Note: RFL policy still adapts even with identical probabilities,
        which can give it a slight edge (policy learns to exploit successful items).
        We use 8% threshold to account for this expected variance.
        """
        baseline_path, rfl_path = generate_pair("synthetic_easy", cycles=500)
        
        baseline_rate = self.compute_success_rate(baseline_path)
        rfl_rate = self.compute_success_rate(rfl_path)
        uplift = rfl_rate - baseline_rate
        
        print(f"\nsynthetic_easy:")
        print(f"  Baseline success rate: {baseline_rate:.4f}")
        print(f"  RFL success rate:      {rfl_rate:.4f}")
        print(f"  Uplift:                {uplift:+.4f}")
        
        # Expect no significant uplift (within 8% - allows for policy adaptation variance)
        assert abs(uplift) < 0.08, (
            f"Expected no significant uplift, got {uplift:.4f}"
        )
    
    def test_synthetic_shifted_positive_uplift(self, generate_pair, synthetic_config):
        """
        synthetic_shifted: RFL should outperform baseline.
        Expected: uplift > 10% (positive direction)
        """
        baseline_path, rfl_path = generate_pair("synthetic_shifted", cycles=500)
        
        baseline_rate = self.compute_success_rate(baseline_path)
        rfl_rate = self.compute_success_rate(rfl_path)
        uplift = rfl_rate - baseline_rate
        
        print(f"\nsynthetic_shifted:")
        print(f"  Baseline success rate: {baseline_rate:.4f}")
        print(f"  RFL success rate:      {rfl_rate:.4f}")
        print(f"  Uplift:                {uplift:+.4f}")
        
        # Expect positive uplift (RFL better than baseline by at least 10%)
        assert uplift > 0.10, (
            f"Expected positive uplift > 10%, got {uplift:.4f}"
        )
    
    def test_synthetic_regression_negative_uplift(self, generate_pair, synthetic_config):
        """
        synthetic_regression: Baseline should outperform RFL.
        Expected: uplift < -8% (negative direction)
        
        Note: We use 8% threshold as variance can occur due to sampling.
        The key test is that the direction is consistently negative.
        """
        baseline_path, rfl_path = generate_pair("synthetic_regression", cycles=500)
        
        baseline_rate = self.compute_success_rate(baseline_path)
        rfl_rate = self.compute_success_rate(rfl_path)
        uplift = rfl_rate - baseline_rate
        
        print(f"\nsynthetic_regression:")
        print(f"  Baseline success rate: {baseline_rate:.4f}")
        print(f"  RFL success rate:      {rfl_rate:.4f}")
        print(f"  Uplift:                {uplift:+.4f}")
        
        # Expect negative uplift (baseline better than RFL by at least 8%)
        assert uplift < -0.08, (
            f"Expected negative uplift < -8%, got {uplift:.4f}"
        )


# ==============================================================================
# INTEGRATION TESTS - Analysis Pipeline Compatibility
# ==============================================================================

class TestAnalysisPipelineIntegration:
    """Tests for integration with existing analysis tools."""
    
    def test_logs_parseable_by_analysis(self, synthetic_config, temp_output_dir):
        """
        Synthetic logs should be parseable by the U1 analysis module.
        
        This tests that the log format is compatible with existing tooling.
        """
        # Generate logs
        baseline_path = generate_synthetic_logs(
            slice_name="synthetic_shifted",
            mode="baseline",
            cycles=100,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        rfl_path = generate_synthetic_logs(
            slice_name="synthetic_shifted",
            mode="rfl",
            cycles=100,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        # Try importing and using the analysis module
        try:
            from experiments.analyze_uplift_u1 import (
                load_jsonl,
                compute_abstention_rate,
            )
            
            # Load and analyze
            baseline_records = load_jsonl(baseline_path)
            rfl_records = load_jsonl(rfl_path)
            
            assert len(baseline_records) == 100
            assert len(rfl_records) == 100
            
            # Compute abstention rates (inverse of success)
            baseline_abstention, _ = compute_abstention_rate(baseline_records, burn_in=0)
            rfl_abstention, _ = compute_abstention_rate(rfl_records, burn_in=0)
            
            print(f"\nAnalysis module integration:")
            print(f"  Baseline abstention: {baseline_abstention:.4f}")
            print(f"  RFL abstention:      {rfl_abstention:.4f}")
            
            # Verify the analysis module can process the logs
            assert 0.0 <= baseline_abstention <= 1.0
            assert 0.0 <= rfl_abstention <= 1.0
            
        except ImportError as e:
            pytest.skip(f"Analysis module not available: {e}")
    
    def test_logs_have_analysis_compatible_fields(self, synthetic_config, temp_output_dir):
        """
        Logs should have fields compatible with multiple analysis modes.
        
        The analysis tools look for various field names; we provide aliases.
        """
        path = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=10,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                
                # Check for analysis-compatible fields
                assert "success" in record
                assert "proof_found" in record  # Alias
                assert "abstention" in record  # Alias
                assert "outcome" in record
                
                # Verify consistency
                assert record["proof_found"] == record["success"]
                assert record["abstention"] == (not record["success"])


# ==============================================================================
# SAFETY TESTS - Isolation and Labeling
# ==============================================================================

class TestSafetyIsolation:
    """Tests to ensure synthetic data cannot contaminate real evidence."""
    
    def test_slice_names_prefixed(self, synthetic_config):
        """All synthetic slice names must start with 'synthetic_'."""
        for slice_name in synthetic_config.get("slices", {}).keys():
            assert slice_name.startswith("synthetic_"), (
                f"Slice '{slice_name}' must start with 'synthetic_' prefix"
            )
    
    def test_output_files_labeled(self, synthetic_config, temp_output_dir):
        """Output filenames should contain 'synthetic' marker."""
        path = generate_synthetic_logs(
            slice_name="synthetic_easy",
            mode="baseline",
            cycles=10,
            out_dir=temp_output_dir,
            seed=42,
            config=synthetic_config,
        )
        
        assert "synthetic" in path.name.lower()
    
    def test_prereg_hashes_distinct(self, synthetic_config):
        """Synthetic prereg hashes should not match real slice hashes."""
        for slice_name, slice_config in synthetic_config.get("slices", {}).items():
            prereg_hash = slice_config.get("prereg_hash", "")
            
            # Verify hash contains 'synthetic' marker
            assert "synthetic" in prereg_hash.lower(), (
                f"Slice {slice_name}: prereg_hash should contain 'synthetic'"
            )
    
    def test_real_config_not_modified(self):
        """Verify that real curriculum config is not modified."""
        real_config_path = Path(project_root) / "config" / "curriculum_uplift_phase2.yaml"
        
        if not real_config_path.exists():
            pytest.skip("Real config not found (expected in fresh checkout)")
        
        with open(real_config_path) as f:
            real_config = yaml.safe_load(f)
        
        # Verify no synthetic slices in real config
        for slice_name in real_config.get("slices", {}):
            assert not slice_name.startswith("synthetic_"), (
                f"Real config should not contain synthetic slice: {slice_name}"
            )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

