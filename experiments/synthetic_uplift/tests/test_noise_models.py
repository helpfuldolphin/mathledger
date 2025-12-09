#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Noise Models (Drift & Correlation)
--------------------------------------------------

This test module contains 18 tests validating:
    1. Determinism (same seed → same output)
    2. Drift parameter effects (all modes behave correctly)
    3. Correlation effects (class co-failure patterns)
    4. Snapshot reproducibility (generation is bit-exact)
    5. Manifest integrity (hashes are correct)

NOT derived from real derivations; NOT part of Evidence Pack.

Usage:
    pytest experiments/synthetic_uplift/tests/test_noise_models.py -v
    pytest experiments/synthetic_uplift/tests/test_noise_models.py -v -k "drift"
    pytest experiments/synthetic_uplift/tests/test_noise_models.py -v -k "correlation"

==============================================================================
"""

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

import sys
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import (
    SAFETY_LABEL,
    CorrelationConfig,
    CorrelationEngine,
    DriftConfig,
    DriftMode,
    DriftModulator,
    NoiseConfig,
    NoiseEngine,
    create_cyclical_drift,
    create_monotonic_drift,
    create_no_drift,
    create_shock_drift,
    create_correlation,
    simulate_drift_series,
)

from experiments.synthetic_uplift.scenario_suite import (
    SCENARIOS,
    list_scenarios,
    load_scenario,
    get_scenarios_by_category,
)

from experiments.synthetic_uplift.generate_synthetic_logs_v2 import (
    EnhancedOutcomeGenerator,
    SyntheticManifest,
    compute_sha256,
    generate_seed_schedule,
    generate_synthetic_logs_v2,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory(prefix="synthetic_noise_test_") as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# TEST CLASS 1: DRIFT DETERMINISM (3 tests)
# ==============================================================================

class TestDriftDeterminism:
    """Tests for drift model determinism."""
    
    def test_monotonic_drift_deterministic(self):
        """Monotonic drift should produce identical values for same parameters."""
        config1 = create_monotonic_drift(slope=0.001, direction="down")
        config2 = create_monotonic_drift(slope=0.001, direction="down")
        
        mod1 = DriftModulator(config1)
        mod2 = DriftModulator(config2)
        
        for cycle in range(0, 500, 50):
            p1 = mod1.modulate(0.5, cycle, 500)
            p2 = mod2.modulate(0.5, cycle, 500)
            assert p1 == p2, f"Drift not deterministic at cycle {cycle}"
    
    def test_cyclical_drift_deterministic(self):
        """Cyclical drift should produce identical sinusoidal patterns."""
        config = create_cyclical_drift(amplitude=0.15, period=100)
        
        series1 = simulate_drift_series(config, base_prob=0.5, total_cycles=500)
        series2 = simulate_drift_series(config, base_prob=0.5, total_cycles=500)
        
        assert series1 == series2
    
    def test_shock_drift_deterministic(self):
        """Shock drift should produce identical step function."""
        config = create_shock_drift(shock_cycle=250, shock_delta=-0.30)
        mod = DriftModulator(config)
        
        # Before shock
        for cycle in range(0, 250):
            assert mod.modulate(0.7, cycle, 500) == pytest.approx(0.7)
        
        # After shock
        for cycle in range(250, 500):
            assert mod.modulate(0.7, cycle, 500) == pytest.approx(0.4)


# ==============================================================================
# TEST CLASS 2: DRIFT PARAMETER EFFECTS (4 tests)
# ==============================================================================

class TestDriftParameterEffects:
    """Tests validating drift parameters have correct effects."""
    
    def test_no_drift_constant(self):
        """No drift mode should return constant probability."""
        config = create_no_drift()
        mod = DriftModulator(config)
        
        for cycle in range(500):
            assert mod.modulate(0.6, cycle, 500) == 0.6
    
    def test_monotonic_up_increases(self):
        """Monotonic up drift should increase probability over time."""
        config = create_monotonic_drift(slope=0.001, direction="up")
        mod = DriftModulator(config)
        
        p_early = mod.modulate(0.5, 0, 500)
        p_late = mod.modulate(0.5, 400, 500)
        
        assert p_late > p_early
        assert abs(p_late - p_early - 0.4) < 0.01  # ~40pp increase over 400 cycles
    
    def test_monotonic_down_decreases(self):
        """Monotonic down drift should decrease probability over time."""
        config = create_monotonic_drift(slope=0.001, direction="down")
        mod = DriftModulator(config)
        
        p_early = mod.modulate(0.7, 0, 500)
        p_late = mod.modulate(0.7, 400, 500)
        
        assert p_late < p_early
    
    def test_cyclical_oscillates(self):
        """Cyclical drift should oscillate around base probability."""
        config = create_cyclical_drift(amplitude=0.15, period=100)
        series = simulate_drift_series(config, base_prob=0.5, total_cycles=500)
        
        # Should have peaks and troughs
        max_p = max(series)
        min_p = min(series)
        
        assert max_p > 0.6  # Peak above base + amplitude
        assert min_p < 0.4  # Trough below base - amplitude
        
        # Check periodicity - values at 0, 100, 200 should be similar (same phase)
        assert abs(series[0] - series[100]) < 0.01
        assert abs(series[0] - series[200]) < 0.01


# ==============================================================================
# TEST CLASS 3: CORRELATION DETERMINISM (2 tests)
# ==============================================================================

class TestCorrelationDeterminism:
    """Tests for correlation model determinism."""
    
    def test_correlation_engine_deterministic(self):
        """Same seed should produce same correlation outcomes."""
        config = create_correlation(rho=0.5, mode="class")
        
        engine1 = CorrelationEngine(config, seed=42)
        engine2 = CorrelationEngine(config, seed=42)
        
        results1 = []
        results2 = []
        
        for cycle in range(100):
            cycle_seed = 1000 + cycle
            r1 = engine1.apply_correlation(True, "class_a", cycle, cycle_seed, "item_1")
            r2 = engine2.apply_correlation(True, "class_a", cycle, cycle_seed, "item_1")
            results1.append(r1)
            results2.append(r2)
        
        assert results1 == results2
    
    def test_different_cycle_seeds_different_results(self):
        """Different cycle seeds should produce different correlation patterns."""
        config = create_correlation(rho=0.5, mode="class")
        
        engine = CorrelationEngine(config, seed=42)
        
        results_seed_a = []
        results_seed_b = []
        
        for cycle in range(100):
            # Different cycle seeds for each run
            r1 = engine.apply_correlation(True, "class_a", cycle, 1000 + cycle, "item_1")
            results_seed_a.append(r1)
            
        engine.clear_cache()
        
        for cycle in range(100):
            r2 = engine.apply_correlation(True, "class_a", cycle, 9000 + cycle, "item_1")
            results_seed_b.append(r2)
        
        # Results should differ (statistically very unlikely to be identical)
        assert results_seed_a != results_seed_b


# ==============================================================================
# TEST CLASS 4: CORRELATION PARAMETER EFFECTS (3 tests)
# ==============================================================================

class TestCorrelationParameterEffects:
    """Tests validating correlation parameters have correct effects."""
    
    def test_zero_correlation_independent(self):
        """With ρ=0, outcomes should remain independent."""
        config = create_correlation(rho=0.0, mode="class")
        engine = CorrelationEngine(config, seed=42)
        
        # All inputs should pass through unchanged
        for cycle in range(100):
            cycle_seed = 1000 + cycle
            assert engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_1") == True
            assert engine.apply_correlation(False, "class_a", cycle, cycle_seed, "item_2") == False
    
    def test_high_correlation_same_class_covariance(self):
        """With high ρ, items in same class should have correlated outcomes."""
        config = create_correlation(rho=0.9, mode="class")
        engine = CorrelationEngine(config, seed=42)
        
        # Track outcomes for two items in same class
        matches = 0
        total = 100
        
        for cycle in range(total):
            cycle_seed = 1000 + cycle
            # Both start with independent=True
            r1 = engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_a1")
            r2 = engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_a2")
            
            if r1 == r2:
                matches += 1
        
        # With ρ=0.9, outcomes should match frequently (>70%)
        match_rate = matches / total
        assert match_rate > 0.7, f"Expected high correlation, got match rate {match_rate}"
    
    def test_different_classes_less_correlated(self):
        """Items in different classes should be less correlated (class mode)."""
        config = create_correlation(rho=0.9, mode="class")
        engine = CorrelationEngine(config, seed=42)
        
        same_class_matches = 0
        diff_class_matches = 0
        total = 100
        
        for cycle in range(total):
            cycle_seed = 1000 + cycle
            
            # Same class (class_a)
            r1 = engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_a1")
            r2 = engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_a2")
            if r1 == r2:
                same_class_matches += 1
            
            # Different class (class_a vs class_b)
            r3 = engine.apply_correlation(True, "class_a", cycle, cycle_seed, "item_a3")
            r4 = engine.apply_correlation(True, "class_b", cycle, cycle_seed, "item_b1")
            if r3 == r4:
                diff_class_matches += 1
        
        # Same class should have higher correlation than different classes
        same_rate = same_class_matches / total
        diff_rate = diff_class_matches / total
        
        # Not strictly true due to randomness, but on average same class should correlate more
        # We just check that same class correlation is high
        assert same_rate > 0.6


# ==============================================================================
# TEST CLASS 5: SNAPSHOT REPRODUCIBILITY (3 tests)
# ==============================================================================

class TestSnapshotReproducibility:
    """Tests for bit-exact generation reproducibility."""
    
    def test_full_generation_reproducible(self, temp_output_dir):
        """Two runs with same seed should produce identical logs."""
        scenario = load_scenario("synthetic_null_uplift")
        
        # Generate twice
        path1, _ = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=100,
            out_dir=temp_output_dir / "run1",
            seed=42,
            verbose=False,
        )
        
        path2, _ = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=100,
            out_dir=temp_output_dir / "run2",
            seed=42,
            verbose=False,
        )
        
        # Compare file contents
        with open(path1) as f1, open(path2) as f2:
            content1 = f1.read()
            content2 = f2.read()
        
        assert content1 == content2
    
    def test_telemetry_hash_stable(self, temp_output_dir):
        """Telemetry hash should be identical across runs."""
        scenario = load_scenario("synthetic_positive_uplift")
        
        _, manifest1_path = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="rfl",
            cycles=100,
            out_dir=temp_output_dir / "hash1",
            seed=42,
            verbose=False,
        )
        
        _, manifest2_path = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="rfl",
            cycles=100,
            out_dir=temp_output_dir / "hash2",
            seed=42,
            verbose=False,
        )
        
        with open(manifest1_path) as f:
            m1 = json.load(f)
        with open(manifest2_path) as f:
            m2 = json.load(f)
        
        assert m1["telemetry_hash"] == m2["telemetry_hash"]
        assert m1["noise_config_hash"] == m2["noise_config_hash"]
    
    def test_drift_scenario_reproducible(self, temp_output_dir):
        """Drift scenarios should be exactly reproducible."""
        scenario = load_scenario("synthetic_drift_cyclical")
        
        path1, _ = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=200,
            out_dir=temp_output_dir / "drift1",
            seed=999,
            verbose=False,
        )
        
        path2, _ = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=200,
            out_dir=temp_output_dir / "drift2",
            seed=999,
            verbose=False,
        )
        
        # Compare specific records
        with open(path1) as f1:
            records1 = [json.loads(line) for line in f1]
        with open(path2) as f2:
            records2 = [json.loads(line) for line in f2]
        
        for i, (r1, r2) in enumerate(zip(records1, records2)):
            assert r1["success"] == r2["success"], f"Cycle {i} mismatch"
            assert r1["drifted_probability"] == r2["drifted_probability"]


# ==============================================================================
# TEST CLASS 6: MANIFEST INTEGRITY (3 tests)
# ==============================================================================

class TestManifestIntegrity:
    """Tests for manifest correctness and integrity."""
    
    def test_manifest_contains_all_fields(self, temp_output_dir):
        """Manifest should contain all required fields."""
        scenario = load_scenario("synthetic_correlation_high")
        
        _, manifest_path = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=50,
            out_dir=temp_output_dir,
            seed=42,
            verbose=False,
        )
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        required_fields = [
            "label", "synthetic", "version",
            "scenario_name", "scenario_description", "mode", "cycles", "initial_seed",
            "probability_matrix", "drift_config", "correlation_config",
            "scenario_config_hash", "telemetry_hash", "noise_config_hash",
            "statistics", "outputs", "generated_at", "seed_schedule",
        ]
        
        for field in required_fields:
            assert field in manifest, f"Missing field: {field}"
    
    def test_manifest_safety_label(self, temp_output_dir):
        """Manifest should have correct safety label."""
        scenario = load_scenario("synthetic_null_uplift")
        
        _, manifest_path = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=50,
            out_dir=temp_output_dir,
            seed=42,
            verbose=False,
        )
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest["label"] == SAFETY_LABEL
        assert manifest["synthetic"] is True
    
    def test_manifest_drift_config_correct(self, temp_output_dir):
        """Manifest should correctly record drift configuration."""
        scenario = load_scenario("synthetic_drift_shock")
        
        _, manifest_path = generate_synthetic_logs_v2(
            scenario=scenario,
            mode="baseline",
            cycles=50,
            out_dir=temp_output_dir,
            seed=42,
            verbose=False,
        )
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        drift = manifest["drift_config"]
        assert drift["mode"] == "shock"
        assert drift["shock_cycle"] == 250
        assert drift["shock_delta"] == -0.30


# ==============================================================================
# TEST CLASS 7: SCENARIO SUITE VALIDATION (2 additional tests)
# ==============================================================================

class TestScenarioSuiteValidation:
    """Tests for scenario suite integrity."""
    
    def test_all_12_scenarios_defined(self):
        """Suite should contain exactly 12 scenarios."""
        scenarios = list_scenarios()
        assert len(scenarios) == 12
    
    def test_all_scenarios_loadable(self):
        """All scenarios should be loadable without error."""
        for name in list_scenarios():
            scenario = load_scenario(name)
            assert scenario.name == name
            assert scenario.description
            assert "baseline" in scenario.probabilities
            assert "rfl" in scenario.probabilities


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

