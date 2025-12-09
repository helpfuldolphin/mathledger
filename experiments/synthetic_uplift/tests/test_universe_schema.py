#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Universe Schema and Compiler
--------------------------------------------

Tests for:
    - NoiseSpec schema validation
    - Universe compilation
    - Rare event channels
    - Variance models
    - Determinism guarantees

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import yaml

import sys
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL, DriftMode
from experiments.synthetic_uplift.noise_schema import (
    ItemSpec,
    NoiseSpec,
    NoiseSpecBuilder,
    ProbabilityMatrix,
    RareEventChannel,
    RareEventEngine,
    RareEventType,
    VarianceConfig,
    create_catastrophic_collapse,
    create_class_outlier_burst,
    create_intermittent_failure,
    create_recovery_spike,
    create_sudden_uplift,
    validate_spec,
)
from experiments.synthetic_uplift.universe_compiler import (
    CompilationError,
    SyntheticUniverse,
    compile_universe,
    compare_universes,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory(prefix="universe_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_spec():
    """Create a basic valid spec."""
    return (
        NoiseSpecBuilder("synthetic_test_basic", "Basic test universe")
        .with_seed(42)
        .with_cycles(100)
        .with_probabilities(
            baseline={"class_a": 0.7, "class_b": 0.5, "class_c": 0.3},
            rfl={"class_a": 0.8, "class_b": 0.6, "class_c": 0.4},
        )
        .build()
    )


# ==============================================================================
# SCHEMA TESTS
# ==============================================================================

class TestNoiseSpecSchema:
    """Tests for NoiseSpec schema."""
    
    def test_spec_requires_synthetic_prefix(self):
        """Spec name must start with 'synthetic_'."""
        with pytest.raises(ValueError, match="must start with 'synthetic_'"):
            NoiseSpec(name="invalid_name")
    
    def test_spec_serialization_roundtrip(self, basic_spec):
        """Spec should survive serialization roundtrip."""
        # To dict and back
        spec_dict = basic_spec.to_dict()
        restored = NoiseSpec.from_dict(spec_dict)
        
        assert restored.name == basic_spec.name
        assert restored.seed == basic_spec.seed
        assert restored.num_cycles == basic_spec.num_cycles
    
    def test_spec_yaml_roundtrip(self, basic_spec, temp_output_dir):
        """Spec should survive YAML file roundtrip."""
        yaml_path = temp_output_dir / "test_spec.yaml"
        basic_spec.save(yaml_path)
        
        restored = NoiseSpec.load(yaml_path)
        assert restored.name == basic_spec.name
        assert restored.compute_hash() == basic_spec.compute_hash()
    
    def test_spec_json_roundtrip(self, basic_spec, temp_output_dir):
        """Spec should survive JSON file roundtrip."""
        json_path = temp_output_dir / "test_spec.json"
        basic_spec.save(json_path)
        
        restored = NoiseSpec.load(json_path)
        assert restored.name == basic_spec.name
    
    def test_spec_hash_deterministic(self, basic_spec):
        """Spec hash should be deterministic."""
        hash1 = basic_spec.compute_hash()
        hash2 = basic_spec.compute_hash()
        assert hash1 == hash2
    
    def test_spec_auto_generates_items(self, basic_spec):
        """Spec should auto-generate items when not provided."""
        items = basic_spec.get_items()
        expected_count = len(basic_spec.classes) * basic_spec.num_items_per_class
        assert len(items) == expected_count


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

class TestSpecValidation:
    """Tests for spec validation."""
    
    def test_valid_spec_passes(self, basic_spec):
        """Valid spec should pass validation."""
        errors = validate_spec(basic_spec)
        assert len(errors) == 0
    
    def test_invalid_probability_rejected(self):
        """Invalid probabilities should be rejected."""
        spec = (
            NoiseSpecBuilder("synthetic_test_invalid_prob")
            .with_probabilities(baseline={"class_a": 1.5})  # Invalid: > 1.0
            .build()
        )
        errors = validate_spec(spec)
        assert any("Invalid probability" in e for e in errors)
    
    def test_invalid_correlation_rejected(self):
        """Invalid correlation coefficient should be rejected."""
        spec = (
            NoiseSpecBuilder("synthetic_test_invalid_corr")
            .with_probabilities(baseline={"class_a": 0.5})
            .with_correlation(rho=1.5)  # Invalid: > 1.0
            .build()
        )
        errors = validate_spec(spec)
        assert any("rho" in e.lower() for e in errors)
    
    def test_invalid_cyclical_drift_rejected(self):
        """Cyclical drift with invalid period should be rejected."""
        spec = (
            NoiseSpecBuilder("synthetic_test_invalid_drift")
            .with_probabilities(baseline={"class_a": 0.5})
            .with_drift(mode="cyclical", period=0)  # Invalid: period must be > 0
            .build()
        )
        errors = validate_spec(spec)
        assert any("period" in e.lower() for e in errors)


# ==============================================================================
# RARE EVENT TESTS
# ==============================================================================

class TestRareEvents:
    """Tests for rare event channels."""
    
    def test_catastrophic_collapse_triggers(self):
        """Catastrophic collapse should trigger at specified cycle."""
        event = create_catastrophic_collapse(trigger_cycle=100, magnitude=-0.5, duration=50)
        engine = RareEventEngine([event], master_seed=42)
        
        # Before trigger
        effects = engine.get_active_effects(99, "class_a")
        assert len(effects) == 0
        
        # At trigger
        effects = engine.get_active_effects(100, "class_a")
        assert len(effects) == 1
        assert effects[0][1] == pytest.approx(-0.5)
        
        # During event
        effects = engine.get_active_effects(120, "class_a")
        assert len(effects) == 1
        
        # After event ends
        effects = engine.get_active_effects(151, "class_a")
        assert len(effects) == 0
    
    def test_class_outlier_only_affects_target(self):
        """Class outlier burst should only affect target class."""
        event = create_class_outlier_burst("class_a", trigger_probability=0.0, magnitude=-0.4, duration=5)
        event.trigger_cycles = [50]  # Force trigger at cycle 50
        
        engine = RareEventEngine([event], master_seed=42)
        
        # At trigger - only class_a affected
        effects_a = engine.get_active_effects(50, "class_a")
        effects_b = engine.get_active_effects(50, "class_b")
        
        assert len(effects_a) == 1
        assert len(effects_b) == 0
    
    def test_recovery_decays_over_time(self):
        """Recovery spike should decay based on recovery_rate."""
        event = create_recovery_spike(trigger_cycle=100, magnitude=0.3, duration=50, recovery_rate=0.05)
        engine = RareEventEngine([event], master_seed=42)
        
        # At trigger
        effects_t0 = engine.get_active_effects(100, "class_a")
        mag_t0 = effects_t0[0][1] if effects_t0 else 0
        
        # After some cycles (should decay)
        effects_t20 = engine.get_active_effects(120, "class_a")
        mag_t20 = effects_t20[0][1] if effects_t20 else 0
        
        assert mag_t0 > mag_t20  # Magnitude should have decayed
    
    def test_intermittent_probabilistic_triggers(self):
        """Intermittent failures should trigger probabilistically."""
        event = create_intermittent_failure(trigger_probability=0.1, magnitude=-0.3, duration=3)
        engine = RareEventEngine([event], master_seed=42)
        
        # Count triggers over many cycles
        trigger_count = 0
        for cycle in range(1000):
            effects = engine.get_active_effects(cycle, "class_a")
            if effects:
                trigger_count += 1
        
        # With 10% probability and duration 3, expect roughly 10% * 3 = 30% of cycles to have active effect
        # But this is noisy, so just check it's reasonable (5% to 50%)
        trigger_rate = trigger_count / 1000
        assert 0.05 < trigger_rate < 0.50


# ==============================================================================
# COMPILER TESTS
# ==============================================================================

class TestUniverseCompiler:
    """Tests for universe compilation."""
    
    def test_compile_basic_spec(self, basic_spec):
        """Basic spec should compile successfully."""
        universe = compile_universe(basic_spec)
        
        assert universe.spec.name == basic_spec.name
        assert len(universe.item_ids) == len(basic_spec.classes) * basic_spec.num_items_per_class
        assert len(universe.seed_schedule) == basic_spec.num_cycles
    
    def test_compile_invalid_spec_raises(self):
        """Invalid spec should raise CompilationError."""
        spec = (
            NoiseSpecBuilder("synthetic_test_invalid")
            .with_probabilities(baseline={"class_a": 2.0})  # Invalid
            .build()
        )
        
        with pytest.raises(CompilationError):
            compile_universe(spec)
    
    def test_compiled_universe_generates_logs(self, basic_spec, temp_output_dir):
        """Compiled universe should generate valid logs."""
        universe = compile_universe(basic_spec)
        
        results_path, manifest_path, stats = universe.generate_logs(
            mode="baseline",
            out_dir=temp_output_dir,
            verbose=False,
        )
        
        assert results_path.exists()
        assert manifest_path.exists()
        assert stats.total_cycles == basic_spec.num_cycles
    
    def test_generation_deterministic(self, basic_spec, temp_output_dir):
        """Two generations with same spec should be identical."""
        universe1 = compile_universe(basic_spec)
        universe2 = compile_universe(basic_spec)
        
        path1, _, _ = universe1.generate_logs("baseline", temp_output_dir / "run1", verbose=False)
        path2, _, _ = universe2.generate_logs("baseline", temp_output_dir / "run2", verbose=False)
        
        with open(path1) as f1, open(path2) as f2:
            content1 = f1.read()
            content2 = f2.read()
        
        assert content1 == content2


# ==============================================================================
# VARIANCE TESTS
# ==============================================================================

class TestVarianceModel:
    """Tests for variance/noise model."""
    
    def test_variance_applied(self, temp_output_dir):
        """Variance should create probability fluctuation."""
        spec = (
            NoiseSpecBuilder("synthetic_test_variance")
            .with_cycles(200)
            .with_probabilities(baseline={"class_a": 0.5})
            .with_variance(per_cycle_sigma=0.1, per_item_sigma=0.05)
            .build()
        )
        
        universe = compile_universe(spec)
        _, manifest_path, _ = universe.generate_logs("baseline", temp_output_dir, verbose=False)
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # With variance, probability range should exceed base probability
        prob_range = manifest["statistics"]["probability_range"]
        assert prob_range[0] < 0.5  # Min should be below base
        assert prob_range[1] > 0.5  # Max should be above base


# ==============================================================================
# COMPARISON TESTS
# ==============================================================================

class TestUniverseComparison:
    """Tests for universe comparison."""
    
    def test_compare_identical_universes(self, basic_spec):
        """Identical universes should have no differences."""
        u1 = compile_universe(basic_spec)
        u2 = compile_universe(basic_spec)
        
        diff = compare_universes(u1, u2)
        assert len(diff["differences"]) == 0
    
    def test_compare_different_probabilities(self):
        """Different probabilities should be detected."""
        spec1 = (
            NoiseSpecBuilder("synthetic_test_cmp1")
            .with_probabilities(baseline={"class_a": 0.5})
            .build()
        )
        spec2 = (
            NoiseSpecBuilder("synthetic_test_cmp2")
            .with_probabilities(baseline={"class_a": 0.7})
            .build()
        )
        
        u1 = compile_universe(spec1)
        u2 = compile_universe(spec2)
        
        diff = compare_universes(u1, u2)
        assert "probabilities" in diff["differences"]


# ==============================================================================
# BUILDER TESTS
# ==============================================================================

class TestNoiseSpecBuilder:
    """Tests for fluent spec builder."""
    
    def test_builder_chain(self):
        """Builder should support fluent chaining."""
        spec = (
            NoiseSpecBuilder("synthetic_test_builder", "Test description")
            .with_seed(123)
            .with_cycles(300)
            .with_classes(["a", "b"], items_per_class=5)
            .with_probabilities(baseline={"a": 0.6, "b": 0.4})
            .with_drift(mode="cyclical", amplitude=0.1, period=50)
            .with_correlation(rho=0.3)
            .with_variance(per_cycle_sigma=0.02)
            .with_rare_event(create_catastrophic_collapse(trigger_cycle=150))
            .with_metadata(author="test")
            .build()
        )
        
        assert spec.name == "synthetic_test_builder"
        assert spec.seed == 123
        assert spec.num_cycles == 300
        assert spec.classes == ["a", "b"]
        assert spec.num_items_per_class == 5
        assert spec.drift.mode == DriftMode.CYCLICAL
        assert spec.correlation.rho == 0.3
        assert spec.variance.per_cycle_sigma == 0.02
        assert len(spec.rare_events) == 1
        assert spec.metadata["author"] == "test"


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

