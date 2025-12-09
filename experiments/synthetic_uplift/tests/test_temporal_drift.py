#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Temporal Drift Simulator v1.0
----------------------------------------------

Tests for:
    - Drift mode determinism (sinusoidal, linear, step)
    - No uplift implications
    - Drift disabled by default
    - Configuration validation

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import math
import pytest
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.temporal_drift import (
    TemporalDriftMode,
    TemporalDriftConfig,
    TemporalDriftSimulator,
    DriftedSignalType,
    DriftedSignalConfig,
    ScenarioTemporalDrift,
    create_sinusoidal_drift,
    create_linear_drift,
    create_step_drift,
    create_scenario_drift,
)


# ==============================================================================
# DRIFT DETERMINISM TESTS
# ==============================================================================

class TestDriftDeterminism:
    """Tests for deterministic drift behavior."""
    
    def test_sinusoidal_drift_deterministic(self):
        """Sinusoidal drift should produce same values for same inputs."""
        config = create_sinusoidal_drift(period=100, amplitude=0.2)
        
        sim1 = TemporalDriftSimulator(seed=42)
        sim1.configure_signal(DriftedSignalConfig(
            signal_type=DriftedSignalType.SUCCESS_RATE,
            base_value=0.5,
            drift_config=config,
        ))
        
        sim2 = TemporalDriftSimulator(seed=42)
        sim2.configure_signal(DriftedSignalConfig(
            signal_type=DriftedSignalType.SUCCESS_RATE,
            base_value=0.5,
            drift_config=config,
        ))
        
        for cycle in range(100):
            v1 = sim1.get_value(DriftedSignalType.SUCCESS_RATE, cycle, 100)
            v2 = sim2.get_value(DriftedSignalType.SUCCESS_RATE, cycle, 100)
            assert v1 == v2, f"Cycle {cycle}: {v1} != {v2}"
    
    def test_linear_drift_deterministic(self):
        """Linear drift should produce same values for same inputs."""
        config = create_linear_drift(slope=0.001)
        
        sim = TemporalDriftSimulator(seed=42)
        sim.configure_signal(DriftedSignalConfig(
            signal_type=DriftedSignalType.CONFUSABILITY,
            base_value=0.3,
            drift_config=config,
        ))
        
        # Run twice
        values1 = sim.get_drift_profile(DriftedSignalType.CONFUSABILITY, 100)
        values2 = sim.get_drift_profile(DriftedSignalType.CONFUSABILITY, 100)
        
        assert values1 == values2
    
    def test_step_drift_deterministic(self):
        """Step drift should produce same values for same inputs."""
        config = create_step_drift(
            step_cycles=[50, 100, 150],
            step_values=[0.1, 0.2, 0.3],
        )
        
        sim = TemporalDriftSimulator(seed=42)
        sim.configure_signal(DriftedSignalConfig(
            signal_type=DriftedSignalType.ABSTENTION_RATE,
            base_value=0.1,
            drift_config=config,
        ))
        
        # Check step behavior (use pytest.approx for floating point)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 25, 200) == pytest.approx(0.1)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 75, 200) == pytest.approx(0.2)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 125, 200) == pytest.approx(0.3)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 175, 200) == pytest.approx(0.4)


# ==============================================================================
# DRIFT DISABLED BY DEFAULT
# ==============================================================================

class TestDriftDisabledByDefault:
    """Tests ensuring drift is disabled by default."""
    
    def test_default_config_disabled(self):
        """Default TemporalDriftConfig should have enabled=False."""
        config = TemporalDriftConfig()
        assert config.enabled is False
        assert config.mode == TemporalDriftMode.NONE
    
    def test_scenario_drift_disabled_by_default(self):
        """ScenarioTemporalDrift should not enable drift by default."""
        spec = ScenarioTemporalDrift()
        assert not spec.is_enabled
    
    def test_disabled_drift_returns_base_value(self):
        """Disabled drift should return base value for all cycles."""
        spec = ScenarioTemporalDrift(base_success_rate=0.75)
        sim = spec.build_simulator()
        
        for cycle in range(100):
            value = sim.get_value(DriftedSignalType.SUCCESS_RATE, cycle, 100)
            assert value == 0.75


# ==============================================================================
# NO UPLIFT IMPLICATION TESTS
# ==============================================================================

class TestNoUpliftImplication:
    """Tests ensuring drift doesn't imply uplift."""
    
    def test_drift_output_has_safety_label(self):
        """Drift spec serialization should include safety label."""
        spec = ScenarioTemporalDrift(
            success_drift=create_sinusoidal_drift(),
        )
        data = spec.to_dict()
        assert data["label"] == SAFETY_LABEL
    
    def test_drift_values_are_probabilities(self):
        """All drift values should be valid probabilities [0.01, 0.99]."""
        spec = ScenarioTemporalDrift(
            success_drift=create_sinusoidal_drift(amplitude=0.5),
            base_success_rate=0.5,
        )
        sim = spec.build_simulator()
        
        for cycle in range(500):
            value = sim.get_value(DriftedSignalType.SUCCESS_RATE, cycle, 500)
            assert 0.01 <= value <= 0.99, f"Cycle {cycle}: value {value} out of range"
    
    def test_drift_is_structural_not_empirical(self):
        """Drift configuration should be structural, not claiming empirical validity."""
        # This is a documentation test - the config should be clear
        config = create_sinusoidal_drift()
        config_dict = config.to_dict()
        
        # Should not contain empirical claims
        assert "uplift" not in str(config_dict).lower()
        assert "real" not in str(config_dict).lower()


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_sinusoidal_config(self):
        """Valid sinusoidal config should pass validation."""
        config = create_sinusoidal_drift(period=100, amplitude=0.2)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_invalid_sinusoidal_period(self):
        """Sinusoidal with period <= 0 should fail validation."""
        config = TemporalDriftConfig(
            mode=TemporalDriftMode.SINUSOIDAL,
            period=0,
            amplitude=0.1,
            enabled=True,
        )
        errors = config.validate()
        assert any("period > 0" in e for e in errors)
    
    def test_invalid_amplitude(self):
        """Amplitude outside [-1, 1] should fail validation."""
        config = TemporalDriftConfig(
            mode=TemporalDriftMode.SINUSOIDAL,
            period=100,
            amplitude=1.5,
            enabled=True,
        )
        errors = config.validate()
        assert any("Amplitude" in e for e in errors)
    
    def test_invalid_linear_slope(self):
        """Too steep linear slope should fail validation."""
        config = TemporalDriftConfig(
            mode=TemporalDriftMode.LINEAR,
            slope=0.05,  # Too steep
            enabled=True,
        )
        errors = config.validate()
        assert any("slope" in e.lower() for e in errors)
    
    def test_step_cycles_must_match_values(self):
        """Step cycles and values must have same length."""
        config = TemporalDriftConfig(
            mode=TemporalDriftMode.STEP,
            step_cycles=[50, 100],
            step_values=[0.1],  # Mismatched length
            enabled=True,
        )
        errors = config.validate()
        assert any("same length" in e for e in errors)
    
    def test_step_cycles_must_be_sorted(self):
        """Step cycles must be in ascending order."""
        config = TemporalDriftConfig(
            mode=TemporalDriftMode.STEP,
            step_cycles=[100, 50],  # Not sorted
            step_values=[0.1, 0.2],
            enabled=True,
        )
        errors = config.validate()
        assert any("ascending" in e for e in errors)


# ==============================================================================
# DRIFT PROFILE TESTS
# ==============================================================================

class TestDriftProfiles:
    """Tests for drift profile behavior."""
    
    def test_sinusoidal_oscillates(self):
        """Sinusoidal drift should oscillate around base value."""
        config = create_sinusoidal_drift(period=100, amplitude=0.2)
        spec = ScenarioTemporalDrift(
            success_drift=config,
            base_success_rate=0.5,
        )
        sim = spec.build_simulator()
        
        profile = sim.get_drift_profile(DriftedSignalType.SUCCESS_RATE, 100)
        
        # Should have values above and below base
        assert any(v > 0.5 for v in profile)
        assert any(v < 0.5 for v in profile)
        
        # Maximum deviation should be close to amplitude
        max_dev = max(abs(v - 0.5) for v in profile)
        assert pytest.approx(max_dev, abs=0.01) == 0.2
    
    def test_linear_monotonic_increase(self):
        """Linear drift with positive slope should increase."""
        config = create_linear_drift(slope=0.001)
        spec = ScenarioTemporalDrift(
            confusability_drift=config,
            base_confusability=0.2,
        )
        sim = spec.build_simulator()
        
        profile = sim.get_drift_profile(DriftedSignalType.CONFUSABILITY, 100)
        
        # Should be monotonically increasing
        for i in range(1, len(profile)):
            assert profile[i] >= profile[i-1]
    
    def test_step_discrete_jumps(self):
        """Step drift should have discrete jumps at specified cycles."""
        config = create_step_drift(
            step_cycles=[25, 50, 75],
            step_values=[0.05, 0.10, 0.15],
        )
        spec = ScenarioTemporalDrift(
            abstention_drift=config,
            base_abstention_rate=0.05,
        )
        sim = spec.build_simulator()
        
        # Check values at different stages
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 10, 100) == pytest.approx(0.05)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 30, 100) == pytest.approx(0.10)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 60, 100) == pytest.approx(0.15)
        assert sim.get_value(DriftedSignalType.ABSTENTION_RATE, 80, 100) == pytest.approx(0.20)


# ==============================================================================
# SERIALIZATION TESTS
# ==============================================================================

class TestSerialization:
    """Tests for serialization/deserialization."""
    
    def test_config_roundtrip(self):
        """Config should survive serialization roundtrip."""
        original = create_sinusoidal_drift(period=75, amplitude=0.18)
        
        data = original.to_dict()
        restored = TemporalDriftConfig.from_dict(data)
        
        assert restored.mode == original.mode
        assert restored.period == original.period
        assert restored.amplitude == original.amplitude
        assert restored.enabled == original.enabled
    
    def test_scenario_drift_roundtrip(self):
        """ScenarioTemporalDrift should survive serialization roundtrip."""
        original = ScenarioTemporalDrift(
            success_drift=create_sinusoidal_drift(),
            confusability_drift=create_linear_drift(slope=0.0005),
            base_success_rate=0.65,
            base_confusability=0.25,
        )
        
        data = original.to_dict()
        restored = ScenarioTemporalDrift.from_dict(data)
        
        assert restored.base_success_rate == original.base_success_rate
        assert restored.base_confusability == original.base_confusability
        assert restored.success_drift is not None
        assert restored.confusability_drift is not None


# ==============================================================================
# FACTORY FUNCTION TESTS
# ==============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_sinusoidal_drift(self):
        """create_sinusoidal_drift should create valid config."""
        config = create_sinusoidal_drift(period=50, amplitude=0.1)
        
        assert config.mode == TemporalDriftMode.SINUSOIDAL
        assert config.period == 50
        assert config.amplitude == 0.1
        assert config.enabled is True
    
    def test_create_linear_drift(self):
        """create_linear_drift should create valid config."""
        config = create_linear_drift(slope=0.002)
        
        assert config.mode == TemporalDriftMode.LINEAR
        assert config.slope == 0.002
        assert config.enabled is True
    
    def test_create_step_drift(self):
        """create_step_drift should create valid config."""
        config = create_step_drift([100, 200], [0.1, 0.2])
        
        assert config.mode == TemporalDriftMode.STEP
        assert config.step_cycles == [100, 200]
        assert config.step_values == [0.1, 0.2]
        assert config.enabled is True
    
    def test_create_scenario_drift(self):
        """create_scenario_drift should create valid spec."""
        spec = create_scenario_drift(
            success_mode="sinusoidal",
            confusability_mode="linear",
            period=80,
            amplitude=0.12,
            slope=0.0008,
        )
        
        assert spec.success_drift is not None
        assert spec.success_drift.mode == TemporalDriftMode.SINUSOIDAL
        assert spec.confusability_drift is not None
        assert spec.confusability_drift.mode == TemporalDriftMode.LINEAR


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

