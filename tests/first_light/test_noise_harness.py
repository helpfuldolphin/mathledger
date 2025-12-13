"""
Tests for P3 Noise Model Harness

Tests cover:
1. All distributions (Bernoulli, Gaussian, Exponential, Pareto, Mixture, HMM, Drift)
2. Boundary conditions (P3 parameter bounds)
3. Determinism (same seed produces identical results)
4. Integration with SyntheticStateGenerator

SHADOW MODE CONTRACT:
- All tests verify observational-only behavior
- No governance modification
- Deterministic with seeded PRNG

See: docs/system_law/P3_Noise_Model_Spec.md
"""

import pytest
from typing import Dict, Any, List

from backend.topology.first_light.noise_harness import (
    NoiseDistributionType,
    NoiseRegime,
    DegradationState,
    BaseNoiseParams,
    CorrelatedNoiseParams,
    DegradationParams,
    HeatDeathParams,
    HeavyTailParams,
    NonstationaryParams,
    AdaptiveParams,
    PathologyType,
    PathologySeverity,
    PathologyConfig,
    P3NoiseConfig,
    P3NoiseModel,
    P3NoiseHarness,
    NoiseDecisionType,
    NoiseDecision,
    NoiseStateSnapshot,
    select_noise_model,
    generate_noise_sample,
)
from backend.topology.first_light.runner import SyntheticStateGenerator


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_config() -> P3NoiseConfig:
    """Default P3 noise configuration."""
    return P3NoiseConfig.default_shadow()


@pytest.fixture
def mild_stress_config() -> P3NoiseConfig:
    """Mild stress profile configuration."""
    return P3NoiseConfig.mild_stress()


@pytest.fixture
def moderate_stress_config() -> P3NoiseConfig:
    """Moderate stress profile configuration."""
    return P3NoiseConfig.moderate_stress()


@pytest.fixture
def full_config() -> P3NoiseConfig:
    """Configuration with all regimes enabled."""
    return P3NoiseConfig(
        enabled=True,
        base=BaseNoiseParams(
            timeout_rate=0.08,
            spurious_fail_rate=0.03,
            spurious_pass_rate=0.01,
        ),
        correlated=CorrelatedNoiseParams(
            enabled=True,
            rho=0.10,
            theta={"factor_a": 0.20, "factor_b": 0.15},
            item_factors={"item_1": ["factor_a"], "item_2": ["factor_a", "factor_b"]},
        ),
        degradation=DegradationParams(
            enabled=True,
            alpha=0.05,
            beta=0.20,
            theta_healthy=0.01,
            theta_degraded=0.25,
        ),
        heat_death=HeatDeathParams(
            enabled=True,
            initial_resource=1000.0,
            min_resource=100.0,
            consumption_mean=10.0,
            consumption_std=5.0,
            recovery_mean=8.0,
            recovery_std=3.0,
        ),
        heavy_tail=HeavyTailParams(
            enabled=True,
            pi=0.10,
            lambda_exp=0.10,
            alpha=1.5,
            x_min=100.0,
        ),
        nonstationary=NonstationaryParams(
            enabled=True,
            theta_0=0.08,
            delta=0.0001,
            sigma=0.01,
        ),
        adaptive=AdaptiveParams(
            enabled=True,
            gamma=0.50,
        ),
    )


# =============================================================================
# Test: Distribution Types
# =============================================================================

class TestBernoulliDistribution:
    """Tests for Bernoulli distribution (base noise)."""

    def test_bernoulli_zero_rate(self):
        """Zero rate should never trigger."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.0,
                spurious_fail_rate=0.0,
                spurious_pass_rate=0.0,
            ),
        )
        model = P3NoiseModel(config, seed=42)

        for _ in range(100):
            model.step_cycle()
            assert not model.should_timeout("test_item")
            assert not model.should_spurious_fail("test_item")
            assert not model.should_spurious_pass("test_item")

    def test_bernoulli_one_rate(self):
        """Rate of 0.30 (max allowed) should trigger frequently."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.30,
                spurious_fail_rate=0.0,
                spurious_pass_rate=0.0,
            ),
        )
        model = P3NoiseModel(config, seed=42)

        timeout_count = 0
        for _ in range(100):
            model.step_cycle()
            if model.should_timeout("test_item"):
                timeout_count += 1

        # With 30% rate over 100 trials, expect roughly 30 timeouts
        assert 15 <= timeout_count <= 45

    def test_bernoulli_bounds(self):
        """Rates should be bounded per P3 spec."""
        # Valid bounds
        valid_config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.30,
                spurious_fail_rate=0.10,
                spurious_pass_rate=0.0,  # Total = 0.40
            ),
        )
        assert len(valid_config.validate()) == 0

        # Invalid: timeout rate too high
        with pytest.raises(ValueError):
            invalid_config = P3NoiseConfig(
                enabled=True,
                base=BaseNoiseParams(timeout_rate=0.35),  # > 0.30
            )
            invalid_config.validate_or_raise()


class TestGaussianDistribution:
    """Tests for Gaussian distribution (used in heat-death, nonstationary)."""

    def test_heat_death_gaussian_consumption(self):
        """Heat-death uses Gaussian for consumption/recovery."""
        config = P3NoiseConfig(
            enabled=True,
            heat_death=HeatDeathParams(
                enabled=True,
                initial_resource=1000.0,
                min_resource=100.0,
                consumption_mean=10.0,
                consumption_std=5.0,
                recovery_mean=8.0,
                recovery_std=3.0,
            ),
        )
        model = P3NoiseModel(config, seed=42)

        # Run enough cycles to observe resource changes
        resources = []
        for _ in range(50):
            model.step_cycle()
            resources.append(model._heat_death_resource)

        # Resource should fluctuate around initial value
        # (consumption_mean > recovery_mean, so slight decline expected)
        assert all(r > 0 for r in resources)
        assert resources[-1] < config.heat_death.initial_resource


class TestExponentialDistribution:
    """Tests for Exponential distribution (timeout durations)."""

    def test_exponential_timeout_duration(self):
        """Timeout durations should follow exponential distribution."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(timeout_rate=0.10),
            heavy_tail=HeavyTailParams(enabled=False),
        )
        model = P3NoiseModel(config, seed=42)
        model.step_cycle()

        durations = []
        for i in range(100):
            duration = model.sample_timeout_duration(f"item_{i}")
            durations.append(duration)

        # Exponential distribution has positive support
        assert all(d > 0 for d in durations)
        # Mean should be around 100ms (default lambda = 0.01)
        mean_duration = sum(durations) / len(durations)
        assert 50 < mean_duration < 200


class TestParetoDistribution:
    """Tests for Pareto distribution (heavy-tailed timeouts)."""

    def test_pareto_heavy_tail(self):
        """Heavy-tail model should produce occasional large values."""
        # Use max allowed pi=0.20, but test Pareto behavior
        config = P3NoiseConfig(
            enabled=True,
            heavy_tail=HeavyTailParams(
                enabled=True,
                pi=0.20,  # Max allowed (P3 bound)
                alpha=1.5,
                x_min=100.0,
            ),
        )
        model = P3NoiseModel(config, seed=42)
        model.step_cycle()

        durations = []
        for i in range(500):  # More samples to catch Pareto
            duration = model.sample_timeout_duration(f"item_{i}")
            durations.append(duration)

        # Some durations should be >= x_min (Pareto component)
        large_durations = [d for d in durations if d >= 100.0]
        assert len(large_durations) > 0, "Should have some Pareto-drawn durations"
        # Some should be significantly larger (heavy tail)
        max_duration = max(durations)
        assert max_duration > 50  # Should have variety in durations


class TestMixtureDistribution:
    """Tests for Mixture distribution (Exponential + Pareto)."""

    def test_mixture_has_both_components(self):
        """Mixture should produce both fast and slow timeouts."""
        config = P3NoiseConfig(
            enabled=True,
            heavy_tail=HeavyTailParams(
                enabled=True,
                pi=0.20,  # 20% Pareto, 80% exponential
                lambda_exp=0.10,
                alpha=1.5,
                x_min=100.0,
            ),
        )
        model = P3NoiseModel(config, seed=42)
        model.step_cycle()

        durations = []
        for i in range(500):
            duration = model.sample_timeout_duration(f"item_{i}")
            durations.append(duration)

        # Should have variety in durations (mixture produces range)
        min_duration = min(durations)
        max_duration = max(durations)

        # Exponential component produces small values, Pareto produces large
        assert min_duration < max_duration, "Should have variety in durations"
        # Large values should exist (from Pareto)
        large_count = sum(1 for d in durations if d >= 100)
        assert large_count > 0, "Should have some large (Pareto) durations"


class TestHMMDegradation:
    """Tests for two-state HMM (cluster degradation)."""

    def test_hmm_state_transitions(self):
        """HMM should transition between HEALTHY and DEGRADED states."""
        config = P3NoiseConfig(
            enabled=True,
            degradation=DegradationParams(
                enabled=True,
                alpha=0.10,  # 10% chance HEALTHY -> DEGRADED
                beta=0.30,   # 30% chance DEGRADED -> HEALTHY
            ),
        )
        model = P3NoiseModel(config, seed=42)

        states = []
        for _ in range(100):
            model.step_cycle()
            states.append(model._degradation_state)

        # Should have both states
        healthy_count = sum(1 for s in states if s == DegradationState.HEALTHY)
        degraded_count = sum(1 for s in states if s == DegradationState.DEGRADED)

        assert healthy_count > 0, "Should have some HEALTHY states"
        assert degraded_count > 0, "Should have some DEGRADED states"

    def test_hmm_stationary_distribution(self):
        """Long run should approach stationary distribution."""
        config = P3NoiseConfig(
            enabled=True,
            degradation=DegradationParams(
                enabled=True,
                alpha=0.05,  # P(H -> D)
                beta=0.20,   # P(D -> H)
            ),
        )
        model = P3NoiseModel(config, seed=42)

        # Run many cycles
        healthy_count = 0
        total = 1000
        for _ in range(total):
            model.step_cycle()
            if model._degradation_state == DegradationState.HEALTHY:
                healthy_count += 1

        # Stationary: P(H) = beta / (alpha + beta) = 0.20 / 0.25 = 0.80
        expected_healthy_rate = 0.20 / (0.05 + 0.20)
        actual_healthy_rate = healthy_count / total

        assert abs(actual_healthy_rate - expected_healthy_rate) < 0.10


class TestLinearDrift:
    """Tests for linear drift (nonstationary noise)."""

    def test_drift_increases_rate(self):
        """Positive drift should increase noise rate over time."""
        config = P3NoiseConfig(
            enabled=True,
            nonstationary=NonstationaryParams(
                enabled=True,
                theta_0=0.05,
                delta=0.0005,  # +0.0005 per cycle
                sigma=0.0,     # No noise for deterministic test
            ),
        )
        model = P3NoiseModel(config, seed=42)

        rates = []
        for _ in range(100):
            model.step_cycle()
            rate, _ = model.compute_timeout_rate("test_item")
            rates.append(rate)

        # Rate should increase
        assert rates[-1] > rates[0]
        # Increase should be approximately delta * cycles
        expected_increase = 0.0005 * 100
        actual_increase = rates[-1] - rates[0]
        assert abs(actual_increase - expected_increase) < 0.01


# =============================================================================
# Test: Parameter Bounds Validation
# =============================================================================

class TestParameterBoundsValidation:
    """Tests for P3 parameter bound enforcement."""

    def test_base_noise_bounds(self):
        """Base noise parameters must be within P3 bounds."""
        # Valid
        valid = BaseNoiseParams(
            timeout_rate=0.30,
            spurious_fail_rate=0.10,
            spurious_pass_rate=0.0,
        )
        assert len(valid.validate()) == 0

        # Invalid timeout rate
        invalid_timeout = BaseNoiseParams(timeout_rate=0.35)
        errors = invalid_timeout.validate()
        assert any("timeout_rate" in e for e in errors)

        # Invalid total
        invalid_total = BaseNoiseParams(
            timeout_rate=0.25,
            spurious_fail_rate=0.10,
            spurious_pass_rate=0.10,  # Total = 0.45 > 0.40
        )
        errors = invalid_total.validate()
        assert any("Total base noise" in e for e in errors)

    def test_correlated_bounds(self):
        """Correlated parameters must satisfy P3 constraints."""
        # Valid
        valid = CorrelatedNoiseParams(
            enabled=True,
            rho=0.10,
            theta={"factor": 0.20},
        )
        assert len(valid.validate()) == 0

        # Invalid: rho * max(theta) > 0.15
        invalid = CorrelatedNoiseParams(
            enabled=True,
            rho=0.30,
            theta={"factor": 0.60},  # 0.30 * 0.60 = 0.18 > 0.15
        )
        errors = invalid.validate()
        assert any("impact" in e.lower() for e in errors)

    def test_degradation_bounds(self):
        """Degradation parameters must satisfy P3 constraints."""
        # Valid
        valid = DegradationParams(
            enabled=True,
            alpha=0.05,
            beta=0.20,
            theta_healthy=0.01,
            theta_degraded=0.30,
        )
        assert len(valid.validate()) == 0

        # Invalid: beta <= alpha
        invalid = DegradationParams(
            enabled=True,
            alpha=0.10,
            beta=0.08,  # beta < alpha
        )
        errors = invalid.validate()
        assert any("beta must exceed alpha" in e for e in errors)

    def test_heat_death_bounds(self):
        """Heat-death parameters must satisfy P3 constraints."""
        # Valid
        valid = HeatDeathParams(
            enabled=True,
            initial_resource=1000.0,
            min_resource=100.0,
            consumption_mean=10.0,
            recovery_mean=8.0,  # >= 0.7 * 10 = 7
        )
        assert len(valid.validate()) == 0

        # Invalid: recovery too low
        invalid = HeatDeathParams(
            enabled=True,
            consumption_mean=10.0,
            recovery_mean=5.0,  # < 0.7 * 10 = 7
        )
        errors = invalid.validate()
        assert any("recovery_mean" in e for e in errors)

    def test_heavy_tail_bounds(self):
        """Heavy-tail parameters must satisfy P3 constraints."""
        # Valid
        valid = HeavyTailParams(
            enabled=True,
            pi=0.10,
            alpha=1.5,
        )
        assert len(valid.validate()) == 0

        # Invalid: alpha <= 1.0 (infinite mean)
        invalid = HeavyTailParams(
            enabled=True,
            alpha=0.9,  # Must be > 1.0
        )
        errors = invalid.validate()
        assert any("alpha" in e for e in errors)

    def test_nonstationary_bounds(self):
        """Nonstationary parameters must satisfy P3 constraints."""
        # Valid
        valid = NonstationaryParams(
            enabled=True,
            theta_0=0.10,
            delta=0.0001,
        )
        assert len(valid.validate()) == 0

        # Invalid: drift too large
        invalid = NonstationaryParams(
            enabled=True,
            delta=0.002,  # |0.002| * 1000 = 2.0 > 0.50
        )
        errors = invalid.validate()
        assert any("drift" in e.lower() for e in errors)


# =============================================================================
# Test: Determinism
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior with seeded PRNG."""

    def test_same_seed_same_results(self, full_config):
        """Same seed should produce identical noise decisions."""
        model1 = P3NoiseModel(full_config, seed=12345)
        model2 = P3NoiseModel(full_config, seed=12345)

        for _ in range(50):
            model1.step_cycle()
            model2.step_cycle()

            decision1 = model1.generate_decision("test_item", policy_prob=0.7)
            decision2 = model2.generate_decision("test_item", policy_prob=0.7)

            assert decision1.decision == decision2.decision
            assert decision1.computed_timeout_rate == decision2.computed_timeout_rate
            assert decision1.timeout_duration_ms == decision2.timeout_duration_ms

    def test_different_seed_different_results(self, full_config):
        """Different seeds should produce different noise decisions."""
        model1 = P3NoiseModel(full_config, seed=11111)
        model2 = P3NoiseModel(full_config, seed=22222)

        decisions1 = []
        decisions2 = []

        for _ in range(50):
            model1.step_cycle()
            model2.step_cycle()

            decisions1.append(model1.should_timeout("test_item"))
            decisions2.append(model2.should_timeout("test_item"))

        # Should have at least some differences
        differences = sum(1 for d1, d2 in zip(decisions1, decisions2) if d1 != d2)
        assert differences > 0

    def test_reset_restores_determinism(self, full_config):
        """Reset should restore deterministic behavior."""
        model = P3NoiseModel(full_config, seed=42)

        # Run some cycles
        decisions_before = []
        for _ in range(20):
            model.step_cycle()
            decisions_before.append(model.should_timeout("test_item"))

        # Reset and run again
        model.reset()
        decisions_after = []
        for _ in range(20):
            model.step_cycle()
            decisions_after.append(model.should_timeout("test_item"))

        assert decisions_before == decisions_after


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for select_noise_model and generate_noise_sample."""

    def test_select_noise_model_from_config(self, base_config):
        """select_noise_model should create model from P3NoiseConfig."""
        model = select_noise_model(base_config, seed=42)
        assert isinstance(model, P3NoiseModel)
        assert model.config == base_config

    def test_select_noise_model_from_dict(self):
        """select_noise_model should create model from dict."""
        config_dict = {
            "enabled": True,
            "base": {
                "timeout_rate": 0.05,
                "spurious_fail_rate": 0.02,
            },
        }
        model = select_noise_model(config_dict, seed=42)
        assert isinstance(model, P3NoiseModel)
        assert model.config.base.timeout_rate == 0.05

    def test_generate_noise_sample(self, base_config):
        """generate_noise_sample should return noise contribution."""
        model = select_noise_model(base_config, seed=42)
        model.step_cycle()

        state = {"item": "test_item", "policy_prob": 0.7}
        result = generate_noise_sample(model, state)

        assert "noise_decision" in result
        assert "delta_p_contribution" in result
        assert "computed_rates" in result
        assert "state_snapshot" in result

    def test_generate_noise_sample_delta_p(self, base_config):
        """generate_noise_sample should compute delta_p contribution."""
        model = select_noise_model(base_config, seed=42)

        contributions = []
        for i in range(100):
            model.step_cycle()
            state = {"item": f"item_{i}"}
            result = generate_noise_sample(model, state)
            contributions.append(result["delta_p_contribution"])

        # Should have mix of 0, negative (timeout/fail), positive (pass)
        has_zero = any(c == 0.0 for c in contributions)
        has_negative = any(c < 0 for c in contributions)

        assert has_zero or has_negative


# =============================================================================
# Test: P3 Noise Harness Integration
# =============================================================================

class TestP3NoiseHarness:
    """Tests for P3NoiseHarness integration."""

    def test_harness_initialization(self, base_config):
        """Harness should initialize with config."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)
        assert harness.model is not None
        assert harness.config == base_config

    def test_harness_apply_noise(self, base_config):
        """Harness should apply noise to state dict."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)
        harness.step_cycle()

        state = {
            "success": True,
            "H": 0.7,
            "rho": 0.8,
        }
        result = harness.apply_noise(state, item="test_item")

        assert "noise" in result
        # The noise_decision is nested inside "noise" dict
        assert "decision" in result["noise"]
        assert isinstance(result["noise"]["decision"], (str, type(None)))

    def test_harness_stability_report(self, base_config):
        """Harness should generate stability report."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)

        for i in range(50):
            harness.step_cycle()
            state = {"success": True, "H": 0.7}
            harness.apply_noise(state, item=f"item_{i}")

        report = harness.get_stability_report()

        assert "total_cycles" in report
        assert "rates" in report
        assert "noise_impact" in report
        assert "delta_p_impact" in report

    def test_harness_delta_p_contribution(self, base_config):
        """Harness should track cumulative delta_p contribution."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)

        for i in range(100):
            harness.step_cycle()
            state = {"success": True, "H": 0.7}
            harness.apply_noise(state, item=f"item_{i}")

        delta_p = harness.get_delta_p_noise_contribution(window_size=50)
        # Should be a float (may be positive, negative, or zero)
        assert isinstance(delta_p, float)


# =============================================================================
# Test: SyntheticStateGenerator Integration
# =============================================================================

class TestSyntheticStateGeneratorIntegration:
    """Tests for SyntheticStateGenerator with P3 noise."""

    def test_generator_without_noise(self):
        """Generator should work without noise config."""
        gen = SyntheticStateGenerator(tau_0=0.20, seed=42)

        state = gen.step()
        assert "success" in state
        assert state["noise_decision"] is None
        assert state["noise_caused_failure"] is False

    def test_generator_with_noise(self, base_config):
        """Generator should apply noise when configured."""
        gen = SyntheticStateGenerator(
            tau_0=0.20,
            seed=42,
            noise_config=base_config,
        )

        # Run enough cycles to likely see some noise effects
        noise_decisions = []
        for _ in range(100):
            state = gen.step()
            noise_decisions.append(state["noise_decision"])

        # Should have some non-None decisions
        non_none = [d for d in noise_decisions if d is not None]
        assert len(non_none) > 0

    def test_generator_noise_stability_report(self, base_config):
        """Generator should provide noise stability report."""
        gen = SyntheticStateGenerator(
            tau_0=0.20,
            seed=42,
            noise_config=base_config,
        )

        for _ in range(50):
            gen.step()

        report = gen.get_noise_stability_report()
        assert report is not None
        assert "total_cycles" in report

    def test_generator_noise_caused_failure(self):
        """Generator should track noise-caused failures."""
        # Use high noise config to ensure some failures
        high_noise_config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.25,
                spurious_fail_rate=0.10,
            ),
        )
        gen = SyntheticStateGenerator(
            tau_0=0.20,
            seed=42,
            noise_config=high_noise_config,
        )

        noise_failures = 0
        for _ in range(100):
            state = gen.step()
            if state.get("noise_caused_failure"):
                noise_failures += 1

        # Should have some noise-caused failures with high noise rate
        assert noise_failures > 0

    def test_generator_reset_with_noise(self, base_config):
        """Generator reset should reset noise harness too."""
        gen = SyntheticStateGenerator(
            tau_0=0.20,
            seed=42,
            noise_config=base_config,
        )

        # Run some cycles
        states_before = [gen.step() for _ in range(20)]

        # Reset
        gen.reset()

        # Run again
        states_after = [gen.step() for _ in range(20)]

        # Noise decisions should be identical (determinism)
        for s1, s2 in zip(states_before, states_after):
            assert s1["noise_decision"] == s2["noise_decision"]


# =============================================================================
# Test: Pathologies
# =============================================================================

class TestPathologies:
    """Tests for synthetic pathologies."""

    def test_spike_pathology(self):
        """Spike pathology should increase rate during duration."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(timeout_rate=0.05),
            pathologies=[
                PathologyConfig(
                    pathology_type=PathologyType.SPIKE,
                    t_start=10,
                    duration=5,
                    magnitude=0.15,
                ),
            ],
        )
        model = P3NoiseModel(config, seed=42)

        rates = []
        for _ in range(20):
            model.step_cycle()
            rate, _ = model.compute_timeout_rate("test_item")
            rates.append(rate)

        # Cycles 10-14 should have higher rate
        pre_spike = rates[5:9]
        during_spike = rates[10:15]
        post_spike = rates[16:20]

        avg_pre = sum(pre_spike) / len(pre_spike)
        avg_during = sum(during_spike) / len(during_spike)
        avg_post = sum(post_spike) / len(post_spike)

        assert avg_during > avg_pre
        assert avg_during > avg_post

    def test_drift_pathology(self):
        """Drift pathology should gradually change rate."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(timeout_rate=0.05),
            pathologies=[
                PathologyConfig(
                    pathology_type=PathologyType.DRIFT,
                    t_start=5,
                    rate=0.002,
                    direction=1,
                ),
            ],
        )
        model = P3NoiseModel(config, seed=42)

        rates = []
        for _ in range(50):
            model.step_cycle()
            rate, _ = model.compute_timeout_rate("test_item")
            rates.append(rate)

        # Rate should increase over time after t_start
        early_rate = sum(rates[10:15]) / 5
        late_rate = sum(rates[45:50]) / 5
        assert late_rate > early_rate

    def test_oscillation_pathology(self):
        """Oscillation pathology should create periodic variation."""
        config = P3NoiseConfig(
            enabled=True,
            base=BaseNoiseParams(timeout_rate=0.10),
            pathologies=[
                PathologyConfig(
                    pathology_type=PathologyType.OSCILLATION,
                    period=20,
                    amplitude=0.05,
                ),
            ],
        )
        model = P3NoiseModel(config, seed=42)

        rates = []
        for _ in range(100):
            model.step_cycle()
            rate, _ = model.compute_timeout_rate("test_item")
            rates.append(rate)

        # Rates should vary around base rate
        min_rate = min(rates)
        max_rate = max(rates)
        assert max_rate - min_rate > 0.02  # Should have variation


# =============================================================================
# Test: State Snapshots
# =============================================================================

class TestStateSnapshots:
    """Tests for noise state snapshots."""

    def test_snapshot_schema(self, full_config):
        """Snapshot should conform to P3 schema."""
        model = P3NoiseModel(full_config, seed=42)
        model.step_cycle()
        model.generate_decision("test_item")

        snapshot = model.get_state_snapshot()
        snapshot_dict = snapshot.to_dict()

        assert snapshot_dict["schema_version"] == "p3-noise-state/1.0.0"
        assert snapshot_dict["mode"] == "SHADOW"
        assert "cycle" in snapshot_dict
        assert "base_state" in snapshot_dict
        assert "regime_states" in snapshot_dict
        assert "decisions_this_cycle" in snapshot_dict

    def test_decision_schema(self, base_config):
        """Decision should conform to P3 schema."""
        model = P3NoiseModel(base_config, seed=42)
        model.step_cycle()

        decision = model.generate_decision("test_item")
        decision_dict = decision.to_dict()

        assert decision_dict["schema_version"] == "p3-noise-decision/1.0.0"
        assert "cycle" in decision_dict
        assert "item" in decision_dict
        assert "decision" in decision_dict
        assert "computed_rates" in decision_dict


# =============================================================================
# Test: SHADOW Mode Contract
# =============================================================================

class TestShadowModeContract:
    """Tests verifying SHADOW mode contract."""

    def test_mode_always_shadow(self, full_config):
        """All outputs should have mode = SHADOW."""
        model = P3NoiseModel(full_config, seed=42)
        model.step_cycle()
        model.generate_decision("test_item")

        snapshot = model.get_state_snapshot()
        assert snapshot.mode == "SHADOW"

        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict["mode"] == "SHADOW"

    def test_harness_report_shadow(self, base_config):
        """Harness stability report should have mode = SHADOW."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)
        for i in range(10):
            harness.step_cycle()
            harness.apply_noise({"H": 0.7}, item=f"item_{i}")

        report = harness.get_stability_report()
        assert report["mode"] == "SHADOW"

    def test_no_governance_modification(self, base_config):
        """Noise decisions should not modify governance."""
        harness = P3NoiseHarness(noise_config=base_config, seed=42)
        harness.step_cycle()

        state = {
            "success": True,
            "governance_aligned": True,
            "real_blocked": False,
        }
        result = harness.apply_noise(state.copy(), item="test_item")

        # Governance fields should remain unchanged by noise
        assert result["governance_aligned"] == state["governance_aligned"]
        assert result["real_blocked"] == state["real_blocked"]
