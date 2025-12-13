"""
Phase X P3: Noise Harness for First-Light Shadow Experiments

This module implements the noise model integration for First-Light shadow experiments,
providing deterministic noise injection per the P3_Noise_Model_Spec.md.

SHADOW MODE CONTRACT:
- All noise injection is observational only
- No governance modification
- Deterministic with seeded PRNG
- Bounded per specification

See: docs/system_law/P3_Noise_Model_Spec.md

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from rfl.prng import DeterministicPRNG, int_to_hex_seed


__all__ = [
    "NoiseDistributionType",
    "NoiseRegime",
    "P3NoiseConfig",
    "P3NoiseModel",
    "PathologyType",
    "PathologyConfig",
    "P3NoiseHarness",
    "select_noise_model",
    "generate_noise_sample",
    "NoiseDecision",
    "NoiseStateSnapshot",
]


# =============================================================================
# Distribution Types (per P3_Noise_Model_Spec.md Section 2)
# =============================================================================

class NoiseDistributionType(Enum):
    """Allowed noise distribution types from P3 spec."""

    BERNOULLI = "bernoulli"
    """P(X=1) = p, P(X=0) = 1-p"""

    GAUSSIAN = "gaussian"
    """X ~ N(mu, sigma^2)"""

    TRUNCATED_GAUSSIAN = "truncated_gaussian"
    """Gaussian clamped to [low, high]"""

    EXPONENTIAL = "exponential"
    """X ~ Exp(lambda), E[X] = 1/lambda"""

    PARETO = "pareto"
    """X ~ Pareto(alpha, x_min)"""

    MIXTURE = "mixture"
    """(1-pi)*Exp(lambda) + pi*Pareto(alpha, x_min)"""


class NoiseRegime(Enum):
    """Noise regime types from P3 spec."""

    BASE = "base"
    """Independent Bernoulli events"""

    CORRELATED = "correlated"
    """Latent factor model"""

    DEGRADATION = "degradation"
    """Two-state HMM (HEALTHY/DEGRADED)"""

    HEAT_DEATH = "heat_death"
    """Resource depletion process"""

    HEAVY_TAIL = "heavy_tail"
    """Mixture distribution for timeouts"""

    NONSTATIONARY = "nonstationary"
    """Time-varying drift"""

    ADAPTIVE = "adaptive"
    """Policy-aware confidence scaling"""


class DegradationState(Enum):
    """State for degradation HMM."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"


# =============================================================================
# Configuration Data Classes (per P3_Noise_Model_Spec.md Section 3)
# =============================================================================

@dataclass
class BaseNoiseParams:
    """Base noise parameters with P3 bounds."""

    timeout_rate: float = 0.05
    """P(timeout), bounded [0.0, 0.30]"""

    spurious_fail_rate: float = 0.02
    """P(spurious fail), bounded [0.0, 0.15]"""

    spurious_pass_rate: float = 0.01
    """P(spurious pass), bounded [0.0, 0.10]"""

    def validate(self) -> List[str]:
        """Validate against P3 bounds."""
        errors = []
        if not 0.0 <= self.timeout_rate <= 0.30:
            errors.append(f"timeout_rate {self.timeout_rate} not in [0.0, 0.30]")
        if not 0.0 <= self.spurious_fail_rate <= 0.15:
            errors.append(f"spurious_fail_rate {self.spurious_fail_rate} not in [0.0, 0.15]")
        if not 0.0 <= self.spurious_pass_rate <= 0.10:
            errors.append(f"spurious_pass_rate {self.spurious_pass_rate} not in [0.0, 0.10]")
        total = self.timeout_rate + self.spurious_fail_rate + self.spurious_pass_rate
        if total > 0.40:
            errors.append(f"Total base noise {total:.3f} exceeds 0.40")
        return errors


@dataclass
class CorrelatedNoiseParams:
    """Correlated failure parameters with P3 bounds."""

    enabled: bool = False
    rho: float = 0.10
    """Factor activation probability, bounded [0.0, 0.30]"""

    theta: Dict[str, float] = field(default_factory=dict)
    """Per-factor failure rates, each bounded [0.0, 0.50]"""

    item_factors: Dict[str, List[str]] = field(default_factory=dict)
    """Item to factor mapping"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 0.0 <= self.rho <= 0.30:
                errors.append(f"correlated.rho {self.rho} not in [0.0, 0.30]")
            max_theta = max(self.theta.values()) if self.theta else 0.0
            if self.rho * max_theta > 0.15:
                errors.append(f"Correlated impact rho*max(theta) = {self.rho * max_theta:.3f} exceeds 0.15")
        return errors


@dataclass
class DegradationParams:
    """Cluster degradation HMM parameters with P3 bounds."""

    enabled: bool = False
    alpha: float = 0.05
    """P(HEALTHY -> DEGRADED), bounded [0.0, 0.15]"""

    beta: float = 0.20
    """P(DEGRADED -> HEALTHY), bounded [0.05, 0.50]"""

    theta_healthy: float = 0.01
    """Failure rate in HEALTHY, bounded [0.0, 0.05]"""

    theta_degraded: float = 0.30
    """Failure rate in DEGRADED, bounded [0.10, 0.50]"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 0.0 <= self.alpha <= 0.15:
                errors.append(f"degradation.alpha {self.alpha} not in [0.0, 0.15]")
            if not 0.05 <= self.beta <= 0.50:
                errors.append(f"degradation.beta {self.beta} not in [0.05, 0.50]")
            if self.beta <= self.alpha:
                errors.append("degradation.beta must exceed alpha")
            if self.theta_degraded <= self.theta_healthy:
                errors.append("degradation.theta_degraded must exceed theta_healthy")
        return errors


@dataclass
class HeatDeathParams:
    """Heat-death resource depletion parameters with P3 bounds."""

    enabled: bool = False
    initial_resource: float = 1000.0
    """R_0, bounded [500, 2000]"""

    min_resource: float = 100.0
    """R_min, bounded [50, 200]"""

    consumption_mean: float = 10.0
    """mu_c, bounded [1.0, 20.0]"""

    consumption_std: float = 5.0
    """sigma_c, bounded [0.0, 10.0]"""

    recovery_mean: float = 8.0
    """mu_r, bounded [1.0, 20.0]"""

    recovery_std: float = 3.0
    """sigma_r, bounded [0.0, 10.0]"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 500 <= self.initial_resource <= 2000:
                errors.append(f"heat_death.initial_resource {self.initial_resource} not in [500, 2000]")
            if not 50 <= self.min_resource <= 200:
                errors.append(f"heat_death.min_resource {self.min_resource} not in [50, 200]")
            if self.recovery_mean < 0.7 * self.consumption_mean:
                errors.append("heat_death.recovery_mean must be >= 0.7 * consumption_mean")
            if self.initial_resource < 5 * self.min_resource:
                errors.append("heat_death.initial_resource must be >= 5 * min_resource")
        return errors


@dataclass
class HeavyTailParams:
    """Heavy-tail timeout distribution parameters with P3 bounds."""

    enabled: bool = False
    pi: float = 0.10
    """Pareto mixing probability, bounded [0.0, 0.20]"""

    lambda_exp: float = 0.10
    """Exponential rate, bounded (0.0, 1.0]"""

    alpha: float = 1.5
    """Pareto tail index, bounded (1.0, 3.0]"""

    x_min: float = 100.0
    """Pareto scale (ms), bounded [50, 500]"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 0.0 <= self.pi <= 0.20:
                errors.append(f"heavy_tail.pi {self.pi} not in [0.0, 0.20]")
            if not 0.0 < self.lambda_exp <= 1.0:
                errors.append(f"heavy_tail.lambda {self.lambda_exp} not in (0.0, 1.0]")
            if not 1.0 < self.alpha <= 3.0:
                errors.append(f"heavy_tail.alpha {self.alpha} not in (1.0, 3.0]")
            if not 50 <= self.x_min <= 500:
                errors.append(f"heavy_tail.x_min {self.x_min} not in [50, 500]")
        return errors


@dataclass
class NonstationaryParams:
    """Non-stationary drift parameters with P3 bounds."""

    enabled: bool = False
    theta_0: float = 0.10
    """Initial rate, bounded [0.0, 0.30]"""

    delta: float = 0.0001
    """Drift rate per cycle, bounded [-0.001, 0.001]"""

    sigma: float = 0.01
    """Noise std, bounded [0.0, 0.05]"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 0.0 <= self.theta_0 <= 0.30:
                errors.append(f"nonstationary.theta_0 {self.theta_0} not in [0.0, 0.30]")
            if not -0.001 <= self.delta <= 0.001:
                errors.append(f"nonstationary.delta {self.delta} not in [-0.001, 0.001]")
            if not 0.0 <= self.sigma <= 0.05:
                errors.append(f"nonstationary.sigma {self.sigma} not in [0.0, 0.05]")
            max_drift = abs(self.delta) * 1000
            if max_drift > 0.50:
                errors.append(f"Non-stationary drift over 1000 cycles ({max_drift:.3f}) exceeds 0.50")
        return errors


@dataclass
class AdaptiveParams:
    """Adaptive noise parameters with P3 bounds."""

    enabled: bool = False
    gamma: float = 0.50
    """Adaptation strength, bounded [0.0, 1.0]"""

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not 0.0 <= self.gamma <= 1.0:
                errors.append(f"adaptive.gamma {self.gamma} not in [0.0, 1.0]")
        return errors


# =============================================================================
# Pathology Configuration (per P3_Noise_Model_Spec.md Section 4)
# =============================================================================

class PathologyType(Enum):
    """Synthetic pathology types from P3 spec."""

    SPIKE = "spike"
    DRIFT = "drift"
    OSCILLATION = "oscillation"
    CLUSTER_BURST = "cluster_burst"
    HEAT_DEATH_CASCADE = "heat_death_cascade"


class PathologySeverity(Enum):
    """Pathology severity levels."""
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


@dataclass
class PathologyConfig:
    """Configuration for a synthetic pathology."""

    pathology_type: PathologyType
    severity: PathologySeverity = PathologySeverity.MILD

    # Spike parameters
    t_start: int = 0
    duration: int = 5
    magnitude: float = 0.10

    # Drift parameters
    rate: float = 0.0002
    direction: int = 1  # +1 or -1
    max_rate: float = 0.50

    # Oscillation parameters
    period: int = 50
    amplitude: float = 0.05
    phase: float = 0.0

    # Cluster burst parameters
    cluster_fraction: float = 0.2
    burst_rate: float = 0.3
    affected_factors: List[str] = field(default_factory=list)

    # Heat-death cascade parameters
    depletion_rate: float = 0.005
    R_critical: float = 100.0

    def get_impact(self, cycle: int, base_rate: float) -> float:
        """Compute pathology impact at given cycle."""

        if self.pathology_type == PathologyType.SPIKE:
            if self.t_start <= cycle < self.t_start + self.duration:
                return min(1.0, base_rate + self.magnitude)
            return base_rate

        elif self.pathology_type == PathologyType.DRIFT:
            if cycle >= self.t_start:
                delta = self.rate * (cycle - self.t_start) * self.direction
                return max(0.0, min(self.max_rate, base_rate + delta))
            return base_rate

        elif self.pathology_type == PathologyType.OSCILLATION:
            osc = self.amplitude * math.sin(2 * math.pi * cycle / self.period + self.phase)
            return max(0.0, min(1.0, base_rate + osc))

        else:
            return base_rate

    def is_active(self, cycle: int) -> bool:
        """Check if pathology is active at given cycle."""
        if self.pathology_type == PathologyType.SPIKE:
            return self.t_start <= cycle < self.t_start + self.duration
        elif self.pathology_type == PathologyType.DRIFT:
            return cycle >= self.t_start
        elif self.pathology_type == PathologyType.OSCILLATION:
            return True  # Always active
        elif self.pathology_type == PathologyType.CLUSTER_BURST:
            return self.t_start <= cycle < self.t_start + self.duration
        elif self.pathology_type == PathologyType.HEAT_DEATH_CASCADE:
            return True  # Always active
        return False


# =============================================================================
# Main P3 Noise Configuration
# =============================================================================

@dataclass
class P3NoiseConfig:
    """
    Full P3 Noise Model configuration.

    Combines all regime parameters per P3_Noise_Model_Spec.md.
    """

    enabled: bool = True
    """Master switch for noise injection."""

    base: BaseNoiseParams = field(default_factory=BaseNoiseParams)
    correlated: CorrelatedNoiseParams = field(default_factory=CorrelatedNoiseParams)
    degradation: DegradationParams = field(default_factory=DegradationParams)
    heat_death: HeatDeathParams = field(default_factory=HeatDeathParams)
    heavy_tail: HeavyTailParams = field(default_factory=HeavyTailParams)
    nonstationary: NonstationaryParams = field(default_factory=NonstationaryParams)
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)

    pathologies: List[PathologyConfig] = field(default_factory=list)
    """Active synthetic pathologies."""

    def validate(self) -> List[str]:
        """Validate all parameters against P3 bounds."""
        errors = []
        errors.extend(self.base.validate())
        errors.extend(self.correlated.validate())
        errors.extend(self.degradation.validate())
        errors.extend(self.heat_death.validate())
        errors.extend(self.heavy_tail.validate())
        errors.extend(self.nonstationary.validate())
        errors.extend(self.adaptive.validate())
        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid P3NoiseConfig: {'; '.join(errors)}")

    @classmethod
    def default_shadow(cls) -> "P3NoiseConfig":
        """Default configuration for shadow testing."""
        return cls(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.05,
                spurious_fail_rate=0.02,
                spurious_pass_rate=0.01,
            ),
        )

    @classmethod
    def mild_stress(cls) -> "P3NoiseConfig":
        """Mild stress profile for regression testing."""
        return cls(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.08,
                spurious_fail_rate=0.03,
                spurious_pass_rate=0.01,
            ),
            pathologies=[
                PathologyConfig(
                    pathology_type=PathologyType.SPIKE,
                    severity=PathologySeverity.MILD,
                    t_start=100,
                    duration=3,
                    magnitude=0.05,
                ),
            ],
        )

    @classmethod
    def moderate_stress(cls) -> "P3NoiseConfig":
        """Moderate stress profile for shadow experiment validation."""
        return cls(
            enabled=True,
            base=BaseNoiseParams(
                timeout_rate=0.10,
                spurious_fail_rate=0.04,
                spurious_pass_rate=0.02,
            ),
            degradation=DegradationParams(enabled=True),
            pathologies=[
                PathologyConfig(
                    pathology_type=PathologyType.SPIKE,
                    severity=PathologySeverity.MODERATE,
                    t_start=150,
                    duration=8,
                    magnitude=0.12,
                ),
                PathologyConfig(
                    pathology_type=PathologyType.OSCILLATION,
                    severity=PathologySeverity.MODERATE,
                    period=40,
                    amplitude=0.06,
                ),
            ],
        )


# =============================================================================
# Noise Decision Output
# =============================================================================

class NoiseDecisionType(Enum):
    """Type of noise decision."""
    CLEAN = "CLEAN"
    TIMEOUT = "TIMEOUT"
    SPURIOUS_FAIL = "SPURIOUS_FAIL"
    SPURIOUS_PASS = "SPURIOUS_PASS"


@dataclass
class NoiseDecision:
    """Result of a noise injection decision."""

    decision: NoiseDecisionType
    cycle: int
    item: str

    computed_timeout_rate: float = 0.0
    computed_fail_rate: float = 0.0
    computed_pass_rate: float = 0.0

    timeout_duration_ms: Optional[float] = None
    """If TIMEOUT, the duration in ms."""

    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    """List of regimes that contributed to the decision."""

    prng_draw: float = 0.0
    """The random draw that determined the decision."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema_version": "p3-noise-decision/1.0.0",
            "cycle": self.cycle,
            "item": self.item,
            "decision": self.decision.value,
            "computed_rates": {
                "timeout_rate": round(self.computed_timeout_rate, 6),
                "spurious_fail_rate": round(self.computed_fail_rate, 6),
                "spurious_pass_rate": round(self.computed_pass_rate, 6),
            },
            "timeout_duration_ms": self.timeout_duration_ms,
            "contributing_factors": self.contributing_factors,
            "prng_draw": round(self.prng_draw, 6),
        }


@dataclass
class NoiseStateSnapshot:
    """Snapshot of noise model state at a cycle."""

    cycle: int
    mode: str = "SHADOW"

    # Base state
    base_timeout_rate: float = 0.0
    base_fail_rate: float = 0.0
    base_pass_rate: float = 0.0

    # Regime states
    degradation_state: Optional[str] = None
    heat_death_resource: Optional[float] = None
    nonstationary_current_rate: Optional[float] = None
    correlated_active_factors: Optional[Dict[str, bool]] = None

    # Pathology states
    active_pathologies: List[str] = field(default_factory=list)

    # Cycle decisions summary
    total_items: int = 0
    timeout_count: int = 0
    spurious_fail_count: int = 0
    spurious_pass_count: int = 0
    clean_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema_version": "p3-noise-state/1.0.0",
            "cycle": self.cycle,
            "mode": self.mode,
            "base_state": {
                "timeout_rate": round(self.base_timeout_rate, 6),
                "spurious_fail_rate": round(self.base_fail_rate, 6),
                "spurious_pass_rate": round(self.base_pass_rate, 6),
            },
            "regime_states": {
                "degradation": {"state": self.degradation_state} if self.degradation_state else None,
                "heat_death": {"resource_level": self.heat_death_resource} if self.heat_death_resource else None,
                "nonstationary": {"current_rate": self.nonstationary_current_rate} if self.nonstationary_current_rate else None,
                "correlated": {"active_factors": self.correlated_active_factors} if self.correlated_active_factors else None,
            },
            "active_pathologies": self.active_pathologies,
            "decisions_this_cycle": {
                "total_items": self.total_items,
                "timeout_count": self.timeout_count,
                "spurious_fail_count": self.spurious_fail_count,
                "spurious_pass_count": self.spurious_pass_count,
                "clean_count": self.clean_count,
            },
        }


# =============================================================================
# P3 Noise Model Implementation
# =============================================================================

class P3NoiseModel:
    """
    P3 Noise Model implementation.

    Combines all noise regimes with deterministic PRNG for reproducibility.

    SHADOW MODE CONTRACT:
    - All decisions are observational only
    - No governance modification
    - Deterministic with seeded PRNG

    See: docs/system_law/P3_Noise_Model_Spec.md
    """

    def __init__(self, config: P3NoiseConfig, seed: int = 42) -> None:
        """
        Initialize noise model.

        Args:
            config: P3NoiseConfig with all parameters
            seed: Master seed for deterministic PRNG
        """
        config.validate_or_raise()

        self.config = config
        self.seed = seed
        self.prng = DeterministicPRNG(int_to_hex_seed(seed))

        # State variables
        self._cycle = 0
        self._degradation_state = DegradationState.HEALTHY
        self._heat_death_resource = config.heat_death.initial_resource
        self._correlated_active_factors: Dict[str, bool] = {}

        # Decision tracking for snapshots
        self._cycle_decisions: List[NoiseDecision] = []

    def step_cycle(self) -> None:
        """
        Advance to next cycle and update stateful regimes.

        Call this once per cycle before making noise decisions.
        """
        self._cycle += 1
        self._cycle_decisions = []

        # Update degradation HMM
        if self.config.degradation.enabled:
            self._step_degradation()

        # Update heat-death resource
        if self.config.heat_death.enabled:
            self._step_heat_death()

        # Update correlated factors
        if self.config.correlated.enabled:
            self._step_correlated()

    def _step_degradation(self) -> None:
        """Transition degradation HMM."""
        prng = self.prng.for_path("degradation", "transition", str(self._cycle))
        draw = prng.random()

        if self._degradation_state == DegradationState.HEALTHY:
            if draw < self.config.degradation.alpha:
                self._degradation_state = DegradationState.DEGRADED
        else:
            if draw < self.config.degradation.beta:
                self._degradation_state = DegradationState.HEALTHY

    def _step_heat_death(self) -> None:
        """Update heat-death resource level."""
        cfg = self.config.heat_death
        prng = self.prng.for_path("heat_death", "cycle", str(self._cycle))

        # Sample consumption (truncated Gaussian)
        consumption = self._sample_truncated_gaussian(
            prng.for_path("consumption"),
            cfg.consumption_mean,
            cfg.consumption_std,
            0.0,
            cfg.consumption_mean * 3,
        )

        # Sample recovery (truncated Gaussian)
        recovery = self._sample_truncated_gaussian(
            prng.for_path("recovery"),
            cfg.recovery_mean,
            cfg.recovery_std,
            0.0,
            cfg.recovery_mean * 3,
        )

        # Update resource level
        self._heat_death_resource = max(0.0, self._heat_death_resource - consumption + recovery)

    def _step_correlated(self) -> None:
        """Sample new correlated factor activations."""
        self._correlated_active_factors = {}
        prng = self.prng.for_path("correlated", "cycle", str(self._cycle))

        for factor in self.config.correlated.theta.keys():
            factor_prng = prng.for_path("factor", factor)
            self._correlated_active_factors[factor] = factor_prng.random() < self.config.correlated.rho

    def _sample_truncated_gaussian(
        self,
        prng: DeterministicPRNG,
        mu: float,
        sigma: float,
        low: float,
        high: float,
        max_attempts: int = 100,
    ) -> float:
        """Sample from truncated Gaussian distribution."""
        for _ in range(max_attempts):
            # Box-Muller transform for Gaussian
            u1 = max(prng.random(), 1e-10)
            u2 = prng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            x = mu + sigma * z
            if low <= x <= high:
                return x
        # Fallback to clamped mean
        return max(low, min(high, mu))

    def _sample_gaussian(self, prng: DeterministicPRNG, mu: float, sigma: float) -> float:
        """Sample from Gaussian distribution using Box-Muller."""
        u1 = max(prng.random(), 1e-10)
        u2 = prng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def compute_timeout_rate(self, item: str, policy_prob: Optional[float] = None) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute effective timeout rate for an item.

        Args:
            item: Item identifier
            policy_prob: Optional policy probability for adaptive noise

        Returns:
            Tuple of (rate, contributing_factors)
        """
        factors = []
        rate = self.config.base.timeout_rate
        factors.append({"regime": "base", "contribution": "base_rate", "value": rate})

        # Apply nonstationary drift
        if self.config.nonstationary.enabled:
            cfg = self.config.nonstationary
            drift = cfg.theta_0 + cfg.delta * self._cycle
            prng = self.prng.for_path("nonstationary", str(self._cycle))
            noise = self._sample_gaussian(prng, 0.0, cfg.sigma)
            rate = max(0.0, min(1.0, drift + noise))
            factors.append({"regime": "nonstationary", "contribution": "drift", "value": rate})

        # Apply adaptive scaling
        if self.config.adaptive.enabled and policy_prob is not None:
            confidence = abs(policy_prob - 0.5) * 2
            rate = rate * (1 + self.config.adaptive.gamma * confidence)
            rate = max(0.0, min(1.0, rate))
            factors.append({"regime": "adaptive", "contribution": "scaling", "value": rate})

        # Apply pathologies
        for pathology in self.config.pathologies:
            if pathology.is_active(self._cycle):
                rate = pathology.get_impact(self._cycle, rate)
                factors.append({
                    "regime": "pathology",
                    "contribution": pathology.pathology_type.value,
                    "value": rate,
                })

        return rate, factors

    def should_timeout(self, item: str, policy_prob: Optional[float] = None) -> bool:
        """
        Determine if item should timeout.

        Args:
            item: Item identifier
            policy_prob: Optional policy probability for adaptive noise

        Returns:
            True if timeout should be injected
        """
        if not self.config.enabled:
            return False

        # Check correlated failures first
        if self.config.correlated.enabled:
            item_factors = self.config.correlated.item_factors.get(item, [])
            for factor in item_factors:
                if self._correlated_active_factors.get(factor, False):
                    prng = self.prng.for_path("correlated", "item", item, str(self._cycle))
                    theta = self.config.correlated.theta.get(factor, 0.0)
                    if prng.random() < theta:
                        return True

        # Check degradation
        if self.config.degradation.enabled and self._degradation_state == DegradationState.DEGRADED:
            prng = self.prng.for_path("degradation", "item", item, str(self._cycle))
            if prng.random() < self.config.degradation.theta_degraded:
                return True

        # Check heat-death
        if self.config.heat_death.enabled:
            if self._heat_death_resource < self.config.heat_death.min_resource:
                return True

        # Sample from base + modifiers
        rate, _ = self.compute_timeout_rate(item, policy_prob)
        prng = self.prng.for_path("timeout", item, str(self._cycle))
        return prng.random() < rate

    def should_spurious_fail(self, item: str, policy_prob: Optional[float] = None) -> bool:
        """Determine if item should spuriously fail."""
        if not self.config.enabled:
            return False

        rate = self.config.base.spurious_fail_rate

        # Apply nonstationary
        if self.config.nonstationary.enabled:
            cfg = self.config.nonstationary
            drift = cfg.theta_0 + cfg.delta * self._cycle
            prng = self.prng.for_path("nonstationary", "fail", str(self._cycle))
            noise = self._sample_gaussian(prng, 0.0, cfg.sigma)
            rate = max(0.0, min(1.0, drift + noise))

        # Apply adaptive
        if self.config.adaptive.enabled and policy_prob is not None:
            confidence = abs(policy_prob - 0.5) * 2
            rate = rate * (1 + self.config.adaptive.gamma * confidence)
            rate = max(0.0, min(0.15, rate))  # P3 bound

        prng = self.prng.for_path("spurious_fail", item, str(self._cycle))
        return prng.random() < rate

    def should_spurious_pass(self, item: str, policy_prob: Optional[float] = None) -> bool:
        """Determine if item should spuriously pass."""
        if not self.config.enabled:
            return False

        rate = self.config.base.spurious_pass_rate

        # Apply adaptive
        if self.config.adaptive.enabled and policy_prob is not None:
            confidence = abs(policy_prob - 0.5) * 2
            rate = rate * (1 + self.config.adaptive.gamma * confidence)
            rate = max(0.0, min(0.10, rate))  # P3 bound

        prng = self.prng.for_path("spurious_pass", item, str(self._cycle))
        return prng.random() < rate

    def sample_timeout_duration(self, item: str) -> float:
        """
        Sample timeout duration from heavy-tail or exponential distribution.

        Args:
            item: Item identifier

        Returns:
            Timeout duration in milliseconds
        """
        prng = self.prng.for_path("timeout_duration", item, str(self._cycle))

        if self.config.heavy_tail.enabled:
            cfg = self.config.heavy_tail
            # Mixture: (1-pi)*Exp + pi*Pareto
            if prng.random() < cfg.pi:
                # Pareto: x_min / U^(1/alpha)
                u = max(prng.random(), 1e-10)
                return cfg.x_min / (u ** (1.0 / cfg.alpha))
            else:
                # Exponential: -mean * ln(U)
                u = max(prng.random(), 1e-10)
                mean_ms = 1000.0 / cfg.lambda_exp
                return -mean_ms * math.log(u)
        else:
            # Default exponential with lambda = 0.01
            u = max(prng.random(), 1e-10)
            return -100.0 * math.log(u)  # mean = 100ms

    def generate_decision(
        self,
        item: str,
        policy_prob: Optional[float] = None,
    ) -> NoiseDecision:
        """
        Generate a complete noise decision for an item.

        SHADOW MODE: This decision is observational only.

        Args:
            item: Item identifier
            policy_prob: Optional policy probability

        Returns:
            NoiseDecision with all details
        """
        timeout_rate, factors = self.compute_timeout_rate(item, policy_prob)
        fail_rate = self.config.base.spurious_fail_rate
        pass_rate = self.config.base.spurious_pass_rate

        # Determine decision type
        prng = self.prng.for_path("decision", item, str(self._cycle))
        draw = prng.random()

        decision_type = NoiseDecisionType.CLEAN
        timeout_duration = None

        if self.should_timeout(item, policy_prob):
            decision_type = NoiseDecisionType.TIMEOUT
            timeout_duration = self.sample_timeout_duration(item)
        elif self.should_spurious_fail(item, policy_prob):
            decision_type = NoiseDecisionType.SPURIOUS_FAIL
        elif self.should_spurious_pass(item, policy_prob):
            decision_type = NoiseDecisionType.SPURIOUS_PASS

        decision = NoiseDecision(
            decision=decision_type,
            cycle=self._cycle,
            item=item,
            computed_timeout_rate=timeout_rate,
            computed_fail_rate=fail_rate,
            computed_pass_rate=pass_rate,
            timeout_duration_ms=timeout_duration,
            contributing_factors=factors,
            prng_draw=draw,
        )

        self._cycle_decisions.append(decision)
        return decision

    def get_state_snapshot(self) -> NoiseStateSnapshot:
        """Get current state snapshot for logging."""
        # Count decision types
        timeout_count = sum(1 for d in self._cycle_decisions if d.decision == NoiseDecisionType.TIMEOUT)
        fail_count = sum(1 for d in self._cycle_decisions if d.decision == NoiseDecisionType.SPURIOUS_FAIL)
        pass_count = sum(1 for d in self._cycle_decisions if d.decision == NoiseDecisionType.SPURIOUS_PASS)
        clean_count = sum(1 for d in self._cycle_decisions if d.decision == NoiseDecisionType.CLEAN)

        # Active pathologies
        active_pathologies = [
            p.pathology_type.value
            for p in self.config.pathologies
            if p.is_active(self._cycle)
        ]

        return NoiseStateSnapshot(
            cycle=self._cycle,
            mode="SHADOW",
            base_timeout_rate=self.config.base.timeout_rate,
            base_fail_rate=self.config.base.spurious_fail_rate,
            base_pass_rate=self.config.base.spurious_pass_rate,
            degradation_state=self._degradation_state.value if self.config.degradation.enabled else None,
            heat_death_resource=self._heat_death_resource if self.config.heat_death.enabled else None,
            nonstationary_current_rate=self.config.nonstationary.theta_0 + self.config.nonstationary.delta * self._cycle if self.config.nonstationary.enabled else None,
            correlated_active_factors=self._correlated_active_factors if self.config.correlated.enabled else None,
            active_pathologies=active_pathologies,
            total_items=len(self._cycle_decisions),
            timeout_count=timeout_count,
            spurious_fail_count=fail_count,
            spurious_pass_count=pass_count,
            clean_count=clean_count,
        )

    def reset(self) -> None:
        """Reset model to initial state."""
        self._cycle = 0
        self._degradation_state = DegradationState.HEALTHY
        self._heat_death_resource = self.config.heat_death.initial_resource
        self._correlated_active_factors = {}
        self._cycle_decisions = []
        self.prng = DeterministicPRNG(int_to_hex_seed(self.seed))

    @property
    def cycle(self) -> int:
        """Current cycle number."""
        return self._cycle


# =============================================================================
# Factory Functions
# =============================================================================

def select_noise_model(config: Union[P3NoiseConfig, Dict[str, Any]], seed: int = 42) -> P3NoiseModel:
    """
    Factory function to create a P3NoiseModel from configuration.

    Implements: select_noise_model(config) per STRATCOM task.

    Args:
        config: P3NoiseConfig or dict with parameters
        seed: Master seed for PRNG

    Returns:
        Configured P3NoiseModel

    Raises:
        ValueError: If config validation fails
    """
    if isinstance(config, dict):
        # Build config from dict
        base_params = config.get("base", {})
        noise_config = P3NoiseConfig(
            enabled=config.get("enabled", True),
            base=BaseNoiseParams(**base_params) if base_params else BaseNoiseParams(),
            correlated=CorrelatedNoiseParams(**config.get("correlated", {})) if config.get("correlated") else CorrelatedNoiseParams(),
            degradation=DegradationParams(**config.get("degradation", {})) if config.get("degradation") else DegradationParams(),
            heat_death=HeatDeathParams(**config.get("heat_death", {})) if config.get("heat_death") else HeatDeathParams(),
            heavy_tail=HeavyTailParams(**config.get("heavy_tail", {})) if config.get("heavy_tail") else HeavyTailParams(),
            nonstationary=NonstationaryParams(**config.get("nonstationary", {})) if config.get("nonstationary") else NonstationaryParams(),
            adaptive=AdaptiveParams(**config.get("adaptive", {})) if config.get("adaptive") else AdaptiveParams(),
        )
    else:
        noise_config = config

    return P3NoiseModel(noise_config, seed=seed)


def generate_noise_sample(
    model: P3NoiseModel,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate noise contribution for a state update.

    Implements: generate_noise_sample(model, state) per STRATCOM task.

    SHADOW MODE: Deterministic with PRNG, observational only.

    Args:
        model: P3NoiseModel instance
        state: Current state dict with at least 'item' key

    Returns:
        Dict with noise contribution:
        - noise_decision: NoiseDecisionType value
        - delta_p_contribution: float adjustment to delta_p
        - timeout_duration_ms: Optional float
        - state_snapshot: NoiseStateSnapshot dict
    """
    item = state.get("item", f"item_{model.cycle}")
    policy_prob = state.get("policy_prob")

    decision = model.generate_decision(item, policy_prob)

    # Compute delta_p contribution
    # Noise that causes failures should reduce delta_p
    delta_p_contribution = 0.0
    if decision.decision == NoiseDecisionType.TIMEOUT:
        delta_p_contribution = -0.01  # Timeout hurts learning
    elif decision.decision == NoiseDecisionType.SPURIOUS_FAIL:
        delta_p_contribution = -0.005  # Spurious failure hurts less
    elif decision.decision == NoiseDecisionType.SPURIOUS_PASS:
        delta_p_contribution = 0.002  # Spurious pass slightly helps (but is dangerous)

    return {
        "noise_decision": decision.decision.value,
        "delta_p_contribution": delta_p_contribution,
        "timeout_duration_ms": decision.timeout_duration_ms,
        "computed_rates": {
            "timeout": decision.computed_timeout_rate,
            "spurious_fail": decision.computed_fail_rate,
            "spurious_pass": decision.computed_pass_rate,
        },
        "contributing_factors": decision.contributing_factors,
        "state_snapshot": model.get_state_snapshot().to_dict(),
    }


# =============================================================================
# P3 Noise Harness (Integration with SyntheticStateGenerator)
# =============================================================================

class P3NoiseHarness:
    """
    Harness integrating P3NoiseModel with SyntheticStateGenerator.

    Provides:
    - Noise contribution to synthetic state
    - Delta-p noise adjustment
    - Stability report exposure

    SHADOW MODE CONTRACT:
    - All operations are observational only
    - No governance modification
    - Deterministic with seeded PRNG
    """

    def __init__(
        self,
        noise_config: Optional[P3NoiseConfig] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize noise harness.

        Args:
            noise_config: P3 noise configuration (default: shadow profile)
            seed: Master seed for PRNG
        """
        self.config = noise_config or P3NoiseConfig.default_shadow()
        self.model = select_noise_model(self.config, seed)
        self.seed = seed

        # Tracking for stability report
        self._noise_snapshots: List[NoiseStateSnapshot] = []
        self._decisions: List[NoiseDecision] = []
        self._delta_p_contributions: List[float] = []

    def step_cycle(self) -> None:
        """Advance noise model to next cycle."""
        self.model.step_cycle()

    def apply_noise(
        self,
        state: Dict[str, Any],
        item: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply noise to a state update and return modified state.

        SHADOW MODE: Returns noise-adjusted state for observation.

        Args:
            state: Current state dict from SyntheticStateGenerator
            item: Optional item identifier

        Returns:
            State dict with noise contribution added
        """
        item = item or f"item_{self.model.cycle}"
        state["item"] = item

        # Generate noise sample
        noise_result = generate_noise_sample(self.model, state)

        # Track for stability report
        snapshot = self.model.get_state_snapshot()
        self._noise_snapshots.append(snapshot)
        self._delta_p_contributions.append(noise_result["delta_p_contribution"])

        # Add noise contribution to state
        state["noise"] = {
            "decision": noise_result["noise_decision"],
            "delta_p_contribution": noise_result["delta_p_contribution"],
            "timeout_duration_ms": noise_result["timeout_duration_ms"],
            "rates": noise_result["computed_rates"],
        }

        # Modify success based on noise
        if noise_result["noise_decision"] == "TIMEOUT":
            state["success"] = False
            state["noise_caused_failure"] = True
        elif noise_result["noise_decision"] == "SPURIOUS_FAIL":
            state["success"] = False
            state["noise_caused_failure"] = True
        elif noise_result["noise_decision"] == "SPURIOUS_PASS":
            state["success"] = True
            state["noise_caused_success"] = True

        return state

    def get_delta_p_noise_contribution(self, window_size: int = 50) -> float:
        """
        Get cumulative delta_p contribution from noise over a window.

        Args:
            window_size: Number of recent cycles to consider

        Returns:
            Sum of delta_p contributions from noise
        """
        recent = self._delta_p_contributions[-window_size:]
        return sum(recent)

    def get_stability_report(self) -> Dict[str, Any]:
        """
        Generate stability report for noise effects.

        SHADOW MODE: Observational summary only.

        Returns:
            Dict with noise stability metrics
        """
        if not self._noise_snapshots:
            return {
                "total_cycles": 0,
                "noise_impact": "NONE",
                "recommendation": "No data",
            }

        # Compute summary statistics
        total_cycles = len(self._noise_snapshots)
        total_timeouts = sum(s.timeout_count for s in self._noise_snapshots)
        total_spurious_fails = sum(s.spurious_fail_count for s in self._noise_snapshots)
        total_spurious_passes = sum(s.spurious_pass_count for s in self._noise_snapshots)
        total_clean = sum(s.clean_count for s in self._noise_snapshots)
        total_decisions = total_timeouts + total_spurious_fails + total_spurious_passes + total_clean

        # Compute rates
        timeout_rate = total_timeouts / max(1, total_decisions)
        fail_rate = total_spurious_fails / max(1, total_decisions)
        pass_rate = total_spurious_passes / max(1, total_decisions)
        clean_rate = total_clean / max(1, total_decisions)

        # Compute delta_p impact
        total_delta_p_contribution = sum(self._delta_p_contributions)
        avg_delta_p_contribution = total_delta_p_contribution / max(1, total_cycles)

        # Determine impact level
        if timeout_rate + fail_rate > 0.20:
            impact = "HIGH"
            recommendation = "Consider reducing noise rates or shortening experiment"
        elif timeout_rate + fail_rate > 0.10:
            impact = "MODERATE"
            recommendation = "Noise within expected bounds but monitor closely"
        else:
            impact = "LOW"
            recommendation = "Noise contribution acceptable"

        return {
            "schema_version": "p3-stability-report/1.0.0",
            "mode": "SHADOW",
            "total_cycles": total_cycles,
            "total_decisions": total_decisions,
            "rates": {
                "timeout": round(timeout_rate, 4),
                "spurious_fail": round(fail_rate, 4),
                "spurious_pass": round(pass_rate, 4),
                "clean": round(clean_rate, 4),
            },
            "counts": {
                "timeout": total_timeouts,
                "spurious_fail": total_spurious_fails,
                "spurious_pass": total_spurious_passes,
                "clean": total_clean,
            },
            "delta_p_impact": {
                "total_contribution": round(total_delta_p_contribution, 6),
                "avg_per_cycle": round(avg_delta_p_contribution, 6),
            },
            "noise_impact": impact,
            "recommendation": recommendation,
            "config_summary": {
                "enabled": self.config.enabled,
                "base_timeout_rate": self.config.base.timeout_rate,
                "base_fail_rate": self.config.base.spurious_fail_rate,
                "degradation_enabled": self.config.degradation.enabled,
                "heavy_tail_enabled": self.config.heavy_tail.enabled,
                "pathology_count": len(self.config.pathologies),
            },
        }

    def reset(self) -> None:
        """Reset harness and model."""
        self.model.reset()
        self._noise_snapshots = []
        self._decisions = []
        self._delta_p_contributions = []


# =============================================================================
# P3 Noise Summary Builder (for Stability Report Integration)
# =============================================================================

def build_noise_summary_for_p3(noise_harness: P3NoiseHarness) -> Dict[str, Any]:
    """
    Build comprehensive noise summary for P3 First-Light stability report.

    Implements: build_noise_summary_for_p3(noise_harness) per STRATCOM task.

    Returns:
        - Proportions of each noise regime activated
        - Aggregate effect on delta_p and RSI
        - Suitable for attachment under stability_report["noise_summary"]

    SHADOW MODE: Observational only, no governance modification.

    Args:
        noise_harness: P3NoiseHarness instance with collected noise data

    Returns:
        Dict with noise summary suitable for stability_report["noise_summary"]
    """
    if not noise_harness._noise_snapshots:
        return {
            "schema_version": "p3-noise-summary/1.0.0",
            "mode": "SHADOW",
            "total_cycles": 0,
            "regime_proportions": {},
            "delta_p_aggregate": {},
            "rsi_aggregate": {},
            "interpretation_guidance": "No noise data collected",
        }

    config = noise_harness.config
    model = noise_harness.model
    snapshots = noise_harness._noise_snapshots
    total_cycles = len(snapshots)

    # -------------------------------------------------------------------------
    # Regime Activation Proportions
    # -------------------------------------------------------------------------
    regime_activations = {
        "base": total_cycles,  # Base is always active
        "correlated": 0,
        "degradation_healthy": 0,
        "degradation_degraded": 0,
        "heat_death_nominal": 0,
        "heat_death_stressed": 0,
        "heavy_tail": 0,
        "nonstationary": 0,
        "adaptive": 0,
        "pathology": 0,
    }

    # Track regime-specific metrics
    degraded_cycles = 0
    heat_death_stressed_cycles = 0
    pathology_active_cycles = 0

    for snapshot in snapshots:
        # Correlated: check if any factors were active
        if snapshot.correlated_active_factors:
            if any(snapshot.correlated_active_factors.values()):
                regime_activations["correlated"] += 1

        # Degradation: track state
        if snapshot.degradation_state:
            if snapshot.degradation_state == "DEGRADED":
                regime_activations["degradation_degraded"] += 1
                degraded_cycles += 1
            else:
                regime_activations["degradation_healthy"] += 1

        # Heat-death: check resource level
        if snapshot.heat_death_resource is not None:
            if snapshot.heat_death_resource < config.heat_death.min_resource * 1.5:
                regime_activations["heat_death_stressed"] += 1
                heat_death_stressed_cycles += 1
            else:
                regime_activations["heat_death_nominal"] += 1

        # Heavy-tail: enabled throughout if configured
        if config.heavy_tail.enabled:
            regime_activations["heavy_tail"] += 1

        # Non-stationary: enabled throughout if configured
        if config.nonstationary.enabled:
            regime_activations["nonstationary"] += 1

        # Adaptive: enabled throughout if configured
        if config.adaptive.enabled:
            regime_activations["adaptive"] += 1

        # Pathology: check if any active
        if snapshot.active_pathologies:
            regime_activations["pathology"] += 1
            pathology_active_cycles += 1

    # Compute proportions
    regime_proportions = {
        regime: round(count / max(1, total_cycles), 4)
        for regime, count in regime_activations.items()
    }

    # -------------------------------------------------------------------------
    # Delta-p Aggregate Effects
    # -------------------------------------------------------------------------
    delta_p_contributions = noise_harness._delta_p_contributions
    total_delta_p = sum(delta_p_contributions)
    avg_delta_p = total_delta_p / max(1, total_cycles)

    # Compute delta_p by noise type
    timeout_count = sum(s.timeout_count for s in snapshots)
    fail_count = sum(s.spurious_fail_count for s in snapshots)
    pass_count = sum(s.spurious_pass_count for s in snapshots)

    # Estimated delta_p by type (using default contributions)
    delta_p_from_timeout = timeout_count * (-0.01)
    delta_p_from_fail = fail_count * (-0.005)
    delta_p_from_pass = pass_count * 0.002

    delta_p_aggregate = {
        "total_contribution": round(total_delta_p, 6),
        "avg_per_cycle": round(avg_delta_p, 6),
        "by_noise_type": {
            "timeout": round(delta_p_from_timeout, 6),
            "spurious_fail": round(delta_p_from_fail, 6),
            "spurious_pass": round(delta_p_from_pass, 6),
        },
        "net_direction": "NEGATIVE" if total_delta_p < -0.001 else ("POSITIVE" if total_delta_p > 0.001 else "NEUTRAL"),
        "magnitude_class": _classify_delta_p_magnitude(total_delta_p, total_cycles),
    }

    # -------------------------------------------------------------------------
    # RSI Aggregate Effects
    # -------------------------------------------------------------------------
    # RSI is affected by noise through success rate impact
    # More noise failures -> lower success rate -> lower RSI trend
    total_noise_events = timeout_count + fail_count
    noise_event_rate = total_noise_events / max(1, total_cycles)

    # Estimate RSI suppression factor
    # Noise events reduce effective success rate, which indirectly affects RSI
    rsi_suppression_estimate = noise_event_rate * 0.1  # ~10% RSI impact per noise event rate

    rsi_aggregate = {
        "noise_event_rate": round(noise_event_rate, 4),
        "estimated_rsi_suppression": round(rsi_suppression_estimate, 4),
        "suppression_class": _classify_rsi_suppression(rsi_suppression_estimate),
        "degraded_cycle_fraction": round(degraded_cycles / max(1, total_cycles), 4),
        "pathology_active_fraction": round(pathology_active_cycles / max(1, total_cycles), 4),
    }

    # -------------------------------------------------------------------------
    # Interpretation Guidance
    # -------------------------------------------------------------------------
    guidance_parts = []

    if delta_p_aggregate["magnitude_class"] == "HIGH":
        guidance_parts.append("Significant delta_p impact from noise injection.")
    elif delta_p_aggregate["magnitude_class"] == "MODERATE":
        guidance_parts.append("Moderate delta_p impact from noise injection.")
    else:
        guidance_parts.append("Minimal delta_p impact from noise injection.")

    if rsi_aggregate["suppression_class"] == "HIGH":
        guidance_parts.append("RSI trajectory significantly suppressed by noise.")
    elif rsi_aggregate["suppression_class"] == "MODERATE":
        guidance_parts.append("RSI trajectory moderately affected by noise.")

    if regime_proportions.get("degradation_degraded", 0) > 0.20:
        guidance_parts.append(f"Degradation regime active {regime_proportions['degradation_degraded']*100:.1f}% of cycles.")

    if regime_proportions.get("pathology", 0) > 0.10:
        guidance_parts.append(f"Pathology injection active {regime_proportions['pathology']*100:.1f}% of cycles.")

    interpretation_guidance = " ".join(guidance_parts) if guidance_parts else "Noise contribution within normal bounds."

    # -------------------------------------------------------------------------
    # Build Final Summary
    # -------------------------------------------------------------------------
    return {
        "schema_version": "p3-noise-summary/1.0.0",
        "mode": "SHADOW",
        "total_cycles": total_cycles,
        "regime_proportions": regime_proportions,
        "delta_p_aggregate": delta_p_aggregate,
        "rsi_aggregate": rsi_aggregate,
        "config_profile": {
            "base_timeout_rate": config.base.timeout_rate,
            "base_fail_rate": config.base.spurious_fail_rate,
            "base_pass_rate": config.base.spurious_pass_rate,
            "correlated_enabled": config.correlated.enabled,
            "degradation_enabled": config.degradation.enabled,
            "heat_death_enabled": config.heat_death.enabled,
            "heavy_tail_enabled": config.heavy_tail.enabled,
            "nonstationary_enabled": config.nonstationary.enabled,
            "adaptive_enabled": config.adaptive.enabled,
            "pathology_count": len(config.pathologies),
        },
        "interpretation_guidance": interpretation_guidance,
    }


def _classify_delta_p_magnitude(total_delta_p: float, total_cycles: int) -> str:
    """Classify delta_p magnitude for interpretation."""
    if total_cycles == 0:
        return "NONE"
    normalized = abs(total_delta_p) / total_cycles
    if normalized > 0.005:
        return "HIGH"
    elif normalized > 0.002:
        return "MODERATE"
    else:
        return "LOW"


def _classify_rsi_suppression(suppression: float) -> str:
    """Classify RSI suppression level for interpretation."""
    if suppression > 0.02:
        return "HIGH"
    elif suppression > 0.01:
        return "MODERATE"
    else:
        return "LOW"


# =============================================================================
# Evidence Pack Integration
# =============================================================================

def attach_noise_to_evidence(
    evidence: Dict[str, Any],
    noise_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach noise summary to evidence pack under governance section.

    Implements: attach_noise_to_evidence(evidence, noise_summary) per STRATCOM task.

    SHADOW MODE: Observational attachment only.

    Args:
        evidence: Evidence pack dict (modified in place)
        noise_summary: Output from build_noise_summary_for_p3()

    Returns:
        Modified evidence dict with noise_summary attached at evidence["governance"]["noise"]
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Attach noise summary
    evidence["governance"]["noise"] = {
        "schema_version": noise_summary.get("schema_version", "p3-noise-summary/1.0.0"),
        "mode": noise_summary.get("mode", "SHADOW"),
        "total_cycles": noise_summary.get("total_cycles", 0),
        "regime_proportions": noise_summary.get("regime_proportions", {}),
        "delta_p_aggregate": noise_summary.get("delta_p_aggregate", {}),
        "rsi_aggregate": noise_summary.get("rsi_aggregate", {}),
        "interpretation_guidance": noise_summary.get("interpretation_guidance", ""),
        "config_profile": noise_summary.get("config_profile", {}),
    }

    # Add noise impact assessment to governance advisories if present
    if "advisories" not in evidence["governance"]:
        evidence["governance"]["advisories"] = []

    # Generate advisory based on noise impact
    delta_p_impact = noise_summary.get("delta_p_aggregate", {})
    rsi_impact = noise_summary.get("rsi_aggregate", {})

    if delta_p_impact.get("magnitude_class") == "HIGH":
        evidence["governance"]["advisories"].append({
            "type": "NOISE_IMPACT",
            "severity": "WARN",
            "message": "High noise impact on delta_p trajectory observed",
            "details": {
                "total_delta_p": delta_p_impact.get("total_contribution", 0),
                "direction": delta_p_impact.get("net_direction", "UNKNOWN"),
            },
        })
    elif delta_p_impact.get("magnitude_class") == "MODERATE":
        evidence["governance"]["advisories"].append({
            "type": "NOISE_IMPACT",
            "severity": "INFO",
            "message": "Moderate noise impact on delta_p trajectory",
            "details": {
                "total_delta_p": delta_p_impact.get("total_contribution", 0),
            },
        })

    if rsi_impact.get("suppression_class") == "HIGH":
        evidence["governance"]["advisories"].append({
            "type": "RSI_SUPPRESSION",
            "severity": "WARN",
            "message": "RSI trajectory significantly suppressed by noise injection",
            "details": {
                "suppression_estimate": rsi_impact.get("estimated_rsi_suppression", 0),
                "noise_event_rate": rsi_impact.get("noise_event_rate", 0),
            },
        })

    return evidence
