# Task C3: UnifiedNoiseModel — Complete Code Skeleton

"""
Unified Noise Model for Verifier Imperfection Modeling

This module implements the UnifiedNoiseModel that combines all six advanced noise regimes:
1. Correlated Failures
2. Cluster-Based Degradation
3. Heat-Death Scenarios
4. High-Tactic-Depth Tails
5. Non-Stationary Noise
6. Policy-Aware Adaptive Noise

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Code Skeleton Ready for Implementation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
from backend.verification.noise_sampler import DeterministicPRNG
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


# ============================================================================
# Base Noise Configuration
# ============================================================================

@dataclass
class BaseNoiseConfig:
    """Base noise configuration with independent Bernoulli noise."""
    
    timeout_rate: float = 0.0
    spurious_fail_rate: float = 0.0
    spurious_pass_rate: float = 0.0
    
    def validate(self) -> None:
        """Validate noise rates are in [0, 1]."""
        assert 0.0 <= self.timeout_rate <= 1.0, "timeout_rate must be in [0, 1]"
        assert 0.0 <= self.spurious_fail_rate <= 1.0, "spurious_fail_rate must be in [0, 1]"
        assert 0.0 <= self.spurious_pass_rate <= 1.0, "spurious_pass_rate must be in [0, 1]"


# ============================================================================
# Regime 1: Correlated Failures
# ============================================================================

@dataclass
class CorrelatedNoiseConfig:
    """Configuration for correlated failure noise regime."""
    
    enabled: bool = False
    rho: float = 0.1  # Factor activation probability
    theta: Dict[str, float] = field(default_factory=dict)  # Base failure rate per factor
    item_factors: Dict[str, List[str]] = field(default_factory=dict)  # Item → factors mapping


@dataclass
class CorrelatedNoiseModel:
    """Latent factor model for correlated failures.
    
    Model: P(fail_i | z) = 1 - ∏_{k: A_ik = 1} (1 - z_k * θ_k)
    where z_k ~ Bernoulli(ρ) are latent failure factors.
    """
    
    config: CorrelatedNoiseConfig
    prng: DeterministicPRNG
    
    # State: active factors for current cycle
    active_factors: Dict[str, bool] = field(default_factory=dict)
    cycle_count: int = 0
    
    def step_cycle(self) -> None:
        """Sample new active factors for next cycle."""
        self.cycle_count += 1
        self.active_factors = {}
        
        prng_factors = self.prng.for_path("correlated", "cycle", str(self.cycle_count))
        
        for factor in self.config.theta.keys():
            prng_factor = prng_factors.for_path("factor", factor)
            self.active_factors[factor] = prng_factor.random() < self.config.rho
    
    def should_fail(self, item: str) -> bool:
        """Determine if item should fail due to correlated factors.
        
        Args:
            item: Item identifier
        
        Returns:
            True if item should fail, False otherwise
        """
        if not self.config.enabled:
            return False
        
        factors = self.config.item_factors.get(item, [])
        if not factors:
            return False
        
        # Compute failure probability
        prob_success = 1.0
        for factor in factors:
            if self.active_factors.get(factor, False):
                prob_success *= (1 - self.config.theta[factor])
        
        prob_fail = 1 - prob_success
        
        # Sample failure decision
        prng_item = self.prng.for_path("correlated", "item", item, str(self.cycle_count))
        return prng_item.random() < prob_fail


# ============================================================================
# Regime 2: Cluster-Based Degradation
# ============================================================================

class DegradationState(Enum):
    """Degradation state for cluster-based degradation."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass
class ClusterDegradationConfig:
    """Configuration for cluster-based degradation noise regime."""
    
    enabled: bool = False
    alpha: float = 0.05  # Transition probability HEALTHY → DEGRADED
    beta: float = 0.2    # Transition probability DEGRADED → HEALTHY
    theta_healthy: float = 0.01  # Failure rate in HEALTHY state
    theta_degraded: float = 0.3   # Failure rate in DEGRADED state


@dataclass
class ClusterDegradationModel:
    """Hidden Markov Model for cluster-based degradation.
    
    States: {HEALTHY, DEGRADED}
    Transition: P(HEALTHY → DEGRADED) = α, P(DEGRADED → HEALTHY) = β
    Emission: P(fail | HEALTHY) = θ_healthy, P(fail | DEGRADED) = θ_degraded
    """
    
    config: ClusterDegradationConfig
    prng: DeterministicPRNG
    
    # State
    state: DegradationState = DegradationState.HEALTHY
    cycle_count: int = 0
    
    def step_cycle(self) -> None:
        """Transition to next state."""
        self.cycle_count += 1
        
        prng_transition = self.prng.for_path("degradation", "transition", str(self.cycle_count))
        
        if self.state == DegradationState.HEALTHY:
            if prng_transition.random() < self.config.alpha:
                self.state = DegradationState.DEGRADED
        else:  # DEGRADED
            if prng_transition.random() < self.config.beta:
                self.state = DegradationState.HEALTHY
    
    def should_fail(self, item: str) -> bool:
        """Determine if item should fail based on current degradation state.
        
        Args:
            item: Item identifier
        
        Returns:
            True if item should fail, False otherwise
        """
        if not self.config.enabled:
            return False
        
        # Get failure rate for current state
        if self.state == DegradationState.HEALTHY:
            fail_rate = self.config.theta_healthy
        else:
            fail_rate = self.config.theta_degraded
        
        # Sample failure decision
        prng_item = self.prng.for_path("degradation", "item", item, str(self.cycle_count))
        return prng_item.random() < fail_rate


# ============================================================================
# Regime 3: Heat-Death Scenarios
# ============================================================================

@dataclass
class HeatDeathConfig:
    """Configuration for heat-death (resource exhaustion) noise regime."""
    
    enabled: bool = False
    initial_resource: float = 1000.0  # Initial resource level
    min_resource: float = 100.0       # Minimum resource before failure
    consumption_mean: float = 10.0    # Mean resource consumption per cycle
    consumption_std: float = 5.0      # Std dev of consumption
    recovery_mean: float = 8.0        # Mean resource recovery per cycle
    recovery_std: float = 3.0         # Std dev of recovery


@dataclass
class HeatDeathModel:
    """Resource depletion process for heat-death scenarios.
    
    Model: R(t+1) = R(t) - c(t) + r(t)
    where c(t) ~ N(μ_c, σ_c²) and r(t) ~ N(μ_r, σ_r²)
    Failure: fail(t) = 1 if R(t) < R_min
    """
    
    config: HeatDeathConfig
    prng: DeterministicPRNG
    
    # State
    resource_level: float = field(default_factory=lambda: 0.0)
    cycle_count: int = 0
    
    def __post_init__(self):
        """Initialize resource level."""
        self.resource_level = self.config.initial_resource
    
    def step_cycle(self) -> None:
        """Update resource level for next cycle."""
        self.cycle_count += 1
        
        prng_cycle = self.prng.for_path("heat_death", "cycle", str(self.cycle_count))
        
        # Sample consumption
        prng_consumption = prng_cycle.for_path("consumption")
        consumption = prng_consumption.gauss(
            self.config.consumption_mean,
            self.config.consumption_std,
        )
        consumption = max(0.0, consumption)  # Non-negative
        
        # Sample recovery
        prng_recovery = prng_cycle.for_path("recovery")
        recovery = prng_recovery.gauss(
            self.config.recovery_mean,
            self.config.recovery_std,
        )
        recovery = max(0.0, recovery)  # Non-negative
        
        # Update resource level
        self.resource_level = self.resource_level - consumption + recovery
        self.resource_level = max(0.0, self.resource_level)  # Non-negative
    
    def should_fail(self, item: str) -> bool:
        """Determine if item should fail due to resource exhaustion.
        
        Args:
            item: Item identifier
        
        Returns:
            True if resource exhausted, False otherwise
        """
        if not self.config.enabled:
            return False
        
        return self.resource_level < self.config.min_resource


# ============================================================================
# Regime 4: High-Tactic-Depth Tails
# ============================================================================

@dataclass
class HeavyTailConfig:
    """Configuration for heavy-tailed timeout distribution."""
    
    enabled: bool = False
    pi: float = 0.1          # Mixing probability for Pareto component
    lambda_fast: float = 0.1  # Rate parameter for exponential component
    alpha: float = 1.5        # Tail index for Pareto component
    x_min: float = 100.0      # Scale parameter for Pareto component


@dataclass
class HeavyTailTimeoutModel:
    """Mixture distribution for heavy-tailed timeouts.
    
    Model: T ~ (1 - π) * Exp(λ) + π * Pareto(α, x_min)
    """
    
    config: HeavyTailConfig
    prng: DeterministicPRNG
    
    def sample_timeout_duration(self, item: str, cycle: int) -> float:
        """Sample timeout duration from heavy-tailed distribution.
        
        Args:
            item: Item identifier
            cycle: Cycle number
        
        Returns:
            Timeout duration in milliseconds
        """
        if not self.config.enabled:
            # Fallback to exponential
            prng_item = self.prng.for_path("timeout_duration", item, str(cycle))
            return prng_item.expovariate(self.config.lambda_fast)
        
        prng_item = self.prng.for_path("heavy_tail", item, str(cycle))
        
        # Sample component (exponential or Pareto)
        prng_component = prng_item.for_path("component")
        use_pareto = prng_component.random() < self.config.pi
        
        if use_pareto:
            # Sample from Pareto
            prng_pareto = prng_item.for_path("pareto")
            u = prng_pareto.random()
            duration = self.config.x_min / (u ** (1 / self.config.alpha))
        else:
            # Sample from exponential
            prng_exp = prng_item.for_path("exponential")
            duration = prng_exp.expovariate(self.config.lambda_fast)
        
        return duration


# ============================================================================
# Regime 5: Non-Stationary Noise
# ============================================================================

@dataclass
class NonStationaryConfig:
    """Configuration for non-stationary noise regime."""
    
    enabled: bool = False
    theta_0: float = 0.1    # Initial noise rate
    delta: float = 0.0001   # Drift rate (linear trend)
    sigma: float = 0.01     # Noise standard deviation


@dataclass
class NonStationaryNoiseModel:
    """Time-varying noise parameters for non-stationary noise.
    
    Model: θ(t) = θ_0 + δ * t + ε(t), ε(t) ~ N(0, σ²)
    """
    
    config: NonStationaryConfig
    prng: DeterministicPRNG
    
    def get_noise_rate(self, cycle: int) -> float:
        """Compute noise rate at given cycle.
        
        Args:
            cycle: Cycle number
        
        Returns:
            Noise rate in [0, 1]
        """
        if not self.config.enabled:
            return self.config.theta_0
        
        # Linear drift
        drift = self.config.theta_0 + self.config.delta * cycle
        
        # Add Gaussian noise
        prng_cycle = self.prng.for_path("nonstationary", "cycle", str(cycle))
        noise = prng_cycle.gauss(0.0, self.config.sigma)
        
        # Clamp to [0, 1]
        rate = drift + noise
        rate = max(0.0, min(1.0, rate))
        
        return rate


# ============================================================================
# Regime 6: Policy-Aware Adaptive Noise
# ============================================================================

@dataclass
class AdaptiveNoiseConfig:
    """Configuration for policy-aware adaptive noise regime."""
    
    enabled: bool = False
    gamma: float = 0.5  # Adaptation strength


@dataclass
class AdaptiveNoiseModel:
    """Policy-aware adaptive noise that adjusts to RFL policy confidence.
    
    Model: θ_adaptive(item, π) = θ_base * (1 + γ * confidence(item, π))
    where confidence(item, π) = |π(item) - 0.5| * 2
    """
    
    config: AdaptiveNoiseConfig
    prng: DeterministicPRNG
    
    def get_noise_rate(
        self,
        base_rate: float,
        policy_prob: float,
    ) -> float:
        """Compute adaptive noise rate based on policy confidence.
        
        Args:
            base_rate: Base noise rate
            policy_prob: Policy probability for item (0 to 1)
        
        Returns:
            Adaptive noise rate
        """
        if not self.config.enabled:
            return base_rate
        
        # Compute confidence (distance from uniform)
        confidence = abs(policy_prob - 0.5) * 2
        
        # Scale noise rate
        adaptive_rate = base_rate * (1 + self.config.gamma * confidence)
        
        # Clamp to [0, 1]
        adaptive_rate = max(0.0, min(1.0, adaptive_rate))
        
        return adaptive_rate


# ============================================================================
# Unified Noise Model
# ============================================================================

@dataclass
class UnifiedNoiseConfig:
    """Configuration for unified noise model combining all regimes."""
    
    base_noise: BaseNoiseConfig = field(default_factory=BaseNoiseConfig)
    correlated: CorrelatedNoiseConfig = field(default_factory=CorrelatedNoiseConfig)
    degradation: ClusterDegradationConfig = field(default_factory=ClusterDegradationConfig)
    heat_death: HeatDeathConfig = field(default_factory=HeatDeathConfig)
    heavy_tail: HeavyTailConfig = field(default_factory=HeavyTailConfig)
    nonstationary: NonStationaryConfig = field(default_factory=NonStationaryConfig)
    adaptive: AdaptiveNoiseConfig = field(default_factory=AdaptiveNoiseConfig)
    
    def validate(self) -> None:
        """Validate configuration."""
        self.base_noise.validate()


@dataclass
class UnifiedNoiseModel:
    """Unified noise model combining all six advanced noise regimes.
    
    This model orchestrates all noise regimes and provides a unified interface
    for noise injection decisions.
    
    Regimes:
    1. Base Noise: Independent Bernoulli noise (timeout, spurious fail/pass)
    2. Correlated Failures: Latent factor model with spatial correlation
    3. Cluster Degradation: HMM with HEALTHY/DEGRADED states
    4. Heat-Death: Resource depletion process
    5. Heavy-Tailed Timeouts: Mixture distribution with Pareto tails
    6. Non-Stationary Noise: Time-varying parameters
    7. Adaptive Noise: Policy-aware adjustment
    
    Usage:
        config = UnifiedNoiseConfig(...)
        model = UnifiedNoiseModel(config, master_seed=12345)
        
        # For each cycle
        model.step_cycle()
        
        # For each item
        should_timeout = model.should_timeout(item, meta)
        if should_timeout:
            duration = model.sample_timeout_duration(item)
    """
    
    config: UnifiedNoiseConfig
    master_seed: int
    
    # Sub-models (initialized in __post_init__)
    prng: DeterministicPRNG = field(init=False)
    correlated_model: Optional[CorrelatedNoiseModel] = field(init=False, default=None)
    degradation_model: Optional[ClusterDegradationModel] = field(init=False, default=None)
    heat_death_model: Optional[HeatDeathModel] = field(init=False, default=None)
    heavy_tail_model: Optional[HeavyTailTimeoutModel] = field(init=False, default=None)
    nonstationary_model: Optional[NonStationaryNoiseModel] = field(init=False, default=None)
    adaptive_model: Optional[AdaptiveNoiseModel] = field(init=False, default=None)
    
    # State
    cycle_count: int = 0
    
    def __post_init__(self):
        """Initialize sub-models."""
        # Validate configuration
        self.config.validate()
        
        # Initialize master PRNG
        self.prng = DeterministicPRNG(self._int_to_hex_seed(self.master_seed))
        
        # Initialize sub-models
        if self.config.correlated.enabled:
            self.correlated_model = CorrelatedNoiseModel(
                config=self.config.correlated,
                prng=self.prng.for_path("correlated_model"),
            )
        
        if self.config.degradation.enabled:
            self.degradation_model = ClusterDegradationModel(
                config=self.config.degradation,
                prng=self.prng.for_path("degradation_model"),
            )
        
        if self.config.heat_death.enabled:
            self.heat_death_model = HeatDeathModel(
                config=self.config.heat_death,
                prng=self.prng.for_path("heat_death_model"),
            )
        
        if self.config.heavy_tail.enabled:
            self.heavy_tail_model = HeavyTailTimeoutModel(
                config=self.config.heavy_tail,
                prng=self.prng.for_path("heavy_tail_model"),
            )
        
        if self.config.nonstationary.enabled:
            self.nonstationary_model = NonStationaryNoiseModel(
                config=self.config.nonstationary,
                prng=self.prng.for_path("nonstationary_model"),
            )
        
        if self.config.adaptive.enabled:
            self.adaptive_model = AdaptiveNoiseModel(
                config=self.config.adaptive,
                prng=self.prng.for_path("adaptive_model"),
            )
    
    def step_cycle(self) -> None:
        """Advance to next cycle and update all stateful sub-models.
        
        This should be called once per cycle (e.g., at the start of each RFL update).
        """
        self.cycle_count += 1
        
        # Step all stateful sub-models
        if self.correlated_model:
            self.correlated_model.step_cycle()
        
        if self.degradation_model:
            self.degradation_model.step_cycle()
        
        if self.heat_death_model:
            self.heat_death_model.step_cycle()
    
    def should_timeout(
        self,
        item: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if item should timeout.
        
        Args:
            item: Item identifier (e.g., module name)
            meta: Optional metadata (e.g., policy_prob for adaptive noise)
        
        Returns:
            True if timeout should be injected, False otherwise
        """
        meta = meta or {}
        
        # Start with base timeout rate
        timeout_rate = self.config.base_noise.timeout_rate
        
        # Apply non-stationary drift
        if self.nonstationary_model:
            timeout_rate = self.nonstationary_model.get_noise_rate(self.cycle_count)
        
        # Apply adaptive adjustment
        if self.adaptive_model and "policy_prob" in meta:
            timeout_rate = self.adaptive_model.get_noise_rate(
                timeout_rate,
                meta["policy_prob"],
            )
        
        # Check correlated failures
        if self.correlated_model and self.correlated_model.should_fail(item):
            return True
        
        # Check cluster degradation
        if self.degradation_model and self.degradation_model.should_fail(item):
            return True
        
        # Check heat death
        if self.heat_death_model and self.heat_death_model.should_fail(item):
            return True
        
        # Sample base timeout decision
        prng_item = self.prng.for_path("timeout", item, str(self.cycle_count))
        return prng_item.random() < timeout_rate
    
    def should_spurious_fail(
        self,
        item: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if item should spuriously fail.
        
        Args:
            item: Item identifier
            meta: Optional metadata
        
        Returns:
            True if spurious failure should be injected, False otherwise
        """
        meta = meta or {}
        
        # Start with base spurious fail rate
        fail_rate = self.config.base_noise.spurious_fail_rate
        
        # Apply non-stationary drift
        if self.nonstationary_model:
            fail_rate = self.nonstationary_model.get_noise_rate(self.cycle_count)
        
        # Apply adaptive adjustment
        if self.adaptive_model and "policy_prob" in meta:
            fail_rate = self.adaptive_model.get_noise_rate(
                fail_rate,
                meta["policy_prob"],
            )
        
        # Sample spurious fail decision
        prng_item = self.prng.for_path("spurious_fail", item, str(self.cycle_count))
        return prng_item.random() < fail_rate
    
    def should_spurious_pass(
        self,
        item: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if item should spuriously pass.
        
        Args:
            item: Item identifier
            meta: Optional metadata
        
        Returns:
            True if spurious pass should be injected, False otherwise
        """
        meta = meta or {}
        
        # Start with base spurious pass rate
        pass_rate = self.config.base_noise.spurious_pass_rate
        
        # Apply non-stationary drift
        if self.nonstationary_model:
            pass_rate = self.nonstationary_model.get_noise_rate(self.cycle_count)
        
        # Apply adaptive adjustment
        if self.adaptive_model and "policy_prob" in meta:
            pass_rate = self.adaptive_model.get_noise_rate(
                pass_rate,
                meta["policy_prob"],
            )
        
        # Sample spurious pass decision
        prng_item = self.prng.for_path("spurious_pass", item, str(self.cycle_count))
        return prng_item.random() < pass_rate
    
    def sample_timeout_duration(
        self,
        item: str,
    ) -> float:
        """Sample timeout duration from distribution.
        
        Args:
            item: Item identifier
        
        Returns:
            Timeout duration in milliseconds
        """
        # Use heavy-tail model if enabled
        if self.heavy_tail_model:
            return self.heavy_tail_model.sample_timeout_duration(item, self.cycle_count)
        
        # Fallback to exponential distribution
        prng_item = self.prng.for_path("timeout_duration", item, str(self.cycle_count))
        lambda_rate = 0.01  # Default rate
        return prng_item.expovariate(lambda_rate)
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current noise model state for logging.
        
        Returns:
            Dict with state of all sub-models
        """
        snapshot = {
            "cycle_count": self.cycle_count,
        }
        
        if self.correlated_model:
            snapshot["correlated"] = {
                "active_factors": self.correlated_model.active_factors,
                "cycle_count": self.correlated_model.cycle_count,
            }
        
        if self.degradation_model:
            snapshot["degradation"] = {
                "state": self.degradation_model.state.value,
                "cycle_count": self.degradation_model.cycle_count,
            }
        
        if self.heat_death_model:
            snapshot["heat_death"] = {
                "resource_level": self.heat_death_model.resource_level,
                "cycle_count": self.heat_death_model.cycle_count,
            }
        
        return snapshot
    
    @staticmethod
    def _int_to_hex_seed(seed: int) -> str:
        """Convert integer seed to hex string for DeterministicPRNG."""
        return f"{seed:016x}"


# ============================================================================
# Configuration Loading from YAML
# ============================================================================

def load_unified_noise_config_from_yaml(yaml_path: str) -> UnifiedNoiseConfig:
    """Load unified noise configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
    
    Returns:
        UnifiedNoiseConfig instance
    """
    import yaml
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    config = UnifiedNoiseConfig(
        base_noise=BaseNoiseConfig(**data.get("base_noise", {})),
        correlated=CorrelatedNoiseConfig(**data.get("correlated", {})),
        degradation=ClusterDegradationConfig(**data.get("degradation", {})),
        heat_death=HeatDeathConfig(**data.get("heat_death", {})),
        heavy_tail=HeavyTailConfig(**data.get("heavy_tail", {})),
        nonstationary=NonStationaryConfig(**data.get("nonstationary", {})),
        adaptive=AdaptiveNoiseConfig(**data.get("adaptive", {})),
    )
    
    return config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create unified noise model with all regimes enabled
    
    config = UnifiedNoiseConfig(
        base_noise=BaseNoiseConfig(
            timeout_rate=0.1,
            spurious_fail_rate=0.05,
            spurious_pass_rate=0.02,
        ),
        correlated=CorrelatedNoiseConfig(
            enabled=True,
            rho=0.1,
            theta={"tactic_simp": 0.3, "tactic_ring": 0.2},
            item_factors={
                "Mathlib.Algebra.Ring.Basic.theorem_1": ["tactic_ring"],
                "Mathlib.Algebra.Ring.Basic.theorem_2": ["tactic_ring", "tactic_simp"],
            },
        ),
        degradation=ClusterDegradationConfig(
            enabled=True,
            alpha=0.05,
            beta=0.2,
            theta_healthy=0.01,
            theta_degraded=0.3,
        ),
        heat_death=HeatDeathConfig(
            enabled=True,
            initial_resource=1000.0,
            min_resource=100.0,
            consumption_mean=10.0,
            consumption_std=5.0,
            recovery_mean=8.0,
            recovery_std=3.0,
        ),
        heavy_tail=HeavyTailConfig(
            enabled=True,
            pi=0.1,
            lambda_fast=0.1,
            alpha=1.5,
            x_min=100.0,
        ),
        nonstationary=NonStationaryConfig(
            enabled=True,
            theta_0=0.1,
            delta=0.0001,
            sigma=0.01,
        ),
        adaptive=AdaptiveNoiseConfig(
            enabled=True,
            gamma=0.5,
        ),
    )
    
    # Create model
    model = UnifiedNoiseModel(config, master_seed=12345)
    
    # Simulate 10 cycles
    for cycle in range(10):
        model.step_cycle()
        
        print(f"\n=== Cycle {cycle} ===")
        print(f"State snapshot: {model.get_state_snapshot()}")
        
        # Test noise decisions for sample items
        items = [
            "Mathlib.Algebra.Ring.Basic.theorem_1",
            "Mathlib.Algebra.Ring.Basic.theorem_2",
            "Mathlib.Data.Nat.Basic.theorem_1",
        ]
        
        for item in items:
            meta = {"policy_prob": 0.7}  # Example policy probability
            
            should_timeout = model.should_timeout(item, meta)
            should_fail = model.should_spurious_fail(item, meta)
            should_pass = model.should_spurious_pass(item, meta)
            
            print(f"  {item}:")
            print(f"    Timeout: {should_timeout}")
            print(f"    Spurious fail: {should_fail}")
            print(f"    Spurious pass: {should_pass}")
            
            if should_timeout:
                duration = model.sample_timeout_duration(item)
                print(f"    Timeout duration: {duration:.2f} ms")
