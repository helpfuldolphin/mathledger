"""
UnifiedNoiseModel MVP

Simplified implementation with base + heavy-tail + adaptive regimes.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready (MVP)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
from rfl.prng import DeterministicPRNG


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
        assert 0.0 <= self.timeout_rate <= 1.0
        assert 0.0 <= self.spurious_fail_rate <= 1.0
        assert 0.0 <= self.spurious_pass_rate <= 1.0


# ============================================================================
# Heavy-Tail Timeout Configuration
# ============================================================================

@dataclass
class HeavyTailConfig:
    """Configuration for heavy-tailed timeout distribution."""
    
    enabled: bool = False
    pi: float = 0.1          # Mixing probability for Pareto component
    lambda_fast: float = 0.1  # Rate parameter for exponential component
    alpha: float = 1.5        # Tail index for Pareto component
    x_min: float = 100.0      # Scale parameter for Pareto component


# ============================================================================
# Adaptive Noise Configuration
# ============================================================================

@dataclass
class AdaptiveNoiseConfig:
    """Configuration for policy-aware adaptive noise regime."""
    
    enabled: bool = False
    gamma: float = 0.5  # Adaptation strength


# ============================================================================
# Unified Noise Model MVP Configuration
# ============================================================================

@dataclass
class UnifiedNoiseConfigMVP:
    """Configuration for unified noise model MVP."""
    
    base_noise: BaseNoiseConfig = field(default_factory=BaseNoiseConfig)
    heavy_tail: HeavyTailConfig = field(default_factory=HeavyTailConfig)
    adaptive: AdaptiveNoiseConfig = field(default_factory=AdaptiveNoiseConfig)
    
    def validate(self) -> None:
        """Validate configuration."""
        self.base_noise.validate()


# ============================================================================
# Unified Noise Model MVP
# ============================================================================

@dataclass
class UnifiedNoiseModelMVP:
    """Unified noise model MVP with base + heavy-tail + adaptive regimes.
    
    This is a simplified version for immediate deployment, implementing:
    1. Base Noise: Independent Bernoulli (timeout, spurious fail/pass)
    2. Heavy-Tail Timeouts: Mixture distribution with Pareto tails
    3. Adaptive Noise: Policy-aware adjustment
    
    Usage:
        config = UnifiedNoiseConfigMVP(...)
        model = UnifiedNoiseModelMVP(config, master_seed=12345)
        
        # For each cycle
        model.step_cycle()
        
        # For each item
        should_timeout = model.should_timeout(item, meta)
        if should_timeout:
            duration = model.sample_timeout_duration(item)
    """
    
    config: UnifiedNoiseConfigMVP
    master_seed: int
    
    # PRNG
    prng: DeterministicPRNG = field(init=False)
    
    # State
    cycle_count: int = 0
    
    def __post_init__(self):
        """Initialize PRNG."""
        self.config.validate()
        self.prng = DeterministicPRNG(self._int_to_hex_seed(self.master_seed))
    
    def step_cycle(self) -> None:
        """Advance to next cycle."""
        self.cycle_count += 1
    
    def should_timeout(
        self,
        item: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if item should timeout.
        
        Args:
            item: Item identifier
            meta: Optional metadata (e.g., policy_prob for adaptive noise)
        
        Returns:
            True if timeout should be injected
        """
        meta = meta or {}
        
        # Start with base timeout rate
        timeout_rate = self.config.base_noise.timeout_rate
        
        # Apply adaptive adjustment
        if self.config.adaptive.enabled and "policy_prob" in meta:
            timeout_rate = self._apply_adaptive_adjustment(
                timeout_rate,
                meta["policy_prob"],
            )
        
        # Sample timeout decision
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
            True if spurious failure should be injected
        """
        meta = meta or {}
        
        fail_rate = self.config.base_noise.spurious_fail_rate
        
        if self.config.adaptive.enabled and "policy_prob" in meta:
            fail_rate = self._apply_adaptive_adjustment(
                fail_rate,
                meta["policy_prob"],
            )
        
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
            True if spurious pass should be injected
        """
        meta = meta or {}
        
        pass_rate = self.config.base_noise.spurious_pass_rate
        
        if self.config.adaptive.enabled and "policy_prob" in meta:
            pass_rate = self._apply_adaptive_adjustment(
                pass_rate,
                meta["policy_prob"],
            )
        
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
        
        if not self.config.heavy_tail.enabled:
            # Fallback to exponential
            prng_item = self.prng.for_path("timeout_duration", item, str(self.cycle_count))
            return prng_item.expovariate(0.01)
        
        # Heavy-tail mixture distribution
        prng_item = self.prng.for_path("heavy_tail", item, str(self.cycle_count))
        
        # Sample component (exponential or Pareto)
        prng_component = prng_item.for_path("component")
        use_pareto = prng_component.random() < self.config.heavy_tail.pi
        
        if use_pareto:
            # Sample from Pareto
            prng_pareto = prng_item.for_path("pareto")
            u = prng_pareto.random()
            duration = self.config.heavy_tail.x_min / (u ** (1 / self.config.heavy_tail.alpha))
        else:
            # Sample from exponential
            prng_exp = prng_item.for_path("exponential")
            duration = prng_exp.expovariate(self.config.heavy_tail.lambda_fast)
        
        return duration
    
    def _apply_adaptive_adjustment(
        self,
        base_rate: float,
        policy_prob: float,
    ) -> float:
        """Apply adaptive noise adjustment based on policy confidence.
        
        Args:
            base_rate: Base noise rate
            policy_prob: Policy probability (0 to 1)
        
        Returns:
            Adjusted noise rate
        """
        
        # Compute confidence (distance from uniform)
        confidence = abs(policy_prob - 0.5) * 2
        
        # Scale noise rate
        adaptive_rate = base_rate * (1 + self.config.adaptive.gamma * confidence)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, adaptive_rate))
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current state.
        
        Returns:
            Dict with state
        """
        return {
            "cycle_count": self.cycle_count,
        }
    
    @staticmethod
    def _int_to_hex_seed(seed: int) -> str:
        """Convert integer seed to hex string."""
        return f"{seed:016x}"


# ============================================================================
# Configuration Loading from YAML
# ============================================================================

def load_unified_noise_config_mvp_from_yaml(yaml_path: str) -> UnifiedNoiseConfigMVP:
    """Load unified noise configuration MVP from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
    
    Returns:
        UnifiedNoiseConfigMVP instance
    """
    import yaml
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    config = UnifiedNoiseConfigMVP(
        base_noise=BaseNoiseConfig(**data.get("base_noise", {})),
        heavy_tail=HeavyTailConfig(**data.get("heavy_tail", {})),
        adaptive=AdaptiveNoiseConfig(**data.get("adaptive", {})),
    )
    
    return config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create unified noise model MVP
    
    config = UnifiedNoiseConfigMVP(
        base_noise=BaseNoiseConfig(
            timeout_rate=0.1,
            spurious_fail_rate=0.05,
            spurious_pass_rate=0.02,
        ),
        heavy_tail=HeavyTailConfig(
            enabled=True,
            pi=0.1,
            lambda_fast=0.1,
            alpha=1.5,
            x_min=100.0,
        ),
        adaptive=AdaptiveNoiseConfig(
            enabled=True,
            gamma=0.5,
        ),
    )
    
    # Create model
    model = UnifiedNoiseModelMVP(config, master_seed=12345)
    
    # Simulate 5 cycles
    for cycle in range(5):
        model.step_cycle()
        
        print(f"\n=== Cycle {cycle} ===")
        
        # Test noise decisions for sample items
        items = ["Module.A", "Module.B", "Module.C"]
        
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
