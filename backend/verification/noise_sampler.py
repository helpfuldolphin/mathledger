"""
Deterministic Noise Sampler — Seeded Noise Injection for Imperfect Verifiers

This module implements deterministic, reproducible noise injection for verifier
calls. All noise is generated using seeded PRNGs to ensure:
- Identical seeds produce identical noise signatures
- No stochastic behavior without explicit seeding
- Full reproducibility for debugging and analysis

Design Principles:
- All randomness uses DeterministicPRNG from rfl.prng
- Noise rates are configurable per-tier, per-slice
- Timeout durations sampled from configurable distributions
- Context strings ensure unique noise per call

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rfl.prng import DeterministicPRNG, int_to_hex_seed


class TimeoutDistribution(Enum):
    """Timeout duration distribution types."""
    
    UNIFORM = "uniform"
    """Uniform distribution between min and max."""
    
    EXPONENTIAL = "exponential"
    """Exponential distribution with given mean."""
    
    FIXED = "fixed"
    """Fixed timeout duration."""


@dataclass(frozen=True)
class TimeoutDistributionConfig:
    """Configuration for timeout duration sampling."""
    
    type: TimeoutDistribution
    """Distribution type."""
    
    min_ms: Optional[float] = None
    """Minimum timeout in milliseconds (for UNIFORM)."""
    
    max_ms: Optional[float] = None
    """Maximum timeout in milliseconds (for UNIFORM)."""
    
    mean_ms: Optional[float] = None
    """Mean timeout in milliseconds (for EXPONENTIAL)."""
    
    fixed_ms: Optional[float] = None
    """Fixed timeout in milliseconds (for FIXED)."""
    
    @classmethod
    def uniform(cls, min_ms: float, max_ms: float) -> "TimeoutDistributionConfig":
        """Create uniform distribution config."""
        return cls(type=TimeoutDistribution.UNIFORM, min_ms=min_ms, max_ms=max_ms)
    
    @classmethod
    def exponential(cls, mean_ms: float) -> "TimeoutDistributionConfig":
        """Create exponential distribution config."""
        return cls(type=TimeoutDistribution.EXPONENTIAL, mean_ms=mean_ms)
    
    @classmethod
    def fixed(cls, fixed_ms: float) -> "TimeoutDistributionConfig":
        """Create fixed duration config."""
        return cls(type=TimeoutDistribution.FIXED, fixed_ms=fixed_ms)


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for noise injection rates and distributions.
    
    All rates are probabilities in [0, 1].
    """
    
    noise_enabled: bool
    """Whether noise injection is enabled."""
    
    timeout_rate: float
    """Probability of timeout injection."""
    
    spurious_fail_rate: float
    """Probability of spurious failure (false negative)."""
    
    spurious_pass_rate: float
    """Probability of spurious pass (false positive)."""
    
    timeout_distribution: TimeoutDistributionConfig
    """Configuration for timeout duration sampling."""
    
    def __post_init__(self):
        """Validate noise rates."""
        if not (0.0 <= self.timeout_rate <= 1.0):
            raise ValueError(f"timeout_rate must be in [0, 1], got {self.timeout_rate}")
        if not (0.0 <= self.spurious_fail_rate <= 1.0):
            raise ValueError(f"spurious_fail_rate must be in [0, 1], got {self.spurious_fail_rate}")
        if not (0.0 <= self.spurious_pass_rate <= 1.0):
            raise ValueError(f"spurious_pass_rate must be in [0, 1], got {self.spurious_pass_rate}")
    
    @classmethod
    def no_noise(cls) -> "NoiseConfig":
        """Create config with no noise (for testing)."""
        return cls(
            noise_enabled=False,
            timeout_rate=0.0,
            spurious_fail_rate=0.0,
            spurious_pass_rate=0.0,
            timeout_distribution=TimeoutDistributionConfig.fixed(0.0),
        )
    
    @classmethod
    def default_fast_noisy(cls) -> "NoiseConfig":
        """Create default config for fast_noisy tier."""
        return cls(
            noise_enabled=True,
            timeout_rate=0.10,
            spurious_fail_rate=0.05,
            spurious_pass_rate=0.02,
            timeout_distribution=TimeoutDistributionConfig.uniform(500, 1500),
        )
    
    @classmethod
    def default_balanced(cls) -> "NoiseConfig":
        """Create default config for balanced tier."""
        return cls(
            noise_enabled=True,
            timeout_rate=0.05,
            spurious_fail_rate=0.02,
            spurious_pass_rate=0.01,
            timeout_distribution=TimeoutDistributionConfig.uniform(1000, 2000),
        )
    
    @classmethod
    def default_slow_precise(cls) -> "NoiseConfig":
        """Create default config for slow_precise tier."""
        return cls(
            noise_enabled=True,
            timeout_rate=0.01,
            spurious_fail_rate=0.005,
            spurious_pass_rate=0.001,
            timeout_distribution=TimeoutDistributionConfig.uniform(1500, 3000),
        )


class NoiseSampler:
    """Deterministic noise sampler for verifier calls.
    
    This class uses hierarchical seeded PRNGs to ensure:
    - Identical seeds produce identical noise signatures
    - Different contexts produce independent noise
    - Full reproducibility across runs
    
    Usage:
        sampler = NoiseSampler(config, seed=42)
        if sampler.should_timeout("cycle_1_item_3"):
            # Inject timeout
            duration = sampler.sample_timeout_duration("cycle_1_item_3")
    """
    
    def __init__(self, config: NoiseConfig, seed: int):
        """Initialize noise sampler with config and seed.
        
        Args:
            config: Noise configuration (rates, distributions)
            seed: Master seed for deterministic noise generation
        """
        self.config = config
        self.seed = seed
        self.prng = DeterministicPRNG(int_to_hex_seed(seed))
    
    def should_timeout(self, context: str) -> bool:
        """Deterministically decide if this call should timeout.
        
        Args:
            context: Unique context string for this call (e.g., "cycle_1_item_3")
        
        Returns:
            True if timeout should be injected, False otherwise
        """
        if not self.config.noise_enabled:
            return False
        
        rng = self.prng.for_path("timeout", context)
        return rng.random() < self.config.timeout_rate
    
    def should_spurious_fail(self, context: str) -> bool:
        """Deterministically decide if valid proof should fail (false negative).
        
        Args:
            context: Unique context string for this call
        
        Returns:
            True if spurious failure should be injected, False otherwise
        """
        if not self.config.noise_enabled:
            return False
        
        rng = self.prng.for_path("spurious_fail", context)
        return rng.random() < self.config.spurious_fail_rate
    
    def should_spurious_pass(self, context: str) -> bool:
        """Deterministically decide if invalid proof should pass (false positive).
        
        Args:
            context: Unique context string for this call
        
        Returns:
            True if spurious pass should be injected, False otherwise
        """
        if not self.config.noise_enabled:
            return False
        
        rng = self.prng.for_path("spurious_pass", context)
        return rng.random() < self.config.spurious_pass_rate
    
    def sample_timeout_duration(self, context: str) -> float:
        """Sample timeout duration from configured distribution.
        
        Args:
            context: Unique context string for this call
        
        Returns:
            Timeout duration in seconds (deterministic for given context)
        """
        rng = self.prng.for_path("timeout_duration", context)
        dist = self.config.timeout_distribution
        
        if dist.type == TimeoutDistribution.UNIFORM:
            assert dist.min_ms is not None and dist.max_ms is not None
            duration_ms = rng.uniform(dist.min_ms, dist.max_ms)
            return duration_ms / 1000.0
        
        elif dist.type == TimeoutDistribution.EXPONENTIAL:
            assert dist.mean_ms is not None
            # Sample from exponential distribution
            # Using inverse transform: -mean * ln(U) where U ~ Uniform(0, 1)
            import math
            u = rng.random()
            # Clamp u away from 0 to avoid log(0)
            u = max(u, 1e-10)
            duration_ms = -dist.mean_ms * math.log(u)
            return duration_ms / 1000.0
        
        elif dist.type == TimeoutDistribution.FIXED:
            assert dist.fixed_ms is not None
            return dist.fixed_ms / 1000.0
        
        else:
            raise ValueError(f"Unknown timeout distribution type: {dist.type}")
    
    def get_noise_signature(self, context: str) -> str:
        """Get deterministic noise signature for a context.
        
        This signature uniquely identifies the noise behavior for a given
        context and seed. Useful for debugging and reproducibility checks.
        
        Args:
            context: Unique context string for this call
        
        Returns:
            Hex-encoded hash of noise decisions
        """
        decisions = [
            str(self.should_timeout(context)),
            str(self.should_spurious_fail(context)),
            str(self.should_spurious_pass(context)),
            str(self.sample_timeout_duration(context)),
        ]
        signature_str = "|".join(decisions)
        return hashlib.sha256(signature_str.encode("utf-8")).hexdigest()[:16]


# ==================== Noise Sampler Factory ====================

def create_noise_sampler(
    tier: str,
    seed: int,
    config_overrides: Optional[NoiseConfig] = None,
) -> NoiseSampler:
    """Factory function to create noise sampler for a tier.
    
    Args:
        tier: Verifier tier name ("fast_noisy", "balanced", "slow_precise")
        seed: Master seed for noise generation
        config_overrides: Optional config overrides
    
    Returns:
        Configured NoiseSampler instance
    """
    if config_overrides is not None:
        config = config_overrides
    elif tier == "fast_noisy":
        config = NoiseConfig.default_fast_noisy()
    elif tier == "balanced":
        config = NoiseConfig.default_balanced()
    elif tier == "slow_precise":
        config = NoiseConfig.default_slow_precise()
    else:
        # Default to no noise for unknown tiers
        config = NoiseConfig.no_noise()
    
    return NoiseSampler(config, seed)
