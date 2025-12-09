#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Noise Models for Synthetic Uplift Generation
---------------------------------------------

This module implements temporal drift and class-correlation noise models
for stress-testing the U2 uplift analysis pipeline.

NOT derived from real derivations; NOT part of Evidence Pack.

Drift Modes:
    - none: Constant probabilities (no drift)
    - monotonic: Linear increase or decrease over time
    - cyclical: Sinusoidal modulation p_t = base_p + amplitude * sin(2π * t / period)
    - shock: Sudden probability shift at a specific cycle

Correlation Models:
    - Independent: Items fail independently (ρ = 0)
    - Correlated: Items in same class co-fail with probability ρ

==============================================================================
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ==============================================================================
# SAFETY LABEL
# ==============================================================================
SAFETY_LABEL = "PHASE II — SYNTHETIC TEST DATA ONLY"


# ==============================================================================
# DRIFT MODELS
# ==============================================================================

class DriftMode(Enum):
    """Temporal drift mode enumeration."""
    NONE = "none"
    MONOTONIC = "monotonic"
    CYCLICAL = "cyclical"
    SHOCK = "shock"


@dataclass
class DriftConfig:
    """
    Configuration for temporal probability drift.
    
    Attributes:
        mode: Type of drift (none, monotonic, cyclical, shock)
        amplitude: Maximum deviation from base probability (for cyclical/monotonic)
        period: Number of cycles for one complete oscillation (cyclical only)
        slope: Rate of change per cycle (monotonic only)
        shock_cycle: Cycle at which shock occurs (shock only)
        shock_delta: Probability change at shock point (shock only)
        direction: 'up' or 'down' for monotonic drift
    """
    mode: DriftMode = DriftMode.NONE
    amplitude: float = 0.0
    period: int = 100
    slope: float = 0.0
    shock_cycle: int = 0
    shock_delta: float = 0.0
    direction: str = "up"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for manifest."""
        return {
            "mode": self.mode.value,
            "amplitude": self.amplitude,
            "period": self.period,
            "slope": self.slope,
            "shock_cycle": self.shock_cycle,
            "shock_delta": self.shock_delta,
            "direction": self.direction,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftConfig":
        """Deserialize from dictionary."""
        return cls(
            mode=DriftMode(data.get("mode", "none")),
            amplitude=data.get("amplitude", 0.0),
            period=data.get("period", 100),
            slope=data.get("slope", 0.0),
            shock_cycle=data.get("shock_cycle", 0),
            shock_delta=data.get("shock_delta", 0.0),
            direction=data.get("direction", "up"),
        )


class DriftModulator:
    """
    Applies temporal drift to base probabilities.
    
    Deterministic modulation based on cycle number and drift configuration.
    """
    
    def __init__(self, config: DriftConfig):
        self.config = config
    
    def modulate(self, base_prob: float, cycle: int, total_cycles: int) -> float:
        """
        Apply drift modulation to a base probability.
        
        Args:
            base_prob: The base probability (0.0 to 1.0)
            cycle: Current cycle number (0-indexed)
            total_cycles: Total number of cycles in the run
        
        Returns:
            Modulated probability, clamped to [0.01, 0.99]
        """
        if self.config.mode == DriftMode.NONE:
            return base_prob
        
        elif self.config.mode == DriftMode.MONOTONIC:
            # Linear drift: p_t = base_p + slope * t
            # Direction determines sign
            sign = 1.0 if self.config.direction == "up" else -1.0
            drift = sign * self.config.slope * cycle
            modulated = base_prob + drift
        
        elif self.config.mode == DriftMode.CYCLICAL:
            # Sinusoidal drift: p_t = base_p + amplitude * sin(2π * t / period)
            if self.config.period > 0:
                phase = 2.0 * math.pi * cycle / self.config.period
                drift = self.config.amplitude * math.sin(phase)
            else:
                drift = 0.0
            modulated = base_prob + drift
        
        elif self.config.mode == DriftMode.SHOCK:
            # Sudden shift at shock_cycle
            if cycle >= self.config.shock_cycle:
                modulated = base_prob + self.config.shock_delta
            else:
                modulated = base_prob
        
        else:
            modulated = base_prob
        
        # Clamp to valid probability range
        return max(0.01, min(0.99, modulated))
    
    def get_drift_at_cycle(self, cycle: int, total_cycles: int) -> float:
        """
        Get the raw drift value at a specific cycle (without base probability).
        
        Useful for visualization and debugging.
        """
        return self.modulate(0.5, cycle, total_cycles) - 0.5


# ==============================================================================
# CORRELATION MODELS
# ==============================================================================

@dataclass
class CorrelationConfig:
    """
    Configuration for class-based correlation noise.
    
    Attributes:
        rho: Correlation coefficient (0.0 = independent, 1.0 = fully correlated)
        mode: 'class' for intra-class correlation, 'global' for all items
    """
    rho: float = 0.0
    mode: str = "class"  # 'class' or 'global'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rho": self.rho,
            "mode": self.mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationConfig":
        """Deserialize from dictionary."""
        return cls(
            rho=data.get("rho", 0.0),
            mode=data.get("mode", "class"),
        )


class CorrelationEngine:
    """
    Implements class-based correlation for item outcomes.
    
    When correlation ρ > 0, items in the same class may co-fail based on
    a shared latent Bernoulli variable Z:
    
    - With probability ρ, outcome is determined by shared Z
    - With probability (1-ρ), outcome is determined independently
    
    This creates realistic patterns where related items tend to fail together.
    """
    
    def __init__(self, config: CorrelationConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        
        # Cache for per-class latent variables per cycle
        self._class_latents: Dict[int, Dict[str, bool]] = {}
        self._global_latent: Dict[int, bool] = {}
    
    def _get_class_latent(self, cycle: int, item_class: str, cycle_seed: int) -> bool:
        """
        Get the shared latent Bernoulli variable for a class at a cycle.
        
        Deterministic based on cycle_seed and class name.
        """
        if cycle not in self._class_latents:
            self._class_latents[cycle] = {}
        
        if item_class not in self._class_latents[cycle]:
            # Deterministic latent based on cycle and class
            latent_seed = cycle_seed ^ hash(item_class)
            latent_rng = random.Random(latent_seed)
            self._class_latents[cycle][item_class] = latent_rng.random() < 0.5
        
        return self._class_latents[cycle][item_class]
    
    def _get_global_latent(self, cycle: int, cycle_seed: int) -> bool:
        """
        Get the global latent Bernoulli variable for a cycle.
        """
        if cycle not in self._global_latent:
            latent_rng = random.Random(cycle_seed ^ 0xDEADBEEF)
            self._global_latent[cycle] = latent_rng.random() < 0.5
        
        return self._global_latent[cycle]
    
    def apply_correlation(
        self,
        independent_success: bool,
        item_class: str,
        cycle: int,
        cycle_seed: int,
        item_id: str,
    ) -> bool:
        """
        Apply correlation to an independently generated outcome.
        
        Args:
            independent_success: The outcome generated independently
            item_class: The class of the item
            cycle: Current cycle number
            cycle_seed: Seed for this cycle
            item_id: Unique item identifier
        
        Returns:
            Correlated outcome (may differ from independent_success with prob ρ)
        """
        if self.config.rho <= 0.0:
            return independent_success
        
        # Determine if this outcome should use shared latent
        mix_seed = cycle_seed ^ hash(item_id) ^ 0xCAFEBABE
        mix_rng = random.Random(mix_seed)
        use_shared = mix_rng.random() < self.config.rho
        
        if not use_shared:
            return independent_success
        
        # Use shared latent variable
        if self.config.mode == "global":
            shared_outcome = self._get_global_latent(cycle, cycle_seed)
        else:  # class mode
            shared_outcome = self._get_class_latent(cycle, item_class, cycle_seed)
        
        return shared_outcome
    
    def clear_cache(self):
        """Clear the latent variable cache."""
        self._class_latents.clear()
        self._global_latent.clear()


# ==============================================================================
# COMBINED NOISE MODEL
# ==============================================================================

@dataclass
class NoiseConfig:
    """
    Combined configuration for all noise models.
    """
    drift: DriftConfig = field(default_factory=DriftConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "drift": self.drift.to_dict(),
            "correlation": self.correlation.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseConfig":
        """Deserialize from dictionary."""
        return cls(
            drift=DriftConfig.from_dict(data.get("drift", {})),
            correlation=CorrelationConfig.from_dict(data.get("correlation", {})),
        )


class NoiseEngine:
    """
    Combined noise engine applying both drift and correlation.
    
    Usage:
        engine = NoiseEngine(noise_config, seed=42)
        modulated_prob = engine.apply_drift(base_prob, cycle, total_cycles)
        final_outcome = engine.apply_correlation(
            independent_outcome, item_class, cycle, cycle_seed, item_id
        )
    """
    
    def __init__(self, config: NoiseConfig, seed: int):
        self.config = config
        self.drift_modulator = DriftModulator(config.drift)
        self.correlation_engine = CorrelationEngine(config.correlation, seed)
    
    def apply_drift(self, base_prob: float, cycle: int, total_cycles: int) -> float:
        """Apply temporal drift to a probability."""
        return self.drift_modulator.modulate(base_prob, cycle, total_cycles)
    
    def apply_correlation(
        self,
        independent_success: bool,
        item_class: str,
        cycle: int,
        cycle_seed: int,
        item_id: str,
    ) -> bool:
        """Apply class correlation to an outcome."""
        return self.correlation_engine.apply_correlation(
            independent_success, item_class, cycle, cycle_seed, item_id
        )
    
    def clear_cache(self):
        """Clear internal caches."""
        self.correlation_engine.clear_cache()


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_no_drift() -> DriftConfig:
    """Create a no-drift configuration."""
    return DriftConfig(mode=DriftMode.NONE)


def create_monotonic_drift(
    slope: float = 0.001,
    direction: str = "down",
) -> DriftConfig:
    """
    Create monotonic drift configuration.
    
    Args:
        slope: Rate of change per cycle
        direction: 'up' or 'down'
    """
    return DriftConfig(
        mode=DriftMode.MONOTONIC,
        slope=slope,
        direction=direction,
    )


def create_cyclical_drift(
    amplitude: float = 0.15,
    period: int = 100,
) -> DriftConfig:
    """
    Create cyclical drift configuration.
    
    Args:
        amplitude: Maximum deviation from base probability
        period: Cycles per complete oscillation
    """
    return DriftConfig(
        mode=DriftMode.CYCLICAL,
        amplitude=amplitude,
        period=period,
    )


def create_shock_drift(
    shock_cycle: int = 250,
    shock_delta: float = -0.30,
) -> DriftConfig:
    """
    Create shock drift configuration.
    
    Args:
        shock_cycle: Cycle at which shock occurs
        shock_delta: Probability change (negative for degradation)
    """
    return DriftConfig(
        mode=DriftMode.SHOCK,
        shock_cycle=shock_cycle,
        shock_delta=shock_delta,
    )


def create_correlation(rho: float = 0.3, mode: str = "class") -> CorrelationConfig:
    """
    Create correlation configuration.
    
    Args:
        rho: Correlation coefficient [0, 1]
        mode: 'class' or 'global'
    """
    return CorrelationConfig(rho=rho, mode=mode)


# ==============================================================================
# TESTING UTILITIES
# ==============================================================================

def simulate_drift_series(
    drift_config: DriftConfig,
    base_prob: float = 0.5,
    total_cycles: int = 500,
) -> List[float]:
    """
    Simulate a series of drifted probabilities for visualization.
    
    Returns list of probabilities at each cycle.
    """
    modulator = DriftModulator(drift_config)
    return [
        modulator.modulate(base_prob, cycle, total_cycles)
        for cycle in range(total_cycles)
    ]

