#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Temporal Drift Simulator v1.0
------------------------------

This module implements temporal drift for synthetic scenario generation,
extending beyond simple probability drift to include:

    - Confusability drift (how "tricky" items appear)
    - Success rate drift (probability of correct verification)
    - Abstention pattern drift (tendency to skip items)

Drift Modes:
    - sinusoidal: Smooth oscillation with configurable period
    - linear: Monotonic increase or decrease
    - step: Discrete jumps at specified cycles

Must NOT:
    - Produce claims about real uplift
    - Imply any empirical conclusions
    - Mix synthetic and real data

==============================================================================
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# TEMPORAL DRIFT MODES
# ==============================================================================

class TemporalDriftMode(Enum):
    """Temporal drift mode enumeration for v1.0."""
    NONE = "none"
    SINUSOIDAL = "sinusoidal"
    LINEAR = "linear"
    STEP = "step"


# ==============================================================================
# TEMPORAL DRIFT CONFIG
# ==============================================================================

@dataclass
class TemporalDriftConfig:
    """
    Configuration for temporal drift in synthetic scenarios.
    
    Attributes:
        mode: Type of drift (none, sinusoidal, linear, step)
        period: Oscillation period for sinusoidal mode (cycles)
        amplitude: Maximum deviation from base value
        slope: Rate of change per cycle (linear mode)
        step_cycles: Cycles at which steps occur (step mode)
        step_values: Values at each step (step mode)
        enabled: Whether drift is active (default: False)
    """
    mode: TemporalDriftMode = TemporalDriftMode.NONE
    period: int = 100
    amplitude: float = 0.0
    slope: float = 0.0
    step_cycles: List[int] = field(default_factory=list)
    step_values: List[float] = field(default_factory=list)
    enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for manifest."""
        return {
            "mode": self.mode.value,
            "period": self.period,
            "amplitude": self.amplitude,
            "slope": self.slope,
            "step_cycles": self.step_cycles.copy(),
            "step_values": self.step_values.copy(),
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalDriftConfig":
        """Deserialize from dictionary."""
        mode_str = data.get("mode", "none")
        return cls(
            mode=TemporalDriftMode(mode_str) if mode_str else TemporalDriftMode.NONE,
            period=data.get("period", 100),
            amplitude=data.get("amplitude", 0.0),
            slope=data.get("slope", 0.0),
            step_cycles=data.get("step_cycles", []),
            step_values=data.get("step_values", []),
            enabled=data.get("enabled", False),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        
        if self.mode == TemporalDriftMode.SINUSOIDAL:
            if self.period <= 0:
                errors.append("Sinusoidal drift requires period > 0")
            if not -1.0 <= self.amplitude <= 1.0:
                errors.append("Amplitude must be in [-1.0, 1.0]")
        
        elif self.mode == TemporalDriftMode.LINEAR:
            if abs(self.slope) > 0.01:
                errors.append("Linear slope too steep (max 0.01 per cycle)")
        
        elif self.mode == TemporalDriftMode.STEP:
            if len(self.step_cycles) != len(self.step_values):
                errors.append("step_cycles and step_values must have same length")
            if self.step_cycles and self.step_cycles != sorted(self.step_cycles):
                errors.append("step_cycles must be in ascending order")
        
        return errors


# ==============================================================================
# DRIFTED SIGNAL TYPES
# ==============================================================================

class DriftedSignalType(Enum):
    """Types of signals that can experience temporal drift."""
    SUCCESS_RATE = "success_rate"
    CONFUSABILITY = "confusability"
    ABSTENTION_RATE = "abstention_rate"


@dataclass
class DriftedSignalConfig:
    """
    Configuration for a single drifted signal.
    
    Attributes:
        signal_type: What signal is being drifted
        base_value: Starting/center value for the signal
        drift_config: How the signal drifts over time
    """
    signal_type: DriftedSignalType
    base_value: float
    drift_config: TemporalDriftConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "base_value": self.base_value,
            "drift_config": self.drift_config.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftedSignalConfig":
        """Deserialize from dictionary."""
        return cls(
            signal_type=DriftedSignalType(data.get("signal_type", "success_rate")),
            base_value=data.get("base_value", 0.5),
            drift_config=TemporalDriftConfig.from_dict(data.get("drift_config", {})),
        )


# ==============================================================================
# TEMPORAL DRIFT SIMULATOR
# ==============================================================================

class TemporalDriftSimulator:
    """
    Simulates temporal drift for multiple signal types.
    
    This is deterministic and purely structural - no empirical claims.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize simulator with a seed for determinism.
        
        Args:
            seed: Random seed (used for step jitter if enabled)
        """
        self.seed = seed
        self.signals: Dict[DriftedSignalType, DriftedSignalConfig] = {}
    
    def configure_signal(self, config: DriftedSignalConfig):
        """Add or update a drifted signal configuration."""
        self.signals[config.signal_type] = config
    
    def get_value(
        self,
        signal_type: DriftedSignalType,
        cycle: int,
        total_cycles: int,
    ) -> float:
        """
        Get the drifted value for a signal at a specific cycle.
        
        Args:
            signal_type: Which signal to query
            cycle: Current cycle number (0-indexed)
            total_cycles: Total cycles in the run
        
        Returns:
            Drifted value, clamped to [0.01, 0.99]
        """
        if signal_type not in self.signals:
            return 0.5  # Default neutral value
        
        config = self.signals[signal_type]
        drift_config = config.drift_config
        
        if not drift_config.enabled:
            return config.base_value
        
        # Apply drift based on mode
        drift_value = self._compute_drift(drift_config, cycle, total_cycles)
        final_value = config.base_value + drift_value
        
        # Clamp to valid range
        return max(0.01, min(0.99, final_value))
    
    def _compute_drift(
        self,
        config: TemporalDriftConfig,
        cycle: int,
        total_cycles: int,
    ) -> float:
        """Compute the raw drift value at a cycle."""
        
        if config.mode == TemporalDriftMode.NONE:
            return 0.0
        
        elif config.mode == TemporalDriftMode.SINUSOIDAL:
            if config.period <= 0:
                return 0.0
            phase = 2.0 * math.pi * cycle / config.period
            return config.amplitude * math.sin(phase)
        
        elif config.mode == TemporalDriftMode.LINEAR:
            return config.slope * cycle
        
        elif config.mode == TemporalDriftMode.STEP:
            # Find which step we're at
            current_delta = 0.0
            for i, step_cycle in enumerate(config.step_cycles):
                if cycle >= step_cycle:
                    current_delta = config.step_values[i]
                else:
                    break
            return current_delta
        
        return 0.0
    
    def get_drift_profile(
        self,
        signal_type: DriftedSignalType,
        total_cycles: int,
    ) -> List[float]:
        """
        Get the complete drift profile for a signal.
        
        Returns a list of values, one per cycle.
        """
        return [
            self.get_value(signal_type, cycle, total_cycles)
            for cycle in range(total_cycles)
        ]
    
    def validate(self) -> List[str]:
        """Validate all signal configurations."""
        errors = []
        for signal_type, config in self.signals.items():
            signal_errors = config.drift_config.validate()
            for err in signal_errors:
                errors.append(f"{signal_type.value}: {err}")
        return errors


# ==============================================================================
# SCENARIO TEMPORAL DRIFT SPECIFICATION
# ==============================================================================

@dataclass
class ScenarioTemporalDrift:
    """
    Complete temporal drift specification for a scenario.
    
    Encapsulates drift for all signal types in a single structure
    that can be attached to scenario definitions.
    """
    
    success_drift: Optional[TemporalDriftConfig] = None
    confusability_drift: Optional[TemporalDriftConfig] = None
    abstention_drift: Optional[TemporalDriftConfig] = None
    
    # Base values
    base_success_rate: float = 0.7
    base_confusability: float = 0.3
    base_abstention_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for scenario registry."""
        return {
            "label": SAFETY_LABEL,
            "base_values": {
                "success_rate": self.base_success_rate,
                "confusability": self.base_confusability,
                "abstention_rate": self.base_abstention_rate,
            },
            "success_drift": self.success_drift.to_dict() if self.success_drift else None,
            "confusability_drift": self.confusability_drift.to_dict() if self.confusability_drift else None,
            "abstention_drift": self.abstention_drift.to_dict() if self.abstention_drift else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioTemporalDrift":
        """Deserialize from dictionary."""
        base_values = data.get("base_values", {})
        
        return cls(
            success_drift=TemporalDriftConfig.from_dict(data["success_drift"]) if data.get("success_drift") else None,
            confusability_drift=TemporalDriftConfig.from_dict(data["confusability_drift"]) if data.get("confusability_drift") else None,
            abstention_drift=TemporalDriftConfig.from_dict(data["abstention_drift"]) if data.get("abstention_drift") else None,
            base_success_rate=base_values.get("success_rate", 0.7),
            base_confusability=base_values.get("confusability", 0.3),
            base_abstention_rate=base_values.get("abstention_rate", 0.1),
        )
    
    def build_simulator(self, seed: int = 42) -> TemporalDriftSimulator:
        """Build a simulator from this specification."""
        sim = TemporalDriftSimulator(seed=seed)
        
        if self.success_drift:
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.SUCCESS_RATE,
                base_value=self.base_success_rate,
                drift_config=self.success_drift,
            ))
        else:
            # Configure with no drift
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.SUCCESS_RATE,
                base_value=self.base_success_rate,
                drift_config=TemporalDriftConfig(enabled=False),
            ))
        
        if self.confusability_drift:
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.CONFUSABILITY,
                base_value=self.base_confusability,
                drift_config=self.confusability_drift,
            ))
        else:
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.CONFUSABILITY,
                base_value=self.base_confusability,
                drift_config=TemporalDriftConfig(enabled=False),
            ))
        
        if self.abstention_drift:
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.ABSTENTION_RATE,
                base_value=self.base_abstention_rate,
                drift_config=self.abstention_drift,
            ))
        else:
            sim.configure_signal(DriftedSignalConfig(
                signal_type=DriftedSignalType.ABSTENTION_RATE,
                base_value=self.base_abstention_rate,
                drift_config=TemporalDriftConfig(enabled=False),
            ))
        
        return sim
    
    def validate(self) -> List[str]:
        """Validate all drift configurations."""
        errors = []
        
        for name, config in [
            ("success", self.success_drift),
            ("confusability", self.confusability_drift),
            ("abstention", self.abstention_drift),
        ]:
            if config:
                config_errors = config.validate()
                for err in config_errors:
                    errors.append(f"{name}_drift: {err}")
        
        # Validate base values
        for name, value in [
            ("base_success_rate", self.base_success_rate),
            ("base_confusability", self.base_confusability),
            ("base_abstention_rate", self.base_abstention_rate),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be in [0.0, 1.0]")
        
        return errors
    
    @property
    def is_enabled(self) -> bool:
        """Check if any drift is enabled."""
        return (
            (self.success_drift and self.success_drift.enabled) or
            (self.confusability_drift and self.confusability_drift.enabled) or
            (self.abstention_drift and self.abstention_drift.enabled)
        )


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_sinusoidal_drift(
    period: int = 100,
    amplitude: float = 0.15,
) -> TemporalDriftConfig:
    """Create a sinusoidal drift configuration."""
    return TemporalDriftConfig(
        mode=TemporalDriftMode.SINUSOIDAL,
        period=period,
        amplitude=amplitude,
        enabled=True,
    )


def create_linear_drift(
    slope: float = 0.001,
) -> TemporalDriftConfig:
    """Create a linear drift configuration."""
    return TemporalDriftConfig(
        mode=TemporalDriftMode.LINEAR,
        slope=slope,
        enabled=True,
    )


def create_step_drift(
    step_cycles: List[int],
    step_values: List[float],
) -> TemporalDriftConfig:
    """Create a step drift configuration."""
    return TemporalDriftConfig(
        mode=TemporalDriftMode.STEP,
        step_cycles=step_cycles.copy(),
        step_values=step_values.copy(),
        enabled=True,
    )


def create_scenario_drift(
    success_mode: Optional[str] = None,
    confusability_mode: Optional[str] = None,
    abstention_mode: Optional[str] = None,
    period: int = 100,
    amplitude: float = 0.15,
    slope: float = 0.001,
) -> ScenarioTemporalDrift:
    """
    Create a scenario temporal drift specification.
    
    Args:
        success_mode: Drift mode for success rate (none, sinusoidal, linear, step)
        confusability_mode: Drift mode for confusability
        abstention_mode: Drift mode for abstention rate
        period: Period for sinusoidal mode
        amplitude: Amplitude for sinusoidal mode
        slope: Slope for linear mode
    
    Returns:
        ScenarioTemporalDrift specification
    """
    spec = ScenarioTemporalDrift()
    
    mode_map = {
        "sinusoidal": lambda: create_sinusoidal_drift(period, amplitude),
        "linear": lambda: create_linear_drift(slope),
        "step": lambda: create_step_drift([], []),  # Requires explicit configuration
        None: lambda: None,
        "none": lambda: None,
    }
    
    spec.success_drift = mode_map.get(success_mode, lambda: None)()
    spec.confusability_drift = mode_map.get(confusability_mode, lambda: None)()
    spec.abstention_drift = mode_map.get(abstention_mode, lambda: None)()
    
    return spec


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print(f"\n{SAFETY_LABEL}\n")
    print("Temporal Drift Simulator v1.0")
    print("=" * 60)
    
    # Demo sinusoidal drift
    spec = ScenarioTemporalDrift(
        success_drift=create_sinusoidal_drift(period=50, amplitude=0.2),
        confusability_drift=create_linear_drift(slope=0.001),
        base_success_rate=0.7,
        base_confusability=0.2,
    )
    
    sim = spec.build_simulator(seed=42)
    
    print("\nSuccess rate drift (sinusoidal, period=50):")
    profile = sim.get_drift_profile(DriftedSignalType.SUCCESS_RATE, 100)
    for i in range(0, 100, 10):
        print(f"  Cycle {i:3d}: {profile[i]:.4f}")
    
    print("\nConfusability drift (linear, slope=0.001):")
    profile = sim.get_drift_profile(DriftedSignalType.CONFUSABILITY, 100)
    for i in range(0, 100, 10):
        print(f"  Cycle {i:3d}: {profile[i]:.4f}")
    
    print(f"\n{SAFETY_LABEL}")

