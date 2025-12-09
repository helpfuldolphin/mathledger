#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Noise Universe Specification Schema
------------------------------------

This module defines the complete specification schema for parametric synthetic
universes. A NoiseSpec fully describes a synthetic universe that can be
compiled into a deterministic log generator.

Schema Components:
    - Base probabilities (per mode, per class)
    - Drift model configuration
    - Correlation model configuration
    - Variance parameters
    - Rare event channels (shock, burst, collapse)
    - Item and cycle counts
    - Random seed

Must NOT generate uplift interpretations.
All outputs labeled "PHASE II — SYNTHETIC".
Entire system is deterministic.

==============================================================================
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from experiments.synthetic_uplift.noise_models import (
    SAFETY_LABEL,
    CorrelationConfig,
    DriftConfig,
    DriftMode,
    NoiseConfig,
)


# ==============================================================================
# RARE EVENT CHANNEL TYPES
# ==============================================================================

class RareEventType(Enum):
    """Types of rare event channels."""
    CATASTROPHIC_COLLAPSE = "catastrophic_collapse"
    SUDDEN_UPLIFT = "sudden_uplift"
    CLASS_OUTLIER_BURST = "class_outlier_burst"
    INTERMITTENT_FAILURE = "intermittent_failure"
    RECOVERY_SPIKE = "recovery_spike"


@dataclass
class RareEventChannel:
    """
    Configuration for a rare event channel.
    
    Rare events are deterministically triggered based on seed mixing,
    allowing reproducible "black swan" scenarios in synthetic data.
    
    Attributes:
        event_type: Type of rare event
        trigger_probability: Per-cycle probability of triggering (0.0 to 1.0)
        trigger_cycles: Specific cycles at which event triggers (overrides probability)
        duration: Number of cycles the event lasts
        magnitude: Strength of the event effect (-1.0 to 1.0)
        affected_classes: Classes affected (None = all classes)
        recovery_rate: Rate at which effect decays after event (0.0 = instant, 1.0 = permanent)
    """
    event_type: RareEventType
    trigger_probability: float = 0.0
    trigger_cycles: List[int] = field(default_factory=list)
    duration: int = 1
    magnitude: float = 0.0
    affected_classes: Optional[List[str]] = None
    recovery_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "trigger_probability": self.trigger_probability,
            "trigger_cycles": self.trigger_cycles,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "affected_classes": self.affected_classes,
            "recovery_rate": self.recovery_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RareEventChannel":
        """Deserialize from dictionary."""
        return cls(
            event_type=RareEventType(data["event_type"]),
            trigger_probability=data.get("trigger_probability", 0.0),
            trigger_cycles=data.get("trigger_cycles", []),
            duration=data.get("duration", 1),
            magnitude=data.get("magnitude", 0.0),
            affected_classes=data.get("affected_classes"),
            recovery_rate=data.get("recovery_rate", 0.0),
        )


# ==============================================================================
# VARIANCE PARAMETERS
# ==============================================================================

@dataclass
class VarianceConfig:
    """
    Configuration for probability variance/noise.
    
    Adds per-cycle and per-item variance to base probabilities.
    
    Attributes:
        per_cycle_sigma: Standard deviation of per-cycle noise
        per_item_sigma: Standard deviation of per-item noise
        heteroscedastic: If True, variance scales with base probability
        min_prob: Minimum probability after variance (default 0.01)
        max_prob: Maximum probability after variance (default 0.99)
    """
    per_cycle_sigma: float = 0.0
    per_item_sigma: float = 0.0
    heteroscedastic: bool = False
    min_prob: float = 0.01
    max_prob: float = 0.99
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "per_cycle_sigma": self.per_cycle_sigma,
            "per_item_sigma": self.per_item_sigma,
            "heteroscedastic": self.heteroscedastic,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VarianceConfig":
        """Deserialize from dictionary."""
        return cls(
            per_cycle_sigma=data.get("per_cycle_sigma", 0.0),
            per_item_sigma=data.get("per_item_sigma", 0.0),
            heteroscedastic=data.get("heteroscedastic", False),
            min_prob=data.get("min_prob", 0.01),
            max_prob=data.get("max_prob", 0.99),
        )


# ==============================================================================
# ITEM SPECIFICATION
# ==============================================================================

@dataclass
class ItemSpec:
    """
    Specification for a synthetic item.
    
    Attributes:
        id: Unique item identifier
        item_class: Class membership (e.g., "class_a")
        complexity: Complexity level (affects base probability)
        weight: Sampling weight for item selection
        custom_prob_offset: Per-item probability offset
    """
    id: str
    item_class: str
    complexity: int = 1
    weight: float = 1.0
    custom_prob_offset: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "class": self.item_class,
            "complexity": self.complexity,
            "weight": self.weight,
            "custom_prob_offset": self.custom_prob_offset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemSpec":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            item_class=data.get("class", "class_a"),
            complexity=data.get("complexity", 1),
            weight=data.get("weight", 1.0),
            custom_prob_offset=data.get("custom_prob_offset", 0.0),
        )


# ==============================================================================
# PROBABILITY MATRIX
# ==============================================================================

@dataclass
class ProbabilityMatrix:
    """
    Base probability matrix for modes and classes.
    
    Structure: mode -> class -> probability
    """
    baseline: Dict[str, float] = field(default_factory=dict)
    rfl: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize to dictionary."""
        return {
            "baseline": self.baseline,
            "rfl": self.rfl,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityMatrix":
        """Deserialize from dictionary."""
        return cls(
            baseline=data.get("baseline", {}),
            rfl=data.get("rfl", {}),
        )
    
    def get_probability(self, mode: str, item_class: str) -> float:
        """Get probability for a mode and class."""
        probs = self.baseline if mode == "baseline" else self.rfl
        return probs.get(item_class, 0.5)
    
    def get_classes(self) -> List[str]:
        """Get all defined classes."""
        return list(set(self.baseline.keys()) | set(self.rfl.keys()))


# ==============================================================================
# NOISE UNIVERSE SPECIFICATION (NoiseSpec)
# ==============================================================================

@dataclass
class NoiseSpec:
    """
    Complete specification for a synthetic noise universe.
    
    A NoiseSpec fully describes a parametric universe that can be compiled
    into a deterministic synthetic log generator.
    
    Attributes:
        name: Universe name (must start with 'synthetic_')
        description: Human-readable description
        version: Schema version
        seed: Master random seed for determinism
        num_cycles: Number of cycles to generate
        num_items_per_class: Number of items per class (auto-generated)
        classes: List of class names
        
        probabilities: Base probability matrix
        drift: Temporal drift configuration
        correlation: Class correlation configuration
        variance: Variance/noise parameters
        rare_events: List of rare event channels
        
        items: Explicit item list (optional, overrides auto-generation)
        metadata: Additional metadata
    """
    # Identity
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Dimensions
    seed: int = 42
    num_cycles: int = 500
    num_items_per_class: int = 3
    classes: List[str] = field(default_factory=lambda: ["class_a", "class_b", "class_c"])
    
    # Probability model
    probabilities: ProbabilityMatrix = field(default_factory=ProbabilityMatrix)
    
    # Noise models
    drift: DriftConfig = field(default_factory=DriftConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    variance: VarianceConfig = field(default_factory=VarianceConfig)
    
    # Rare event channels
    rare_events: List[RareEventChannel] = field(default_factory=list)
    
    # Items (optional explicit list)
    items: Optional[List[ItemSpec]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate spec after initialization."""
        if not self.name.startswith("synthetic_"):
            raise ValueError(f"Universe name must start with 'synthetic_': {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": SAFETY_LABEL,
            "schema_version": "noise_spec_v1",
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "seed": self.seed,
            "num_cycles": self.num_cycles,
            "num_items_per_class": self.num_items_per_class,
            "classes": self.classes,
            "probabilities": self.probabilities.to_dict(),
            "drift": self.drift.to_dict(),
            "correlation": self.correlation.to_dict(),
            "variance": self.variance.to_dict(),
            "rare_events": [e.to_dict() for e in self.rare_events],
            "items": [i.to_dict() for i in self.items] if self.items else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseSpec":
        """Deserialize from dictionary."""
        items = None
        if data.get("items"):
            items = [ItemSpec.from_dict(i) for i in data["items"]]
        
        rare_events = []
        if data.get("rare_events"):
            rare_events = [RareEventChannel.from_dict(e) for e in data["rare_events"]]
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            seed=data.get("seed", 42),
            num_cycles=data.get("num_cycles", 500),
            num_items_per_class=data.get("num_items_per_class", 3),
            classes=data.get("classes", ["class_a", "class_b", "class_c"]),
            probabilities=ProbabilityMatrix.from_dict(data.get("probabilities", {})),
            drift=DriftConfig.from_dict(data.get("drift", {})),
            correlation=CorrelationConfig.from_dict(data.get("correlation", {})),
            variance=VarianceConfig.from_dict(data.get("variance", {})),
            rare_events=rare_events,
            items=items,
            metadata=data.get("metadata", {}),
        )
    
    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "NoiseSpec":
        """Deserialize from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    def save(self, path) -> None:
        """Save spec to file (JSON or YAML based on extension)."""
        from pathlib import Path
        path = Path(path)
        
        if path.suffix in (".yaml", ".yml"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_yaml())
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path) -> "NoiseSpec":
        """Load spec from file."""
        from pathlib import Path
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of the spec."""
        spec_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(spec_str.encode("utf-8")).hexdigest()[:16]
    
    def get_items(self) -> List[ItemSpec]:
        """Get item list (auto-generate if not explicitly provided)."""
        if self.items:
            return self.items
        
        # Auto-generate items
        items = []
        for cls in self.classes:
            for i in range(self.num_items_per_class):
                items.append(ItemSpec(
                    id=f"{self.name}_{cls}_{i:02d}",
                    item_class=cls,
                    complexity=self.classes.index(cls) + 1,
                ))
        return items


# ==============================================================================
# RARE EVENT ENGINE
# ==============================================================================

class RareEventEngine:
    """
    Engine for processing rare event channels.
    
    Deterministically triggers and applies rare events based on
    seed mixing and channel configuration.
    """
    
    def __init__(self, channels: List[RareEventChannel], master_seed: int):
        self.channels = channels
        self.master_seed = master_seed
        
        # Track active events: channel_idx -> (start_cycle, remaining_duration)
        self._active_events: Dict[int, Tuple[int, int]] = {}
        
        # Pre-compute triggered cycles for determinism
        self._triggered_cycles: Dict[int, List[int]] = {}
        self._precompute_triggers()
    
    def _precompute_triggers(self, max_cycles: int = 10000):
        """Pre-compute which cycles each channel triggers on."""
        for idx, channel in enumerate(self.channels):
            triggered = set(channel.trigger_cycles)
            
            # Add probabilistic triggers
            if channel.trigger_probability > 0:
                rng = random.Random(self.master_seed ^ (idx * 0x12345678))
                for cycle in range(max_cycles):
                    if rng.random() < channel.trigger_probability:
                        triggered.add(cycle)
            
            self._triggered_cycles[idx] = sorted(triggered)
    
    def get_active_effects(
        self,
        cycle: int,
        item_class: str,
    ) -> List[Tuple[RareEventChannel, float]]:
        """
        Get all active rare event effects for a cycle and class.
        
        Returns list of (channel, effective_magnitude) tuples.
        """
        effects = []
        
        for idx, channel in enumerate(self.channels):
            # Check if this cycle triggers the event
            if cycle in self._triggered_cycles.get(idx, []):
                self._active_events[idx] = (cycle, channel.duration)
            
            # Check if event is active
            if idx in self._active_events:
                start_cycle, remaining = self._active_events[idx]
                cycles_elapsed = cycle - start_cycle
                
                if cycles_elapsed < channel.duration:
                    # Event is active
                    # Check if this class is affected
                    if channel.affected_classes is None or item_class in channel.affected_classes:
                        # Apply recovery decay
                        if channel.recovery_rate > 0 and cycles_elapsed > 0:
                            decay = math.exp(-channel.recovery_rate * cycles_elapsed)
                            effective_mag = channel.magnitude * decay
                        else:
                            effective_mag = channel.magnitude
                        
                        effects.append((channel, effective_mag))
                else:
                    # Event has ended
                    del self._active_events[idx]
        
        return effects
    
    def apply_effects(
        self,
        base_prob: float,
        cycle: int,
        item_class: str,
    ) -> Tuple[float, List[str]]:
        """
        Apply all active rare event effects to a probability.
        
        Returns:
            (modified_probability, list_of_active_event_names)
        """
        effects = self.get_active_effects(cycle, item_class)
        
        if not effects:
            return base_prob, []
        
        modified_prob = base_prob
        event_names = []
        
        for channel, magnitude in effects:
            modified_prob += magnitude
            event_names.append(channel.event_type.value)
        
        # Clamp to valid range
        modified_prob = max(0.01, min(0.99, modified_prob))
        
        return modified_prob, event_names
    
    def is_event_active(self, cycle: int, event_type: RareEventType) -> bool:
        """Check if a specific event type is active at a cycle."""
        for idx, channel in enumerate(self.channels):
            if channel.event_type == event_type:
                if idx in self._active_events:
                    start_cycle, _ = self._active_events[idx]
                    if cycle - start_cycle < channel.duration:
                        return True
        return False


# ==============================================================================
# FACTORY FUNCTIONS FOR RARE EVENTS
# ==============================================================================

def create_catastrophic_collapse(
    trigger_cycle: int = 300,
    magnitude: float = -0.60,
    duration: int = 100,
    recovery_rate: float = 0.0,
) -> RareEventChannel:
    """
    Create a catastrophic collapse event.
    
    Simulates sudden system failure where success probability drops dramatically.
    """
    return RareEventChannel(
        event_type=RareEventType.CATASTROPHIC_COLLAPSE,
        trigger_cycles=[trigger_cycle],
        magnitude=magnitude,
        duration=duration,
        recovery_rate=recovery_rate,
    )


def create_sudden_uplift(
    trigger_cycle: int = 200,
    magnitude: float = 0.30,
    duration: int = 50,
    recovery_rate: float = 0.02,
) -> RareEventChannel:
    """
    Create a sudden uplift event.
    
    Simulates a breakthrough that temporarily boosts success probability.
    """
    return RareEventChannel(
        event_type=RareEventType.SUDDEN_UPLIFT,
        trigger_cycles=[trigger_cycle],
        magnitude=magnitude,
        duration=duration,
        recovery_rate=recovery_rate,
    )


def create_class_outlier_burst(
    affected_class: str,
    trigger_probability: float = 0.02,
    magnitude: float = -0.40,
    duration: int = 5,
) -> RareEventChannel:
    """
    Create a class-specific outlier burst event.
    
    Simulates sporadic failures affecting only one class.
    """
    return RareEventChannel(
        event_type=RareEventType.CLASS_OUTLIER_BURST,
        trigger_probability=trigger_probability,
        magnitude=magnitude,
        duration=duration,
        affected_classes=[affected_class],
    )


def create_intermittent_failure(
    trigger_probability: float = 0.05,
    magnitude: float = -0.50,
    duration: int = 3,
) -> RareEventChannel:
    """
    Create an intermittent failure event.
    
    Simulates random transient failures.
    """
    return RareEventChannel(
        event_type=RareEventType.INTERMITTENT_FAILURE,
        trigger_probability=trigger_probability,
        magnitude=magnitude,
        duration=duration,
    )


def create_recovery_spike(
    trigger_cycle: int,
    magnitude: float = 0.25,
    duration: int = 20,
    recovery_rate: float = 0.05,
) -> RareEventChannel:
    """
    Create a recovery spike event.
    
    Simulates temporary recovery after a failure.
    """
    return RareEventChannel(
        event_type=RareEventType.RECOVERY_SPIKE,
        trigger_cycles=[trigger_cycle],
        magnitude=magnitude,
        duration=duration,
        recovery_rate=recovery_rate,
    )


# ==============================================================================
# SPEC BUILDER
# ==============================================================================

class NoiseSpecBuilder:
    """
    Fluent builder for NoiseSpec objects.
    
    Usage:
        spec = (NoiseSpecBuilder("synthetic_my_universe")
            .with_probabilities(baseline={"class_a": 0.7}, rfl={"class_a": 0.8})
            .with_drift(mode="cyclical", amplitude=0.1, period=100)
            .with_correlation(rho=0.3)
            .with_rare_event(create_catastrophic_collapse(trigger_cycle=250))
            .build())
    """
    
    def __init__(self, name: str, description: str = ""):
        self._spec = NoiseSpec(name=name, description=description)
    
    def with_seed(self, seed: int) -> "NoiseSpecBuilder":
        """Set master random seed."""
        self._spec.seed = seed
        return self
    
    def with_cycles(self, num_cycles: int) -> "NoiseSpecBuilder":
        """Set number of cycles."""
        self._spec.num_cycles = num_cycles
        return self
    
    def with_classes(self, classes: List[str], items_per_class: int = 3) -> "NoiseSpecBuilder":
        """Set class names and items per class."""
        self._spec.classes = classes
        self._spec.num_items_per_class = items_per_class
        return self
    
    def with_probabilities(
        self,
        baseline: Dict[str, float],
        rfl: Optional[Dict[str, float]] = None,
    ) -> "NoiseSpecBuilder":
        """Set base probability matrix."""
        if rfl is None:
            rfl = baseline.copy()
        self._spec.probabilities = ProbabilityMatrix(baseline=baseline, rfl=rfl)
        return self
    
    def with_drift(
        self,
        mode: str = "none",
        amplitude: float = 0.0,
        period: int = 100,
        slope: float = 0.0,
        shock_cycle: int = 0,
        shock_delta: float = 0.0,
        direction: str = "up",
    ) -> "NoiseSpecBuilder":
        """Set drift configuration."""
        self._spec.drift = DriftConfig(
            mode=DriftMode(mode),
            amplitude=amplitude,
            period=period,
            slope=slope,
            shock_cycle=shock_cycle,
            shock_delta=shock_delta,
            direction=direction,
        )
        return self
    
    def with_correlation(
        self,
        rho: float = 0.0,
        mode: str = "class",
    ) -> "NoiseSpecBuilder":
        """Set correlation configuration."""
        self._spec.correlation = CorrelationConfig(rho=rho, mode=mode)
        return self
    
    def with_variance(
        self,
        per_cycle_sigma: float = 0.0,
        per_item_sigma: float = 0.0,
        heteroscedastic: bool = False,
    ) -> "NoiseSpecBuilder":
        """Set variance configuration."""
        self._spec.variance = VarianceConfig(
            per_cycle_sigma=per_cycle_sigma,
            per_item_sigma=per_item_sigma,
            heteroscedastic=heteroscedastic,
        )
        return self
    
    def with_rare_event(self, event: RareEventChannel) -> "NoiseSpecBuilder":
        """Add a rare event channel."""
        self._spec.rare_events.append(event)
        return self
    
    def with_items(self, items: List[ItemSpec]) -> "NoiseSpecBuilder":
        """Set explicit item list."""
        self._spec.items = items
        return self
    
    def with_metadata(self, **kwargs) -> "NoiseSpecBuilder":
        """Add metadata."""
        self._spec.metadata.update(kwargs)
        return self
    
    def build(self) -> NoiseSpec:
        """Build and return the NoiseSpec."""
        return self._spec


# ==============================================================================
# VALIDATION
# ==============================================================================

class SpecValidationError(Exception):
    """Raised when a NoiseSpec fails validation."""
    pass


def validate_spec(spec: NoiseSpec) -> List[str]:
    """
    Validate a NoiseSpec for consistency and correctness.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Name validation
    if not spec.name.startswith("synthetic_"):
        errors.append(f"Name must start with 'synthetic_': {spec.name}")
    
    # Probability validation
    for mode in ["baseline", "rfl"]:
        probs = getattr(spec.probabilities, mode)
        for cls, prob in probs.items():
            if not 0.0 <= prob <= 1.0:
                errors.append(f"Invalid probability for {mode}/{cls}: {prob}")
    
    # Class consistency
    defined_classes = set(spec.classes)
    prob_classes = set(spec.probabilities.get_classes())
    
    if not prob_classes.issubset(defined_classes):
        extra = prob_classes - defined_classes
        errors.append(f"Probabilities defined for undefined classes: {extra}")
    
    # Drift validation
    if spec.drift.mode == DriftMode.CYCLICAL and spec.drift.period <= 0:
        errors.append("Cyclical drift requires period > 0")
    
    if spec.drift.mode == DriftMode.SHOCK:
        if spec.drift.shock_cycle < 0:
            errors.append("Shock cycle must be >= 0")
        if spec.drift.shock_cycle >= spec.num_cycles:
            errors.append(f"Shock cycle ({spec.drift.shock_cycle}) >= num_cycles ({spec.num_cycles})")
    
    # Correlation validation
    if not 0.0 <= spec.correlation.rho <= 1.0:
        errors.append(f"Correlation rho must be in [0, 1]: {spec.correlation.rho}")
    
    # Variance validation
    if spec.variance.per_cycle_sigma < 0:
        errors.append("per_cycle_sigma must be >= 0")
    if spec.variance.per_item_sigma < 0:
        errors.append("per_item_sigma must be >= 0")
    
    # Rare event validation
    for i, event in enumerate(spec.rare_events):
        if event.trigger_probability < 0 or event.trigger_probability > 1:
            errors.append(f"Rare event {i}: trigger_probability must be in [0, 1]")
        if event.duration < 1:
            errors.append(f"Rare event {i}: duration must be >= 1")
        if event.affected_classes:
            invalid_classes = set(event.affected_classes) - defined_classes
            if invalid_classes:
                errors.append(f"Rare event {i}: affected classes not defined: {invalid_classes}")
    
    # Dimension validation
    if spec.num_cycles < 1:
        errors.append("num_cycles must be >= 1")
    if spec.num_items_per_class < 1:
        errors.append("num_items_per_class must be >= 1")
    if len(spec.classes) < 1:
        errors.append("At least one class must be defined")
    
    return errors

