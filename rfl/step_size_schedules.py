"""
RFL Step-Size Schedules Module
===============================

Implements adaptive step-size schedules η_t for RFL policy updates.

The step-size schedule controls the magnitude of policy updates at each
epoch, balancing exploration (large steps) and exploitation (small steps).

Supported schedules:
- Constant: η_t = η_0 (fixed learning rate)
- Linear decay: η_t = η_0 * (1 - t/T)
- Exponential decay: η_t = η_0 * exp(-λt)
- Inverse sqrt: η_t = η_0 / sqrt(1 + t)
- Cosine annealing: η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(πt/T))
- Adaptive: η_t based on gradient norm and convergence metrics

Usage:
    from rfl.step_size_schedules import (
        StepSizeSchedule,
        ConstantSchedule,
        ExponentialDecaySchedule,
        AdaptiveSchedule,
    )

    # Create schedule
    schedule = ExponentialDecaySchedule(initial_rate=0.1, decay_rate=0.01)

    # Get step size for epoch t
    eta_t = schedule.get_step_size(epoch=10, gradient_norm=0.5)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class StepSizeSchedule(ABC):
    """
    Abstract base class for step-size schedules.
    
    All schedules must implement get_step_size() which returns η_t
    for a given epoch and optional context (gradient norm, loss, etc.).
    """
    
    @abstractmethod
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute step size η_t for epoch t.
        
        Args:
            epoch: Current epoch (0-indexed)
            gradient_norm: L2 norm of gradient ||Φ||
            context: Optional additional context (loss, metrics, etc.)
        
        Returns:
            Step size η_t in [0, 1]
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schedule to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepSizeSchedule:
        """Deserialize schedule from dictionary."""
        pass


@dataclass
class ConstantSchedule(StepSizeSchedule):
    """
    Constant step-size schedule: η_t = η_0.
    
    Simplest schedule, useful for stable environments where
    optimal step size is known a priori.
    
    Attributes:
        learning_rate: Fixed learning rate η_0
    """
    learning_rate: float = 0.1
    
    def __post_init__(self):
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in [0, 1], got {self.learning_rate}")
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self.learning_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "constant",
            "learning_rate": self.learning_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstantSchedule:
        return cls(learning_rate=data["learning_rate"])


@dataclass
class LinearDecaySchedule(StepSizeSchedule):
    """
    Linear decay schedule: η_t = η_0 * (1 - t/T).
    
    Linearly decreases step size from η_0 to 0 over T epochs.
    Useful for finite-horizon optimization.
    
    Attributes:
        initial_rate: Initial learning rate η_0
        total_epochs: Total number of epochs T
        min_rate: Minimum learning rate (floor)
    """
    initial_rate: float = 0.1
    total_epochs: int = 100
    min_rate: float = 0.0
    
    def __post_init__(self):
        if not (0.0 <= self.initial_rate <= 1.0):
            raise ValueError(f"initial_rate must be in [0, 1], got {self.initial_rate}")
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {self.total_epochs}")
        if not (0.0 <= self.min_rate < self.initial_rate):
            raise ValueError(f"min_rate must be in [0, initial_rate), got {self.min_rate}")
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if epoch >= self.total_epochs:
            return self.min_rate
        
        decay_factor = 1.0 - (epoch / self.total_epochs)
        eta_t = self.initial_rate * decay_factor
        return max(self.min_rate, eta_t)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "linear_decay",
            "initial_rate": self.initial_rate,
            "total_epochs": self.total_epochs,
            "min_rate": self.min_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinearDecaySchedule:
        return cls(
            initial_rate=data["initial_rate"],
            total_epochs=data["total_epochs"],
            min_rate=data.get("min_rate", 0.0),
        )


@dataclass
class ExponentialDecaySchedule(StepSizeSchedule):
    """
    Exponential decay schedule: η_t = η_0 * exp(-λt).
    
    Exponentially decreases step size over time. Decay rate λ controls
    how quickly the step size shrinks.
    
    Attributes:
        initial_rate: Initial learning rate η_0
        decay_rate: Decay rate λ (larger = faster decay)
        min_rate: Minimum learning rate (floor)
    """
    initial_rate: float = 0.1
    decay_rate: float = 0.01
    min_rate: float = 0.001
    
    def __post_init__(self):
        if not (0.0 <= self.initial_rate <= 1.0):
            raise ValueError(f"initial_rate must be in [0, 1], got {self.initial_rate}")
        if self.decay_rate < 0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")
        if not (0.0 <= self.min_rate < self.initial_rate):
            raise ValueError(f"min_rate must be in [0, initial_rate), got {self.min_rate}")
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        eta_t = self.initial_rate * math.exp(-self.decay_rate * epoch)
        return max(self.min_rate, eta_t)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "exponential_decay",
            "initial_rate": self.initial_rate,
            "decay_rate": self.decay_rate,
            "min_rate": self.min_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExponentialDecaySchedule:
        return cls(
            initial_rate=data["initial_rate"],
            decay_rate=data["decay_rate"],
            min_rate=data.get("min_rate", 0.001),
        )


@dataclass
class InverseSqrtSchedule(StepSizeSchedule):
    """
    Inverse square root schedule: η_t = η_0 / sqrt(1 + t).
    
    Commonly used in transformer training. Decays slower than exponential
    in early epochs, faster in later epochs.
    
    Attributes:
        initial_rate: Initial learning rate η_0
        warmup_epochs: Number of warmup epochs (constant rate)
        min_rate: Minimum learning rate (floor)
    """
    initial_rate: float = 0.1
    warmup_epochs: int = 0
    min_rate: float = 0.001
    
    def __post_init__(self):
        if not (0.0 <= self.initial_rate <= 1.0):
            raise ValueError(f"initial_rate must be in [0, 1], got {self.initial_rate}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if not (0.0 <= self.min_rate < self.initial_rate):
            raise ValueError(f"min_rate must be in [0, initial_rate), got {self.min_rate}")
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if epoch < self.warmup_epochs:
            return self.initial_rate
        
        adjusted_epoch = epoch - self.warmup_epochs + 1
        eta_t = self.initial_rate / math.sqrt(adjusted_epoch)
        return max(self.min_rate, eta_t)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "inverse_sqrt",
            "initial_rate": self.initial_rate,
            "warmup_epochs": self.warmup_epochs,
            "min_rate": self.min_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InverseSqrtSchedule:
        return cls(
            initial_rate=data["initial_rate"],
            warmup_epochs=data.get("warmup_epochs", 0),
            min_rate=data.get("min_rate", 0.001),
        )


@dataclass
class CosineAnnealingSchedule(StepSizeSchedule):
    """
    Cosine annealing schedule: η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(πt/T)).
    
    Smoothly anneals learning rate from η_max to η_min following a cosine curve.
    Popular in deep learning for smooth convergence.
    
    Attributes:
        max_rate: Maximum learning rate η_max
        min_rate: Minimum learning rate η_min
        total_epochs: Total number of epochs T
    """
    max_rate: float = 0.1
    min_rate: float = 0.001
    total_epochs: int = 100
    
    def __post_init__(self):
        if not (0.0 <= self.max_rate <= 1.0):
            raise ValueError(f"max_rate must be in [0, 1], got {self.max_rate}")
        if not (0.0 <= self.min_rate < self.max_rate):
            raise ValueError(f"min_rate must be in [0, max_rate), got {self.min_rate}")
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {self.total_epochs}")
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if epoch >= self.total_epochs:
            return self.min_rate
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * epoch / self.total_epochs))
        eta_t = self.min_rate + (self.max_rate - self.min_rate) * cosine_factor
        return eta_t
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "cosine_annealing",
            "max_rate": self.max_rate,
            "min_rate": self.min_rate,
            "total_epochs": self.total_epochs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CosineAnnealingSchedule:
        return cls(
            max_rate=data["max_rate"],
            min_rate=data["min_rate"],
            total_epochs=data["total_epochs"],
        )


@dataclass
class AdaptiveSchedule(StepSizeSchedule):
    """
    Adaptive schedule: η_t based on gradient norm and convergence metrics.
    
    Adjusts step size dynamically based on:
    - Gradient norm (larger gradients → smaller steps)
    - Recent loss/metric trends (improving → maintain, worsening → reduce)
    - Convergence detection (plateau → increase for exploration)
    
    Attributes:
        initial_rate: Initial learning rate
        min_rate: Minimum learning rate
        max_rate: Maximum learning rate
        gradient_scale: Scaling factor for gradient norm
        patience: Number of epochs to wait before adapting
        adaptation_factor: Multiplicative factor for rate changes
    """
    initial_rate: float = 0.1
    min_rate: float = 0.001
    max_rate: float = 0.5
    gradient_scale: float = 1.0
    patience: int = 5
    adaptation_factor: float = 0.5
    
    # Internal state (mutable)
    _current_rate: float = field(default=0.1, init=False, repr=False)
    _best_metric: Optional[float] = field(default=None, init=False, repr=False)
    _epochs_without_improvement: int = field(default=0, init=False, repr=False)
    _metric_history: List[float] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if not (0.0 <= self.initial_rate <= 1.0):
            raise ValueError(f"initial_rate must be in [0, 1], got {self.initial_rate}")
        if not (0.0 <= self.min_rate < self.max_rate <= 1.0):
            raise ValueError(f"min_rate < max_rate required, got {self.min_rate}, {self.max_rate}")
        if self.gradient_scale <= 0:
            raise ValueError(f"gradient_scale must be > 0, got {self.gradient_scale}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        
        self._current_rate = self.initial_rate
    
    def get_step_size(
        self,
        epoch: int,
        gradient_norm: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute adaptive step size based on gradient and metrics.
        
        Context keys:
            - abstention_rate: Current abstention rate (lower is better)
            - verified_count: Number of verified proofs (higher is better)
        """
        context = context or {}
        
        # Base rate adjustment based on gradient norm
        # Larger gradients → smaller steps (stability)
        if gradient_norm > 0:
            gradient_factor = 1.0 / (1.0 + self.gradient_scale * gradient_norm)
            base_rate = self._current_rate * gradient_factor
        else:
            base_rate = self._current_rate
        
        # Metric-based adaptation
        # Use abstention_rate as primary metric (lower is better)
        current_metric = context.get("abstention_rate")
        
        if current_metric is not None:
            self._metric_history.append(current_metric)
            
            # Check for improvement
            if self._best_metric is None or current_metric < self._best_metric:
                # Improvement: maintain or slightly increase rate
                self._best_metric = current_metric
                self._epochs_without_improvement = 0
                self._current_rate = min(self.max_rate, self._current_rate * 1.05)
            else:
                # No improvement: count epochs
                self._epochs_without_improvement += 1
                
                # If patience exceeded, reduce rate
                if self._epochs_without_improvement >= self.patience:
                    self._current_rate = max(
                        self.min_rate,
                        self._current_rate * self.adaptation_factor
                    )
                    self._epochs_without_improvement = 0  # Reset counter
        
        # Clamp to bounds
        eta_t = max(self.min_rate, min(self.max_rate, base_rate))
        return eta_t
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "adaptive",
            "initial_rate": self.initial_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "gradient_scale": self.gradient_scale,
            "patience": self.patience,
            "adaptation_factor": self.adaptation_factor,
            "current_rate": self._current_rate,
            "best_metric": self._best_metric,
            "epochs_without_improvement": self._epochs_without_improvement,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveSchedule:
        schedule = cls(
            initial_rate=data["initial_rate"],
            min_rate=data["min_rate"],
            max_rate=data["max_rate"],
            gradient_scale=data.get("gradient_scale", 1.0),
            patience=data.get("patience", 5),
            adaptation_factor=data.get("adaptation_factor", 0.5),
        )
        
        # Restore internal state if present
        if "current_rate" in data:
            schedule._current_rate = data["current_rate"]
        if "best_metric" in data:
            schedule._best_metric = data["best_metric"]
        if "epochs_without_improvement" in data:
            schedule._epochs_without_improvement = data["epochs_without_improvement"]
        
        return schedule


# -----------------------------------------------------------------------------
# Schedule Factory
# -----------------------------------------------------------------------------

def create_schedule(schedule_type: str, **kwargs) -> StepSizeSchedule:
    """
    Factory function to create step-size schedules.
    
    Args:
        schedule_type: Type of schedule ("constant", "exponential_decay", etc.)
        **kwargs: Schedule-specific parameters
    
    Returns:
        StepSizeSchedule instance
    
    Raises:
        ValueError: If schedule_type is unknown
    """
    schedule_map = {
        "constant": ConstantSchedule,
        "linear_decay": LinearDecaySchedule,
        "exponential_decay": ExponentialDecaySchedule,
        "inverse_sqrt": InverseSqrtSchedule,
        "cosine_annealing": CosineAnnealingSchedule,
        "adaptive": AdaptiveSchedule,
    }
    
    if schedule_type not in schedule_map:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Available: {list(schedule_map.keys())}"
        )
    
    return schedule_map[schedule_type](**kwargs)


def load_schedule(data: Dict[str, Any]) -> StepSizeSchedule:
    """
    Load schedule from dictionary.
    
    Args:
        data: Dictionary with "type" key and schedule parameters
    
    Returns:
        StepSizeSchedule instance
    """
    schedule_type = data.get("type")
    if not schedule_type:
        raise ValueError("Schedule data must contain 'type' field")
    
    schedule_map = {
        "constant": ConstantSchedule,
        "linear_decay": LinearDecaySchedule,
        "exponential_decay": ExponentialDecaySchedule,
        "inverse_sqrt": InverseSqrtSchedule,
        "cosine_annealing": CosineAnnealingSchedule,
        "adaptive": AdaptiveSchedule,
    }
    
    if schedule_type not in schedule_map:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return schedule_map[schedule_type].from_dict(data)


__all__ = [
    "StepSizeSchedule",
    "ConstantSchedule",
    "LinearDecaySchedule",
    "ExponentialDecaySchedule",
    "InverseSqrtSchedule",
    "CosineAnnealingSchedule",
    "AdaptiveSchedule",
    "create_schedule",
    "load_schedule",
]
