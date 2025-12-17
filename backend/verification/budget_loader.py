"""
Budget Configuration Loader — Load and Validate Verifier Budgets

This module loads and validates verifier budget configuration from YAML files.
Provides type-safe access to budget parameters for Phase II slices.

Author: Agent B1 (Budget Enforcement Architect)
Date: 2025-01-XX
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "verifier_budget_phase2.yaml"


@dataclass(frozen=True, slots=True)
class VerifierBudget:
    """Budget configuration for verifier operations."""
    
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int
    
    def __post_init__(self) -> None:
        """Validate budget parameters."""
        if self.cycle_budget_s < 0.0:
            raise ValueError(f"cycle_budget_s must be non-negative, got {self.cycle_budget_s}")
        if self.taut_timeout_s < 0.0:
            raise ValueError(f"taut_timeout_s must be non-negative, got {self.taut_timeout_s}")
        if self.max_candidates_per_cycle < 1:
            raise ValueError(
                f"max_candidates_per_cycle must be >= 1, got {self.max_candidates_per_cycle}"
            )
        # Reasonable upper bounds
        if self.cycle_budget_s > 3600.0:
            raise ValueError(f"cycle_budget_s exceeds reasonable limit (3600s), got {self.cycle_budget_s}")
        if self.taut_timeout_s > 60.0:
            raise ValueError(f"taut_timeout_s exceeds reasonable limit (60s), got {self.taut_timeout_s}")
        if self.max_candidates_per_cycle > 10000:
            raise ValueError(
                f"max_candidates_per_cycle exceeds reasonable limit (10000), got {self.max_candidates_per_cycle}"
            )
    
    @property
    def max_candidates(self) -> int:
        """Alias for max_candidates_per_cycle for pipeline compatibility."""
        return self.max_candidates_per_cycle
    
    @property
    def taut_timeout_ms(self) -> int:
        """Convert taut_timeout_s to milliseconds (rounded down)."""
        return int(self.taut_timeout_s * 1000.0)


def load_budget_for_slice(
    slice_name: str,
    path: Path | str | None = None,
) -> VerifierBudget:
    """
    Load budget configuration for a specific slice.
    
    Loads YAML config, merges defaults with slice-specific overrides,
    and returns a validated VerifierBudget object.
    
    Args:
        slice_name: Name of the slice (e.g., "slice_uplift_goal")
        path: Path to budget config YAML file (default: config/verifier_budget_phase2.yaml)
    
    Returns:
        VerifierBudget with merged defaults and slice overrides
    
    Raises:
        FileNotFoundError: If config file does not exist
        KeyError: If defaults or requested slice is missing
        ValueError: If budget values are invalid (negative, wrong type, etc.)
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Budget config not found at {config_path}. "
            "Please create config/verifier_budget_phase2.yaml"
        )
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid config format: expected dict, got {type(config_data).__name__}")
    
    # Load defaults
    defaults = config_data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise KeyError("Config missing 'defaults' section")
    
    # Validate default keys
    required_keys = {"cycle_budget_s", "taut_timeout_s", "max_candidates_per_cycle"}
    missing_keys = required_keys - set(defaults.keys())
    if missing_keys:
        raise KeyError(f"Defaults missing required keys: {missing_keys}")
    
    # Load slice-specific overrides
    slices = config_data.get("slices", {})
    if not isinstance(slices, dict):
        raise KeyError("Config missing 'slices' section")
    
    if slice_name not in slices:
        raise KeyError(
            f"Slice '{slice_name}' not found in config. Available slices: {list(slices.keys())}"
        )
    
    slice_config = slices[slice_name]
    if not isinstance(slice_config, dict):
        raise ValueError(
            f"Invalid slice config for '{slice_name}': expected dict, got {type(slice_config).__name__}"
        )
    
    # Merge: slice overrides defaults
    merged = {**defaults, **slice_config}
    
    # Extract and validate types
    try:
        cycle_budget_s = float(merged["cycle_budget_s"])
        taut_timeout_s = float(merged["taut_timeout_s"])
        max_candidates_per_cycle = int(merged["max_candidates_per_cycle"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid budget value types for slice '{slice_name}': {e}. "
            f"Expected cycle_budget_s: float, taut_timeout_s: float, max_candidates_per_cycle: int"
        ) from e
    
    return VerifierBudget(
        cycle_budget_s=cycle_budget_s,
        taut_timeout_s=taut_timeout_s,
        max_candidates_per_cycle=max_candidates_per_cycle,
    )


def is_phase2_slice(slice_name: str) -> bool:
    """
    Check if a slice name indicates a Phase II slice.
    
    Phase II slices follow the pattern: slice_uplift_*
    
    Args:
        slice_name: Name of the slice
    
    Returns:
        True if slice_name matches Phase II pattern
    """
    return slice_name.startswith("slice_uplift_")

