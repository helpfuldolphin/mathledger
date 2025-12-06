# PHASE II — NOT USED IN PHASE I
# Budget enforcement for Phase II slices

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml

DEFAULT_CONFIG_PATH = Path("config/verifier_budget_phase2.yaml")


@dataclass
class VerifierBudget:
    """Budget configuration for verifier."""
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int


def is_phase2_slice(slice_name: str) -> bool:
    """
    Check if slice is a Phase II slice.
    
    Args:
        slice_name: Name of the slice
        
    Returns:
        True if slice requires Phase II budget enforcement
    """
    # Phase II slices typically have specific naming conventions
    phase2_prefixes = ["slice_uplift_", "slice_phase2_", "u2_"]
    return any(slice_name.startswith(prefix) for prefix in phase2_prefixes)


def load_budget_for_slice(
    slice_name: str,
    config_path: Optional[Path] = None,
) -> VerifierBudget:
    """
    Load budget configuration for a slice.
    
    Args:
        slice_name: Name of the slice
        config_path: Path to budget config file (default: DEFAULT_CONFIG_PATH)
        
    Returns:
        VerifierBudget with loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If slice not found in config
        ValueError: If budget values are invalid
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Budget config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if "slices" not in config:
        raise KeyError("Budget config missing 'slices' key")
    
    slices = config["slices"]
    if slice_name not in slices:
        raise KeyError(f"Slice '{slice_name}' not found in budget config")
    
    slice_config = slices[slice_name]
    
    try:
        budget = VerifierBudget(
            cycle_budget_s=float(slice_config["cycle_budget_s"]),
            taut_timeout_s=float(slice_config["taut_timeout_s"]),
            max_candidates_per_cycle=int(slice_config["max_candidates_per_cycle"]),
        )
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid budget config for slice '{slice_name}': {e}")
    
    # Validate budget values
    if budget.cycle_budget_s <= 0:
        raise ValueError("cycle_budget_s must be positive")
    if budget.taut_timeout_s <= 0:
        raise ValueError("taut_timeout_s must be positive")
    if budget.max_candidates_per_cycle <= 0:
        raise ValueError("max_candidates_per_cycle must be positive")
    
    return budget
