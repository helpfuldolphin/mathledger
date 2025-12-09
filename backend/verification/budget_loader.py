"""
Backward-compatible budget_loader shim.

This module provides the BudgetLoader class for loading verification budgets.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Default configuration path
DEFAULT_CONFIG_PATH = Path("config/verifier_budget.yaml")


@dataclass
class VerificationBudget:
    """Budget constraints for verification."""
    timeout_seconds: int = 30
    memory_mb: int = 2048
    disk_mb: int = 100
    max_proofs: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "max_proofs": self.max_proofs,
        }


class BudgetLoader:
    """
    Loads verification budgets from configuration.

    This is a backward-compatible shim. In Phase IIb, this will load
    real budgets from YAML or JSON configuration files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the budget loader.

        Args:
            config_path: Optional path to budget configuration file.
        """
        self._config_path = config_path
        self._budgets: Dict[str, VerificationBudget] = {}

    def load(self) -> Dict[str, VerificationBudget]:
        """
        Load budgets from configuration.

        Returns:
            Dictionary mapping slice names to budgets.
        """
        # Default budgets for Phase II
        return {
            "default": VerificationBudget(),
            "minimal": VerificationBudget(
                timeout_seconds=5,
                memory_mb=512,
                disk_mb=50,
                max_proofs=1,
            ),
            "generous": VerificationBudget(
                timeout_seconds=60,
                memory_mb=4096,
                disk_mb=500,
                max_proofs=50,
            ),
        }

    def get_budget(self, slice_name: str) -> VerificationBudget:
        """
        Get budget for a specific slice.

        Args:
            slice_name: Name of the curriculum slice.

        Returns:
            VerificationBudget for the slice.
        """
        if not self._budgets:
            self._budgets = self.load()
        return self._budgets.get(slice_name, self._budgets.get("default", VerificationBudget()))


# Alias for backward compatibility
VerifierBudget = VerificationBudget

# Singleton instance for convenience
_default_loader = None


def load_budget_for_slice(slice_name: str) -> VerificationBudget:
    """
    Convenience function to load a budget for a slice.

    Uses a singleton BudgetLoader instance.

    Args:
        slice_name: Name of the curriculum slice.

    Returns:
        VerificationBudget for the slice.
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = BudgetLoader()
    return _default_loader.get_budget(slice_name)


def load_default_budget() -> VerificationBudget:
    """
    Load the default verification budget.

    Returns:
        Default VerificationBudget.
    """
    return load_budget_for_slice("default")


def is_phase2_slice(slice_name: str) -> bool:
    """
    Check if a slice is a Phase 2 slice.

    Phase 2 slices have specific naming conventions.

    Args:
        slice_name: Name of the slice to check.

    Returns:
        True if it's a Phase 2 slice.
    """
    phase2_prefixes = ["phase2", "phase_2", "p2_", "uplift_"]
    return any(slice_name.lower().startswith(prefix) for prefix in phase2_prefixes)


__all__ = [
    "BudgetLoader",
    "VerificationBudget",
    "VerifierBudget",
    "load_budget_for_slice",
    "load_default_budget",
    "is_phase2_slice",
    "DEFAULT_CONFIG_PATH",
]
