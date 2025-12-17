"""Analysis plan package.

Provides plan creation and management for analysis operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Plan:
    """Analysis plan structure."""
    name: str
    description: str = ""
    steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "metadata": self.metadata,
        }


@dataclass
class ConsolePlan:
    """Console-specific plan for runbook operations."""
    title: str
    commands: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)


def create_plan(
    name: str,
    description: str = "",
    steps: Optional[List[str]] = None,
) -> Plan:
    """Create an analysis plan."""
    return Plan(name=name, description=description, steps=steps or [])


def create_console_plan(title: str, commands: Optional[List[str]] = None) -> ConsolePlan:
    """Create a console plan."""
    return ConsolePlan(title=title, commands=commands or [])


def load_plan_from_file(path: str) -> Plan:
    """Load analysis plan from file (stub)."""
    return Plan(name="loaded_plan", description=f"Loaded from {path}")


def validate_plan(plan: Plan) -> bool:
    """Validate an analysis plan."""
    return bool(plan.name)


# Import submodule
from .global_console_plan import (
    GlobalConsolePlan,
    create_global_console_plan,
    load_global_console_plan,
)

__all__ = [
    "Plan",
    "ConsolePlan",
    "GlobalConsolePlan",
    "create_plan",
    "create_console_plan",
    "create_global_console_plan",
    "load_plan_from_file",
    "load_global_console_plan",
    "validate_plan",
]
