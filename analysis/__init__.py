"""Analysis module.

Provides analysis plan, console runbook, dynamics, and conjecture utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisPlan:
    """Analysis plan structure."""
    name: str
    steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsoleRunbook:
    """Console runbook for analysis operations."""
    title: str
    commands: List[str] = field(default_factory=list)


def create_plan(name: str, steps: Optional[List[str]] = None) -> AnalysisPlan:
    """Create an analysis plan."""
    return AnalysisPlan(name=name, steps=steps or [])


def load_plan(path: str) -> AnalysisPlan:
    """Load analysis plan from file (stub)."""
    return AnalysisPlan(name="default")


# Re-export submodules for convenience
from .u2_dynamics import (
    DynamicsState,
    DynamicsTile,
    compute_dynamics_state,
    build_dynamics_tile,
    summarize_dynamics_for_global_health,
    attach_dynamics_tile,
)

from .conjecture_engine import (
    ConjectureStatus,
    Conjecture,
    ConjectureResult,
    generate_conjectures,
    evaluate_conjecture,
    summarize_conjectures_for_global_health,
    attach_conjecture_tile,
)

from .governance import (
    GovernanceAnalysis,
    analyze_governance,
    summarize_governance_for_report,
)

from .global_health import (
    build_pq_policy_tile,
    attach_pq_policy_tile,
)


__all__ = [
    # Core
    "AnalysisPlan",
    "ConsoleRunbook",
    "create_plan",
    "load_plan",
    # Dynamics
    "DynamicsState",
    "DynamicsTile",
    "compute_dynamics_state",
    "build_dynamics_tile",
    "summarize_dynamics_for_global_health",
    "attach_dynamics_tile",
    # Conjecture
    "ConjectureStatus",
    "Conjecture",
    "ConjectureResult",
    "generate_conjectures",
    "evaluate_conjecture",
    "summarize_conjectures_for_global_health",
    "attach_conjecture_tile",
    # Governance
    "GovernanceAnalysis",
    "analyze_governance",
    "summarize_governance_for_report",
    # Global Health
    "build_pq_policy_tile",
    "attach_pq_policy_tile",
]
