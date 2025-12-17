"""Global console plan module.

Provides global console plan utilities for runbook operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


REQUIRED_TILES = ["Health", "Metrics", "Alerts", "Security", "Conjectures"]


@dataclass
class TileNode:
    """Console tile node for DAG-based planning."""
    name: str
    description: str = ""
    depends_on: Tuple[str, ...] = field(default_factory=tuple)
    hard_gate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for backwards compatibility
ConsoleTileNode = TileNode


@dataclass
class GlobalConsolePlan:
    """Global console plan for orchestrating multiple runbooks."""
    name: str
    description: str = ""
    runbooks: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "runbooks": self.runbooks,
            "config": self.config,
        }


def create_global_console_plan(
    name: str,
    description: str = "",
    runbooks: Optional[List[str]] = None,
) -> GlobalConsolePlan:
    """Create a global console plan."""
    return GlobalConsolePlan(
        name=name,
        description=description,
        runbooks=runbooks or [],
    )


def load_global_console_plan(path: str) -> GlobalConsolePlan:
    """Load global console plan from file (stub)."""
    return GlobalConsolePlan(name="loaded_plan", description=f"Loaded from {path}")


def execute_global_console_plan(plan: GlobalConsolePlan) -> Dict[str, Any]:
    """Execute a global console plan (stub)."""
    return {
        "status": "ok",
        "executed_runbooks": plan.runbooks,
    }


def build_console_plan(
    tiles: Optional[List[str]] = None,
) -> Dict[str, TileNode]:
    """Build a console plan with specified tiles as a DAG.

    Returns a dict mapping tile name to TileNode with dependencies.
    """
    tiles = tiles or REQUIRED_TILES

    # Default dependency graph with hard_gate info
    tile_defs = {
        "Health": {"deps": (), "hard_gate": True},
        "Metrics": {"deps": ("Health",), "hard_gate": True},
        "Alerts": {"deps": ("Health", "Metrics"), "hard_gate": False},
        "Security": {"deps": ("Health",), "hard_gate": True},
        "Conjectures": {"deps": ("Metrics", "Security"), "hard_gate": False},
    }

    plan = {}
    for tile in tiles:
        tile_def = tile_defs.get(tile, {"deps": (), "hard_gate": False})
        deps = tile_def["deps"]
        # Filter deps to only include tiles in the plan
        valid_deps = tuple(d for d in deps if d in tiles)
        plan[tile] = TileNode(
            name=tile,
            description=f"{tile} tile",
            depends_on=valid_deps,
            hard_gate=tile_def["hard_gate"],
        )

    return plan


def topological_order(plan: Mapping[str, TileNode]) -> Tuple[str, ...]:
    """Compute topological order of tiles in the plan.

    Returns tiles in dependency order (dependencies before dependents).
    """
    visited = set()
    order = []

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        node = plan.get(name)
        if node:
            for dep in node.depends_on:
                visit(dep)
        order.append(name)

    for name in plan:
        visit(name)

    return tuple(order)


def dependents_map(plan: Mapping[str, TileNode]) -> Dict[str, List[str]]:
    """Build a map of tile -> list of tiles that depend on it."""
    deps: Dict[str, List[str]] = {name: [] for name in plan}
    for name, node in plan.items():
        for dependency in node.depends_on:
            if dependency in deps:
                deps[dependency].append(name)
    return deps


def minimal_deployment_sequence(
    plan: Optional[Mapping[str, TileNode]] = None,
) -> Tuple[str, ...]:
    """Get minimal deployment sequence for a plan."""
    if plan is None:
        plan = build_console_plan()
    return topological_order(plan)


def plan_summary(
    plan: Optional[Mapping[str, TileNode]] = None,
) -> Dict[str, Any]:
    """Get summary of a global console plan."""
    if plan is None:
        plan = build_console_plan()

    tile_names = set(plan.keys())
    missing = tuple(t for t in REQUIRED_TILES if t not in tile_names)

    # Count edges
    edge_count = sum(len(node.depends_on) for node in plan.values())

    return {
        "node_count": len(plan),
        "edge_count": edge_count,
        "tile_count": len(plan),
        "tiles": list(plan.keys()),
        "required_tiles": REQUIRED_TILES,
        "missing_required_tiles": missing,
        "topological_order": topological_order(plan),
    }


__all__ = [
    "REQUIRED_TILES",
    "TileNode",
    "ConsoleTileNode",
    "GlobalConsolePlan",
    "create_global_console_plan",
    "load_global_console_plan",
    "execute_global_console_plan",
    "build_console_plan",
    "topological_order",
    "dependents_map",
    "minimal_deployment_sequence",
    "plan_summary",
]
