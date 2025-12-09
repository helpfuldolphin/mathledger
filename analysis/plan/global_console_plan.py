"""
Global Health Console planning graph.

Defines the static dependency graph for the major console tiles and exposes
helpers to reason about ordering guarantees and hard gating semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class TileNode:
    """Graph node describing an interactive console tile."""

    name: str
    depends_on: Tuple[str, ...]
    hard_gate: bool = True


def build_console_plan() -> Dict[str, TileNode]:
    """
    Construct the static plan definition.

    Returns:
        Dict keyed by tile name that enumerates dependency relationships.
    """

    nodes: List[TileNode] = [
        TileNode("Preflight", (), True),
        TileNode("Bundle", ("Preflight",), True),
        TileNode("Slice Identity", ("Bundle",), True),
        TileNode("Topology", ("Bundle",), True),
        TileNode("Telemetry", ("Topology", "Slice Identity"), True),
        TileNode("Metrics", ("Telemetry",), True),
        TileNode("Budget", ("Metrics",), True),
        TileNode("Security", ("Telemetry", "Metrics"), True),
        TileNode("Replay", ("Bundle", "Telemetry"), False),
        TileNode("TDA", ("Replay", "Topology"), False),
        TileNode("Conjectures", ("TDA", "Budget", "Security"), False),
    ]

    plan = {node.name: node for node in nodes}
    if len(plan) != len(nodes):
        raise ValueError("Duplicate tile names detected in console plan")
    return plan


def dependency_graph(plan: Mapping[str, TileNode] | None = None) -> Dict[str, Tuple[str, ...]]:
    """
    Return the dependency graph for the provided plan (or default plan).
    """

    source = plan or build_console_plan()
    return {name: node.depends_on for name, node in source.items()}


def dependents_map(plan: Mapping[str, TileNode]) -> Dict[str, Tuple[str, ...]]:
    """Compute downstream dependents for each tile."""

    dependents: Dict[str, List[str]] = {name: [] for name in plan}
    for node in plan.values():
        for dependency in node.depends_on:
            if dependency not in plan:
                raise KeyError(f"Unknown dependency {dependency!r} for tile {node.name}")
            dependents[dependency].append(node.name)
    return {name: tuple(sorted(children)) for name, children in dependents.items()}


def topological_order(plan: Mapping[str, TileNode] | None = None) -> List[str]:
    """
    Return a deterministic topological ordering for the tiles.

    Raises:
        ValueError: if the graph contains a cycle.
    """

    source = plan or build_console_plan()
    graph = dependency_graph(source)
    incoming = {name: len(deps) for name, deps in graph.items()}

    from heapq import heappop, heappush

    ready: List[str] = []
    for name, degree in incoming.items():
        if degree == 0:
            heappush(ready, name)

    order: List[str] = []
    dependents = dependents_map(source)

    while ready:
        current = heappop(ready)
        order.append(current)

        for child in dependents[current]:
            incoming[child] -= 1
            if incoming[child] == 0:
                heappush(ready, child)

    if len(order) != len(source):
        raise ValueError("Dependency graph contains a cycle")

    return order


def minimal_deployment_sequence(plan: Mapping[str, TileNode] | None = None) -> Tuple[str, ...]:
    """
    Expose the minimal ordered set of tiles for a fresh deployment.
    """

    return tuple(topological_order(plan))


__all__ = [
    "TileNode",
    "build_console_plan",
    "dependency_graph",
    "dependents_map",
    "topological_order",
    "minimal_deployment_sequence",
]
