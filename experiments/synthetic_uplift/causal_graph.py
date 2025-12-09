#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Scenario Causal Graph
----------------------

This module implements causal links between scenarios for deterministic
test harness ordering.

IMPORTANT: Causal links affect ONLY the order in which scenarios are
processed in the test harness. They do NOT affect:
    - Sampling behavior
    - Outcome probabilities
    - Any metric computation

This is purely structural for organization and reproducibility.

Must NOT:
    - Influence actual test outcomes
    - Produce claims about real uplift
    - Mix synthetic and real data

==============================================================================
"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# CAUSAL GRAPH STRUCTURE
# ==============================================================================

@dataclass
class CausalLink:
    """
    A directed causal link from source to target.
    
    Meaning: source should be processed BEFORE target in test harness.
    This is purely organizational, not outcome-affecting.
    """
    source: str
    target: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalLink":
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            description=data.get("description"),
        )


@dataclass
class ScenarioCausalGraph:
    """
    Directed acyclic graph of scenario causal relationships.
    
    Used for deterministic test harness ordering.
    """
    
    scenarios: Set[str] = field(default_factory=set)
    links: List[CausalLink] = field(default_factory=list)
    
    # Adjacency lists
    _forward: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    _backward: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def add_scenario(self, name: str):
        """Add a scenario node to the graph."""
        if not name.startswith("synthetic_"):
            raise ValueError(f"Scenario must start with 'synthetic_': {name}")
        self.scenarios.add(name)
    
    def add_link(self, source: str, target: str, description: Optional[str] = None):
        """
        Add a causal link: source -> target.
        
        Meaning: source should be processed before target.
        """
        if source not in self.scenarios:
            self.add_scenario(source)
        if target not in self.scenarios:
            self.add_scenario(target)
        
        link = CausalLink(source=source, target=target, description=description)
        self.links.append(link)
        self._forward[source].append(target)
        self._backward[target].append(source)
    
    def get_dependencies(self, scenario: str) -> List[str]:
        """Get scenarios that must be processed before this one."""
        return self._backward.get(scenario, [])
    
    def get_dependents(self, scenario: str) -> List[str]:
        """Get scenarios that depend on this one."""
        return self._forward.get(scenario, [])
    
    def has_cycle(self) -> bool:
        """Check if the graph has any cycles."""
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._forward.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.scenarios:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def topological_sort(self) -> List[str]:
        """
        Return scenarios in topological order.
        
        This is the deterministic processing order for the test harness.
        """
        if self.has_cycle():
            raise ValueError("Graph has cycles - cannot produce topological order")
        
        # Kahn's algorithm
        in_degree = {node: 0 for node in self.scenarios}
        for source in self._forward:
            for target in self._forward[source]:
                in_degree[target] = in_degree.get(target, 0) + 1
        
        # Start with nodes that have no incoming edges
        queue = deque(sorted([n for n, d in in_degree.items() if d == 0]))
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in sorted(self._forward.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining isolated nodes (sorted for determinism)
        remaining = sorted(self.scenarios - set(result))
        result.extend(remaining)
        
        return result
    
    def validate(self) -> List[str]:
        """Validate the graph structure."""
        errors = []
        
        if self.has_cycle():
            errors.append("Graph contains cycles")
        
        for link in self.links:
            if not link.source.startswith("synthetic_"):
                errors.append(f"Link source must start with 'synthetic_': {link.source}")
            if not link.target.startswith("synthetic_"):
                errors.append(f"Link target must start with 'synthetic_': {link.target}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": SAFETY_LABEL,
            "scenarios": sorted(self.scenarios),
            "links": [link.to_dict() for link in self.links],
            "causal_links": {
                src: sorted(self._forward[src])
                for src in sorted(self._forward.keys())
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioCausalGraph":
        """Deserialize from dictionary."""
        graph = cls()
        
        for scenario in data.get("scenarios", []):
            graph.add_scenario(scenario)
        
        for link_data in data.get("links", []):
            link = CausalLink.from_dict(link_data)
            graph.add_link(link.source, link.target, link.description)
        
        # Also support the compact "causal_links" format
        for source, targets in data.get("causal_links", {}).items():
            if source not in graph.scenarios:
                graph.add_scenario(source)
            for target in targets:
                if target not in graph.scenarios:
                    graph.add_scenario(target)
                # Avoid duplicates
                if target not in graph._forward[source]:
                    graph.add_link(source, target)
        
        return graph


# ==============================================================================
# GRAPH VISUALIZATION
# ==============================================================================

def visualize_scenario_graph(
    graph: ScenarioCausalGraph,
    format: str = "text",
) -> str:
    """
    Visualize the scenario causal graph.
    
    Args:
        graph: The causal graph to visualize
        format: Output format ('text', 'dot', 'mermaid')
    
    Returns:
        String representation of the graph
    """
    if format == "text":
        return _visualize_text(graph)
    elif format == "dot":
        return _visualize_dot(graph)
    elif format == "mermaid":
        return _visualize_mermaid(graph)
    else:
        raise ValueError(f"Unknown format: {format}")


def _visualize_text(graph: ScenarioCausalGraph) -> str:
    """Text-based visualization."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "SCENARIO CAUSAL GRAPH",
        "=" * 60,
        f"Scenarios: {len(graph.scenarios)}",
        f"Links:     {len(graph.links)}",
        "",
    ]
    
    # Show topological order
    try:
        order = graph.topological_sort()
        lines.append("Processing Order (Topological Sort):")
        lines.append("-" * 40)
        for i, scenario in enumerate(order, 1):
            deps = graph.get_dependencies(scenario)
            deps_str = f" <- {', '.join(deps)}" if deps else ""
            lines.append(f"  {i:2d}. {scenario}{deps_str}")
    except ValueError as e:
        lines.append(f"ERROR: {e}")
    
    lines.append("")
    
    # Show all links
    if graph.links:
        lines.append("Causal Links:")
        lines.append("-" * 40)
        for link in graph.links:
            desc = f" ({link.description})" if link.description else ""
            lines.append(f"  {link.source} -> {link.target}{desc}")
    
    lines.append("")
    return "\n".join(lines)


def _visualize_dot(graph: ScenarioCausalGraph) -> str:
    """GraphViz DOT format visualization."""
    lines = [
        f"// {SAFETY_LABEL}",
        "digraph ScenarioCausalGraph {",
        "    rankdir=TB;",
        "    node [shape=box, style=filled, fillcolor=lightblue];",
        "",
    ]
    
    # Add nodes
    for scenario in sorted(graph.scenarios):
        # Shorten name for readability
        short_name = scenario.replace("synthetic_", "")
        lines.append(f'    "{scenario}" [label="{short_name}"];')
    
    lines.append("")
    
    # Add edges
    for link in graph.links:
        label = f' [label="{link.description}"]' if link.description else ""
        lines.append(f'    "{link.source}" -> "{link.target}"{label};')
    
    lines.append("}")
    return "\n".join(lines)


def _visualize_mermaid(graph: ScenarioCausalGraph) -> str:
    """Mermaid diagram format visualization."""
    lines = [
        f"%%{SAFETY_LABEL}",
        "graph TD",
    ]
    
    # Add nodes with short names
    for scenario in sorted(graph.scenarios):
        short_name = scenario.replace("synthetic_", "")
        lines.append(f"    {scenario}[{short_name}]")
    
    # Add edges
    for link in graph.links:
        if link.description:
            lines.append(f"    {link.source} -->|{link.description}| {link.target}")
        else:
            lines.append(f"    {link.source} --> {link.target}")
    
    return "\n".join(lines)


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

def build_default_causal_graph() -> ScenarioCausalGraph:
    """
    Build the default causal graph for synthetic scenarios.
    
    This defines the default processing order based on logical
    dependencies (e.g., simpler scenarios before complex ones).
    """
    graph = ScenarioCausalGraph()
    
    # Base scenarios
    base_scenarios = [
        "synthetic_null_uplift",
        "synthetic_positive_uplift",
        "synthetic_negative_uplift",
    ]
    
    # Drift scenarios depend on base
    drift_scenarios = [
        "synthetic_drift_monotonic",
        "synthetic_drift_cyclical",
        "synthetic_drift_shock",
    ]
    
    # Correlation scenarios
    correlation_scenarios = [
        "synthetic_correlation_low",
        "synthetic_correlation_high",
    ]
    
    # Rare event scenarios
    rare_scenarios = [
        "synthetic_catastrophic",
        "synthetic_sudden_uplift",
        "synthetic_outlier_bursts",
        "synthetic_intermittent",
        "synthetic_recovery",
    ]
    
    # Mixed scenarios depend on simpler ones
    mixed_scenarios = [
        "synthetic_variance",
        "synthetic_mixed_chaos",
    ]
    
    # Add all scenarios
    for scenario in (base_scenarios + drift_scenarios + correlation_scenarios +
                     rare_scenarios + mixed_scenarios):
        graph.add_scenario(scenario)
    
    # Define causal relationships (ordering only, not outcomes)
    # Drift scenarios should come after base validation
    for drift in drift_scenarios:
        graph.add_link("synthetic_null_uplift", drift, "base before drift")
    
    # Correlation scenarios after base
    for corr in correlation_scenarios:
        graph.add_link("synthetic_null_uplift", corr, "base before correlation")
    
    # Mixed scenarios last
    for mixed in mixed_scenarios:
        graph.add_link("synthetic_drift_cyclical", mixed, "drift before mixed")
        graph.add_link("synthetic_correlation_high", mixed, "correlation before mixed")
    
    return graph


# ==============================================================================
# REGISTRY INTEGRATION
# ==============================================================================

def load_causal_graph_from_registry(registry_path: Optional[Path] = None) -> ScenarioCausalGraph:
    """
    Load causal graph from the scenario registry.
    
    If the registry doesn't have causal links, returns the default graph.
    """
    if registry_path is None:
        registry_path = Path(__file__).parent / "scenario_registry.json"
    
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    # Check for causal_links in registry
    if "causal_links" in registry:
        graph = ScenarioCausalGraph()
        for scenario in registry.get("scenarios", {}):
            graph.add_scenario(scenario)
        
        for source, targets in registry["causal_links"].items():
            for target in targets:
                graph.add_link(source, target)
        
        return graph
    
    # Fall back to default graph with registry scenarios
    graph = build_default_causal_graph()
    
    # Add any registry scenarios not in default graph
    for scenario in registry.get("scenarios", {}):
        if scenario not in graph.scenarios:
            graph.add_scenario(scenario)
    
    return graph


def save_causal_graph_to_file(
    graph: ScenarioCausalGraph,
    path: Path,
    format: str = "json",
):
    """Save causal graph to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, indent=2)
    elif format in ("text", "dot", "mermaid"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(visualize_scenario_graph(graph, format=format))
    else:
        raise ValueError(f"Unknown format: {format}")


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print(f"\n{SAFETY_LABEL}\n")
    
    # Build and display default graph
    graph = build_default_causal_graph()
    
    if len(sys.argv) > 1:
        fmt = sys.argv[1]
    else:
        fmt = "text"
    
    print(visualize_scenario_graph(graph, format=fmt))

