"""
Causal graph data structures and manipulation.

Represents causal relationships as a directed acyclic graph (DAG).
Nodes are variables, edges are causal relationships.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import hashlib
import json


class VariableType(Enum):
    """Types of causal variables."""
    POLICY = "policy"
    ABSTENTION = "abstention"
    THROUGHPUT = "throughput"
    VERIFICATION_TIME = "verification_time"
    DEPTH = "depth"
    SYSTEM = "system"  # Logical system (PL, FOL, etc.)


@dataclass
class CausalNode:
    """
    Node in causal graph representing a variable.

    Attributes:
        name: Variable name (e.g., "policy_hash", "abstain_pct")
        var_type: Type of variable
        description: Human-readable description
        observed_values: Historical observations of this variable
    """
    name: str
    var_type: VariableType
    description: str = ""
    observed_values: List[float] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, CausalNode):
            return False
        return self.name == other.name


@dataclass
class CausalEdge:
    """
    Directed edge representing causal relationship.

    Attributes:
        source: Cause variable
        target: Effect variable
        coefficient: Estimated causal coefficient (effect size)
        confidence: Confidence interval [lower, upper]
        mechanism: Description of causal mechanism
    """
    source: CausalNode
    target: CausalNode
    coefficient: Optional[float] = None
    confidence: Optional[Tuple[float, float]] = None
    mechanism: str = ""

    def __hash__(self):
        return hash((self.source.name, self.target.name))

    def __eq__(self, other):
        if not isinstance(other, CausalEdge):
            return False
        return (self.source == other.source and self.target == other.target)


class CausalGraph:
    """
    Directed acyclic graph (DAG) representing causal structure.

    Implements core RFL causal model:
    Policy → Abstention → Throughput
            ↘ Verification Time ↗
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Set[CausalEdge] = set()
        self._adjacency: Dict[str, Set[str]] = {}  # source -> {targets}
        self._reverse_adj: Dict[str, Set[str]] = {}  # target -> {sources}

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
        if node.name not in self._adjacency:
            self._adjacency[node.name] = set()
        if node.name not in self._reverse_adj:
            self._reverse_adj[node.name] = set()

    def add_edge(self, edge: CausalEdge) -> None:
        """
        Add a causal edge to the graph.

        Raises:
            ValueError: If edge would create a cycle
        """
        # Ensure nodes exist
        if edge.source.name not in self.nodes:
            self.add_node(edge.source)
        if edge.target.name not in self.nodes:
            self.add_node(edge.target)

        # Check for cycle
        if self._would_create_cycle(edge.source.name, edge.target.name):
            raise ValueError(
                f"Adding edge {edge.source.name} → {edge.target.name} "
                "would create a cycle"
            )

        self.edges.add(edge)
        self._adjacency[edge.source.name].add(edge.target.name)
        self._reverse_adj[edge.target.name].add(edge.source.name)

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge source→target would create a cycle."""
        # BFS from target to see if we can reach source
        visited = set()
        queue = [target]

        while queue:
            current = queue.pop(0)
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._adjacency.get(current, set()))

        return False

    def get_parents(self, node_name: str) -> List[CausalNode]:
        """Get all direct causal parents of a node."""
        parent_names = self._reverse_adj.get(node_name, set())
        return [self.nodes[name] for name in parent_names]

    def get_children(self, node_name: str) -> List[CausalNode]:
        """Get all direct causal children of a node."""
        child_names = self._adjacency.get(node_name, set())
        return [self.nodes[name] for name in child_names]

    def get_ancestors(self, node_name: str) -> Set[str]:
        """Get all causal ancestors of a node (transitive closure)."""
        ancestors = set()
        queue = list(self._reverse_adj.get(node_name, set()))

        while queue:
            current = queue.pop(0)
            if current in ancestors:
                continue
            ancestors.add(current)
            queue.extend(self._reverse_adj.get(current, set()))

        return ancestors

    def get_descendants(self, node_name: str) -> Set[str]:
        """Get all causal descendants of a node (transitive closure)."""
        descendants = set()
        queue = list(self._adjacency.get(node_name, set()))

        while queue:
            current = queue.pop(0)
            if current in descendants:
                continue
            descendants.add(current)
            queue.extend(self._adjacency.get(current, set()))

        return descendants

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (causes before effects).

        Returns:
            List of node names in topological order

        Raises:
            ValueError: If graph contains a cycle
        """
        in_degree = {name: len(self._reverse_adj.get(name, set()))
                     for name in self.nodes}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for child in self._adjacency.get(current, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two nodes, if it exists."""
        for edge in self.edges:
            if edge.source.name == source and edge.target.name == target:
                return edge
        return None

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": {
                name: {
                    "var_type": node.var_type.value,
                    "description": node.description,
                    "n_observations": len(node.observed_values)
                }
                for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source.name,
                    "target": edge.target.name,
                    "coefficient": edge.coefficient,
                    "confidence": edge.confidence,
                    "mechanism": edge.mechanism
                }
                for edge in self.edges
            ]
        }

    def hash(self) -> str:
        """Compute deterministic hash of graph structure."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


def build_rfl_graph() -> CausalGraph:
    """
    Build the canonical RFL causal graph.

    Structure:
        Policy → Score Distribution → Abstention Rate → Proof Throughput
                                    ↘ Verification Time ↗

    Returns:
        Initialized causal graph with RFL structure
    """
    graph = CausalGraph()

    # Define nodes
    policy = CausalNode(
        name="policy_hash",
        var_type=VariableType.POLICY,
        description="Policy identifier determining scoring behavior"
    )

    abstention = CausalNode(
        name="abstain_pct",
        var_type=VariableType.ABSTENTION,
        description="Percentage of derivations abstained from"
    )

    throughput = CausalNode(
        name="proofs_per_sec",
        var_type=VariableType.THROUGHPUT,
        description="Proof generation rate (proofs/second)"
    )

    verify_time = CausalNode(
        name="verify_ms_p50",
        var_type=VariableType.VERIFICATION_TIME,
        description="Median verification time (milliseconds)"
    )

    depth = CausalNode(
        name="depth_max",
        var_type=VariableType.DEPTH,
        description="Maximum formula depth reached"
    )

    # Add nodes
    graph.add_node(policy)
    graph.add_node(abstention)
    graph.add_node(throughput)
    graph.add_node(verify_time)
    graph.add_node(depth)

    # Define causal edges
    # Policy → Abstention
    graph.add_edge(CausalEdge(
        source=policy,
        target=abstention,
        mechanism="Policy scoring determines which derivations to attempt vs abstain"
    ))

    # Policy → Depth
    graph.add_edge(CausalEdge(
        source=policy,
        target=depth,
        mechanism="Policy prioritizes formulas by complexity"
    ))

    # Abstention → Verification Time
    graph.add_edge(CausalEdge(
        source=abstention,
        target=verify_time,
        mechanism="Higher abstention → attempt easier proofs → faster verification"
    ))

    # Depth → Verification Time
    graph.add_edge(CausalEdge(
        source=depth,
        target=verify_time,
        mechanism="Deeper formulas require longer verification"
    ))

    # Verification Time → Throughput
    graph.add_edge(CausalEdge(
        source=verify_time,
        target=throughput,
        mechanism="Faster verification → higher throughput"
    ))

    # Abstention → Throughput (direct effect)
    graph.add_edge(CausalEdge(
        source=abstention,
        target=throughput,
        mechanism="Abstaining from hard problems → focus resources on solvable proofs"
    ))

    return graph


def validate_graph(graph: CausalGraph) -> List[str]:
    """
    Validate causal graph structure.

    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []

    # Check for cycles
    try:
        graph.topological_sort()
    except ValueError:
        warnings.append("Graph contains cycles")

    # Check for isolated nodes
    for name, node in graph.nodes.items():
        parents = graph.get_parents(name)
        children = graph.get_children(name)
        if not parents and not children:
            warnings.append(f"Node {name} is isolated (no causal connections)")

    # Check for missing coefficients
    for edge in graph.edges:
        if edge.coefficient is None:
            warnings.append(
                f"Edge {edge.source.name} → {edge.target.name} "
                "has no estimated coefficient"
            )

    return warnings
