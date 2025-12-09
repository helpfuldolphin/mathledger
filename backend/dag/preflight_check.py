"""
PHASE II — NOT RUN IN PHASE I

Pre-Flight DAG Health and Drift Eligibility Checks.

This module implements the checks specified in docs/DAG_PRE_FLIGHT_AUDIT.md:
- CHECK-001..008: Structural health validations
- DRIFT-001..005: Run eligibility comparisons

Author: CLAUDE G — DAG Pre-Flight Auditor Engineer
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading Pre-Flight Check.", file=sys.stderr)


# =============================================================================
# Enums and Data Structures
# =============================================================================

class CheckStatus(str, Enum):
    """Status for individual checks."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Severity(str, Enum):
    """Severity level for checks."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class CheckResult:
    """Result of a single CHECK-* validation."""
    id: str
    name: str
    status: CheckStatus
    severity: Severity
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "details": self.details,
        }


@dataclass
class DriftCheckResult:
    """Result of a single DRIFT-* validation."""
    id: str
    name: str
    status: CheckStatus
    metric_value: Any
    threshold: Any
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class DriftEligibilityResult:
    """Result of drift eligibility assessment."""
    eligible: bool
    reasons: List[str]
    drift_checks: List[DriftCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": self.reasons,
            "drift_checks": [c.to_dict() for c in self.drift_checks],
        }


@dataclass
class PreflightReport:
    """Complete pre-flight audit report."""
    preflight_version: str
    timestamp: str
    label: str
    inputs: Dict[str, Any]
    checks: List[CheckResult]
    drift_eligibility: Optional[DriftEligibilityResult]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preflight_version": self.preflight_version,
            "timestamp": self.timestamp,
            "label": self.label,
            "inputs": self.inputs,
            "checks": {c.id: c.to_dict() for c in self.checks},
            "drift_eligibility": self.drift_eligibility.to_dict() if self.drift_eligibility else None,
            "summary": self.summary,
        }


@dataclass
class PreflightConfig:
    """Configuration for pre-flight checks."""
    # Scope
    scope: str = "FULL"  # FULL, BOUNDED, EXPERIMENT

    # CHECK-004 tolerance
    dangling_tolerance: int = 0

    # CHECK-007 tolerance
    depth_tolerance: int = 2
    max_configured_depth: Optional[int] = None

    # DRIFT thresholds
    max_vertex_divergence: float = 0.5
    max_edge_divergence: float = 0.6
    max_depth_difference: int = 3
    cycle_tolerance: int = 10

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreflightConfig":
        """Load configuration from dictionary."""
        preflight = d.get("preflight", {})
        drift = d.get("drift", {})
        return cls(
            scope=preflight.get("scope", "FULL"),
            dangling_tolerance=preflight.get("dangling_tolerance", 0),
            depth_tolerance=preflight.get("depth_tolerance", 2),
            max_configured_depth=preflight.get("max_configured_depth"),
            max_vertex_divergence=drift.get("max_vertex_divergence", 0.5),
            max_edge_divergence=drift.get("max_edge_divergence", 0.6),
            max_depth_difference=drift.get("max_depth_difference", 3),
            cycle_tolerance=drift.get("cycle_tolerance", 10),
        )


# =============================================================================
# DAG Data Structures
# =============================================================================

@dataclass
class DagSnapshot:
    """Snapshot of a DAG for analysis."""
    vertices: Set[str]  # Set of hashes
    edges: Set[Tuple[str, str]]  # Set of (child, parent) tuples
    hash_to_formula: Dict[str, str]  # hash -> normalized formula
    vertex_timestamps: Dict[str, float]  # hash -> timestamp (optional)
    depths: Dict[str, int]  # hash -> computed depth
    axioms: Set[str]  # Vertices with no parents
    cycle_count: int  # Number of cycles in the log


def load_dag_from_jsonl(path: Path) -> Tuple[DagSnapshot, List[str]]:
    """
    Load a DAG snapshot from a JSONL log file.

    Returns (DagSnapshot, list of parse errors)
    """
    vertices: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    hash_to_formula: Dict[str, str] = {}
    vertex_timestamps: Dict[str, float] = {}
    axioms: Set[str] = set()
    parse_errors: List[str] = []
    cycle_count = 0

    if not path.exists():
        parse_errors.append(f"File not found: {path}")
        return DagSnapshot(
            vertices=vertices,
            edges=edges,
            hash_to_formula=hash_to_formula,
            vertex_timestamps=vertex_timestamps,
            depths={},
            axioms=axioms,
            cycle_count=0,
        ), parse_errors

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {line_num}: JSON parse error: {e}")
                continue

            # Track cycle records
            if "cycle" in record:
                cycle_count = max(cycle_count, record.get("cycle", 0) + 1)

            # Extract derivations
            derivations = record.get("derivations", [])
            if isinstance(derivations, list):
                for d in derivations:
                    h = d.get("hash") or d.get("conclusion")
                    if h:
                        vertices.add(h)
                        # Track formula mapping
                        formula = d.get("normalized") or d.get("formula") or d.get("text")
                        if formula:
                            hash_to_formula[h] = formula
                        # Track timestamp
                        ts = d.get("timestamp")
                        if ts:
                            vertex_timestamps[h] = ts
                        # Track parents/premises
                        parents = d.get("premises", []) or d.get("parents", [])
                        if not parents:
                            axioms.add(h)
                        else:
                            for p in parents:
                                vertices.add(p)
                                edges.add((h, p))

            # Also handle single derivation records
            h = record.get("hash") or record.get("conclusion")
            if h and "derivations" not in record:
                vertices.add(h)
                formula = record.get("normalized") or record.get("formula") or record.get("text")
                if formula:
                    hash_to_formula[h] = formula
                ts = record.get("timestamp")
                if ts:
                    vertex_timestamps[h] = ts
                parents = record.get("premises", []) or record.get("parents", [])
                if not parents:
                    axioms.add(h)
                else:
                    for p in parents:
                        vertices.add(p)
                        edges.add((h, p))

    # Compute depths after loading
    depths = _compute_all_depths(vertices, edges)

    # Identify axioms (vertices with no parents in edges)
    children_with_parents = {e[0] for e in edges}
    axioms = vertices - children_with_parents

    return DagSnapshot(
        vertices=vertices,
        edges=edges,
        hash_to_formula=hash_to_formula,
        vertex_timestamps=vertex_timestamps,
        depths=depths,
        axioms=axioms,
        cycle_count=cycle_count,
    ), parse_errors


def _compute_all_depths(vertices: Set[str], edges: Set[Tuple[str, str]]) -> Dict[str, int]:
    """Compute depth for all vertices using topological ordering."""
    # Build adjacency: child -> parents
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    for child, parent in edges:
        parents_of[child].add(parent)

    depths: Dict[str, int] = {}

    # Process in topological order using Kahn's algorithm
    # First, find in-degree (number of unique parents)
    in_degree: Dict[str, int] = defaultdict(int)
    for v in vertices:
        in_degree[v] = len(parents_of[v])

    # Start with axioms (no parents)
    queue = deque([v for v in vertices if in_degree[v] == 0])
    for v in queue:
        depths[v] = 1

    # For cycle-safe depth computation, use iterative approach
    visited: Set[str] = set()

    def get_depth(v: str) -> int:
        if v in depths:
            return depths[v]
        if v not in vertices:
            return 1  # Unknown vertex treated as axiom
        if v in visited:
            return 0  # Cycle detected

        visited.add(v)
        parents = parents_of.get(v, set())
        if not parents:
            depths[v] = 1
        else:
            max_parent_depth = max((get_depth(p) for p in parents), default=0)
            depths[v] = 1 + max_parent_depth
        visited.discard(v)
        return depths[v]

    for v in vertices:
        if v not in depths:
            get_depth(v)

    return depths


def merge_dags(dag1: DagSnapshot, dag2: DagSnapshot) -> DagSnapshot:
    """Merge two DAG snapshots."""
    vertices = dag1.vertices | dag2.vertices
    edges = dag1.edges | dag2.edges

    # Merge hash_to_formula (dag2 overwrites dag1 on conflict)
    hash_to_formula = dict(dag1.hash_to_formula)
    hash_to_formula.update(dag2.hash_to_formula)

    # Merge timestamps
    vertex_timestamps = dict(dag1.vertex_timestamps)
    vertex_timestamps.update(dag2.vertex_timestamps)

    # Recompute depths and axioms
    depths = _compute_all_depths(vertices, edges)
    children_with_parents = {e[0] for e in edges}
    axioms = vertices - children_with_parents

    return DagSnapshot(
        vertices=vertices,
        edges=edges,
        hash_to_formula=hash_to_formula,
        vertex_timestamps=vertex_timestamps,
        depths=depths,
        axioms=axioms,
        cycle_count=dag1.cycle_count + dag2.cycle_count,
    )


# =============================================================================
# CHECK-* Implementations
# =============================================================================

def check_001_acyclicity(dag: DagSnapshot) -> CheckResult:
    """
    CHECK-001: Acyclicity (INV-001)

    The merged DAG must contain no cycles.
    Uses Kahn's algorithm (topological sort).
    """
    # Build adjacency for Kahn's algorithm
    # For topological sort, we traverse parent->child (reverse of our edges)
    children_of: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = defaultdict(int)

    for child, parent in dag.edges:
        children_of[parent].add(child)
        in_degree[child] += 1

    # Initialize with all vertices
    for v in dag.vertices:
        if v not in in_degree:
            in_degree[v] = 0

    # Start with vertices that have no incoming edges (axioms)
    queue = deque([v for v in dag.vertices if in_degree[v] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for child in children_of.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    has_cycle = visited_count != len(dag.vertices)
    cycle_nodes = [v for v in dag.vertices if in_degree[v] > 0] if has_cycle else []

    status = CheckStatus.FAIL if has_cycle else CheckStatus.PASS

    return CheckResult(
        id="CHECK-001",
        name="Acyclicity",
        status=status,
        severity=Severity.CRITICAL,
        details={
            "vertices_checked": len(dag.vertices),
            "cycles_found": 1 if has_cycle else 0,
            "cycle_nodes": cycle_nodes[:10],  # Limit for report size
        },
    )


def check_002_no_self_loops(dag: DagSnapshot) -> CheckResult:
    """
    CHECK-002: No Self-Loops (INV-002)

    No vertex may reference itself as a parent.
    """
    self_loops = [(child, parent) for child, parent in dag.edges if child == parent]

    status = CheckStatus.FAIL if self_loops else CheckStatus.PASS

    return CheckResult(
        id="CHECK-002",
        name="No Self-Loops",
        status=status,
        severity=Severity.CRITICAL,
        details={
            "edges_checked": len(dag.edges),
            "self_loops_found": len(self_loops),
            "self_loop_vertices": list(set(e[0] for e in self_loops))[:10],
        },
    )


def check_003_hash_integrity(dag: DagSnapshot) -> CheckResult:
    """
    CHECK-003: Hash Integrity (INV-004)

    Each hash must map to exactly one normalized formula.
    """
    # Build reverse mapping: formula -> set of hashes
    formula_to_hashes: Dict[str, Set[str]] = defaultdict(set)
    for h, formula in dag.hash_to_formula.items():
        formula_to_hashes[formula].add(h)

    # Check for same hash mapping to different formulas
    hash_to_formulas: Dict[str, Set[str]] = defaultdict(set)
    for h, formula in dag.hash_to_formula.items():
        hash_to_formulas[h].add(formula)

    collisions = {h: list(formulas) for h, formulas in hash_to_formulas.items() if len(formulas) > 1}

    status = CheckStatus.FAIL if collisions else CheckStatus.PASS

    return CheckResult(
        id="CHECK-003",
        name="Hash Integrity",
        status=status,
        severity=Severity.CRITICAL,
        details={
            "hashes_checked": len(dag.hash_to_formula),
            "collisions_found": len(collisions),
            "collision_hashes": list(collisions.keys())[:10],
        },
    )


def check_004_parent_resolution(
    dag: DagSnapshot,
    axiom_manifest: Optional[Set[str]] = None,
    dangling_tolerance: int = 0,
) -> CheckResult:
    """
    CHECK-004: Parent Resolution

    All parent references must resolve to known vertices OR be in the allowed axiom set.
    """
    axiom_manifest = axiom_manifest or set()

    dangling_refs: List[Tuple[str, str]] = []
    resolved_to_axioms = 0

    for child, parent in dag.edges:
        if parent in dag.vertices:
            continue  # Resolved to known vertex
        if parent in axiom_manifest:
            resolved_to_axioms += 1
            continue  # Resolved to manifest axiom
        dangling_refs.append((child, parent))

    dangling_count = len(dangling_refs)

    if dangling_count == 0:
        status = CheckStatus.PASS
    elif dangling_count <= dangling_tolerance:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return CheckResult(
        id="CHECK-004",
        name="Parent Resolution",
        status=status,
        severity=Severity.ERROR,
        details={
            "parents_checked": len(dag.edges),
            "dangling_found": dangling_count,
            "resolved_to_axioms": resolved_to_axioms,
            "dangling_refs": dangling_refs[:10],
            "tolerance": dangling_tolerance,
        },
    )


def check_005_axiom_set_validity(axiom_manifest_path: Optional[Path]) -> CheckResult:
    """
    CHECK-005: Axiom Set Validity

    The axiom manifest must be present and parseable.
    """
    if axiom_manifest_path is None:
        return CheckResult(
            id="CHECK-005",
            name="Axiom Set Valid",
            status=CheckStatus.WARN,
            severity=Severity.ERROR,
            details={
                "axiom_manifest_provided": False,
                "message": "No axiom manifest path provided",
            },
        )

    if not axiom_manifest_path.exists():
        return CheckResult(
            id="CHECK-005",
            name="Axiom Set Valid",
            status=CheckStatus.FAIL,
            severity=Severity.ERROR,
            details={
                "axiom_manifest_path": str(axiom_manifest_path),
                "exists": False,
            },
        )

    try:
        content = axiom_manifest_path.read_text(encoding='utf-8')
        # Try JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                # YAML not available, try line-by-line hashes
                data = {"axioms": [line.strip() for line in content.splitlines() if line.strip()]}
            except Exception as e:
                return CheckResult(
                    id="CHECK-005",
                    name="Axiom Set Valid",
                    status=CheckStatus.FAIL,
                    severity=Severity.ERROR,
                    details={
                        "axiom_manifest_path": str(axiom_manifest_path),
                        "parse_error": str(e),
                    },
                )

        # Extract axiom hashes
        axiom_hashes = data.get("axioms", []) if isinstance(data, dict) else data
        if not isinstance(axiom_hashes, list):
            axiom_hashes = []

        # Validate hash format (64 hex chars for SHA256)
        valid_hashes = []
        invalid_hashes = []
        sha256_pattern = re.compile(r'^[a-fA-F0-9]{64}$')

        for h in axiom_hashes:
            if isinstance(h, str) and sha256_pattern.match(h):
                valid_hashes.append(h)
            else:
                invalid_hashes.append(h)

        if not valid_hashes:
            return CheckResult(
                id="CHECK-005",
                name="Axiom Set Valid",
                status=CheckStatus.FAIL,
                severity=Severity.ERROR,
                details={
                    "axiom_manifest_path": str(axiom_manifest_path),
                    "axiom_count": 0,
                    "message": "No valid SHA256 hashes found",
                },
            )

        status = CheckStatus.PASS if not invalid_hashes else CheckStatus.WARN

        return CheckResult(
            id="CHECK-005",
            name="Axiom Set Valid",
            status=status,
            severity=Severity.ERROR,
            details={
                "axiom_manifest_path": str(axiom_manifest_path),
                "axiom_count": len(valid_hashes),
                "invalid_entries": len(invalid_hashes),
                "axiom_hashes": valid_hashes[:5],  # Sample
            },
        )

    except Exception as e:
        return CheckResult(
            id="CHECK-005",
            name="Axiom Set Valid",
            status=CheckStatus.FAIL,
            severity=Severity.ERROR,
            details={
                "axiom_manifest_path": str(axiom_manifest_path),
                "error": str(e),
            },
        )


def check_006_log_integrity(
    baseline_path: Path,
    rfl_path: Optional[Path] = None,
) -> CheckResult:
    """
    CHECK-006: Log File Integrity

    Experiment log files must be complete and parseable.
    """
    results = {}
    total_errors = 0

    for name, path in [("baseline", baseline_path), ("rfl", rfl_path)]:
        if path is None:
            continue

        if not path.exists():
            results[name] = {"exists": False, "error": "File not found"}
            total_errors += 1
            continue

        line_count = 0
        parse_errors = 0
        cycle_records = 0
        truncated = False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                last_line = ""
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    last_line = line
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                        if "cycle" in record:
                            cycle_records += 1
                    except json.JSONDecodeError:
                        parse_errors += 1

                # Check for truncation (incomplete final line)
                if last_line and not last_line.endswith('\n'):
                    # Check if it's valid JSON
                    try:
                        json.loads(last_line.strip())
                    except json.JSONDecodeError:
                        truncated = True

            results[name] = {
                "exists": True,
                "lines": line_count,
                "parse_errors": parse_errors,
                "cycle_records": cycle_records,
                "truncated": truncated,
            }

            if parse_errors > 0 or truncated or cycle_records == 0:
                total_errors += 1

        except Exception as e:
            results[name] = {"exists": True, "error": str(e)}
            total_errors += 1

    if total_errors > 0:
        status = CheckStatus.FAIL
    else:
        status = CheckStatus.PASS

    return CheckResult(
        id="CHECK-006",
        name="Log Integrity",
        status=status,
        severity=Severity.ERROR,
        details=results,
    )


def check_007_depth_bounds(
    dag: DagSnapshot,
    max_configured_depth: Optional[int],
    depth_tolerance: int = 2,
) -> CheckResult:
    """
    CHECK-007: Depth Bound Compliance (conditional)

    No vertex should exceed configured maximum depth.
    """
    if max_configured_depth is None:
        return CheckResult(
            id="CHECK-007",
            name="Depth Bounds",
            status=CheckStatus.PASS,
            severity=Severity.WARNING,
            details={
                "skipped": True,
                "reason": "No max_configured_depth specified",
            },
        )

    max_depth = max(dag.depths.values()) if dag.depths else 0
    exceeded_vertices = [
        v for v, d in dag.depths.items()
        if d > max_configured_depth
    ]
    severely_exceeded = [
        v for v, d in dag.depths.items()
        if d > max_configured_depth + depth_tolerance
    ]

    if severely_exceeded:
        status = CheckStatus.FAIL
    elif exceeded_vertices:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.PASS

    return CheckResult(
        id="CHECK-007",
        name="Depth Bounds",
        status=status,
        severity=Severity.WARNING,
        details={
            "max_configured_depth": max_configured_depth,
            "actual_max_depth": max_depth,
            "exceeded_count": len(exceeded_vertices),
            "severely_exceeded_count": len(severely_exceeded),
            "tolerance": depth_tolerance,
        },
    )


def check_008_temporal_consistency(dag: DagSnapshot) -> CheckResult:
    """
    CHECK-008: Temporal Consistency (conditional)

    Child statements should not precede their parents in log order.
    """
    if not dag.vertex_timestamps:
        return CheckResult(
            id="CHECK-008",
            name="Temporal Consistency",
            status=CheckStatus.PASS,
            severity=Severity.WARNING,
            details={
                "skipped": True,
                "reason": "No timestamp data available",
            },
        )

    violations: List[Dict[str, Any]] = []

    for child, parent in dag.edges:
        child_ts = dag.vertex_timestamps.get(child)
        parent_ts = dag.vertex_timestamps.get(parent)

        if child_ts is not None and parent_ts is not None:
            if child_ts < parent_ts:
                violations.append({
                    "child": child[:16] + "...",
                    "parent": parent[:16] + "...",
                    "child_ts": child_ts,
                    "parent_ts": parent_ts,
                })

    status = CheckStatus.WARN if violations else CheckStatus.PASS

    return CheckResult(
        id="CHECK-008",
        name="Temporal Consistency",
        status=status,
        severity=Severity.WARNING,
        details={
            "edges_with_timestamps": sum(
                1 for c, p in dag.edges
                if c in dag.vertex_timestamps and p in dag.vertex_timestamps
            ),
            "violations_found": len(violations),
            "violations": violations[:10],
        },
    )


# =============================================================================
# DRIFT-* Implementations
# =============================================================================

def drift_001_axiom_alignment(
    baseline: DagSnapshot,
    rfl: DagSnapshot,
    axiom_manifest: Optional[Set[str]] = None,
) -> DriftCheckResult:
    """
    DRIFT-001: Axiom Set Alignment

    Both runs must use compatible axiom sets.
    """
    axiom_manifest = axiom_manifest or set()

    baseline_axioms = baseline.axioms
    rfl_axioms = rfl.axioms

    # Check if axioms are within manifest
    baseline_unknown = baseline_axioms - axiom_manifest if axiom_manifest else set()
    rfl_unknown = rfl_axioms - axiom_manifest if axiom_manifest else set()

    axiom_difference = baseline_axioms.symmetric_difference(rfl_axioms)

    if baseline_unknown or rfl_unknown:
        status = CheckStatus.FAIL
    elif axiom_difference:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.PASS

    return DriftCheckResult(
        id="DRIFT-001",
        name="Axiom Alignment",
        status=status,
        metric_value={
            "baseline_axiom_count": len(baseline_axioms),
            "rfl_axiom_count": len(rfl_axioms),
            "symmetric_difference": len(axiom_difference),
        },
        threshold="Identical manifest",
        details={
            "axiom_alignment": len(axiom_difference) == 0,
            "baseline_unknown_axioms": len(baseline_unknown),
            "rfl_unknown_axioms": len(rfl_unknown),
        },
    )


def drift_002_vertex_divergence(
    baseline: DagSnapshot,
    rfl: DagSnapshot,
    max_divergence: float = 0.5,
) -> DriftCheckResult:
    """
    DRIFT-002: Vertex Set Divergence

    Vertex sets must not diverge beyond threshold.
    """
    sym_diff = baseline.vertices.symmetric_difference(rfl.vertices)
    max_size = max(len(baseline.vertices), len(rfl.vertices), 1)
    divergence = len(sym_diff) / max_size

    if divergence <= max_divergence:
        status = CheckStatus.PASS
    elif divergence <= 2 * max_divergence:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return DriftCheckResult(
        id="DRIFT-002",
        name="Vertex Divergence",
        status=status,
        metric_value=round(divergence, 4),
        threshold=max_divergence,
        details={
            "baseline_vertices": len(baseline.vertices),
            "rfl_vertices": len(rfl.vertices),
            "symmetric_difference": len(sym_diff),
            "only_in_baseline": len(baseline.vertices - rfl.vertices),
            "only_in_rfl": len(rfl.vertices - baseline.vertices),
        },
    )


def drift_003_edge_divergence(
    baseline: DagSnapshot,
    rfl: DagSnapshot,
    max_divergence: float = 0.6,
) -> DriftCheckResult:
    """
    DRIFT-003: Edge Set Divergence

    Edge sets must not diverge beyond threshold.
    """
    sym_diff = baseline.edges.symmetric_difference(rfl.edges)
    max_size = max(len(baseline.edges), len(rfl.edges), 1)
    divergence = len(sym_diff) / max_size

    if divergence <= max_divergence:
        status = CheckStatus.PASS
    elif divergence <= 2 * max_divergence:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return DriftCheckResult(
        id="DRIFT-003",
        name="Edge Divergence",
        status=status,
        metric_value=round(divergence, 4),
        threshold=max_divergence,
        details={
            "baseline_edges": len(baseline.edges),
            "rfl_edges": len(rfl.edges),
            "symmetric_difference": len(sym_diff),
            "only_in_baseline": len(baseline.edges - rfl.edges),
            "only_in_rfl": len(rfl.edges - baseline.edges),
        },
    )


def drift_004_depth_distribution(
    baseline: DagSnapshot,
    rfl: DagSnapshot,
    max_difference: int = 3,
) -> DriftCheckResult:
    """
    DRIFT-004: Depth Distribution Alignment

    Max depth distributions must be comparable.
    """
    baseline_max = max(baseline.depths.values()) if baseline.depths else 0
    rfl_max = max(rfl.depths.values()) if rfl.depths else 0
    difference = abs(baseline_max - rfl_max)

    if difference <= max_difference:
        status = CheckStatus.PASS
    elif difference <= 2 * max_difference:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return DriftCheckResult(
        id="DRIFT-004",
        name="Depth Distribution",
        status=status,
        metric_value=difference,
        threshold=max_difference,
        details={
            "baseline_max_depth": baseline_max,
            "rfl_max_depth": rfl_max,
            "depth_difference": difference,
        },
    )


def drift_005_cycle_count(
    baseline: DagSnapshot,
    rfl: DagSnapshot,
    tolerance: int = 10,
) -> DriftCheckResult:
    """
    DRIFT-005: Cycle Count Alignment

    Both runs must have comparable cycle counts.
    """
    difference = abs(baseline.cycle_count - rfl.cycle_count)

    if difference == 0:
        status = CheckStatus.PASS
    elif difference <= tolerance:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return DriftCheckResult(
        id="DRIFT-005",
        name="Cycle Count",
        status=status,
        metric_value=difference,
        threshold=tolerance,
        details={
            "baseline_cycles": baseline.cycle_count,
            "rfl_cycles": rfl.cycle_count,
            "cycle_difference": difference,
        },
    )


# =============================================================================
# Main Pre-Flight Auditor
# =============================================================================

class PreflightAuditor:
    """
    Pre-Flight DAG Auditor.

    Executes all CHECK-* and DRIFT-* validations per DAG_PRE_FLIGHT_AUDIT.md.
    """

    VERSION = "1.0.0"
    LABEL = "PHASE II — NOT RUN IN PHASE I"

    def __init__(self, config: Optional[PreflightConfig] = None):
        self.config = config or PreflightConfig()
        self._checks: List[CheckResult] = []
        self._drift_checks: List[DriftCheckResult] = []

    def load_axiom_manifest(self, path: Optional[Path]) -> Set[str]:
        """Load axiom hashes from manifest file."""
        if path is None or not path.exists():
            return set()

        try:
            content = path.read_text(encoding='utf-8')
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except (ImportError, Exception):
                    # Assume line-by-line hashes
                    return set(line.strip() for line in content.splitlines() if line.strip())

            axioms = data.get("axioms", []) if isinstance(data, dict) else data
            return set(axioms) if isinstance(axioms, list) else set()
        except Exception:
            return set()

    def run_health_checks(
        self,
        dag: DagSnapshot,
        axiom_manifest: Optional[Set[str]] = None,
        axiom_manifest_path: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
        rfl_path: Optional[Path] = None,
    ) -> List[CheckResult]:
        """Run all CHECK-* validations."""
        checks: List[CheckResult] = []

        # CHECK-001: Acyclicity
        checks.append(check_001_acyclicity(dag))

        # CHECK-002: No Self-Loops
        checks.append(check_002_no_self_loops(dag))

        # CHECK-003: Hash Integrity
        checks.append(check_003_hash_integrity(dag))

        # CHECK-004: Parent Resolution
        checks.append(check_004_parent_resolution(
            dag,
            axiom_manifest=axiom_manifest,
            dangling_tolerance=self.config.dangling_tolerance,
        ))

        # CHECK-005: Axiom Set Validity
        checks.append(check_005_axiom_set_validity(axiom_manifest_path))

        # CHECK-006: Log Integrity
        if baseline_path or rfl_path:
            checks.append(check_006_log_integrity(
                baseline_path=baseline_path or Path(""),
                rfl_path=rfl_path,
            ))

        # CHECK-007: Depth Bounds (conditional)
        checks.append(check_007_depth_bounds(
            dag,
            max_configured_depth=self.config.max_configured_depth,
            depth_tolerance=self.config.depth_tolerance,
        ))

        # CHECK-008: Temporal Consistency (conditional)
        checks.append(check_008_temporal_consistency(dag))

        self._checks = checks
        return checks

    def run_drift_checks(
        self,
        baseline: DagSnapshot,
        rfl: DagSnapshot,
        axiom_manifest: Optional[Set[str]] = None,
    ) -> DriftEligibilityResult:
        """Run all DRIFT-* validations."""
        drift_checks: List[DriftCheckResult] = []

        # DRIFT-001: Axiom Alignment
        drift_checks.append(drift_001_axiom_alignment(
            baseline, rfl, axiom_manifest
        ))

        # DRIFT-002: Vertex Divergence
        drift_checks.append(drift_002_vertex_divergence(
            baseline, rfl,
            max_divergence=self.config.max_vertex_divergence,
        ))

        # DRIFT-003: Edge Divergence
        drift_checks.append(drift_003_edge_divergence(
            baseline, rfl,
            max_divergence=self.config.max_edge_divergence,
        ))

        # DRIFT-004: Depth Distribution
        drift_checks.append(drift_004_depth_distribution(
            baseline, rfl,
            max_difference=self.config.max_depth_difference,
        ))

        # DRIFT-005: Cycle Count
        drift_checks.append(drift_005_cycle_count(
            baseline, rfl,
            tolerance=self.config.cycle_tolerance,
        ))

        self._drift_checks = drift_checks

        # Determine eligibility
        fail_checks = [c for c in drift_checks if c.status == CheckStatus.FAIL]
        warn_checks = [c for c in drift_checks if c.status == CheckStatus.WARN]

        eligible = len(fail_checks) == 0
        reasons: List[str] = []

        for c in fail_checks:
            reasons.append(f"FAIL: {c.id} - {c.name}")
        for c in warn_checks:
            reasons.append(f"WARN: {c.id} - {c.name}")

        if not reasons:
            reasons.append("All drift checks passed")

        return DriftEligibilityResult(
            eligible=eligible,
            reasons=reasons,
            drift_checks=drift_checks,
        )

    def run_full_preflight(
        self,
        baseline_path: Path,
        rfl_path: Optional[Path] = None,
        axiom_manifest_path: Optional[Path] = None,
        slice_config_path: Optional[Path] = None,
    ) -> PreflightReport:
        """
        Run full pre-flight audit.

        Args:
            baseline_path: Path to baseline JSONL log
            rfl_path: Optional path to RFL JSONL log
            axiom_manifest_path: Optional path to axiom manifest
            slice_config_path: Optional path to slice configuration

        Returns:
            PreflightReport with all check results
        """
        # Load DAGs
        baseline_dag, baseline_errors = load_dag_from_jsonl(baseline_path)

        rfl_dag: Optional[DagSnapshot] = None
        rfl_errors: List[str] = []
        if rfl_path:
            rfl_dag, rfl_errors = load_dag_from_jsonl(rfl_path)

        # Load axiom manifest
        axiom_manifest = self.load_axiom_manifest(axiom_manifest_path)

        # Load slice config for depth bounds
        if slice_config_path and slice_config_path.exists():
            try:
                content = slice_config_path.read_text(encoding='utf-8')
                try:
                    slice_data = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        slice_data = yaml.safe_load(content)
                    except (ImportError, Exception):
                        slice_data = {}
                self.config.max_configured_depth = slice_data.get("max_chain_length") or slice_data.get("max_depth")
            except Exception:
                pass

        # Merge DAGs for health checks
        if rfl_dag:
            merged_dag = merge_dags(baseline_dag, rfl_dag)
        else:
            merged_dag = baseline_dag

        # Run health checks on merged DAG
        health_checks = self.run_health_checks(
            dag=merged_dag,
            axiom_manifest=axiom_manifest,
            axiom_manifest_path=axiom_manifest_path,
            baseline_path=baseline_path,
            rfl_path=rfl_path,
        )

        # Run drift checks if both DAGs available
        drift_result: Optional[DriftEligibilityResult] = None
        if rfl_dag:
            drift_result = self.run_drift_checks(
                baseline=baseline_dag,
                rfl=rfl_dag,
                axiom_manifest=axiom_manifest,
            )

        # Build summary
        critical_failures = sum(1 for c in health_checks if c.status == CheckStatus.FAIL and c.severity == Severity.CRITICAL)
        error_failures = sum(1 for c in health_checks if c.status == CheckStatus.FAIL and c.severity == Severity.ERROR)
        warnings = sum(1 for c in health_checks if c.status == CheckStatus.WARN)

        audit_eligible = critical_failures == 0 and (drift_result is None or drift_result.eligible)

        if critical_failures > 0:
            overall_status = "FAIL"
        elif error_failures > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        summary = {
            "overall_status": overall_status,
            "critical_failures": critical_failures,
            "errors": error_failures,
            "warnings": warnings,
            "audit_eligible": audit_eligible,
            "parse_errors": {
                "baseline": baseline_errors,
                "rfl": rfl_errors,
            },
        }

        inputs = {
            "baseline_log": str(baseline_path),
            "rfl_log": str(rfl_path) if rfl_path else None,
            "axiom_manifest": str(axiom_manifest_path) if axiom_manifest_path else None,
            "slice_config": str(slice_config_path) if slice_config_path else None,
            "scope": self.config.scope,
        }

        return PreflightReport(
            preflight_version=self.VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            label=self.LABEL,
            inputs=inputs,
            checks=health_checks,
            drift_eligibility=drift_result,
            summary=summary,
        )


def get_exit_code(report: PreflightReport) -> int:
    """
    Determine exit code from report.

    Returns:
        0: All checks PASS
        1: WARN conditions (audit may proceed with flag)
        2: FAIL conditions (audit blocked)
    """
    if report.summary["critical_failures"] > 0:
        return 2
    if report.summary["errors"] > 0:
        return 2
    if report.summary["warnings"] > 0:
        return 1
    return 0


# =============================================================================
# DAG Posture Snapshot (TASK 1)
# =============================================================================

# Schema version for posture snapshots - increment on breaking changes
POSTURE_SCHEMA_VERSION = "1.0.0"


def build_dag_posture_snapshot(report: PreflightReport) -> Dict[str, Any]:
    """
    Build a compact posture description for a DAG or DAG pair.

    This extracts key structural metrics from a PreflightReport into a
    path/timestamp-independent snapshot suitable for cross-run comparisons.

    Args:
        report: A PreflightReport from a preflight audit

    Returns:
        A dictionary containing:
        - schema_version: Version of the posture schema
        - has_cycles: Whether cycles were detected (from CHECK-001)
        - max_depth: Maximum chain depth observed
        - vertex_count: Total number of vertices
        - edge_count: Total number of edges
        - drift_eligible: Whether the run is eligible for drift comparison
        - drift_ineligibility_reason: Reason if not eligible (optional)
    """
    # Extract cycle status from CHECK-001
    has_cycles = False
    check_001 = next((c for c in report.checks if c.id == "CHECK-001"), None)
    if check_001:
        has_cycles = check_001.status == CheckStatus.FAIL
        # Also check details for explicit cycle count
        if check_001.details.get("cycles_found", 0) > 0:
            has_cycles = True

    # Extract depth info from CHECK-007 or summary
    max_depth = 0
    check_007 = next((c for c in report.checks if c.id == "CHECK-007"), None)
    if check_007 and not check_007.details.get("skipped"):
        max_depth = check_007.details.get("actual_max_depth", 0)

    # Extract vertex/edge counts from CHECK-001 or CHECK-002
    vertex_count = 0
    edge_count = 0

    if check_001:
        vertex_count = check_001.details.get("vertices_checked", 0)

    check_002 = next((c for c in report.checks if c.id == "CHECK-002"), None)
    if check_002:
        edge_count = check_002.details.get("edges_checked", 0)

    # Extract drift eligibility
    drift_eligible = True
    drift_ineligibility_reason: Optional[str] = None

    if report.drift_eligibility:
        drift_eligible = report.drift_eligibility.eligible
        if not drift_eligible:
            # Get first FAIL reason
            fail_reasons = [r for r in report.drift_eligibility.reasons if r.startswith("FAIL:")]
            if fail_reasons:
                drift_ineligibility_reason = fail_reasons[0]
            elif report.drift_eligibility.reasons:
                drift_ineligibility_reason = report.drift_eligibility.reasons[0]

    # Build posture snapshot with deterministic key ordering
    posture: Dict[str, Any] = {
        "schema_version": POSTURE_SCHEMA_VERSION,
        "has_cycles": has_cycles,
        "max_depth": max_depth,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "drift_eligible": drift_eligible,
    }

    if drift_ineligibility_reason:
        posture["drift_ineligibility_reason"] = drift_ineligibility_reason

    return posture


def build_dag_posture_from_snapshot(dag: DagSnapshot) -> Dict[str, Any]:
    """
    Build a posture snapshot directly from a DagSnapshot.

    This is useful when you have raw DAG data without a full PreflightReport.

    Args:
        dag: A DagSnapshot containing DAG structure

    Returns:
        A posture dictionary (without drift eligibility info)
    """
    # Detect cycles using Kahn's algorithm
    children_of: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = defaultdict(int)

    for child, parent in dag.edges:
        children_of[parent].add(child)
        in_degree[child] += 1

    for v in dag.vertices:
        if v not in in_degree:
            in_degree[v] = 0

    queue = deque([v for v in dag.vertices if in_degree[v] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for child in children_of.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    has_cycles = visited_count != len(dag.vertices)

    max_depth = max(dag.depths.values()) if dag.depths else 0

    return {
        "schema_version": POSTURE_SCHEMA_VERSION,
        "has_cycles": has_cycles,
        "max_depth": max_depth,
        "vertex_count": len(dag.vertices),
        "edge_count": len(dag.edges),
        "drift_eligible": True,  # Unknown without drift checks
    }


# =============================================================================
# Cross-Run DAG Drift Radar (TASK 2)
# =============================================================================

class DriftDirection(str, Enum):
    """Direction of drift eligibility change."""
    STABLE_ELIGIBLE = "STABLE_ELIGIBLE"
    STABLE_INELIGIBLE = "STABLE_INELIGIBLE"
    ELIGIBLE_TO_INELIGIBLE = "ELIGIBLE_TO_INELIGIBLE"
    INELIGIBLE_TO_ELIGIBLE = "INELIGIBLE_TO_ELIGIBLE"


def compare_dag_postures(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two DAG posture snapshots to detect drift across runs.

    This enables tracking of DAG evolution over time by comparing
    posture snapshots from different experiments or time periods.

    Args:
        old: Previous posture snapshot
        new: Current posture snapshot

    Returns:
        A dictionary containing:
        - depth_delta: Change in max depth (new - old)
        - vertex_count_delta: Change in vertex count
        - edge_count_delta: Change in edge count
        - drift_eligibility_change: Enum describing eligibility transition
        - cycle_status_changed: Whether cycle status changed
        - schema_compatible: Whether schemas are compatible
    """
    # Check schema compatibility
    old_version = old.get("schema_version", "0.0.0")
    new_version = new.get("schema_version", "0.0.0")

    # Major version must match for compatibility
    old_major = old_version.split(".")[0] if old_version else "0"
    new_major = new_version.split(".")[0] if new_version else "0"
    schema_compatible = old_major == new_major

    # Compute deltas
    old_depth = old.get("max_depth", 0)
    new_depth = new.get("max_depth", 0)
    depth_delta = new_depth - old_depth

    old_vertices = old.get("vertex_count", 0)
    new_vertices = new.get("vertex_count", 0)
    vertex_count_delta = new_vertices - old_vertices

    old_edges = old.get("edge_count", 0)
    new_edges = new.get("edge_count", 0)
    edge_count_delta = new_edges - old_edges

    # Determine drift eligibility change
    old_eligible = old.get("drift_eligible", True)
    new_eligible = new.get("drift_eligible", True)

    if old_eligible and new_eligible:
        drift_eligibility_change = DriftDirection.STABLE_ELIGIBLE
    elif not old_eligible and not new_eligible:
        drift_eligibility_change = DriftDirection.STABLE_INELIGIBLE
    elif old_eligible and not new_eligible:
        drift_eligibility_change = DriftDirection.ELIGIBLE_TO_INELIGIBLE
    else:
        drift_eligibility_change = DriftDirection.INELIGIBLE_TO_ELIGIBLE

    # Check cycle status change
    old_cycles = old.get("has_cycles", False)
    new_cycles = new.get("has_cycles", False)
    cycle_status_changed = old_cycles != new_cycles

    # Build comparison result with deterministic key ordering
    comparison: Dict[str, Any] = {
        "depth_delta": depth_delta,
        "vertex_count_delta": vertex_count_delta,
        "edge_count_delta": edge_count_delta,
        "drift_eligibility_change": drift_eligibility_change.value,
        "cycle_status_changed": cycle_status_changed,
        "schema_compatible": schema_compatible,
    }

    # Add context for significant changes
    if cycle_status_changed:
        comparison["cycle_transition"] = f"{'CYCLIC' if old_cycles else 'ACYCLIC'} -> {'CYCLIC' if new_cycles else 'ACYCLIC'}"

    if drift_eligibility_change in (DriftDirection.ELIGIBLE_TO_INELIGIBLE, DriftDirection.INELIGIBLE_TO_ELIGIBLE):
        old_reason = old.get("drift_ineligibility_reason")
        new_reason = new.get("drift_ineligibility_reason")
        if old_reason or new_reason:
            comparison["eligibility_reasons"] = {
                "old": old_reason,
                "new": new_reason,
            }

    return comparison


def compare_posture_files(
    old_path: Path,
    new_path: Path,
) -> Dict[str, Any]:
    """
    Load and compare two posture snapshot files.

    Args:
        old_path: Path to previous posture JSON file
        new_path: Path to current posture JSON file

    Returns:
        Comparison result dictionary

    Raises:
        FileNotFoundError: If either file doesn't exist
        json.JSONDecodeError: If either file is invalid JSON
    """
    with open(old_path, 'r', encoding='utf-8') as f:
        old = json.load(f)

    with open(new_path, 'r', encoding='utf-8') as f:
        new = json.load(f)

    return compare_dag_postures(old, new)


# =============================================================================
# PHASE III: DAG Drift Ledger & Eligibility Oracle
# =============================================================================

# Schema version for drift ledger
LEDGER_SCHEMA_VERSION = "1.0.0"


class GatingLevel(str, Enum):
    """Gating level for eligibility oracle decisions."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class DriftLedgerEntry:
    """A single entry in the drift ledger tracking posture evolution."""
    index: int
    posture: Dict[str, Any]
    comparison: Optional[Dict[str, Any]]  # None for first entry
    cumulative_drift_score: float
    timestamp: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "posture": self.posture,
            "comparison": self.comparison,
            "cumulative_drift_score": self.cumulative_drift_score,
            "timestamp": self.timestamp,
            "label": self.label,
        }


@dataclass
class DriftLedger:
    """
    Drift Ledger tracking DAG posture evolution over time.

    Tracks eligibility transitions, depth/vertex/edge trends,
    and computes cumulative drift scores.
    """
    schema_version: str
    entries: List[DriftLedgerEntry]
    trends: Dict[str, Any]
    eligibility_transitions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [e.to_dict() for e in self.entries],
            "trends": self.trends,
            "eligibility_transitions": self.eligibility_transitions,
        }


def _compute_drift_score(comparison: Dict[str, Any]) -> float:
    """
    Compute a drift score from a posture comparison.

    The drift score is a weighted sum of:
    - Absolute depth change (weight: 0.3)
    - Relative vertex change (weight: 0.4)
    - Relative edge change (weight: 0.3)
    - Cycle status change penalty (+1.0)
    - Eligibility change penalty (+0.5)

    Returns:
        A non-negative float representing drift magnitude.
    """
    score = 0.0

    # Depth change contribution (absolute)
    depth_delta = abs(comparison.get("depth_delta", 0))
    score += 0.3 * min(depth_delta / 10.0, 1.0)  # Normalize to [0, 1]

    # Vertex change contribution (relative)
    vertex_delta = abs(comparison.get("vertex_count_delta", 0))
    score += 0.4 * min(vertex_delta / 100.0, 1.0)  # Normalize

    # Edge change contribution (relative)
    edge_delta = abs(comparison.get("edge_count_delta", 0))
    score += 0.3 * min(edge_delta / 100.0, 1.0)  # Normalize

    # Cycle status change penalty
    if comparison.get("cycle_status_changed", False):
        score += 1.0

    # Eligibility transition penalty
    eligibility_change = comparison.get("drift_eligibility_change", "")
    if eligibility_change in (
        DriftDirection.ELIGIBLE_TO_INELIGIBLE.value,
        DriftDirection.INELIGIBLE_TO_ELIGIBLE.value,
    ):
        score += 0.5

    return round(score, 4)


def build_dag_drift_ledger(
    posture_snapshots: List[Dict[str, Any]],
    timestamps: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> DriftLedger:
    """
    Build a drift ledger from a sequence of posture snapshots.

    Tracks:
    - Eligibility transitions between consecutive snapshots
    - Depth, vertex, and edge count trends
    - Cumulative drift score across all transitions

    Args:
        posture_snapshots: List of posture snapshots in chronological order
        timestamps: Optional list of timestamps for each snapshot
        labels: Optional list of labels for each snapshot

    Returns:
        A DriftLedger containing entries, trends, and transition history
    """
    if not posture_snapshots:
        return DriftLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            entries=[],
            trends={
                "depth": [],
                "vertex_count": [],
                "edge_count": [],
                "drift_scores": [],
            },
            eligibility_transitions=[],
        )

    entries: List[DriftLedgerEntry] = []
    eligibility_transitions: List[Dict[str, Any]] = []

    # Trend tracking lists
    depth_trend: List[int] = []
    vertex_trend: List[int] = []
    edge_trend: List[int] = []
    drift_scores: List[float] = []

    cumulative_drift = 0.0
    prev_posture: Optional[Dict[str, Any]] = None

    for i, posture in enumerate(posture_snapshots):
        timestamp = timestamps[i] if timestamps and i < len(timestamps) else None
        label = labels[i] if labels and i < len(labels) else None

        # Extract trend data
        depth_trend.append(posture.get("max_depth", 0))
        vertex_trend.append(posture.get("vertex_count", 0))
        edge_trend.append(posture.get("edge_count", 0))

        # Compute comparison with previous posture
        comparison: Optional[Dict[str, Any]] = None
        if prev_posture is not None:
            comparison = compare_dag_postures(prev_posture, posture)
            drift_score = _compute_drift_score(comparison)
            cumulative_drift += drift_score
            drift_scores.append(drift_score)

            # Track eligibility transitions
            eligibility_change = comparison.get("drift_eligibility_change")
            if eligibility_change in (
                DriftDirection.ELIGIBLE_TO_INELIGIBLE.value,
                DriftDirection.INELIGIBLE_TO_ELIGIBLE.value,
            ):
                eligibility_transitions.append({
                    "from_index": i - 1,
                    "to_index": i,
                    "direction": eligibility_change,
                    "from_label": labels[i - 1] if labels and i - 1 < len(labels) else None,
                    "to_label": label,
                    "reason": comparison.get("eligibility_reasons", {}),
                })
        else:
            drift_scores.append(0.0)

        entry = DriftLedgerEntry(
            index=i,
            posture=posture,
            comparison=comparison,
            cumulative_drift_score=round(cumulative_drift, 4),
            timestamp=timestamp,
            label=label,
        )
        entries.append(entry)
        prev_posture = posture

    # Compute trend summaries
    trends = {
        "depth": depth_trend,
        "vertex_count": vertex_trend,
        "edge_count": edge_trend,
        "drift_scores": drift_scores,
        # Summary statistics
        "depth_min": min(depth_trend) if depth_trend else 0,
        "depth_max": max(depth_trend) if depth_trend else 0,
        "depth_delta_total": depth_trend[-1] - depth_trend[0] if len(depth_trend) > 1 else 0,
        "vertex_min": min(vertex_trend) if vertex_trend else 0,
        "vertex_max": max(vertex_trend) if vertex_trend else 0,
        "vertex_delta_total": vertex_trend[-1] - vertex_trend[0] if len(vertex_trend) > 1 else 0,
        "edge_min": min(edge_trend) if edge_trend else 0,
        "edge_max": max(edge_trend) if edge_trend else 0,
        "edge_delta_total": edge_trend[-1] - edge_trend[0] if len(edge_trend) > 1 else 0,
        "cumulative_drift": round(cumulative_drift, 4),
        "average_drift": round(sum(drift_scores) / len(drift_scores), 4) if drift_scores else 0.0,
    }

    return DriftLedger(
        schema_version=LEDGER_SCHEMA_VERSION,
        entries=entries,
        trends=trends,
        eligibility_transitions=eligibility_transitions,
    )


# =============================================================================
# TASK 2: Eligibility Oracle
# =============================================================================

@dataclass
class EligibilityOracleResult:
    """Result from the eligibility oracle evaluation."""
    eligibility_status: str  # "ELIGIBLE" or "INELIGIBLE"
    gating_level: GatingLevel
    reasons: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligibility_status": self.eligibility_status,
            "gating_level": self.gating_level.value,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def evaluate_dag_eligibility(
    old_posture: Dict[str, Any],
    new_posture: Dict[str, Any],
    *,
    max_depth_regression: int = 5,
    max_vertex_loss_pct: float = 0.2,
    max_edge_loss_pct: float = 0.3,
    max_drift_score: float = 2.0,
    block_on_cycle_introduction: bool = True,
) -> EligibilityOracleResult:
    """
    Evaluate DAG eligibility based on posture comparison.

    This oracle determines whether a DAG transition is safe for production
    and assigns a gating level:
    - OK: Transition is safe, no issues detected
    - WARN: Transition has minor issues, proceed with caution
    - BLOCK: Transition has critical issues, should not proceed

    Args:
        old_posture: Previous posture snapshot
        new_posture: Current posture snapshot
        max_depth_regression: Max allowed depth decrease before WARN
        max_vertex_loss_pct: Max allowed vertex loss percentage before WARN
        max_edge_loss_pct: Max allowed edge loss percentage before WARN
        max_drift_score: Max drift score before BLOCK
        block_on_cycle_introduction: Whether to BLOCK on cycle introduction

    Returns:
        EligibilityOracleResult with status, gating level, and reasons
    """
    comparison = compare_dag_postures(old_posture, new_posture)
    drift_score = _compute_drift_score(comparison)

    reasons: List[str] = []
    gating_level = GatingLevel.OK
    eligibility_status = "ELIGIBLE"

    # Check for cycle introduction (BLOCK by default)
    if comparison.get("cycle_status_changed", False):
        cycle_transition = comparison.get("cycle_transition", "")
        if "ACYCLIC -> CYCLIC" in cycle_transition:
            reasons.append("CRITICAL: Cycle introduced in DAG")
            if block_on_cycle_introduction:
                gating_level = GatingLevel.BLOCK
                eligibility_status = "INELIGIBLE"
            else:
                gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))
        elif "CYCLIC -> ACYCLIC" in cycle_transition:
            reasons.append("GOOD: Cycle resolved in DAG")

    # Check for depth regression
    depth_delta = comparison.get("depth_delta", 0)
    if depth_delta < -max_depth_regression:
        reasons.append(f"WARNING: Significant depth regression ({depth_delta})")
        gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))

    # Check for vertex loss
    old_vertices = old_posture.get("vertex_count", 0)
    vertex_delta = comparison.get("vertex_count_delta", 0)
    if old_vertices > 0 and vertex_delta < 0:
        loss_pct = abs(vertex_delta) / old_vertices
        if loss_pct > max_vertex_loss_pct:
            reasons.append(f"WARNING: Significant vertex loss ({vertex_delta}, {loss_pct:.1%})")
            gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))

    # Check for edge loss
    old_edges = old_posture.get("edge_count", 0)
    edge_delta = comparison.get("edge_count_delta", 0)
    if old_edges > 0 and edge_delta < 0:
        loss_pct = abs(edge_delta) / old_edges
        if loss_pct > max_edge_loss_pct:
            reasons.append(f"WARNING: Significant edge loss ({edge_delta}, {loss_pct:.1%})")
            gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))

    # Check drift score threshold
    if drift_score > max_drift_score:
        reasons.append(f"CRITICAL: Drift score exceeds threshold ({drift_score:.4f} > {max_drift_score})")
        gating_level = GatingLevel.BLOCK
        eligibility_status = "INELIGIBLE"

    # Check eligibility transition
    eligibility_change = comparison.get("drift_eligibility_change", "")
    if eligibility_change == DriftDirection.ELIGIBLE_TO_INELIGIBLE.value:
        reasons.append("WARNING: Drift eligibility lost")
        gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))
    elif eligibility_change == DriftDirection.INELIGIBLE_TO_ELIGIBLE.value:
        reasons.append("GOOD: Drift eligibility restored")

    # Check schema compatibility
    if not comparison.get("schema_compatible", True):
        reasons.append("WARNING: Schema version incompatibility")
        gating_level = max(gating_level, GatingLevel.WARN, key=lambda x: [GatingLevel.OK, GatingLevel.WARN, GatingLevel.BLOCK].index(x))

    # If no issues found
    if not reasons:
        reasons.append("OK: DAG transition is healthy")

    # Compile metrics
    metrics = {
        "depth_delta": depth_delta,
        "vertex_count_delta": vertex_delta,
        "edge_count_delta": edge_delta,
        "drift_score": drift_score,
        "cycle_status_changed": comparison.get("cycle_status_changed", False),
        "eligibility_change": eligibility_change,
        "schema_compatible": comparison.get("schema_compatible", True),
    }

    return EligibilityOracleResult(
        eligibility_status=eligibility_status,
        gating_level=gating_level,
        reasons=reasons,
        metrics=metrics,
    )


# =============================================================================
# TASK 3: Global Health Hook
# =============================================================================

@dataclass
class GlobalHealthSummary:
    """Summary of DAG posture for global health monitoring."""
    dag_ok: bool
    sustained_regressions: List[Dict[str, Any]]
    drift_hotspots: List[Dict[str, Any]]
    health_score: float  # 0.0 to 1.0
    summary_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_ok": self.dag_ok,
            "sustained_regressions": self.sustained_regressions,
            "drift_hotspots": self.drift_hotspots,
            "health_score": self.health_score,
            "summary_metrics": self.summary_metrics,
        }


def summarize_dag_posture_for_global_health(
    ledger: DriftLedger,
    *,
    regression_window: int = 3,
    drift_hotspot_threshold: float = 0.5,
    max_sustained_regressions: int = 2,
) -> GlobalHealthSummary:
    """
    Summarize DAG posture from a drift ledger for global health monitoring.

    This function analyzes the drift ledger to detect:
    - Sustained regressions: Consecutive decreases in key metrics
    - Drift hotspots: Points with high drift scores
    - Overall health assessment

    Args:
        ledger: The drift ledger to analyze
        regression_window: Number of consecutive entries to check for regression
        drift_hotspot_threshold: Drift score threshold for hotspot detection
        max_sustained_regressions: Max allowed sustained regressions before unhealthy

    Returns:
        GlobalHealthSummary with health status and detected issues
    """
    if not ledger.entries:
        return GlobalHealthSummary(
            dag_ok=True,
            sustained_regressions=[],
            drift_hotspots=[],
            health_score=1.0,
            summary_metrics={
                "total_entries": 0,
                "total_transitions": 0,
                "eligibility_transitions": 0,
            },
        )

    sustained_regressions: List[Dict[str, Any]] = []
    drift_hotspots: List[Dict[str, Any]] = []
    health_penalties = 0.0

    # Detect sustained regressions (consecutive decreases)
    if len(ledger.entries) >= regression_window:
        depth_trend = ledger.trends.get("depth", [])
        vertex_trend = ledger.trends.get("vertex_count", [])
        edge_trend = ledger.trends.get("edge_count", [])

        # Check depth regression
        for i in range(len(depth_trend) - regression_window + 1):
            window = depth_trend[i:i + regression_window]
            if all(window[j] > window[j + 1] for j in range(len(window) - 1)):
                sustained_regressions.append({
                    "metric": "depth",
                    "start_index": i,
                    "end_index": i + regression_window - 1,
                    "values": window,
                    "total_regression": window[0] - window[-1],
                })
                health_penalties += 0.1

        # Check vertex regression
        for i in range(len(vertex_trend) - regression_window + 1):
            window = vertex_trend[i:i + regression_window]
            if all(window[j] > window[j + 1] for j in range(len(window) - 1)):
                sustained_regressions.append({
                    "metric": "vertex_count",
                    "start_index": i,
                    "end_index": i + regression_window - 1,
                    "values": window,
                    "total_regression": window[0] - window[-1],
                })
                health_penalties += 0.15

        # Check edge regression
        for i in range(len(edge_trend) - regression_window + 1):
            window = edge_trend[i:i + regression_window]
            if all(window[j] > window[j + 1] for j in range(len(window) - 1)):
                sustained_regressions.append({
                    "metric": "edge_count",
                    "start_index": i,
                    "end_index": i + regression_window - 1,
                    "values": window,
                    "total_regression": window[0] - window[-1],
                })
                health_penalties += 0.15

    # Detect drift hotspots
    drift_scores = ledger.trends.get("drift_scores", [])
    for i, score in enumerate(drift_scores):
        if score >= drift_hotspot_threshold:
            entry = ledger.entries[i]
            drift_hotspots.append({
                "index": i,
                "drift_score": score,
                "label": entry.label,
                "timestamp": entry.timestamp,
                "details": entry.comparison if entry.comparison else {},
            })
            health_penalties += 0.1

    # Penalize eligibility transitions
    health_penalties += len(ledger.eligibility_transitions) * 0.1

    # Check for cycle introduction in any entry
    for entry in ledger.entries:
        if entry.comparison and entry.comparison.get("cycle_status_changed"):
            cycle_transition = entry.comparison.get("cycle_transition", "")
            if "ACYCLIC -> CYCLIC" in cycle_transition:
                health_penalties += 0.3

    # Compute health score (capped at 0.0)
    health_score = max(0.0, 1.0 - health_penalties)

    # Determine overall health
    dag_ok = (
        len(sustained_regressions) <= max_sustained_regressions
        and health_score >= 0.5
        and not any(
            entry.posture.get("has_cycles", False)
            for entry in ledger.entries[-3:] if ledger.entries  # Check last 3 entries
        )
    )

    # Summary metrics
    summary_metrics = {
        "total_entries": len(ledger.entries),
        "total_transitions": len(ledger.entries) - 1 if ledger.entries else 0,
        "eligibility_transitions": len(ledger.eligibility_transitions),
        "cumulative_drift": ledger.trends.get("cumulative_drift", 0.0),
        "average_drift": ledger.trends.get("average_drift", 0.0),
        "depth_delta_total": ledger.trends.get("depth_delta_total", 0),
        "vertex_delta_total": ledger.trends.get("vertex_delta_total", 0),
        "edge_delta_total": ledger.trends.get("edge_delta_total", 0),
        "sustained_regression_count": len(sustained_regressions),
        "drift_hotspot_count": len(drift_hotspots),
        "health_penalties": round(health_penalties, 4),
    }

    return GlobalHealthSummary(
        dag_ok=dag_ok,
        sustained_regressions=sustained_regressions,
        drift_hotspots=drift_hotspots,
        health_score=round(health_score, 4),
        summary_metrics=summary_metrics,
    )


# =============================================================================
# PHASE IV: Structural Eligibility Gate & Director DAG Panel
# =============================================================================

class ReleaseStatus(str, Enum):
    """Status for release eligibility decisions."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class MaasStatus(str, Enum):
    """Status for MAAS/Global Health integration."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


class StatusLight(str, Enum):
    """Visual status indicator for Director panel."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


# =============================================================================
# TASK 1: DAG-Based Release Eligibility View
# =============================================================================

def evaluate_dag_for_release(
    drift_ledger: Dict[str, Any],
    oracle_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate DAG posture for release eligibility.

    This function determines whether a release should proceed based on
    DAG structural health and oracle assessment.

    Args:
        drift_ledger: Serialized DriftLedger (from DriftLedger.to_dict())
        oracle_result: Serialized EligibilityOracleResult (from EligibilityOracleResult.to_dict())

    Returns:
        Dictionary containing:
        - release_ok: bool - Whether release should proceed
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: List of neutral strings describing issues
    """
    blocking_reasons: List[str] = []

    # Extract oracle gating level
    oracle_gating = oracle_result.get("gating_level", "OK")
    oracle_reasons = oracle_result.get("reasons", [])

    # Check oracle BLOCK conditions
    if oracle_gating == GatingLevel.BLOCK.value:
        # Filter to critical reasons only
        for reason in oracle_reasons:
            if reason.startswith("CRITICAL:"):
                # Convert to neutral language
                neutral_reason = reason.replace("CRITICAL: ", "")
                blocking_reasons.append(neutral_reason)

    # Check drift ledger for additional concerns
    trends = drift_ledger.get("trends", {})
    entries = drift_ledger.get("entries", [])
    eligibility_transitions = drift_ledger.get("eligibility_transitions", [])

    # Check for recent cycle introduction
    if entries:
        recent_entries = entries[-3:] if len(entries) >= 3 else entries
        for entry in recent_entries:
            posture = entry.get("posture", {})
            if posture.get("has_cycles", False):
                if "Cycle introduced in DAG" not in blocking_reasons:
                    blocking_reasons.append("DAG contains cycles in recent posture")
                break

    # Check cumulative drift threshold
    cumulative_drift = trends.get("cumulative_drift", 0.0)
    if cumulative_drift > 5.0:  # High cumulative drift
        blocking_reasons.append(f"Cumulative drift score exceeds threshold ({cumulative_drift:.2f})")

    # Check for frequent eligibility transitions (instability)
    if len(eligibility_transitions) >= 3:
        blocking_reasons.append(f"Frequent eligibility transitions detected ({len(eligibility_transitions)})")

    # Determine final status
    if blocking_reasons:
        status = ReleaseStatus.BLOCK
        release_ok = False
    elif oracle_gating == GatingLevel.WARN.value:
        status = ReleaseStatus.WARN
        release_ok = True  # WARN allows release with caution
        # Add warnings as informational (not blocking)
        for reason in oracle_reasons:
            if reason.startswith("WARNING:"):
                neutral_reason = reason.replace("WARNING: ", "")
                blocking_reasons.append(f"[WARN] {neutral_reason}")
    else:
        status = ReleaseStatus.OK
        release_ok = True

    return {
        "release_ok": release_ok,
        "status": status.value,
        "blocking_reasons": blocking_reasons,
    }


# =============================================================================
# TASK 2: MAAS / Global Health DAG Adapter
# =============================================================================

def summarize_dag_for_maas(
    oracle_result: Dict[str, Any],
    global_health_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize DAG status for MAAS / Global Health integration.

    This adapter provides a simplified view of DAG health suitable
    for integration with monitoring and alerting systems.

    Args:
        oracle_result: Serialized EligibilityOracleResult
        global_health_summary: Serialized GlobalHealthSummary

    Returns:
        Dictionary containing:
        - dag_structurally_ok: bool - Overall structural health
        - has_sustained_regressions: bool - Whether regressions detected
        - status: "OK" | "ATTENTION" | "BLOCK"
    """
    # Extract global health data
    dag_ok = global_health_summary.get("dag_ok", True)
    sustained_regressions = global_health_summary.get("sustained_regressions", [])
    health_score = global_health_summary.get("health_score", 1.0)
    drift_hotspots = global_health_summary.get("drift_hotspots", [])

    # Extract oracle data
    oracle_gating = oracle_result.get("gating_level", "OK")
    eligibility_status = oracle_result.get("eligibility_status", "ELIGIBLE")

    # Determine if structurally OK
    dag_structurally_ok = (
        dag_ok
        and eligibility_status == "ELIGIBLE"
        and health_score >= 0.5
    )

    # Check for sustained regressions
    has_sustained_regressions = len(sustained_regressions) > 0

    # Determine MAAS status
    if oracle_gating == GatingLevel.BLOCK.value:
        status = MaasStatus.BLOCK
    elif (
        oracle_gating == GatingLevel.WARN.value
        or has_sustained_regressions
        or len(drift_hotspots) > 0
        or health_score < 0.7
    ):
        status = MaasStatus.ATTENTION
    else:
        status = MaasStatus.OK

    return {
        "dag_structurally_ok": dag_structurally_ok,
        "has_sustained_regressions": has_sustained_regressions,
        "status": status.value,
    }


# =============================================================================
# TASK 3: Director DAG Panel
# =============================================================================

def _generate_dag_headline(
    health_score: float,
    status_light: StatusLight,
    drift_hotspot_count: int,
    has_cycles: bool,
    has_regressions: bool,
) -> str:
    """Generate a neutral headline describing DAG structural posture."""
    if status_light == StatusLight.RED:
        if has_cycles:
            return "DAG structure contains cycles requiring resolution"
        elif drift_hotspot_count > 0:
            return f"DAG posture has {drift_hotspot_count} drift hotspot(s) requiring attention"
        else:
            return "DAG structural health requires immediate attention"
    elif status_light == StatusLight.YELLOW:
        if has_regressions:
            return "DAG shows sustained metric regressions"
        elif health_score < 0.7:
            return f"DAG health score is below optimal ({health_score:.0%})"
        else:
            return "DAG posture has minor issues to monitor"
    else:
        if health_score >= 0.95:
            return "DAG structural posture is optimal"
        elif health_score >= 0.8:
            return "DAG structural posture is healthy"
        else:
            return "DAG structural posture is acceptable"


def build_dag_director_panel(
    drift_summary: Dict[str, Any],
    release_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a Director DAG panel for executive visibility.

    This provides a high-level view of DAG health suitable for
    dashboards and director-level reporting.

    Args:
        drift_summary: Serialized GlobalHealthSummary
        release_eval: Result from evaluate_dag_for_release()

    Returns:
        Dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - health_score: float (0.0 to 1.0)
        - drift_hotspots: List of hotspot summaries
        - headline: Neutral sentence describing DAG structural posture
    """
    # Extract health data
    health_score = drift_summary.get("health_score", 1.0)
    drift_hotspots_raw = drift_summary.get("drift_hotspots", [])
    sustained_regressions = drift_summary.get("sustained_regressions", [])
    dag_ok = drift_summary.get("dag_ok", True)

    # Extract release evaluation
    release_status = release_eval.get("status", "OK")
    release_ok = release_eval.get("release_ok", True)

    # Check for cycles in summary metrics
    summary_metrics = drift_summary.get("summary_metrics", {})

    # Determine status light
    if release_status == ReleaseStatus.BLOCK.value or not dag_ok:
        status_light = StatusLight.RED
    elif release_status == ReleaseStatus.WARN.value or health_score < 0.7:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Simplify drift hotspots for panel display
    drift_hotspots = []
    for hotspot in drift_hotspots_raw:
        drift_hotspots.append({
            "index": hotspot.get("index"),
            "drift_score": hotspot.get("drift_score"),
            "label": hotspot.get("label"),
        })

    # Check for cycles (look at blocking reasons)
    blocking_reasons = release_eval.get("blocking_reasons", [])
    has_cycles = any("cycle" in reason.lower() for reason in blocking_reasons)

    # Generate headline
    headline = _generate_dag_headline(
        health_score=health_score,
        status_light=status_light,
        drift_hotspot_count=len(drift_hotspots),
        has_cycles=has_cycles,
        has_regressions=len(sustained_regressions) > 0,
    )

    return {
        "status_light": status_light.value,
        "health_score": health_score,
        "drift_hotspots": drift_hotspots,
        "headline": headline,
    }


# =============================================================================
# PHASE V: DAG × Topology × HT Consistency
# =============================================================================

class CrossLayerStatus(str, Enum):
    """Status for cross-layer consistency checks."""
    CONSISTENT = "CONSISTENT"
    TENSION = "TENSION"
    CONFLICT = "CONFLICT"


def _normalize_health_status(status: str) -> str:
    """
    Normalize various health status strings to a common format.

    Maps different status representations to: OK, WARN, BLOCK
    """
    status_upper = status.upper()

    # Map OK variants
    if status_upper in ("OK", "PASS", "HEALTHY", "GREEN", "CONSISTENT", "ELIGIBLE"):
        return "OK"

    # Map WARN variants
    if status_upper in ("WARN", "WARNING", "YELLOW", "ATTENTION", "CAUTION", "TENSION"):
        return "WARN"

    # Map BLOCK variants
    if status_upper in ("BLOCK", "FAIL", "ERROR", "RED", "CONFLICT", "INELIGIBLE", "CRITICAL"):
        return "BLOCK"

    # Default to WARN for unknown statuses
    return "WARN"


# =============================================================================
# TASK 1: DAG × Topology Consistency Checker
# =============================================================================

def check_dag_topology_consistency(
    dag_health: Dict[str, Any],
    topology_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check consistency between DAG structural health and topology signal.

    This function detects misalignments between DAG structure and
    topology layer to ensure multi-layer consistency.

    Args:
        dag_health: DAG health status dict with keys like:
            - status: "OK" | "WARN" | "BLOCK" (or variants)
            - dag_ok: bool (optional)
            - health_score: float (optional)
        topology_signal: Topology health dict with keys like:
            - status: "OK" | "WARN" | "BLOCK" (or variants)
            - healthy: bool (optional)
            - score: float (optional)

    Returns:
        Dictionary containing:
        - consistent: bool - Whether layers are consistent
        - status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - reasons: List of strings explaining the status
    """
    reasons: List[str] = []

    # Extract and normalize DAG status
    dag_status_raw = dag_health.get("status", dag_health.get("gating_level", "OK"))
    dag_status = _normalize_health_status(str(dag_status_raw))
    dag_ok = dag_health.get("dag_ok", dag_health.get("release_ok", dag_status == "OK"))
    dag_score = dag_health.get("health_score", 1.0 if dag_status == "OK" else 0.5)

    # Extract and normalize topology status
    topo_status_raw = topology_signal.get("status", topology_signal.get("health", "OK"))
    topo_status = _normalize_health_status(str(topo_status_raw))
    topo_healthy = topology_signal.get("healthy", topology_signal.get("ok", topo_status == "OK"))
    topo_score = topology_signal.get("score", topology_signal.get("health_score", 1.0 if topo_status == "OK" else 0.5))

    # Determine consistency status
    if dag_status == "OK" and topo_status == "OK":
        # Both OK - consistent
        status = CrossLayerStatus.CONSISTENT
        consistent = True
        reasons.append("DAG and Topology layers are both healthy")

    elif dag_status == "BLOCK" and topo_status == "OK":
        # DAG blocked but topology OK - conflict
        status = CrossLayerStatus.CONFLICT
        consistent = False
        reasons.append("DAG structure is BLOCK but Topology reports OK")
        reasons.append("Topology may not reflect DAG structural issues")

    elif dag_status == "OK" and topo_status == "BLOCK":
        # DAG OK but topology blocked - conflict
        status = CrossLayerStatus.CONFLICT
        consistent = False
        reasons.append("DAG structure is OK but Topology reports BLOCK")
        reasons.append("Topology issues may not be reflected in DAG structure")

    elif dag_status == "BLOCK" and topo_status == "BLOCK":
        # Both blocked - consistent (both agree there's a problem)
        status = CrossLayerStatus.CONSISTENT
        consistent = True
        reasons.append("DAG and Topology layers both report critical issues")

    elif dag_status == "WARN" or topo_status == "WARN":
        # At least one is WARN - tension
        status = CrossLayerStatus.TENSION
        consistent = True  # Tension is not inconsistent, just needs attention

        if dag_status == "WARN" and topo_status == "WARN":
            reasons.append("Both DAG and Topology report warnings")
        elif dag_status == "WARN":
            reasons.append(f"DAG reports WARN while Topology is {topo_status}")
        else:
            reasons.append(f"Topology reports WARN while DAG is {dag_status}")

    else:
        # Fallback - check if statuses match
        if dag_status == topo_status:
            status = CrossLayerStatus.CONSISTENT
            consistent = True
            reasons.append(f"DAG and Topology both report {dag_status}")
        else:
            status = CrossLayerStatus.TENSION
            consistent = True
            reasons.append(f"DAG reports {dag_status}, Topology reports {topo_status}")

    # Additional score-based checks
    if abs(dag_score - topo_score) > 0.3:
        if status == CrossLayerStatus.CONSISTENT:
            status = CrossLayerStatus.TENSION
        reasons.append(f"Health score divergence: DAG={dag_score:.2f}, Topology={topo_score:.2f}")

    return {
        "consistent": consistent,
        "status": status.value,
        "reasons": reasons,
    }


# =============================================================================
# TASK 2: DAG × HT Alignment View
# =============================================================================

def build_dag_ht_alignment_view(
    dag_health: Dict[str, Any],
    ht_replay_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an alignment view between DAG structural health and HT replay invariants.

    Detects cases where DAG structure and HT replay disagree:
    - DAG BLOCK + HT OK: Structural issues not caught by HT replay
    - DAG OK + HT BLOCK: HT invariant violations in structurally sound DAG

    Args:
        dag_health: DAG health status dict with keys like:
            - status: "OK" | "WARN" | "BLOCK"
            - release_ok: bool (optional)
            - blocking_reasons: list (optional)
        ht_replay_health: HT replay health dict with keys like:
            - status: "OK" | "WARN" | "BLOCK"
            - invariants_ok: bool (optional)
            - violations: list (optional)

    Returns:
        Dictionary containing:
        - aligned: bool - Whether DAG and HT are aligned
        - status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - dag_status: Normalized DAG status
        - ht_status: Normalized HT status
        - misalignment_type: None | "DAG_BLOCK_HT_OK" | "DAG_OK_HT_BLOCK" | "BOTH_WARN"
        - reasons: List of strings explaining alignment
    """
    reasons: List[str] = []

    # Extract and normalize DAG status
    dag_status_raw = dag_health.get("status", dag_health.get("gating_level", "OK"))
    dag_status = _normalize_health_status(str(dag_status_raw))
    dag_release_ok = dag_health.get("release_ok", dag_status != "BLOCK")
    dag_blocking = dag_health.get("blocking_reasons", [])

    # Extract and normalize HT status
    ht_status_raw = ht_replay_health.get("status", ht_replay_health.get("health", "OK"))
    ht_status = _normalize_health_status(str(ht_status_raw))
    ht_invariants_ok = ht_replay_health.get("invariants_ok", ht_replay_health.get("ok", ht_status == "OK"))
    ht_violations = ht_replay_health.get("violations", [])

    # Determine alignment
    misalignment_type = None

    if dag_status == "BLOCK" and ht_status == "OK":
        # DAG sees problems that HT doesn't
        aligned = False
        status = CrossLayerStatus.CONFLICT
        misalignment_type = "DAG_BLOCK_HT_OK"
        reasons.append("DAG structure is BLOCK but HT replay reports OK")
        reasons.append("Structural issues may not violate HT invariants")
        if dag_blocking:
            reasons.append(f"DAG blocking reasons: {', '.join(dag_blocking[:3])}")

    elif dag_status == "OK" and ht_status == "BLOCK":
        # HT sees problems that DAG doesn't
        aligned = False
        status = CrossLayerStatus.CONFLICT
        misalignment_type = "DAG_OK_HT_BLOCK"
        reasons.append("DAG structure is OK but HT replay reports BLOCK")
        reasons.append("HT invariant violations exist in structurally sound DAG")
        if ht_violations:
            reasons.append(f"HT violations: {', '.join(str(v) for v in ht_violations[:3])}")

    elif dag_status == "WARN" and ht_status == "WARN":
        # Both have warnings - tension but aligned
        aligned = True
        status = CrossLayerStatus.TENSION
        misalignment_type = "BOTH_WARN"
        reasons.append("Both DAG and HT report warnings")
        reasons.append("Review both layers for potential issues")

    elif dag_status == "WARN" or ht_status == "WARN":
        # One has warnings
        aligned = True
        status = CrossLayerStatus.TENSION
        if dag_status == "WARN":
            reasons.append(f"DAG reports WARN, HT reports {ht_status}")
        else:
            reasons.append(f"HT reports WARN, DAG reports {dag_status}")

    elif dag_status == "OK" and ht_status == "OK":
        # Both OK - fully aligned
        aligned = True
        status = CrossLayerStatus.CONSISTENT
        reasons.append("DAG structure and HT replay are both healthy")

    elif dag_status == "BLOCK" and ht_status == "BLOCK":
        # Both blocked - aligned on problem
        aligned = True
        status = CrossLayerStatus.CONSISTENT
        reasons.append("Both DAG and HT report critical issues")

    else:
        # Fallback
        aligned = dag_status == ht_status
        status = CrossLayerStatus.CONSISTENT if aligned else CrossLayerStatus.TENSION
        reasons.append(f"DAG: {dag_status}, HT: {ht_status}")

    return {
        "aligned": aligned,
        "status": status.value,
        "dag_status": dag_status,
        "ht_status": ht_status,
        "misalignment_type": misalignment_type,
        "reasons": reasons,
    }


# =============================================================================
# TASK 3: Extended Director Panel with Cross-Layer Status
# =============================================================================

def build_dag_director_panel_extended(
    drift_summary: Dict[str, Any],
    release_eval: Dict[str, Any],
    topology_signal: Optional[Dict[str, Any]] = None,
    ht_replay_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an extended Director DAG panel with cross-layer consistency.

    This extends the basic director panel with optional cross-layer
    status information from topology and HT replay layers.

    Args:
        drift_summary: Serialized GlobalHealthSummary
        release_eval: Result from evaluate_dag_for_release()
        topology_signal: Optional topology health signal
        ht_replay_health: Optional HT replay health signal

    Returns:
        Dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - health_score: float (0.0 to 1.0)
        - drift_hotspots: List of hotspot summaries
        - headline: Neutral sentence describing DAG structural posture
        - cross_layer_status: Optional "CONSISTENT" | "TENSION" | "CONFLICT"
        - cross_layer_reasons: Optional list of cross-layer consistency reasons
    """
    # Build base panel
    panel = build_dag_director_panel(drift_summary, release_eval)

    # If no cross-layer signals provided, return base panel
    if topology_signal is None and ht_replay_health is None:
        return panel

    cross_layer_reasons: List[str] = []
    cross_layer_statuses: List[str] = []

    # Check topology consistency if provided
    if topology_signal is not None:
        topo_consistency = check_dag_topology_consistency(release_eval, topology_signal)
        cross_layer_statuses.append(topo_consistency["status"])
        if topo_consistency["status"] != CrossLayerStatus.CONSISTENT.value:
            cross_layer_reasons.extend([f"[Topology] {r}" for r in topo_consistency["reasons"]])
        elif topo_consistency["reasons"]:
            cross_layer_reasons.append(f"[Topology] {topo_consistency['reasons'][0]}")

    # Check HT alignment if provided
    if ht_replay_health is not None:
        ht_alignment = build_dag_ht_alignment_view(release_eval, ht_replay_health)
        cross_layer_statuses.append(ht_alignment["status"])
        if ht_alignment["status"] != CrossLayerStatus.CONSISTENT.value:
            cross_layer_reasons.extend([f"[HT] {r}" for r in ht_alignment["reasons"]])
        elif ht_alignment["reasons"]:
            cross_layer_reasons.append(f"[HT] {ht_alignment['reasons'][0]}")

    # Determine overall cross-layer status (worst case)
    if CrossLayerStatus.CONFLICT.value in cross_layer_statuses:
        cross_layer_status = CrossLayerStatus.CONFLICT.value
    elif CrossLayerStatus.TENSION.value in cross_layer_statuses:
        cross_layer_status = CrossLayerStatus.TENSION.value
    else:
        cross_layer_status = CrossLayerStatus.CONSISTENT.value

    # Upgrade status light if cross-layer conflict
    if cross_layer_status == CrossLayerStatus.CONFLICT.value:
        if panel["status_light"] == StatusLight.GREEN.value:
            panel["status_light"] = StatusLight.YELLOW.value
    elif cross_layer_status == CrossLayerStatus.TENSION.value:
        if panel["status_light"] == StatusLight.GREEN.value:
            # Tension doesn't upgrade green, but add note
            pass

    # Add cross-layer fields
    panel["cross_layer_status"] = cross_layer_status
    panel["cross_layer_reasons"] = cross_layer_reasons

    return panel


def check_multilayer_consistency(
    dag_health: Dict[str, Any],
    topology_signal: Optional[Dict[str, Any]] = None,
    ht_replay_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check consistency across all available layers (DAG, Topology, HT).

    This is a convenience function that aggregates all cross-layer
    consistency checks into a single result.

    Args:
        dag_health: DAG health status
        topology_signal: Optional topology health signal
        ht_replay_health: Optional HT replay health signal

    Returns:
        Dictionary containing:
        - overall_consistent: bool
        - overall_status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - layer_statuses: Dict of individual layer statuses
        - all_reasons: Aggregated list of reasons
        - conflict_count: Number of conflicts detected
        - tension_count: Number of tensions detected
    """
    layer_statuses: Dict[str, str] = {}
    all_reasons: List[str] = []
    conflict_count = 0
    tension_count = 0

    # Normalize DAG status
    dag_status = _normalize_health_status(str(dag_health.get("status", "OK")))
    layer_statuses["dag"] = dag_status

    # Check topology if provided
    if topology_signal is not None:
        topo_result = check_dag_topology_consistency(dag_health, topology_signal)
        layer_statuses["topology"] = topo_result["status"]
        all_reasons.extend(topo_result["reasons"])
        if topo_result["status"] == CrossLayerStatus.CONFLICT.value:
            conflict_count += 1
        elif topo_result["status"] == CrossLayerStatus.TENSION.value:
            tension_count += 1

    # Check HT if provided
    if ht_replay_health is not None:
        ht_result = build_dag_ht_alignment_view(dag_health, ht_replay_health)
        layer_statuses["ht"] = ht_result["status"]
        all_reasons.extend(ht_result["reasons"])
        if ht_result["status"] == CrossLayerStatus.CONFLICT.value:
            conflict_count += 1
        elif ht_result["status"] == CrossLayerStatus.TENSION.value:
            tension_count += 1

    # Determine overall status
    if conflict_count > 0:
        overall_status = CrossLayerStatus.CONFLICT.value
        overall_consistent = False
    elif tension_count > 0:
        overall_status = CrossLayerStatus.TENSION.value
        overall_consistent = True  # Tension is not inconsistent
    else:
        overall_status = CrossLayerStatus.CONSISTENT.value
        overall_consistent = True

    return {
        "overall_consistent": overall_consistent,
        "overall_status": overall_status,
        "layer_statuses": layer_statuses,
        "all_reasons": all_reasons,
        "conflict_count": conflict_count,
        "tension_count": tension_count,
    }


# =============================================================================
# PHASE VI: Cross-Layer Governance Signal
# =============================================================================

class GovernanceStatus(str, Enum):
    """Governance status for structural cohesion."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def to_governance_signal_for_structure(
    consistency_result: Dict[str, Any],
    ht_alignment_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert cross-layer consistency results into a consolidated governance signal.

    This function produces a single, legible constraint signal for CLAUDE I
    and the Console, mapping multi-layer structure checks to governance actions.

    Mapping:
    - Any CONFLICT → BLOCK
    - Any TENSION (without BLOCK) → WARN
    - Full CONSISTENT → OK

    Args:
        consistency_result: Result from check_multilayer_consistency() or
            check_dag_topology_consistency()
        ht_alignment_view: Optional result from build_dag_ht_alignment_view()

    Returns:
        Dictionary containing:
        - status: "OK" | "WARN" | "BLOCK"
        - structural_cohesion: bool - Whether structure is cohesive
        - blocking_rules: List of rule identifiers that triggered blocking
        - warning_rules: List of rule identifiers that triggered warnings
        - layer_summary: Dict summarizing each layer's status
        - reasons: List of human-readable reasons
    """
    blocking_rules: List[str] = []
    warning_rules: List[str] = []
    reasons: List[str] = []

    # Extract consistency result fields
    overall_status = consistency_result.get("overall_status", "CONSISTENT")
    overall_consistent = consistency_result.get("overall_consistent", True)
    layer_statuses = consistency_result.get("layer_statuses", {})
    all_reasons = consistency_result.get("all_reasons", [])
    conflict_count = consistency_result.get("conflict_count", 0)
    tension_count = consistency_result.get("tension_count", 0)

    # Check if this is a simple two-layer result (from check_dag_topology_consistency)
    if "consistent" in consistency_result and "overall_status" not in consistency_result:
        # Convert simple result to multilayer format
        overall_status = consistency_result.get("status", "CONSISTENT")
        overall_consistent = consistency_result.get("consistent", True)
        all_reasons = consistency_result.get("reasons", [])
        if overall_status == CrossLayerStatus.CONFLICT.value:
            conflict_count = 1
        elif overall_status == CrossLayerStatus.TENSION.value:
            tension_count = 1

    # Process HT alignment view if provided
    ht_misalignment_type = None
    if ht_alignment_view is not None:
        ht_status = ht_alignment_view.get("status", "CONSISTENT")
        ht_aligned = ht_alignment_view.get("aligned", True)
        ht_misalignment_type = ht_alignment_view.get("misalignment_type")
        ht_reasons = ht_alignment_view.get("reasons", [])

        if ht_status == CrossLayerStatus.CONFLICT.value:
            conflict_count += 1
            if ht_misalignment_type:
                blocking_rules.append(ht_misalignment_type)
            all_reasons.extend(ht_reasons)
        elif ht_status == CrossLayerStatus.TENSION.value:
            tension_count += 1
            if ht_misalignment_type:
                warning_rules.append(ht_misalignment_type)
            all_reasons.extend(ht_reasons)

    # Determine governance status based on conflicts and tensions
    if conflict_count > 0 or overall_status == CrossLayerStatus.CONFLICT.value:
        governance_status = GovernanceStatus.BLOCK
        structural_cohesion = False

        # Identify blocking rules from layer mismatches
        if "topology" in layer_statuses:
            topo_status = layer_statuses["topology"]
            dag_status = layer_statuses.get("dag", "OK")
            if topo_status == CrossLayerStatus.CONFLICT.value:
                if dag_status == "BLOCK":
                    blocking_rules.append("DAG_BLOCK_TOPOLOGY_OK")
                elif dag_status == "OK":
                    blocking_rules.append("DAG_OK_TOPOLOGY_BLOCK")

        if "ht" in layer_statuses:
            ht_layer_status = layer_statuses["ht"]
            dag_status = layer_statuses.get("dag", "OK")
            if ht_layer_status == CrossLayerStatus.CONFLICT.value:
                if dag_status == "BLOCK":
                    if "DAG_BLOCK_HT_OK" not in blocking_rules:
                        blocking_rules.append("DAG_BLOCK_HT_OK")
                elif dag_status == "OK":
                    if "DAG_OK_HT_BLOCK" not in blocking_rules:
                        blocking_rules.append("DAG_OK_HT_BLOCK")

        reasons.append("Structural cohesion BLOCKED due to cross-layer conflicts")

    elif tension_count > 0 or overall_status == CrossLayerStatus.TENSION.value:
        governance_status = GovernanceStatus.WARN
        structural_cohesion = True  # Tension doesn't break cohesion

        # Identify warning rules
        if "topology" in layer_statuses:
            if layer_statuses["topology"] == CrossLayerStatus.TENSION.value:
                warning_rules.append("DAG_TOPOLOGY_TENSION")

        if "ht" in layer_statuses:
            if layer_statuses["ht"] == CrossLayerStatus.TENSION.value:
                if ht_misalignment_type and ht_misalignment_type not in warning_rules:
                    warning_rules.append(ht_misalignment_type)
                elif "DAG_HT_TENSION" not in warning_rules:
                    warning_rules.append("DAG_HT_TENSION")

        reasons.append("Structural cohesion has warnings requiring attention")

    else:
        governance_status = GovernanceStatus.OK
        structural_cohesion = True
        reasons.append("Structural cohesion is fully consistent across all layers")

    # Build layer summary
    layer_summary = {
        "dag": layer_statuses.get("dag", "OK"),
    }
    if "topology" in layer_statuses:
        layer_summary["topology"] = layer_statuses["topology"]
    if "ht" in layer_statuses:
        layer_summary["ht"] = layer_statuses["ht"]
    if ht_alignment_view is not None and "ht" not in layer_summary:
        layer_summary["ht"] = ht_alignment_view.get("status", "CONSISTENT")

    # Add detailed reasons
    reasons.extend(all_reasons)

    return {
        "status": governance_status.value,
        "structural_cohesion": structural_cohesion,
        "blocking_rules": blocking_rules,
        "warning_rules": warning_rules,
        "layer_summary": layer_summary,
        "reasons": reasons,
    }


# =============================================================================
# TASK 2: Global Console Pane Adapter
# =============================================================================

def build_structure_console_pane(
    governance_signal: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structure console pane ready to slot into global_health["structure"].

    This provides a compact summary of structural health suitable for
    the global console display.

    Args:
        governance_signal: Result from to_governance_signal_for_structure()
        director_panel: Optional result from build_dag_director_panel_extended()

    Returns:
        Dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - headline: Single sentence describing structural status
        - cross_layer_status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - cohesion_ok: bool
        - blocking_rules: List of blocking rule identifiers (if any)
    """
    # Extract governance signal fields
    governance_status = governance_signal.get("status", "OK")
    structural_cohesion = governance_signal.get("structural_cohesion", True)
    blocking_rules = governance_signal.get("blocking_rules", [])
    warning_rules = governance_signal.get("warning_rules", [])
    layer_summary = governance_signal.get("layer_summary", {})

    # Determine status light
    if governance_status == GovernanceStatus.BLOCK.value:
        status_light = StatusLight.RED.value
    elif governance_status == GovernanceStatus.WARN.value:
        status_light = StatusLight.YELLOW.value
    else:
        status_light = StatusLight.GREEN.value

    # Override with director panel if provided and more severe
    if director_panel is not None:
        panel_light = director_panel.get("status_light", "GREEN")
        if panel_light == StatusLight.RED.value and status_light != StatusLight.RED.value:
            status_light = StatusLight.RED.value
        elif panel_light == StatusLight.YELLOW.value and status_light == StatusLight.GREEN.value:
            status_light = StatusLight.YELLOW.value

    # Determine cross-layer status
    if governance_status == GovernanceStatus.BLOCK.value:
        cross_layer_status = CrossLayerStatus.CONFLICT.value
    elif governance_status == GovernanceStatus.WARN.value:
        cross_layer_status = CrossLayerStatus.TENSION.value
    else:
        cross_layer_status = CrossLayerStatus.CONSISTENT.value

    # Generate headline
    headline = _generate_structure_headline(
        status_light=status_light,
        cross_layer_status=cross_layer_status,
        blocking_rules=blocking_rules,
        warning_rules=warning_rules,
        layer_summary=layer_summary,
    )

    result = {
        "status_light": status_light,
        "headline": headline,
        "cross_layer_status": cross_layer_status,
        "cohesion_ok": structural_cohesion,
    }

    # Include blocking rules if any
    if blocking_rules:
        result["blocking_rules"] = blocking_rules

    return result


def _generate_structure_headline(
    status_light: str,
    cross_layer_status: str,
    blocking_rules: List[str],
    warning_rules: List[str],
    layer_summary: Dict[str, str],
) -> str:
    """Generate a headline for the structure console pane."""
    if status_light == StatusLight.RED.value:
        if blocking_rules:
            # Describe the specific conflict
            rule = blocking_rules[0]
            if rule == "DAG_BLOCK_HT_OK":
                return "DAG structure blocked but HT invariants pass"
            elif rule == "DAG_OK_HT_BLOCK":
                return "HT invariants fail despite healthy DAG structure"
            elif rule == "DAG_BLOCK_TOPOLOGY_OK":
                return "DAG structure blocked but Topology reports healthy"
            elif rule == "DAG_OK_TOPOLOGY_BLOCK":
                return "Topology blocked despite healthy DAG structure"
            else:
                return f"Cross-layer conflict detected: {rule}"
        else:
            return "Structural cohesion blocked due to cross-layer conflicts"

    elif status_light == StatusLight.YELLOW.value:
        if warning_rules:
            rule = warning_rules[0]
            if rule == "BOTH_WARN":
                return "Both DAG and HT layers report warnings"
            elif rule == "DAG_TOPOLOGY_TENSION":
                return "Tension detected between DAG and Topology layers"
            elif rule == "DAG_HT_TENSION":
                return "Tension detected between DAG and HT layers"
            else:
                return f"Cross-layer tension: {rule}"
        else:
            return "Structural layers show tension requiring attention"

    else:
        # Count healthy layers
        healthy_count = sum(1 for s in layer_summary.values()
                           if s in ("OK", "CONSISTENT"))
        total_count = len(layer_summary)

        if total_count > 1:
            return f"Structural cohesion healthy across {healthy_count} layers"
        else:
            return "Structural cohesion is healthy"


def build_global_health_structure_entry(
    dag_health: Dict[str, Any],
    topology_signal: Optional[Dict[str, Any]] = None,
    ht_replay_health: Optional[Dict[str, Any]] = None,
    drift_summary: Optional[Dict[str, Any]] = None,
    release_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete structure entry for global_health integration.

    This is a convenience function that chains all the necessary calls
    to produce a final structure entry ready for the global health console.

    Args:
        dag_health: DAG health status
        topology_signal: Optional topology health signal
        ht_replay_health: Optional HT replay health signal
        drift_summary: Optional GlobalHealthSummary (for director panel)
        release_eval: Optional release evaluation (for director panel)

    Returns:
        Dictionary ready to be assigned to global_health["structure"]
    """
    # Check multilayer consistency
    consistency_result = check_multilayer_consistency(
        dag_health,
        topology_signal,
        ht_replay_health,
    )

    # Build HT alignment view if HT health provided
    ht_alignment = None
    if ht_replay_health is not None:
        ht_alignment = build_dag_ht_alignment_view(dag_health, ht_replay_health)

    # Build governance signal
    governance_signal = to_governance_signal_for_structure(
        consistency_result,
        ht_alignment,
    )

    # Build director panel if we have drift summary and release eval
    director_panel = None
    if drift_summary is not None and release_eval is not None:
        director_panel = build_dag_director_panel_extended(
            drift_summary,
            release_eval,
            topology_signal,
            ht_replay_health,
        )

    # Build console pane
    console_pane = build_structure_console_pane(governance_signal, director_panel)

    # Merge governance signal details into console pane for complete entry
    return {
        **console_pane,
        "governance_status": governance_signal["status"],
        "layer_summary": governance_signal["layer_summary"],
        "warning_rules": governance_signal.get("warning_rules", []),
        "reasons": governance_signal.get("reasons", [])[:3],  # Limit to top 3 reasons
    }
