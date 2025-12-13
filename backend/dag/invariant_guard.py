"""DAG Invariant Guard module.

Provides DAG invariant verification and enforcement.

Structural Governance Signal emission implemented per:
docs/system_law/Structural_Cohesion_PhaseX.md Section 6.1

SHADOW MODE: All structural signals are for observation/logging only.
"""

import json
import secrets
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Structural Governance Data Structures
# =============================================================================

@dataclass
class StructuralViolation:
    """
    Single structural invariant violation.

    SHADOW MODE: Violations are logged, not enforced.
    See: docs/system_law/schemas/structural/structural_governance_signal.schema.json
    """
    invariant_id: str  # SI-001 through SI-010
    layer: str  # DAG, TOPOLOGY, HT
    severity: str  # TENSION or CONFLICT
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation_hint: Optional[str] = None


@dataclass
class LayerStatus:
    """Status for a single structural layer."""
    name: str
    status: str  # CONSISTENT, TENSION, CONFLICT
    score: float  # 0.0 to 1.0
    violation_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralGovernanceSignal:
    """
    Cross-layer structural governance signal.

    SHADOW MODE: This signal is for observation/logging only.
    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 6.3
    """
    signal_id: str
    timestamp: str
    run_id: Optional[str] = None
    cycle: int = 0

    # Layer statuses
    dag_status: str = "CONSISTENT"  # CONSISTENT, TENSION, CONFLICT
    topology_status: str = "CONSISTENT"
    ht_status: str = "CONSISTENT"

    # Combined
    combined_severity: str = "CONSISTENT"
    cohesion_score: float = 1.0
    admissible: bool = True

    # Details
    dag_details: Dict[str, Any] = field(default_factory=dict)
    topology_details: Dict[str, Any] = field(default_factory=dict)
    ht_details: Dict[str, Any] = field(default_factory=dict)

    # Violations
    violations: List[StructuralViolation] = field(default_factory=list)

    # Layer scores
    layer_scores: Dict[str, float] = field(default_factory=dict)
    layer_weights: Dict[str, float] = field(default_factory=lambda: {
        "dag_weight": 0.4,
        "topology_weight": 0.4,
        "ht_weight": 0.2,
    })

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary conforming to schema."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "cycle": self.cycle,
            "dag_status": self.dag_status,
            "topology_status": self.topology_status,
            "ht_status": self.ht_status,
            "combined_severity": self.combined_severity,
            "cohesion_score": round(self.cohesion_score, 4),
            "admissible": self.admissible,
            "dag_details": self.dag_details,
            "topology_details": self.topology_details,
            "ht_details": self.ht_details,
            "violations": [
                {
                    "invariant_id": v.invariant_id,
                    "layer": v.layer,
                    "severity": v.severity,
                    "message": v.message,
                    "details": v.details,
                    "remediation_hint": v.remediation_hint,
                }
                for v in self.violations
            ],
            "layer_scores": self.layer_scores,
            "layer_weights": self.layer_weights,
            "metadata": self.metadata,
        }

    @staticmethod
    def generate_signal_id() -> str:
        """Generate unique signal ID per schema (sgs_ + 16 hex chars)."""
        return f"sgs_{secrets.token_hex(8)}"


@dataclass
class InvariantViolation:
    """DAG invariant violation record."""
    invariant_name: str
    node_id: str
    message: str
    severity: str = "error"


@dataclass
class InvariantCheckResult:
    """Result of invariant check."""
    valid: bool = True
    violations: List[InvariantViolation] = field(default_factory=list)
    checked_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def check_dag_invariants(
    nodes: List[Dict[str, Any]],
    edges: List[tuple],
) -> InvariantCheckResult:
    """Check DAG invariants on nodes and edges."""
    violations = []

    # Check for cycles
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    node_ids = {n.get("id", str(i)) for i, n in enumerate(nodes)}

    # Build adjacency
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for src, dst, *_ in edges:
        if src in adj:
            adj[src].append(dst)

    def has_cycle(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if has_cycle(neighbor):
                return True
        in_stack.discard(node)
        return False

    for node_id in node_ids:
        if has_cycle(node_id):
            violations.append(InvariantViolation(
                invariant_name="acyclic",
                node_id=node_id,
                message="Cycle detected in DAG",
            ))
            break

    return InvariantCheckResult(
        valid=len(violations) == 0,
        violations=violations,
        checked_nodes=len(nodes),
        metadata={"edge_count": len(edges)},
    )


def verify_node_invariant(
    node: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
) -> List[InvariantViolation]:
    """Verify invariants for a single node."""
    violations = []
    required_fields = required_fields or ["id"]

    node_id = node.get("id", "unknown")

    for fld in required_fields:
        if fld not in node:
            violations.append(InvariantViolation(
                invariant_name=f"required_field_{fld}",
                node_id=node_id,
                message=f"Missing required field: {fld}",
            ))

    return violations


def enforce_dag_structure(
    nodes: List[Dict[str, Any]],
    edges: List[tuple],
) -> Dict[str, Any]:
    """Enforce DAG structure and return sanitized result."""
    result = check_dag_invariants(nodes, edges)

    return {
        "valid": result.valid,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "violations": [
            {"name": v.invariant_name, "node": v.node_id, "message": v.message}
            for v in result.violations
        ],
    }


@dataclass
class SliceProfile:
    """Slice profile for DAG analysis."""
    slice_id: str
    max_depth: int = 0
    max_branching_factor: float = 1.0
    node_kind_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofDag:
    """Proof DAG representation."""
    slices: Dict[str, SliceProfile] = field(default_factory=dict)
    metric_ledger: List[Dict[str, Any]] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node_id: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a node to the DAG."""
        self.nodes.append({"id": node_id, **(data or {})})

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add an edge to the DAG."""
        self.edges.append((from_id, to_id))

    def check_invariants(self) -> InvariantCheckResult:
        """Check DAG invariants."""
        return check_dag_invariants(self.nodes, list(self.edges))


def evaluate_dag_invariants(
    dag: ProofDag,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate all DAG invariants."""
    rules = rules or {}
    violations = []

    # Check slice-specific rules
    for slice_id, profile in dag.slices.items():
        max_depth_rule = rules.get("max_depth_per_slice", {}).get(slice_id)
        if max_depth_rule is not None and profile.max_depth > max_depth_rule:
            violations.append({
                "invariant": "max_depth",
                "slice": slice_id,
                "message": f"Depth {profile.max_depth} exceeds max {max_depth_rule}",
            })

        allowed_kinds = rules.get("allowed_node_kinds", {}).get(slice_id)
        if allowed_kinds is not None:
            for kind in profile.node_kind_counts:
                if kind not in allowed_kinds:
                    violations.append({
                        "invariant": "allowed_node_kinds",
                        "slice": slice_id,
                        "message": f"Node kind {kind} not allowed",
                    })

    return {
        "status": "OK" if not violations else "VIOLATED",
        "violated_invariants": violations,
        "slices_checked": list(dag.slices.keys()),
    }


def load_invariant_rules(path: str) -> Dict[str, Any]:
    """Load invariant rules from file."""
    p = Path(path)
    if not p.exists():
        return {}

    content = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(content)
    elif p.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content)

    return {}


def summarize_dag_invariants_for_global_health(
    dag: ProofDag,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize DAG invariants for global health display."""
    result = evaluate_dag_invariants(dag, rules)
    return {
        "status": result["status"],
        "violations": len(result["violated_invariants"]),
        "slices": len(dag.slices),
    }


# =============================================================================
# Structural Governance Signal Emission
# =============================================================================

def emit_structural_signal(
    dag: Optional[ProofDag] = None,
    topology_state: Optional[Dict[str, Any]] = None,
    ht_state: Optional[Dict[str, Any]] = None,
    rules: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    cycle: int = 0,
    triggered_by: str = "MANUAL",
) -> StructuralGovernanceSignal:
    """
    Emit a structural governance signal from cross-layer invariant checks.

    SHADOW MODE: This signal is for observation/logging only.
    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 6.3

    Args:
        dag: ProofDag to check DAG layer invariants
        topology_state: Dict with topology state (H, rho, tau, beta, in_omega, omega_exit_streak)
        ht_state: Dict with HT state (total_anchors, verified_anchors, pending_anchors, failed_anchors)
        rules: Optional invariant rules
        run_id: Associated P4 run ID
        cycle: Current cycle number
        triggered_by: What triggered this signal (PRE_RUN, CYCLE, MANUAL, PERIODIC)

    Returns:
        StructuralGovernanceSignal with cross-layer status
    """
    import time
    start_time = time.perf_counter()

    violations: List[StructuralViolation] = []
    rules = rules or {}

    # -------------------------------------------------------------------------
    # DAG Layer Checks (SI-001, SI-002, SI-003, SI-004)
    # -------------------------------------------------------------------------
    dag_score = 1.0
    dag_status = "CONSISTENT"
    dag_details: Dict[str, Any] = {
        "acyclic": True,
        "node_integrity": True,
        "depth_violations": 0,
        "kind_violations": 0,
        "nodes_checked": 0,
        "edges_checked": 0,
        "slices_checked": [],
    }

    if dag is not None:
        # SI-001: DAG Acyclicity (CRITICAL)
        dag_result = check_dag_invariants(dag.nodes, list(dag.edges))
        dag_details["nodes_checked"] = dag_result.checked_nodes
        dag_details["edges_checked"] = dag_result.metadata.get("edge_count", 0)

        if not dag_result.valid:
            dag_details["acyclic"] = False
            dag_status = "CONFLICT"
            dag_score = 0.0
            violations.append(StructuralViolation(
                invariant_id="SI-001",
                layer="DAG",
                severity="CONFLICT",
                message="DAG acyclicity violation: cycle detected",
                details={"violations": [v.message for v in dag_result.violations]},
                remediation_hint="Review recent edges for circular references",
            ))

        # SI-002: Node Required Fields
        node_violations = 0
        for node in dag.nodes:
            node_errs = verify_node_invariant(node, rules.get("required_fields"))
            if node_errs:
                node_violations += len(node_errs)
        if node_violations > 0:
            dag_details["node_integrity"] = False
            if dag_status == "CONSISTENT":
                dag_status = "TENSION"
            violations.append(StructuralViolation(
                invariant_id="SI-002",
                layer="DAG",
                severity="TENSION",
                message=f"Node integrity violations: {node_violations} nodes missing required fields",
                details={"count": node_violations},
            ))
            dag_score = max(0.0, dag_score - 0.1 * node_violations)

        # SI-003, SI-004: Slice constraints
        slice_result = evaluate_dag_invariants(dag, rules)
        dag_details["slices_checked"] = slice_result.get("slices_checked", [])

        for v in slice_result.get("violated_invariants", []):
            if v["invariant"] == "max_depth":
                dag_details["depth_violations"] += 1
                if dag_status == "CONSISTENT":
                    dag_status = "TENSION"
                violations.append(StructuralViolation(
                    invariant_id="SI-003",
                    layer="DAG",
                    severity="TENSION",
                    message=v["message"],
                    details={"slice": v.get("slice")},
                ))
                dag_score = max(0.0, dag_score - 0.05)

            elif v["invariant"] == "allowed_node_kinds":
                dag_details["kind_violations"] += 1
                if dag_status == "CONSISTENT":
                    dag_status = "TENSION"
                violations.append(StructuralViolation(
                    invariant_id="SI-004",
                    layer="DAG",
                    severity="TENSION",
                    message=v["message"],
                    details={"slice": v.get("slice")},
                ))
                dag_score = max(0.0, dag_score - 0.05)

    # -------------------------------------------------------------------------
    # Topology Layer Checks (SI-005, SI-006, SI-007, SI-008, SI-009)
    # -------------------------------------------------------------------------
    topology_score = 1.0
    topology_status = "CONSISTENT"
    topology_details: Dict[str, Any] = {
        "state_bounds_ok": True,
        "safe_region_defined": True,
        "observation_schema_ok": True,
        "twin_schema_ok": True,
        "divergence_schema_ok": True,
        "current_state": {},
        "omega_exit_streak": 0,
    }

    if topology_state is not None:
        H = topology_state.get("H", 0.5)
        rho = topology_state.get("rho", 0.5)
        tau = topology_state.get("tau", 0.2)
        beta = topology_state.get("beta", 0.0)
        in_omega = topology_state.get("in_omega", True)
        omega_exit_streak = topology_state.get("omega_exit_streak", 0)

        topology_details["current_state"] = {
            "H": H, "rho": rho, "tau": tau, "beta": beta, "in_omega": in_omega
        }
        topology_details["omega_exit_streak"] = omega_exit_streak

        # SI-005: State Space Bounds
        bounds_ok = True
        if not (0.0 <= H <= 1.0):
            bounds_ok = False
        if not (0.0 <= rho <= 1.0):
            bounds_ok = False
        if not (0.16 <= tau <= 0.24):
            bounds_ok = False
        if not (0.0 <= beta <= 1.0):
            bounds_ok = False

        if not bounds_ok:
            topology_details["state_bounds_ok"] = False
            topology_status = "CONFLICT"
            topology_score = 0.5
            violations.append(StructuralViolation(
                invariant_id="SI-005",
                layer="TOPOLOGY",
                severity="CONFLICT",
                message="State space bounds violation",
                details={"H": H, "rho": rho, "tau": tau, "beta": beta},
                remediation_hint="Check USLA state transition logic",
            ))

        # SI-006: Safe Region (Omega) tracking
        if omega_exit_streak > 100:
            if topology_status == "CONSISTENT":
                topology_status = "TENSION"
            topology_score = max(0.0, topology_score - 0.2)
            violations.append(StructuralViolation(
                invariant_id="SI-006",
                layer="TOPOLOGY",
                severity="TENSION",
                message=f"State outside safe region for {omega_exit_streak} consecutive cycles",
                details={"omega_exit_streak": omega_exit_streak},
            ))
        elif omega_exit_streak > 0 and not in_omega:
            topology_score = max(0.0, topology_score - 0.01 * omega_exit_streak)

        # SI-007, SI-008, SI-009: Schema compliance (assume OK unless explicitly provided)
        topology_details["observation_schema_ok"] = topology_state.get("observation_schema_ok", True)
        topology_details["twin_schema_ok"] = topology_state.get("twin_schema_ok", True)
        topology_details["divergence_schema_ok"] = topology_state.get("divergence_schema_ok", True)

    # -------------------------------------------------------------------------
    # HT Layer Checks (SI-010)
    # -------------------------------------------------------------------------
    ht_score = 1.0
    ht_status = "CONSISTENT"
    ht_details: Dict[str, Any] = {
        "anchor_integrity": True,
        "total_anchors": 0,
        "verified_anchors": 0,
        "pending_anchors": 0,
        "failed_anchors": 0,
    }

    if ht_state is not None:
        total = ht_state.get("total_anchors", 0)
        verified = ht_state.get("verified_anchors", 0)
        pending = ht_state.get("pending_anchors", 0)
        failed = ht_state.get("failed_anchors", 0)

        ht_details["total_anchors"] = total
        ht_details["verified_anchors"] = verified
        ht_details["pending_anchors"] = pending
        ht_details["failed_anchors"] = failed

        # SI-010: Truth Anchor Integrity (CRITICAL for admissibility)
        if failed > 0:
            ht_details["anchor_integrity"] = False
            ht_status = "CONFLICT"
            ht_score = 0.0
            violations.append(StructuralViolation(
                invariant_id="SI-010",
                layer="HT",
                severity="CONFLICT",
                message=f"Truth anchor integrity violation: {failed} anchors failed verification",
                details={"failed_anchors": failed},
                remediation_hint="Re-verify failed anchors with Lean",
            ))
        elif total > 0:
            ht_score = verified / total

    # -------------------------------------------------------------------------
    # Compute Combined Severity and Cohesion Score
    # -------------------------------------------------------------------------
    severity_order = {"CONSISTENT": 0, "TENSION": 1, "CONFLICT": 2}
    severities = [dag_status, topology_status, ht_status]
    combined_severity = max(severities, key=lambda s: severity_order.get(s, 0))

    # Cohesion score = weighted average
    w_dag = 0.4
    w_topo = 0.4
    w_ht = 0.2
    cohesion_score = w_dag * dag_score + w_topo * topology_score + w_ht * ht_score

    # Admissibility: SI-001 or SI-010 CONFLICT blocks
    admissible = True
    blocking_violations = [v for v in violations if v.invariant_id in ("SI-001", "SI-010")]
    if blocking_violations:
        admissible = False

    # Build signal
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    signal = StructuralGovernanceSignal(
        signal_id=StructuralGovernanceSignal.generate_signal_id(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        cycle=cycle,
        dag_status=dag_status,
        topology_status=topology_status,
        ht_status=ht_status,
        combined_severity=combined_severity,
        cohesion_score=cohesion_score,
        admissible=admissible,
        dag_details=dag_details,
        topology_details=topology_details,
        ht_details=ht_details,
        violations=violations,
        layer_scores={
            "dag_score": round(dag_score, 4),
            "topology_score": round(topology_score, 4),
            "ht_score": round(ht_score, 4),
        },
        layer_weights={
            "dag_weight": w_dag,
            "topology_weight": w_topo,
            "ht_weight": w_ht,
        },
        metadata={
            "check_duration_ms": round(elapsed_ms, 2),
            "schema_version": "1.0.0",
            "triggered_by": triggered_by,
        },
    )

    return signal


def build_structural_cohesion_tile(
    signal: StructuralGovernanceSignal,
    sparkline_history: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Build a structural console tile from a governance signal.

    SHADOW MODE: This tile is for display only.
    See: docs/system_law/schemas/structural/structural_console_tile.schema.json

    Args:
        signal: StructuralGovernanceSignal to convert
        sparkline_history: Optional list of recent cohesion scores for sparkline

    Returns:
        Dict conforming to structural_console_tile.schema.json
    """
    # Map combined severity to overall status
    status_map = {
        "CONSISTENT": ("HEALTHY", "check_circle", "green"),
        "TENSION": ("DEGRADED", "warning", "yellow"),
        "CONFLICT": ("CRITICAL", "error", "red"),
    }
    overall_status, status_icon, status_color = status_map.get(
        signal.combined_severity, ("HEALTHY", "check_circle", "green")
    )

    # Map layer status to icon/color
    layer_icon_map = {
        "CONSISTENT": ("check", "green"),
        "TENSION": ("minus", "yellow"),
        "CONFLICT": ("x", "red"),
    }

    # Build layer tiles
    def make_layer_tile(name: str, status: str, score: float, details: Dict[str, Any]) -> Dict[str, Any]:
        icon, color = layer_icon_map.get(status, ("check", "green"))
        violation_count = len([v for v in signal.violations if v.layer == name.upper()])

        # Pick key metric based on layer
        key_metric: Dict[str, Any] = {"name": "Score", "value": f"{int(score * 100)}%"}
        if name == "DAG":
            key_metric = {"name": "Nodes", "value": details.get("nodes_checked", 0)}
        elif name == "Topology":
            omega_streak = details.get("omega_exit_streak", 0)
            if omega_streak > 0:
                key_metric = {"name": "Omega Exit Streak", "value": omega_streak, "unit": "cycles"}
            else:
                key_metric = {"name": "Omega Occupancy", "value": "OK"}
        elif name == "HT":
            key_metric = {"name": "Verified Anchors", "value": details.get("verified_anchors", 0)}

        return {
            "name": name,
            "status": status,
            "status_icon": icon,
            "status_color": color,
            "score": round(score, 4),
            "score_display": f"{int(score * 100)}%",
            "violation_count": violation_count,
            "key_metric": key_metric,
        }

    layers = {
        "dag": make_layer_tile(
            "DAG",
            signal.dag_status,
            signal.layer_scores.get("dag_score", 1.0),
            signal.dag_details,
        ),
        "topology": make_layer_tile(
            "Topology",
            signal.topology_status,
            signal.layer_scores.get("topology_score", 1.0),
            signal.topology_details,
        ),
        "ht": make_layer_tile(
            "HT",
            signal.ht_status,
            signal.layer_scores.get("ht_score", 1.0),
            signal.ht_details,
        ),
    }

    # Violation breakdown
    conflict_count = len([v for v in signal.violations if v.severity == "CONFLICT"])
    tension_count = len([v for v in signal.violations if v.severity == "TENSION"])

    # Top violations (max 3)
    top_violations = []
    for v in signal.violations[:3]:
        top_violations.append({
            "invariant_id": v.invariant_id,
            "layer": v.layer,
            "severity": v.severity,
            "short_message": v.message[:50] if len(v.message) > 50 else v.message,
        })

    # Admissibility
    admissibility: Dict[str, Any] = {"admissible": signal.admissible}
    if not signal.admissible:
        blocking = [v.invariant_id for v in signal.violations if v.invariant_id in ("SI-001", "SI-010")]
        admissibility["reason"] = "Blocking invariant violated"
        admissibility["blocking_invariants"] = blocking

    # Sparkline
    sparkline_data: Dict[str, Any] = {}
    if sparkline_history:
        sparkline_data = {
            "values": sparkline_history[-20:],  # Last 20 values
            "window_size": len(sparkline_history),
        }

    # Compute trend if we have history
    cohesion_trend = "STABLE"
    cohesion_delta = 0.0
    if sparkline_history and len(sparkline_history) >= 2:
        cohesion_delta = signal.cohesion_score - sparkline_history[-1]
        if cohesion_delta > 0.02:
            cohesion_trend = "IMPROVING"
        elif cohesion_delta < -0.02:
            cohesion_trend = "DEGRADING"

    tile = {
        "tile_id": f"sct_{secrets.token_hex(4)}",
        "timestamp": signal.timestamp,
        "run_id": signal.run_id,
        "cycle": signal.cycle,
        "overall_status": overall_status,
        "overall_status_icon": status_icon,
        "overall_status_color": status_color,
        "cohesion_score": round(signal.cohesion_score, 4),
        "cohesion_score_display": f"{int(signal.cohesion_score * 100)}%",
        "cohesion_trend": cohesion_trend,
        "cohesion_delta": round(cohesion_delta, 4),
        "layers": layers,
        "active_violations": len(signal.violations),
        "violation_breakdown": {
            "conflict": conflict_count,
            "tension": tension_count,
        },
        "top_violations": top_violations,
        "admissibility": admissibility,
        "sparkline_data": sparkline_data,
        "refresh_interval_ms": 2000 if signal.combined_severity != "CONSISTENT" else 5000,
        "last_signal_id": signal.signal_id,
    }

    return tile


# =============================================================================
# P4 Calibration & Evidence Integration (CLAUDE G → First Light)
# =============================================================================

def build_structural_calibration_for_p4(
    structural_signal: StructuralGovernanceSignal,
) -> Dict[str, Any]:
    """
    Build structural calibration summary for P4 calibration report.

    SHADOW MODE: This summary is for observation/logging only.
    Intended to attach under p4_calibration_report["structural_cohesion"].

    See: docs/system_law/Structural_Cohesion_PhaseX.md

    Args:
        structural_signal: StructuralGovernanceSignal from emit_structural_signal()

    Returns:
        Dict suitable for p4_calibration_report["structural_cohesion"]
    """
    # Extract blocking invariant statuses
    si_001_status = "PASS"
    si_010_status = "PASS"

    for v in structural_signal.violations:
        if v.invariant_id == "SI-001":
            si_001_status = "FAIL"
        elif v.invariant_id == "SI-010":
            si_010_status = "FAIL"

    # Collect key violations (first 5)
    key_violations = []
    for v in structural_signal.violations[:5]:
        key_violations.append({
            "id": v.invariant_id,
            "layer": v.layer,
            "severity": v.severity,
            "message": v.message[:80] if len(v.message) > 80 else v.message,
        })

    # Build calibration summary
    calibration = {
        "scs": round(structural_signal.cohesion_score, 4),
        "scs_percent": f"{int(structural_signal.cohesion_score * 100)}%",
        "combined_severity": structural_signal.combined_severity,
        "admissible": structural_signal.admissible,
        "blocking_invariants": {
            "SI-001_dag_acyclicity": si_001_status,
            "SI-010_truth_anchor_integrity": si_010_status,
        },
        "layer_summary": {
            "dag": {
                "status": structural_signal.dag_status,
                "score": structural_signal.layer_scores.get("dag_score", 1.0),
            },
            "topology": {
                "status": structural_signal.topology_status,
                "score": structural_signal.layer_scores.get("topology_score", 1.0),
            },
            "ht": {
                "status": structural_signal.ht_status,
                "score": structural_signal.layer_scores.get("ht_score", 1.0),
            },
        },
        "violation_count": len(structural_signal.violations),
        "key_violations": key_violations,
        "signal_id": structural_signal.signal_id,
        "checked_at": structural_signal.timestamp,
        "mode": "SHADOW",
    }

    return calibration


def attach_structural_governance_to_evidence(
    evidence: Dict[str, Any],
    structural_signal: StructuralGovernanceSignal,
) -> Dict[str, Any]:
    """
    Attach structural governance data to evidence record.

    SHADOW MODE: This attachment is for observation/logging only.
    Attaches under evidence["governance"]["structure"].

    See: docs/system_law/Structural_Cohesion_PhaseX.md

    Args:
        evidence: Evidence dict to augment (modified in-place and returned)
        structural_signal: StructuralGovernanceSignal to attach

    Returns:
        Modified evidence dict with structural governance attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Extract blocking invariants
    blocking_invariants = [
        v.invariant_id
        for v in structural_signal.violations
        if v.invariant_id in ("SI-001", "SI-010")
    ]

    # Build structure attachment
    structure_attachment = {
        "cohesion_score": round(structural_signal.cohesion_score, 4),
        "combined_severity": structural_signal.combined_severity,
        "admissible": structural_signal.admissible,
        "blocking_invariants": blocking_invariants,
        "layer_statuses": {
            "dag": structural_signal.dag_status,
            "topology": structural_signal.topology_status,
            "ht": structural_signal.ht_status,
        },
        "violation_summary": {
            "total": len(structural_signal.violations),
            "conflict_count": len([v for v in structural_signal.violations if v.severity == "CONFLICT"]),
            "tension_count": len([v for v in structural_signal.violations if v.severity == "TENSION"]),
        },
        "signal_id": structural_signal.signal_id,
        "timestamp": structural_signal.timestamp,
        "mode": "SHADOW",
    }

    evidence["governance"]["structure"] = structure_attachment

    return evidence


def build_escalation_advisory(
    snapshot_severity: str,
    structural_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an advisory describing what escalation WOULD have occurred.

    SHADOW MODE: This is purely informational. No escalation is applied.
    Describes what would have happened if structural escalation were enforced.

    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 4.4

    Args:
        snapshot_severity: Original divergence severity (NONE, INFO, WARN, CRITICAL)
        structural_signal: StructuralGovernanceSignal.to_dict() or None

    Returns:
        Advisory dict describing hypothetical escalation
    """
    advisory = {
        "original_severity": snapshot_severity,
        "would_escalate": False,
        "escalated_severity": snapshot_severity,
        "escalation_reason": None,
        "structural_context": None,
        "mode": "SHADOW_ADVISORY",
    }

    if structural_signal is None:
        advisory["structural_context"] = "NO_SIGNAL"
        return advisory

    combined_severity = structural_signal.get("combined_severity", "CONSISTENT")
    cohesion_score = structural_signal.get("cohesion_score", 1.0)
    admissible = structural_signal.get("admissible", True)

    advisory["structural_context"] = {
        "combined_severity": combined_severity,
        "cohesion_score": round(cohesion_score, 4),
        "admissible": admissible,
    }

    # Determine hypothetical escalation
    if combined_severity == "CONFLICT" or not admissible:
        if snapshot_severity != "NONE":
            advisory["would_escalate"] = True
            advisory["escalated_severity"] = "CRITICAL"
            advisory["escalation_reason"] = (
                f"CONFLICT detected (admissible={admissible}): "
                f"{snapshot_severity} → CRITICAL"
            )

    elif combined_severity == "TENSION":
        if snapshot_severity == "INFO":
            advisory["would_escalate"] = True
            advisory["escalated_severity"] = "WARN"
            advisory["escalation_reason"] = "TENSION detected: INFO → WARN"
        elif snapshot_severity == "WARN":
            advisory["would_escalate"] = True
            advisory["escalated_severity"] = "CRITICAL"
            advisory["escalation_reason"] = "TENSION detected: WARN → CRITICAL"

    # Add cohesion degradation note
    if cohesion_score < 0.8:
        if advisory["escalation_reason"]:
            advisory["escalation_reason"] += f" (cohesion degraded: {cohesion_score:.2%})"
        else:
            advisory["escalation_reason"] = f"Cohesion degraded: {cohesion_score:.2%} (no severity escalation)"

    return advisory


__all__ = [
    # Original exports
    "InvariantViolation",
    "InvariantCheckResult",
    "SliceProfile",
    "ProofDag",
    "check_dag_invariants",
    "verify_node_invariant",
    "enforce_dag_structure",
    "evaluate_dag_invariants",
    "load_invariant_rules",
    "summarize_dag_invariants_for_global_health",
    # Structural governance exports
    "StructuralViolation",
    "LayerStatus",
    "StructuralGovernanceSignal",
    "emit_structural_signal",
    "build_structural_cohesion_tile",
    # P4/Evidence integration exports
    "build_structural_calibration_for_p4",
    "attach_structural_governance_to_evidence",
    "build_escalation_advisory",
]
