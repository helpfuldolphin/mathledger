"""
U2 Safety SLO Engine

Provides:
- Safety envelope tracking across multiple experiments
- Scenario-aware safety matrix (slice × mode)
- Policy-level SLO enforcement with typed contracts
- Deterministic timeline aggregation

INVARIANTS:
- Timeline building is deterministic (same envelopes → same timeline)
- All types are fully annotated for mypy verification
- SLO evaluation uses stable thresholds
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, TypedDict

# Type definitions
SafetyStatus = Literal["OK", "WARN", "BLOCK"]


class SafetyEnvelope(TypedDict):
    """
    Safety envelope for a single U2 experiment run.
    
    INVARIANTS:
    - schema_version must be "1.0" for current format
    - timestamp is ISO-8601 formatted
    - mode is either "baseline" or "rfl"
    """
    schema_version: str
    config: Dict[str, object]
    perf_ok: bool
    safety_status: SafetyStatus
    lint_issues: List[str]
    warnings: List[str]
    run_id: str
    slice_name: str
    mode: Literal["baseline", "rfl"]
    timestamp: str  # ISO-8601


@dataclass(frozen=True)
class SafetySLOPoint:
    """
    A single point in the safety SLO timeline.
    
    Represents one experiment run with computed metrics.
    """
    run_id: str
    slice_name: str
    mode: Literal["baseline", "rfl"]
    safety_status: SafetyStatus
    perf_ok: bool
    lint_issue_count: int
    warnings_count: int
    timestamp: datetime


@dataclass(frozen=True)
class SafetySLOTimeline:
    """
    Aggregated timeline of safety SLO points.
    
    INVARIANTS:
    - points are sorted by (timestamp, run_id) for determinism
    - rates are computed as floats in [0.0, 1.0]
    """
    schema_version: str
    points: List[SafetySLOPoint]
    status_counts: Dict[SafetyStatus, int]
    perf_ok_rate: float
    lint_issue_rate: float


@dataclass(frozen=True)
class ScenarioSafetyCell:
    """
    Safety metrics for a single scenario (slice_name × mode).
    
    A "scenario" is a unique combination of slice and execution mode.
    """
    slice_name: str
    mode: Literal["baseline", "rfl"]
    runs: int
    blocked_runs: int
    warn_runs: int
    ok_runs: int
    perf_failure_runs: int
    worst_status: SafetyStatus


@dataclass(frozen=True)
class ScenarioSafetyMatrix:
    """
    Safety matrix across all scenarios.
    
    INVARIANTS:
    - cells are sorted by (slice_name, mode) for determinism
    """
    schema_version: str
    cells: List[ScenarioSafetyCell]
    total_runs: int
    total_slices: int


@dataclass(frozen=True)
class SafetySLOEvaluation:
    """
    Overall safety SLO evaluation with pass/warn/block decision.
    
    Uses configurable thresholds to determine overall status.
    """
    schema_version: str
    matrix: ScenarioSafetyMatrix
    overall_status: SafetyStatus
    failing_scenarios: List[str]  # Format: "slice_name:mode"
    reasons: List[str]


# Configurable SLO thresholds
MAX_BLOCK_RATE = 0.05  # ≤ 5% BLOCK across all runs
MAX_WARN_RATE = 0.20  # ≤ 20% WARN across all runs
MAX_PERF_FAILURE_RATE = 0.10  # ≤ 10% performance failures


def build_safety_slo_timeline(envelopes: List[SafetyEnvelope]) -> SafetySLOTimeline:
    """
    Deterministically aggregate per-run envelopes into a timeline.
    
    Args:
        envelopes: List of safety envelopes from experiment runs
        
    Returns:
        SafetySLOTimeline with sorted points and computed metrics
        
    INVARIANTS:
    - Envelopes are sorted by (timestamp, run_id) for determinism
    - Empty envelopes list produces empty timeline with zero rates
    """
    if not envelopes:
        return SafetySLOTimeline(
            schema_version="1.0",
            points=[],
            status_counts={"OK": 0, "WARN": 0, "BLOCK": 0},
            perf_ok_rate=0.0,
            lint_issue_rate=0.0,
        )
    
    # Convert envelopes to SLO points
    points: List[SafetySLOPoint] = []
    for envelope in envelopes:
        point = SafetySLOPoint(
            run_id=envelope["run_id"],
            slice_name=envelope["slice_name"],
            mode=envelope["mode"],
            safety_status=envelope["safety_status"],
            perf_ok=envelope["perf_ok"],
            lint_issue_count=len(envelope["lint_issues"]),
            warnings_count=len(envelope["warnings"]),
            timestamp=datetime.fromisoformat(envelope["timestamp"]),
        )
        points.append(point)
    
    # Sort by timestamp, then run_id for determinism
    points.sort(key=lambda p: (p.timestamp, p.run_id))
    
    # Compute status counts
    status_counts: Dict[SafetyStatus, int] = {"OK": 0, "WARN": 0, "BLOCK": 0}
    for point in points:
        status_counts[point.safety_status] += 1
    
    # Compute rates
    total = len(points)
    perf_ok_count = sum(1 for p in points if p.perf_ok)
    lint_issue_count = sum(1 for p in points if p.lint_issue_count > 0)
    
    perf_ok_rate = perf_ok_count / total if total > 0 else 0.0
    lint_issue_rate = lint_issue_count / total if total > 0 else 0.0
    
    return SafetySLOTimeline(
        schema_version="1.0",
        points=points,
        status_counts=status_counts,
        perf_ok_rate=perf_ok_rate,
        lint_issue_rate=lint_issue_rate,
    )


def build_scenario_safety_matrix(timeline: SafetySLOTimeline) -> ScenarioSafetyMatrix:
    """
    Group SafetySLOPoints by (slice_name, mode), compute per-scenario counts and worst_status.
    
    Args:
        timeline: SafetySLOTimeline with sorted points
        
    Returns:
        ScenarioSafetyMatrix with cells sorted by (slice_name, mode)
        
    INVARIANTS:
    - Cells are sorted lexicographically by (slice_name, mode)
    - worst_status is the most severe status seen in the scenario
    """
    # Group points by scenario
    scenarios: Dict[tuple[str, str], List[SafetySLOPoint]] = {}
    for point in timeline.points:
        key = (point.slice_name, point.mode)
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(point)
    
    # Build cells
    cells: List[ScenarioSafetyCell] = []
    for (slice_name, mode), scenario_points in scenarios.items():
        # Count statuses
        ok_runs = sum(1 for p in scenario_points if p.safety_status == "OK")
        warn_runs = sum(1 for p in scenario_points if p.safety_status == "WARN")
        blocked_runs = sum(1 for p in scenario_points if p.safety_status == "BLOCK")
        perf_failure_runs = sum(1 for p in scenario_points if not p.perf_ok)
        
        # Determine worst status (BLOCK > WARN > OK)
        if blocked_runs > 0:
            worst_status: SafetyStatus = "BLOCK"
        elif warn_runs > 0:
            worst_status = "WARN"
        else:
            worst_status = "OK"
        
        cell = ScenarioSafetyCell(
            slice_name=slice_name,
            mode=mode,  # type: ignore
            runs=len(scenario_points),
            blocked_runs=blocked_runs,
            warn_runs=warn_runs,
            ok_runs=ok_runs,
            perf_failure_runs=perf_failure_runs,
            worst_status=worst_status,
        )
        cells.append(cell)
    
    # Sort cells by (slice_name, mode) for determinism
    cells.sort(key=lambda c: (c.slice_name, c.mode))
    
    # Count unique slices
    unique_slices = len(set(c.slice_name for c in cells))
    
    return ScenarioSafetyMatrix(
        schema_version="1.0",
        cells=cells,
        total_runs=len(timeline.points),
        total_slices=unique_slices,
    )


def evaluate_safety_slo(
    matrix: ScenarioSafetyMatrix,
    max_block_rate: float = MAX_BLOCK_RATE,
    max_warn_rate: float = MAX_WARN_RATE,
    max_perf_failure_rate: float = MAX_PERF_FAILURE_RATE,
) -> SafetySLOEvaluation:
    """
    Evaluate safety SLO and determine overall status.
    
    Args:
        matrix: ScenarioSafetyMatrix with per-scenario metrics
        max_block_rate: Maximum acceptable BLOCK rate (default 0.05)
        max_warn_rate: Maximum acceptable WARN rate (default 0.20)
        max_perf_failure_rate: Maximum acceptable performance failure rate (default 0.10)
        
    Returns:
        SafetySLOEvaluation with overall status and reasons
        
    Rules:
    - overall_status == "BLOCK" if:
      - any scenario has blocked_runs / runs > max_block_rate, or
      - global BLOCK rate > max_block_rate
    - overall_status == "WARN" if:
      - global WARN rate > max_warn_rate, or
      - perf_failure rate > max_perf_failure_rate
    - Else overall_status == "OK"
    """
    if matrix.total_runs == 0:
        # Empty matrix is OK
        return SafetySLOEvaluation(
            schema_version="1.0",
            matrix=matrix,
            overall_status="OK",
            failing_scenarios=[],
            reasons=["No runs to evaluate"],
        )
    
    failing_scenarios: List[str] = []
    reasons: List[str] = []
    overall_status: SafetyStatus = "OK"
    
    # Check per-scenario block rates
    for cell in matrix.cells:
        scenario_block_rate = cell.blocked_runs / cell.runs if cell.runs > 0 else 0.0
        if scenario_block_rate > max_block_rate:
            scenario_key = f"{cell.slice_name}:{cell.mode}"
            failing_scenarios.append(scenario_key)
            reasons.append(
                f"scenario {scenario_key} block_rate={scenario_block_rate:.2f} exceeds {max_block_rate:.2f}"
            )
            overall_status = "BLOCK"
    
    # Check global rates
    total_blocked = sum(c.blocked_runs for c in matrix.cells)
    total_warned = sum(c.warn_runs for c in matrix.cells)
    total_perf_failures = sum(c.perf_failure_runs for c in matrix.cells)
    
    global_block_rate = total_blocked / matrix.total_runs
    global_warn_rate = total_warned / matrix.total_runs
    global_perf_failure_rate = total_perf_failures / matrix.total_runs
    
    # Check global block rate
    if global_block_rate > max_block_rate:
        reasons.append(
            f"global block_rate={global_block_rate:.2f} exceeds {max_block_rate:.2f}"
        )
        overall_status = "BLOCK"
    
    # Check warn and perf rates (only if not already blocked)
    if overall_status != "BLOCK":
        if global_warn_rate > max_warn_rate:
            reasons.append(
                f"global warn_rate={global_warn_rate:.2f} exceeds {max_warn_rate:.2f}"
            )
            overall_status = "WARN"
        
        if global_perf_failure_rate > max_perf_failure_rate:
            reasons.append(
                f"global perf_failure_rate={global_perf_failure_rate:.2f} exceeds {max_perf_failure_rate:.2f}"
            )
            if overall_status != "WARN":
                overall_status = "WARN"
    
    # If no issues found, add success reason
    if not reasons:
        reasons.append("All safety SLO thresholds met")
    
    return SafetySLOEvaluation(
        schema_version="1.0",
        matrix=matrix,
        overall_status=overall_status,
        failing_scenarios=failing_scenarios,
        reasons=reasons,
    )
