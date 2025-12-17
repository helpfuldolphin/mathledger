"""Structural Drill Runner — P5 STRUCTURAL_BREAK Event Simulation.

Implements the 5-phase structural drill specified in:
docs/system_law/P5_Structural_Drill_Package.md

SHADOW MODE: All drill operations are observational only. No enforcement actions.
"""

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.dag.invariant_guard import (
    ProofDag,
    StructuralGovernanceSignal,
    build_escalation_advisory,
    build_structural_cohesion_tile,
    emit_structural_signal,
)


# =============================================================================
# Drill Phase Definitions
# =============================================================================

@dataclass
class DrillPhase:
    """Single phase configuration for structural drill."""
    name: str
    cycle_start: int
    cycle_end: int
    dag_config: Dict[str, Any]
    topology_config: Dict[str, Any]
    ht_config: Dict[str, Any]
    expected_pattern: str
    expected_severity: str
    description: str = ""


@dataclass
class DrillCycleResult:
    """Result of a single drill cycle."""
    cycle: int
    phase_name: str
    signal: StructuralGovernanceSignal
    tile: Dict[str, Any]
    advisory: Dict[str, Any]
    pattern: str
    severity: str
    streak: int
    timestamp: str


@dataclass
class DrillArtifact:
    """Complete drill artifact containing all cycle results."""
    drill_id: str
    scenario_id: str
    started_at: str
    completed_at: Optional[str] = None
    phases: List[DrillPhase] = field(default_factory=list)
    cycle_results: List[DrillCycleResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "drill_id": self.drill_id,
            "scenario_id": self.scenario_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "phases": [
                {
                    "name": p.name,
                    "cycle_range": [p.cycle_start, p.cycle_end],
                    "expected_pattern": p.expected_pattern,
                    "expected_severity": p.expected_severity,
                    "description": p.description,
                }
                for p in self.phases
            ],
            "cycle_results": [
                {
                    "cycle": r.cycle,
                    "phase_name": r.phase_name,
                    "signal_id": r.signal.signal_id,
                    "cohesion_score": r.signal.cohesion_score,
                    "combined_severity": r.signal.combined_severity,
                    "admissible": r.signal.admissible,
                    "pattern": r.pattern,
                    "severity": r.severity,
                    "streak": r.streak,
                    "timestamp": r.timestamp,
                }
                for r in self.cycle_results
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# =============================================================================
# STRUCTURAL_BREAK Injection Script
# =============================================================================

def inject_dag_cycle(dag: ProofDag, cycle_nodes: List[str]) -> ProofDag:
    """
    Inject a cycle into the DAG to trigger SI-001 violation.

    SHADOW MODE: This creates a synthetic cycle for drill purposes only.

    Args:
        dag: ProofDag to inject cycle into
        cycle_nodes: List of node IDs forming the cycle (last connects to first)

    Returns:
        Modified ProofDag with cycle injected
    """
    # Ensure nodes exist
    existing_ids = {n.get("id") for n in dag.nodes}
    for node_id in cycle_nodes:
        if node_id not in existing_ids:
            dag.add_node(node_id, {"synthetic": True, "drill_injected": True})

    # Add edges to form cycle
    for i in range(len(cycle_nodes) - 1):
        dag.add_edge(cycle_nodes[i], cycle_nodes[i + 1])
    # Close the cycle
    dag.add_edge(cycle_nodes[-1], cycle_nodes[0])

    return dag


def inject_anchor_failure(ht_state: Dict[str, Any], fail_count: int = 1) -> Dict[str, Any]:
    """
    Inject anchor failure into HT state to trigger SI-010 violation.

    SHADOW MODE: This creates a synthetic failure for drill purposes only.

    Args:
        ht_state: HT state dict to modify
        fail_count: Number of anchors to mark as failed

    Returns:
        Modified HT state with failed anchors
    """
    ht_state = dict(ht_state)  # Don't mutate original
    ht_state["failed_anchors"] = ht_state.get("failed_anchors", 0) + fail_count
    return ht_state


def inject_omega_exit(topology_state: Dict[str, Any], exit_cycles: int) -> Dict[str, Any]:
    """
    Inject omega exit streak into topology state for SI-006 tension.

    SHADOW MODE: This creates synthetic tension for drill purposes only.

    Args:
        topology_state: Topology state dict to modify
        exit_cycles: Number of consecutive cycles outside omega

    Returns:
        Modified topology state with omega exit
    """
    topology_state = dict(topology_state)  # Don't mutate original
    topology_state["in_omega"] = False
    topology_state["omega_exit_streak"] = exit_cycles
    return topology_state


# =============================================================================
# Severity & Streak Calculation
# =============================================================================

@dataclass
class StreakTracker:
    """Tracks STRUCTURAL_BREAK streak state."""
    current_streak: int = 0
    streak_start_cycle: Optional[int] = None
    max_streak: int = 0
    break_events: List[int] = field(default_factory=list)
    last_severity: str = "INFO"

    def update(self, cycle: int, pattern: str, severity: str) -> int:
        """
        Update streak based on current cycle state.

        Args:
            cycle: Current cycle number
            pattern: Detected pattern (NONE, DRIFT, STRUCTURAL_BREAK, etc.)
            severity: Current severity

        Returns:
            Current streak count
        """
        if pattern == "STRUCTURAL_BREAK":
            if self.current_streak == 0:
                # New break event
                self.streak_start_cycle = cycle
                self.break_events.append(cycle)
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            # Break ended
            self.current_streak = 0
            self.streak_start_cycle = None

        self.last_severity = severity
        return self.current_streak

    def is_repeated_break(self) -> bool:
        """Check if current streak constitutes a repeated break (>= 2)."""
        return self.current_streak >= 2


def calculate_pattern(signal: StructuralGovernanceSignal) -> str:
    """
    Calculate divergence pattern from structural signal.

    Args:
        signal: StructuralGovernanceSignal to analyze

    Returns:
        Pattern string: NONE, DRIFT, STRUCTURAL_BREAK
    """
    # SI-001 or SI-010 CONFLICT triggers STRUCTURAL_BREAK
    if not signal.admissible:
        return "STRUCTURAL_BREAK"

    # TENSION without blocking violations indicates DRIFT
    if signal.combined_severity == "TENSION":
        return "DRIFT"

    # CONSISTENT indicates no pattern
    return "NONE"


def calculate_severity(
    signal: StructuralGovernanceSignal,
    base_severity: str = "INFO",
    streak: int = 0,
) -> str:
    """
    Calculate escalated severity from structural signal and streak.

    Args:
        signal: StructuralGovernanceSignal to analyze
        base_severity: Base divergence severity before structural escalation
        streak: Current STRUCTURAL_BREAK streak

    Returns:
        Escalated severity: INFO, WARN, CRITICAL
    """
    # CONFLICT or non-admissible → CRITICAL
    if signal.combined_severity == "CONFLICT" or not signal.admissible:
        return "CRITICAL"

    # TENSION escalates INFO→WARN and WARN→CRITICAL
    if signal.combined_severity == "TENSION":
        if base_severity == "INFO":
            return "WARN"
        elif base_severity == "WARN":
            return "CRITICAL"

    # Streak >= 2 also triggers CRITICAL
    if streak >= 2:
        return "CRITICAL"

    return base_severity


# =============================================================================
# 5-Phase Structural Drill Runner
# =============================================================================

def create_default_phases() -> List[DrillPhase]:
    """
    Create default 5-phase drill configuration.

    Phases:
    1. Baseline (cycles 1-100): Normal operation
    2. Tension Onset (cycles 101-150): SI-006 omega exit
    3. Structural Break (cycle 151): SI-001 cycle injection
    4. Escalation Active (cycles 152-200): Sustained break
    5. Recovery Simulation (cycles 201-210): DAG repaired

    Returns:
        List of DrillPhase configurations
    """
    return [
        DrillPhase(
            name="baseline",
            cycle_start=1,
            cycle_end=100,
            dag_config={"inject_cycle": False, "node_count": 150},
            topology_config={"H": 0.12, "rho": 0.85, "in_omega": True, "omega_exit_streak": 0},
            ht_config={"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0},
            expected_pattern="NONE",
            expected_severity="INFO",
            description="Normal operation baseline",
        ),
        DrillPhase(
            name="tension_onset",
            cycle_start=101,
            cycle_end=150,
            dag_config={"inject_cycle": False, "node_count": 200},
            topology_config={"H": 0.18, "rho": 0.72, "in_omega": False, "omega_exit_streak": 45},
            ht_config={"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0},
            expected_pattern="DRIFT",
            expected_severity="WARN",
            description="Omega exit triggers SI-006 tension",
        ),
        DrillPhase(
            name="structural_break",
            cycle_start=151,
            cycle_end=151,
            dag_config={"inject_cycle": True, "cycle_nodes": ["node_42", "node_87", "node_42"], "node_count": 201},
            topology_config={"H": 0.22, "rho": 0.65, "in_omega": False, "omega_exit_streak": 50},
            ht_config={"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0},
            expected_pattern="STRUCTURAL_BREAK",
            expected_severity="CRITICAL",
            description="SI-001 cycle injection triggers STRUCTURAL_BREAK",
        ),
        DrillPhase(
            name="escalation_active",
            cycle_start=152,
            cycle_end=200,
            dag_config={"inject_cycle": True, "cycle_nodes": ["node_42", "node_87", "node_42"], "node_count": 201},
            topology_config={"H": 0.25, "rho": 0.60, "in_omega": False, "omega_exit_streak": 100},
            ht_config={"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0},
            expected_pattern="STRUCTURAL_BREAK",
            expected_severity="CRITICAL",
            description="Sustained STRUCTURAL_BREAK with escalating streak",
        ),
        DrillPhase(
            name="recovery",
            cycle_start=201,
            cycle_end=210,
            dag_config={"inject_cycle": False, "node_count": 201},
            topology_config={"H": 0.15, "rho": 0.80, "in_omega": True, "omega_exit_streak": 0},
            ht_config={"total_anchors": 10, "verified_anchors": 10, "failed_anchors": 0},
            expected_pattern="NONE",
            expected_severity="INFO",
            description="DAG repaired, system recovering",
        ),
    ]


def build_dag_for_phase(phase: DrillPhase, cycle: int) -> ProofDag:
    """
    Build synthetic DAG state for a drill phase.

    Args:
        phase: Current DrillPhase configuration
        cycle: Current cycle number

    Returns:
        ProofDag configured for the phase
    """
    dag = ProofDag()
    node_count = phase.dag_config.get("node_count", 100)

    # Add synthetic nodes
    for i in range(node_count):
        dag.add_node(f"node_{i}", {"cycle_created": cycle, "synthetic": True})

    # Add some edges (not forming cycles)
    for i in range(1, node_count):
        if i % 3 == 0:
            dag.add_edge(f"node_{i-1}", f"node_{i}")

    # Inject cycle if configured
    if phase.dag_config.get("inject_cycle", False):
        cycle_nodes = phase.dag_config.get("cycle_nodes", ["node_42", "node_87", "node_42"])
        dag = inject_dag_cycle(dag, cycle_nodes)

    return dag


def run_structural_drill(
    scenario_id: str = "DRILL-SB-001",
    phases: Optional[List[DrillPhase]] = None,
    sample_rate: int = 10,
    output_dir: Optional[Path] = None,
) -> DrillArtifact:
    """
    Execute the 5-phase structural drill.

    SHADOW MODE: All drill operations are observational only.

    Args:
        scenario_id: Identifier for this drill scenario
        phases: List of DrillPhase configurations (uses default if None)
        sample_rate: Record every Nth cycle (default 10 for efficiency)
        output_dir: Optional directory to write artifacts

    Returns:
        DrillArtifact containing all results
    """
    phases = phases or create_default_phases()

    drill_id = f"drill_{secrets.token_hex(8)}"
    artifact = DrillArtifact(
        drill_id=drill_id,
        scenario_id=scenario_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        phases=phases,
        metadata={
            "sample_rate": sample_rate,
            "shadow_mode": True,
            "version": "1.0.0",
        },
    )

    streak_tracker = StreakTracker()
    cohesion_history: List[float] = []

    for phase in phases:
        for cycle in range(phase.cycle_start, phase.cycle_end + 1):
            # Build state for this cycle
            dag = build_dag_for_phase(phase, cycle)
            topology_state = dict(phase.topology_config)
            ht_state = dict(phase.ht_config)

            # Emit structural signal
            signal = emit_structural_signal(
                dag=dag,
                topology_state=topology_state,
                ht_state=ht_state,
                run_id=drill_id,
                cycle=cycle,
                triggered_by="DRILL",
            )

            # Calculate pattern and severity
            pattern = calculate_pattern(signal)
            severity = calculate_severity(signal, base_severity="INFO", streak=streak_tracker.current_streak)

            # Update streak
            streak = streak_tracker.update(cycle, pattern, severity)

            # Track cohesion history
            cohesion_history.append(signal.cohesion_score)

            # Build tile and advisory
            tile = build_structural_cohesion_tile(signal, cohesion_history[-20:])
            advisory = build_escalation_advisory("INFO", signal.to_dict())

            # Record result (sample or key cycles)
            is_key_cycle = (
                cycle == phase.cycle_start
                or cycle == phase.cycle_end
                or pattern == "STRUCTURAL_BREAK"
                or cycle % sample_rate == 0
            )

            if is_key_cycle:
                result = DrillCycleResult(
                    cycle=cycle,
                    phase_name=phase.name,
                    signal=signal,
                    tile=tile,
                    advisory=advisory,
                    pattern=pattern,
                    severity=severity,
                    streak=streak,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                artifact.cycle_results.append(result)

    # Build summary
    artifact.completed_at = datetime.now(timezone.utc).isoformat()
    artifact.summary = build_drill_summary(artifact, streak_tracker)

    # Write artifacts if output_dir provided
    if output_dir:
        write_drill_artifacts(artifact, output_dir)

    return artifact


def build_drill_summary(artifact: DrillArtifact, streak_tracker: StreakTracker) -> Dict[str, Any]:
    """
    Build summary statistics for completed drill.

    Args:
        artifact: DrillArtifact with cycle results
        streak_tracker: StreakTracker with accumulated state

    Returns:
        Summary dict with key metrics
    """
    cycle_results = artifact.cycle_results

    # Count patterns
    pattern_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    for r in cycle_results:
        pattern_counts[r.pattern] = pattern_counts.get(r.pattern, 0) + 1
        severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

    # Phase summaries
    phase_summaries = []
    for phase in artifact.phases:
        phase_results = [r for r in cycle_results if r.phase_name == phase.name]
        if phase_results:
            phase_summaries.append({
                "name": phase.name,
                "cycles_sampled": len(phase_results),
                "patterns_detected": list({r.pattern for r in phase_results}),
                "expected_pattern": phase.expected_pattern,
                "pattern_match": phase.expected_pattern in {r.pattern for r in phase_results},
            })

    return {
        "total_cycles_sampled": len(cycle_results),
        "pattern_counts": pattern_counts,
        "severity_counts": severity_counts,
        "max_streak": streak_tracker.max_streak,
        "break_events": streak_tracker.break_events,
        "phase_summaries": phase_summaries,
        "drill_success": all(p.get("pattern_match", False) for p in phase_summaries),
    }


def write_drill_artifacts(artifact: DrillArtifact, output_dir: Path) -> List[Path]:
    """
    Write drill artifacts to files.

    Args:
        artifact: DrillArtifact to write
        output_dir: Directory to write files

    Returns:
        List of written file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # Main artifact JSON (drill-id specific)
    artifact_path = output_dir / f"{artifact.drill_id}_artifact.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2)
    written.append(artifact_path)

    # Canonical artifact JSON (for evidence pack detection)
    canonical_path = output_dir / "structural_drill_artifact.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2)
    written.append(canonical_path)

    # Timeline data for plotting
    timeline_path = output_dir / f"{artifact.drill_id}_timeline.json"
    timeline_data = {
        "drill_id": artifact.drill_id,
        "cycles": [r.cycle for r in artifact.cycle_results],
        "cohesion_scores": [r.signal.cohesion_score for r in artifact.cycle_results],
        "patterns": [r.pattern for r in artifact.cycle_results],
        "severities": [r.severity for r in artifact.cycle_results],
        "streaks": [r.streak for r in artifact.cycle_results],
        "phases": [r.phase_name for r in artifact.cycle_results],
    }
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)
    written.append(timeline_path)

    # Summary
    summary_path = output_dir / f"{artifact.drill_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(artifact.summary, f, indent=2)
    written.append(summary_path)

    return written


# =============================================================================
# Drift Timeline Plot Stub
# =============================================================================

def generate_drift_timeline_plot_data(artifact: DrillArtifact) -> Dict[str, Any]:
    """
    Generate data structure for drift timeline plot.

    STUB: Returns plot data specification. Actual rendering requires matplotlib/plotly.

    Args:
        artifact: DrillArtifact with cycle results

    Returns:
        Plot data specification dict
    """
    cycle_results = artifact.cycle_results

    # Extract time series
    cycles = [r.cycle for r in cycle_results]
    cohesion_scores = [r.signal.cohesion_score for r in cycle_results]
    severities = [r.severity for r in cycle_results]
    patterns = [r.pattern for r in cycle_results]
    streaks = [r.streak for r in cycle_results]

    # Phase boundaries
    phase_boundaries = []
    for phase in artifact.phases:
        phase_boundaries.append({
            "name": phase.name,
            "start": phase.cycle_start,
            "end": phase.cycle_end,
            "color": _phase_color(phase.name),
        })

    # Event markers
    event_markers = []
    for r in cycle_results:
        if r.pattern == "STRUCTURAL_BREAK" and r.streak == 1:
            event_markers.append({
                "cycle": r.cycle,
                "type": "BREAK_START",
                "label": f"SI-001 @ cycle {r.cycle}",
            })

    return {
        "plot_id": f"drift_timeline_{artifact.drill_id}",
        "title": "Structural Drift Timeline",
        "subtitle": f"Drill: {artifact.scenario_id}",
        "x_axis": {
            "label": "Cycle",
            "values": cycles,
        },
        "y_axes": [
            {
                "id": "cohesion",
                "label": "Cohesion Score",
                "values": cohesion_scores,
                "color": "#2196F3",
                "line_style": "solid",
            },
            {
                "id": "streak",
                "label": "Break Streak",
                "values": streaks,
                "color": "#F44336",
                "line_style": "dashed",
                "secondary": True,
            },
        ],
        "categorical_tracks": [
            {
                "id": "severity",
                "label": "Severity",
                "values": severities,
                "color_map": {
                    "INFO": "#4CAF50",
                    "WARN": "#FF9800",
                    "CRITICAL": "#F44336",
                },
            },
            {
                "id": "pattern",
                "label": "Pattern",
                "values": patterns,
                "color_map": {
                    "NONE": "#9E9E9E",
                    "DRIFT": "#FF9800",
                    "STRUCTURAL_BREAK": "#F44336",
                },
            },
        ],
        "phase_regions": phase_boundaries,
        "event_markers": event_markers,
        "annotations": [
            {
                "cycle": 151,
                "y": 0.0,
                "text": "STRUCTURAL_BREAK",
                "arrow": True,
            },
        ],
        "shadow_mode_banner": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _phase_color(phase_name: str) -> str:
    """Get color for phase region."""
    colors = {
        "baseline": "#E3F2FD",
        "tension_onset": "#FFF3E0",
        "structural_break": "#FFEBEE",
        "escalation_active": "#FCE4EC",
        "recovery": "#E8F5E9",
    }
    return colors.get(phase_name, "#F5F5F5")


# =============================================================================
# Artifact Schema Definition
# =============================================================================

DRILL_ARTIFACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "structural_drill_artifact.schema.json",
    "title": "Structural Drill Artifact",
    "description": "Complete artifact from P5 structural drill execution",
    "type": "object",
    "required": ["drill_id", "scenario_id", "started_at", "phases", "cycle_results", "summary"],
    "properties": {
        "drill_id": {
            "type": "string",
            "pattern": "^drill_[a-f0-9]{16}$",
        },
        "scenario_id": {
            "type": "string",
        },
        "started_at": {
            "type": "string",
            "format": "date-time",
        },
        "completed_at": {
            "type": ["string", "null"],
            "format": "date-time",
        },
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "cycle_range", "expected_pattern", "expected_severity"],
                "properties": {
                    "name": {"type": "string"},
                    "cycle_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "expected_pattern": {"type": "string"},
                    "expected_severity": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "cycle_results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cycle", "phase_name", "signal_id", "cohesion_score", "pattern", "severity", "streak"],
                "properties": {
                    "cycle": {"type": "integer"},
                    "phase_name": {"type": "string"},
                    "signal_id": {"type": "string"},
                    "cohesion_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "combined_severity": {"type": "string", "enum": ["CONSISTENT", "TENSION", "CONFLICT"]},
                    "admissible": {"type": "boolean"},
                    "pattern": {"type": "string", "enum": ["NONE", "DRIFT", "STRUCTURAL_BREAK"]},
                    "severity": {"type": "string", "enum": ["INFO", "WARN", "CRITICAL"]},
                    "streak": {"type": "integer", "minimum": 0},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
        },
        "summary": {
            "type": "object",
            "properties": {
                "total_cycles_sampled": {"type": "integer"},
                "pattern_counts": {"type": "object"},
                "severity_counts": {"type": "object"},
                "max_streak": {"type": "integer"},
                "break_events": {"type": "array", "items": {"type": "integer"}},
                "drill_success": {"type": "boolean"},
            },
        },
        "metadata": {
            "type": "object",
            "properties": {
                "sample_rate": {"type": "integer"},
                "shadow_mode": {"type": "boolean"},
                "version": {"type": "string"},
            },
        },
    },
}


__all__ = [
    # Data structures
    "DrillPhase",
    "DrillCycleResult",
    "DrillArtifact",
    "StreakTracker",
    # Injection functions
    "inject_dag_cycle",
    "inject_anchor_failure",
    "inject_omega_exit",
    # Calculation functions
    "calculate_pattern",
    "calculate_severity",
    # Drill runner
    "create_default_phases",
    "build_dag_for_phase",
    "run_structural_drill",
    "build_drill_summary",
    "write_drill_artifacts",
    # Plot stub
    "generate_drift_timeline_plot_data",
    # Schema
    "DRILL_ARTIFACT_SCHEMA",
]
