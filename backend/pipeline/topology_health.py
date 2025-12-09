"""
Topology Health Evaluator and Degradation Policy Engine.

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.

This module implements runtime and CI enforcement of the topology health
and degradation policy defined in docs/U2_PIPELINE_TOPOLOGY.md Sections 10-12.

Key components:
- TopologyHealthEvaluator: Evaluates pipeline health from node statuses
- DegradationPolicyEngine: Applies degradation rules and constraints

References:
- Section 10: Degraded Pipeline Modes (FULL_PIPELINE, DEGRADED_ANALYSIS, EVIDENCE_ONLY)
- Section 11: Topology Health Matrix (health signals, failure patterns)
- Section 12: CI Degradation Policy (hard-fail/soft-fail semantics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# =============================================================================
# Schema Version for Snapshot Format
# =============================================================================

HEALTH_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


# =============================================================================
# Enums
# =============================================================================


class PipelineMode(Enum):
    """
    Pipeline degradation modes as defined in Section 10.1.

    FULL_PIPELINE: All nodes operational, full Δp computation permitted
    DEGRADED_ANALYSIS: 2-3 slice pairs complete, restricted Δp
    EVIDENCE_ONLY: Critical failure, no Δp permitted
    """
    FULL_PIPELINE = "FULL_PIPELINE"
    DEGRADED_ANALYSIS = "DEGRADED_ANALYSIS"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"


class GovernanceLabel(Enum):
    """Governance labels for Evidence Pack status."""
    OK = "OK"
    WARN = "WARN"
    DO_NOT_USE = "DO_NOT_USE"


class NodeStatus(Enum):
    """Status of a pipeline node."""
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"


class NodeType(Enum):
    """
    Node types as defined in Section 11.2.

    Each type has specific health metrics and failure patterns.
    """
    GATEKEEPER = "GATEKEEPER"      # N01, N02
    LOADER = "LOADER"              # N03
    VALIDATOR = "VALIDATOR"        # N04, N05
    RUNNER = "RUNNER"              # N10-N13
    EVALUATOR = "EVALUATOR"        # N20-N23
    SYNC_BARRIER = "SYNC_BARRIER"  # N30
    ANALYZER = "ANALYZER"          # N40, N41
    SUMMARIZER = "SUMMARIZER"      # N50
    AUDITOR = "AUDITOR"            # N60
    PACKAGER = "PACKAGER"          # N70
    ATTESTER = "ATTESTER"          # N80
    SEALER = "SEALER"              # N90


class FailBehavior(Enum):
    """Failure behavior for CI stages."""
    HARD_FAIL = "HARD_FAIL"  # Immediately triggers EVIDENCE_ONLY
    SOFT_FAIL = "SOFT_FAIL"  # May allow DEGRADED_ANALYSIS


class PatternSeverity(Enum):
    """
    Severity levels for failure patterns.

    Used in Pattern Library to map pattern codes to governance implications.
    """
    CRITICAL = "CRITICAL"  # Strongly suggests EVIDENCE_ONLY
    HIGH = "HIGH"          # May trigger EVIDENCE_ONLY or DEGRADED
    MEDIUM = "MEDIUM"      # Typically allows DEGRADED_ANALYSIS
    LOW = "LOW"            # Informational, minimal impact on mode


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class HealthSignals:
    """
    Health signals for a node as defined in Section 11.1.

    All signals are immutable to ensure deterministic evaluation.
    """
    heartbeat: bool = True
    progress: float = 1.0
    memory_ok: bool = True
    disk_ok: bool = True
    latency_ok: bool = True
    integrity_ok: bool = True
    dependencies_ok: bool = True

    def __post_init__(self):
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"progress must be in [0.0, 1.0], got {self.progress}")

    def is_healthy(self) -> bool:
        """Check if all signals indicate healthy status."""
        return all([
            self.heartbeat,
            self.progress >= 1.0,
            self.memory_ok,
            self.disk_ok,
            self.latency_ok,
            self.integrity_ok,
            self.dependencies_ok,
        ])


@dataclass(frozen=True)
class FailurePattern:
    """
    Observable failure pattern as defined in Section 11.2.

    Each node type has specific failure patterns (GK-001, VD-001, etc.).
    """
    pattern_id: str
    symptoms: str
    root_cause: str
    recovery_action: str
    node_type: NodeType


@dataclass(frozen=True)
class PatternLibraryEntry:
    """
    Pattern library entry mapping pattern codes to severity and mode hints.

    This provides governance guidance based on detected patterns.
    """
    pattern_id: str
    severity: PatternSeverity
    suggested_mode: PipelineMode
    fail_behavior: FailBehavior
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "severity": self.severity.value,
            "suggested_mode": self.suggested_mode.value,
            "fail_behavior": self.fail_behavior.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class NodeHealth:
    """Health status of a single pipeline node."""
    node_id: str
    node_type: NodeType
    status: NodeStatus
    signals: HealthSignals
    failure_patterns: Tuple[FailurePattern, ...] = field(default_factory=tuple)
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

    def is_critical_node(self) -> bool:
        """Check if this is a critical (hard-fail) node."""
        return self.node_type in {
            NodeType.GATEKEEPER,
            NodeType.LOADER,
            NodeType.VALIDATOR,
        }


@dataclass(frozen=True)
class SliceResult:
    """
    Result of a slice execution (baseline + RFL pair).

    Critical Constraint (Section 10.1):
    A slice pair must BOTH complete for that slice's Δp to be computed.
    """
    slice_name: str  # goal, sparse, tree, dep
    baseline_completed: bool
    rfl_completed: bool
    runner_node_id: str  # N10, N11, N12, N13
    evaluator_node_id: str  # N20, N21, N22, N23
    baseline_exit_code: Optional[int] = None
    rfl_exit_code: Optional[int] = None

    def is_complete(self) -> bool:
        """Check if both baseline and RFL completed successfully."""
        return self.baseline_completed and self.rfl_completed

    def allows_delta_p(self) -> bool:
        """
        Check if Δp computation is allowed for this slice.

        Cardinal Rule (Section 12.1):
        If a baseline run fails, the corresponding RFL run's Δp MUST be excluded.
        If an RFL run fails, the corresponding baseline run's Δp MUST be excluded.
        """
        return self.is_complete()


@dataclass(frozen=True)
class ValidationStatus:
    """Status of all validation stages (N01-N05)."""
    gate_check_passed: bool
    prereg_verify_passed: bool
    curriculum_load_passed: bool
    dry_run_passed: bool
    manifest_init_passed: bool

    @property
    def all_passed(self) -> bool:
        """Check if all validation stages passed."""
        return all([
            self.gate_check_passed,
            self.prereg_verify_passed,
            self.curriculum_load_passed,
            self.dry_run_passed,
            self.manifest_init_passed,
        ])


@dataclass(frozen=True)
class IntegrityCheck:
    """Result of an integrity check."""
    check_name: str
    passed: bool
    details: Optional[str] = None


@dataclass(frozen=True)
class PipelineHealth:
    """
    Overall pipeline health assessment.

    Produced by TopologyHealthEvaluator.evaluate_pipeline_health().
    """
    mode: PipelineMode
    governance_label: GovernanceLabel
    failed_nodes: Tuple[str, ...]
    successful_slices: Tuple[str, ...]
    failed_slices: Tuple[str, ...]
    health_signals: Dict[str, HealthSignals]
    detected_patterns: Tuple[FailurePattern, ...]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode.value,
            "governance_label": self.governance_label.value,
            "failed_nodes": list(self.failed_nodes),
            "successful_slices": list(self.successful_slices),
            "failed_slices": list(self.failed_slices),
            "detected_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "symptoms": p.symptoms,
                    "root_cause": p.root_cause,
                    "node_type": p.node_type.value,
                }
                for p in self.detected_patterns
            ],
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class DegradationDecision:
    """
    Decision from the Degradation Policy Engine.

    Encodes the outcome of applying constraints C1-C5.
    """
    mode: PipelineMode
    governance_label: GovernanceLabel
    allow_delta_p: bool
    allowed_delta_p_slices: FrozenSet[str]  # Slices where Δp is permitted
    allow_evidence_pack: bool
    evidence_pack_status: str  # COMPLETE, PARTIAL, FORENSIC
    halt_ci: bool
    halt_reason: Optional[str]
    quarantine: bool
    violations: Tuple[str, ...]  # List of constraint violations
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode.value,
            "governance_label": self.governance_label.value,
            "allow_delta_p": self.allow_delta_p,
            "allowed_delta_p_slices": sorted(self.allowed_delta_p_slices),
            "allow_evidence_pack": self.allow_evidence_pack,
            "evidence_pack_status": self.evidence_pack_status,
            "halt_ci": self.halt_ci,
            "halt_reason": self.halt_reason,
            "quarantine": self.quarantine,
            "violations": list(self.violations),
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class DecisionExplanation:
    """
    Human-friendly, machine-stable explanation of pipeline mode decision.

    Provides transparency into why a particular mode was chosen.
    """
    reason_codes: Tuple[str, ...]  # Pattern codes and rule IDs (C1-C5)
    short_summary: str  # Single-line explanation
    detailed_reasons: Tuple[str, ...]  # Detailed breakdown
    patterns_detected: Tuple[str, ...]  # Pattern IDs that fired
    constraints_violated: Tuple[str, ...]  # Constraint IDs violated
    mode: PipelineMode
    governance_label: GovernanceLabel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "reason_codes": list(self.reason_codes),
            "short_summary": self.short_summary,
            "detailed_reasons": list(self.detailed_reasons),
            "patterns_detected": list(self.patterns_detected),
            "constraints_violated": list(self.constraints_violated),
            "mode": self.mode.value,
            "governance_label": self.governance_label.value,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def format_for_ci(self) -> str:
        """Format explanation for CI step summary."""
        lines = [
            f"## Pipeline Mode: {self.mode.value}",
            f"**Governance Label:** {self.governance_label.value}",
            "",
            f"**Summary:** {self.short_summary}",
            "",
        ]

        if self.patterns_detected:
            lines.append("### Patterns Detected")
            for p in self.patterns_detected:
                lines.append(f"- `{p}`")
            lines.append("")

        if self.constraints_violated:
            lines.append("### Constraints Violated")
            for c in self.constraints_violated:
                lines.append(f"- `{c}`")
            lines.append("")

        if self.detailed_reasons:
            lines.append("### Detailed Reasons")
            for r in self.detailed_reasons:
                lines.append(f"- {r}")

        return "\n".join(lines)


@dataclass(frozen=True)
class HealthSnapshot:
    """
    Longitudinal health snapshot for trend analysis.

    Captures pipeline health at a point in time for comparison across runs.
    """
    schema_version: str
    mode: PipelineMode
    governance_label: GovernanceLabel
    hard_fail_count: int
    soft_fail_count: int
    slice_baseline_fail_count: int
    slice_rfl_fail_count: int
    successful_slice_count: int
    failed_slice_count: int
    patterns_detected: Tuple[str, ...]
    pattern_severities: Dict[str, str]  # pattern_id -> severity
    timestamp: str
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "governance_label": self.governance_label.value,
            "hard_fail_count": self.hard_fail_count,
            "soft_fail_count": self.soft_fail_count,
            "slice_baseline_fail_count": self.slice_baseline_fail_count,
            "slice_rfl_fail_count": self.slice_rfl_fail_count,
            "successful_slice_count": self.successful_slice_count,
            "failed_slice_count": self.failed_slice_count,
            "patterns_detected": list(self.patterns_detected),
            "pattern_severities": self.pattern_severities,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthSnapshot":
        """Create from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            mode=PipelineMode(data["mode"]),
            governance_label=GovernanceLabel(data["governance_label"]),
            hard_fail_count=data["hard_fail_count"],
            soft_fail_count=data["soft_fail_count"],
            slice_baseline_fail_count=data["slice_baseline_fail_count"],
            slice_rfl_fail_count=data["slice_rfl_fail_count"],
            successful_slice_count=data["successful_slice_count"],
            failed_slice_count=data["failed_slice_count"],
            patterns_detected=tuple(data["patterns_detected"]),
            pattern_severities=data["pattern_severities"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id"),
        )


@dataclass(frozen=True)
class SnapshotComparison:
    """
    Comparison between two health snapshots.

    Tracks mode transitions and failure count deltas.
    """
    mode_delta: Optional[str]  # e.g., "FULL_PIPELINE -> DEGRADED_ANALYSIS"
    label_delta: Optional[str]  # e.g., "OK -> WARN"
    hard_fail_delta: int
    soft_fail_delta: int
    slice_fail_delta: int
    any_new_critical_pattern: bool
    new_patterns: Tuple[str, ...]
    removed_patterns: Tuple[str, ...]
    degraded: bool  # True if new snapshot is worse
    improved: bool  # True if new snapshot is better
    unchanged: bool  # True if snapshots are equivalent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode_delta": self.mode_delta,
            "label_delta": self.label_delta,
            "hard_fail_delta": self.hard_fail_delta,
            "soft_fail_delta": self.soft_fail_delta,
            "slice_fail_delta": self.slice_fail_delta,
            "any_new_critical_pattern": self.any_new_critical_pattern,
            "new_patterns": list(self.new_patterns),
            "removed_patterns": list(self.removed_patterns),
            "degraded": self.degraded,
            "improved": self.improved,
            "unchanged": self.unchanged,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Node Configuration
# =============================================================================


# Critical nodes that trigger EVIDENCE_ONLY on failure (Section 11.2)
CRITICAL_NODES: FrozenSet[str] = frozenset({"N01", "N02", "N03", "N04", "N05"})

# Slice pair mappings (Runner → Evaluator)
SLICE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("goal", "N10", "N20"),
    ("sparse", "N11", "N21"),
    ("tree", "N12", "N22"),
    ("dep", "N13", "N23"),
)

# Node ID to type mapping
NODE_TYPE_MAP: Dict[str, NodeType] = {
    "N01": NodeType.GATEKEEPER,
    "N02": NodeType.GATEKEEPER,
    "N03": NodeType.LOADER,
    "N04": NodeType.VALIDATOR,
    "N05": NodeType.VALIDATOR,
    "N10": NodeType.RUNNER,
    "N11": NodeType.RUNNER,
    "N12": NodeType.RUNNER,
    "N13": NodeType.RUNNER,
    "N20": NodeType.EVALUATOR,
    "N21": NodeType.EVALUATOR,
    "N22": NodeType.EVALUATOR,
    "N23": NodeType.EVALUATOR,
    "N30": NodeType.SYNC_BARRIER,
    "N40": NodeType.ANALYZER,
    "N41": NodeType.ANALYZER,
    "N50": NodeType.SUMMARIZER,
    "N60": NodeType.AUDITOR,
    "N70": NodeType.PACKAGER,
    "N80": NodeType.ATTESTER,
    "N90": NodeType.SEALER,
    "N98": NodeType.PACKAGER,  # QUARANTINE
    "N99": NodeType.GATEKEEPER,  # ABORT_GATE
}

# Hard-fail stages (Section 12.2)
HARD_FAIL_STAGES: FrozenSet[str] = frozenset({
    "gate-check",
    "prereg-verify",
    "curriculum-load",
    "dry-run",
    "manifest-init",
    "integrity-check",
    "contamination-check",
})

# Exit code ranges for failure classification (Section 12.2)
EXIT_CODE_RANGES: Dict[str, Tuple[int, int]] = {
    "gate-check": (100, 109),
    "prereg-verify": (110, 119),
    "curriculum-load": (120, 129),
    "dry-run": (130, 139),
    "manifest-init": (140, 149),
    "slice-runner": (200, 249),
    "evaluator": (300, 349),
    "sync-barrier": (400, 449),
    "analyzer": (500, 549),
    "statistics": (600, 649),
    "audit": (700, 749),
    "packaging": (800, 849),
    "sealing": (900, 949),
    "infrastructure": (990, 999),
}


# =============================================================================
# Failure Pattern Definitions (Section 11.2)
# =============================================================================


FAILURE_PATTERNS: Dict[str, FailurePattern] = {
    # Gatekeeper patterns
    "GK-001": FailurePattern(
        "GK-001", "Gate file missing", "File not deployed",
        "Deploy gate file", NodeType.GATEKEEPER
    ),
    "GK-002": FailurePattern(
        "GK-002", "Hash mismatch", "File corrupted/modified",
        "Restore from VCS", NodeType.GATEKEEPER
    ),
    "GK-003": FailurePattern(
        "GK-003", "Permission denied", "ACL misconfiguration",
        "Fix file permissions", NodeType.GATEKEEPER
    ),
    "GK-004": FailurePattern(
        "GK-004", "Timeout on read", "Disk I/O issue",
        "Check storage health", NodeType.GATEKEEPER
    ),
    "GK-005": FailurePattern(
        "GK-005", "Invalid JSON/YAML", "Syntax error in gate file",
        "Fix syntax, re-deploy", NodeType.GATEKEEPER
    ),
    # Validator patterns
    "VD-001": FailurePattern(
        "VD-001", "Schema validation fail", "Config schema mismatch",
        "Update config to schema", NodeType.VALIDATOR
    ),
    "VD-002": FailurePattern(
        "VD-002", "Missing required field", "Incomplete configuration",
        "Add required fields", NodeType.VALIDATOR
    ),
    "VD-003": FailurePattern(
        "VD-003", "Invalid seed value", "Seed out of range",
        "Use valid MDAP_SEED", NodeType.VALIDATOR
    ),
    "VD-004": FailurePattern(
        "VD-004", "Cycle count mismatch", "n_cycles inconsistent",
        "Align cycle counts", NodeType.VALIDATOR
    ),
    "VD-005": FailurePattern(
        "VD-005", "Dry-run assertion fail", "Logic error in config",
        "Debug configuration", NodeType.VALIDATOR
    ),
    # Loader patterns
    "LD-001": FailurePattern(
        "LD-001", "YAML parse error", "Invalid YAML syntax",
        "Fix YAML file", NodeType.LOADER
    ),
    "LD-002": FailurePattern(
        "LD-002", "Missing slice config", "Slice not defined",
        "Add slice definition", NodeType.LOADER
    ),
    "LD-003": FailurePattern(
        "LD-003", "Circular reference", "Config references loop",
        "Break circular refs", NodeType.LOADER
    ),
    "LD-004": FailurePattern(
        "LD-004", "Memory exhaustion", "Config too large",
        "Optimize config size", NodeType.LOADER
    ),
    "LD-005": FailurePattern(
        "LD-005", "File not found", "Path incorrect",
        "Fix config path", NodeType.LOADER
    ),
    # Runner patterns
    "RN-001": FailurePattern(
        "RN-001", "Zero proofs generated", "Empty formula pool",
        "Expand formula pool", NodeType.RUNNER
    ),
    "RN-002": FailurePattern(
        "RN-002", "Cycle timeout", "Verifier hanging",
        "Increase timeout/retry", NodeType.RUNNER
    ),
    "RN-003": FailurePattern(
        "RN-003", "Memory exhaustion", "Large derivation tree",
        "Reduce breadth", NodeType.RUNNER
    ),
    "RN-004": FailurePattern(
        "RN-004", "Verifier crash", "Lean process failure",
        "Restart verifier", NodeType.RUNNER
    ),
    "RN-005": FailurePattern(
        "RN-005", "Seed mismatch", "Non-deterministic init",
        "Fix seed propagation", NodeType.RUNNER
    ),
    "RN-006": FailurePattern(
        "RN-006", "Partial cycle", "Interrupted execution",
        "Invalidate + retry", NodeType.RUNNER
    ),
    "RN-007": FailurePattern(
        "RN-007", "Baseline/RFL divergence", "Different cycle counts",
        "Align counts", NodeType.RUNNER
    ),
    # Evaluator patterns
    "EV-001": FailurePattern(
        "EV-001", "Missing metric field", "Evaluator bug",
        "Fix evaluator code", NodeType.EVALUATOR
    ),
    "EV-002": FailurePattern(
        "EV-002", "NaN in computation", "Division by zero",
        "Add zero-check", NodeType.EVALUATOR
    ),
    "EV-003": FailurePattern(
        "EV-003", "Metric out of range", "Computation error",
        "Validate bounds", NodeType.EVALUATOR
    ),
    "EV-004": FailurePattern(
        "EV-004", "JSON serialization fail", "Complex object",
        "Simplify output", NodeType.EVALUATOR
    ),
    "EV-005": FailurePattern(
        "EV-005", "Log file unreadable", "Encoding issue",
        "Fix log encoding", NodeType.EVALUATOR
    ),
    # Sync barrier patterns
    "SY-001": FailurePattern(
        "SY-001", "Indefinite wait", "Upstream deadlock",
        "Timeout and degrade", NodeType.SYNC_BARRIER
    ),
    "SY-002": FailurePattern(
        "SY-002", "Incomplete slice set", "Missing evaluators",
        "Check evaluator status", NodeType.SYNC_BARRIER
    ),
    "SY-003": FailurePattern(
        "SY-003", "Memory spike", "Large result sets",
        "Stream results", NodeType.SYNC_BARRIER
    ),
    # Analyzer patterns
    "AN-001": FailurePattern(
        "AN-001", "Chain depth overflow", "Unbounded recursion",
        "Add depth limit", NodeType.ANALYZER
    ),
    "AN-002": FailurePattern(
        "AN-002", "Graph cycle detected", "Corrupted proof DAG",
        "Flag and skip", NodeType.ANALYZER
    ),
    "AN-003": FailurePattern(
        "AN-003", "Memory exhaustion", "Large DAG",
        "Batch processing", NodeType.ANALYZER
    ),
    "AN-004": FailurePattern(
        "AN-004", "Missing dependencies", "Incomplete proof data",
        "Report missing data", NodeType.ANALYZER
    ),
    # Summarizer patterns
    "SM-001": FailurePattern(
        "SM-001", "CI computation fail", "Invalid input data",
        "Validate inputs", NodeType.SUMMARIZER
    ),
    "SM-002": FailurePattern(
        "SM-002", "Cohen's h undefined", "Zero variance",
        "Report as N/A", NodeType.SUMMARIZER
    ),
    "SM-003": FailurePattern(
        "SM-003", "Z-test failure", "Sample size issue",
        "Report limitations", NodeType.SUMMARIZER
    ),
    "SM-004": FailurePattern(
        "SM-004", "Summary incomplete", "Missing slice data",
        "Note missing slices", NodeType.SUMMARIZER
    ),
    # Auditor patterns
    "AU-001": FailurePattern(
        "AU-001", "Checksum mismatch", "File corruption",
        "Flag corrupt files", NodeType.AUDITOR
    ),
    "AU-002": FailurePattern(
        "AU-002", "Missing audit trail", "Incomplete logging",
        "Add audit events", NodeType.AUDITOR
    ),
    "AU-003": FailurePattern(
        "AU-003", "Timestamp anomaly", "Clock skew",
        "Flag and report", NodeType.AUDITOR
    ),
    "AU-004": FailurePattern(
        "AU-004", "Duplicate entries", "Retry artifacts",
        "Deduplicate", NodeType.AUDITOR
    ),
    "AU-005": FailurePattern(
        "AU-005", "Phase I reference", "Contamination",
        "EVIDENCE_ONLY mode", NodeType.AUDITOR
    ),
    # Packager patterns
    "PK-001": FailurePattern(
        "PK-001", "Missing artifacts", "Upstream failure",
        "Include available only", NodeType.PACKAGER
    ),
    "PK-002": FailurePattern(
        "PK-002", "Compression failure", "Corrupt data",
        "Skip compression", NodeType.PACKAGER
    ),
    "PK-003": FailurePattern(
        "PK-003", "Disk space exhausted", "Insufficient storage",
        "Clean temp files", NodeType.PACKAGER
    ),
    "PK-004": FailurePattern(
        "PK-004", "Manifest incomplete", "Missing metadata",
        "Generate partial manifest", NodeType.PACKAGER
    ),
    # Attester patterns
    "AT-001": FailurePattern(
        "AT-001", "Signature failure", "Key issue",
        "Use backup key", NodeType.ATTESTER
    ),
    "AT-002": FailurePattern(
        "AT-002", "Hash mismatch", "Concurrent modification",
        "Lock files", NodeType.ATTESTER
    ),
    "AT-003": FailurePattern(
        "AT-003", "Timestamp service down", "External dependency",
        "Use local timestamp", NodeType.ATTESTER
    ),
    # Sealer patterns
    "SL-001": FailurePattern(
        "SL-001", "Archive creation fail", "Disk full",
        "Clean and retry", NodeType.SEALER
    ),
    "SL-002": FailurePattern(
        "SL-002", "Final hash mismatch", "Late modification",
        "Recompute from source", NodeType.SEALER
    ),
    "SL-003": FailurePattern(
        "SL-003", "Metadata incomplete", "Upstream failure",
        "Seal with available", NodeType.SEALER
    ),
}


# =============================================================================
# Pattern Library - Severity Mappings (Task 2)
# =============================================================================


PATTERN_LIBRARY: Dict[str, PatternLibraryEntry] = {
    # Gatekeeper patterns - CRITICAL severity, strongly suggest EVIDENCE_ONLY
    "GK-001": PatternLibraryEntry(
        "GK-001", PatternSeverity.CRITICAL, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Gate file missing - pipeline cannot validate"
    ),
    "GK-002": PatternLibraryEntry(
        "GK-002", PatternSeverity.CRITICAL, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Hash mismatch - integrity compromised"
    ),
    "GK-003": PatternLibraryEntry(
        "GK-003", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Permission denied - access control issue"
    ),
    "GK-004": PatternLibraryEntry(
        "GK-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Timeout on read - infrastructure issue"
    ),
    "GK-005": PatternLibraryEntry(
        "GK-005", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Invalid config syntax - cannot parse"
    ),
    # Validator patterns - HIGH severity, typically EVIDENCE_ONLY
    "VD-001": PatternLibraryEntry(
        "VD-001", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Schema validation failed - config invalid"
    ),
    "VD-002": PatternLibraryEntry(
        "VD-002", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Missing required field - config incomplete"
    ),
    "VD-003": PatternLibraryEntry(
        "VD-003", PatternSeverity.MEDIUM, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Invalid seed value - determinism at risk"
    ),
    "VD-004": PatternLibraryEntry(
        "VD-004", PatternSeverity.MEDIUM, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Cycle count mismatch - comparison invalid"
    ),
    "VD-005": PatternLibraryEntry(
        "VD-005", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Dry-run assertion failed - logic error"
    ),
    # Loader patterns - HIGH severity, typically EVIDENCE_ONLY
    "LD-001": PatternLibraryEntry(
        "LD-001", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "YAML parse error - cannot load config"
    ),
    "LD-002": PatternLibraryEntry(
        "LD-002", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Missing slice config - cannot run slice"
    ),
    "LD-003": PatternLibraryEntry(
        "LD-003", PatternSeverity.MEDIUM, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Circular reference - config error"
    ),
    "LD-004": PatternLibraryEntry(
        "LD-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Memory exhaustion - resource issue"
    ),
    "LD-005": PatternLibraryEntry(
        "LD-005", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "File not found - config missing"
    ),
    # Runner patterns - MEDIUM severity, allow DEGRADED_ANALYSIS
    "RN-001": PatternLibraryEntry(
        "RN-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Zero proofs generated - empty output"
    ),
    "RN-002": PatternLibraryEntry(
        "RN-002", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Cycle timeout - verifier hung"
    ),
    "RN-003": PatternLibraryEntry(
        "RN-003", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Memory exhaustion - resource limit"
    ),
    "RN-004": PatternLibraryEntry(
        "RN-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Verifier crash - Lean process failed"
    ),
    "RN-005": PatternLibraryEntry(
        "RN-005", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Seed mismatch - non-deterministic"
    ),
    "RN-006": PatternLibraryEntry(
        "RN-006", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Partial cycle - incomplete execution"
    ),
    "RN-007": PatternLibraryEntry(
        "RN-007", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Baseline/RFL divergence - comparison invalid"
    ),
    # Evaluator patterns - MEDIUM severity, allow DEGRADED_ANALYSIS
    "EV-001": PatternLibraryEntry(
        "EV-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Missing metric field - evaluator bug"
    ),
    "EV-002": PatternLibraryEntry(
        "EV-002", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "NaN in computation - edge case"
    ),
    "EV-003": PatternLibraryEntry(
        "EV-003", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Metric out of range - unexpected value"
    ),
    "EV-004": PatternLibraryEntry(
        "EV-004", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "JSON serialization failed - output issue"
    ),
    "EV-005": PatternLibraryEntry(
        "EV-005", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Log file unreadable - encoding issue"
    ),
    # Sync barrier patterns - MEDIUM severity
    "SY-001": PatternLibraryEntry(
        "SY-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Indefinite wait - upstream deadlock"
    ),
    "SY-002": PatternLibraryEntry(
        "SY-002", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Incomplete slice set - missing evaluators"
    ),
    "SY-003": PatternLibraryEntry(
        "SY-003", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Memory spike - large result sets"
    ),
    # Analyzer patterns - MEDIUM severity
    "AN-001": PatternLibraryEntry(
        "AN-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Chain depth overflow - unbounded recursion"
    ),
    "AN-002": PatternLibraryEntry(
        "AN-002", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Graph cycle detected - corrupted DAG"
    ),
    "AN-003": PatternLibraryEntry(
        "AN-003", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Memory exhaustion - large DAG"
    ),
    "AN-004": PatternLibraryEntry(
        "AN-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Missing dependencies - incomplete data"
    ),
    # Summarizer patterns - MEDIUM severity
    "SM-001": PatternLibraryEntry(
        "SM-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "CI computation failed - invalid input"
    ),
    "SM-002": PatternLibraryEntry(
        "SM-002", PatternSeverity.LOW, PipelineMode.FULL_PIPELINE,
        FailBehavior.SOFT_FAIL, "Cohen's h undefined - zero variance (info only)"
    ),
    "SM-003": PatternLibraryEntry(
        "SM-003", PatternSeverity.LOW, PipelineMode.FULL_PIPELINE,
        FailBehavior.SOFT_FAIL, "Z-test failure - sample size issue (info only)"
    ),
    "SM-004": PatternLibraryEntry(
        "SM-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Summary incomplete - missing slice data"
    ),
    # Auditor patterns - HIGH severity for integrity issues
    "AU-001": PatternLibraryEntry(
        "AU-001", PatternSeverity.CRITICAL, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Checksum mismatch - file corruption"
    ),
    "AU-002": PatternLibraryEntry(
        "AU-002", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Missing audit trail - incomplete logging"
    ),
    "AU-003": PatternLibraryEntry(
        "AU-003", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Timestamp anomaly - clock skew"
    ),
    "AU-004": PatternLibraryEntry(
        "AU-004", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Duplicate entries - retry artifacts"
    ),
    "AU-005": PatternLibraryEntry(
        "AU-005", PatternSeverity.CRITICAL, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Phase I reference - contamination detected"
    ),
    # Packager patterns - LOW to MEDIUM severity
    "PK-001": PatternLibraryEntry(
        "PK-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Missing artifacts - upstream failure"
    ),
    "PK-002": PatternLibraryEntry(
        "PK-002", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Compression failure - corrupt data"
    ),
    "PK-003": PatternLibraryEntry(
        "PK-003", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Disk space exhausted - storage issue"
    ),
    "PK-004": PatternLibraryEntry(
        "PK-004", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Manifest incomplete - missing metadata"
    ),
    # Attester patterns - HIGH severity for seal issues
    "AT-001": PatternLibraryEntry(
        "AT-001", PatternSeverity.HIGH, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Signature failure - key issue"
    ),
    "AT-002": PatternLibraryEntry(
        "AT-002", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Hash mismatch - concurrent modification"
    ),
    "AT-003": PatternLibraryEntry(
        "AT-003", PatternSeverity.LOW, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Timestamp service down - external dependency"
    ),
    # Sealer patterns - MEDIUM severity
    "SL-001": PatternLibraryEntry(
        "SL-001", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Archive creation failed - disk full"
    ),
    "SL-002": PatternLibraryEntry(
        "SL-002", PatternSeverity.HIGH, PipelineMode.EVIDENCE_ONLY,
        FailBehavior.HARD_FAIL, "Final hash mismatch - late modification"
    ),
    "SL-003": PatternLibraryEntry(
        "SL-003", PatternSeverity.MEDIUM, PipelineMode.DEGRADED_ANALYSIS,
        FailBehavior.SOFT_FAIL, "Metadata incomplete - upstream failure"
    ),
}


def get_pattern_severity(pattern_id: str) -> Optional[PatternSeverity]:
    """Get severity for a pattern ID from the pattern library."""
    entry = PATTERN_LIBRARY.get(pattern_id)
    return entry.severity if entry else None


def get_pattern_suggested_mode(pattern_id: str) -> Optional[PipelineMode]:
    """Get suggested mode for a pattern ID from the pattern library."""
    entry = PATTERN_LIBRARY.get(pattern_id)
    return entry.suggested_mode if entry else None


def is_critical_pattern(pattern_id: str) -> bool:
    """Check if a pattern is critical severity."""
    severity = get_pattern_severity(pattern_id)
    return severity == PatternSeverity.CRITICAL


# =============================================================================
# Topology Health Evaluator
# =============================================================================


class TopologyHealthEvaluator:
    """
    Evaluates pipeline health from node statuses.

    Implements the health evaluation logic defined in Section 11.3
    (Health Aggregation Rules) and Section 10.3 (Mode Detection Logic).

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self._node_type_map = NODE_TYPE_MAP
        self._critical_nodes = CRITICAL_NODES
        self._slice_pairs = SLICE_PAIRS
        self._failure_patterns = FAILURE_PATTERNS

    def evaluate_pipeline_health(
        self,
        node_statuses: Dict[str, NodeStatus],
        node_health_map: Optional[Dict[str, NodeHealth]] = None,
        integrity_checks: Optional[List[IntegrityCheck]] = None,
    ) -> PipelineHealth:
        """
        Evaluate overall pipeline health from node statuses.

        Implements the aggregation rules from Section 11.3.

        Args:
            node_statuses: Mapping of node_id → NodeStatus
            node_health_map: Optional detailed health info per node
            integrity_checks: Optional integrity check results

        Returns:
            PipelineHealth assessment with mode and governance label
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        failed_nodes: List[str] = []
        detected_patterns: List[FailurePattern] = []
        health_signals: Dict[str, HealthSignals] = {}

        # Extract health signals if available
        if node_health_map:
            for node_id, health in node_health_map.items():
                health_signals[node_id] = health.signals
                if health.failure_patterns:
                    detected_patterns.extend(health.failure_patterns)

        # Rule 1: Check critical nodes (Section 11.3)
        # Critical nodes trigger EVIDENCE_ONLY on failure
        for node_id in self._critical_nodes:
            status = node_statuses.get(node_id)
            if status == NodeStatus.FAIL:
                failed_nodes.append(node_id)
                return PipelineHealth(
                    mode=PipelineMode.EVIDENCE_ONLY,
                    governance_label=GovernanceLabel.DO_NOT_USE,
                    failed_nodes=tuple(failed_nodes),
                    successful_slices=(),
                    failed_slices=("goal", "sparse", "tree", "dep"),
                    health_signals=health_signals,
                    detected_patterns=tuple(detected_patterns),
                    timestamp=timestamp,
                )

        # Rule 2: Check integrity (Section 10.3)
        if integrity_checks:
            for check in integrity_checks:
                if not check.passed:
                    return PipelineHealth(
                        mode=PipelineMode.EVIDENCE_ONLY,
                        governance_label=GovernanceLabel.DO_NOT_USE,
                        failed_nodes=tuple(failed_nodes),
                        successful_slices=(),
                        failed_slices=("goal", "sparse", "tree", "dep"),
                        health_signals=health_signals,
                        detected_patterns=tuple(detected_patterns),
                        timestamp=timestamp,
                    )

        # Rule 3: Count successful slice pairs (Section 11.3)
        successful_slices: List[str] = []
        failed_slices: List[str] = []

        for slice_name, runner_id, evaluator_id in self._slice_pairs:
            runner_status = node_statuses.get(runner_id, NodeStatus.NOT_RUN)
            evaluator_status = node_statuses.get(evaluator_id, NodeStatus.NOT_RUN)

            # A pair succeeds only if BOTH runner AND evaluator pass
            runner_ok = runner_status in (NodeStatus.OK, NodeStatus.WARN)
            evaluator_ok = evaluator_status in (NodeStatus.OK, NodeStatus.WARN)

            if runner_ok and evaluator_ok:
                successful_slices.append(slice_name)
            else:
                failed_slices.append(slice_name)
                if runner_status == NodeStatus.FAIL:
                    failed_nodes.append(runner_id)
                if evaluator_status == NodeStatus.FAIL:
                    failed_nodes.append(evaluator_id)

        # Determine mode based on successful pair count
        successful_count = len(successful_slices)

        if successful_count == 4:
            # All 4 pairs successful → FULL_PIPELINE
            mode = PipelineMode.FULL_PIPELINE
            governance_label = GovernanceLabel.OK
        elif successful_count >= 2:
            # 2-3 pairs successful → DEGRADED_ANALYSIS
            mode = PipelineMode.DEGRADED_ANALYSIS
            governance_label = GovernanceLabel.WARN
        else:
            # Fewer than 2 pairs → EVIDENCE_ONLY
            mode = PipelineMode.EVIDENCE_ONLY
            governance_label = GovernanceLabel.DO_NOT_USE

        return PipelineHealth(
            mode=mode,
            governance_label=governance_label,
            failed_nodes=tuple(failed_nodes),
            successful_slices=tuple(successful_slices),
            failed_slices=tuple(failed_slices),
            health_signals=health_signals,
            detected_patterns=tuple(detected_patterns),
            timestamp=timestamp,
        )

    def evaluate_from_slice_results(
        self,
        validation_status: ValidationStatus,
        slice_results: List[SliceResult],
        integrity_checks: Optional[List[IntegrityCheck]] = None,
    ) -> PipelineHealth:
        """
        Evaluate pipeline health from slice results.

        Implements Section 10.3 Mode Detection Logic.

        Args:
            validation_status: Status of validation stages (N01-N05)
            slice_results: Results of slice executions
            integrity_checks: Optional integrity check results

        Returns:
            PipelineHealth assessment
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Rule 1: Validation failure → EVIDENCE_ONLY
        if not validation_status.all_passed:
            return PipelineHealth(
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                failed_nodes=self._get_failed_validation_nodes(validation_status),
                successful_slices=(),
                failed_slices=("goal", "sparse", "tree", "dep"),
                health_signals={},
                detected_patterns=(),
                timestamp=timestamp,
            )

        # Rule 2: Corruption detected → EVIDENCE_ONLY
        if integrity_checks:
            for check in integrity_checks:
                if not check.passed:
                    return PipelineHealth(
                        mode=PipelineMode.EVIDENCE_ONLY,
                        governance_label=GovernanceLabel.DO_NOT_USE,
                        failed_nodes=(),
                        successful_slices=(),
                        failed_slices=("goal", "sparse", "tree", "dep"),
                        health_signals={},
                        detected_patterns=(),
                        timestamp=timestamp,
                    )

        # Count successful slice pairs
        successful_slices = [s.slice_name for s in slice_results if s.is_complete()]
        failed_slices = [s.slice_name for s in slice_results if not s.is_complete()]
        successful_count = len(successful_slices)

        # Collect failed nodes
        failed_nodes: List[str] = []
        for s in slice_results:
            if not s.baseline_completed or not s.rfl_completed:
                failed_nodes.append(s.runner_node_id)
                failed_nodes.append(s.evaluator_node_id)

        # Determine mode
        if successful_count == 4:
            mode = PipelineMode.FULL_PIPELINE
            governance_label = GovernanceLabel.OK
        elif successful_count >= 2:
            mode = PipelineMode.DEGRADED_ANALYSIS
            governance_label = GovernanceLabel.WARN
        else:
            mode = PipelineMode.EVIDENCE_ONLY
            governance_label = GovernanceLabel.DO_NOT_USE

        return PipelineHealth(
            mode=mode,
            governance_label=governance_label,
            failed_nodes=tuple(failed_nodes),
            successful_slices=tuple(successful_slices),
            failed_slices=tuple(failed_slices),
            health_signals={},
            detected_patterns=(),
            timestamp=timestamp,
        )

    def _get_failed_validation_nodes(
        self, validation_status: ValidationStatus
    ) -> Tuple[str, ...]:
        """Get list of failed validation node IDs."""
        failed = []
        if not validation_status.gate_check_passed:
            failed.extend(["N01", "N02"])
        if not validation_status.prereg_verify_passed:
            failed.append("N02")
        if not validation_status.curriculum_load_passed:
            failed.append("N03")
        if not validation_status.dry_run_passed:
            failed.append("N04")
        if not validation_status.manifest_init_passed:
            failed.append("N05")
        return tuple(failed)

    def detect_failure_patterns(
        self,
        node_id: str,
        exit_code: Optional[int],
        error_message: Optional[str],
    ) -> List[FailurePattern]:
        """
        Detect failure patterns based on exit code and error message.

        Args:
            node_id: The node ID (e.g., "N01", "N10")
            exit_code: Exit code from the node
            error_message: Error message if available

        Returns:
            List of detected failure patterns
        """
        patterns = []
        node_type = self._node_type_map.get(node_id)

        if not node_type or not error_message:
            return patterns

        # Pattern matching based on error message keywords
        error_lower = error_message.lower()

        pattern_keywords = {
            "GK-001": ["gate file", "missing", "not found"],
            "GK-002": ["hash", "mismatch", "corrupted"],
            "GK-003": ["permission", "denied", "access"],
            "GK-004": ["timeout", "read"],
            "GK-005": ["invalid", "json", "yaml", "syntax"],
            "VD-001": ["schema", "validation"],
            "VD-002": ["required", "field", "missing"],
            "VD-003": ["seed", "invalid"],
            "VD-004": ["cycle", "count", "mismatch"],
            "VD-005": ["dry-run", "assertion"],
            "LD-001": ["yaml", "parse"],
            "LD-002": ["slice", "config", "missing"],
            "LD-003": ["circular", "reference"],
            "LD-004": ["memory", "exhaustion"],
            "LD-005": ["file", "not found"],
            "RN-001": ["zero proofs", "no proofs"],
            "RN-002": ["cycle", "timeout"],
            "RN-003": ["memory", "exhaustion"],
            "RN-004": ["verifier", "crash"],
            "RN-005": ["seed", "mismatch"],
            "RN-006": ["partial", "cycle"],
            "RN-007": ["baseline", "rfl", "divergence"],
            "EV-001": ["metric", "field", "missing"],
            "EV-002": ["nan", "division"],
            "EV-003": ["out of range", "bounds"],
            "EV-004": ["serialization", "json"],
            "EV-005": ["log", "encoding", "unreadable"],
            "AU-001": ["checksum", "mismatch"],
            "AU-002": ["audit", "trail", "missing"],
            "AU-003": ["timestamp", "anomaly"],
            "AU-004": ["duplicate"],
            "AU-005": ["phase i", "contamination"],
        }

        for pattern_id, keywords in pattern_keywords.items():
            pattern = self._failure_patterns.get(pattern_id)
            if pattern and pattern.node_type == node_type:
                if any(kw in error_lower for kw in keywords):
                    patterns.append(pattern)

        return patterns


# =============================================================================
# Degradation Policy Engine
# =============================================================================


class DegradationPolicyEngine:
    """
    Applies degradation policy rules and constraints.

    Implements the CI Degradation Policy from Section 12, including:
    - The Cardinal Rule (Section 12.1)
    - Hard-fail vs soft-fail semantics (Sections 12.2-12.3)
    - Constraints C1-C5

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """

    # Constraints from the pipeline topology specification
    CONSTRAINT_C1 = "C1: No skip edges - all validation must complete"
    CONSTRAINT_C2 = "C2: No parallel hazard - avoid concurrent mutations"
    CONSTRAINT_C3 = "C3: Barrier sync - all evaluators must sync"
    CONSTRAINT_C4 = "C4: Failure isolation - failed slices must not pollute"
    CONSTRAINT_C5 = "C5: Quarantine before retry - failed experiments isolated"

    def __init__(self):
        """Initialize the engine."""
        self._hard_fail_stages = HARD_FAIL_STAGES
        self._exit_code_ranges = EXIT_CODE_RANGES

    def evaluate_degradation(
        self,
        ci_stage_results: Dict[str, Dict],
        pipeline_health: Optional[PipelineHealth] = None,
    ) -> DegradationDecision:
        """
        Apply degradation rules to CI stage results.

        Implements constraints C1-C5 and the Cardinal Rule:
        "No node may fail in such a way that Δp is computed from
        partial or corrupted data."

        Args:
            ci_stage_results: Dict of stage_name → {status, exit_code, ...}
            pipeline_health: Optional pre-computed health assessment

        Returns:
            DegradationDecision with mode, permissions, and violations
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        violations: List[str] = []

        # Check hard-fail stages first
        hard_fail_triggered = False
        hard_fail_stage = None

        for stage_name in self._hard_fail_stages:
            result = ci_stage_results.get(stage_name)
            if result and result.get("status") == "FAIL":
                hard_fail_triggered = True
                hard_fail_stage = stage_name
                violations.append(
                    f"HARD-FAIL: Stage '{stage_name}' failed - "
                    f"triggers EVIDENCE_ONLY mode"
                )
                break

        if hard_fail_triggered:
            return DegradationDecision(
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label=GovernanceLabel.DO_NOT_USE,
                allow_delta_p=False,
                allowed_delta_p_slices=frozenset(),
                allow_evidence_pack=True,
                evidence_pack_status="FORENSIC",
                halt_ci=True,
                halt_reason=f"Hard-fail stage '{hard_fail_stage}' failed",
                quarantine=True,
                violations=tuple(violations),
                timestamp=timestamp,
            )

        # Count successful slice pairs
        slice_stages = {
            "goal": ("run-slice-goal", "eval-goal"),
            "sparse": ("run-slice-sparse", "eval-sparse"),
            "tree": ("run-slice-tree", "eval-tree"),
            "dep": ("run-slice-dep", "eval-dep"),
        }

        successful_slices: Set[str] = set()
        failed_slices: Set[str] = set()

        for slice_name, (runner_stage, eval_stage) in slice_stages.items():
            runner_result = ci_stage_results.get(runner_stage, {})
            eval_result = ci_stage_results.get(eval_stage, {})

            runner_ok = runner_result.get("status") in ("OK", "SUCCESS", "PASS")
            eval_ok = eval_result.get("status") in ("OK", "SUCCESS", "PASS")

            if runner_ok and eval_ok:
                successful_slices.add(slice_name)
            else:
                failed_slices.add(slice_name)
                # Apply Cardinal Rule: if either fails, Δp is forbidden
                if not runner_ok:
                    violations.append(
                        f"CARDINAL RULE: Runner for '{slice_name}' failed - "
                        f"Δp({slice_name}) EXCLUDED"
                    )
                if not eval_ok:
                    violations.append(
                        f"CARDINAL RULE: Evaluator for '{slice_name}' failed - "
                        f"Δp({slice_name}) EXCLUDED"
                    )

        # Determine mode based on successful pairs
        successful_count = len(successful_slices)

        if successful_count == 4:
            mode = PipelineMode.FULL_PIPELINE
            governance_label = GovernanceLabel.OK
            allow_delta_p = True
            allowed_delta_p_slices = frozenset(successful_slices)
            evidence_pack_status = "COMPLETE"
            halt_ci = False
            quarantine = False
        elif successful_count >= 2:
            mode = PipelineMode.DEGRADED_ANALYSIS
            governance_label = GovernanceLabel.WARN
            allow_delta_p = True  # But only for successful pairs
            allowed_delta_p_slices = frozenset(successful_slices)
            evidence_pack_status = "PARTIAL"
            halt_ci = False
            quarantine = False
            violations.append(
                f"DEGRADED: Only {successful_count}/4 slice pairs succeeded - "
                f"Δp restricted to: {', '.join(sorted(successful_slices))}"
            )
        else:
            mode = PipelineMode.EVIDENCE_ONLY
            governance_label = GovernanceLabel.DO_NOT_USE
            allow_delta_p = False
            allowed_delta_p_slices = frozenset()
            evidence_pack_status = "FORENSIC"
            halt_ci = True
            quarantine = True
            violations.append(
                f"EVIDENCE_ONLY: Only {successful_count}/4 slice pairs succeeded - "
                f"Δp computation FORBIDDEN"
            )

        # Apply constraint C4: Failure isolation
        for failed_slice in failed_slices:
            if failed_slice in allowed_delta_p_slices:
                # This should never happen, but enforce the constraint
                violations.append(
                    f"C4 VIOLATION: Failed slice '{failed_slice}' was in allowed set"
                )
                allowed_delta_p_slices = allowed_delta_p_slices - {failed_slice}

        return DegradationDecision(
            mode=mode,
            governance_label=governance_label,
            allow_delta_p=allow_delta_p,
            allowed_delta_p_slices=allowed_delta_p_slices,
            allow_evidence_pack=True,
            evidence_pack_status=evidence_pack_status,
            halt_ci=halt_ci,
            halt_reason="Insufficient successful slice pairs" if halt_ci else None,
            quarantine=quarantine,
            violations=tuple(violations),
            timestamp=timestamp,
        )

    def check_cardinal_rule(
        self,
        slice_name: str,
        baseline_completed: bool,
        rfl_completed: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if Δp computation is allowed for a slice.

        Cardinal Rule (Section 12.1):
        No node may fail in such a way that Δp is computed from
        partial or corrupted data.

        Args:
            slice_name: Name of the slice
            baseline_completed: Whether baseline run completed
            rfl_completed: Whether RFL run completed

        Returns:
            Tuple of (allowed, violation_message)
        """
        if baseline_completed and rfl_completed:
            return True, None

        if not baseline_completed and not rfl_completed:
            return False, (
                f"CARDINAL RULE: Both baseline and RFL failed for '{slice_name}' - "
                f"Δp({slice_name}) FORBIDDEN"
            )

        if not baseline_completed:
            return False, (
                f"CARDINAL RULE: Baseline failed for '{slice_name}' - "
                f"RFL's Δp({slice_name}) MUST be excluded"
            )

        # not rfl_completed
        return False, (
            f"CARDINAL RULE: RFL failed for '{slice_name}' - "
            f"Baseline's Δp({slice_name}) MUST be excluded"
        )

    def validate_delta_p_request(
        self,
        slice_name: str,
        decision: DegradationDecision,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate whether Δp computation can proceed for a slice.

        Args:
            slice_name: Name of the slice requesting Δp
            decision: Current degradation decision

        Returns:
            Tuple of (allowed, rejection_reason)
        """
        if not decision.allow_delta_p:
            return False, (
                f"Δp computation globally forbidden in {decision.mode.value} mode"
            )

        if slice_name not in decision.allowed_delta_p_slices:
            return False, (
                f"Δp computation for '{slice_name}' not in allowed set: "
                f"{sorted(decision.allowed_delta_p_slices)}"
            )

        return True, None

    def get_exit_code_category(self, exit_code: int) -> Optional[str]:
        """
        Categorize an exit code based on the ranges in Section 12.

        Args:
            exit_code: The exit code to categorize

        Returns:
            Category name or None if not in any range
        """
        for category, (low, high) in self._exit_code_ranges.items():
            if low <= exit_code <= high:
                return category
        return None

    def is_hard_fail_exit_code(self, exit_code: int) -> bool:
        """Check if an exit code indicates a hard-fail condition."""
        return 100 <= exit_code <= 149 or 700 <= exit_code <= 750

    def explain(
        self,
        decision: DegradationDecision,
        ci_stage_results: Dict[str, Dict],
        pipeline_health: Optional[PipelineHealth] = None,
    ) -> DecisionExplanation:
        """
        Generate a human-friendly, machine-stable explanation of the decision.

        Produces a DecisionExplanation block suitable for CI step summaries
        and MAAS (Monitoring as a Service) integration.

        Args:
            decision: The degradation decision to explain
            ci_stage_results: CI stage results used in the decision
            pipeline_health: Optional pipeline health for pattern info

        Returns:
            DecisionExplanation with reason codes and formatted output
        """
        reason_codes: List[str] = []
        detailed_reasons: List[str] = []
        patterns_detected: List[str] = []
        constraints_violated: List[str] = []

        # Analyze mode and build explanations
        if decision.mode == PipelineMode.FULL_PIPELINE:
            short_summary = (
                "All slices completed successfully. Full Delta-p computation permitted."
            )
            reason_codes.append("MODE:FULL_PIPELINE")

        elif decision.mode == PipelineMode.DEGRADED_ANALYSIS:
            successful_count = len(decision.allowed_delta_p_slices)
            short_summary = (
                f"{successful_count}/4 slice pairs succeeded. "
                f"Delta-p restricted to: {', '.join(sorted(decision.allowed_delta_p_slices))}."
            )
            reason_codes.append("MODE:DEGRADED_ANALYSIS")
            reason_codes.append(f"SLICES:{successful_count}/4")

        else:  # EVIDENCE_ONLY
            short_summary = (
                "Critical failure detected. Delta-p computation forbidden. "
                "Evidence preserved for forensic analysis."
            )
            reason_codes.append("MODE:EVIDENCE_ONLY")

        # Add governance label to reasons
        reason_codes.append(f"LABEL:{decision.governance_label.value}")

        # Analyze violations for detailed reasons
        for violation in decision.violations:
            if "HARD-FAIL" in violation:
                # Extract stage name
                if "Stage '" in violation:
                    stage = violation.split("Stage '")[1].split("'")[0]
                    detailed_reasons.append(
                        f"Hard-fail stage '{stage}' failed, triggering EVIDENCE_ONLY"
                    )
                    constraints_violated.append("HARD_FAIL_POLICY")
                    # Check for associated patterns
                    result = ci_stage_results.get(stage, {})
                    if result.get("pattern_id"):
                        patterns_detected.append(result["pattern_id"])

            elif "CARDINAL RULE" in violation:
                constraints_violated.append("CARDINAL_RULE")
                detailed_reasons.append(violation.replace("CARDINAL RULE: ", ""))

            elif "DEGRADED:" in violation:
                detailed_reasons.append(violation.replace("DEGRADED: ", ""))

            elif "EVIDENCE_ONLY:" in violation:
                detailed_reasons.append(violation.replace("EVIDENCE_ONLY: ", ""))

            elif "C4 VIOLATION" in violation:
                constraints_violated.append("C4_FAILURE_ISOLATION")
                detailed_reasons.append(violation)

        # Add pattern information from pipeline health if available
        if pipeline_health:
            for pattern in pipeline_health.detected_patterns:
                if pattern.pattern_id not in patterns_detected:
                    patterns_detected.append(pattern.pattern_id)
                    # Add pattern severity context
                    severity = get_pattern_severity(pattern.pattern_id)
                    if severity:
                        detailed_reasons.append(
                            f"Pattern {pattern.pattern_id} ({severity.value}): "
                            f"{pattern.symptoms}"
                        )

        # Deduplicate and sort
        reason_codes = list(dict.fromkeys(reason_codes))
        patterns_detected = list(dict.fromkeys(patterns_detected))
        constraints_violated = list(dict.fromkeys(constraints_violated))

        return DecisionExplanation(
            reason_codes=tuple(reason_codes),
            short_summary=short_summary,
            detailed_reasons=tuple(detailed_reasons),
            patterns_detected=tuple(sorted(patterns_detected)),
            constraints_violated=tuple(sorted(constraints_violated)),
            mode=decision.mode,
            governance_label=decision.governance_label,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_pipeline_health(
    node_statuses: Dict[str, NodeStatus],
    node_health_map: Optional[Dict[str, NodeHealth]] = None,
    integrity_checks: Optional[List[IntegrityCheck]] = None,
) -> PipelineHealth:
    """
    Convenience function to evaluate pipeline health.

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """
    evaluator = TopologyHealthEvaluator()
    return evaluator.evaluate_pipeline_health(
        node_statuses, node_health_map, integrity_checks
    )


def evaluate_degradation(
    ci_stage_results: Dict[str, Dict],
    pipeline_health: Optional[PipelineHealth] = None,
) -> DegradationDecision:
    """
    Convenience function to evaluate degradation.

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """
    engine = DegradationPolicyEngine()
    return engine.evaluate_degradation(ci_stage_results, pipeline_health)


# =============================================================================
# Longitudinal Health Snapshot & Trend Functions (Task 1)
# =============================================================================


def build_pipeline_health_snapshot(
    health: PipelineHealth,
    decision: DegradationDecision,
    run_id: Optional[str] = None,
) -> HealthSnapshot:
    """
    Build a longitudinal health snapshot from pipeline health and decision.

    This captures the pipeline state at a point in time for trend analysis
    and comparison across runs.

    Args:
        health: The evaluated pipeline health
        decision: The degradation decision
        run_id: Optional run identifier for correlation

    Returns:
        HealthSnapshot capturing the current state
    """
    # Count failures by category
    hard_fail_count = 0
    soft_fail_count = 0
    slice_baseline_fail_count = 0
    slice_rfl_fail_count = 0

    # Classify failed nodes
    for node_id in health.failed_nodes:
        node_type = NODE_TYPE_MAP.get(node_id)
        if node_type in {NodeType.GATEKEEPER, NodeType.LOADER, NodeType.VALIDATOR}:
            hard_fail_count += 1
        elif node_type == NodeType.RUNNER:
            # Runners are baseline executors
            slice_baseline_fail_count += 1
            soft_fail_count += 1
        elif node_type == NodeType.EVALUATOR:
            # Evaluators run on RFL results
            slice_rfl_fail_count += 1
            soft_fail_count += 1
        else:
            soft_fail_count += 1

    # Extract pattern IDs from detected patterns
    pattern_ids = tuple(p.pattern_id for p in health.detected_patterns)

    # Build severity map for detected patterns
    pattern_severities: Dict[str, str] = {}
    for pattern_id in pattern_ids:
        severity = get_pattern_severity(pattern_id)
        if severity:
            pattern_severities[pattern_id] = severity.value

    return HealthSnapshot(
        schema_version=HEALTH_SNAPSHOT_SCHEMA_VERSION,
        mode=decision.mode,
        governance_label=decision.governance_label,
        hard_fail_count=hard_fail_count,
        soft_fail_count=soft_fail_count,
        slice_baseline_fail_count=slice_baseline_fail_count,
        slice_rfl_fail_count=slice_rfl_fail_count,
        successful_slice_count=len(health.successful_slices),
        failed_slice_count=len(health.failed_slices),
        patterns_detected=pattern_ids,
        pattern_severities=pattern_severities,
        timestamp=health.timestamp,
        run_id=run_id,
    )


def compare_health_snapshots(
    old_snapshot: HealthSnapshot,
    new_snapshot: HealthSnapshot,
) -> SnapshotComparison:
    """
    Compare two health snapshots to detect trends.

    Tracks mode transitions, failure count deltas, and new critical patterns.

    Args:
        old_snapshot: The previous/baseline snapshot
        new_snapshot: The current/new snapshot

    Returns:
        SnapshotComparison with delta analysis
    """
    # Mode ordering for comparison (worse modes have higher values)
    mode_order = {
        PipelineMode.FULL_PIPELINE: 0,
        PipelineMode.DEGRADED_ANALYSIS: 1,
        PipelineMode.EVIDENCE_ONLY: 2,
    }

    # Label ordering
    label_order = {
        GovernanceLabel.OK: 0,
        GovernanceLabel.WARN: 1,
        GovernanceLabel.DO_NOT_USE: 2,
    }

    # Compute mode delta
    mode_delta = None
    if old_snapshot.mode != new_snapshot.mode:
        mode_delta = f"{old_snapshot.mode.value} -> {new_snapshot.mode.value}"

    # Compute label delta
    label_delta = None
    if old_snapshot.governance_label != new_snapshot.governance_label:
        label_delta = (
            f"{old_snapshot.governance_label.value} -> "
            f"{new_snapshot.governance_label.value}"
        )

    # Compute failure count deltas
    hard_fail_delta = new_snapshot.hard_fail_count - old_snapshot.hard_fail_count
    soft_fail_delta = new_snapshot.soft_fail_count - old_snapshot.soft_fail_count
    slice_fail_delta = (
        new_snapshot.failed_slice_count - old_snapshot.failed_slice_count
    )

    # Detect new and removed patterns
    old_patterns = set(old_snapshot.patterns_detected)
    new_patterns = set(new_snapshot.patterns_detected)

    new_pattern_ids = tuple(sorted(new_patterns - old_patterns))
    removed_pattern_ids = tuple(sorted(old_patterns - new_patterns))

    # Check for any new critical patterns
    any_new_critical = any(
        is_critical_pattern(p) for p in new_pattern_ids
    )

    # Determine overall health trend
    old_mode_score = mode_order[old_snapshot.mode]
    new_mode_score = mode_order[new_snapshot.mode]

    old_label_score = label_order[old_snapshot.governance_label]
    new_label_score = label_order[new_snapshot.governance_label]

    # Degraded if mode or label got worse, or more failures
    degraded = (
        new_mode_score > old_mode_score
        or new_label_score > old_label_score
        or hard_fail_delta > 0
        or any_new_critical
    )

    # Improved if mode or label got better, and no new critical patterns
    improved = (
        (new_mode_score < old_mode_score or new_label_score < old_label_score)
        and not any_new_critical
        and hard_fail_delta <= 0
    )

    # Unchanged if no significant changes
    unchanged = (
        mode_delta is None
        and label_delta is None
        and hard_fail_delta == 0
        and soft_fail_delta == 0
        and not new_pattern_ids
        and not removed_pattern_ids
    )

    return SnapshotComparison(
        mode_delta=mode_delta,
        label_delta=label_delta,
        hard_fail_delta=hard_fail_delta,
        soft_fail_delta=soft_fail_delta,
        slice_fail_delta=slice_fail_delta,
        any_new_critical_pattern=any_new_critical,
        new_patterns=new_pattern_ids,
        removed_patterns=removed_pattern_ids,
        degraded=degraded,
        improved=improved,
        unchanged=unchanged,
    )


# =============================================================================
# Phase III: Long-Horizon Governance & Forecasting
# =============================================================================

# Schema version for trajectory format
TRAJECTORY_SCHEMA_VERSION = "1.0.0"


class RiskLevel(Enum):
    """Risk level for degradation forecasting."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ModeStability(Enum):
    """Mode stability classification for trajectory analysis."""
    STABLE = "STABLE"          # Consistent mode across runs
    IMPROVING = "IMPROVING"    # Trending toward better modes
    DEGRADING = "DEGRADING"    # Trending toward worse modes
    VOLATILE = "VOLATILE"      # Frequent mode changes


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    A single point in the health trajectory.

    Represents pipeline health at a specific run.
    """
    run_id: Optional[str]
    timestamp: str
    mode: PipelineMode
    governance_label: GovernanceLabel
    hard_fail_count: int
    soft_fail_count: int
    patterns_detected: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode.value,
            "governance_label": self.governance_label.value,
            "hard_fail_count": self.hard_fail_count,
            "soft_fail_count": self.soft_fail_count,
            "patterns_detected": list(self.patterns_detected),
        }

    @classmethod
    def from_snapshot(cls, snapshot: HealthSnapshot) -> "TrajectoryPoint":
        """Create a trajectory point from a health snapshot."""
        return cls(
            run_id=snapshot.run_id,
            timestamp=snapshot.timestamp,
            mode=snapshot.mode,
            governance_label=snapshot.governance_label,
            hard_fail_count=snapshot.hard_fail_count,
            soft_fail_count=snapshot.soft_fail_count,
            patterns_detected=snapshot.patterns_detected,
        )


@dataclass(frozen=True)
class ModeEvolution:
    """
    Tracks how pipeline mode has evolved over time.
    """
    total_runs: int
    full_pipeline_count: int
    degraded_analysis_count: int
    evidence_only_count: int
    mode_transitions: int  # Number of times mode changed
    current_streak: int    # Consecutive runs at current mode
    current_mode: PipelineMode
    stability: ModeStability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_runs": self.total_runs,
            "full_pipeline_count": self.full_pipeline_count,
            "degraded_analysis_count": self.degraded_analysis_count,
            "evidence_only_count": self.evidence_only_count,
            "mode_transitions": self.mode_transitions,
            "current_streak": self.current_streak,
            "current_mode": self.current_mode.value,
            "stability": self.stability.value,
        }


@dataclass(frozen=True)
class PatternTrend:
    """
    Trend information for a specific failure pattern.
    """
    pattern_id: str
    severity: PatternSeverity
    occurrence_count: int
    first_seen_run: Optional[str]
    last_seen_run: Optional[str]
    consecutive_occurrences: int  # Current streak
    resolved: bool  # Not present in latest run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "severity": self.severity.value,
            "occurrence_count": self.occurrence_count,
            "first_seen_run": self.first_seen_run,
            "last_seen_run": self.last_seen_run,
            "consecutive_occurrences": self.consecutive_occurrences,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class FailurePatternTrends:
    """
    Aggregated trends for all failure patterns in trajectory.
    """
    total_unique_patterns: int
    active_patterns: Tuple[str, ...]       # Present in latest run
    resolved_patterns: Tuple[str, ...]     # Were present, now resolved
    recurring_patterns: Tuple[str, ...]    # Appeared in 2+ runs
    chronic_patterns: Tuple[str, ...]      # Appeared in 50%+ of runs
    pattern_details: Tuple[PatternTrend, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_unique_patterns": self.total_unique_patterns,
            "active_patterns": list(self.active_patterns),
            "resolved_patterns": list(self.resolved_patterns),
            "recurring_patterns": list(self.recurring_patterns),
            "chronic_patterns": list(self.chronic_patterns),
            "pattern_details": [p.to_dict() for p in self.pattern_details],
        }


@dataclass(frozen=True)
class HealthTrajectory:
    """
    Long-horizon health trajectory for pipeline governance.

    Captures the evolution of pipeline health across multiple runs
    for trend analysis and forecasting.
    """
    schema_version: str
    trajectory: Tuple[TrajectoryPoint, ...]
    mode_evolution: ModeEvolution
    failure_pattern_trends: FailurePatternTrends
    stability_index: float  # 0.0 (unstable) to 1.0 (perfectly stable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "trajectory": [p.to_dict() for p in self.trajectory],
            "mode_evolution": self.mode_evolution.to_dict(),
            "failure_pattern_trends": self.failure_pattern_trends.to_dict(),
            "stability_index": self.stability_index,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class DegradationRiskPrediction:
    """
    Prediction of degradation risk for the next run.

    Based on deterministic analysis of trajectory patterns.
    """
    next_run_mode_prediction: PipelineMode
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    explanatory_patterns: Tuple[str, ...]
    risk_factors: Tuple[str, ...]
    protective_factors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "next_run_mode_prediction": self.next_run_mode_prediction.value,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "explanatory_patterns": list(self.explanatory_patterns),
            "risk_factors": list(self.risk_factors),
            "protective_factors": list(self.protective_factors),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class GlobalHealthSummary:
    """
    High-level governance summary for pipeline health.

    Suitable for executive dashboards and alerting systems.
    """
    pipeline_ok: bool
    mode_stability: ModeStability
    current_mode: PipelineMode
    current_label: GovernanceLabel
    unresolved_critical_patterns: Tuple[str, ...]
    runs_since_last_full_pipeline: int
    stability_index: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline_ok": self.pipeline_ok,
            "mode_stability": self.mode_stability.value,
            "current_mode": self.current_mode.value,
            "current_label": self.current_label.value,
            "unresolved_critical_patterns": list(self.unresolved_critical_patterns),
            "runs_since_last_full_pipeline": self.runs_since_last_full_pipeline,
            "stability_index": self.stability_index,
            "recommendation": self.recommendation,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Task 1: Health Trajectory Ledger
# =============================================================================


def build_pipeline_health_trajectory(
    snapshots: List[HealthSnapshot],
) -> HealthTrajectory:
    """
    Build a health trajectory from a sequence of snapshots.

    Analyzes the evolution of pipeline health over multiple runs
    to identify trends, patterns, and stability metrics.

    Args:
        snapshots: List of health snapshots in chronological order

    Returns:
        HealthTrajectory with mode evolution, pattern trends, and stability
    """
    if not snapshots:
        # Empty trajectory
        return HealthTrajectory(
            schema_version=TRAJECTORY_SCHEMA_VERSION,
            trajectory=(),
            mode_evolution=ModeEvolution(
                total_runs=0,
                full_pipeline_count=0,
                degraded_analysis_count=0,
                evidence_only_count=0,
                mode_transitions=0,
                current_streak=0,
                current_mode=PipelineMode.EVIDENCE_ONLY,
                stability=ModeStability.VOLATILE,
            ),
            failure_pattern_trends=FailurePatternTrends(
                total_unique_patterns=0,
                active_patterns=(),
                resolved_patterns=(),
                recurring_patterns=(),
                chronic_patterns=(),
                pattern_details=(),
            ),
            stability_index=0.0,
        )

    # Build trajectory points
    trajectory_points = tuple(
        TrajectoryPoint.from_snapshot(s) for s in snapshots
    )

    # Analyze mode evolution
    mode_evolution = _analyze_mode_evolution(snapshots)

    # Analyze pattern trends
    pattern_trends = _analyze_pattern_trends(snapshots)

    # Calculate stability index
    stability_index = _calculate_stability_index(snapshots, mode_evolution)

    return HealthTrajectory(
        schema_version=TRAJECTORY_SCHEMA_VERSION,
        trajectory=trajectory_points,
        mode_evolution=mode_evolution,
        failure_pattern_trends=pattern_trends,
        stability_index=stability_index,
    )


def _analyze_mode_evolution(snapshots: List[HealthSnapshot]) -> ModeEvolution:
    """Analyze how pipeline mode has evolved over time."""
    if not snapshots:
        return ModeEvolution(
            total_runs=0,
            full_pipeline_count=0,
            degraded_analysis_count=0,
            evidence_only_count=0,
            mode_transitions=0,
            current_streak=0,
            current_mode=PipelineMode.EVIDENCE_ONLY,
            stability=ModeStability.VOLATILE,
        )

    total_runs = len(snapshots)

    # Count modes
    full_count = sum(1 for s in snapshots if s.mode == PipelineMode.FULL_PIPELINE)
    degraded_count = sum(1 for s in snapshots if s.mode == PipelineMode.DEGRADED_ANALYSIS)
    evidence_count = sum(1 for s in snapshots if s.mode == PipelineMode.EVIDENCE_ONLY)

    # Count transitions
    transitions = 0
    for i in range(1, len(snapshots)):
        if snapshots[i].mode != snapshots[i - 1].mode:
            transitions += 1

    # Calculate current streak
    current_mode = snapshots[-1].mode
    current_streak = 1
    for i in range(len(snapshots) - 2, -1, -1):
        if snapshots[i].mode == current_mode:
            current_streak += 1
        else:
            break

    # Determine stability
    stability = _determine_mode_stability(snapshots, transitions)

    return ModeEvolution(
        total_runs=total_runs,
        full_pipeline_count=full_count,
        degraded_analysis_count=degraded_count,
        evidence_only_count=evidence_count,
        mode_transitions=transitions,
        current_streak=current_streak,
        current_mode=current_mode,
        stability=stability,
    )


def _determine_mode_stability(
    snapshots: List[HealthSnapshot],
    transitions: int,
) -> ModeStability:
    """Determine the stability classification based on mode history."""
    if len(snapshots) < 2:
        return ModeStability.STABLE

    total_runs = len(snapshots)
    transition_rate = transitions / (total_runs - 1) if total_runs > 1 else 0

    # High transition rate = volatile
    if transition_rate > 0.5:
        return ModeStability.VOLATILE

    # Check for trend direction (compare first half to second half)
    if total_runs >= 4:
        mid = total_runs // 2
        first_half = snapshots[:mid]
        second_half = snapshots[mid:]

        # Mode ordering (higher = worse)
        mode_score = {
            PipelineMode.FULL_PIPELINE: 0,
            PipelineMode.DEGRADED_ANALYSIS: 1,
            PipelineMode.EVIDENCE_ONLY: 2,
        }

        first_avg = sum(mode_score[s.mode] for s in first_half) / len(first_half)
        second_avg = sum(mode_score[s.mode] for s in second_half) / len(second_half)

        if second_avg < first_avg - 0.3:
            return ModeStability.IMPROVING
        elif second_avg > first_avg + 0.3:
            return ModeStability.DEGRADING

    # Low transition rate and no clear trend = stable
    if transition_rate <= 0.2:
        return ModeStability.STABLE

    return ModeStability.VOLATILE


def _analyze_pattern_trends(
    snapshots: List[HealthSnapshot],
) -> FailurePatternTrends:
    """Analyze failure pattern trends across snapshots."""
    if not snapshots:
        return FailurePatternTrends(
            total_unique_patterns=0,
            active_patterns=(),
            resolved_patterns=(),
            recurring_patterns=(),
            chronic_patterns=(),
            pattern_details=(),
        )

    # Collect all patterns and their occurrences
    pattern_runs: Dict[str, List[int]] = {}  # pattern_id -> list of run indices
    for i, snapshot in enumerate(snapshots):
        for pattern_id in snapshot.patterns_detected:
            if pattern_id not in pattern_runs:
                pattern_runs[pattern_id] = []
            pattern_runs[pattern_id].append(i)

    total_runs = len(snapshots)
    latest_patterns = set(snapshots[-1].patterns_detected) if snapshots else set()

    # Build pattern details
    pattern_details: List[PatternTrend] = []
    recurring: List[str] = []
    chronic: List[str] = []
    resolved: List[str] = []

    for pattern_id, run_indices in pattern_runs.items():
        occurrence_count = len(run_indices)
        first_idx = run_indices[0]
        last_idx = run_indices[-1]

        # Calculate consecutive occurrences at end
        consecutive = 0
        for i in range(len(snapshots) - 1, -1, -1):
            if pattern_id in snapshots[i].patterns_detected:
                consecutive += 1
            else:
                break

        is_resolved = pattern_id not in latest_patterns
        severity = get_pattern_severity(pattern_id) or PatternSeverity.MEDIUM

        pattern_details.append(PatternTrend(
            pattern_id=pattern_id,
            severity=severity,
            occurrence_count=occurrence_count,
            first_seen_run=snapshots[first_idx].run_id,
            last_seen_run=snapshots[last_idx].run_id,
            consecutive_occurrences=consecutive,
            resolved=is_resolved,
        ))

        # Categorize
        if occurrence_count >= 2:
            recurring.append(pattern_id)
        if occurrence_count >= total_runs * 0.5:
            chronic.append(pattern_id)
        if is_resolved:
            resolved.append(pattern_id)

    return FailurePatternTrends(
        total_unique_patterns=len(pattern_runs),
        active_patterns=tuple(sorted(latest_patterns)),
        resolved_patterns=tuple(sorted(resolved)),
        recurring_patterns=tuple(sorted(recurring)),
        chronic_patterns=tuple(sorted(chronic)),
        pattern_details=tuple(sorted(pattern_details, key=lambda p: p.pattern_id)),
    )


def _calculate_stability_index(
    snapshots: List[HealthSnapshot],
    mode_evolution: ModeEvolution,
) -> float:
    """
    Calculate a stability index from 0.0 (unstable) to 1.0 (perfectly stable).

    Factors:
    - Mode consistency (low transitions)
    - FULL_PIPELINE prevalence
    - Absence of critical patterns
    - Low failure counts
    """
    if not snapshots:
        return 0.0

    total_runs = len(snapshots)

    # Factor 1: Mode consistency (0.0-0.3)
    # 0 transitions = 0.3, many transitions = 0.0
    transition_rate = mode_evolution.mode_transitions / max(total_runs - 1, 1)
    mode_consistency = 0.3 * (1.0 - min(transition_rate, 1.0))

    # Factor 2: FULL_PIPELINE prevalence (0.0-0.3)
    full_rate = mode_evolution.full_pipeline_count / total_runs
    full_prevalence = 0.3 * full_rate

    # Factor 3: Absence of critical patterns (0.0-0.2)
    critical_count = sum(
        1 for s in snapshots
        for p in s.patterns_detected
        if is_critical_pattern(p)
    )
    critical_rate = critical_count / (total_runs * 5)  # Normalize assuming max 5 per run
    pattern_health = 0.2 * (1.0 - min(critical_rate, 1.0))

    # Factor 4: Low hard failure counts (0.0-0.2)
    avg_hard_fails = sum(s.hard_fail_count for s in snapshots) / total_runs
    fail_health = 0.2 * (1.0 - min(avg_hard_fails / 3, 1.0))  # Normalize assuming max 3

    stability = mode_consistency + full_prevalence + pattern_health + fail_health
    return round(min(max(stability, 0.0), 1.0), 3)


# =============================================================================
# Task 2: Failure Pattern Forecasting
# =============================================================================


def predict_degradation_risk(
    trajectory: HealthTrajectory,
) -> DegradationRiskPrediction:
    """
    Predict the degradation risk for the next run.

    Uses deterministic analysis of trajectory patterns to forecast
    the likely mode and risk level.

    Args:
        trajectory: The health trajectory to analyze

    Returns:
        DegradationRiskPrediction with mode prediction and risk factors
    """
    if not trajectory.trajectory:
        # No data - assume high risk
        return DegradationRiskPrediction(
            next_run_mode_prediction=PipelineMode.EVIDENCE_ONLY,
            risk_level=RiskLevel.HIGH,
            confidence=0.1,
            explanatory_patterns=(),
            risk_factors=("No historical data available",),
            protective_factors=(),
        )

    risk_factors: List[str] = []
    protective_factors: List[str] = []
    explanatory_patterns: List[str] = []

    # Get current state
    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution
    pattern_trends = trajectory.failure_pattern_trends

    # Analyze risk factors

    # Factor 1: Current mode
    if latest.mode == PipelineMode.EVIDENCE_ONLY:
        risk_factors.append("Currently in EVIDENCE_ONLY mode")
    elif latest.mode == PipelineMode.DEGRADED_ANALYSIS:
        risk_factors.append("Currently in DEGRADED_ANALYSIS mode")
    else:
        protective_factors.append("Currently in FULL_PIPELINE mode")

    # Factor 2: Mode stability
    if mode_evo.stability == ModeStability.DEGRADING:
        risk_factors.append("Mode trend is degrading")
    elif mode_evo.stability == ModeStability.VOLATILE:
        risk_factors.append("Mode is volatile (frequent changes)")
    elif mode_evo.stability == ModeStability.IMPROVING:
        protective_factors.append("Mode trend is improving")
    elif mode_evo.stability == ModeStability.STABLE:
        if latest.mode == PipelineMode.FULL_PIPELINE:
            protective_factors.append("Stable at FULL_PIPELINE")
        else:
            risk_factors.append(f"Stable at {latest.mode.value}")

    # Factor 3: Active critical patterns
    active_critical = [
        p for p in pattern_trends.active_patterns
        if is_critical_pattern(p)
    ]
    if active_critical:
        risk_factors.append(f"{len(active_critical)} active critical pattern(s)")
        explanatory_patterns.extend(active_critical)

    # Factor 4: Chronic patterns
    if pattern_trends.chronic_patterns:
        risk_factors.append(f"{len(pattern_trends.chronic_patterns)} chronic pattern(s)")
        for p in pattern_trends.chronic_patterns:
            if p not in explanatory_patterns:
                explanatory_patterns.append(p)

    # Factor 5: Current failure counts
    if latest.hard_fail_count > 0:
        risk_factors.append(f"{latest.hard_fail_count} hard failure(s) in latest run")
    elif latest.soft_fail_count == 0:
        protective_factors.append("No failures in latest run")

    # Factor 6: Current streak
    if mode_evo.current_streak >= 3:
        if latest.mode == PipelineMode.FULL_PIPELINE:
            protective_factors.append(f"{mode_evo.current_streak} consecutive FULL_PIPELINE runs")
        elif latest.mode == PipelineMode.EVIDENCE_ONLY:
            risk_factors.append(f"{mode_evo.current_streak} consecutive EVIDENCE_ONLY runs")

    # Factor 7: FULL_PIPELINE prevalence
    if mode_evo.total_runs >= 3:
        full_rate = mode_evo.full_pipeline_count / mode_evo.total_runs
        if full_rate >= 0.8:
            protective_factors.append(f"{int(full_rate * 100)}% FULL_PIPELINE rate")
        elif full_rate <= 0.2:
            risk_factors.append(f"Only {int(full_rate * 100)}% FULL_PIPELINE rate")

    # Calculate risk level and prediction
    risk_score = len(risk_factors) - len(protective_factors) * 0.5

    if risk_score >= 3:
        risk_level = RiskLevel.HIGH
    elif risk_score >= 1:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    # Predict next mode based on current state and trends
    predicted_mode = _predict_next_mode(trajectory, risk_level)

    # Calculate confidence based on data quality
    confidence = _calculate_prediction_confidence(trajectory, risk_level)

    return DegradationRiskPrediction(
        next_run_mode_prediction=predicted_mode,
        risk_level=risk_level,
        confidence=confidence,
        explanatory_patterns=tuple(sorted(explanatory_patterns)),
        risk_factors=tuple(risk_factors),
        protective_factors=tuple(protective_factors),
    )


def _predict_next_mode(
    trajectory: HealthTrajectory,
    risk_level: RiskLevel,
) -> PipelineMode:
    """Predict the most likely mode for the next run."""
    if not trajectory.trajectory:
        return PipelineMode.EVIDENCE_ONLY

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution

    # Simple heuristic based on current state and trend
    if mode_evo.stability == ModeStability.STABLE:
        # Likely to stay at current mode
        return latest.mode

    if mode_evo.stability == ModeStability.IMPROVING:
        # Might improve one step
        if latest.mode == PipelineMode.EVIDENCE_ONLY:
            return PipelineMode.DEGRADED_ANALYSIS
        elif latest.mode == PipelineMode.DEGRADED_ANALYSIS:
            return PipelineMode.FULL_PIPELINE
        return PipelineMode.FULL_PIPELINE

    if mode_evo.stability == ModeStability.DEGRADING:
        # Might degrade one step
        if latest.mode == PipelineMode.FULL_PIPELINE:
            return PipelineMode.DEGRADED_ANALYSIS
        return PipelineMode.EVIDENCE_ONLY

    # Volatile - predict based on majority
    if mode_evo.full_pipeline_count > mode_evo.evidence_only_count:
        return PipelineMode.FULL_PIPELINE
    elif mode_evo.evidence_only_count > mode_evo.full_pipeline_count:
        return PipelineMode.EVIDENCE_ONLY
    return PipelineMode.DEGRADED_ANALYSIS


def _calculate_prediction_confidence(
    trajectory: HealthTrajectory,
    risk_level: RiskLevel,
) -> float:
    """Calculate confidence in the prediction based on data quality."""
    if not trajectory.trajectory:
        return 0.1

    total_runs = len(trajectory.trajectory)
    mode_evo = trajectory.mode_evolution

    # Base confidence from data volume
    if total_runs >= 10:
        base_confidence = 0.7
    elif total_runs >= 5:
        base_confidence = 0.5
    elif total_runs >= 3:
        base_confidence = 0.3
    else:
        base_confidence = 0.2

    # Adjust for stability
    if mode_evo.stability == ModeStability.STABLE:
        base_confidence += 0.2
    elif mode_evo.stability == ModeStability.VOLATILE:
        base_confidence -= 0.2

    # Adjust for current streak
    streak_bonus = min(mode_evo.current_streak * 0.05, 0.15)
    base_confidence += streak_bonus

    return round(min(max(base_confidence, 0.1), 0.95), 2)


# =============================================================================
# Task 3: Governance Summary Extraction
# =============================================================================


def summarize_topology_for_global_health(
    trajectory: HealthTrajectory,
) -> GlobalHealthSummary:
    """
    Generate a high-level governance summary for the pipeline.

    Produces a summary suitable for executive dashboards and alerting.

    Args:
        trajectory: The health trajectory to summarize

    Returns:
        GlobalHealthSummary with key health indicators
    """
    if not trajectory.trajectory:
        return GlobalHealthSummary(
            pipeline_ok=False,
            mode_stability=ModeStability.VOLATILE,
            current_mode=PipelineMode.EVIDENCE_ONLY,
            current_label=GovernanceLabel.DO_NOT_USE,
            unresolved_critical_patterns=(),
            runs_since_last_full_pipeline=-1,
            stability_index=0.0,
            recommendation="No trajectory data available. Run pipeline to establish baseline.",
        )

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution
    pattern_trends = trajectory.failure_pattern_trends

    # Determine if pipeline is OK
    pipeline_ok = (
        latest.mode == PipelineMode.FULL_PIPELINE
        and latest.governance_label == GovernanceLabel.OK
        and not any(is_critical_pattern(p) for p in pattern_trends.active_patterns)
    )

    # Find unresolved critical patterns
    unresolved_critical = tuple(
        p for p in pattern_trends.active_patterns
        if is_critical_pattern(p)
    )

    # Calculate runs since last FULL_PIPELINE
    runs_since_full = 0
    for point in reversed(trajectory.trajectory):
        if point.mode == PipelineMode.FULL_PIPELINE:
            break
        runs_since_full += 1

    if runs_since_full == len(trajectory.trajectory):
        # Never achieved FULL_PIPELINE
        runs_since_full = -1

    # Generate recommendation
    recommendation = _generate_recommendation(
        latest, mode_evo, pattern_trends, unresolved_critical
    )

    return GlobalHealthSummary(
        pipeline_ok=pipeline_ok,
        mode_stability=mode_evo.stability,
        current_mode=latest.mode,
        current_label=latest.governance_label,
        unresolved_critical_patterns=unresolved_critical,
        runs_since_last_full_pipeline=runs_since_full,
        stability_index=trajectory.stability_index,
        recommendation=recommendation,
    )


def _generate_recommendation(
    latest: TrajectoryPoint,
    mode_evo: ModeEvolution,
    pattern_trends: FailurePatternTrends,
    unresolved_critical: Tuple[str, ...],
) -> str:
    """Generate a human-readable recommendation based on trajectory analysis."""
    if latest.mode == PipelineMode.FULL_PIPELINE and not unresolved_critical:
        if mode_evo.stability == ModeStability.STABLE:
            return "Pipeline healthy and stable. Continue monitoring."
        elif mode_evo.stability == ModeStability.IMPROVING:
            return "Pipeline health improving. Recent issues resolved."
        else:
            return "Pipeline currently healthy but historically volatile. Monitor closely."

    if latest.mode == PipelineMode.EVIDENCE_ONLY:
        if unresolved_critical:
            patterns_str = ", ".join(unresolved_critical[:3])
            return f"CRITICAL: Pipeline in EVIDENCE_ONLY mode. Address critical patterns: {patterns_str}"
        else:
            return "CRITICAL: Pipeline in EVIDENCE_ONLY mode. Investigate hard failures."

    if latest.mode == PipelineMode.DEGRADED_ANALYSIS:
        if mode_evo.stability == ModeStability.DEGRADING:
            return "WARNING: Pipeline degrading. Address soft failures before escalation."
        elif pattern_trends.chronic_patterns:
            chronic_str = ", ".join(pattern_trends.chronic_patterns[:3])
            return f"WARNING: Chronic patterns detected: {chronic_str}. Review root causes."
        else:
            return "WARNING: Pipeline partially degraded. Some slices failing."

    return "Review pipeline health trajectory for detailed analysis."


# =============================================================================
# Phase IV: Pipeline Degradation Guardrail & Director Outlook
# =============================================================================


class ReleaseStatus(Enum):
    """Release guardrail status classification."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class MaasStatus(Enum):
    """MAAS (Monitoring as a Service) status classification."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


class StatusLight(Enum):
    """Director panel status light (traffic light pattern)."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class ReleaseGuardrailResult:
    """
    Result of release guardrail evaluation.

    Determines whether a release should proceed based on
    pipeline topology health and degradation risk.
    """
    release_ok: bool
    status: ReleaseStatus
    blocking_reasons: Tuple[str, ...]
    watch_patterns: Tuple[str, ...]
    risk_level: RiskLevel
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "release_ok": self.release_ok,
            "status": self.status.value,
            "blocking_reasons": list(self.blocking_reasons),
            "watch_patterns": list(self.watch_patterns),
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class MaasHealthSignal:
    """
    MAAS (Monitoring as a Service) health signal.

    Provides a standardized health signal for external monitoring
    and alerting systems.
    """
    topology_ok_for_evidence: bool
    mode_stability: ModeStability
    unresolved_critical_patterns: Tuple[str, ...]
    status: MaasStatus
    current_mode: PipelineMode
    stability_index: float
    alert_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topology_ok_for_evidence": self.topology_ok_for_evidence,
            "mode_stability": self.mode_stability.value,
            "unresolved_critical_patterns": list(self.unresolved_critical_patterns),
            "status": self.status.value,
            "current_mode": self.current_mode.value,
            "stability_index": self.stability_index,
            "alert_summary": self.alert_summary,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class DirectorTopologyPanel:
    """
    Director-level topology forecast panel.

    Provides a high-level executive view of pipeline health
    for decision-makers and dashboards.
    """
    status_light: StatusLight
    current_mode: PipelineMode
    mode_stability: ModeStability
    risk_level: RiskLevel
    headline: str
    stability_index: float
    runs_analyzed: int
    next_run_prediction: PipelineMode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status_light": self.status_light.value,
            "current_mode": self.current_mode.value,
            "mode_stability": self.mode_stability.value,
            "risk_level": self.risk_level.value,
            "headline": self.headline,
            "stability_index": self.stability_index,
            "runs_analyzed": self.runs_analyzed,
            "next_run_prediction": self.next_run_prediction.value,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Task 1: Degradation Guardrail for Release
# =============================================================================


def evaluate_topology_for_release(
    trajectory: HealthTrajectory,
    prediction: DegradationRiskPrediction,
) -> ReleaseGuardrailResult:
    """
    Evaluate pipeline topology health for release decision.

    Determines whether a release should proceed based on the
    current trajectory state and degradation risk prediction.

    Rules:
    - BLOCK if risk_level == HIGH or current_mode == EVIDENCE_ONLY
    - WARN if risk_level == MEDIUM or ModeStability == DEGRADING/VOLATILE
    - OK otherwise

    Args:
        trajectory: The health trajectory to evaluate
        prediction: The degradation risk prediction

    Returns:
        ReleaseGuardrailResult with release decision and reasons
    """
    blocking_reasons: List[str] = []
    watch_patterns: List[str] = []

    # Handle empty trajectory
    if not trajectory.trajectory:
        return ReleaseGuardrailResult(
            release_ok=False,
            status=ReleaseStatus.BLOCK,
            blocking_reasons=("No trajectory data available for release evaluation",),
            watch_patterns=(),
            risk_level=RiskLevel.HIGH,
            confidence=0.0,
        )

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution
    pattern_trends = trajectory.failure_pattern_trends

    # Determine blocking conditions
    is_blocked = False

    # Rule 1: Block if risk_level == HIGH
    if prediction.risk_level == RiskLevel.HIGH:
        is_blocked = True
        blocking_reasons.append(
            f"High degradation risk detected (confidence: {prediction.confidence:.0%})"
        )
        # Add risk factors as context
        for factor in prediction.risk_factors[:3]:
            blocking_reasons.append(f"  - {factor}")

    # Rule 2: Block if current_mode == EVIDENCE_ONLY
    if latest.mode == PipelineMode.EVIDENCE_ONLY:
        is_blocked = True
        blocking_reasons.append(
            "Pipeline currently in EVIDENCE_ONLY mode - no Delta-p computation permitted"
        )

    # Rule 3: Block if unresolved critical patterns
    critical_patterns = [
        p for p in pattern_trends.active_patterns
        if is_critical_pattern(p)
    ]
    if critical_patterns:
        is_blocked = True
        patterns_str = ", ".join(critical_patterns[:3])
        blocking_reasons.append(f"Unresolved critical patterns: {patterns_str}")

    # Determine warning conditions (if not blocked)
    is_warned = False

    if not is_blocked:
        # Rule 4: Warn if risk_level == MEDIUM
        if prediction.risk_level == RiskLevel.MEDIUM:
            is_warned = True
            blocking_reasons.append("Medium degradation risk - proceed with caution")

        # Rule 5: Warn if ModeStability == DEGRADING or VOLATILE
        if mode_evo.stability in (ModeStability.DEGRADING, ModeStability.VOLATILE):
            is_warned = True
            blocking_reasons.append(
                f"Mode stability is {mode_evo.stability.value} - pipeline health unstable"
            )

        # Rule 6: Warn if recent failures (but not blocked)
        if latest.hard_fail_count > 0 or latest.soft_fail_count > 0:
            is_warned = True
            blocking_reasons.append(
                f"Recent failures: {latest.hard_fail_count} hard, "
                f"{latest.soft_fail_count} soft"
            )

    # Collect watch patterns (patterns to monitor even if release proceeds)
    # Include all active patterns plus explanatory patterns from prediction
    watch_set: Set[str] = set(pattern_trends.active_patterns)
    watch_set.update(prediction.explanatory_patterns)
    watch_set.update(pattern_trends.chronic_patterns)
    watch_patterns = list(sorted(watch_set))

    # Determine final status
    if is_blocked:
        status = ReleaseStatus.BLOCK
        release_ok = False
    elif is_warned:
        status = ReleaseStatus.WARN
        release_ok = True  # Can proceed with caution
    else:
        status = ReleaseStatus.OK
        release_ok = True

    return ReleaseGuardrailResult(
        release_ok=release_ok,
        status=status,
        blocking_reasons=tuple(blocking_reasons),
        watch_patterns=tuple(watch_patterns),
        risk_level=prediction.risk_level,
        confidence=prediction.confidence,
    )


# =============================================================================
# Task 2: MAAS / Global Health Adapter
# =============================================================================


def summarize_topology_for_maas(
    trajectory: HealthTrajectory,
    release_eval: ReleaseGuardrailResult,
) -> MaasHealthSignal:
    """
    Generate a MAAS (Monitoring as a Service) health signal.

    Provides a standardized signal for external monitoring systems
    and alerting infrastructure.

    Args:
        trajectory: The health trajectory
        release_eval: The release guardrail evaluation result

    Returns:
        MaasHealthSignal for external monitoring integration
    """
    # Handle empty trajectory
    if not trajectory.trajectory:
        return MaasHealthSignal(
            topology_ok_for_evidence=False,
            mode_stability=ModeStability.VOLATILE,
            unresolved_critical_patterns=(),
            status=MaasStatus.BLOCK,
            current_mode=PipelineMode.EVIDENCE_ONLY,
            stability_index=0.0,
            alert_summary="No trajectory data - pipeline health unknown",
        )

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution
    pattern_trends = trajectory.failure_pattern_trends

    # Determine if topology is OK for evidence collection
    # (even EVIDENCE_ONLY mode can collect evidence, just not compute Delta-p)
    topology_ok_for_evidence = (
        latest.mode != PipelineMode.EVIDENCE_ONLY
        or mode_evo.stability != ModeStability.VOLATILE
    )

    # Collect unresolved critical patterns
    unresolved_critical = tuple(
        p for p in pattern_trends.active_patterns
        if is_critical_pattern(p)
    )

    # Map release status to MAAS status
    if release_eval.status == ReleaseStatus.BLOCK:
        maas_status = MaasStatus.BLOCK
    elif release_eval.status == ReleaseStatus.WARN:
        maas_status = MaasStatus.ATTENTION
    else:
        maas_status = MaasStatus.OK

    # Generate alert summary
    alert_summary = _generate_maas_alert_summary(
        latest, mode_evo, release_eval, unresolved_critical
    )

    return MaasHealthSignal(
        topology_ok_for_evidence=topology_ok_for_evidence,
        mode_stability=mode_evo.stability,
        unresolved_critical_patterns=unresolved_critical,
        status=maas_status,
        current_mode=latest.mode,
        stability_index=trajectory.stability_index,
        alert_summary=alert_summary,
    )


def _generate_maas_alert_summary(
    latest: TrajectoryPoint,
    mode_evo: ModeEvolution,
    release_eval: ReleaseGuardrailResult,
    unresolved_critical: Tuple[str, ...],
) -> str:
    """Generate a concise alert summary for MAAS."""
    if release_eval.status == ReleaseStatus.BLOCK:
        if unresolved_critical:
            return f"BLOCK: {len(unresolved_critical)} critical pattern(s) unresolved"
        elif latest.mode == PipelineMode.EVIDENCE_ONLY:
            return "BLOCK: Pipeline in EVIDENCE_ONLY mode"
        else:
            return "BLOCK: High degradation risk"

    if release_eval.status == ReleaseStatus.WARN:
        if mode_evo.stability == ModeStability.DEGRADING:
            return "ATTENTION: Pipeline health degrading"
        elif mode_evo.stability == ModeStability.VOLATILE:
            return "ATTENTION: Pipeline health volatile"
        else:
            return "ATTENTION: Medium risk detected"

    # OK status
    if mode_evo.stability == ModeStability.IMPROVING:
        return "OK: Pipeline health improving"
    elif mode_evo.stability == ModeStability.STABLE:
        return "OK: Pipeline health stable"
    else:
        return "OK: Pipeline healthy"


# =============================================================================
# Task 3: Director Topology Forecast Panel
# =============================================================================


def build_topology_director_panel(
    trajectory: HealthTrajectory,
    prediction: DegradationRiskPrediction,
    release_eval: ReleaseGuardrailResult,
) -> DirectorTopologyPanel:
    """
    Build a Director-level topology forecast panel.

    Provides a high-level executive view suitable for dashboards
    and decision-maker briefings.

    Args:
        trajectory: The health trajectory
        prediction: The degradation risk prediction
        release_eval: The release guardrail evaluation result

    Returns:
        DirectorTopologyPanel with executive summary
    """
    # Handle empty trajectory
    if not trajectory.trajectory:
        return DirectorTopologyPanel(
            status_light=StatusLight.RED,
            current_mode=PipelineMode.EVIDENCE_ONLY,
            mode_stability=ModeStability.VOLATILE,
            risk_level=RiskLevel.HIGH,
            headline="No pipeline data available. Establish baseline before proceeding.",
            stability_index=0.0,
            runs_analyzed=0,
            next_run_prediction=PipelineMode.EVIDENCE_ONLY,
        )

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution

    # Determine status light (traffic light pattern)
    status_light = _determine_status_light(release_eval, mode_evo, prediction)

    # Generate headline
    headline = _generate_director_headline(
        latest, mode_evo, prediction, release_eval, trajectory
    )

    return DirectorTopologyPanel(
        status_light=status_light,
        current_mode=latest.mode,
        mode_stability=mode_evo.stability,
        risk_level=prediction.risk_level,
        headline=headline,
        stability_index=trajectory.stability_index,
        runs_analyzed=mode_evo.total_runs,
        next_run_prediction=prediction.next_run_mode_prediction,
    )


def _determine_status_light(
    release_eval: ReleaseGuardrailResult,
    mode_evo: ModeEvolution,
    prediction: DegradationRiskPrediction,
) -> StatusLight:
    """Determine the traffic light status for the director panel."""
    # RED: Blocked or critical issues
    if release_eval.status == ReleaseStatus.BLOCK:
        return StatusLight.RED

    # RED: HIGH risk even if not blocked yet
    if prediction.risk_level == RiskLevel.HIGH:
        return StatusLight.RED

    # YELLOW: Warnings or medium risk
    if release_eval.status == ReleaseStatus.WARN:
        return StatusLight.YELLOW

    if prediction.risk_level == RiskLevel.MEDIUM:
        return StatusLight.YELLOW

    # YELLOW: Unstable mode trends
    if mode_evo.stability in (ModeStability.DEGRADING, ModeStability.VOLATILE):
        return StatusLight.YELLOW

    # GREEN: All clear
    return StatusLight.GREEN


def _generate_director_headline(
    latest: TrajectoryPoint,
    mode_evo: ModeEvolution,
    prediction: DegradationRiskPrediction,
    release_eval: ReleaseGuardrailResult,
    trajectory: HealthTrajectory,
) -> str:
    """Generate a neutral headline for the director panel."""
    runs = mode_evo.total_runs

    # Critical state headlines
    if release_eval.status == ReleaseStatus.BLOCK:
        if latest.mode == PipelineMode.EVIDENCE_ONLY:
            return (
                f"Pipeline blocked: EVIDENCE_ONLY mode. "
                f"{mode_evo.current_streak} consecutive run(s) at this state."
            )
        else:
            return (
                f"Pipeline blocked: {len(release_eval.blocking_reasons)} issue(s) detected. "
                f"Risk level: {prediction.risk_level.value}."
            )

    # Warning state headlines
    if release_eval.status == ReleaseStatus.WARN:
        if mode_evo.stability == ModeStability.DEGRADING:
            return (
                f"Pipeline health degrading over {runs} run(s). "
                f"Currently at {latest.mode.value}. Monitor closely."
            )
        elif mode_evo.stability == ModeStability.VOLATILE:
            return (
                f"Pipeline health volatile: {mode_evo.mode_transitions} transition(s) "
                f"over {runs} run(s). Stability index: {trajectory.stability_index:.1%}."
            )
        else:
            return (
                f"Pipeline at {latest.mode.value} with medium risk. "
                f"{len(release_eval.watch_patterns)} pattern(s) to monitor."
            )

    # OK state headlines
    if mode_evo.stability == ModeStability.IMPROVING:
        return (
            f"Pipeline health improving. {mode_evo.current_streak} consecutive "
            f"{latest.mode.value} run(s). Stability index: {trajectory.stability_index:.1%}."
        )

    if mode_evo.stability == ModeStability.STABLE:
        full_rate = mode_evo.full_pipeline_count / max(runs, 1) * 100
        return (
            f"Pipeline stable at {latest.mode.value}. "
            f"{full_rate:.0f}% FULL_PIPELINE rate over {runs} run(s)."
        )

    # Default
    return (
        f"Pipeline at {latest.mode.value}. "
        f"Risk: {prediction.risk_level.value}. "
        f"Stability: {trajectory.stability_index:.1%}."
    )


# =============================================================================
# Phase V: Topology × Bundle × DAG Unification
# =============================================================================

# Schema version for Phase V cross-system integration
CROSS_SYSTEM_SCHEMA_VERSION = "1.0.0"


class JointStatus(Enum):
    """Joint status for cross-system integration views."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class ConsistencyStatus(Enum):
    """Consistency status for cross-system posture checks."""
    CONSISTENT = "CONSISTENT"
    TENSION = "TENSION"
    CONFLICT = "CONFLICT"


# =============================================================================
# Task 1: Bundle + Topology Joint View
# =============================================================================


def build_topology_bundle_joint_view(
    trajectory: HealthTrajectory,
    bundle_evolution: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a joint view combining topology health and bundle evolution.

    Provides a unified perspective for integration decisions by combining
    topology stability with bundle integration status.

    Args:
        trajectory: The health trajectory from topology analysis
        bundle_evolution: Bundle evolution ledger data containing:
            - stability_rating: str ("STABLE", "UNSTABLE", "UNKNOWN")
            - integration_status: str ("OK", "WARN", "BLOCK")
            - reasons: Optional[List[str]] - reasons for status

    Returns:
        Dictionary containing:
        - topology_ok_for_integration: bool
        - topology_stability: ModeStability value
        - bundle_stability_rating: str from bundle evolution
        - joint_status: "OK" | "WARN" | "BLOCK"
        - reasons: list of neutral strings combining both perspectives
        - schema_version: version string for format compatibility
    """
    reasons: List[str] = []

    # Extract topology status
    if not trajectory.trajectory:
        topology_ok = False
        topology_stability = ModeStability.VOLATILE
        reasons.append("Topology: No trajectory data available")
    else:
        latest = trajectory.trajectory[-1]
        mode_evo = trajectory.mode_evolution

        # Topology is OK for integration if not in EVIDENCE_ONLY and stable
        topology_ok = (
            latest.mode != PipelineMode.EVIDENCE_ONLY
            and mode_evo.stability not in (ModeStability.DEGRADING, ModeStability.VOLATILE)
        )
        topology_stability = mode_evo.stability

        if not topology_ok:
            if latest.mode == PipelineMode.EVIDENCE_ONLY:
                reasons.append(
                    f"Topology: Pipeline in EVIDENCE_ONLY mode "
                    f"(stability: {topology_stability.value})"
                )
            elif mode_evo.stability in (ModeStability.DEGRADING, ModeStability.VOLATILE):
                reasons.append(
                    f"Topology: Mode stability is {topology_stability.value}"
                )

    # Extract bundle status
    bundle_stability = bundle_evolution.get("stability_rating", "UNKNOWN")
    bundle_integration = bundle_evolution.get("integration_status", "UNKNOWN")
    bundle_reasons = bundle_evolution.get("reasons", [])

    # Determine if bundle blocks integration
    bundle_blocks = bundle_integration == "BLOCK"
    bundle_warns = bundle_integration == "WARN"

    if bundle_blocks:
        reasons.append(f"Bundle: Integration status is BLOCK")
        for br in bundle_reasons[:2]:
            reasons.append(f"  - {br}")
    elif bundle_warns:
        reasons.append(f"Bundle: Integration status is WARN")
        for br in bundle_reasons[:2]:
            reasons.append(f"  - {br}")
    elif bundle_stability == "UNSTABLE":
        reasons.append(f"Bundle: Stability rating is UNSTABLE")

    # Determine joint status
    # BLOCK if: topology release would BLOCK OR bundle integration is BLOCK
    topology_would_block = (
        not topology_ok
        and trajectory.trajectory
        and trajectory.trajectory[-1].mode == PipelineMode.EVIDENCE_ONLY
    )

    if topology_would_block or bundle_blocks:
        joint_status = JointStatus.BLOCK
    elif not topology_ok or bundle_warns or bundle_stability == "UNSTABLE":
        joint_status = JointStatus.WARN
    else:
        joint_status = JointStatus.OK

    # Add positive indicators if all clear
    if not reasons:
        reasons.append("Topology: Pipeline health stable for integration")
        reasons.append(f"Bundle: Stability rating is {bundle_stability}")

    return {
        "topology_ok_for_integration": topology_ok,
        "topology_stability": topology_stability.value,
        "bundle_stability_rating": bundle_stability,
        "joint_status": joint_status.value,
        "reasons": reasons,
        "schema_version": CROSS_SYSTEM_SCHEMA_VERSION,
    }


# =============================================================================
# Task 2: Global Console Adapter
# =============================================================================


def summarize_topology_for_global_console(
    trajectory: HealthTrajectory,
    release_eval: ReleaseGuardrailResult,
) -> Dict[str, Any]:
    """
    Generate a summary for the global health console/dashboard.

    Provides a standardized object for global health dashboard ingestion
    with key topology metrics and status indicators.

    Args:
        trajectory: The health trajectory
        release_eval: The release guardrail evaluation result

    Returns:
        Dictionary containing:
        - topology_ok: bool
        - status_light: StatusLight value
        - stability_index: float (0.0-1.0)
        - mode_stability: ModeStability value
        - alert_summary: str describing current state
        - current_mode: PipelineMode value
        - runs_analyzed: int
        - schema_version: version string
    """
    # Handle empty trajectory
    if not trajectory.trajectory:
        return {
            "topology_ok": False,
            "status_light": StatusLight.RED.value,
            "stability_index": 0.0,
            "mode_stability": ModeStability.VOLATILE.value,
            "alert_summary": "No trajectory data - pipeline health unknown",
            "current_mode": PipelineMode.EVIDENCE_ONLY.value,
            "runs_analyzed": 0,
            "schema_version": CROSS_SYSTEM_SCHEMA_VERSION,
        }

    latest = trajectory.trajectory[-1]
    mode_evo = trajectory.mode_evolution

    # Determine topology_ok (can proceed with operations)
    topology_ok = release_eval.release_ok

    # Determine status light
    if release_eval.status == ReleaseStatus.BLOCK:
        status_light = StatusLight.RED
    elif release_eval.status == ReleaseStatus.WARN:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Generate alert summary
    alert_summary = _generate_console_alert_summary(
        latest, mode_evo, release_eval, trajectory.stability_index
    )

    return {
        "topology_ok": topology_ok,
        "status_light": status_light.value,
        "stability_index": trajectory.stability_index,
        "mode_stability": mode_evo.stability.value,
        "alert_summary": alert_summary,
        "current_mode": latest.mode.value,
        "runs_analyzed": mode_evo.total_runs,
        "schema_version": CROSS_SYSTEM_SCHEMA_VERSION,
    }


def _generate_console_alert_summary(
    latest: TrajectoryPoint,
    mode_evo: ModeEvolution,
    release_eval: ReleaseGuardrailResult,
    stability_index: float,
) -> str:
    """Generate a concise alert summary for the global console."""
    if release_eval.status == ReleaseStatus.BLOCK:
        if latest.mode == PipelineMode.EVIDENCE_ONLY:
            return f"BLOCKED: EVIDENCE_ONLY mode ({mode_evo.current_streak} consecutive)"
        else:
            return f"BLOCKED: {len(release_eval.blocking_reasons)} issue(s) detected"

    if release_eval.status == ReleaseStatus.WARN:
        return f"WARNING: {latest.mode.value} with {mode_evo.stability.value} stability"

    # OK status
    return f"OK: {latest.mode.value} (stability: {stability_index:.0%})"


# =============================================================================
# Task 3: Cross-Check with DAG Posture
# =============================================================================


def check_consistency_with_dag_posture(
    trajectory: HealthTrajectory,
    dag_global_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check consistency between topology health and DAG posture signals.

    Detects cases where topology and DAG posture disagree, which may
    indicate systemic issues or measurement drift.

    Args:
        trajectory: The health trajectory from topology analysis
        dag_global_health: DAG global health data containing:
            - drift_status: str ("OK", "WARN", "BLOCKED")
            - reasons: Optional[List[str]]

    Returns:
        Dictionary containing:
        - consistency_status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - topology_status: str (derived OK/WARN/BLOCK)
        - dag_status: str (from dag_global_health)
        - reasons: list of explanatory strings
        - schema_version: version string
    """
    reasons: List[str] = []

    # Determine topology status
    if not trajectory.trajectory:
        topology_status = "BLOCK"
        reasons.append("Topology: No data available (treated as BLOCK)")
    else:
        latest = trajectory.trajectory[-1]
        mode_evo = trajectory.mode_evolution

        if latest.mode == PipelineMode.EVIDENCE_ONLY:
            topology_status = "BLOCK"
        elif mode_evo.stability in (ModeStability.DEGRADING, ModeStability.VOLATILE):
            topology_status = "WARN"
        elif latest.mode == PipelineMode.DEGRADED_ANALYSIS:
            topology_status = "WARN"
        else:
            topology_status = "OK"

    # Extract DAG status (normalize BLOCKED -> BLOCK for comparison)
    dag_status_raw = dag_global_health.get("drift_status", "UNKNOWN")
    dag_status = "BLOCK" if dag_status_raw == "BLOCKED" else dag_status_raw
    dag_reasons = dag_global_health.get("reasons", [])

    # Determine consistency
    consistency_status = _determine_consistency(topology_status, dag_status)

    # Generate explanatory reasons
    if consistency_status == ConsistencyStatus.CONFLICT:
        if topology_status == "OK" and dag_status == "BLOCK":
            reasons.append(
                "Topology indicates OK but DAG posture is BLOCK - "
                "DAG structural issues may not be reflected in pipeline mode"
            )
        elif topology_status == "BLOCK" and dag_status == "OK":
            reasons.append(
                "Topology indicates BLOCK but DAG posture is OK - "
                "pipeline issues may not be structural"
            )
        for dr in dag_reasons[:2]:
            reasons.append(f"DAG: {dr}")

    elif consistency_status == ConsistencyStatus.TENSION:
        if topology_status == "OK" and dag_status == "WARN":
            reasons.append(
                "Topology is OK but DAG posture has warnings - monitor for drift"
            )
        elif topology_status == "WARN" and dag_status == "OK":
            reasons.append(
                "Topology has warnings but DAG posture is OK - "
                "pipeline instability may not affect structure"
            )
        elif topology_status == "BLOCK" and dag_status == "WARN":
            reasons.append(
                "Topology is BLOCK with DAG warnings - multiple concerns present"
            )
        elif topology_status == "WARN" and dag_status == "BLOCK":
            reasons.append(
                "Topology has warnings but DAG is BLOCK - "
                "structural issues may be primary cause"
            )

    else:  # CONSISTENT
        if topology_status == "OK" and dag_status == "OK":
            reasons.append("Both topology and DAG posture indicate healthy state")
        elif topology_status == "WARN" and dag_status == "WARN":
            reasons.append("Both topology and DAG posture indicate warnings")
        elif topology_status == "BLOCK" and dag_status == "BLOCK":
            reasons.append("Both topology and DAG posture indicate blocked state")
        else:
            reasons.append(f"Topology: {topology_status}, DAG: {dag_status}")

    return {
        "consistency_status": consistency_status.value,
        "topology_status": topology_status,
        "dag_status": dag_status,
        "reasons": reasons,
        "schema_version": CROSS_SYSTEM_SCHEMA_VERSION,
    }


def _determine_consistency(
    topology_status: str,
    dag_status: str,
) -> ConsistencyStatus:
    """
    Determine consistency status between topology and DAG signals.

    CONSISTENT: Both agree (OK/OK, WARN/WARN, BLOCK/BLOCK)
    TENSION: Minor disagreement (OK/WARN, WARN/OK, BLOCK/WARN, WARN/BLOCK)
    CONFLICT: Major disagreement (OK/BLOCK, BLOCK/OK)
    """
    # Normalize unknown to WARN for comparison
    if topology_status == "UNKNOWN":
        topology_status = "WARN"
    if dag_status == "UNKNOWN":
        dag_status = "WARN"

    # Check for conflict (OK vs BLOCK in either direction)
    if (topology_status == "OK" and dag_status == "BLOCK") or \
       (topology_status == "BLOCK" and dag_status == "OK"):
        return ConsistencyStatus.CONFLICT

    # Check for consistent (same status)
    if topology_status == dag_status:
        return ConsistencyStatus.CONSISTENT

    # All other cases are tension
    return ConsistencyStatus.TENSION


# =============================================================================
# Task 4: Extended Director Panel with Cross-System Consistency
# =============================================================================


@dataclass(frozen=True)
class DirectorTopologyPanelExtended:
    """
    Extended Director-level topology forecast panel with cross-system consistency.

    Includes all fields from DirectorTopologyPanel plus optional
    cross-system consistency indicator from DAG posture checks.
    """
    status_light: StatusLight
    current_mode: PipelineMode
    mode_stability: ModeStability
    risk_level: RiskLevel
    headline: str
    stability_index: float
    runs_analyzed: int
    next_run_prediction: PipelineMode
    cross_system_consistency: Optional[str]  # "CONSISTENT" | "TENSION" | "CONFLICT" | None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status_light": self.status_light.value,
            "current_mode": self.current_mode.value,
            "mode_stability": self.mode_stability.value,
            "risk_level": self.risk_level.value,
            "headline": self.headline,
            "stability_index": self.stability_index,
            "runs_analyzed": self.runs_analyzed,
            "next_run_prediction": self.next_run_prediction.value,
            "cross_system_consistency": self.cross_system_consistency,
            "schema_version": CROSS_SYSTEM_SCHEMA_VERSION,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def build_topology_director_panel_extended(
    trajectory: HealthTrajectory,
    prediction: DegradationRiskPrediction,
    release_eval: ReleaseGuardrailResult,
    dag_global_health: Optional[Dict[str, Any]] = None,
) -> DirectorTopologyPanelExtended:
    """
    Build an extended Director-level topology forecast panel.

    Includes cross-system consistency check if DAG health data is provided.
    Traffic light logic remains unchanged from base panel.

    Args:
        trajectory: The health trajectory
        prediction: The degradation risk prediction
        release_eval: The release guardrail evaluation result
        dag_global_health: Optional DAG global health data for consistency check

    Returns:
        DirectorTopologyPanelExtended with cross-system consistency
    """
    # Get base panel data
    base_panel = build_topology_director_panel(trajectory, prediction, release_eval)

    # Determine cross-system consistency if DAG health provided
    cross_system_consistency: Optional[str] = None
    if dag_global_health is not None:
        consistency_result = check_consistency_with_dag_posture(
            trajectory, dag_global_health
        )
        cross_system_consistency = consistency_result["consistency_status"]

    return DirectorTopologyPanelExtended(
        status_light=base_panel.status_light,
        current_mode=base_panel.current_mode,
        mode_stability=base_panel.mode_stability,
        risk_level=base_panel.risk_level,
        headline=base_panel.headline,
        stability_index=base_panel.stability_index,
        runs_analyzed=base_panel.runs_analyzed,
        next_run_prediction=base_panel.next_run_prediction,
        cross_system_consistency=cross_system_consistency,
    )


# =============================================================================
# Phase VI: Topology/Bundle as a Coherent Governance Layer
# =============================================================================
#
# INTEGRATION WITH CLAUDE I's build_global_alignment_view
# --------------------------------------------------------
#
# This layer provides unified topology+bundle governance signals that feed into
# the meta-governance synthesizer (CLAUDE I). The key integration points are:
#
# 1. summarize_topology_bundle_for_global_console() returns a dashboard tile
#    suitable for the global health console. This should be called after
#    build_topology_bundle_joint_view() and build_topology_director_panel_extended().
#
# 2. to_governance_signal_for_topology_bundle() returns a GovernanceSignal-like
#    dict that CLAUDE I's build_global_alignment_view() can consume directly.
#    The signal includes:
#    - status: "OK"|"WARN"|"BLOCK" (maps from JointStatus)
#    - blocking_rules: list of rule codes that triggered blocking/warning
#    - blocking_rate: derived from stability_index and risk assessment
#
# Expected call flow in build_global_alignment_view:
#   1. Build trajectory: build_pipeline_health_trajectory(snapshots)
#   2. Build prediction: predict_degradation_risk(trajectory)
#   3. Build release eval: evaluate_topology_for_release(trajectory, prediction)
#   4. Build joint view: build_topology_bundle_joint_view(trajectory, bundle_evo)
#   5. Build panel: build_topology_director_panel_extended(...)
#   6. Get signal: to_governance_signal_for_topology_bundle(joint_view)
#   7. Merge signal into global alignment view
#
# The GovernanceSignal is designed to be compatible with other governance
# layers (DAG posture, metric conformance, etc.) for unified dashboard display.
#

# Schema version for Phase VI governance layer
GOVERNANCE_SIGNAL_SCHEMA_VERSION = "1.0.0"


# Blocking rule codes for topology/bundle governance
class TopologyBundleBlockingRule:
    """Blocking rule codes for topology/bundle governance layer."""
    TOPOLOGY_EVIDENCE_ONLY = "TOPOLOGY_EVIDENCE_ONLY"
    TOPOLOGY_DEGRADING = "TOPOLOGY_DEGRADING"
    TOPOLOGY_VOLATILE = "TOPOLOGY_VOLATILE"
    TOPOLOGY_NO_DATA = "TOPOLOGY_NO_DATA"
    BUNDLE_INTEGRATION_BLOCKED = "BUNDLE_INTEGRATION_BLOCKED"
    BUNDLE_INTEGRATION_WARN = "BUNDLE_INTEGRATION_WARN"
    BUNDLE_UNSTABLE = "BUNDLE_UNSTABLE"
    CROSS_SYSTEM_CONFLICT = "CROSS_SYSTEM_CONFLICT"
    CROSS_SYSTEM_TENSION = "CROSS_SYSTEM_TENSION"


# =============================================================================
# Task 1: Global Console Adapter
# =============================================================================


def summarize_topology_bundle_for_global_console(
    joint_view: Dict[str, Any],
    director_panel: DirectorTopologyPanelExtended,
) -> Dict[str, Any]:
    """
    Generate a global console tile for topology+bundle unified view.

    Provides a single dashboard tile that summarizes the combined
    topology and bundle health status for the global console.

    Args:
        joint_view: Result from build_topology_bundle_joint_view()
        director_panel: Result from build_topology_director_panel_extended()

    Returns:
        Dictionary containing:
        - topology_bundle_ok: bool - overall health status
        - status_light: "GREEN"|"YELLOW"|"RED" - traffic light indicator
        - headline: str - neutral, single-sentence summary
        - consistency_status: str - cross-system consistency
        - stability_index: float - overall stability (0.0-1.0)
        - topology_mode: str - current pipeline mode
        - bundle_stability: str - bundle stability rating
        - runs_analyzed: int - number of runs in trajectory
        - schema_version: str - format version
    """
    # Extract key values
    joint_status = joint_view.get("joint_status", "BLOCK")
    topology_ok = joint_view.get("topology_ok_for_integration", False)
    topology_stability = joint_view.get("topology_stability", "VOLATILE")
    bundle_stability = joint_view.get("bundle_stability_rating", "UNKNOWN")
    consistency = director_panel.cross_system_consistency or "UNKNOWN"

    # Determine overall OK status
    topology_bundle_ok = joint_status == "OK"

    # Determine status light
    if joint_status == "BLOCK":
        status_light = StatusLight.RED
    elif joint_status == "WARN":
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Override to RED if there's a cross-system conflict
    if consistency == "CONFLICT":
        status_light = StatusLight.RED
        topology_bundle_ok = False

    # Generate headline
    headline = _generate_topology_bundle_headline(
        joint_status, topology_ok, topology_stability,
        bundle_stability, consistency, director_panel
    )

    return {
        "topology_bundle_ok": topology_bundle_ok,
        "status_light": status_light.value,
        "headline": headline,
        "consistency_status": consistency,
        "stability_index": director_panel.stability_index,
        "topology_mode": director_panel.current_mode.value,
        "bundle_stability": bundle_stability,
        "runs_analyzed": director_panel.runs_analyzed,
        "risk_level": director_panel.risk_level.value,
        "schema_version": GOVERNANCE_SIGNAL_SCHEMA_VERSION,
    }


def _generate_topology_bundle_headline(
    joint_status: str,
    topology_ok: bool,
    topology_stability: str,
    bundle_stability: str,
    consistency: str,
    director_panel: DirectorTopologyPanelExtended,
) -> str:
    """Generate a neutral, single-sentence headline for the console tile."""
    # Conflict state - highest priority
    if consistency == "CONFLICT":
        return (
            f"Cross-system conflict detected: topology and DAG posture disagree. "
            f"Review required."
        )

    # Blocked state
    if joint_status == "BLOCK":
        if not topology_ok and director_panel.current_mode == PipelineMode.EVIDENCE_ONLY:
            return (
                f"Integration blocked: topology in EVIDENCE_ONLY mode, "
                f"bundle stability is {bundle_stability}."
            )
        elif bundle_stability == "UNSTABLE":
            return (
                f"Integration blocked: bundle stability is UNSTABLE, "
                f"topology at {director_panel.current_mode.value}."
            )
        else:
            return (
                f"Integration blocked: joint status is BLOCK. "
                f"Review topology ({director_panel.current_mode.value}) and bundle ({bundle_stability})."
            )

    # Warning state
    if joint_status == "WARN":
        if consistency == "TENSION":
            return (
                f"Cross-system tension: topology and DAG posture have minor disagreement. "
                f"Bundle is {bundle_stability}."
            )
        elif topology_stability in ("DEGRADING", "VOLATILE"):
            return (
                f"Topology stability is {topology_stability}, bundle is {bundle_stability}. "
                f"Monitor closely."
            )
        else:
            return (
                f"Integration warnings present: topology at {director_panel.current_mode.value}, "
                f"bundle is {bundle_stability}."
            )

    # OK state
    if consistency == "CONSISTENT":
        return (
            f"Topology and bundle healthy: {director_panel.current_mode.value} mode, "
            f"{bundle_stability} bundle, systems consistent."
        )
    else:
        return (
            f"Topology and bundle operational: {director_panel.current_mode.value} mode, "
            f"{bundle_stability} bundle stability."
        )


# =============================================================================
# Task 2: GovernanceSignal Adapter for CLAUDE I
# =============================================================================


def to_governance_signal_for_topology_bundle(
    joint_view: Dict[str, Any],
    consistency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert topology+bundle joint view to a GovernanceSignal for CLAUDE I.

    Produces a signal compatible with build_global_alignment_view() that
    can be merged with other governance layer signals (DAG posture,
    metric conformance, etc.).

    Args:
        joint_view: Result from build_topology_bundle_joint_view()
        consistency_result: Optional result from check_consistency_with_dag_posture()

    Returns:
        GovernanceSignal-like dictionary containing:
        - layer: "topology_bundle" - identifies this governance layer
        - status: "OK"|"WARN"|"BLOCK" - overall status
        - blocking_rules: list of rule codes that triggered blocking/warning
        - blocking_rate: float - derived from stability/risk (0.0-1.0)
        - reasons: list of explanatory strings
        - metrics: dict of key metrics for this layer
        - schema_version: str - format version
    """
    # Extract values from joint view
    joint_status = joint_view.get("joint_status", "BLOCK")
    topology_ok = joint_view.get("topology_ok_for_integration", False)
    topology_stability = joint_view.get("topology_stability", "VOLATILE")
    bundle_stability = joint_view.get("bundle_stability_rating", "UNKNOWN")
    bundle_integration = joint_view.get("bundle_integration_status", "UNKNOWN")
    reasons = list(joint_view.get("reasons", []))

    # Build blocking rules list
    blocking_rules: List[str] = []

    # Topology rules
    if not topology_ok:
        if "EVIDENCE_ONLY" in str(reasons):
            blocking_rules.append(TopologyBundleBlockingRule.TOPOLOGY_EVIDENCE_ONLY)
        if topology_stability == "DEGRADING":
            blocking_rules.append(TopologyBundleBlockingRule.TOPOLOGY_DEGRADING)
        elif topology_stability == "VOLATILE":
            blocking_rules.append(TopologyBundleBlockingRule.TOPOLOGY_VOLATILE)

    # Check for no data condition
    if any("No trajectory data" in r for r in reasons):
        blocking_rules.append(TopologyBundleBlockingRule.TOPOLOGY_NO_DATA)

    # Bundle rules
    if bundle_integration == "BLOCK" or any("Bundle" in r and "BLOCK" in r for r in reasons):
        blocking_rules.append(TopologyBundleBlockingRule.BUNDLE_INTEGRATION_BLOCKED)
    elif bundle_integration == "WARN" or any("Bundle" in r and "WARN" in r for r in reasons):
        blocking_rules.append(TopologyBundleBlockingRule.BUNDLE_INTEGRATION_WARN)

    if bundle_stability == "UNSTABLE":
        blocking_rules.append(TopologyBundleBlockingRule.BUNDLE_UNSTABLE)

    # Cross-system consistency rules
    if consistency_result:
        consistency_status = consistency_result.get("consistency_status", "UNKNOWN")
        if consistency_status == "CONFLICT":
            blocking_rules.append(TopologyBundleBlockingRule.CROSS_SYSTEM_CONFLICT)
        elif consistency_status == "TENSION":
            blocking_rules.append(TopologyBundleBlockingRule.CROSS_SYSTEM_TENSION)

    # Calculate blocking rate
    # Higher rate = more blocking pressure
    blocking_rate = _calculate_blocking_rate(
        joint_status, topology_stability, bundle_stability, blocking_rules
    )

    # Map joint status to governance status
    status = joint_status  # Already "OK"|"WARN"|"BLOCK"

    # Upgrade to BLOCK if cross-system conflict
    if consistency_result and consistency_result.get("consistency_status") == "CONFLICT":
        status = "BLOCK"

    # Build metrics dict
    metrics = {
        "topology_ok_for_integration": topology_ok,
        "topology_stability": topology_stability,
        "bundle_stability_rating": bundle_stability,
        "blocking_rule_count": len(blocking_rules),
    }

    if consistency_result:
        metrics["cross_system_consistency"] = consistency_result.get("consistency_status")
        metrics["dag_status"] = consistency_result.get("dag_status")

    return {
        "layer": "topology_bundle",
        "status": status,
        "blocking_rules": blocking_rules,
        "blocking_rate": blocking_rate,
        "reasons": reasons,
        "metrics": metrics,
        "schema_version": GOVERNANCE_SIGNAL_SCHEMA_VERSION,
    }


def _calculate_blocking_rate(
    joint_status: str,
    topology_stability: str,
    bundle_stability: str,
    blocking_rules: List[str],
) -> float:
    """
    Calculate a blocking rate (0.0-1.0) based on governance factors.

    Higher values indicate more blocking pressure / higher risk.

    The rate is calculated as:
    - Base rate from joint status: BLOCK=1.0, WARN=0.5, OK=0.0
    - Adjustments for stability factors
    - Adjustments for number of blocking rules
    """
    # Base rate from status
    if joint_status == "BLOCK":
        base_rate = 1.0
    elif joint_status == "WARN":
        base_rate = 0.5
    else:
        base_rate = 0.0

    # Stability adjustments (max +0.3)
    stability_adjustment = 0.0
    if topology_stability == "VOLATILE":
        stability_adjustment += 0.15
    elif topology_stability == "DEGRADING":
        stability_adjustment += 0.10

    if bundle_stability == "UNSTABLE":
        stability_adjustment += 0.15

    # Rule count adjustment (each rule adds 0.05, max 0.2)
    rule_adjustment = min(len(blocking_rules) * 0.05, 0.2)

    # Calculate final rate (capped at 1.0)
    final_rate = min(base_rate + stability_adjustment + rule_adjustment, 1.0)

    # Round to 2 decimal places
    return round(final_rate, 2)


# =============================================================================
# Convenience: Build Full Governance Signal Pipeline
# =============================================================================


def build_topology_bundle_governance_signal(
    trajectory: HealthTrajectory,
    bundle_evolution: Dict[str, Any],
    dag_global_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete governance signal for the topology+bundle layer.

    This is a convenience function that orchestrates the full pipeline:
    1. Build joint view
    2. Check DAG consistency (if dag_global_health provided)
    3. Generate governance signal

    Args:
        trajectory: The health trajectory from topology analysis
        bundle_evolution: Bundle evolution ledger data
        dag_global_health: Optional DAG global health for consistency check

    Returns:
        Complete GovernanceSignal for CLAUDE I's build_global_alignment_view()
    """
    # Build joint view
    joint_view = build_topology_bundle_joint_view(trajectory, bundle_evolution)

    # Check DAG consistency if provided
    consistency_result = None
    if dag_global_health is not None:
        consistency_result = check_consistency_with_dag_posture(
            trajectory, dag_global_health
        )

    # Generate governance signal
    return to_governance_signal_for_topology_bundle(joint_view, consistency_result)
