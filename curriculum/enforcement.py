"""
Curriculum Phase X Enforcement Module.

Implements invariant checking and governance signal generation for P3/P4
shadow operations as specified in docs/system_law/Curriculum_PhaseX_Invariants.md.

SHADOW MODE CONTRACT:
- All functions return advisory results only
- No curriculum mutations are performed
- Violations are logged, not enforced
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from substrate.repro.determinism import (
    deterministic_hash,
    deterministic_isoformat,
    deterministic_uuid,
)

# -----------------------------------------------------------------------------
# Severity and Status Enums
# -----------------------------------------------------------------------------

class DriftSeverity(Enum):
    """Drift severity classification."""
    NONE = "NONE"
    PARAMETRIC = "PARAMETRIC"
    SEMANTIC = "SEMANTIC"


class DriftStatus(Enum):
    """Drift response status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class GovernanceSignalType(Enum):
    """Types of governance signals."""
    DRIFT_DETECTED = "DRIFT_DETECTED"
    TRANSITION_REQUESTED = "TRANSITION_REQUESTED"
    TRANSITION_VALIDATED = "TRANSITION_VALIDATED"
    TRANSITION_BLOCKED = "TRANSITION_BLOCKED"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    SNAPSHOT_CAPTURED = "SNAPSHOT_CAPTURED"
    SNAPSHOT_VERIFIED = "SNAPSHOT_VERIFIED"


Constraint = Literal["increasing", "decreasing", "boolean_true", "any"]
Phase = Literal["P3", "P4"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GateEvolutionRule:
    """Rule for validating gate threshold evolution."""
    path: Tuple[str, ...]
    constraint: Constraint


GATE_EVOLUTION_RULES: List[GateEvolutionRule] = [
    GateEvolutionRule(("coverage", "ci_lower_min"), "increasing"),
    GateEvolutionRule(("coverage", "sample_min"), "increasing"),
    GateEvolutionRule(("coverage", "require_attestation"), "boolean_true"),
    GateEvolutionRule(("abstention", "max_rate_pct"), "decreasing"),
    GateEvolutionRule(("abstention", "max_mass"), "decreasing"),
    GateEvolutionRule(("velocity", "min_pph"), "increasing"),
    GateEvolutionRule(("velocity", "stability_cv_max"), "decreasing"),
    GateEvolutionRule(("velocity", "window_minutes"), "any"),
    GateEvolutionRule(("caps", "min_attempt_mass"), "increasing"),
    GateEvolutionRule(("caps", "min_runtime_minutes"), "increasing"),
    GateEvolutionRule(("caps", "backlog_max"), "decreasing"),
]


@dataclass
class Violation:
    """Represents a single invariant violation."""
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MonotonicityViolation:
    """Represents a monotonicity constraint violation."""
    axis: str
    type: Literal["MISSING_AXIS", "REGRESSION"]
    before: Optional[Any]
    after: Optional[Any]
    delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "axis": self.axis,
            "type": self.type,
            "before": self.before,
            "after": self.after,
        }
        if self.delta is not None:
            result["delta"] = self.delta
        return result


@dataclass
class GateEvolutionViolation:
    """Represents a gate evolution rule violation."""
    gate: str
    threshold: str
    violation_type: Literal["REGRESSION", "RELAXATION", "DISABLE"]
    before: Any
    after: Any
    delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "gate": self.gate,
            "threshold": self.threshold,
            "violation_type": self.violation_type,
            "before": self.before,
            "after": self.after,
        }
        if self.delta is not None:
            result["delta"] = self.delta
        return result


@dataclass
class ChangedParam:
    """Represents a changed parameter in drift analysis."""
    path: str
    baseline: Any
    current: Any
    classification: DriftSeverity
    constraint: Constraint
    delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "path": self.path,
            "baseline": self.baseline,
            "current": self.current,
            "classification": self.classification.value,
            "constraint": self.constraint,
        }
        if self.delta is not None:
            result["delta"] = self.delta
        return result


@dataclass
class CurriculumSnapshot:
    """Immutable snapshot of curriculum state."""
    fingerprint: str
    slice_names: List[str]
    active_index: int
    active_slice_name: str
    params: Dict[str, Any]
    gates: Dict[str, Any]
    monotonic_axes: Tuple[str, ...]
    raw_config: Dict[str, Any]

    @classmethod
    def from_config(cls, config: Dict[str, Any], system_slug: str) -> "CurriculumSnapshot":
        """Create snapshot from curriculum config dict."""
        fingerprint = deterministic_hash(
            json.dumps(config, sort_keys=True, separators=(",", ":"))
        )

        system_cfg = config.get("systems", {}).get(system_slug, {})
        slices = system_cfg.get("slices", [])
        slice_names = [s.get("name", f"slice_{i}") for i, s in enumerate(slices)]

        active_name = system_cfg.get("active")
        active_index = 0
        for i, name in enumerate(slice_names):
            if name == active_name:
                active_index = i
                break

        active_slice = slices[active_index] if slices else {}
        params = dict(active_slice.get("params", {}))
        gates = dict(active_slice.get("gates", {}))

        invariants = system_cfg.get("invariants", {})
        monotonic_axes = tuple(invariants.get("monotonic_axes", []))

        return cls(
            fingerprint=fingerprint,
            slice_names=slice_names,
            active_index=active_index,
            active_slice_name=active_name or "",
            params=params,
            gates=gates,
            monotonic_axes=monotonic_axes,
            raw_config=deepcopy(config),
        )


@dataclass
class P3VerificationResult:
    """Result of P3 pre-execution verification."""
    valid: bool
    phase: str = "P3"
    mode: str = "SHADOW"
    curriculum_fingerprint: str = ""
    active_slice: str = ""
    violations: List[Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    snapshot: Optional[CurriculumSnapshot] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "phase": self.phase,
            "mode": self.mode,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "active_slice": self.active_slice,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
        }


@dataclass
class DriftTimelineEvent:
    """A single event in the curriculum drift timeline."""
    event_id: str
    timestamp: str
    phase: Phase
    mode: str = "SHADOW"
    curriculum_fingerprint: str = ""
    slice_name: str = ""
    baseline_slice_name: str = ""
    drift_status: DriftStatus = DriftStatus.OK
    drift_severity: DriftSeverity = DriftSeverity.NONE
    changed_params: List[ChangedParam] = field(default_factory=list)
    monotonicity_violations: List[MonotonicityViolation] = field(default_factory=list)
    gate_evolution_violations: List[GateEvolutionViolation] = field(default_factory=list)
    action_taken: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "curriculum-drift-timeline/1.0.0",
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "mode": self.mode,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "slice_name": self.slice_name,
            "baseline_slice_name": self.baseline_slice_name,
            "drift_status": self.drift_status.value,
            "drift_severity": self.drift_severity.value,
            "changed_params": [p.to_dict() for p in self.changed_params],
            "monotonicity_violations": [v.to_dict() for v in self.monotonicity_violations],
            "gate_evolution_violations": [v.to_dict() for v in self.gate_evolution_violations],
            "action_taken": self.action_taken,
        }

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass
class GovernanceSignal:
    """Curriculum governance signal for audit logging."""
    signal_id: str
    timestamp: str
    phase: Phase
    mode: str = "SHADOW"
    signal_type: GovernanceSignalType = GovernanceSignalType.DRIFT_DETECTED
    curriculum_fingerprint: str = ""
    active_slice: str = ""
    target_slice: Optional[str] = None
    severity: DriftSeverity = DriftSeverity.NONE
    status: DriftStatus = DriftStatus.OK
    violations: List[Violation] = field(default_factory=list)
    governance_action: str = "LOGGED_ONLY"
    hypothetical: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": "curriculum-governance-signal/1.0.0",
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "mode": self.mode,
            "signal_type": self.signal_type.value,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "active_slice": self.active_slice,
            "severity": self.severity.value,
            "status": self.status.value,
            "violations": [v.to_dict() for v in self.violations],
            "governance_action": self.governance_action,
        }
        if self.target_slice is not None:
            result["target_slice"] = self.target_slice
        if self.hypothetical is not None:
            result["hypothetical"] = self.hypothetical
        if self.context is not None:
            result["context"] = self.context
        return result

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _get_nested(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    """Get nested value from dict by path."""
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _is_number(value: Any) -> bool:
    """Check if value is a number (not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _classify_drift(
    path: str,
    baseline: Any,
    current: Any,
    constraint: Constraint,
) -> DriftSeverity:
    """Classify the severity of a parameter drift."""
    if baseline == current:
        return DriftSeverity.NONE

    if constraint == "any":
        return DriftSeverity.PARAMETRIC

    if constraint == "boolean_true":
        if baseline and not current:
            return DriftSeverity.SEMANTIC
        return DriftSeverity.PARAMETRIC

    if constraint == "increasing":
        if _is_number(baseline) and _is_number(current):
            if float(current) < float(baseline):
                return DriftSeverity.SEMANTIC
        return DriftSeverity.PARAMETRIC

    if constraint == "decreasing":
        if _is_number(baseline) and _is_number(current):
            if float(current) > float(baseline):
                return DriftSeverity.SEMANTIC
        return DriftSeverity.PARAMETRIC

    return DriftSeverity.PARAMETRIC


def _severity_to_status(severity: DriftSeverity) -> DriftStatus:
    """Map severity to status."""
    if severity == DriftSeverity.SEMANTIC:
        return DriftStatus.BLOCK
    if severity == DriftSeverity.PARAMETRIC:
        return DriftStatus.WARN
    return DriftStatus.OK


def _max_severity(severities: List[DriftSeverity]) -> DriftSeverity:
    """Return maximum severity from list."""
    order = {DriftSeverity.NONE: 0, DriftSeverity.PARAMETRIC: 1, DriftSeverity.SEMANTIC: 2}
    if not severities:
        return DriftSeverity.NONE
    return max(severities, key=lambda s: order[s])


# -----------------------------------------------------------------------------
# P3 Pre-Execution Verification
# -----------------------------------------------------------------------------

def verify_curriculum_for_p3(
    config: Dict[str, Any],
    system_slug: str = "propositional",
) -> Dict[str, Any]:
    """
    Verify curriculum configuration for P3 shadow execution.

    Checks CUR-P3 invariants and returns advisory result.

    SHADOW MODE: Returns advisory result only; no enforcement.

    Args:
        config: Curriculum configuration dict (from curriculum.yaml)
        system_slug: System slug to verify

    Returns:
        Dict with verification result including:
        - valid: bool
        - phase: "P3"
        - mode: "SHADOW"
        - curriculum_fingerprint: hash of config
        - active_slice: name of active slice
        - violations: list of violation dicts
        - warnings: list of warning strings
    """
    violations: List[Violation] = []
    warnings: List[str] = []

    # Compute fingerprint
    fingerprint = deterministic_hash(
        json.dumps(config, sort_keys=True, separators=(",", ":"))
    )

    # Check version
    version = config.get("version")
    if version != 2:
        violations.append(Violation(
            code="CUR-P3-VERSION",
            message=f"Unsupported curriculum version: {version}",
            details={"expected": 2, "actual": version},
        ))

    # Check system exists
    systems = config.get("systems", {})
    if system_slug not in systems:
        violations.append(Violation(
            code="CUR-P3-SYSTEM",
            message=f"System '{system_slug}' not found",
            details={"available": list(systems.keys())},
        ))
        return P3VerificationResult(
            valid=False,
            curriculum_fingerprint=fingerprint,
            violations=violations,
            warnings=warnings,
        ).to_dict()

    system_cfg = systems[system_slug]

    # Check description
    if not system_cfg.get("description"):
        warnings.append(f"System '{system_slug}' missing description")

    # Check slices exist
    slices = system_cfg.get("slices", [])
    if not slices:
        violations.append(Violation(
            code="CUR-P3-SLICES",
            message=f"System '{system_slug}' has no slices",
        ))
        return P3VerificationResult(
            valid=False,
            curriculum_fingerprint=fingerprint,
            violations=violations,
            warnings=warnings,
        ).to_dict()

    # Check active slice
    active_name = system_cfg.get("active")
    if not active_name:
        warnings.append("No active slice specified; will use first incomplete")
        for s in slices:
            if s.get("completed_at") is None:
                active_name = s.get("name")
                break

    active_slice = None
    active_index = -1
    for i, s in enumerate(slices):
        if s.get("name") == active_name:
            active_slice = s
            active_index = i
            break

    if active_slice is None:
        violations.append(Violation(
            code="CUR-P3-ACTIVE",
            message=f"Active slice '{active_name}' not found in slices",
            details={"available": [s.get("name") for s in slices]},
        ))
        return P3VerificationResult(
            valid=False,
            curriculum_fingerprint=fingerprint,
            violations=violations,
            warnings=warnings,
        ).to_dict()

    # Check slice structure
    for i, slice_cfg in enumerate(slices):
        slice_name = slice_cfg.get("name", f"slice_{i}")

        # Check required fields
        if "params" not in slice_cfg:
            violations.append(Violation(
                code="CUR-P3-SLICE-PARAMS",
                message=f"Slice '{slice_name}' missing params",
            ))
        if "gates" not in slice_cfg:
            violations.append(Violation(
                code="CUR-P3-SLICE-GATES",
                message=f"Slice '{slice_name}' missing gates",
            ))

        # Check gate structure
        gates = slice_cfg.get("gates", {})
        required_gates = ["coverage", "abstention", "velocity", "caps"]
        for gate_name in required_gates:
            if gate_name not in gates:
                violations.append(Violation(
                    code="CUR-P3-GATE-MISSING",
                    message=f"Slice '{slice_name}' missing gate '{gate_name}'",
                ))

    # Check monotonicity invariants
    invariants = system_cfg.get("invariants", {})
    monotonic_axes = invariants.get("monotonic_axes", [])

    for axis in monotonic_axes:
        prev_value: Optional[int] = None
        for i, s in enumerate(slices):
            params = s.get("params", {})
            value = params.get(axis)

            if value is None:
                violations.append(Violation(
                    code="CUR-P3-MONO-MISSING",
                    message=f"Slice '{s.get('name')}' missing monotonic axis '{axis}'",
                ))
                continue

            if prev_value is not None and int(value) < prev_value:
                violations.append(Violation(
                    code="CUR-P3-MONO-VIOLATION",
                    message=f"Monotonicity violation on '{axis}' at slice '{s.get('name')}'",
                    details={
                        "axis": axis,
                        "previous": prev_value,
                        "current": int(value),
                    },
                ))
            prev_value = int(value)

    # Create snapshot
    snapshot = CurriculumSnapshot.from_config(config, system_slug)

    result = P3VerificationResult(
        valid=len(violations) == 0,
        curriculum_fingerprint=fingerprint,
        active_slice=active_name or "",
        violations=violations,
        warnings=warnings,
        snapshot=snapshot,
    )

    return result.to_dict()


# -----------------------------------------------------------------------------
# P4 Drift Timeline Generator
# -----------------------------------------------------------------------------

class DriftTimelineGenerator:
    """
    Generates curriculum drift timeline events for P4 shadow observation.

    SHADOW MODE: All events are observational only.
    """

    def __init__(self, phase: Phase = "P4"):
        self.phase = phase
        self._baseline_snapshot: Optional[CurriculumSnapshot] = None

    def capture_baseline(self, config: Dict[str, Any], system_slug: str) -> CurriculumSnapshot:
        """Capture baseline snapshot at run start."""
        self._baseline_snapshot = CurriculumSnapshot.from_config(config, system_slug)
        return self._baseline_snapshot

    def generate_drift_event(
        self,
        current_config: Dict[str, Any],
        system_slug: str,
        run_id: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> DriftTimelineEvent:
        """
        Generate a drift timeline event comparing current config to baseline.

        Args:
            current_config: Current curriculum configuration
            system_slug: System slug
            run_id: Optional run identifier
            cycle: Optional cycle number

        Returns:
            DriftTimelineEvent with comparison results
        """
        if self._baseline_snapshot is None:
            raise ValueError("No baseline captured; call capture_baseline first")

        baseline = self._baseline_snapshot
        current = CurriculumSnapshot.from_config(current_config, system_slug)

        # Generate deterministic event ID
        event_id = deterministic_uuid(
            f"{baseline.fingerprint}:{current.fingerprint}:{run_id or ''}:{cycle or 0}"
        )

        # Generate deterministic timestamp
        timestamp = deterministic_isoformat(event_id, self.phase)

        # Analyze drift
        changed_params: List[ChangedParam] = []
        mono_violations: List[MonotonicityViolation] = []
        gate_violations: List[GateEvolutionViolation] = []
        max_sev = DriftSeverity.NONE

        # Check ladder changes
        if baseline.slice_names != current.slice_names:
            changed_params.append(ChangedParam(
                path="slice_ladder",
                baseline=baseline.slice_names,
                current=current.slice_names,
                classification=DriftSeverity.SEMANTIC,
                constraint="any",
            ))
            max_sev = DriftSeverity.SEMANTIC

        # Check active index
        if baseline.active_index != current.active_index:
            sev = DriftSeverity.SEMANTIC if current.active_index < baseline.active_index else DriftSeverity.PARAMETRIC
            changed_params.append(ChangedParam(
                path="active_index",
                baseline=baseline.active_index,
                current=current.active_index,
                classification=sev,
                constraint="increasing",
                delta=float(current.active_index - baseline.active_index),
            ))
            max_sev = _max_severity([max_sev, sev])

        # Check params
        param_axes = ["atoms", "depth_max", "breadth_max", "total_max"]
        for axis in param_axes:
            b_val = baseline.params.get(axis)
            c_val = current.params.get(axis)
            if b_val != c_val:
                sev = _classify_drift(axis, b_val, c_val, "increasing")
                delta = None
                if _is_number(b_val) and _is_number(c_val):
                    delta = float(c_val) - float(b_val)
                changed_params.append(ChangedParam(
                    path=f"params.{axis}",
                    baseline=b_val,
                    current=c_val,
                    classification=sev,
                    constraint="increasing",
                    delta=delta,
                ))
                max_sev = _max_severity([max_sev, sev])

                if sev == DriftSeverity.SEMANTIC:
                    mono_violations.append(MonotonicityViolation(
                        axis=axis,
                        type="REGRESSION",
                        before=b_val,
                        after=c_val,
                        delta=delta,
                    ))

        # Check gates
        for rule in GATE_EVOLUTION_RULES:
            b_val = _get_nested(baseline.gates, rule.path)
            c_val = _get_nested(current.gates, rule.path)
            if b_val == c_val:
                continue

            sev = _classify_drift(".".join(rule.path), b_val, c_val, rule.constraint)
            delta = None
            if _is_number(b_val) and _is_number(c_val):
                delta = float(c_val) - float(b_val)

            changed_params.append(ChangedParam(
                path=f"gates.{'.'.join(rule.path)}",
                baseline=b_val,
                current=c_val,
                classification=sev,
                constraint=rule.constraint,
                delta=delta,
            ))
            max_sev = _max_severity([max_sev, sev])

            if sev == DriftSeverity.SEMANTIC:
                violation_type: Literal["REGRESSION", "RELAXATION", "DISABLE"] = "REGRESSION"
                if rule.constraint == "boolean_true":
                    violation_type = "DISABLE"
                elif rule.constraint == "decreasing":
                    violation_type = "RELAXATION"

                gate_violations.append(GateEvolutionViolation(
                    gate=rule.path[0],
                    threshold=rule.path[1] if len(rule.path) > 1 else "",
                    violation_type=violation_type,
                    before=b_val,
                    after=c_val,
                    delta=delta,
                ))

        return DriftTimelineEvent(
            event_id=event_id,
            timestamp=timestamp,
            phase=self.phase,
            mode="SHADOW",
            curriculum_fingerprint=current.fingerprint,
            slice_name=current.active_slice_name,
            baseline_slice_name=baseline.active_slice_name,
            drift_status=_severity_to_status(max_sev),
            drift_severity=max_sev,
            changed_params=changed_params,
            monotonicity_violations=mono_violations,
            gate_evolution_violations=gate_violations,
            action_taken="LOGGED_ONLY",
        )


# -----------------------------------------------------------------------------
# Governance Signal Builder
# -----------------------------------------------------------------------------

class GovernanceSignalBuilder:
    """
    Builds governance signals from drift events and verification results.

    SHADOW MODE: All signals record observational data only.
    """

    def __init__(self, phase: Phase = "P4"):
        self.phase = phase

    def from_drift_event(
        self,
        event: DriftTimelineEvent,
        run_id: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> GovernanceSignal:
        """
        Build governance signal from drift timeline event.

        Args:
            event: Drift timeline event
            run_id: Optional run identifier
            cycle: Optional cycle number

        Returns:
            GovernanceSignal
        """
        signal_id = deterministic_uuid(f"signal:{event.event_id}")
        timestamp = deterministic_isoformat(signal_id, self.phase, event.timestamp)

        # Convert drift violations to Violations
        violations: List[Violation] = []

        for mv in event.monotonicity_violations:
            violations.append(Violation(
                code=f"MONO_{mv.type}_{mv.axis.upper()}",
                message=f"Monotonicity {mv.type.lower()} on axis '{mv.axis}'",
                details=mv.to_dict(),
            ))

        for gv in event.gate_evolution_violations:
            violations.append(Violation(
                code=f"GATE_{gv.violation_type}_{gv.gate.upper()}",
                message=f"Gate {gv.violation_type.lower()} on {gv.gate}.{gv.threshold}",
                details=gv.to_dict(),
            ))

        # Determine signal type
        signal_type = GovernanceSignalType.DRIFT_DETECTED
        if event.drift_severity == DriftSeverity.SEMANTIC:
            signal_type = GovernanceSignalType.INVARIANT_VIOLATION

        # Build hypothetical
        hypothetical = {
            "would_allow_transition": event.drift_severity != DriftSeverity.SEMANTIC,
            "would_trigger_alert": event.drift_severity != DriftSeverity.NONE,
            "blocking_violations": [v.code for v in violations],
        }

        # Build context
        context: Dict[str, Any] = {}
        if run_id:
            context["run_id"] = run_id
        if cycle is not None:
            context["cycle"] = cycle
        context["triggering_event"] = event.event_id

        return GovernanceSignal(
            signal_id=signal_id,
            timestamp=timestamp,
            phase=self.phase,
            mode="SHADOW",
            signal_type=signal_type,
            curriculum_fingerprint=event.curriculum_fingerprint,
            active_slice=event.slice_name,
            target_slice=event.baseline_slice_name if event.baseline_slice_name != event.slice_name else None,
            severity=event.drift_severity,
            status=event.drift_status,
            violations=violations,
            governance_action="LOGGED_ONLY",
            hypothetical=hypothetical,
            context=context if context else None,
        )

    def from_verification_result(
        self,
        result: Dict[str, Any],
        signal_type: GovernanceSignalType = GovernanceSignalType.SNAPSHOT_CAPTURED,
    ) -> GovernanceSignal:
        """
        Build governance signal from P3 verification result.

        Args:
            result: P3 verification result dict
            signal_type: Type of signal to generate

        Returns:
            GovernanceSignal
        """
        fingerprint = result.get("curriculum_fingerprint", "")
        signal_id = deterministic_uuid(f"signal:verify:{fingerprint}")
        timestamp = deterministic_isoformat(signal_id, self.phase)

        violations = [
            Violation(**v) for v in result.get("violations", [])
        ]

        severity = DriftSeverity.SEMANTIC if violations else DriftSeverity.NONE
        status = DriftStatus.BLOCK if violations else DriftStatus.OK

        hypothetical = {
            "would_allow_transition": result.get("valid", False),
            "would_trigger_alert": not result.get("valid", True),
            "blocking_violations": [v.code for v in violations],
        }

        return GovernanceSignal(
            signal_id=signal_id,
            timestamp=timestamp,
            phase=self.phase,
            mode="SHADOW",
            signal_type=signal_type,
            curriculum_fingerprint=fingerprint,
            active_slice=result.get("active_slice", ""),
            severity=severity,
            status=status,
            violations=violations,
            governance_action="LOGGED_ONLY",
            hypothetical=hypothetical,
        )

    def transition_requested(
        self,
        fingerprint: str,
        from_slice: str,
        to_slice: str,
        run_id: Optional[str] = None,
    ) -> GovernanceSignal:
        """Create signal for transition request (observational)."""
        signal_id = deterministic_uuid(f"signal:transition:{fingerprint}:{from_slice}:{to_slice}")
        timestamp = deterministic_isoformat(signal_id, self.phase)

        return GovernanceSignal(
            signal_id=signal_id,
            timestamp=timestamp,
            phase=self.phase,
            mode="SHADOW",
            signal_type=GovernanceSignalType.TRANSITION_REQUESTED,
            curriculum_fingerprint=fingerprint,
            active_slice=from_slice,
            target_slice=to_slice,
            severity=DriftSeverity.NONE,
            status=DriftStatus.OK,
            governance_action="LOGGED_ONLY",
            hypothetical={
                "would_allow_transition": True,
                "would_trigger_alert": False,
                "blocking_violations": [],
            },
            context={"run_id": run_id} if run_id else None,
        )


# -----------------------------------------------------------------------------
# Runtime Guard
# -----------------------------------------------------------------------------

class CurriculumRuntimeGuard:
    """
    Runtime enforcement guard for curriculum invariants.

    SHADOW MODE: Checks are advisory only; violations are logged.
    """

    def __init__(self, phase: Phase = "P3"):
        self.phase = phase
        self._snapshot: Optional[CurriculumSnapshot] = None
        self._drift_generator = DriftTimelineGenerator(phase)
        self._signal_builder = GovernanceSignalBuilder(phase)
        self._events: List[DriftTimelineEvent] = []
        self._signals: List[GovernanceSignal] = []

    def capture_snapshot(self, config: Dict[str, Any], system_slug: str) -> CurriculumSnapshot:
        """Capture curriculum state at run start."""
        self._snapshot = self._drift_generator.capture_baseline(config, system_slug)

        # Generate snapshot captured signal
        verify_result = verify_curriculum_for_p3(config, system_slug)
        signal = self._signal_builder.from_verification_result(
            verify_result,
            GovernanceSignalType.SNAPSHOT_CAPTURED,
        )
        self._signals.append(signal)

        return self._snapshot

    def verify_unchanged(
        self,
        config: Dict[str, Any],
        system_slug: str,
        run_id: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> Tuple[bool, Optional[DriftTimelineEvent], Optional[GovernanceSignal]]:
        """
        Verify curriculum matches snapshot (advisory in shadow mode).

        Returns:
            (unchanged: bool, drift_event: optional, signal: optional)
        """
        if self._snapshot is None:
            raise ValueError("No snapshot captured; call capture_snapshot first")

        event = self._drift_generator.generate_drift_event(config, system_slug, run_id, cycle)
        self._events.append(event)

        if event.drift_severity != DriftSeverity.NONE:
            signal = self._signal_builder.from_drift_event(event, run_id, cycle)
            self._signals.append(signal)
            return (event.drift_severity == DriftSeverity.NONE, event, signal)

        # Generate verification signal
        signal = GovernanceSignal(
            signal_id=deterministic_uuid(f"signal:verify:{event.event_id}"),
            timestamp=event.timestamp,
            phase=self.phase,
            mode="SHADOW",
            signal_type=GovernanceSignalType.SNAPSHOT_VERIFIED,
            curriculum_fingerprint=event.curriculum_fingerprint,
            active_slice=event.slice_name,
            severity=DriftSeverity.NONE,
            status=DriftStatus.OK,
            governance_action="LOGGED_ONLY",
        )
        self._signals.append(signal)

        return (True, event, signal)

    def get_events(self) -> List[DriftTimelineEvent]:
        """Return all drift events."""
        return list(self._events)

    def get_signals(self) -> List[GovernanceSignal]:
        """Return all governance signals."""
        return list(self._signals)

    def export_timeline_jsonl(self) -> str:
        """Export all drift events as JSONL."""
        return "\n".join(e.to_jsonl() for e in self._events)

    def export_signals_jsonl(self) -> str:
        """Export all governance signals as JSONL."""
        return "\n".join(s.to_jsonl() for s in self._signals)


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Enums
    "DriftSeverity",
    "DriftStatus",
    "GovernanceSignalType",
    # Data structures
    "Violation",
    "MonotonicityViolation",
    "GateEvolutionViolation",
    "ChangedParam",
    "CurriculumSnapshot",
    "P3VerificationResult",
    "DriftTimelineEvent",
    "GovernanceSignal",
    # Functions
    "verify_curriculum_for_p3",
    # Classes
    "DriftTimelineGenerator",
    "GovernanceSignalBuilder",
    "CurriculumRuntimeGuard",
    # Constants
    "GATE_EVOLUTION_RULES",
]
