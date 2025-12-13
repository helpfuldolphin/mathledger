"""
P5 Divergence Interpreter -- Real-telemetry divergence diagnosis.

Implements the 8-rule deterministic rule engine specified in
docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md

SHADOW MODE CONTRACT:
- All output is observational only
- No control flow depends on diagnostic results
- Output is for logging, analysis, and evidence collection

Signal Precedence (from spec Section 1.3):
    Priority 1: Identity Signal (cryptographic)     -> Security-critical
    Priority 2: Structure Signal (DAG coherence)    -> Data integrity
    Priority 3: Replay Safety Signal                -> Determinism verification
    Priority 4: Topology Signal                     -> Model/manifold health
    Priority 5: Budget Signal                       -> Resource constraints
    Priority 6: Metrics Signal                      -> Performance indicators

Rule Engine (from spec Section 3.2):
    Rule 1: NOMINAL           - All OK, divergence NONE/INFO
    Rule 2: REPLAY_FAILURE    - Replay BLOCK, Topology STABLE, Budget STABLE
    Rule 3: STRUCTURAL_BREAK  - Replay OK, Topology TURBULENT/CRITICAL, Type STATE
    Rule 4: PHASE_LAG         - Replay WARN, Topology DRIFT, Budget DRIFTING/VOLATILE
    Rule 5: BUDGET_CONFOUND   - Budget VOLATILE, Divergence WARN/CRITICAL
    Rule 6: CASCADING_FAILURE - 2+ BLOCK/CRITICAL signals
    Rule 7: REPLAY_FAILURE    - Replay WARN, Topology STABLE (soft)
    Rule 8: UNKNOWN           - Default fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================


class RootCauseHypothesis(str, Enum):
    """Root cause attribution hypotheses."""

    NOMINAL = "NOMINAL"
    REPLAY_FAILURE = "REPLAY_FAILURE"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    PHASE_LAG = "PHASE_LAG"
    BUDGET_CONFOUND = "BUDGET_CONFOUND"
    IDENTITY_VIOLATION = "IDENTITY_VIOLATION"
    STRUCTURE_VIOLATION = "STRUCTURE_VIOLATION"
    CASCADING_FAILURE = "CASCADING_FAILURE"
    UNKNOWN = "UNKNOWN"


class DiagnosticAction(str, Enum):
    """Recommended action levels (advisory only)."""

    LOGGED_ONLY = "LOGGED_ONLY"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AttributionStep:
    """Single step in the attribution chain."""

    signal: str
    status: str
    contributed_to_diagnosis: bool
    reason: str


@dataclass
class DiagnosticResult:
    """Result of divergence diagnostic evaluation."""

    root_cause_hypothesis: RootCauseHypothesis
    root_cause_confidence: float
    action: DiagnosticAction
    action_rationale: str
    headline: str
    attribution_chain: List[AttributionStep] = field(default_factory=list)
    confounding_factors: List[str] = field(default_factory=list)


@dataclass
class SupportingSignals:
    """Collected signals supporting the diagnosis."""

    replay: Optional[Dict[str, Any]] = None
    topology: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    divergence: Optional[Dict[str, Any]] = None
    identity: Optional[Dict[str, Any]] = None
    structure: Optional[Dict[str, Any]] = None


# =============================================================================
# Core Interpreter Class
# =============================================================================


class P5DivergenceInterpreter:
    """
    Interprets P5 divergence by synthesizing replay, topology, budget, and
    divergence signals into a unified diagnostic.

    SHADOW MODE CONTRACT:
    - interpret_divergence() returns observational data only
    - No control flow or enforcement depends on output
    - All output is for logging and analysis

    Usage:
        interpreter = P5DivergenceInterpreter()
        diagnostic = interpreter.interpret_divergence(
            divergence_snapshot=snapshot,
            replay_signal=replay,
            topology_signal=topology,
            budget_signal=budget,
        )
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        enable_identity_check: bool = True,
        enable_structure_check: bool = True,
    ) -> None:
        """
        Initialize the interpreter.

        Args:
            enable_identity_check: Whether to check identity signals (if provided)
            enable_structure_check: Whether to check structure signals (if provided)
        """
        self._enable_identity_check = enable_identity_check
        self._enable_structure_check = enable_structure_check

    def interpret_divergence(
        self,
        divergence_snapshot: Dict[str, Any],
        replay_signal: Dict[str, Any],
        topology_signal: Dict[str, Any],
        budget_signal: Dict[str, Any],
        *,
        identity_signal: Optional[Dict[str, Any]] = None,
        structure_signal: Optional[Dict[str, Any]] = None,
        cycle: int = 0,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Interpret divergence and produce diagnostic panel output.

        SHADOW MODE: Output is observational only.

        Args:
            divergence_snapshot: P4 divergence snapshot with severity, type, divergence_pct
            replay_signal: Replay safety governance signal with status, alignment, conflict
            topology_signal: Topology bundle signal with mode, persistence_drift, betti stats
            budget_signal: Budget signal with stability_class, health_score
            identity_signal: Optional identity signal for security checks
            structure_signal: Optional structure signal for DAG coherence checks
            cycle: Current cycle number
            run_id: Optional run identifier

        Returns:
            Dict conforming to p5_divergence_diagnostic.schema.json
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Collect supporting signals
        supporting = self._collect_supporting_signals(
            replay_signal,
            topology_signal,
            budget_signal,
            divergence_snapshot,
            identity_signal,
            structure_signal,
        )

        # Run diagnostic evaluation
        result = self._evaluate(
            identity_signal=identity_signal,
            structure_signal=structure_signal,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            divergence_snapshot=divergence_snapshot,
        )

        # Extract status values for output
        replay_status = self._normalize_status(replay_signal.get("status", "OK"))
        divergence_severity = divergence_snapshot.get("severity", "NONE").upper()
        divergence_type = divergence_snapshot.get("type", "NONE").upper()
        topology_mode = topology_signal.get("mode", "STABLE").upper()
        budget_stability = budget_signal.get("stability_class", "STABLE").upper()

        # Build output
        output: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": timestamp,
            "cycle": cycle,
            "replay_status": replay_status,
            "divergence_severity": divergence_severity,
            "divergence_type": divergence_type,
            "topology_mode": topology_mode,
            "budget_stability": budget_stability,
            "root_cause_hypothesis": result.root_cause_hypothesis.value,
            "root_cause_confidence": result.root_cause_confidence,
            "supporting_signals": supporting,
            "attribution_chain": [
                {
                    "signal": step.signal,
                    "status": step.status,
                    "contributed_to_diagnosis": step.contributed_to_diagnosis,
                    "reason": step.reason,
                }
                for step in result.attribution_chain
            ],
            "confounding_factors": result.confounding_factors,
            "action": result.action.value,
            "action_rationale": result.action_rationale,
            "headline": result.headline,
        }

        if run_id:
            output["run_id"] = run_id

        return output

    # =========================================================================
    # Rule Engine Implementation
    # =========================================================================

    def _evaluate(
        self,
        identity_signal: Optional[Dict[str, Any]],
        structure_signal: Optional[Dict[str, Any]],
        replay_signal: Dict[str, Any],
        topology_signal: Dict[str, Any],
        budget_signal: Dict[str, Any],
        divergence_snapshot: Dict[str, Any],
    ) -> DiagnosticResult:
        """
        Core evaluation logic implementing the 8-rule engine.

        See Section 3.1 of P5_Divergence_Diagnostic_Panel_Spec.md
        """
        attribution_chain: List[AttributionStep] = []
        confounding_factors: List[str] = []

        # =====================================================================
        # Phase 1: High-priority security/integrity signals
        # =====================================================================

        # Priority 1: Identity Signal (cryptographic)
        if self._enable_identity_check and identity_signal:
            identity_status = self._evaluate_identity(identity_signal)
            attribution_chain.append(
                AttributionStep(
                    signal="identity",
                    status=identity_status,
                    contributed_to_diagnosis=(identity_status == "BLOCK"),
                    reason=self._get_identity_reason(identity_signal, identity_status),
                )
            )

            if identity_status == "BLOCK":
                return DiagnosticResult(
                    root_cause_hypothesis=RootCauseHypothesis.IDENTITY_VIOLATION,
                    root_cause_confidence=1.0,
                    action=DiagnosticAction.INVESTIGATION_REQUIRED,
                    action_rationale="Cryptographic integrity failure detected",
                    headline="IDENTITY_VIOLATION: Cryptographic integrity compromised",
                    attribution_chain=attribution_chain,
                )

        # Priority 2: Structure Signal (DAG coherence)
        if self._enable_structure_check and structure_signal:
            structure_status = self._evaluate_structure(structure_signal)
            attribution_chain.append(
                AttributionStep(
                    signal="structure",
                    status=structure_status,
                    contributed_to_diagnosis=(structure_status == "BLOCK"),
                    reason=self._get_structure_reason(structure_signal, structure_status),
                )
            )

            if structure_status == "BLOCK":
                return DiagnosticResult(
                    root_cause_hypothesis=RootCauseHypothesis.STRUCTURE_VIOLATION,
                    root_cause_confidence=1.0,
                    action=DiagnosticAction.INVESTIGATION_REQUIRED,
                    action_rationale="DAG coherence or cycle detection failure",
                    headline="STRUCTURE_VIOLATION: DAG coherence broken",
                    attribution_chain=attribution_chain,
                )

        # =====================================================================
        # Phase 2: Extract core signal values
        # =====================================================================

        replay_status = self._normalize_status(replay_signal.get("status", "OK"))
        topology_mode = topology_signal.get("mode", "STABLE").upper()
        budget_stability = budget_signal.get("stability_class", "STABLE").upper()
        divergence_severity = divergence_snapshot.get("severity", "NONE").upper()
        divergence_type = divergence_snapshot.get("type", "NONE").upper()

        # =====================================================================
        # Phase 3: Check confounding factors
        # =====================================================================

        if budget_stability == "VOLATILE":
            confounding_factors.append("budget_volatile")
        elif budget_stability == "DRIFTING":
            confounding_factors.append("budget_drifting")

        if topology_mode in ("TURBULENT", "CRITICAL"):
            confounding_factors.append("topology_unstable")

        # Add remaining signals to attribution chain
        attribution_chain.append(
            AttributionStep(
                signal="replay",
                status=replay_status,
                contributed_to_diagnosis=False,
                reason=self._get_replay_reason(replay_signal, replay_status),
            )
        )
        attribution_chain.append(
            AttributionStep(
                signal="topology",
                status=topology_mode,
                contributed_to_diagnosis=False,
                reason=f"Mode: {topology_mode}",
            )
        )
        attribution_chain.append(
            AttributionStep(
                signal="budget",
                status=budget_stability,
                contributed_to_diagnosis=False,
                reason=f"Stability: {budget_stability}",
            )
        )
        attribution_chain.append(
            AttributionStep(
                signal="divergence",
                status=divergence_severity,
                contributed_to_diagnosis=False,
                reason=f"Severity: {divergence_severity}, Type: {divergence_type}",
            )
        )

        # =====================================================================
        # Phase 4: Apply diagnostic rules (8 rules, deterministic precedence)
        # =====================================================================

        # Rule 1: NOMINAL - All systems healthy
        if (
            replay_status == "OK"
            and topology_mode == "STABLE"
            and budget_stability == "STABLE"
            and divergence_severity in ("NONE", "INFO")
        ):
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.NOMINAL,
                root_cause_confidence=1.0,
                action=DiagnosticAction.LOGGED_ONLY,
                action_rationale="All systems operating within normal parameters",
                headline="NOMINAL: All systems healthy",
                attribution_chain=attribution_chain,
            )

        # Rule 2: REPLAY_FAILURE (strict) - Replay BLOCK in stable environment
        if (
            replay_status == "BLOCK"
            and topology_mode == "STABLE"
            and budget_stability == "STABLE"
        ):
            self._mark_contributed(attribution_chain, "replay")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.REPLAY_FAILURE,
                root_cause_confidence=0.95,
                action=DiagnosticAction.INVESTIGATION_REQUIRED,
                action_rationale="Replay failure in stable environment indicates determinism bug",
                headline="REPLAY_FAILURE: Determinism bug in stable environment",
                attribution_chain=attribution_chain,
            )

        # Rule 3: STRUCTURAL_BREAK - Replay OK but topology turbulent/critical
        if (
            replay_status == "OK"
            and topology_mode in ("TURBULENT", "CRITICAL")
            and divergence_type == "STATE"
        ):
            self._mark_contributed(attribution_chain, "replay")
            self._mark_contributed(attribution_chain, "topology")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.STRUCTURAL_BREAK,
                root_cause_confidence=0.85,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Topology turbulence with STATE divergence indicates model shift",
                headline=f"STRUCTURAL_BREAK: Topology {topology_mode}, Replay OK -- Model shift",
                attribution_chain=attribution_chain,
            )

        # Rule 4: PHASE_LAG - Multiple systems showing correlated strain
        if (
            replay_status == "WARN"
            and topology_mode == "DRIFT"
            and budget_stability in ("DRIFTING", "VOLATILE")
        ):
            self._mark_contributed(attribution_chain, "replay")
            self._mark_contributed(attribution_chain, "topology")
            self._mark_contributed(attribution_chain, "budget")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.PHASE_LAG,
                root_cause_confidence=0.70,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Multiple systems showing correlated drift",
                headline="PHASE_LAG: Correlated multi-system drift",
                attribution_chain=attribution_chain,
                confounding_factors=confounding_factors,
            )

        # Rule 5: BUDGET_CONFOUND - Divergence during budget instability
        if budget_stability == "VOLATILE" and divergence_severity in (
            "WARN",
            "CRITICAL",
        ):
            self._mark_contributed(attribution_chain, "budget")
            self._mark_contributed(attribution_chain, "divergence")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.BUDGET_CONFOUND,
                root_cause_confidence=0.60,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Budget instability may be obscuring true root cause",
                headline="BUDGET_CONFOUND: Divergence during budget instability",
                attribution_chain=attribution_chain,
                confounding_factors=["budget_volatile"],
            )

        # Rule 6: CASCADING_FAILURE - Multiple high-severity signals
        block_count = sum(
            [
                1 if replay_status == "BLOCK" else 0,
                1 if topology_mode == "CRITICAL" else 0,
                1 if divergence_severity == "CRITICAL" else 0,
            ]
        )

        if block_count >= 2:
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.CASCADING_FAILURE,
                root_cause_confidence=0.75,
                action=DiagnosticAction.INVESTIGATION_REQUIRED,
                action_rationale="Multiple subsystems in critical state",
                headline="CASCADING_FAILURE: Multiple systems in critical state",
                attribution_chain=attribution_chain,
                confounding_factors=confounding_factors,
            )

        # Rule 7: REPLAY_FAILURE (soft) - Replay WARN in stable environment
        if replay_status == "WARN" and topology_mode == "STABLE":
            self._mark_contributed(attribution_chain, "replay")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.REPLAY_FAILURE,
                root_cause_confidence=0.70,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Minor replay concern warrants investigation",
                headline="REPLAY_FAILURE: Minor replay concern in stable environment",
                attribution_chain=attribution_chain,
            )

        # Rule 8: UNKNOWN - Default fallback
        return DiagnosticResult(
            root_cause_hypothesis=RootCauseHypothesis.UNKNOWN,
            root_cause_confidence=0.50,
            action=DiagnosticAction.REVIEW_RECOMMENDED,
            action_rationale="Pattern not recognized by rule engine",
            headline=f"UNKNOWN: Replay={replay_status}, Topology={topology_mode}, Budget={budget_stability}",
            attribution_chain=attribution_chain,
            confounding_factors=confounding_factors,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _normalize_status(self, status: Any) -> str:
        """Normalize status to uppercase string."""
        if hasattr(status, "value"):
            return str(status.value).upper()
        return str(status).upper()

    def _evaluate_identity(self, signal: Dict[str, Any]) -> str:
        """Evaluate identity signal status."""
        if not signal.get("block_hash_valid", True):
            return "BLOCK"
        if not signal.get("chain_continuous", True):
            return "BLOCK"
        if not signal.get("merkle_root_valid", True):
            return "BLOCK"
        if not signal.get("dual_root_consistent", True):
            return "BLOCK"
        if not signal.get("signature_valid", True):
            return "BLOCK"
        if not signal.get("pq_attestation_valid", True):
            return "BLOCK"
        return "OK"

    def _evaluate_structure(self, signal: Dict[str, Any]) -> str:
        """Evaluate structure signal status."""
        if not signal.get("dag_coherent", True):
            return "BLOCK"
        if signal.get("cycle_detected", False):
            return "BLOCK"
        return "OK"

    def _get_identity_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for identity status."""
        if status == "BLOCK":
            if not signal.get("block_hash_valid", True):
                return "Block hash validation failed"
            if not signal.get("chain_continuous", True):
                return "Chain continuity broken"
            if not signal.get("merkle_root_valid", True):
                return "Merkle root invalid"
            if not signal.get("dual_root_consistent", True):
                return "Dual root inconsistent"
            if not signal.get("signature_valid", True):
                return "Signature validation failed"
            if not signal.get("pq_attestation_valid", True):
                return "PQ attestation invalid"
            return "Identity violation detected"
        return "Cryptographic integrity intact"

    def _get_structure_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for structure status."""
        if status == "BLOCK":
            if not signal.get("dag_coherent", True):
                return "DAG coherence check failed"
            if signal.get("cycle_detected", False):
                return "Cycle detected in DAG"
            return "Structure violation detected"
        return "DAG coherent"

    def _get_replay_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for replay status."""
        reasons = signal.get("reasons", [])
        if reasons:
            return reasons[0] if len(reasons) == 1 else f"{len(reasons)} issues"
        if status == "OK":
            return "Determinism confirmed"
        if status == "WARN":
            return "Minor replay concern"
        return "Replay verification failed"

    def _collect_supporting_signals(
        self,
        replay: Dict[str, Any],
        topology: Dict[str, Any],
        budget: Dict[str, Any],
        divergence: Dict[str, Any],
        identity: Optional[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect supporting signals for output."""
        supporting: Dict[str, Any] = {}

        supporting["replay"] = {
            "status": self._normalize_status(replay.get("status", "OK")),
            "alignment": str(
                replay.get("governance_alignment", replay.get("alignment", "aligned"))
            ).lower(),
            "conflict": replay.get("conflict", False),
            "reasons": replay.get("reasons", []),
        }

        supporting["topology"] = {
            "mode": topology.get("mode", "STABLE").upper(),
            "persistence_drift": topology.get("persistence_drift"),
            "betti_0": topology.get("betti_0") or topology.get("betti", {}).get("b0"),
            "betti_1": topology.get("betti_1") or topology.get("betti", {}).get("b1"),
            "within_omega": topology.get("within_omega"),
        }

        supporting["budget"] = {
            "stability_class": budget.get("stability_class", "STABLE").upper(),
            "health_score": budget.get("health_score"),
            "stability_index": budget.get("stability_index"),
        }

        supporting["divergence"] = {
            "severity": divergence.get("severity", "NONE").upper(),
            "type": divergence.get("type", "NONE").upper(),
            "divergence_pct": divergence.get("divergence_pct"),
            "H_diff": divergence.get("H_diff"),
            "rho_diff": divergence.get("rho_diff"),
        }

        if identity:
            supporting["identity"] = {
                "status": self._evaluate_identity(identity),
                "hash_valid": identity.get("block_hash_valid", True),
                "chain_continuous": identity.get("chain_continuous", True),
            }

        if structure:
            supporting["structure"] = {
                "status": self._evaluate_structure(structure),
                "dag_coherent": structure.get("dag_coherent", True),
                "cycle_detected": structure.get("cycle_detected", False),
            }

        return supporting

    @staticmethod
    def _mark_contributed(chain: List[AttributionStep], signal: str) -> None:
        """Mark a signal as having contributed to the diagnosis."""
        for step in chain:
            if step.signal == signal:
                step.contributed_to_diagnosis = True
                break


# =============================================================================
# Convenience Functions
# =============================================================================


def interpret_p5_divergence(
    divergence_snapshot: Dict[str, Any],
    replay_signal: Dict[str, Any],
    topology_signal: Dict[str, Any],
    budget_signal: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function for P5 divergence interpretation.

    SHADOW MODE: Output is observational only.

    Args:
        divergence_snapshot: P4 divergence snapshot
        replay_signal: Replay safety governance signal
        topology_signal: Topology bundle signal
        budget_signal: Budget signal
        **kwargs: Additional arguments (identity_signal, structure_signal, cycle, run_id)

    Returns:
        Diagnostic panel dict conforming to p5_divergence_diagnostic.schema.json
    """
    interpreter = P5DivergenceInterpreter()
    return interpreter.interpret_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
        **kwargs,
    )


def diagnose_from_p4_artifacts(
    divergence_log_path: str,
    replay_signal: Dict[str, Any],
    topology_signal: Dict[str, Any],
    budget_signal: Dict[str, Any],
    cycle_index: int = -1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate diagnostic from existing P4 divergence log.

    SHADOW MODE: Output is observational only.

    Args:
        divergence_log_path: Path to p4_divergence_log.jsonl
        replay_signal: Current replay safety signal
        topology_signal: Current topology signal
        budget_signal: Current budget signal
        cycle_index: Index into divergence log (-1 for latest)
        **kwargs: Additional arguments passed to interpret_p5_divergence

    Returns:
        Diagnostic panel dict
    """
    import json
    from pathlib import Path

    log_path = Path(divergence_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Divergence log not found: {log_path}")

    # Read JSONL and get specified entry
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        raise ValueError("Divergence log is empty")

    snapshot = entries[cycle_index]

    return interpret_p5_divergence(
        divergence_snapshot=snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
        cycle=snapshot.get("cycle", 0),
        **kwargs,
    )


# =============================================================================
# Director Tile Integration
# =============================================================================


def build_p5_diagnostic_tile(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a P5 diagnostic tile for director panel integration.

    This tile is designed to wire into build_replay_safety_director_tile()
    or the global director panel.

    Args:
        diagnostic: Output from interpret_p5_divergence()

    Returns:
        Director tile dict with summary for dashboard display
    """
    hypothesis = diagnostic.get("root_cause_hypothesis", "UNKNOWN")
    confidence = diagnostic.get("root_cause_confidence", 0.5)
    action = diagnostic.get("action", "REVIEW_RECOMMENDED")
    headline = diagnostic.get("headline", "")

    # Determine severity badge
    if action == "INVESTIGATION_REQUIRED":
        severity_badge = "CRITICAL"
    elif action == "REVIEW_RECOMMENDED":
        if confidence >= 0.8:
            severity_badge = "WARN"
        else:
            severity_badge = "INFO"
    else:
        severity_badge = "OK"

    # Build tile
    return {
        "tile_type": "p5_diagnostic",
        "schema_version": "1.0.0",
        "timestamp": diagnostic.get("timestamp"),
        "cycle": diagnostic.get("cycle", 0),
        "summary": {
            "hypothesis": hypothesis,
            "confidence": confidence,
            "severity_badge": severity_badge,
            "headline": headline,
        },
        "signal_snapshot": {
            "replay_status": diagnostic.get("replay_status", "OK"),
            "topology_mode": diagnostic.get("topology_mode", "STABLE"),
            "budget_stability": diagnostic.get("budget_stability", "STABLE"),
            "divergence_severity": diagnostic.get("divergence_severity", "NONE"),
        },
        "action": {
            "level": action,
            "rationale": diagnostic.get("action_rationale", ""),
        },
        "shadow_mode_notice": "P5 diagnostic is observational only",
    }


def attach_p5_diagnostic_to_evidence(
    evidence: Dict[str, Any],
    diagnostic: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach P5 diagnostic to evidence pack.

    Args:
        evidence: Evidence pack dict
        diagnostic: Output from interpret_p5_divergence()

    Returns:
        Updated evidence pack with p5_diagnostic key
    """
    evidence["p5_diagnostic"] = {
        "schema_version": diagnostic.get("schema_version", "1.0.0"),
        "cycle": diagnostic.get("cycle", 0),
        "root_cause_hypothesis": diagnostic.get("root_cause_hypothesis", "UNKNOWN"),
        "root_cause_confidence": diagnostic.get("root_cause_confidence", 0.5),
        "action": diagnostic.get("action", "REVIEW_RECOMMENDED"),
        "headline": diagnostic.get("headline", ""),
        "signal_summary": {
            "replay": diagnostic.get("replay_status", "OK"),
            "topology": diagnostic.get("topology_mode", "STABLE"),
            "budget": diagnostic.get("budget_stability", "STABLE"),
            "divergence": diagnostic.get("divergence_severity", "NONE"),
        },
        "confounding_factors": diagnostic.get("confounding_factors", []),
        "shadow_mode": True,
    }
    return evidence


# =============================================================================
# GGFL Adapter
# =============================================================================


def p5_diagnostic_for_alignment_view(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare P5 diagnostic for Global Governance Fusion Layer (GGFL) alignment view.

    Returns a GGFL-compatible signal stub with status, hypothesis, and severity.
    Influence is kept low - this is advisory only.

    SHADOW MODE CONTRACT:
    - Output is observational only
    - No influence on governance decisions
    - Low weight in fusion calculations

    Args:
        diagnostic: Output from interpret_p5_divergence()

    Returns:
        GGFL-compatible signal dict suitable for build_global_alignment_view()
    """
    hypothesis = diagnostic.get("root_cause_hypothesis", "UNKNOWN")
    action = diagnostic.get("action", "REVIEW_RECOMMENDED")
    confidence = diagnostic.get("root_cause_confidence", 0.5)

    # Map action to status
    if action == "INVESTIGATION_REQUIRED":
        status = "unhealthy"
    elif action == "REVIEW_RECOMMENDED":
        status = "degraded"
    else:
        status = "healthy"

    # Map hypothesis to severity for GGFL
    severity_map = {
        "NOMINAL": "INFO",
        "REPLAY_FAILURE": "WARN",
        "STRUCTURAL_BREAK": "WARN",
        "PHASE_LAG": "WARN",
        "BUDGET_CONFOUND": "INFO",
        "IDENTITY_VIOLATION": "CRITICAL",
        "STRUCTURE_VIOLATION": "CRITICAL",
        "CASCADING_FAILURE": "CRITICAL",
        "UNKNOWN": "INFO",
    }
    severity = severity_map.get(hypothesis, "INFO")

    return {
        "signal_type": "p5_diagnostic",
        "status": status,
        "hypothesis": hypothesis,
        "severity": severity,
        "confidence": confidence,
        "action": action,
        "headline": diagnostic.get("headline", ""),
        "signal_summary": {
            "replay_status": diagnostic.get("replay_status", "OK"),
            "topology_mode": diagnostic.get("topology_mode", "STABLE"),
            "budget_stability": diagnostic.get("budget_stability", "STABLE"),
            "divergence_severity": diagnostic.get("divergence_severity", "NONE"),
        },
        "advisory_only": True,
        "shadow_mode": True,
        # Low weight in fusion - advisory signal
        "weight": 0.1,
    }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "RootCauseHypothesis",
    "DiagnosticAction",
    # Data classes
    "AttributionStep",
    "DiagnosticResult",
    "SupportingSignals",
    # Core interpreter
    "P5DivergenceInterpreter",
    # Convenience functions
    "interpret_p5_divergence",
    "diagnose_from_p4_artifacts",
    # Director tile integration
    "build_p5_diagnostic_tile",
    "attach_p5_diagnostic_to_evidence",
    # GGFL adapter
    "p5_diagnostic_for_alignment_view",
]
