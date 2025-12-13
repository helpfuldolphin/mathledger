"""
Phase X P4: Divergence Analyzer for Real vs Twin Comparison

This module implements the DivergenceAnalyzer for comparing real runner
observations against shadow twin predictions.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- Analysis is for LOGGING only
- Divergences do NOT trigger remediation
- No feedback flows from analysis to real execution

Status: P4 DESIGN FREEZE (STUBS ONLY)

TODO[PhaseX-Divergence-Metric]: Implement severity classification per
    docs/system_law/Phase_X_Divergence_Metric.md
    Thresholds: NONE < 0.01, INFO < 0.05, WARN < 0.15, CRITICAL >= 0.15
    Epsilon floor: 0.001

TODO[PhaseX-TDA-P4]: Include TDA divergence in analysis.
    - Compare SNS, PCS, DRS, HSS between twin and real
    - Log TDA divergence alongside Δp divergence

TODO[P4-BUDGET-TDA-001]: Budget-modulated TDA interpretation
    Budget instability affects how divergence should be interpreted.
    See docs/system_law/Budget_PhaseX_Doctrine.md Section 3.3

    Implementation (when authorized):
    1. Import GovernanceSignal from backend.analytics.governance_verifier
    2. Consume budget layer signal for stability_class metadata
    3. Apply severity multiplier based on budget stability:
       - STABLE: 1.0 (no adjustment)
       - DRIFTING: 0.7
       - VOLATILE: 0.4
    4. Update root_cause_vector with budget attribution when unstable
    5. Flag divergence records with "budget_confounded" marker
    6. Add tda_context field to DivergenceSnapshot: NOMINAL | BUDGET_DRIFT | BUDGET_UNSTABLE

    Severity Multiplier Logic:
    - When budget_stability_class == "VOLATILE":
      * Adjust divergence_severity by 0.4 multiplier
      * Add "BUDGET" to root_cause_attribution
    - When budget_stability_class == "DRIFTING":
      * Adjust divergence_severity by 0.7 multiplier
      * Consider budget as contributing factor

    Dependencies:
    - budget_governance_signal.schema.json
    - budget_director_panel.schema.json
    - GovernanceSignal from governance_verifier.py
    - LAYER_BUDGET constant

    Status: NOT AUTHORIZED (requires P4 execution auth)

TODO[CLAUDE-G-Structural]: Structural invariants feeding into divergence analyzer
    The _classify_severity() method should incorporate structural layer signals:
      1. Accept StructuralGovernanceSignal as optional parameter
      2. Escalate severity when structural CONFLICT detected
      3. Adjust confidence metrics based on cohesion_score
    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 4.4

    Implementation (when authorized):
    - Import StructuralGovernanceSignal schema types
    - Add structural_signal parameter to analyze() method
    - In _classify_severity():
      * If structural_signal.combined_severity == "CONFLICT":
        - Escalate any divergence to SEVERE
        - Set structural_conflict flag in snapshot
      * If structural_signal.combined_severity == "TENSION":
        - Escalate MINOR -> MODERATE
        - Add cohesion_degraded flag
    - Adjust confidence in summary by cohesion_score factor

    Status: NOT AUTHORIZED (requires P4 execution auth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.topology.first_light.data_structures_p4 import (
    DivergenceSnapshot,
    RealCycleObservation,
    TwinCycleObservation,
)
from backend.topology.first_light.budget_binding import (
    BudgetRiskSignal,
    BudgetStabilityClass,
    build_budget_risk_signal,
    adjust_divergence_severity,
    compute_tda_context,
)

__all__ = [
    "DivergenceAnalyzer",
    "DivergenceSummary",
    "DivergenceThresholds",
    "BudgetAwareDivergenceSummary",
    # Structural governance exports (CLAUDE G)
    "apply_structural_severity_escalation",
    "analyze_with_structural_context",
]


@dataclass
class DivergenceThresholds:
    """
    Thresholds for divergence classification.

    SHADOW MODE: These thresholds are for LOGGING classification only.
    They do NOT trigger any enforcement actions.
    """

    # State divergence thresholds
    H_threshold: float = 0.1
    rho_threshold: float = 0.1
    tau_threshold: float = 0.05
    beta_threshold: float = 0.1

    # Streak thresholds (for LOGGING escalation only)
    minor_streak_threshold: int = 5
    moderate_streak_threshold: int = 10
    severe_streak_threshold: int = 20


@dataclass
class DivergenceSummary:
    """
    Summary of divergence analysis across a run.

    SHADOW MODE: This summary is observational only.
    """

    # Total counts
    total_comparisons: int = 0
    total_divergences: int = 0

    # By type
    state_divergences: int = 0
    outcome_divergences: int = 0
    combined_divergences: int = 0

    # By severity
    minor_divergences: int = 0
    moderate_divergences: int = 0
    severe_divergences: int = 0

    # Streak tracking
    max_divergence_streak: int = 0
    current_streak: int = 0

    # Accuracy metrics (twin prediction accuracy)
    success_matches: int = 0
    blocked_matches: int = 0
    omega_matches: int = 0
    hard_ok_matches: int = 0

    # Computed rates
    divergence_rate: float = 0.0
    success_accuracy: float = 0.0
    blocked_accuracy: float = 0.0
    omega_accuracy: float = 0.0
    hard_ok_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_comparisons": self.total_comparisons,
            "total_divergences": self.total_divergences,
            "by_type": {
                "state": self.state_divergences,
                "outcome": self.outcome_divergences,
                "combined": self.combined_divergences,
            },
            "by_severity": {
                "minor": self.minor_divergences,
                "moderate": self.moderate_divergences,
                "severe": self.severe_divergences,
            },
            "streaks": {
                "max_divergence_streak": self.max_divergence_streak,
                "current_streak": self.current_streak,
            },
            "accuracy": {
                "divergence_rate": round(self.divergence_rate, 4),
                "success_accuracy": round(self.success_accuracy, 4),
                "blocked_accuracy": round(self.blocked_accuracy, 4),
                "omega_accuracy": round(self.omega_accuracy, 4),
                "hard_ok_accuracy": round(self.hard_ok_accuracy, 4),
            },
        }


@dataclass
class BudgetAwareDivergenceSummary:
    """
    Extended divergence summary with budget-modulated severity.

    SHADOW MODE CONTRACT:
    - Budget multipliers are computed but do NOT alter behavior
    - Adjusted severity counts are for analysis only
    - No remediation is triggered by any values

    See: docs/system_law/Budget_PhaseX_Doctrine.md Section 3.3
    """

    # Base summary
    base_summary: DivergenceSummary = field(default_factory=DivergenceSummary)

    # Budget context
    budget_signal: BudgetRiskSignal = field(
        default_factory=lambda: build_budget_risk_signal()
    )

    # Adjusted severity counts (SHADOW: for analysis only)
    adjusted_minor_divergences: int = 0
    adjusted_moderate_divergences: int = 0
    adjusted_severe_divergences: int = 0

    # TDA context
    tda_context: str = "NOMINAL"

    # Budget-confounded divergences (where budget instability may explain divergence)
    budget_confounded_divergences: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with budget context."""
        base = self.base_summary.to_dict()
        base["budget_context"] = {
            "stability_class": self.budget_signal.stability_class.value,
            "severity_multiplier": round(self.budget_signal.severity_multiplier, 4),
            "tda_context": self.tda_context,
            "budget_confounded": self.budget_signal.budget_confounded,
            "budget_confounded_divergences": self.budget_confounded_divergences,
            "health_score": round(self.budget_signal.health_score, 2),
            "stability_index": round(self.budget_signal.stability_index, 4),
        }
        base["adjusted_severity"] = {
            "minor": self.adjusted_minor_divergences,
            "moderate": self.adjusted_moderate_divergences,
            "severe": self.adjusted_severe_divergences,
            "note": "SHADOW MODE: adjusted counts for analysis only",
        }
        return base

    @classmethod
    def from_summary_and_budget(
        cls,
        summary: DivergenceSummary,
        health_score: float = 100.0,
        stability_index: float = 1.0,
        inv_bud_failures: Optional[List[str]] = None,
    ) -> "BudgetAwareDivergenceSummary":
        """
        Create budget-aware summary from base summary and budget metrics.

        Applies severity multiplier to adjust divergence counts for analysis.

        Args:
            summary: Base DivergenceSummary
            health_score: Budget health score
            stability_index: Stability index
            inv_bud_failures: List of failing invariants

        Returns:
            BudgetAwareDivergenceSummary with adjusted counts
        """
        budget_signal = build_budget_risk_signal(
            drift_value=0.0,  # P4 uses stability class, not drift
            health_score=health_score,
            stability_index=stability_index,
            inv_bud_failures=inv_bud_failures,
        )

        multiplier = budget_signal.severity_multiplier
        tda_context = compute_tda_context(budget_signal)

        # Compute adjusted severity counts
        # Interpretation: multiplier < 1.0 means some severe -> moderate, etc.
        # This is a simplified model for analysis

        # With multiplier 0.4 (VOLATILE): most severe become moderate
        # With multiplier 0.7 (DRIFTING): some severe become moderate

        if multiplier < 0.5:
            # VOLATILE: Severe -> Moderate, Moderate -> Minor
            adjusted_severe = 0
            adjusted_moderate = summary.severe_divergences
            adjusted_minor = summary.minor_divergences + summary.moderate_divergences
        elif multiplier < 0.8:
            # DRIFTING: Some severe -> moderate
            adjusted_severe = int(summary.severe_divergences * multiplier)
            adjusted_moderate = (
                summary.moderate_divergences +
                (summary.severe_divergences - adjusted_severe)
            )
            adjusted_minor = summary.minor_divergences
        else:
            # STABLE: No adjustment
            adjusted_severe = summary.severe_divergences
            adjusted_moderate = summary.moderate_divergences
            adjusted_minor = summary.minor_divergences

        # Count budget-confounded divergences
        budget_confounded_count = 0
        if budget_signal.budget_confounded:
            budget_confounded_count = summary.total_divergences

        return cls(
            base_summary=summary,
            budget_signal=budget_signal,
            adjusted_minor_divergences=adjusted_minor,
            adjusted_moderate_divergences=adjusted_moderate,
            adjusted_severe_divergences=adjusted_severe,
            tda_context=tda_context,
            budget_confounded_divergences=budget_confounded_count,
        )


class DivergenceAnalyzer:
    """
    Analyzes divergence between real runner and shadow twin.

    SHADOW MODE CONTRACT:
    - Analysis is for LOGGING only
    - Divergences do NOT trigger remediation
    - No feedback flows from analysis to real execution
    - All methods are observation-only

    See: docs/system_law/Phase_X_P4_Spec.md Section 4.1
    """

    def __init__(
        self,
        thresholds: Optional[DivergenceThresholds] = None,
    ) -> None:
        """
        Initialize divergence analyzer.

        Args:
            thresholds: Divergence classification thresholds
        """
        self._thresholds = thresholds or DivergenceThresholds()
        self._history: List[DivergenceSnapshot] = []
        self._summary = DivergenceSummary()
        self._current_streak = 0
        self._streak_start: Optional[int] = None

    def analyze(
        self,
        real: RealCycleObservation,
        twin: TwinCycleObservation,
    ) -> DivergenceSnapshot:
        """
        Analyze divergence between real and twin observations.

        SHADOW MODE: This analysis is for LOGGING only.
        The returned DivergenceSnapshot has action="LOGGED_ONLY".

        Args:
            real: Real runner observation
            twin: Shadow twin prediction

        Returns:
            DivergenceSnapshot with divergence analysis
        """
        # Build thresholds dict for from_observations
        thresholds = {
            "state_threshold": 0.05,
            "epsilon": 0.001,
            "threshold_none": 0.01,
            "threshold_info": 0.05,
            "threshold_warn": 0.15,
        }

        # Create divergence snapshot using the class method
        snapshot = DivergenceSnapshot.from_observations(
            real=real,
            twin=twin,
            thresholds=thresholds,
            consecutive=self._current_streak,
            streak_start=self._streak_start,
        )

        # Update streak tracking
        self._update_streaks(snapshot.is_diverged())

        # Update summary statistics
        self._update_summary(real, twin, snapshot)

        # Record in history
        self._history.append(snapshot)

        return snapshot

    def analyze_batch(
        self,
        pairs: List[Tuple[RealCycleObservation, TwinCycleObservation]],
    ) -> List[DivergenceSnapshot]:
        """
        Analyze divergence for multiple observation pairs.

        Args:
            pairs: List of (real, twin) observation pairs

        Returns:
            List of DivergenceSnapshot for each pair
        """
        results: List[DivergenceSnapshot] = []
        for real, twin in pairs:
            snapshot = self.analyze(real, twin)
            results.append(snapshot)
        return results

    def get_summary(self) -> DivergenceSummary:
        """
        Get summary of all divergence analysis.

        Returns:
            DivergenceSummary with accumulated statistics
        """
        # Compute rates before returning
        if self._summary.total_comparisons > 0:
            self._summary.divergence_rate = (
                self._summary.total_divergences / self._summary.total_comparisons
            )
            self._summary.success_accuracy = (
                self._summary.success_matches / self._summary.total_comparisons
            )
            self._summary.blocked_accuracy = (
                self._summary.blocked_matches / self._summary.total_comparisons
            )
            self._summary.omega_accuracy = (
                self._summary.omega_matches / self._summary.total_comparisons
            )
            self._summary.hard_ok_accuracy = (
                self._summary.hard_ok_matches / self._summary.total_comparisons
            )
        return self._summary

    def get_divergence_history(self) -> List[DivergenceSnapshot]:
        """
        Get history of all divergence snapshots.

        Returns:
            List of all DivergenceSnapshot objects recorded
        """
        return list(self._history)

    def get_current_streak(self) -> int:
        """
        Get current divergence streak length.

        Returns:
            Number of consecutive divergent cycles
        """
        return self._current_streak

    def hypothetical_should_alert(self) -> Tuple[bool, Optional[str]]:
        """
        Check if alert WOULD be triggered (analysis only, NEVER enforced).

        SHADOW MODE: This is for analysis only. No alert is actually sent.

        Returns:
            Tuple of (would_alert, reason)
        """
        # Check severe streak threshold
        if self._current_streak >= self._thresholds.severe_streak_threshold:
            return (
                True,
                f"Severe divergence streak: {self._current_streak} cycles "
                f"(threshold: {self._thresholds.severe_streak_threshold})",
            )

        # Check high divergence rate
        if self._summary.total_comparisons >= 10:
            rate = self._summary.total_divergences / self._summary.total_comparisons
            if rate >= 0.30:
                return (True, f"High divergence rate: {rate:.2%}")

        return (False, None)

    def reset(self) -> None:
        """Reset analyzer state."""
        self._history.clear()
        self._summary = DivergenceSummary()
        self._current_streak = 0
        self._streak_start = None

    def _classify_severity(
        self,
        snapshot: DivergenceSnapshot,
    ) -> str:
        """
        Classify divergence severity.

        Args:
            snapshot: Divergence snapshot to classify

        Returns:
            Severity string: "NONE", "MINOR", "MODERATE", or "SEVERE"
        """
        # Map from DivergenceSnapshot severity to our classification
        severity_map = {
            "NONE": "NONE",
            "INFO": "MINOR",
            "WARN": "MODERATE",
            "CRITICAL": "SEVERE",
        }
        return severity_map.get(snapshot.divergence_severity, "NONE")

    def _classify_type(
        self,
        snapshot: DivergenceSnapshot,
    ) -> str:
        """
        Classify divergence type.

        Args:
            snapshot: Divergence snapshot to classify

        Returns:
            Type string: "NONE", "STATE", "OUTCOME", or "BOTH"
        """
        return snapshot.divergence_type

    def _update_streaks(
        self,
        is_diverged: bool,
    ) -> None:
        """
        Update divergence streak tracking.

        Args:
            is_diverged: Whether current cycle diverged
        """
        if is_diverged:
            self._current_streak += 1
            if self._streak_start is None:
                self._streak_start = len(self._history)
            self._summary.max_divergence_streak = max(
                self._summary.max_divergence_streak, self._current_streak
            )
        else:
            self._current_streak = 0
            self._streak_start = None
        self._summary.current_streak = self._current_streak

    def _update_summary(
        self,
        real: RealCycleObservation,
        twin: TwinCycleObservation,
        snapshot: DivergenceSnapshot,
    ) -> None:
        """
        Update summary statistics after analyzing a pair.

        Args:
            real: Real observation
            twin: Twin prediction
            snapshot: Resulting divergence snapshot
        """
        self._summary.total_comparisons += 1

        # Track accuracy matches
        if real.success == twin.predicted_success:
            self._summary.success_matches += 1
        if real.real_blocked == twin.predicted_blocked:
            self._summary.blocked_matches += 1
        if real.in_omega == twin.predicted_in_omega:
            self._summary.omega_matches += 1
        if real.hard_ok == twin.predicted_hard_ok:
            self._summary.hard_ok_matches += 1

        # Track divergences
        if snapshot.is_diverged():
            self._summary.total_divergences += 1

            # By type
            div_type = self._classify_type(snapshot)
            if div_type == "STATE":
                self._summary.state_divergences += 1
            elif div_type == "OUTCOME":
                self._summary.outcome_divergences += 1
            elif div_type == "BOTH":
                self._summary.combined_divergences += 1

            # By severity
            severity = self._classify_severity(snapshot)
            if severity == "MINOR":
                self._summary.minor_divergences += 1
            elif severity == "MODERATE":
                self._summary.moderate_divergences += 1
            elif severity == "SEVERE":
                self._summary.severe_divergences += 1


# =============================================================================
# Structural Severity Escalation (CLAUDE G Integration)
# =============================================================================

def apply_structural_severity_escalation(
    snapshot: DivergenceSnapshot,
    structural_signal: Optional[Dict[str, Any]] = None,
) -> DivergenceSnapshot:
    """
    Apply structural severity escalation to a divergence snapshot.

    SHADOW MODE: This escalation is for LOGGING/analysis only.
    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 4.4

    Escalation rules:
    - If combined_severity == "CONFLICT": escalate any divergence to CRITICAL
    - If combined_severity == "TENSION": escalate INFO -> WARN, WARN -> CRITICAL
    - cohesion_score < 0.8: set cohesion_degraded flag

    Args:
        snapshot: DivergenceSnapshot to potentially escalate
        structural_signal: StructuralGovernanceSignal.to_dict() or None

    Returns:
        Updated DivergenceSnapshot with structural fields populated
    """
    if structural_signal is None:
        # No structural signal, return unchanged (default structural fields)
        return snapshot

    # Extract structural values
    combined_severity = structural_signal.get("combined_severity", "CONSISTENT")
    cohesion_score = structural_signal.get("cohesion_score", 1.0)
    admissible = structural_signal.get("admissible", True)

    # Save original severity
    original_severity = snapshot.divergence_severity
    new_severity = original_severity
    severity_escalated = False
    structural_conflict = False
    cohesion_degraded = False

    # Check for structural CONFLICT (SI-001 or SI-010 violated)
    if combined_severity == "CONFLICT" or not admissible:
        structural_conflict = True
        # Escalate any non-NONE severity to CRITICAL
        if original_severity != "NONE":
            new_severity = "CRITICAL"
            severity_escalated = (new_severity != original_severity)

    # Check for structural TENSION
    elif combined_severity == "TENSION":
        # Escalate INFO -> WARN, WARN -> CRITICAL
        if original_severity == "INFO":
            new_severity = "WARN"
            severity_escalated = True
        elif original_severity == "WARN":
            new_severity = "CRITICAL"
            severity_escalated = True

    # Check cohesion degradation
    if cohesion_score < 0.8:
        cohesion_degraded = True

    # Update snapshot fields
    snapshot.structural_conflict = structural_conflict
    snapshot.cohesion_degraded = cohesion_degraded
    snapshot.cohesion_score = cohesion_score
    snapshot.original_severity = original_severity
    snapshot.severity_escalated = severity_escalated

    if severity_escalated:
        snapshot.divergence_severity = new_severity

    return snapshot


def analyze_with_structural_context(
    analyzer: "DivergenceAnalyzer",
    real: RealCycleObservation,
    twin: TwinCycleObservation,
    structural_signal: Optional[Dict[str, Any]] = None,
) -> DivergenceSnapshot:
    """
    Analyze divergence with structural context applied.

    SHADOW MODE: Structural escalation is for LOGGING only.

    Args:
        analyzer: DivergenceAnalyzer instance
        real: Real runner observation
        twin: Shadow twin prediction
        structural_signal: StructuralGovernanceSignal.to_dict() or None

    Returns:
        DivergenceSnapshot with structural fields populated
    """
    # First do standard analysis
    snapshot = analyzer.analyze(real, twin)

    # Then apply structural escalation
    if structural_signal is not None:
        snapshot = apply_structural_severity_escalation(snapshot, structural_signal)

    return snapshot
