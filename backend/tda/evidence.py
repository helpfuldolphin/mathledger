"""
TDA Evidence Pack Attachment

Provides functions to attach TDA metrics to Evidence Pack governance section.

See: docs/system_law/TDA_PhaseX_Binding.md Section 8 (External Verifier)
See: docs/system_law/Evidence_Pack_Spec_PhaseX.md

SHADOW MODE CONTRACT:
- All TDA evidence is observational only
- No governance modification based on TDA values
- Evidence attachment is for audit trail and external verification
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.tda.monitor import TDASummary
from backend.tda.metrics import TDAWindowMetrics

__all__ = [
    "attach_tda_to_evidence",
    "format_tda_evidence_summary",
    "compute_topology_match_score",
    "TDAEvidenceBlock",
]


# =============================================================================
# TDA Evidence Block
# =============================================================================

class TDAEvidenceBlock:
    """
    TDA evidence block for Evidence Pack governance section.

    This block encapsulates all TDA metrics for external verifier consumption.
    """

    def __init__(
        self,
        p3_summary: Optional[TDASummary] = None,
        p4_summary: Optional[TDASummary] = None,
        p3_windows: Optional[List[TDAWindowMetrics]] = None,
        p4_windows: Optional[List[TDAWindowMetrics]] = None,
    ):
        """
        Initialize TDA evidence block.

        Args:
            p3_summary: P3 First-Light TDA summary
            p4_summary: P4 real-coupling TDA summary
            p3_windows: P3 per-window TDA metrics (optional)
            p4_windows: P4 per-window TDA metrics (optional)
        """
        self.p3_summary = p3_summary
        self.p4_summary = p4_summary
        self.p3_windows = p3_windows or []
        self.p4_windows = p4_windows or []
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "generated_at": self.generated_at,
            "p3_synthetic": self._format_p3_section(),
            "p4_shadow": self._format_p4_section(),
            "topology_matching": self._compute_topology_matching(),
            "verifier_guidance": self._verifier_guidance(),
        }

    def _format_p3_section(self) -> Dict[str, Any]:
        """Format P3 TDA evidence section."""
        if not self.p3_summary:
            return {"available": False, "reason": "P3 TDA summary not provided"}

        return {
            "available": True,
            "summary": self.p3_summary.to_dict(),
            "metrics": {
                "sns": {
                    "mean": round(self.p3_summary.sns_mean, 6),
                    "max": round(self.p3_summary.sns_max, 6),
                    "anomaly_count": self.p3_summary.sns_anomaly_count,
                    "interpretation": self._interpret_sns(self.p3_summary.sns_mean),
                },
                "pcs": {
                    "mean": round(self.p3_summary.pcs_mean, 6),
                    "min": round(self.p3_summary.pcs_min, 6),
                    "collapse_count": self.p3_summary.pcs_collapse_count,
                    "interpretation": self._interpret_pcs(self.p3_summary.pcs_mean),
                },
                "hss": {
                    "mean": round(self.p3_summary.hss_mean, 6),
                    "min": round(self.p3_summary.hss_min, 6),
                    "degradation_count": self.p3_summary.hss_degradation_count,
                    "interpretation": self._interpret_hss(self.p3_summary.hss_mean),
                },
            },
            "envelope": {
                "occupancy": round(self.p3_summary.envelope_occupancy, 4),
                "exit_total": self.p3_summary.envelope_exit_total,
                "max_exit_streak": self.p3_summary.max_envelope_exit_streak,
                "interpretation": self._interpret_envelope(self.p3_summary.envelope_occupancy),
            },
            "red_flags": {
                "total": self.p3_summary.total_red_flags,
                "by_type": dict(self.p3_summary.red_flags_by_type),
            },
            "window_count": len(self.p3_windows),
        }

    def _format_p4_section(self) -> Dict[str, Any]:
        """Format P4 TDA evidence section."""
        if not self.p4_summary:
            return {"available": False, "reason": "P4 TDA summary not provided"}

        return {
            "available": True,
            "summary": self.p4_summary.to_dict(),
            "metrics": {
                "drs": {
                    "mean": round(self.p4_summary.drs_mean, 6),
                    "max": round(self.p4_summary.drs_max, 6),
                    "critical_count": self.p4_summary.drs_critical_count,
                    "interpretation": self._interpret_drs(self.p4_summary.drs_mean),
                },
                "sns": {
                    "mean": round(self.p4_summary.sns_mean, 6),
                    "max": round(self.p4_summary.sns_max, 6),
                },
                "pcs": {
                    "mean": round(self.p4_summary.pcs_mean, 6),
                    "min": round(self.p4_summary.pcs_min, 6),
                },
                "hss": {
                    "mean": round(self.p4_summary.hss_mean, 6),
                    "min": round(self.p4_summary.hss_min, 6),
                },
            },
            "envelope": {
                "occupancy": round(self.p4_summary.envelope_occupancy, 4),
                "exit_total": self.p4_summary.envelope_exit_total,
                "max_exit_streak": self.p4_summary.max_envelope_exit_streak,
            },
            "window_count": len(self.p4_windows),
        }

    def _compute_topology_matching(self) -> Dict[str, Any]:
        """
        Compute topology matching between P3 and P4.

        External verifiers use this to assess whether synthetic (P3) and
        real (P4) observations exhibit consistent topological structure.
        """
        if not self.p3_summary or not self.p4_summary:
            return {
                "available": False,
                "reason": "Both P3 and P4 summaries required for matching",
            }

        # Compute per-metric deltas
        sns_delta = abs(self.p3_summary.sns_mean - self.p4_summary.sns_mean)
        pcs_delta = abs(self.p3_summary.pcs_mean - self.p4_summary.pcs_mean)
        hss_delta = abs(self.p3_summary.hss_mean - self.p4_summary.hss_mean)
        envelope_delta = abs(
            self.p3_summary.envelope_occupancy - self.p4_summary.envelope_occupancy
        )

        # Compute match score (1.0 = perfect match, 0.0 = complete mismatch)
        # Using inverse of normalized deltas
        match_score = compute_topology_match_score(
            self.p3_summary, self.p4_summary
        )

        # Classify match quality
        if match_score >= 0.9:
            match_quality = "EXCELLENT"
            match_interpretation = (
                "P3 synthetic and P4 real observations exhibit highly consistent "
                "topological structure. Twin model accurately captures system dynamics."
            )
        elif match_score >= 0.75:
            match_quality = "GOOD"
            match_interpretation = (
                "P3 and P4 observations show good topological agreement. "
                "Minor discrepancies may indicate model tuning opportunities."
            )
        elif match_score >= 0.5:
            match_quality = "MODERATE"
            match_interpretation = (
                "P3 and P4 observations show moderate topological agreement. "
                "Significant discrepancies warrant investigation of model assumptions."
            )
        else:
            match_quality = "POOR"
            match_interpretation = (
                "P3 and P4 observations show poor topological agreement. "
                "Synthetic model may not accurately reflect real system behavior."
            )

        return {
            "available": True,
            "match_score": round(match_score, 4),
            "match_quality": match_quality,
            "interpretation": match_interpretation,
            "deltas": {
                "sns": round(sns_delta, 6),
                "pcs": round(pcs_delta, 6),
                "hss": round(hss_delta, 6),
                "envelope_occupancy": round(envelope_delta, 6),
            },
            "thresholds": {
                "excellent": 0.9,
                "good": 0.75,
                "moderate": 0.5,
            },
        }

    def _verifier_guidance(self) -> Dict[str, Any]:
        """
        Provide external verifier interpretation guidance.

        See: docs/system_law/TDA_PhaseX_Binding.md Section 8
        """
        return {
            "metric_definitions": {
                "SNS": "Structural Novelty Score - measures unexpected proof structure emergence [0,1]",
                "PCS": "Proof Coherence Score - measures consistency of derivation patterns [0,1]",
                "DRS": "Drift Rate Score - measures state trajectory deviation rate [0,∞)",
                "HSS": "Homological Stability Score - measures persistence of topological features [0,1]",
            },
            "envelope_definition": {
                "description": "Omega_TDA stability envelope: SNS ≤ 0.4 AND PCS ≥ 0.6 AND HSS ≥ 0.6",
                "occupancy_meaning": "Fraction of cycles where system state remained within stability envelope",
            },
            "topology_matching_criteria": {
                "what_it_measures": (
                    "Agreement between P3 synthetic simulation and P4 real observation. "
                    "High agreement suggests the synthetic model accurately captures "
                    "system topological dynamics."
                ),
                "expected_for_valid_system": (
                    "Match score ≥ 0.75 indicates the twin model adequately represents "
                    "the real system. Lower scores may indicate model drift, parameter "
                    "miscalibration, or fundamental behavioral differences."
                ),
            },
            "red_flag_semantics": {
                "TDA_SNS_ANOMALY": "Proof structure deviated significantly from historical patterns",
                "TDA_PCS_COLLAPSE": "Derivation coherence dropped below safety threshold",
                "TDA_HSS_DEGRADATION": "Topological features degraded beyond acceptable bounds",
                "TDA_ENVELOPE_EXIT": "System exited stability envelope for extended period",
            },
            "shadow_mode_contract": (
                "All TDA metrics are OBSERVATIONAL ONLY. They do not influence "
                "governance decisions or trigger enforcement actions. Red-flags "
                "are logged for analysis but never acted upon."
            ),
        }

    def _interpret_sns(self, value: float) -> str:
        """Interpret SNS value."""
        if value <= 0.2:
            return "Normal - proof structures follow expected patterns"
        elif value <= 0.4:
            return "Elevated - some novel proof structures emerging"
        elif value <= 0.6:
            return "High - significant structural novelty detected"
        else:
            return "Anomalous - unexpected proof structures dominate"

    def _interpret_pcs(self, value: float) -> str:
        """Interpret PCS value."""
        if value >= 0.8:
            return "High coherence - derivation patterns highly consistent"
        elif value >= 0.6:
            return "Good coherence - derivation patterns mostly consistent"
        elif value >= 0.4:
            return "Low coherence - derivation patterns showing inconsistency"
        else:
            return "Incoherent - derivation patterns lack consistency"

    def _interpret_hss(self, value: float) -> str:
        """Interpret HSS value."""
        if value >= 0.8:
            return "Highly stable - topological features well preserved"
        elif value >= 0.6:
            return "Stable - topological features mostly preserved"
        elif value >= 0.4:
            return "Unstable - some topological feature degradation"
        else:
            return "Degraded - significant topological feature loss"

    def _interpret_drs(self, value: float) -> str:
        """Interpret DRS value."""
        if value <= 0.05:
            return "Minimal drift - twin tracks real system accurately"
        elif value <= 0.10:
            return "Low drift - minor prediction deviations"
        elif value <= 0.20:
            return "Moderate drift - noticeable prediction errors"
        else:
            return "High drift - significant twin-real divergence"

    def _interpret_envelope(self, occupancy: float) -> str:
        """Interpret envelope occupancy."""
        if occupancy >= 0.95:
            return "Excellent - system remained stable throughout"
        elif occupancy >= 0.85:
            return "Good - occasional stability excursions"
        elif occupancy >= 0.70:
            return "Moderate - frequent stability boundary crossings"
        else:
            return "Poor - system frequently outside stability envelope"


# =============================================================================
# Public API
# =============================================================================

def attach_tda_to_evidence(
    evidence: Dict[str, Any],
    tda_summary_p3: Optional[TDASummary] = None,
    tda_summary_p4: Optional[TDASummary] = None,
    p3_windows: Optional[List[TDAWindowMetrics]] = None,
    p4_windows: Optional[List[TDAWindowMetrics]] = None,
) -> Dict[str, Any]:
    """
    Attach TDA metrics to Evidence Pack governance section.

    This function adds a "tda" key under evidence["governance"] containing
    P3 window metrics, P4 DRS statistics, and envelope membership analysis.

    Args:
        evidence: Evidence dictionary to augment (modified in place)
        tda_summary_p3: P3 First-Light TDA summary
        tda_summary_p4: P4 real-coupling TDA summary
        p3_windows: Optional list of P3 per-window TDA metrics
        p4_windows: Optional list of P4 per-window TDA metrics

    Returns:
        The modified evidence dictionary with TDA section added.

    Example:
        >>> evidence = {"governance": {}}
        >>> attach_tda_to_evidence(
        ...     evidence,
        ...     tda_summary_p3=p3_monitor.get_summary(),
        ...     tda_summary_p4=p4_monitor.get_summary(),
        ... )
        >>> print(evidence["governance"]["tda"]["topology_matching"]["match_score"])
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Create TDA evidence block
    tda_block = TDAEvidenceBlock(
        p3_summary=tda_summary_p3,
        p4_summary=tda_summary_p4,
        p3_windows=p3_windows,
        p4_windows=p4_windows,
    )

    # Attach to evidence
    evidence["governance"]["tda"] = tda_block.to_dict()

    return evidence


def format_tda_evidence_summary(
    tda_summary_p3: Optional[TDASummary] = None,
    tda_summary_p4: Optional[TDASummary] = None,
) -> str:
    """
    Format TDA evidence as human-readable summary.

    Args:
        tda_summary_p3: P3 TDA summary
        tda_summary_p4: P4 TDA summary

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "TDA Evidence Summary (SHADOW MODE)",
        "=" * 60,
        "",
    ]

    if tda_summary_p3:
        lines.extend([
            "--- P3 First-Light (Synthetic) ---",
            f"  SNS: mean={tda_summary_p3.sns_mean:.4f}, max={tda_summary_p3.sns_max:.4f}",
            f"  PCS: mean={tda_summary_p3.pcs_mean:.4f}, min={tda_summary_p3.pcs_min:.4f}",
            f"  HSS: mean={tda_summary_p3.hss_mean:.4f}, min={tda_summary_p3.hss_min:.4f}",
            f"  Envelope: occupancy={tda_summary_p3.envelope_occupancy:.2%}",
            f"  Red-flags: {tda_summary_p3.total_red_flags}",
            "",
        ])

    if tda_summary_p4:
        lines.extend([
            "--- P4 Shadow Coupling (Real) ---",
            f"  DRS: mean={tda_summary_p4.drs_mean:.4f}, max={tda_summary_p4.drs_max:.4f}",
            f"  SNS: mean={tda_summary_p4.sns_mean:.4f}",
            f"  PCS: mean={tda_summary_p4.pcs_mean:.4f}",
            f"  HSS: mean={tda_summary_p4.hss_mean:.4f}",
            f"  Envelope: occupancy={tda_summary_p4.envelope_occupancy:.2%}",
            "",
        ])

    if tda_summary_p3 and tda_summary_p4:
        match_score = compute_topology_match_score(tda_summary_p3, tda_summary_p4)
        lines.extend([
            "--- Topology Matching ---",
            f"  Match score: {match_score:.4f}",
            f"  SNS delta: {abs(tda_summary_p3.sns_mean - tda_summary_p4.sns_mean):.4f}",
            f"  PCS delta: {abs(tda_summary_p3.pcs_mean - tda_summary_p4.pcs_mean):.4f}",
            f"  HSS delta: {abs(tda_summary_p3.hss_mean - tda_summary_p4.hss_mean):.4f}",
            "",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)


def compute_topology_match_score(
    p3_summary: TDASummary,
    p4_summary: TDASummary,
) -> float:
    """
    Compute topology matching score between P3 and P4 summaries.

    The match score indicates how well the synthetic (P3) topology
    agrees with real (P4) observations. Score ranges from 0.0 (no match)
    to 1.0 (perfect match).

    Matching Criteria (per TDA_PhaseX_Binding.md Section 8):
    - SNS delta should be < 0.2 for strong match
    - PCS delta should be < 0.15 for strong match
    - HSS delta should be < 0.15 for strong match
    - Envelope occupancy delta should be < 0.1 for strong match

    Args:
        p3_summary: P3 TDA summary
        p4_summary: P4 TDA summary

    Returns:
        Match score in [0.0, 1.0]
    """
    # Compute deltas
    sns_delta = abs(p3_summary.sns_mean - p4_summary.sns_mean)
    pcs_delta = abs(p3_summary.pcs_mean - p4_summary.pcs_mean)
    hss_delta = abs(p3_summary.hss_mean - p4_summary.hss_mean)
    envelope_delta = abs(p3_summary.envelope_occupancy - p4_summary.envelope_occupancy)

    # Thresholds for "perfect" match
    sns_threshold = 0.2
    pcs_threshold = 0.15
    hss_threshold = 0.15
    envelope_threshold = 0.1

    # Compute per-metric scores (1.0 when delta=0, 0.0 when delta >= threshold)
    sns_score = max(0.0, 1.0 - sns_delta / sns_threshold)
    pcs_score = max(0.0, 1.0 - pcs_delta / pcs_threshold)
    hss_score = max(0.0, 1.0 - hss_delta / hss_threshold)
    envelope_score = max(0.0, 1.0 - envelope_delta / envelope_threshold)

    # Weighted average (envelope given less weight as it's derived)
    match_score = (
        0.30 * sns_score +
        0.30 * pcs_score +
        0.25 * hss_score +
        0.15 * envelope_score
    )

    return max(0.0, min(1.0, match_score))
