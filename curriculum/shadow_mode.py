"""
Shadow Mode Isolation for AI Proof Ingestion

Implements the shadow mode isolation policy for AI-submitted proofs.
Shadow mode proofs are recorded but do NOT:
- Advance slice progression
- Trigger governance enforcement
- Contribute to curriculum metrics

See: docs/architecture/AI_PROOF_INGESTION_ADAPTER.md Section 2.4, 4.4

SHADOW MODE CONTRACT:
- All shadow mode proofs are observational only
- Shadow mode is mandatory for external_ai source type (Phase 1)
- Graduation from shadow mode requires UVI CONFIRMATION
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# =============================================================================
# Constants
# =============================================================================

SHADOW_MODE_SCHEMA_VERSION = "1.0.0"

# Source types that require shadow mode
SHADOW_REQUIRED_SOURCE_TYPES = frozenset({"external_ai"})


# =============================================================================
# Shadow Mode Filter
# =============================================================================


@dataclass(frozen=True)
class ShadowModeFilter:
    """
    Filter for excluding shadow mode proofs from curriculum metrics.

    This filter is applied when computing metrics that affect slice progression.
    """
    exclude_shadow: bool = True
    exclude_source_types: frozenset = SHADOW_REQUIRED_SOURCE_TYPES

    def should_include(self, proof: Dict[str, Any]) -> bool:
        """
        Determine if a proof should be included in curriculum metrics.

        Args:
            proof: Proof record with shadow_mode and source_type fields

        Returns:
            True if the proof should be included, False otherwise
        """
        # Exclude if shadow mode is set
        if self.exclude_shadow and proof.get("shadow_mode", False):
            return False

        # Exclude if source type requires shadow mode
        source_type = proof.get("source_type", "internal")
        if source_type in self.exclude_source_types:
            return False

        return True

    def filter_proofs(self, proofs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of proofs, excluding shadow mode proofs.

        Args:
            proofs: List of proof records

        Returns:
            Filtered list excluding shadow mode proofs
        """
        return [p for p in proofs if self.should_include(p)]


# Default filter instance
DEFAULT_SHADOW_FILTER = ShadowModeFilter()


# =============================================================================
# Shadow Mode Enforcement
# =============================================================================


def require_shadow_mode_for_source(source_type: str) -> bool:
    """
    Check if a source type requires shadow mode.

    Args:
        source_type: The proof source type

    Returns:
        True if shadow mode is required
    """
    return source_type in SHADOW_REQUIRED_SOURCE_TYPES


def validate_shadow_mode_constraint(
    source_type: str,
    shadow_mode: bool,
) -> Optional[str]:
    """
    Validate the shadow mode constraint for a proof.

    Args:
        source_type: The proof source type
        shadow_mode: The shadow mode flag value

    Returns:
        Error message if constraint is violated, None if valid
    """
    if source_type in SHADOW_REQUIRED_SOURCE_TYPES and not shadow_mode:
        return (
            f"Shadow mode is required for source_type='{source_type}'. "
            f"Set shadow_mode=true or wait for graduation."
        )
    return None


# =============================================================================
# Curriculum Integration
# =============================================================================


def filter_metrics_for_curriculum(
    proofs: Sequence[Dict[str, Any]],
    *,
    include_shadow: bool = False,
) -> Dict[str, Any]:
    """
    Filter proofs and compute curriculum-relevant metrics.

    Shadow mode proofs are excluded from curriculum metrics by default.
    This ensures AI-submitted proofs do not affect slice progression
    until they graduate from shadow mode.

    Args:
        proofs: List of proof records
        include_shadow: If True, include shadow mode proofs (for reporting only)

    Returns:
        Dictionary with filtered counts and metrics
    """
    filter_obj = ShadowModeFilter(exclude_shadow=not include_shadow)
    filtered = filter_obj.filter_proofs(proofs)

    # Count by status
    verified = sum(1 for p in filtered if p.get("status") == "success")
    failed = sum(1 for p in filtered if p.get("status") == "failure")
    queued = sum(1 for p in filtered if p.get("status") == "queued")

    # Count excluded
    excluded_count = len(proofs) - len(filtered)
    shadow_count = sum(1 for p in proofs if p.get("shadow_mode", False))
    ai_count = sum(1 for p in proofs if p.get("source_type") == "external_ai")

    return {
        "schema_version": SHADOW_MODE_SCHEMA_VERSION,
        "total_proofs": len(filtered),
        "verified_count": verified,
        "failed_count": failed,
        "queued_count": queued,
        "excluded_count": excluded_count,
        "shadow_mode_count": shadow_count,
        "external_ai_count": ai_count,
        "mode": "FILTERED" if not include_shadow else "UNFILTERED",
    }


def build_shadow_mode_report(
    proofs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a shadow mode status report.

    This report is informational only and does not affect governance.

    Args:
        proofs: List of proof records

    Returns:
        Shadow mode status report
    """
    shadow_proofs = [p for p in proofs if p.get("shadow_mode", False)]
    non_shadow_proofs = [p for p in proofs if not p.get("shadow_mode", False)]

    # Shadow mode breakdown by source type
    ai_shadow = [p for p in shadow_proofs if p.get("source_type") == "external_ai"]
    internal_shadow = [p for p in shadow_proofs if p.get("source_type") == "internal"]

    return {
        "schema_version": SHADOW_MODE_SCHEMA_VERSION,
        "report_type": "shadow_mode_status",
        "mode": "SHADOW",  # This report is itself observational

        "summary": {
            "total_proofs": len(proofs),
            "shadow_mode_count": len(shadow_proofs),
            "non_shadow_count": len(non_shadow_proofs),
        },

        "shadow_breakdown": {
            "external_ai": len(ai_shadow),
            "internal": len(internal_shadow),
        },

        "observations": build_shadow_observations(proofs, shadow_proofs),
    }


def build_shadow_observations(
    all_proofs: Sequence[Dict[str, Any]],
    shadow_proofs: Sequence[Dict[str, Any]],
) -> List[str]:
    """
    Build observation notes for shadow mode report.

    These are informational only.
    """
    observations = []

    if not shadow_proofs:
        observations.append("No proofs currently in shadow mode")
        return observations

    # Check for AI proofs
    ai_proofs = [p for p in shadow_proofs if p.get("source_type") == "external_ai"]
    if ai_proofs:
        verified = sum(1 for p in ai_proofs if p.get("status") == "success")
        total = len(ai_proofs)
        rate = verified / total if total > 0 else 0
        observations.append(
            f"AI proofs in shadow mode: {total} ({verified} verified, {rate:.1%} rate)"
        )

    # Check for unexpected shadow mode proofs
    internal_shadow = [p for p in shadow_proofs if p.get("source_type") == "internal"]
    if internal_shadow:
        observations.append(
            f"WARNING: {len(internal_shadow)} internal proofs in shadow mode (unexpected)"
        )

    # Graduation readiness check (informational)
    if ai_proofs:
        if len(ai_proofs) >= 1000:
            observations.append("AI proof volume threshold (1000) reached for graduation consideration")
        else:
            observations.append(f"AI proof volume: {len(ai_proofs)}/1000 for graduation")

    return observations


# =============================================================================
# Slice Progression Guard
# =============================================================================


def guard_slice_progression(
    proof: Dict[str, Any],
) -> Optional[str]:
    """
    Guard against slice progression for shadow mode proofs.

    This function should be called before crediting a proof
    toward slice progression.

    Args:
        proof: The proof record

    Returns:
        Rejection reason if the proof should not advance the slice,
        None if the proof can contribute to progression
    """
    if proof.get("shadow_mode", False):
        return "Shadow mode proofs do not contribute to slice progression"

    source_type = proof.get("source_type", "internal")
    if source_type in SHADOW_REQUIRED_SOURCE_TYPES:
        return f"Source type '{source_type}' requires shadow mode; cannot contribute to progression"

    return None
