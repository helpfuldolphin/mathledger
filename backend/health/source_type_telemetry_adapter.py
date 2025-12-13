"""Source type telemetry adapter for AI proof ingestion.

STATUS: PHASE 1 — AI PROOF INGESTION

Provides telemetry tagging and metrics for proofs by source type (internal vs external_ai).
Enables separate tracking of H/rho/tau metrics for AI vs internal proofs.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Metrics are observational only
- No governance enforcement based on source type metrics
- No control flow depends on this adapter

See: docs/architecture/AI_PROOF_INGESTION_ADAPTER.md Section 4.5
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone


SOURCE_TYPE_TELEMETRY_SCHEMA_VERSION = "1.0.0"

# Valid source types
SOURCE_TYPE_INTERNAL = "internal"
SOURCE_TYPE_EXTERNAL_AI = "external_ai"
VALID_SOURCE_TYPES = {SOURCE_TYPE_INTERNAL, SOURCE_TYPE_EXTERNAL_AI}


@dataclass(frozen=True)
class SourceTypeMetrics:
    """Metrics for a specific source type."""
    source_type: str
    total_proofs: int
    verified_count: int
    failed_count: int
    queued_count: int
    verification_rate: float  # verified / (verified + failed)
    shadow_mode_count: int


def compute_verification_rate(verified: int, failed: int) -> float:
    """Compute verification rate as verified / total attempts."""
    total = verified + failed
    if total == 0:
        return 1.0  # No attempts = 100% (no failures)
    return verified / total


def build_source_type_metrics(
    proofs_by_source: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, SourceTypeMetrics]:
    """
    Build metrics breakdown by source type.

    Args:
        proofs_by_source: Dictionary mapping source_type to list of proof records.
            Each proof record should have: status, shadow_mode

    Returns:
        Dictionary mapping source_type to SourceTypeMetrics
    """
    result = {}

    for source_type, proofs in proofs_by_source.items():
        if source_type not in VALID_SOURCE_TYPES:
            continue

        total = len(proofs)
        verified = sum(1 for p in proofs if p.get("status") == "success")
        failed = sum(1 for p in proofs if p.get("status") == "failure")
        queued = sum(1 for p in proofs if p.get("status") == "queued")
        shadow = sum(1 for p in proofs if p.get("shadow_mode", False))

        result[source_type] = SourceTypeMetrics(
            source_type=source_type,
            total_proofs=total,
            verified_count=verified,
            failed_count=failed,
            queued_count=queued,
            verification_rate=compute_verification_rate(verified, failed),
            shadow_mode_count=shadow,
        )

    return result


def build_source_type_telemetry_tile(
    metrics: Dict[str, SourceTypeMetrics],
    *,
    ai_submission_count_24h: int = 0,
    ai_verification_rate_24h: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build telemetry tile for source type metrics.

    SHADOW MODE CONTRACT:
    - This tile is purely observational
    - No governance decisions are based on this tile
    - No enforcement of any kind

    Args:
        metrics: Source type metrics from build_source_type_metrics()
        ai_submission_count_24h: Number of AI submissions in last 24 hours
        ai_verification_rate_24h: AI verification rate in last 24 hours

    Returns:
        Telemetry tile dictionary
    """
    internal_metrics = metrics.get(SOURCE_TYPE_INTERNAL)
    ai_metrics = metrics.get(SOURCE_TYPE_EXTERNAL_AI)

    return {
        "schema_version": SOURCE_TYPE_TELEMETRY_SCHEMA_VERSION,
        "tile_type": "source_type_telemetry",
        "mode": "SHADOW",  # Always shadow for AI proofs in Phase 1

        # Internal proof metrics
        "internal": {
            "total_proofs": internal_metrics.total_proofs if internal_metrics else 0,
            "verified_count": internal_metrics.verified_count if internal_metrics else 0,
            "failed_count": internal_metrics.failed_count if internal_metrics else 0,
            "verification_rate": internal_metrics.verification_rate if internal_metrics else 1.0,
        } if internal_metrics else None,

        # External AI proof metrics
        "external_ai": {
            "total_proofs": ai_metrics.total_proofs if ai_metrics else 0,
            "verified_count": ai_metrics.verified_count if ai_metrics else 0,
            "failed_count": ai_metrics.failed_count if ai_metrics else 0,
            "queued_count": ai_metrics.queued_count if ai_metrics else 0,
            "verification_rate": ai_metrics.verification_rate if ai_metrics else 1.0,
            "shadow_mode_count": ai_metrics.shadow_mode_count if ai_metrics else 0,
            "submissions_24h": ai_submission_count_24h,
            "verification_rate_24h": ai_verification_rate_24h,
        } if ai_metrics else None,

        # Divergence between internal and AI verification rates
        "divergence": compute_rate_divergence(metrics),

        # Observation notes (shadow mode)
        "observations": build_observations(metrics, ai_submission_count_24h),
    }


def compute_rate_divergence(
    metrics: Dict[str, SourceTypeMetrics],
) -> Optional[Dict[str, Any]]:
    """
    Compute divergence between internal and AI verification rates.

    This is observational only - used to detect if AI proofs have
    systematically different verification characteristics.

    Returns None if insufficient data for comparison.
    """
    internal = metrics.get(SOURCE_TYPE_INTERNAL)
    ai = metrics.get(SOURCE_TYPE_EXTERNAL_AI)

    if not internal or not ai:
        return None

    # Need at least 10 attempts each for meaningful comparison
    internal_attempts = internal.verified_count + internal.failed_count
    ai_attempts = ai.verified_count + ai.failed_count

    if internal_attempts < 10 or ai_attempts < 10:
        return {
            "sufficient_data": False,
            "internal_attempts": internal_attempts,
            "ai_attempts": ai_attempts,
            "divergence": None,
        }

    divergence = abs(internal.verification_rate - ai.verification_rate)

    return {
        "sufficient_data": True,
        "internal_rate": internal.verification_rate,
        "ai_rate": ai.verification_rate,
        "divergence": divergence,
        "within_threshold": divergence <= 0.1,  # 10% threshold
    }


def build_observations(
    metrics: Dict[str, SourceTypeMetrics],
    ai_submission_count_24h: int,
) -> List[str]:
    """
    Build observation notes for shadow mode telemetry.

    These are informational only and do not trigger any actions.
    """
    observations = []

    ai = metrics.get(SOURCE_TYPE_EXTERNAL_AI)
    if ai:
        if ai.total_proofs == 0:
            observations.append("No AI proofs submitted yet")
        elif ai.shadow_mode_count == ai.total_proofs:
            observations.append(f"All {ai.total_proofs} AI proofs in shadow mode (expected)")
        else:
            # This would be unexpected in Phase 1
            observations.append(
                f"WARNING: {ai.total_proofs - ai.shadow_mode_count} AI proofs not in shadow mode"
            )

        if ai.queued_count > 0:
            observations.append(f"{ai.queued_count} AI proofs awaiting verification")

        if ai.verification_rate < 0.95 and (ai.verified_count + ai.failed_count) >= 10:
            observations.append(
                f"AI verification rate ({ai.verification_rate:.1%}) below 95% threshold"
            )

    if ai_submission_count_24h > 100:
        observations.append(f"High AI submission volume: {ai_submission_count_24h} in 24h")

    return observations


def tag_telemetry_event(
    event: Dict[str, Any],
    source_type: str,
    shadow_mode: bool = True,
) -> Dict[str, Any]:
    """
    Tag a telemetry event with source type information.

    This function adds source_type tagging to existing telemetry events
    to enable filtering and analysis by proof origin.

    Args:
        event: The telemetry event to tag
        source_type: The source type (internal or external_ai)
        shadow_mode: Whether this is a shadow mode event

    Returns:
        Tagged event dictionary (new dict, does not mutate input)
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type: {source_type}")

    return {
        **event,
        "source_type": source_type,
        "shadow_mode": shadow_mode,
        "_tagged_at": datetime.now(timezone.utc).isoformat(),
        "_tagger_version": SOURCE_TYPE_TELEMETRY_SCHEMA_VERSION,
    }


def filter_events_by_source(
    events: List[Dict[str, Any]],
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Filter telemetry events by source type.

    Args:
        events: List of telemetry events
        source_type: Source type to filter for

    Returns:
        Filtered list of events
    """
    return [e for e in events if e.get("source_type") == source_type]
