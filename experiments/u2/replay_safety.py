"""U2 Replay Safety module.

Provides replay safety verification and governance signals.

System Law Reference:
    docs/system_law/Replay_Safety_Governance_Law.md

Schema Reference:
    docs/system_law/schemas/replay_safety/replay_safety_governance_signal.schema.json

Integration Points:
    - Global Alignment View (TODO: CLAUDE I)
    - Phase X Director Panel (TODO: CLAUDE I)
    - USLA Governance Bridge (TODO: Phase X P5+)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PromotionStatus(str, Enum):
    """Promotion status for governance signals."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ABSTAIN = "abstain"
    OK = "ok"
    WARN = "warn"
    BLOCK = "block"


class SafetyLevel(str, Enum):
    """Safety level for replay verification."""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    OK = "ok"
    WARN = "warn"
    BLOCK = "block"
    FAIL = "fail"  # Alias for test compatibility


class GovernanceAlignment(str, Enum):
    """Governance alignment status."""
    ALIGNED = "aligned"
    TENSION = "tension"
    DIVERGENT = "divergent"


@dataclass
class ReplaySafetyResult:
    """Replay safety verification result."""
    safe: bool = True
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceSignal:
    """Governance signal for replay safety."""
    signal_type: str
    severity: str = "info"
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


def verify_replay_safety(
    trace_data: Dict[str, Any],
    expected_hash: Optional[str] = None,
) -> ReplaySafetyResult:
    """Verify replay safety for trace data."""
    violations = []
    warnings = []

    if expected_hash and trace_data.get("hash") != expected_hash:
        violations.append("Hash mismatch: replay not deterministic")

    return ReplaySafetyResult(
        safe=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        metadata={"checked_hash": expected_hash is not None},
    )


def emit_governance_signal(
    signal_type: str,
    severity: str = "info",
    message: str = "",
    **context: Any,
) -> GovernanceSignal:
    """Emit a governance signal."""
    return GovernanceSignal(
        signal_type=signal_type,
        severity=severity,
        message=message,
        context=dict(context),
    )


def check_replay_determinism(
    trace1: Dict[str, Any],
    trace2: Dict[str, Any],
) -> bool:
    """Check if two traces are deterministically equivalent."""
    return trace1.get("hash") == trace2.get("hash")


def evaluate_replay_safety_for_promotion(
    envelope_or_trace: Dict[str, Any],
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate replay safety for promotion decision.

    Supports both legacy trace_data format and new envelope format.

    Args:
        envelope_or_trace: Either a trace_data dict with "hash" key,
            or an envelope dict with "safety_level", "is_fully_deterministic", etc.
        expected_hash: Optional expected hash for trace verification.

    Returns:
        Dict with status, reasons, safe_for_policy_update, safe_for_promotion.
    """
    # Detect envelope format vs legacy trace format
    if "safety_level" in envelope_or_trace:
        # Envelope format
        envelope = envelope_or_trace
        safety_level = envelope.get("safety_level")

        # Normalize safety_level to string for comparison
        if isinstance(safety_level, Enum):
            safety_level_str = safety_level.value
        else:
            safety_level_str = str(safety_level).lower()

        is_deterministic = envelope.get("is_fully_deterministic", True)
        policy_allowed = envelope.get("policy_update_allowed", True)

        # Determine status from envelope
        if safety_level_str in ("fail", "block", "unsafe"):
            status = PromotionStatus.BLOCK
            reasons = ["Safety level indicates failure"]
        elif safety_level_str in ("warn", "caution"):
            status = PromotionStatus.WARN
            reasons = ["Safety level indicates caution"]
        elif not is_deterministic:
            status = PromotionStatus.BLOCK
            reasons = ["Replay is not fully deterministic"]
        elif not policy_allowed:
            status = PromotionStatus.WARN
            reasons = ["Policy update not allowed"]
        else:
            status = PromotionStatus.OK
            reasons = ["All safety checks passed"]

        return {
            "status": status,
            "reasons": reasons,
            "safe_for_policy_update": policy_allowed,
            "safe_for_promotion": status != PromotionStatus.BLOCK,
        }
    else:
        # Legacy trace_data format
        result = verify_replay_safety(envelope_or_trace, expected_hash)
        return {
            "safe": result.safe,
            "violations": result.violations,
            "warnings": result.warnings,
            "status": PromotionStatus.OK if result.safe else PromotionStatus.BLOCK,
            "promotion_status": PromotionStatus.APPROVED if result.safe else PromotionStatus.REJECTED,
            "safety_level": SafetyLevel.SAFE if result.safe else SafetyLevel.UNSAFE,
            "reasons": result.violations if result.violations else ["All checks passed"],
            "safe_for_policy_update": result.safe,
            "safe_for_promotion": result.safe,
        }


# =============================================================================
# PHASE IV: Evidence & Director Panel
# =============================================================================

def summarize_replay_safety_for_evidence(
    envelope: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
    governance_view: Optional[Dict[str, Any]] = None,
    # Legacy parameters
    trace_data: Optional[Dict[str, Any]] = None,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize replay safety for evidence pack.

    Supports both new envelope-based format and legacy trace_data format.

    Args:
        envelope: Safety envelope with safety_level, is_fully_deterministic, etc.
        confidence: Confidence score (0.0 to 1.0).
        governance_signal: Optional governance signal to extract governance_status.
        governance_view: Optional governance view to extract governance_alignment.
        trace_data: (Legacy) Trace data dict with "hash" key.
        expected_hash: (Legacy) Expected hash for verification.

    Returns:
        Evidence summary dict with replay_safety_ok, status, confidence_score,
        and optional governance_status/governance_alignment fields.
    """
    # Handle legacy format
    if trace_data is not None and envelope is None:
        result = verify_replay_safety(trace_data, expected_hash)
        evaluation = evaluate_replay_safety_for_promotion(trace_data, expected_hash)
        return {
            "safe": result.safe,
            "violations": result.violations,
            "warnings": result.warnings,
            "promotion_status": evaluation.get("promotion_status", PromotionStatus.OK).value
                if isinstance(evaluation.get("promotion_status"), Enum)
                else evaluation.get("promotion_status", "ok"),
            "safety_level": evaluation.get("safety_level", SafetyLevel.OK).value
                if isinstance(evaluation.get("safety_level"), Enum)
                else evaluation.get("safety_level", "ok"),
            "evidence_hash": trace_data.get("hash"),
        }

    # New envelope-based format
    envelope = envelope or {}
    safety_level = envelope.get("safety_level", SafetyLevel.OK)

    # Normalize safety_level
    if isinstance(safety_level, Enum):
        safety_level_str = safety_level.value
    else:
        safety_level_str = str(safety_level).lower()

    is_deterministic = envelope.get("is_fully_deterministic", True)
    policy_allowed = envelope.get("policy_update_allowed", True)

    # Determine replay_safety_ok
    replay_ok = safety_level_str in ("ok", "safe") and is_deterministic

    # Determine status
    if safety_level_str in ("fail", "block", "unsafe") or not is_deterministic:
        status = PromotionStatus.BLOCK
    elif safety_level_str in ("warn", "caution"):
        status = PromotionStatus.WARN
    else:
        status = PromotionStatus.OK

    evidence: Dict[str, Any] = {
        "replay_safety_ok": replay_ok,
        "status": status,
        "confidence_score": confidence if confidence is not None else envelope.get("confidence_score", 1.0),
        "is_fully_deterministic": is_deterministic,
        "policy_update_allowed": policy_allowed,
    }

    # Add governance_status from governance_signal if provided
    if governance_signal is not None:
        gov_status = governance_signal.get("status")
        if gov_status is not None:
            evidence["governance_status"] = gov_status

    # Add governance_alignment from governance_view if provided
    if governance_view is not None:
        gov_alignment = governance_view.get("governance_alignment")
        if gov_alignment is not None:
            evidence["governance_alignment"] = gov_alignment

    return evidence


def build_replay_safety_director_panel(
    envelope: Optional[Dict[str, Any]] = None,
    promotion_eval: Optional[Dict[str, Any]] = None,
    governance_view: Optional[Dict[str, Any]] = None,
    # Legacy parameters
    trace_data: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build replay safety director panel.

    Supports both new envelope-based format and legacy trace_data format.

    Args:
        envelope: Safety envelope with safety_level, confidence_score, etc.
        promotion_eval: Promotion evaluation result with status, reasons.
        governance_view: Governance view with conflict, safety_status, governance_status.
        trace_data: (Legacy) Trace data dict.
        config: (Legacy) Configuration dict.

    Returns:
        Director panel dict with status, conflict_flag, conflict_note, headline.
    """
    # Handle legacy format
    if trace_data is not None and envelope is None:
        config = config or {}
        result = verify_replay_safety(trace_data)
        return {
            "status": "ok" if result.safe else "warn",
            "safe": result.safe,
            "violations": len(result.violations),
            "warnings": len(result.warnings),
        }

    # New envelope-based format
    envelope = envelope or {}
    promotion_eval = promotion_eval or {}
    governance_view = governance_view or {}

    # Extract conflict information
    conflict = governance_view.get("conflict", False)
    safety_status = governance_view.get("safety_status", PromotionStatus.OK)
    gov_status = governance_view.get("governance_status", PromotionStatus.OK)

    # Normalize statuses
    if isinstance(safety_status, Enum):
        safety_status = safety_status.value
    if isinstance(gov_status, Enum):
        gov_status = gov_status.value

    # Determine overall status
    promo_status = promotion_eval.get("status", PromotionStatus.OK)
    if isinstance(promo_status, Enum):
        promo_status_str = promo_status.value
    else:
        promo_status_str = str(promo_status).lower()

    if conflict or promo_status_str == "block":
        panel_status = "block"
    elif promo_status_str == "warn":
        panel_status = "warn"
    else:
        panel_status = "ok"

    # Build conflict note
    conflict_note: Optional[str] = None
    if conflict:
        conflict_note = f"CONFLICT: Safety says {safety_status}, Governance says {gov_status}. Manual review required."

    # Build headline
    confidence = envelope.get("confidence_score", 1.0)
    if conflict:
        headline = f"Replay Safety: CONFLICT (confidence: {confidence:.0%})"
    elif panel_status == "block":
        headline = f"Replay Safety: BLOCKED (confidence: {confidence:.0%})"
    elif panel_status == "warn":
        headline = f"Replay Safety: WARNING (confidence: {confidence:.0%})"
    else:
        headline = f"Replay Safety: OK (confidence: {confidence:.0%})"

    return {
        "status": panel_status,
        "conflict_flag": conflict,
        "conflict_note": conflict_note,
        "headline": headline,
        "safety_status": safety_status,
        "governance_status": gov_status,
        "confidence_score": confidence,
    }


def build_replay_safety_envelope(
    traces: List[Dict[str, Any]],
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Build replay safety envelope from multiple traces."""
    results = [verify_replay_safety(t, expected_hash) for t in traces]
    all_safe = all(r.safe for r in results)

    return {
        "all_safe": all_safe,
        "trace_count": len(traces),
        "safe_count": sum(1 for r in results if r.safe),
        "total_violations": sum(len(r.violations) for r in results),
    }


def compute_replay_confidence(
    traces: List[Dict[str, Any]],
    expected_hash: Optional[str] = None,
) -> float:
    """Compute replay confidence score."""
    if not traces:
        return 0.0

    results = [verify_replay_safety(t, expected_hash) for t in traces]
    safe_count = sum(1 for r in results if r.safe)
    return safe_count / len(traces)


# =============================================================================
# PHASE V: Governance View
# =============================================================================

# TODO: CLAUDE I — Connect to Global Alignment View
# Integration: This governance view should be consumed by backend/analytics/global_alignment_view.py
# to provide unified cross-subsystem alignment visualization.
# See: docs/system_law/Replay_Safety_Governance_Law.md Section 6.1

def build_replay_safety_governance_view(
    envelope: Optional[Dict[str, Any]] = None,
    promotion_eval: Optional[Dict[str, Any]] = None,
    radar: Optional[Dict[str, Any]] = None,
    # Legacy parameters
    trace_data: Optional[Dict[str, Any]] = None,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Build replay safety governance view.

    Computes alignment between Safety evaluation and Radar evaluation.

    Mapping Rules (from Replay_Safety_Governance_Law.md):
        ALIGNED:   Safety and Radar agree on status
        TENSION:   Minor disagreement (OK vs WARN)
        DIVERGENT: Major disagreement (OK vs BLOCK, or explicit conflict)

    Args:
        envelope: Safety envelope with safety_level, is_fully_deterministic, etc.
        promotion_eval: Promotion evaluation result with status.
        radar: Radar evaluation dict with status, reasons.
        trace_data: (Legacy) Trace data dict.
        expected_hash: (Legacy) Expected hash for verification.

    Returns:
        Governance view with governance_alignment, conflict, safety_status,
        governance_status, and reasons.
    """
    # Handle legacy format
    if trace_data is not None and envelope is None:
        result = verify_replay_safety(trace_data, expected_hash)

        if result.safe:
            alignment = GovernanceAlignment.ALIGNED
        elif result.warnings and not result.violations:
            alignment = GovernanceAlignment.TENSION
        else:
            alignment = GovernanceAlignment.DIVERGENT

        return {
            "alignment": alignment.value,
            "governance_alignment": alignment,
            "safe": result.safe,
            "violations": result.violations,
            "warnings": result.warnings,
        }

    # New envelope-based format
    envelope = envelope or {}
    promotion_eval = promotion_eval or {}
    radar = radar or {}

    # Extract safety status from promotion_eval
    safety_status = promotion_eval.get("status", PromotionStatus.OK)
    if isinstance(safety_status, Enum):
        safety_status_str = safety_status.value
    else:
        safety_status_str = str(safety_status).upper()

    # Extract radar status
    radar_status = radar.get("status", "OK")
    if isinstance(radar_status, Enum):
        radar_status_str = radar_status.value
    else:
        radar_status_str = str(radar_status).upper()

    # Normalize to comparable form
    safety_norm = safety_status_str.upper()
    radar_norm = radar_status_str.upper()

    # Determine alignment based on status comparison
    # ALIGNED: Both same status
    # TENSION: One OK, one WARN (or both WARN)
    # DIVERGENT: One OK/WARN, one BLOCK (major disagreement)

    if safety_norm == radar_norm:
        alignment = GovernanceAlignment.ALIGNED
        conflict = False
    elif {safety_norm, radar_norm} == {"OK", "WARN"}:
        alignment = GovernanceAlignment.TENSION
        conflict = False
    elif "BLOCK" in {safety_norm, radar_norm}:
        alignment = GovernanceAlignment.DIVERGENT
        conflict = True
    else:
        # Default to tension for other mismatches
        alignment = GovernanceAlignment.TENSION
        conflict = False

    # Collect reasons
    reasons: List[str] = []
    if promotion_eval.get("reasons"):
        reasons.extend(promotion_eval["reasons"])
    if radar.get("reasons"):
        reasons.extend(radar["reasons"])

    return {
        "governance_alignment": alignment,
        "alignment": alignment.value,  # Legacy field
        "conflict": conflict,
        "safety_status": safety_status if isinstance(safety_status, Enum) else PromotionStatus(safety_status_str.lower()),
        "governance_status": PromotionStatus(radar_status_str.lower()) if radar_status_str.lower() in ("ok", "warn", "block") else radar_status,
        "reasons": reasons,
    }


# =============================================================================
# PHASE VI: Governance Signal Adapter
# =============================================================================

# TODO: CLAUDE I — Connect to Phase X Director Panel
# Integration: This governance signal should feed into backend/topology/director_panel.py
# to provide real-time conflict visualization and manual review queue.
# See: docs/system_law/Replay_Safety_Governance_Law.md Section 6.2

def to_governance_signal_for_replay_safety(
    safety_eval: Dict[str, Any],
    radar_eval: Optional[Dict[str, Any]] = None,
    *,
    radar_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert safety and radar evaluations to a governance signal.

    Collapses Safety and Radar into a single, normalized governance signal.

    Schema: docs/system_law/schemas/replay_safety/replay_safety_governance_signal.schema.json

    Mapping Rules (from Replay_Safety_Governance_Law.md):
        BLOCK: safety_status==BLOCK OR radar_status==BLOCK OR alignment==DIVERGENT OR conflict==True
        WARN:  (not BLOCK) AND (safety_status==WARN OR radar_status==WARN OR alignment==TENSION)
        OK:    All conditions nominal

    Args:
        safety_eval: Safety evaluation dict with status, reasons, safe_for_policy_update, safe_for_promotion.
        radar_eval: Radar evaluation dict with status/governance_status, governance_alignment, conflict, reasons.
        radar_view: Alias for radar_eval (keyword-only, for test compatibility).

    Returns:
        Unified governance signal with all required schema fields.
    """
    # Support radar_view as alias for radar_eval
    if radar_view is not None and radar_eval is None:
        radar_eval = radar_view
    elif radar_eval is None:
        radar_eval = {}

    # Extract safety status
    safety_status = safety_eval.get("status", PromotionStatus.OK)
    if isinstance(safety_status, str):
        safety_status = PromotionStatus(safety_status.lower()) if safety_status.lower() in ("ok", "warn", "block") else PromotionStatus.OK

    # Extract radar status (may be in "status" or "governance_status")
    radar_status = radar_eval.get("status") or radar_eval.get("governance_status", PromotionStatus.OK)
    if isinstance(radar_status, str):
        radar_status = PromotionStatus(radar_status.lower()) if radar_status.lower() in ("ok", "warn", "block") else PromotionStatus.OK

    # Extract alignment from radar_eval
    alignment = radar_eval.get("governance_alignment") or radar_eval.get("alignment")
    if isinstance(alignment, str):
        alignment_str = alignment.lower()
    elif isinstance(alignment, GovernanceAlignment):
        alignment_str = alignment.value
    else:
        alignment_str = "aligned"

    # Extract conflict flag
    conflict = radar_eval.get("conflict", False)

    # Determine combined status using documented mapping rules
    # BLOCK conditions (checked first, most restrictive)
    if safety_status == PromotionStatus.BLOCK:
        status = PromotionStatus.BLOCK
    elif radar_status == PromotionStatus.BLOCK:
        status = PromotionStatus.BLOCK
    elif alignment_str == "divergent":
        status = PromotionStatus.BLOCK
    elif conflict:
        status = PromotionStatus.BLOCK
    # WARN conditions
    elif safety_status == PromotionStatus.WARN:
        status = PromotionStatus.WARN
    elif radar_status == PromotionStatus.WARN:
        status = PromotionStatus.WARN
    elif alignment_str == "tension":
        status = PromotionStatus.WARN
    # OK (all nominal)
    else:
        status = PromotionStatus.OK

    # Collect reasons with proper prefixing
    reasons: List[str] = []
    if safety_eval.get("reasons"):
        for r in safety_eval["reasons"]:
            # Don't double-prefix
            if r.startswith("[Safety]") or r.startswith("[Radar]") or r.startswith("[CONFLICT]"):
                reasons.append(r)
            else:
                reasons.append(f"[Safety] {r}")
    if radar_eval.get("reasons"):
        for r in radar_eval["reasons"]:
            # Don't double-prefix
            if r.startswith("[Safety]") or r.startswith("[Radar]") or r.startswith("[CONFLICT]"):
                reasons.append(r)
            else:
                reasons.append(f"[Radar] {r}")

    # Add CONFLICT reason if divergent or conflict flag
    if alignment_str == "divergent":
        reasons.append("[CONFLICT] Safety and Radar alignment diverges - manual review required")
    elif conflict and not any("[CONFLICT]" in r for r in reasons):
        reasons.append("[CONFLICT] Explicit conflict flag set - manual review required")

    # Determine governance_alignment enum
    if alignment_str == "aligned":
        gov_alignment = GovernanceAlignment.ALIGNED
    elif alignment_str == "tension":
        gov_alignment = GovernanceAlignment.TENSION
    else:
        gov_alignment = GovernanceAlignment.DIVERGENT

    return {
        "schema_version": "1.0.0",
        "signal_type": "replay_safety",
        "status": status,
        "governance_status": status,
        "governance_alignment": gov_alignment,
        "safety_status": safety_status,
        "radar_status": radar_status,
        "conflict": conflict or alignment_str == "divergent",
        "reasons": reasons,
        "safe_for_policy_update": safety_eval.get("safe_for_policy_update", True) and status != PromotionStatus.BLOCK,
        "safe_for_promotion": safety_eval.get("safe_for_promotion", True) and status != PromotionStatus.BLOCK,
    }


# =============================================================================
# PHASE VII: Director Tile & Alignment View Adapters
# =============================================================================


def build_replay_safety_director_tile(
    governance_view: Dict[str, Any],
) -> Dict[str, Any]:
    """Build replay safety director tile for Phase X director panel.

    Creates a schema-compliant tile for the director panel visualization.

    Schema: docs/system_law/schemas/replay_safety/replay_safety_governance_signal.schema.json

    Args:
        governance_view: Governance view from build_replay_safety_governance_view().

    Returns:
        Director tile dict with status, alignment, conflict_flag, headline.
    """
    # Extract alignment
    alignment = governance_view.get("governance_alignment") or governance_view.get("alignment")
    if isinstance(alignment, GovernanceAlignment):
        alignment_enum = alignment
        alignment_str = alignment.value
    elif isinstance(alignment, str):
        alignment_str = alignment.lower()
        alignment_enum = GovernanceAlignment(alignment_str) if alignment_str in ("aligned", "tension", "divergent") else GovernanceAlignment.ALIGNED
    else:
        alignment_enum = GovernanceAlignment.ALIGNED
        alignment_str = "aligned"

    # Extract conflict
    conflict = governance_view.get("conflict", False)

    # Extract statuses
    safety_status = governance_view.get("safety_status", PromotionStatus.OK)
    gov_status = governance_view.get("governance_status", PromotionStatus.OK)

    # Normalize statuses to strings
    if isinstance(safety_status, Enum):
        safety_str = safety_status.value
    else:
        safety_str = str(safety_status).lower()

    if isinstance(gov_status, Enum):
        gov_str = gov_status.value
    else:
        gov_str = str(gov_status).lower()

    # Determine tile status
    if conflict or alignment_str == "divergent":
        tile_status = "block"
    elif alignment_str == "tension" or safety_str == "warn" or gov_str == "warn":
        tile_status = "warn"
    elif safety_str == "block" or gov_str == "block":
        tile_status = "block"
    else:
        tile_status = "ok"

    # Build headline
    if conflict:
        headline = f"Replay Safety: CONFLICT ({safety_str.upper()} vs {gov_str.upper()})"
    elif tile_status == "block":
        headline = "Replay Safety: BLOCKED"
    elif tile_status == "warn":
        headline = "Replay Safety: WARNING"
    else:
        headline = "Replay Safety: OK"

    return {
        "schema_version": "1.0.0",
        "tile_type": "replay_safety",
        "status": tile_status,
        "alignment": alignment_str,
        "conflict_flag": conflict,
        "headline": headline,
        "safety_status": safety_str,
        "governance_status": gov_str,
        "reasons": governance_view.get("reasons", []),
    }


def replay_safety_for_alignment_view(
    governance_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare replay safety signal for global alignment view consumption.

    This is the handoff function for CLAUDE I's global governance synthesis.
    Exposes status, alignment state, and root causes in a normalized format.

    Args:
        governance_signal: Signal from to_governance_signal_for_replay_safety().

    Returns:
        Dict ready for global alignment view integration with:
        - signal_type: "replay_safety"
        - status: OK | WARN | BLOCK
        - alignment: ALIGNED | TENSION | DIVERGENT
        - conflict: boolean
        - root_causes: list of reason strings
        - safe_for_fusion: boolean (whether this signal allows overall OK)
    """
    # Extract status
    status = governance_signal.get("status") or governance_signal.get("governance_status", PromotionStatus.OK)
    if isinstance(status, Enum):
        status_str = status.value
    else:
        status_str = str(status).lower()

    # Extract alignment
    alignment = governance_signal.get("governance_alignment") or governance_signal.get("alignment")
    if isinstance(alignment, GovernanceAlignment):
        alignment_str = alignment.value
    elif isinstance(alignment, str):
        alignment_str = alignment.lower()
    else:
        alignment_str = "aligned"

    # Extract conflict
    conflict = governance_signal.get("conflict", False)

    # Extract reasons as root causes
    reasons = governance_signal.get("reasons", [])

    # Determine if safe for fusion (allows overall OK in global view)
    # Only OK and non-conflicting allows global OK
    safe_for_fusion = status_str == "ok" and not conflict

    return {
        "signal_type": "replay_safety",
        "status": status_str,
        "alignment": alignment_str,
        "conflict": conflict,
        "root_causes": reasons,
        "safe_for_fusion": safe_for_fusion,
        "safe_for_policy_update": governance_signal.get("safe_for_policy_update", True),
        "safe_for_promotion": governance_signal.get("safe_for_promotion", True),
    }


def _build_compact_reason_summary(reasons: List[str], max_reasons: int = 3) -> str:
    """Build a compact summary of reasons for evidence display.

    Args:
        reasons: List of prefixed reason strings.
        max_reasons: Maximum number of reasons to include.

    Returns:
        Compact summary string.
    """
    if not reasons:
        return "All checks passed"

    # Filter to most important reasons (CONFLICT > Safety > Radar)
    conflict_reasons = [r for r in reasons if "[CONFLICT]" in r]
    safety_reasons = [r for r in reasons if "[Safety]" in r and "[CONFLICT]" not in r]
    radar_reasons = [r for r in reasons if "[Radar]" in r and "[CONFLICT]" not in r]

    # Prioritize: conflicts first, then safety, then radar
    prioritized = conflict_reasons + safety_reasons + radar_reasons
    selected = prioritized[:max_reasons]

    if len(prioritized) > max_reasons:
        remaining = len(prioritized) - max_reasons
        return "; ".join(selected) + f" (+{remaining} more)"

    return "; ".join(selected) if selected else "All checks passed"


def attach_replay_safety_to_evidence(
    evidence: Dict[str, Any],
    governance_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach replay safety governance signal to evidence pack.

    CONSTRAINT: Advisory only. No gating. This function adds observational
    data to the evidence pack without influencing control flow.

    Features:
    - Includes governance_alignment and conflict flag
    - Provides compact reason_summary for quick review
    - Deterministic field ordering via sorted keys
    - JSON-safe (no Enum objects in output)

    Args:
        evidence: Existing evidence pack dict (will be modified in-place).
        governance_signal: Signal from to_governance_signal_for_replay_safety().

    Returns:
        Modified evidence dict with replay_safety governance attached.
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Extract status (JSON-safe normalization)
    status = governance_signal.get("status") or governance_signal.get("governance_status", PromotionStatus.OK)
    if isinstance(status, Enum):
        status_str = status.value
    else:
        status_str = str(status).lower()

    # Extract alignment (JSON-safe normalization)
    alignment = governance_signal.get("governance_alignment") or governance_signal.get("alignment")
    if isinstance(alignment, GovernanceAlignment):
        alignment_str = alignment.value
    elif isinstance(alignment, str):
        alignment_str = alignment.lower()
    else:
        alignment_str = "aligned"

    # Extract safety and radar statuses (JSON-safe)
    safety_status = governance_signal.get("safety_status", PromotionStatus.OK)
    if isinstance(safety_status, Enum):
        safety_status_str = safety_status.value
    else:
        safety_status_str = str(safety_status).lower()

    radar_status = governance_signal.get("radar_status")
    if radar_status is not None:
        if isinstance(radar_status, Enum):
            radar_status_str = radar_status.value
        else:
            radar_status_str = str(radar_status).lower()
    else:
        radar_status_str = None

    # Extract reasons and build compact summary
    reasons = governance_signal.get("reasons", [])
    reason_summary = _build_compact_reason_summary(reasons)

    # Extract conflict flag
    conflict = governance_signal.get("conflict", False)

    # Build replay safety tile with deterministic field ordering
    # Using explicit ordering for JSON consistency
    replay_tile = {
        "advisory_only": True,  # Advisory flag - this tile does not gate evidence pack
        "conflict": conflict,
        "governance_alignment": alignment_str,
        "radar_status": radar_status_str,
        "reason_summary": reason_summary,
        "reasons": sorted(reasons),  # Deterministic ordering
        "safe_for_policy_update": governance_signal.get("safe_for_policy_update", True),
        "safe_for_promotion": governance_signal.get("safe_for_promotion", True),
        "safety_status": safety_status_str,
        "schema_version": "1.0.0",
        "status": status_str,
    }

    # Attach to governance section
    evidence["governance"]["replay_safety"] = replay_tile

    return evidence


# =============================================================================
# PHASE X: Integration TODOs
# =============================================================================

# TODO(PHASE-X-REPLAY): Register replay_governance tile in build_global_health_surface()
# Location: backend/health/global_surface.py
# Integration pattern: Follow attach_*_tile() pattern used by other governance tiles
# Depends on: replay_governance_adapter.py creation
# Schema: docs/system_law/schemas/replay/replay_global_console_tile.schema.json
# SHADOW MODE CONTRACT: Tile is purely observational, no control flow influence

# TODO(PHASE-X-REPLAY): Add replay_safety signal to CLAUDE I governance registry
# Location: backend/analytics/governance_verifier.py (or designated registry)
# Signal type: "replay_safety"
# Fusion rule: Conservative BLOCK propagation (if replay BLOCKs, overall BLOCKs)
# Integration: to_governance_signal_for_replay_safety() produces registry-compatible signal
# Schema: docs/system_law/schemas/replay/replay_governance_radar.schema.json

# TODO(PHASE-X-REPLAY): Create replay_governance_adapter.py
# Location: backend/health/replay_governance_adapter.py
# Functions:
#   - build_replay_governance_tile_for_global_health()
#   - attach_replay_governance_tile()
# Contract: SHADOW mode only, observational, no control flow influence

__all__ = [
    # Enums
    "PromotionStatus",
    "SafetyLevel",
    "GovernanceAlignment",
    # Data structures
    "ReplaySafetyResult",
    "GovernanceSignal",
    # Core functions
    "verify_replay_safety",
    "emit_governance_signal",
    "check_replay_determinism",
    "evaluate_replay_safety_for_promotion",
    # Phase IV: Evidence & Director Panel
    "summarize_replay_safety_for_evidence",
    "build_replay_safety_director_panel",
    "build_replay_safety_envelope",
    "compute_replay_confidence",
    # Phase V: Governance View
    "build_replay_safety_governance_view",
    # Phase VI: Governance Signal Adapter
    "to_governance_signal_for_replay_safety",
    # Phase VII: Director Tile & Alignment View Adapters
    "build_replay_safety_director_tile",
    "replay_safety_for_alignment_view",
    "attach_replay_safety_to_evidence",
]
