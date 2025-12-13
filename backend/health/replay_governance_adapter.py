"""Replay Governance Adapter for Global Health Surface.

Provides replay governance tile builders and signal adapters for Phase X integration.

SHADOW MODE CONTRACT:
- All functions in this module are purely observational
- No function influences governance decisions or control flow
- Tiles are advisory only and do NOT gate any operations
- All outputs are JSON-safe and deterministically ordered

Schema References:
    docs/system_law/schemas/replay/replay_governance_radar.schema.json
    docs/system_law/schemas/replay/replay_promotion_eval.schema.json
    docs/system_law/schemas/replay/replay_director_panel.schema.json
    docs/system_law/schemas/replay/replay_global_console_tile.schema.json

System Law Reference:
    docs/system_law/Replay_Governance_PhaseX_Binding.md
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__version__ = "1.0.0"

# Schema version for replay governance tile
REPLAY_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"

# Status light mapping from director panel status
STATUS_LIGHT_MAP = {
    "ok": "GREEN",
    "warn": "YELLOW",
    "block": "RED",
}

# Governance alignment enum values
ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_TENSION = "tension"
ALIGNMENT_DIVERGENT = "divergent"


def build_replay_console_tile(
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build replay governance console tile for global health surface.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - This is purely for observability and logging

    Args:
        radar: Replay governance radar view (from build_replay_safety_governance_view)
        promotion_eval: Promotion evaluation (from evaluate_replay_safety_for_promotion)
        director_panel: Optional director panel (from build_replay_safety_director_panel)

    Returns:
        Console tile conforming to replay_global_console_tile.schema.json

    Schema: docs/system_law/schemas/replay/replay_global_console_tile.schema.json
    """
    # Extract status from radar (default to "ok" if not present)
    status = radar.get("status", "ok")
    if isinstance(status, str):
        status = status.lower()
    else:
        # Handle enum values
        status = getattr(status, "value", str(status)).lower()

    # Normalize status to valid enum
    if status not in ("ok", "warn", "block"):
        status = "ok"

    # Extract governance alignment
    alignment = radar.get("governance_alignment", radar.get("alignment", ALIGNMENT_ALIGNED))
    if hasattr(alignment, "value"):
        alignment = alignment.value

    # Extract safety status
    safe = promotion_eval.get("safe", True)

    # Build summary metrics
    summary_metrics = {
        "determinism_rate": _extract_determinism_rate(radar, promotion_eval),
        "critical_incident_rate": _extract_critical_incident_rate(radar, director_panel),
        "hot_fingerprints_count": _extract_hot_fingerprints_count(radar),
        "replay_ok_for_promotion": promotion_eval.get("safe", True),
    }

    # Build governance signal (collapsed from radar)
    governance_signal = {
        "signal_type": "replay_safety",
        "status": status,
        "governance_status": status,
        "governance_alignment": alignment,
        "conflict": radar.get("conflict", False),
        "reasons": _extract_reasons(radar),
    }

    # Build promotion eval summary
    promotion_eval_summary = {
        "promotion_status": _normalize_promotion_status(promotion_eval.get("promotion_status")),
        "safety_level": _normalize_safety_level(promotion_eval.get("safety_level")),
    }

    # Build director panel summary (if available)
    director_panel_summary = None
    if director_panel is not None:
        director_panel_summary = {
            "headline": director_panel.get("headline", ""),
            "recommendation": director_panel.get("recommendation", "proceed"),
            "conflict_flag": director_panel.get("conflict_flag", False),
        }

    # Build radar summary
    radar_summary = {
        "governance_alignment": alignment,
        "safe_for_policy_update": radar.get("safe_for_policy_update", True),
        "safe_for_promotion": radar.get("safe_for_promotion", True),
    }

    # Build SHADOW mode contract attestation (always true for this adapter)
    shadow_mode_contract = {
        "observational_only": True,
        "no_control_flow_influence": True,
        "no_governance_modification": True,
    }

    # Build Phase X metadata
    phase_x_metadata = {
        "phase": "P3",  # Default to P3 (synthetic validation)
        "doctrine_ref": "docs/system_law/Replay_Governance_PhaseX_Binding.md",
        "whitepaper_evidence_tag": "replay_governance_v1",
    }

    # Assemble tile with deterministic key ordering
    tile = {
        "schema_version": REPLAY_GOVERNANCE_TILE_SCHEMA_VERSION,
        "tile_type": "replay_governance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "status": status,
        "safe": safe,
        "status_light": STATUS_LIGHT_MAP.get(status, "GREEN"),
        "governance_alignment": alignment,
        "governance_signal": governance_signal,
        "summary_metrics": summary_metrics,
        "promotion_eval": promotion_eval_summary,
        "radar_summary": radar_summary,
        "shadow_mode_contract": shadow_mode_contract,
        "phase_x_metadata": phase_x_metadata,
    }

    # Add director panel if available
    if director_panel_summary is not None:
        tile["director_panel"] = director_panel_summary

    return tile


def replay_to_governance_signal(
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert replay radar and promotion eval to a governance signal.

    Follows the signal collapse algorithm from Replay_Governance_PhaseX_Binding.md Section 3.

    COLLAPSE RULES:
        1. IF safety_status == BLOCK OR radar_status == BLOCK: status = BLOCK
        2. ELSE IF governance_alignment == DIVERGENT: status = BLOCK
        3. ELSE IF conflict == TRUE: status = BLOCK
        4. ELSE IF safety_status == WARN OR radar_status == WARN: status = WARN
        5. ELSE IF governance_alignment == TENSION: status = WARN
        6. ELSE: status = OK

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The signal it produces does NOT influence any governance decisions
    - This is purely for observability and logging

    Args:
        radar: Replay governance radar view
        promotion_eval: Promotion evaluation

    Returns:
        Governance signal conforming to replay_governance_radar.schema.json

    Schema: docs/system_law/schemas/replay/replay_governance_radar.schema.json
    """
    # Extract statuses (normalize to lowercase strings)
    safety_status = _normalize_status(promotion_eval.get("status", "ok"))
    radar_status = _normalize_status(radar.get("status", "ok"))

    # Extract alignment
    alignment = radar.get("governance_alignment", radar.get("alignment", ALIGNMENT_ALIGNED))
    if hasattr(alignment, "value"):
        alignment = alignment.value

    # Extract conflict flag
    conflict = radar.get("conflict", False)

    # Apply collapse rules (from Replay_Governance_PhaseX_Binding.md Section 3.1)
    if safety_status == "block" or radar_status == "block":
        status = "block"
    elif alignment == ALIGNMENT_DIVERGENT:
        status = "block"
    elif conflict:
        status = "block"
    elif safety_status == "warn" or radar_status == "warn":
        status = "warn"
    elif alignment == ALIGNMENT_TENSION:
        status = "warn"
    else:
        status = "ok"

    # Collect and prefix reasons
    reasons: List[str] = []

    # Add safety reasons with [Replay] prefix
    safety_reasons = promotion_eval.get("reasons", promotion_eval.get("violations", []))
    for reason in safety_reasons:
        if isinstance(reason, str):
            if not reason.startswith("[Replay]"):
                reasons.append(f"[Replay] {reason}")
            else:
                reasons.append(reason)

    # Add radar reasons with [Replay] prefix
    radar_reasons = radar.get("reasons", [])
    for reason in radar_reasons:
        if isinstance(reason, str):
            if not reason.startswith("[Replay]"):
                reasons.append(f"[Replay] {reason}")
            else:
                reasons.append(reason)

    # Add conflict reason if divergent
    if alignment == ALIGNMENT_DIVERGENT:
        reasons.append("[Replay] [CONFLICT] Safety and Radar alignment is DIVERGENT")

    # Deduplicate while preserving order
    seen = set()
    unique_reasons = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    return {
        "signal_type": "replay_safety",
        "status": status,
        "governance_status": status,
        "governance_alignment": alignment,
        "safety_status": safety_status,
        "radar_status": radar_status,
        "conflict": conflict or (alignment == ALIGNMENT_DIVERGENT),
        "reasons": unique_reasons,
        "safe_for_policy_update": radar.get("safe_for_policy_update", status != "block"),
        "safe_for_promotion": radar.get("safe_for_promotion", status != "block"),
    }


def attach_replay_governance_to_evidence(
    evidence: Dict[str, Any],
    replay_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach replay governance signal to evidence pack.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment does NOT influence any governance decisions
    - This is purely for observability and evidence collection

    Args:
        evidence: Existing evidence pack (not modified)
        replay_signal: Replay governance signal to attach

    Returns:
        New evidence dict with replay signal attached under evidence["governance"]["replay"]
    """
    # Create shallow copy of evidence
    result = dict(evidence)

    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}
    else:
        # Shallow copy governance to avoid mutating original
        result["governance"] = dict(result["governance"])

    # Attach replay signal (read-only copy)
    result["governance"]["replay"] = dict(replay_signal)

    return result


def build_replay_governance_tile_for_global_health(
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build replay governance tile for global health surface attachment.

    This is the entry point for global_surface.py integration.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - The tile does NOT influence any other tiles
    - No control flow depends on this tile

    Args:
        radar: Replay governance radar view
        promotion_eval: Promotion evaluation
        director_panel: Optional director panel

    Returns:
        Tile suitable for global health surface attachment
    """
    return build_replay_console_tile(radar, promotion_eval, director_panel)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_status(status: Any) -> str:
    """Normalize status to lowercase string."""
    if hasattr(status, "value"):
        status = status.value
    if isinstance(status, str):
        return status.lower()
    return "ok"


def _normalize_promotion_status(status: Any) -> str:
    """Normalize promotion status to valid enum value."""
    if hasattr(status, "value"):
        status = status.value
    if isinstance(status, str):
        status = status.lower()
        if status in ("pending", "approved", "rejected", "abstain"):
            return status
    return "pending"


def _normalize_safety_level(level: Any) -> str:
    """Normalize safety level to valid enum value."""
    if hasattr(level, "value"):
        level = level.value
    if isinstance(level, str):
        level = level.lower()
        if level in ("safe", "caution", "unsafe", "unknown"):
            return level
    return "unknown"


def _extract_determinism_rate(radar: Dict[str, Any], promotion_eval: Dict[str, Any]) -> float:
    """Extract determinism rate from radar or promotion eval."""
    # Try metrics.determinism_score first
    metrics = radar.get("metrics", {})
    if "determinism_score" in metrics:
        return float(metrics["determinism_score"]) / 100.0

    # Try hash_match_rate
    if "hash_match_rate" in metrics:
        return float(metrics["hash_match_rate"])

    # Try trace_summary.confidence_score
    trace_summary = promotion_eval.get("trace_summary", {})
    if "confidence_score" in trace_summary:
        return float(trace_summary["confidence_score"])

    # Default to 1.0 (fully deterministic) if safe
    return 1.0 if promotion_eval.get("safe", True) else 0.0


def _extract_critical_incident_rate(
    radar: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]],
) -> float:
    """Extract critical incident rate."""
    # Try metrics.violation_count
    metrics = radar.get("metrics", {})
    violation_count = metrics.get("violation_count", 0)

    # Try director panel violation count
    if director_panel is not None:
        violation_count = max(violation_count, director_panel.get("violation_count", 0))

    # Normalize: 0 violations = 0.0 rate, any violations = proportional rate
    # Cap at 1.0 for 10+ violations
    return min(1.0, violation_count / 10.0)


def _extract_hot_fingerprints_count(radar: Dict[str, Any]) -> int:
    """Extract hot fingerprints count from radar."""
    # Try drift_indicators
    drift = radar.get("drift_indicators", {})
    count = 0
    if drift.get("h_t_drift_detected", False):
        count += 1
    if drift.get("config_drift_detected", False):
        count += 1
    if drift.get("state_drift_detected", False):
        count += 1

    # Try metrics.violation_count as fallback
    if count == 0:
        metrics = radar.get("metrics", {})
        count = metrics.get("violation_count", 0)

    return count


def _extract_reasons(radar: Dict[str, Any]) -> List[str]:
    """Extract reasons from radar."""
    reasons = radar.get("reasons", [])
    if isinstance(reasons, list):
        return [str(r) for r in reasons]
    return []


# =============================================================================
# PHASE X P5: FIRST-LIGHT BINDING
# =============================================================================

def attach_replay_governance_to_p3_stability_report(
    stability_report: Dict[str, Any],
    replay_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach replay governance summary to P3 stability report.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment is advisory only, does NOT gate any operations
    - This is purely for observability and evidence collection

    P3 Summary Fields:
    - status: Collapsed governance status (ok/warn/block)
    - determinism_rate: Determinism confidence (0.0-1.0)
    - critical_incident_rate: Incident rate (0.0-1.0)
    - hot_fingerprints_count: Count of drift indicators
    - governance_alignment: Alignment status (aligned/tension/divergent)

    Args:
        stability_report: P3 stability report (not modified)
        replay_signal: Replay governance signal from replay_to_governance_signal()

    Returns:
        New stability report dict with replay summary under report["replay_governance"]
    """
    # Create shallow copy
    result = dict(stability_report)

    # Extract P3 summary fields from signal
    status = _normalize_status(replay_signal.get("status", "ok"))
    alignment = replay_signal.get("governance_alignment", ALIGNMENT_ALIGNED)

    # Compute determinism_rate from signal metadata or default
    determinism_rate = 1.0
    if replay_signal.get("status") == "block":
        determinism_rate = 0.0
    elif replay_signal.get("status") == "warn":
        determinism_rate = 0.7

    # Compute critical_incident_rate based on conflict/block
    critical_incident_rate = 0.0
    if replay_signal.get("conflict", False):
        critical_incident_rate = 0.5
    if status == "block":
        critical_incident_rate = 1.0

    # Count hot fingerprints from reasons
    hot_fingerprints_count = len(replay_signal.get("reasons", []))

    # Build P3 summary
    p3_summary = {
        "status": status,
        "determinism_rate": determinism_rate,
        "critical_incident_rate": critical_incident_rate,
        "hot_fingerprints_count": hot_fingerprints_count,
        "governance_alignment": alignment,
    }

    result["replay_governance"] = p3_summary
    return result


def attach_replay_governance_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    replay_signal: Dict[str, Any],
    recency_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach replay governance summary to P4 calibration report.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment is advisory only, does NOT gate any operations
    - This is purely for observability and calibration evidence

    P4 Calibration Fields:
    - status: Collapsed governance status (ok/warn/block)
    - recency_of_replay: ISO timestamp of most recent replay validation
    - safety_status: Safety subsystem status
    - radar_status: Radar subsystem status
    - conflict: True if safety-radar conflict detected

    Args:
        calibration_report: P4 calibration report (not modified)
        replay_signal: Replay governance signal from replay_to_governance_signal()
        recency_timestamp: Optional ISO timestamp for recency_of_replay (defaults to now)

    Returns:
        New calibration report dict with replay calibration under report["replay_calibration"]
    """
    # Create shallow copy
    result = dict(calibration_report)

    # Extract P4 calibration fields from signal
    status = _normalize_status(replay_signal.get("status", "ok"))
    safety_status = _normalize_status(replay_signal.get("safety_status", "ok"))
    radar_status = _normalize_status(replay_signal.get("radar_status", "ok"))
    conflict = replay_signal.get("conflict", False)

    # Use provided timestamp or generate current
    recency = recency_timestamp or datetime.now(timezone.utc).isoformat()

    # Build P4 calibration summary
    p4_calibration = {
        "status": status,
        "recency_of_replay": recency,
        "safety_status": safety_status,
        "radar_status": radar_status,
        "conflict": conflict,
    }

    result["replay_calibration"] = p4_calibration
    return result


# =============================================================================
# PHASE X P5: GGFL FUSION ADAPTER
# =============================================================================

def replay_for_alignment_view(
    replay_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize replay GovernanceSignal into GGFL unified format.

    This function converts a replay governance signal into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.

    READ-ONLY: Does not modify input signal.

    GGFL Unified Format:
    - status: Lowercase status (ok/warn/block)
    - alignment: Governance alignment (aligned/tension/divergent)
    - conflict: Boolean conflict flag
    - top_reasons: List of reasons with [Replay] prefix stripped for human readability

    Prefix Stripping Rules:
    - "[Replay] " prefix is stripped from start of reasons
    - "[CONFLICT] " is preserved as it indicates critical state
    - Other prefixes are preserved

    Args:
        replay_signal: Replay governance signal from replay_to_governance_signal()

    Returns:
        GGFL-normalized dict with status, alignment, conflict, top_reasons
    """
    # Extract and normalize status (lowercase for fusion)
    status = _normalize_status(replay_signal.get("status", "ok"))

    # Extract alignment
    alignment = replay_signal.get("governance_alignment", ALIGNMENT_ALIGNED)
    if hasattr(alignment, "value"):
        alignment = alignment.value

    # Extract conflict flag
    conflict = replay_signal.get("conflict", False)

    # Process reasons: strip [Replay] prefix for human summaries
    raw_reasons = replay_signal.get("reasons", [])
    top_reasons: List[str] = []

    for reason in raw_reasons:
        if isinstance(reason, str):
            # Strip "[Replay] " prefix if present at start
            stripped = reason
            if stripped.startswith("[Replay] "):
                stripped = stripped[9:]  # len("[Replay] ") = 9
            elif stripped.startswith("[Replay]"):
                stripped = stripped[8:]  # len("[Replay]") = 8

            # Preserve [CONFLICT] as it's critical
            # Don't strip other prefixes

            if stripped:
                top_reasons.append(stripped)

    # Limit to top 5 reasons for alignment view
    top_reasons = top_reasons[:5]

    return {
        "status": status,
        "alignment": alignment,
        "conflict": conflict,
        "top_reasons": top_reasons,
    }


# =============================================================================
# PHASE X P5: REAL-TELEMETRY REPLAY FUNCTIONS
# =============================================================================

# P5 Schema version for real-telemetry signals
P5_REPLAY_SCHEMA_VERSION = "1.0.0"

# P5 Determinism band thresholds (from Appendix C)
P5_DETERMINISM_GREEN_THRESHOLD = 0.85
P5_DETERMINISM_YELLOW_THRESHOLD = 0.70

# =============================================================================
# CANONICALIZATION: Extraction Source Enum (v1.3.0)
# =============================================================================
# Provenance tracking for signal extraction source

EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_EVIDENCE_JSON = "EVIDENCE_JSON"
EXTRACTION_SOURCE_DIRECT_LOG = "DIRECT_LOG"
EXTRACTION_SOURCE_MISSING = "MISSING"

# =============================================================================
# CANONICALIZATION: Driver Reason Codes (v1.3.0)
# =============================================================================
# Frozen driver codes for GGFL/Status - NO PROSE, deterministic ordering
# Cap = 3 reasons max per signal

DRIVER_SCHEMA_NOT_OK = "DRIVER_SCHEMA_NOT_OK"
DRIVER_SAFETY_MISMATCH_PRESENT = "DRIVER_SAFETY_MISMATCH_PRESENT"
DRIVER_STATE_MISMATCH_PRESENT = "DRIVER_STATE_MISMATCH_PRESENT"
DRIVER_DETERMINISM_RED_BAND = "DRIVER_DETERMINISM_RED_BAND"

# Driver reason code priority order (for single-cap and ordering)
DRIVER_PRIORITY_ORDER = [
    DRIVER_SCHEMA_NOT_OK,           # Priority 1: Schema validation failed
    DRIVER_SAFETY_MISMATCH_PRESENT, # Priority 2: Ω/blocked mismatch
    DRIVER_STATE_MISMATCH_PRESENT,  # Priority 3: H/rho/tau/beta mismatch
    DRIVER_DETERMINISM_RED_BAND,    # Priority 4: Fallback legacy RED band
]

# Maximum driver codes per signal
DRIVER_REASON_CAP = 3


def compute_driver_codes(
    schema_ok: bool,
    safety_mismatch_rate: float,
    state_mismatch_rate: float,
    determinism_band: str,
) -> List[str]:
    """Compute frozen driver reason codes for GGFL/Status.

    CANONICALIZATION CONTRACT (v1.3.0):
    - Driver codes are frozen enum values, NO PROSE
    - Priority order determines selection
    - Cap = 3 codes max per signal
    - Deterministic ordering (sorted by priority)

    Args:
        schema_ok: Whether schema validation passed
        safety_mismatch_rate: Ω/blocked mismatch rate (0.0-1.0)
        state_mismatch_rate: H/rho/tau/beta mismatch rate (0.0-1.0)
        determinism_band: Determinism band (GREEN/YELLOW/RED)

    Returns:
        List of driver codes (max 3), sorted by priority
    """
    codes: List[str] = []

    # Priority 1: Schema validation failed
    if not schema_ok:
        codes.append(DRIVER_SCHEMA_NOT_OK)

    # Priority 2: Safety mismatch present
    if safety_mismatch_rate > 0.0:
        codes.append(DRIVER_SAFETY_MISMATCH_PRESENT)

    # Priority 3: State mismatch present
    if state_mismatch_rate > 0.0:
        codes.append(DRIVER_STATE_MISMATCH_PRESENT)

    # Priority 4: Determinism RED band (fallback)
    if determinism_band == "RED":
        codes.append(DRIVER_DETERMINISM_RED_BAND)

    # Apply cap and sort by priority order
    result: List[str] = []
    for driver in DRIVER_PRIORITY_ORDER:
        if driver in codes:
            result.append(driver)
            if len(result) >= DRIVER_REASON_CAP:
                break

    return result


def _compute_p5_determinism_band(determinism_rate: float) -> str:
    """Compute P5 determinism band from rate.

    P5 Band Thresholds (from Replay_Safety_Governance_Law.md Appendix C):
        GREEN: determinism_rate >= 0.85
        YELLOW: 0.70 <= determinism_rate < 0.85
        RED: determinism_rate < 0.70
    """
    if determinism_rate >= P5_DETERMINISM_GREEN_THRESHOLD:
        return "GREEN"
    elif determinism_rate >= P5_DETERMINISM_YELLOW_THRESHOLD:
        return "YELLOW"
    else:
        return "RED"


def _compute_p5_status_from_band(band: str) -> str:
    """Compute status from P5 determinism band."""
    if band == "GREEN":
        return "ok"
    elif band == "YELLOW":
        return "warn"
    else:
        return "block"


def extract_p5_replay_safety_from_logs(
    replay_logs: List[Dict[str, Any]],
    production_run_id: str,
    expected_hashes: Optional[Dict[str, str]] = None,
    *,
    telemetry_source: str = "real",
    extraction_source: str = EXTRACTION_SOURCE_DIRECT_LOG,
    input_schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract P5 replay safety signal from raw replay logs.

    This function processes production replay logs and produces a replay safety
    signal suitable for P5 real-telemetry validation.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The signal it produces does NOT influence any governance decisions
    - This is purely for observability and evidence collection

    CANONICALIZATION CONTRACT (v1.3.0):
    - extraction_source tracks provenance (MANIFEST|EVIDENCE_JSON|DIRECT_LOG|MISSING)
    - input_schema_version passes through source schema version (or "UNKNOWN")
    - driver_codes are frozen enum values (NO PROSE), cap=3, deterministic order

    Args:
        replay_logs: List of replay log entries from production run.
            Each entry should contain:
            - "trace_hash": SHA-256 hash of trace data
            - "timestamp": ISO 8601 timestamp
            - "cycle_id": Cycle identifier
            - "determinism_check": Optional dict with verification results
        production_run_id: Unique identifier for the production run.
        expected_hashes: Optional dict mapping cycle_id -> expected_hash
            for determinism verification.
        telemetry_source: Source identifier ("real" for P5, "shadow" for P4,
            "synthetic" for P3).
        extraction_source: Provenance source (MANIFEST|EVIDENCE_JSON|DIRECT_LOG|MISSING).
        input_schema_version: Schema version from source data (or None for UNKNOWN).

    Returns:
        P5 replay safety signal with:
        - schema_version: str
        - status: ok | warn | block
        - determinism_rate: float (0.0-1.0)
        - determinism_band: GREEN | YELLOW | RED
        - hash_match_count: int
        - hash_mismatch_count: int
        - critical_incidents: List[Dict]
        - telemetry_source: str
        - production_run_id: str
        - replay_latency_ms: Optional[float]
        - p5_grade: bool (True if meets P5 requirements)
        - reasons: List[str]
        - extraction_source: str (v1.3.0 provenance)
        - input_schema_version: str (v1.3.0 provenance)
        - driver_codes: List[str] (v1.3.0 frozen codes, cap=3)

    P5 Band Thresholds (from Appendix C):
        GREEN: determinism_rate >= 0.85
        YELLOW: 0.70 <= determinism_rate < 0.85
        RED: determinism_rate < 0.70
    """
    # Initialize counters
    hash_match_count = 0
    hash_mismatch_count = 0
    critical_incidents: List[Dict[str, Any]] = []
    reasons: List[str] = []
    total_latency_ms = 0.0
    latency_count = 0

    # True Divergence v1 counters (breakdown by mismatch type)
    safety_mismatch_count = 0  # Ω/blocked mismatches
    state_mismatch_count = 0   # H/rho/tau/beta mismatches
    outcome_mismatch_count = 0  # General outcome mismatches
    total_entries_with_metrics = 0
    prob_success_sum = 0.0
    prob_success_count = 0

    # Process each replay log entry
    for log_entry in replay_logs:
        cycle_id = log_entry.get("cycle_id", "unknown")
        trace_hash = log_entry.get("trace_hash")
        timestamp = log_entry.get("timestamp")

        # Check for latency data
        if "latency_ms" in log_entry:
            total_latency_ms += float(log_entry["latency_ms"])
            latency_count += 1

        # Track prob(success) for Brier score if available
        if "prob_success" in log_entry:
            prob_success_sum += float(log_entry["prob_success"])
            prob_success_count += 1

        # Verify hash if expected_hashes provided
        if expected_hashes is not None and cycle_id in expected_hashes:
            expected = expected_hashes[cycle_id]
            if trace_hash == expected:
                hash_match_count += 1
            else:
                hash_mismatch_count += 1
                outcome_mismatch_count += 1
                reasons.append(f"[Replay] Hash mismatch at cycle {cycle_id}")
                critical_incidents.append({
                    "cycle_id": cycle_id,
                    "type": "hash_mismatch",
                    "expected": expected,
                    "actual": trace_hash,
                    "timestamp": timestamp,
                })
        elif trace_hash is not None:
            # No expected hash, count as match (no verification possible)
            hash_match_count += 1

        # Check determinism_check field if present
        det_check = log_entry.get("determinism_check", {})
        if det_check.get("failed", False):
            reasons.append(f"[Replay] Determinism check failed at cycle {cycle_id}")
            critical_incidents.append({
                "cycle_id": cycle_id,
                "type": "determinism_failure",
                "details": det_check.get("reason", "Unknown"),
                "timestamp": timestamp,
            })

        # True Divergence v1: Track safety mismatches (Ω/blocked)
        # Safety fields: omega, blocked, gating_decision
        safety_fields = log_entry.get("safety_mismatch", {})
        if safety_fields.get("omega_mismatch") or safety_fields.get("blocked_mismatch"):
            safety_mismatch_count += 1
            if not any("[Safety]" in r for r in reasons):
                reasons.append("[Safety] Ω/blocked mismatch detected")

        # True Divergence v1: Track state mismatches (H/rho/tau/beta)
        # State fields: H, rho, tau, beta
        state_fields = log_entry.get("state_mismatch", {})
        if any(state_fields.get(f"{k}_mismatch") for k in ["H", "rho", "tau", "beta"]):
            state_mismatch_count += 1

        # Count entries that have mismatch tracking data
        if "safety_mismatch" in log_entry or "state_mismatch" in log_entry:
            total_entries_with_metrics += 1

    # Compute determinism rate
    total_checks = hash_match_count + hash_mismatch_count
    if total_checks > 0:
        determinism_rate = hash_match_count / total_checks
    else:
        # No checks performed, assume deterministic
        determinism_rate = 1.0

    # Compute band and status
    determinism_band = _compute_p5_determinism_band(determinism_rate)
    status = _compute_p5_status_from_band(determinism_band)

    # Compute average latency
    replay_latency_ms: Optional[float] = None
    if latency_count > 0:
        replay_latency_ms = total_latency_ms / latency_count

    # Determine P5 grade (must have real telemetry source and production_run_id)
    p5_grade = (
        telemetry_source == "real"
        and bool(production_run_id)
        and total_checks > 0
    )

    # Add summary reason if not OK
    if status == "warn":
        reasons.insert(0, f"[Replay] Determinism rate {determinism_rate:.2%} in YELLOW band")
    elif status == "block":
        reasons.insert(0, f"[Replay] Determinism rate {determinism_rate:.2%} in RED band")

    # =========================================================================
    # True Divergence v1: Compute breakdown rates
    # =========================================================================
    # Use total_entries_with_metrics as denominator if available, else total_checks
    td_denominator = total_entries_with_metrics if total_entries_with_metrics > 0 else total_checks

    # Outcome mismatch rate (legacy determinism_rate inverse)
    outcome_mismatch_rate = (
        outcome_mismatch_count / td_denominator if td_denominator > 0 else 0.0
    )

    # Safety mismatch rate (Ω/blocked only)
    safety_mismatch_rate = (
        safety_mismatch_count / td_denominator if td_denominator > 0 else 0.0
    )

    # State mismatch rate (H/rho/tau/beta only)
    state_mismatch_rate = (
        state_mismatch_count / td_denominator if td_denominator > 0 else 0.0
    )

    # Brier score for prob(success) if available
    brier_score_success: Optional[float] = None
    if prob_success_count > 0:
        # Simplified Brier: mean squared error of predicted success prob
        # For full Brier, we'd need actual outcomes; here we use mean as proxy
        mean_prob = prob_success_sum / prob_success_count
        brier_score_success = mean_prob * (1.0 - mean_prob)

    # Build true_divergence_v1 vector (deterministic key ordering)
    true_divergence_v1 = {
        "brier_score_success": brier_score_success,
        "outcome_mismatch_count": outcome_mismatch_count,
        "outcome_mismatch_rate": outcome_mismatch_rate,
        "safety_mismatch_count": safety_mismatch_count,
        "safety_mismatch_rate": safety_mismatch_rate,
        "state_mismatch_count": state_mismatch_count,
        "state_mismatch_rate": state_mismatch_rate,
        "total_entries_with_metrics": total_entries_with_metrics,
        "version": "1.0.0",
    }

    # =========================================================================
    # Canonicalization v1.3.0: Compute driver codes
    # =========================================================================
    # schema_ok defaults to True for direct log extraction (no schema validation)
    schema_ok = True  # Will be set by caller if schema validation fails
    driver_codes = compute_driver_codes(
        schema_ok=schema_ok,
        safety_mismatch_rate=safety_mismatch_rate,
        state_mismatch_rate=state_mismatch_rate,
        determinism_band=determinism_band,
    )

    # Build signal with deterministic key ordering (sorted)
    signal = {
        "critical_incidents": critical_incidents,
        "determinism_band": determinism_band,
        "determinism_rate": determinism_rate,
        # Canonicalization v1.3.0: Frozen driver codes (NO PROSE, cap=3)
        "driver_codes": driver_codes,
        # Canonicalization v1.3.0: Extraction provenance
        "extraction_source": extraction_source,
        "hash_match_count": hash_match_count,
        "hash_mismatch_count": hash_mismatch_count,
        # Canonicalization v1.3.0: Input schema version passthrough
        "input_schema_version": input_schema_version or "UNKNOWN",
        # Legacy metric label to prevent confusion with true_divergence_v1
        # FROZEN (v1.3.0): This label is non-equivalent to true_divergence_v1
        "legacy_metric_label": "RAW_ANY_MISMATCH",
        "p5_grade": p5_grade,
        "production_run_id": production_run_id,
        "reasons": reasons,
        "replay_latency_ms": replay_latency_ms,
        "schema_version": P5_REPLAY_SCHEMA_VERSION,
        "status": status,
        "telemetry_source": telemetry_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_logs_processed": len(replay_logs),
        # True Divergence v1 vector (new metric breakdown)
        "true_divergence_v1": true_divergence_v1,
    }

    return signal


def build_p5_replay_governance_tile(
    p5_signal: Dict[str, Any],
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    *,
    include_p5_extensions: bool = True,
) -> Dict[str, Any]:
    """Build P5-grade replay governance tile for global health surface.

    This function extends build_replay_console_tile() with P5-specific fields
    and validation.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - The tile does NOT influence any other tiles
    - No control flow depends on this tile

    Args:
        p5_signal: P5 replay safety signal from extract_p5_replay_safety_from_logs().
        radar: Replay governance radar view.
        promotion_eval: Promotion evaluation result.
        include_p5_extensions: If True, include P5-only fields
            (telemetry_source, production_run_id, replay_latency_ms).

    Returns:
        P5 replay governance tile with:
        - All fields from build_replay_console_tile()
        - p5_grade: bool (True if signal meets P5 requirements)
        - telemetry_source: "real" (P5-only)
        - production_run_id: str (P5-only)
        - replay_latency_ms: float (P5-only, optional)
        - determinism_band: "GREEN" | "YELLOW" | "RED"
        - phase: "P5"

    P5 Grade Requirements:
        - telemetry_source == "real"
        - production_run_id is present and non-empty
        - determinism_rate is computed from actual replay logs
    """
    # Start with base tile
    base_tile = build_replay_console_tile(radar, promotion_eval)

    # Extract P5-specific values from signal
    p5_grade = p5_signal.get("p5_grade", False)
    determinism_band = p5_signal.get("determinism_band", "GREEN")
    determinism_rate = p5_signal.get("determinism_rate", 1.0)
    telemetry_source = p5_signal.get("telemetry_source", "real")
    production_run_id = p5_signal.get("production_run_id", "")
    replay_latency_ms = p5_signal.get("replay_latency_ms")

    # Update status from P5 signal if more severe
    p5_status = _normalize_status(p5_signal.get("status", "ok"))
    base_status = base_tile.get("status", "ok")

    # Use more severe status
    if p5_status == "block" or base_status == "block":
        final_status = "block"
    elif p5_status == "warn" or base_status == "warn":
        final_status = "warn"
    else:
        final_status = "ok"

    # Build P5 tile with deterministic key ordering
    tile = {
        "determinism_band": determinism_band,
        "governance_alignment": base_tile.get("governance_alignment", ALIGNMENT_ALIGNED),
        "governance_signal": base_tile.get("governance_signal", {}),
        "mode": "SHADOW",
        "p5_grade": p5_grade,
        "phase": "P5",
        "phase_x_metadata": {
            "doctrine_ref": "docs/system_law/Replay_Governance_PhaseX_Binding.md",
            "phase": "P5",
            "whitepaper_evidence_tag": "replay_governance_p5_v1",
        },
        "promotion_eval": base_tile.get("promotion_eval", {}),
        "radar_summary": base_tile.get("radar_summary", {}),
        "safe": base_tile.get("safe", True) and p5_status != "block",
        "schema_version": P5_REPLAY_SCHEMA_VERSION,
        "shadow_mode_contract": {
            "no_control_flow_influence": True,
            "no_governance_modification": True,
            "observational_only": True,
        },
        "status": final_status,
        "status_light": STATUS_LIGHT_MAP.get(final_status, "GREEN"),
        "summary_metrics": {
            "critical_incident_rate": base_tile.get("summary_metrics", {}).get("critical_incident_rate", 0.0),
            "determinism_rate": determinism_rate,
            "hot_fingerprints_count": base_tile.get("summary_metrics", {}).get("hot_fingerprints_count", 0),
            "replay_ok_for_promotion": final_status != "block",
        },
        "tile_type": "replay_governance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add P5 extensions if requested
    if include_p5_extensions:
        tile["production_run_id"] = production_run_id
        tile["replay_latency_ms"] = replay_latency_ms
        tile["telemetry_source"] = telemetry_source

    # Add director panel if present in base tile
    if "director_panel" in base_tile:
        tile["director_panel"] = base_tile["director_panel"]

    return tile


def attach_p5_replay_governance_to_evidence(
    evidence: Dict[str, Any],
    p5_signal: Dict[str, Any],
    p5_tile: Dict[str, Any],
    *,
    validate_p5_grade: bool = True,
) -> Dict[str, Any]:
    """Attach P5 replay governance to evidence pack.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment is advisory only, does NOT gate any operations
    - This is purely for observability and evidence collection

    Args:
        evidence: Existing evidence pack (not modified).
        p5_signal: P5 replay safety signal from extract_p5_replay_safety_from_logs().
        p5_tile: P5 replay governance tile from build_p5_replay_governance_tile().
        validate_p5_grade: If True, raise ValueError if signal is not P5-grade.

    Returns:
        New evidence dict with P5 replay governance attached under:
        - evidence["governance"]["replay"]: Collapsed governance signal
        - evidence["governance"]["replay_p5"]: P5-specific extension fields
        - evidence["replay_safety_ok"]: bool
        - evidence["replay_p5_grade"]: bool

    Raises:
        ValueError: If validate_p5_grade=True and signal is not P5-grade.

    P5 Grade Validation:
        - p5_signal["telemetry_source"] == "real"
        - p5_signal["production_run_id"] is present
        - p5_tile["phase"] == "P5"
    """
    # Validate P5 grade if requested
    is_p5_grade = (
        p5_signal.get("telemetry_source") == "real"
        and bool(p5_signal.get("production_run_id"))
        and p5_tile.get("phase") == "P5"
    )

    if validate_p5_grade and not is_p5_grade:
        raise ValueError(
            f"Signal is not P5-grade: telemetry_source={p5_signal.get('telemetry_source')}, "
            f"production_run_id={p5_signal.get('production_run_id')}, "
            f"phase={p5_tile.get('phase')}"
        )

    # Create shallow copy of evidence
    result = dict(evidence)

    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}
    else:
        result["governance"] = dict(result["governance"])

    # Build collapsed governance signal
    collapsed_signal = {
        "conflict": p5_tile.get("governance_signal", {}).get("conflict", False),
        "governance_alignment": p5_tile.get("governance_alignment", ALIGNMENT_ALIGNED),
        "governance_status": p5_tile.get("status", "ok"),
        "reasons": p5_signal.get("reasons", []),
        "safe_for_policy_update": p5_tile.get("safe", True),
        "safe_for_promotion": p5_tile.get("summary_metrics", {}).get("replay_ok_for_promotion", True),
        "signal_type": "replay_safety",
        "status": p5_tile.get("status", "ok"),
    }

    # Build P5 extension fields
    p5_extensions = {
        "determinism_band": p5_signal.get("determinism_band", "GREEN"),
        "determinism_rate": p5_signal.get("determinism_rate", 1.0),
        "hash_match_count": p5_signal.get("hash_match_count", 0),
        "hash_mismatch_count": p5_signal.get("hash_mismatch_count", 0),
        "production_run_id": p5_signal.get("production_run_id", ""),
        "replay_latency_ms": p5_signal.get("replay_latency_ms"),
        "telemetry_source": p5_signal.get("telemetry_source", "real"),
    }

    # Attach to governance
    result["governance"]["replay"] = collapsed_signal
    result["governance"]["replay_p5"] = p5_extensions

    # Add top-level evidence fields
    result["replay_safety_ok"] = p5_tile.get("status", "ok") == "ok"
    result["replay_p5_grade"] = is_p5_grade

    return result


# =============================================================================
# PHASE X P5: GGFL INTEGRATION PATCH
# =============================================================================

def replay_for_alignment_view_p5(
    p5_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize P5 replay signal into GGFL unified format.

    This extends replay_for_alignment_view() with P5-specific fields.

    READ-ONLY: Does not modify input signal.

    Args:
        p5_signal: P5 replay safety signal from extract_p5_replay_safety_from_logs()

    Returns:
        GGFL-normalized dict with P5 extensions:
        - status: Lowercase status (ok/warn/block)
        - alignment: Governance alignment (defaults to aligned for P5 signals)
        - conflict: Boolean conflict flag (derived from status)
        - top_reasons: List of reasons with [Replay] prefix stripped
        - p5_grade: bool
        - determinism_band: GREEN | YELLOW | RED
        - telemetry_source: str
    """
    # Extract status
    status = _normalize_status(p5_signal.get("status", "ok"))

    # P5 signals don't have separate safety/radar, derive alignment from status
    if status == "block":
        alignment = ALIGNMENT_DIVERGENT
        conflict = True
    elif status == "warn":
        alignment = ALIGNMENT_TENSION
        conflict = False
    else:
        alignment = ALIGNMENT_ALIGNED
        conflict = False

    # Process reasons: strip [Replay] prefix
    raw_reasons = p5_signal.get("reasons", [])
    top_reasons: List[str] = []

    for reason in raw_reasons:
        if isinstance(reason, str):
            stripped = reason
            if stripped.startswith("[Replay] "):
                stripped = stripped[9:]
            elif stripped.startswith("[Replay]"):
                stripped = stripped[8:]
            if stripped:
                top_reasons.append(stripped)

    # Limit to top 5
    top_reasons = top_reasons[:5]

    return {
        "alignment": alignment,
        "conflict": conflict,
        "determinism_band": p5_signal.get("determinism_band", "GREEN"),
        "p5_grade": p5_signal.get("p5_grade", False),
        "status": status,
        "telemetry_source": p5_signal.get("telemetry_source", "real"),
        "top_reasons": top_reasons,
    }


__all__ = [
    "REPLAY_GOVERNANCE_TILE_SCHEMA_VERSION",
    "STATUS_LIGHT_MAP",
    "ALIGNMENT_ALIGNED",
    "ALIGNMENT_TENSION",
    "ALIGNMENT_DIVERGENT",
    "build_replay_console_tile",
    "replay_to_governance_signal",
    "attach_replay_governance_to_evidence",
    "build_replay_governance_tile_for_global_health",
    # Phase X P5: First-Light Binding
    "attach_replay_governance_to_p3_stability_report",
    "attach_replay_governance_to_p4_calibration_report",
    # Phase X P5: GGFL Fusion
    "replay_for_alignment_view",
    # Phase X P5: Real-Telemetry Functions
    "P5_REPLAY_SCHEMA_VERSION",
    "P5_DETERMINISM_GREEN_THRESHOLD",
    "P5_DETERMINISM_YELLOW_THRESHOLD",
    "extract_p5_replay_safety_from_logs",
    "build_p5_replay_governance_tile",
    "attach_p5_replay_governance_to_evidence",
    "replay_for_alignment_view_p5",
    # Canonicalization v1.3.0: Extraction Source
    "EXTRACTION_SOURCE_MANIFEST",
    "EXTRACTION_SOURCE_EVIDENCE_JSON",
    "EXTRACTION_SOURCE_DIRECT_LOG",
    "EXTRACTION_SOURCE_MISSING",
    # Canonicalization v1.3.0: Driver Codes
    "DRIVER_SCHEMA_NOT_OK",
    "DRIVER_SAFETY_MISMATCH_PRESENT",
    "DRIVER_STATE_MISMATCH_PRESENT",
    "DRIVER_DETERMINISM_RED_BAND",
    "DRIVER_PRIORITY_ORDER",
    "DRIVER_REASON_CAP",
    "compute_driver_codes",
]
