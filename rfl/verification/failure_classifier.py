"""
Unified Failure-State Classifier for RFL Verification Layer
============================================================

Provides deterministic classification of execution failures into
canonical failure states for transparent logging and metrics aggregation.

SAFETY GUARANTEES:
- No alteration of success metrics
- Deterministic mapping: same input → same classification
- Transparent logging with source evidence
- No interpretation of uplift or statistical inference

INTEGRATION WITH ABSTENTION TAXONOMY (Phase II Harmonization):
This module's FailureState enum is complementary to AbstentionType from
abstention_taxonomy.py. The key distinction:
- FailureState: Classifies *execution* failures (crashes, timeouts, budget)
- AbstentionType: Classifies *verification* abstentions (lean-disabled, etc.)

Both taxonomies can coexist and are bridged via mapping functions.

PHASE II — VERIFICATION BUREAU
Agent B4 (verifier-ops-4), Agent B6 (verifier-ops-6)
"""

from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging
import subprocess

# Import canonical abstention taxonomy for bridging
from rfl.verification.abstention_taxonomy import (
    AbstentionType,
    classify_breakdown_key as taxonomy_classify_breakdown,
)

if TYPE_CHECKING:
    from rfl.experiment import ExperimentResult

logger = logging.getLogger("rfl.verification.failure_classifier")


class FailureState(Enum):
    """
    Canonical failure state taxonomy for RFL verification.
    
    These states provide a unified language for classifying execution failures
    across the RFL runtime, enabling deterministic logging and metrics aggregation.
    
    INVARIANT: SUCCESS is a reference state only. Exception classification
    functions will never return SUCCESS - that determination is made by
    higher-level code based on actual execution outcomes.
    """
    SUCCESS = "success"                      # Reference state (not a failure)
    TIMEOUT_ABSTAIN = "timeout_abstain"      # Verification/execution timeout
    BUDGET_EXHAUSTED = "budget_exhausted"    # Cycle/step budget exceeded
    CRASH_ABSTAIN = "crash_abstain"          # Executor crash, non-zero exit
    INVALID_FORMULA = "invalid_formula"      # Parse/normalize failure
    SKIPPED_BY_BUDGET = "skipped_by_budget"  # Not attempted due to budget
    UNKNOWN_ERROR = "unknown_error"          # Catch-all for unclassified errors


# Legacy key mapping for backward compatibility
# Maps old ad-hoc string labels to canonical FailureState values
LEGACY_KEY_MAP: Dict[str, str] = {
    # Direct mappings from rfl/experiment.py
    "engine_failure": FailureState.CRASH_ABSTAIN.value,
    "timeout": FailureState.TIMEOUT_ABSTAIN.value,
    "unexpected_error": FailureState.UNKNOWN_ERROR.value,
    
    # Mappings from rfl/runner.py abstention histogram
    "pending_validation": FailureState.UNKNOWN_ERROR.value,  # Ambiguous legacy state
    "empty_run": FailureState.UNKNOWN_ERROR.value,
    "no_successful_proofs": FailureState.UNKNOWN_ERROR.value,
    "zero_throughput": FailureState.UNKNOWN_ERROR.value,
    
    # Mappings from attestation/lean verification
    "lean_failure": FailureState.CRASH_ABSTAIN.value,
    "derivation_abstain": FailureState.UNKNOWN_ERROR.value,
    
    # Budget-related (from lean_control_sandbox_plan.md)
    "budget_skip": FailureState.SKIPPED_BY_BUDGET.value,
    "cycle_budget_exhausted": FailureState.BUDGET_EXHAUSTED.value,
    
    # Pass-through for already-canonical keys
    FailureState.SUCCESS.value: FailureState.SUCCESS.value,
    FailureState.TIMEOUT_ABSTAIN.value: FailureState.TIMEOUT_ABSTAIN.value,
    FailureState.BUDGET_EXHAUSTED.value: FailureState.BUDGET_EXHAUSTED.value,
    FailureState.CRASH_ABSTAIN.value: FailureState.CRASH_ABSTAIN.value,
    FailureState.INVALID_FORMULA.value: FailureState.INVALID_FORMULA.value,
    FailureState.SKIPPED_BY_BUDGET.value: FailureState.SKIPPED_BY_BUDGET.value,
    FailureState.UNKNOWN_ERROR.value: FailureState.UNKNOWN_ERROR.value,
}


def normalize_legacy_key(key: str) -> str:
    """
    Map legacy abstention breakdown keys to canonical FailureState values.
    
    This function provides backward compatibility when reading old logs or
    merging abstention breakdowns that use ad-hoc string labels.
    
    Args:
        key: The legacy or canonical key to normalize
        
    Returns:
        Canonical FailureState.value string
        
    Example:
        >>> normalize_legacy_key("engine_failure")
        'crash_abstain'
        >>> normalize_legacy_key("timeout")
        'timeout_abstain'
        >>> normalize_legacy_key("crash_abstain")  # Already canonical
        'crash_abstain'
    """
    normalized = LEGACY_KEY_MAP.get(key)
    if normalized is not None:
        if normalized != key:
            logger.debug(f"Normalized legacy key '{key}' → '{normalized}'")
        return normalized
    
    # Unknown key - log and return as-is to avoid data loss
    logger.warning(f"Unknown abstention key '{key}' - passing through unchanged")
    return key


def classify_exception(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None
) -> FailureState:
    """
    Classify an exception into a canonical failure state.
    
    This function provides deterministic mapping from Python exceptions
    to FailureState enum values. It examines both the exception type
    and optional context signals to determine the appropriate classification.
    
    Args:
        exc: The exception to classify
        context: Optional context dict with additional signals:
            - budget_exceeded: bool - Cycle budget was exceeded
            - budget_exhausted: bool - Budget fully consumed
            - skipped_by_budget: bool - Candidate skipped due to budget
            - returncode: int - Subprocess return code
            
    Returns:
        FailureState enum value (never SUCCESS)
        
    INVARIANT: This function never returns FailureState.SUCCESS.
    Success determination is the responsibility of higher-level code.
    """
    context = context or {}
    
    # --- Timeout signals ---
    if isinstance(exc, subprocess.TimeoutExpired):
        logger.debug(f"Classified as TIMEOUT_ABSTAIN: {type(exc).__name__}")
        return FailureState.TIMEOUT_ABSTAIN
    
    # Check for timeout in context (e.g., from async operations)
    if context.get("timeout") or context.get("timed_out"):
        logger.debug("Classified as TIMEOUT_ABSTAIN from context flag")
        return FailureState.TIMEOUT_ABSTAIN
    
    # --- Memory exhaustion ---
    if isinstance(exc, MemoryError):
        logger.debug(f"Classified as CRASH_ABSTAIN (memory exhaustion): {type(exc).__name__}")
        return FailureState.CRASH_ABSTAIN
    
    # --- Subprocess crash ---
    if isinstance(exc, subprocess.CalledProcessError):
        logger.debug(f"Classified as CRASH_ABSTAIN (subprocess): returncode={exc.returncode}")
        return FailureState.CRASH_ABSTAIN
    
    # Check for non-zero returncode in context
    returncode = context.get("returncode")
    if returncode is not None and returncode != 0:
        logger.debug(f"Classified as CRASH_ABSTAIN from context: returncode={returncode}")
        return FailureState.CRASH_ABSTAIN
    
    # --- Parse/syntax errors ---
    if isinstance(exc, SyntaxError):
        logger.debug(f"Classified as INVALID_FORMULA: {type(exc).__name__}")
        return FailureState.INVALID_FORMULA
    
    # ValueError with parsing-related messages
    if isinstance(exc, ValueError):
        msg = str(exc).lower()
        parse_keywords = ("syntax", "parse", "invalid", "normalize", "malformed", "formula")
        if any(kw in msg for kw in parse_keywords):
            logger.debug(f"Classified as INVALID_FORMULA: ValueError with parse-related message")
            return FailureState.INVALID_FORMULA
    
    # Check for invalid formula flag in context
    if context.get("invalid_formula") or context.get("parse_error"):
        logger.debug("Classified as INVALID_FORMULA from context flag")
        return FailureState.INVALID_FORMULA
    
    # --- Budget signals ---
    # SKIPPED_BY_BUDGET takes precedence (candidate not attempted)
    if context.get("skipped_by_budget") or context.get("budget_skip"):
        logger.debug("Classified as SKIPPED_BY_BUDGET from context flag")
        return FailureState.SKIPPED_BY_BUDGET
    
    # BUDGET_EXHAUSTED (cycle budget fully consumed)
    if context.get("budget_exceeded") or context.get("budget_exhausted"):
        logger.debug("Classified as BUDGET_EXHAUSTED from context flag")
        return FailureState.BUDGET_EXHAUSTED
    
    # --- Fallback ---
    logger.debug(f"Classified as UNKNOWN_ERROR: {type(exc).__name__}: {exc}")
    return FailureState.UNKNOWN_ERROR


def classify_from_result(result: "ExperimentResult") -> FailureState:
    """
    Classify an ExperimentResult into a canonical failure state.
    
    This function examines the status and error_message fields of an
    ExperimentResult to determine the appropriate FailureState.
    
    Args:
        result: The ExperimentResult to classify
        
    Returns:
        FailureState enum value
        
    Note:
        Returns FailureState.SUCCESS for results with status="success".
        This is the only classification function that may return SUCCESS.
    """
    # Success case
    if result.status == "success":
        return FailureState.SUCCESS
    
    # Use status and error_message for classification
    return classify_from_status(result.status, result.error_message)


def classify_from_status(
    status: str,
    error_msg: Optional[str] = None
) -> FailureState:
    """
    Classify a status string and optional error message into a failure state.
    
    This function provides classification based on status codes and error
    message content, useful when the original exception is not available.
    
    Args:
        status: The status string ("success", "failed", "aborted", etc.)
        error_msg: Optional error message for additional context
        
    Returns:
        FailureState enum value
    """
    # Success case
    if status == "success":
        return FailureState.SUCCESS
    
    # Aborted typically means timeout
    if status == "aborted":
        logger.debug(f"Classified as TIMEOUT_ABSTAIN: status='aborted'")
        return FailureState.TIMEOUT_ABSTAIN
    
    # Failed status - examine error message for more specific classification
    if status == "failed" and error_msg:
        msg_lower = error_msg.lower()
        
        # Timeout indicators in message
        if "timeout" in msg_lower or "timed out" in msg_lower:
            logger.debug(f"Classified as TIMEOUT_ABSTAIN: 'timeout' in error message")
            return FailureState.TIMEOUT_ABSTAIN
        
        # Crash/failure indicators
        if any(kw in msg_lower for kw in ("crash", "memory", "killed", "segfault", "returncode")):
            logger.debug(f"Classified as CRASH_ABSTAIN: crash-related keyword in error message")
            return FailureState.CRASH_ABSTAIN
        
        # CLI/engine failure (legacy)
        if "derive cli failed" in msg_lower or "engine" in msg_lower:
            logger.debug(f"Classified as CRASH_ABSTAIN: CLI/engine failure in error message")
            return FailureState.CRASH_ABSTAIN
        
        # Parse/syntax errors
        if any(kw in msg_lower for kw in ("syntax", "parse", "invalid", "normalize", "formula")):
            logger.debug(f"Classified as INVALID_FORMULA: parse-related keyword in error message")
            return FailureState.INVALID_FORMULA
        
        # Budget-related
        if "budget" in msg_lower:
            if "skip" in msg_lower:
                logger.debug(f"Classified as SKIPPED_BY_BUDGET: budget skip in error message")
                return FailureState.SKIPPED_BY_BUDGET
            logger.debug(f"Classified as BUDGET_EXHAUSTED: budget in error message")
            return FailureState.BUDGET_EXHAUSTED
    
    # Default for failed status without specific indicators
    if status == "failed":
        logger.debug(f"Classified as UNKNOWN_ERROR: status='failed' without specific indicators")
        return FailureState.UNKNOWN_ERROR
    
    # Catch-all for unrecognized status values
    logger.debug(f"Classified as UNKNOWN_ERROR: unrecognized status='{status}'")
    return FailureState.UNKNOWN_ERROR


def classify_and_log(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
    source: str = "unknown"
) -> FailureState:
    """
    Classify an exception and log the classification with source context.
    
    This is a convenience wrapper that combines classification with
    structured logging for audit purposes.
    
    Args:
        exc: The exception to classify
        context: Optional context dict with additional signals
        source: Description of where the failure occurred (for logging)
        
    Returns:
        FailureState enum value
    """
    state = classify_exception(exc, context)
    logger.info(
        f"[{source}] Classified failure: {type(exc).__name__} → {state.value}",
        extra={
            "failure_state": state.value,
            "exception_type": type(exc).__name__,
            "source": source,
            "context": context or {},
        }
    )
    return state


# ---------------------------------------------------------------------------
# Bridging with AbstentionType Taxonomy (Phase II Harmonization)
# ---------------------------------------------------------------------------

# Mapping from FailureState to AbstentionType
# This allows unified analysis across both taxonomies
FAILURE_TO_ABSTENTION_MAP: Dict[FailureState, AbstentionType] = {
    FailureState.TIMEOUT_ABSTAIN: AbstentionType.ABSTAIN_TIMEOUT,
    FailureState.BUDGET_EXHAUSTED: AbstentionType.ABSTAIN_BUDGET,
    FailureState.CRASH_ABSTAIN: AbstentionType.ABSTAIN_CRASH,
    FailureState.INVALID_FORMULA: AbstentionType.ABSTAIN_INVALID,
    FailureState.SKIPPED_BY_BUDGET: AbstentionType.ABSTAIN_BUDGET,
    FailureState.UNKNOWN_ERROR: AbstentionType.ABSTAIN_CRASH,  # Defensive mapping
    # SUCCESS is not an abstention
}


def failure_to_abstention(state: FailureState) -> Optional[AbstentionType]:
    """
    Bridge FailureState to canonical AbstentionType.
    
    This function allows unified reporting across both taxonomies.
    
    Args:
        state: The FailureState to convert
        
    Returns:
        Corresponding AbstentionType, or None if state is SUCCESS
        
    Examples:
        >>> failure_to_abstention(FailureState.TIMEOUT_ABSTAIN)
        AbstentionType.ABSTAIN_TIMEOUT
        
        >>> failure_to_abstention(FailureState.SUCCESS)
        None
    """
    if state == FailureState.SUCCESS:
        return None
    return FAILURE_TO_ABSTENTION_MAP.get(state, AbstentionType.ABSTAIN_CRASH)


def classify_breakdown_key_unified(key: str) -> Optional[AbstentionType]:
    """
    Classify a breakdown key using both legacy FailureState mapping and
    canonical AbstentionType taxonomy.
    
    This function provides a unified classification path that:
    1. First tries the canonical AbstentionType taxonomy
    2. Falls back to legacy FailureState mapping if needed
    3. Converts legacy mappings to AbstentionType for consistency
    
    Args:
        key: The breakdown key to classify
        
    Returns:
        Corresponding AbstentionType, or None if unrecognized
        
    Examples:
        >>> classify_breakdown_key_unified("engine_failure")
        AbstentionType.ABSTAIN_CRASH
        
        >>> classify_breakdown_key_unified("lean-timeout")
        AbstentionType.ABSTAIN_LEAN_TIMEOUT
    """
    # First try canonical taxonomy
    abst_type = taxonomy_classify_breakdown(key)
    if abst_type is not None:
        return abst_type
    
    # Fall back to legacy mapping
    normalized = LEGACY_KEY_MAP.get(key)
    if normalized is not None:
        # Convert legacy FailureState value to AbstentionType
        for fs, at in FAILURE_TO_ABSTENTION_MAP.items():
            if fs.value == normalized:
                return at
    
    # Unrecognized key
    return None

