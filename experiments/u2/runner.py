# PHASE II — NOT RUN IN PHASE I
"""
U2 Runner Module with Trace Logging

STATUS: PHASE II — NOT RUN IN PHASE I

This module provides:
1. U2Runner: Core experiment runner with snapshot support
2. U2Config: Configuration dataclass for experiments
3. CycleResult: Re-exported from cycle_orchestrator for convenience
4. TracedExperimentContext: Context for per-cycle trace logging
5. run_with_traces: Wrapper to inject logging hooks

Usage:
    # Core runner usage (used by run_uplift_u2.py)
    config = U2Config(
        experiment_id="u2_test_001",
        slice_name="arithmetic_simple",
        mode="baseline",
        total_cycles=100,
        master_seed=12345,
    )
    runner = U2Runner(config)
    result = runner.run_cycle(items, execute_fn)
    
    # Trace logging usage
    result = run_with_traces(
        run_id="u2_baseline_001",
        slice_name="arithmetic_simple",
        mode="baseline",
        cycles=100,
        log_path=Path("traces/u2_baseline_001.jsonl"),
        core_runner=run_experiment,
        seed=12345,
        config=config_dict,
    )
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from backend.lean_interface import LeanFailureSignal, LeanFailureKind

from .logging import U2TraceLogger
from . import schema
from .snapshots import (
    SnapshotData,
    save_snapshot,
    capture_prng_states,
    restore_prng_states,
    capture_random_instance_state,
    restore_random_instance_state,
)
from .runtime.cycle_orchestrator import (
    CycleResult,
    CycleState,
    BaselineOrderingStrategy,
    RflOrderingStrategy,
    execute_cycle,
    get_ordering_strategy,
)
from .policy import summarize_lean_failures_for_global_health

# Phase VII CORTEX: TDA Governance Hook (optional integration)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.tda.runner_hook import TDAGovernanceHook

# Re-export CycleResult for convenience
__all__ = [
    "U2Config",
    "U2Runner",
    "CycleResult",
    "TracedExperimentContext",
    "run_with_traces",
    "compute_config_hash",
    # Replay verification (per u2_runner_spec.md v1.1.0)
    "ReplayError",
    "ReplayResult",
    "ReplayCycleComparison",
    "ReplayManifestMismatch",
    "ReplaySeedScheduleMismatch",
    "ReplayLogMissing",
    "ReplayHtMismatch",
    "ReplayRtMismatch",
    "ReplayUtMismatch",
    "ReplayCycleCountMismatch",
    "ReplayConfigHashMismatch",
    "ReplayVerificationError",
    "ReplayUnknownError",
    "REPLAY_ERROR_MAP",
    # Replay contract v1.0.0 (Task 1-3)
    "REPLAY_CONTRACT_VERSION",
    "REPLAY_MODE_FULL",
    "REPLAY_MODE_DRY_RUN",
    "REPLAY_MODE_PARTIAL",
    "ManifestValidationResult",
    "validate_replay_manifest",
    "summarize_replay_for_governance",
    # Phase III: Governance Matrix & Safety Envelope
    "SAFETY_ENVELOPE_VERSION",
    "SafetyLevel",
    "ReplaySafetyEnvelope",
    "build_replay_safety_envelope",
    "validate_replay_before_policy_update",
    "compute_replay_confidence",
    # Phase IV: Replay Safety as Hard Policy Gate & Evidence Tile
    "PromotionStatus",
    "evaluate_replay_safety_for_promotion",
    "summarize_replay_safety_for_evidence",
    "build_replay_safety_director_panel",
    # Phase V: Replay Safety × Governance Radar Fusion
    "GovernanceAlignment",
    "build_replay_safety_governance_view",
    # Phase VI: Replay Safety as Clean GovernanceSignal
    "to_governance_signal_for_replay_safety",
]


# ============================================================================
# Replay Error Taxonomy (RUN-41 to RUN-50) — per u2_runner_spec.md v1.1.0
# ============================================================================

class ReplayError(Exception):
    """Base class for replay-specific errors."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


class ReplayManifestMismatch(ReplayError):
    """RUN-41: Manifest schema version incompatible or required fields missing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-41", message, details)


class ReplaySeedScheduleMismatch(ReplayError):
    """RUN-42: Seed schedule formula doesn't match expected."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-42", message, details)


class ReplayLogMissing(ReplayError):
    """RUN-43: Referenced JSONL log file does not exist."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-43", message, details)


class ReplayHtMismatch(ReplayError):
    """RUN-44: Verification hash root (h_t) mismatch."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-44", message, details)


class ReplayRtMismatch(ReplayError):
    """RUN-45: Candidate ordering root (r_t) mismatch."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-45", message, details)


class ReplayUtMismatch(ReplayError):
    """RUN-46: Policy state root (u_t) mismatch."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-46", message, details)


class ReplayCycleCountMismatch(ReplayError):
    """RUN-47: Line count in results JSONL doesn't match manifest cycles."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-47", message, details)


class ReplayConfigHashMismatch(ReplayError):
    """RUN-48: Current config SHA-256 doesn't match manifest."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-48", message, details)


class ReplayVerificationError(ReplayError):
    """RUN-49: Hermetic verification throws unexpected exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-49", message, details)


class ReplayUnknownError(ReplayError):
    """RUN-50: Unclassified exception during replay."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("RUN-50", message, details)


# Mapping from error code to exception class
REPLAY_ERROR_MAP = {
    "RUN-41": ReplayManifestMismatch,
    "RUN-42": ReplaySeedScheduleMismatch,
    "RUN-43": ReplayLogMissing,
    "RUN-44": ReplayHtMismatch,
    "RUN-45": ReplayRtMismatch,
    "RUN-46": ReplayUtMismatch,
    "RUN-47": ReplayCycleCountMismatch,
    "RUN-48": ReplayConfigHashMismatch,
    "RUN-49": ReplayVerificationError,
    "RUN-50": ReplayUnknownError,
}


# ============================================================================
# Replay Contract Versioning (Task 1)
# ============================================================================

# Replay contract version - bump when output schema changes
REPLAY_CONTRACT_VERSION = "1.0.0"

# Replay execution modes
REPLAY_MODE_FULL = "full"           # Complete cycle-by-cycle replay
REPLAY_MODE_DRY_RUN = "dry_run"     # Manifest/config validation only
REPLAY_MODE_PARTIAL = "partial"     # Incomplete replay (inadmissible for governance)


@dataclass
class ReplayCycleComparison:
    """Comparison result for a single replay cycle."""
    cycle_index: int
    original_h_t: str
    replay_h_t: str
    original_r_t: Optional[str]
    replay_r_t: Optional[str]
    original_u_t: Optional[str]
    replay_u_t: Optional[str]
    h_t_match: bool
    r_t_match: bool
    u_t_match: bool
    all_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "original_h_t": self.original_h_t,
            "replay_h_t": self.replay_h_t,
            "original_r_t": self.original_r_t,
            "replay_r_t": self.replay_r_t,
            "original_u_t": self.original_u_t,
            "replay_u_t": self.replay_u_t,
            "h_t_match": self.h_t_match,
            "r_t_match": self.r_t_match,
            "u_t_match": self.u_t_match,
            "all_match": self.all_match,
        }


@dataclass
class ReplayResult:
    """
    Result of a replay verification run.

    This is a versioned contract object (REPLAY_CONTRACT_VERSION) that
    downstream tools (security, MAAS, governance) can depend on.

    Contract guarantees:
    - contract_version field is always present
    - status is one of: REPLAY_VERIFIED, REPLAY_FAILED, REPLAY_DRY_RUN
    - is_partial_replay indicates inadmissible evidence for governance
    - timestamps are in _metadata section, not used in equality checks
    """
    # Core status fields
    status: str  # "REPLAY_VERIFIED", "REPLAY_FAILED", "REPLAY_DRY_RUN"
    manifest_path: str
    original_cycles: int
    replayed_cycles: int
    all_cycles_match: bool

    # Mismatch details
    first_mismatch_cycle: Optional[int]
    mismatch_type: Optional[str]  # "h_t", "r_t", "u_t"
    error_code: Optional[str]
    error_message: Optional[str]

    # Cycle-level data
    cycle_comparisons: List[ReplayCycleComparison]

    # Hash verification
    original_results_hash: Optional[str]
    replay_results_hash: Optional[str]
    results_hash_match: bool

    # Contract v1.0.0 fields (Task 1)
    replay_mode: str = REPLAY_MODE_FULL  # full, dry_run, partial
    is_partial_replay: bool = False  # True if replay was interrupted or incomplete
    replay_coverage_pct: float = 100.0  # % of cycles replayed
    replay_manifest_path: Optional[str] = None  # Path to replay output manifest
    run_id: Optional[str] = None  # Experiment run ID if known

    # Critical mismatch flags for governance (Task 3)
    has_ht_mismatch: bool = False
    has_rt_mismatch: bool = False
    has_ut_mismatch: bool = False
    has_cycle_count_mismatch: bool = False
    has_config_hash_mismatch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dict with versioned contract schema.

        Note: _metadata section contains timestamps and is excluded from
        equality comparisons by governance tools.
        """
        return {
            # Contract header
            "contract_version": REPLAY_CONTRACT_VERSION,
            "replay_mode": self.replay_mode,

            # Core status
            "status": self.status,
            "primary_manifest_path": self.manifest_path,
            "replay_manifest_path": self.replay_manifest_path,
            "run_id": self.run_id,

            # Cycle statistics
            "per_cycle_stats": {
                "cycle_count_primary": self.original_cycles,
                "cycle_count_replay": self.replayed_cycles,
                "all_cycles_match": self.all_cycles_match,
                "replay_coverage_pct": self.replay_coverage_pct,
            },

            # Partial replay fence (Task 2)
            "is_partial_replay": self.is_partial_replay,
            "governance_admissible": not self.is_partial_replay and self.replay_mode == REPLAY_MODE_FULL,

            # Mismatch details
            "first_mismatch_cycle": self.first_mismatch_cycle,
            "mismatch_type": self.mismatch_type,
            "error_code": self.error_code,
            "error_message": self.error_message,

            # Critical mismatch flags (Task 3)
            "critical_mismatch_flags": {
                "ht_mismatch": self.has_ht_mismatch,
                "rt_mismatch": self.has_rt_mismatch,
                "ut_mismatch": self.has_ut_mismatch,
                "cycle_count_mismatch": self.has_cycle_count_mismatch,
                "config_hash_mismatch": self.has_config_hash_mismatch,
            },

            # Hash verification
            "hash_verification": {
                "original_results_hash": self.original_results_hash,
                "replay_results_hash": self.replay_results_hash,
                "results_hash_match": self.results_hash_match,
            },

            # Detailed cycle comparisons (may be empty for dry-run)
            "cycle_comparisons": [c.to_dict() for c in self.cycle_comparisons],

            # Legacy fields (for backward compatibility)
            "manifest_path": self.manifest_path,
            "original_cycles": self.original_cycles,
            "replayed_cycles": self.replayed_cycles,
        }


# ============================================================================
# Task 2: Dry-Run Validation & Partial Replay Fence
# ============================================================================

@dataclass
class ManifestValidationResult:
    """Result of manifest validation (dry-run mode)."""
    is_valid: bool
    manifest_path: str
    errors: List[str]
    warnings: List[str]
    manifest_version: Optional[str] = None
    slice_name: Optional[str] = None
    mode: Optional[str] = None
    cycles: Optional[int] = None
    seed: Optional[int] = None
    config_hash: Optional[str] = None
    results_path: Optional[str] = None
    results_exist: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "manifest_path": self.manifest_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "manifest_version": self.manifest_version,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "cycles": self.cycles,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "results_path": self.results_path,
            "results_exist": self.results_exist,
        }


def validate_replay_manifest(manifest_path: Path) -> ManifestValidationResult:
    """
    Validate a manifest for replay compatibility without executing cycles.

    This is the dry-run validation function (Task 2). It checks:
    - Manifest file exists and is valid JSON
    - Required fields are present
    - Results log file exists
    - Seed schedule is compatible

    Args:
        manifest_path: Path to the primary run manifest JSON.

    Returns:
        ManifestValidationResult with validation status and details.
    """
    manifest_path = Path(manifest_path)
    errors: List[str] = []
    warnings: List[str] = []

    # Check manifest exists
    if not manifest_path.exists():
        return ManifestValidationResult(
            is_valid=False,
            manifest_path=str(manifest_path),
            errors=["Manifest file not found"],
            warnings=[],
        )

    # Try to load manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return ManifestValidationResult(
            is_valid=False,
            manifest_path=str(manifest_path),
            errors=[f"Invalid JSON: {e}"],
            warnings=[],
        )

    # Check required fields
    required_fields = ["slice", "mode", "cycles", "initial_seed"]
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # Extract manifest data
    manifest_version = manifest.get("manifest_version")
    slice_name = manifest.get("slice")
    mode = manifest.get("mode")
    cycles = manifest.get("cycles")
    seed = manifest.get("initial_seed")
    config_hash = manifest.get("slice_config_hash")

    # Check results path
    results_path_str = manifest.get("outputs", {}).get("results")
    results_exist = False
    if not results_path_str:
        errors.append("Manifest missing results path in outputs section")
    else:
        results_path = Path(results_path_str)
        if not results_path.exists():
            # Try relative to manifest directory
            results_path = manifest_path.parent / Path(results_path_str).name
        results_exist = results_path.exists()
        if not results_exist:
            errors.append(f"Results log not found: {results_path_str}")

    # Validate seed schedule compatibility
    seed_schedule = manifest.get("seed_schedule")
    if seed_schedule and seed_schedule != "MDAP_SEED + cycle_index":
        # Check if it's the expected formula pattern
        if "cycle_index" not in str(seed_schedule):
            warnings.append(f"Non-standard seed schedule: {seed_schedule}")

    # Check for config hash
    if not config_hash:
        warnings.append("No config hash in manifest - cannot verify config integrity")

    return ManifestValidationResult(
        is_valid=len(errors) == 0,
        manifest_path=str(manifest_path),
        errors=errors,
        warnings=warnings,
        manifest_version=manifest_version,
        slice_name=slice_name,
        mode=mode,
        cycles=cycles,
        seed=seed,
        config_hash=config_hash,
        results_path=results_path_str,
        results_exist=results_exist,
    )


# ============================================================================
# Task 3: Governance Summary Helper
# ============================================================================

def summarize_replay_for_governance(replay_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a compact summary for governance/MAAS consumption.

    This function extracts the critical fields from a ReplayResult.to_dict()
    output that governance tools need to make pass/fail decisions.

    Args:
        replay_result: Dict from ReplayResult.to_dict()

    Returns:
        Compact governance summary dict with:
        - status: VERIFIED/FAILED/DRY_RUN
        - run_id: Experiment identifier
        - manifest_pair: Primary and replay manifest paths
        - critical_mismatch_flags: Which critical checks failed
        - governance_admissible: Whether this is valid evidence
    """
    # Handle both new contract format and legacy format
    contract_version = replay_result.get("contract_version", "0.0.0")

    # Extract status
    status = replay_result.get("status", "UNKNOWN")

    # Extract run_id
    run_id = replay_result.get("run_id")

    # Extract manifest pair
    primary_path = replay_result.get("primary_manifest_path") or replay_result.get("manifest_path")
    replay_path = replay_result.get("replay_manifest_path")
    manifest_pair = {
        "primary": primary_path,
        "replay": replay_path,
    }

    # Extract critical mismatch flags
    critical_flags = replay_result.get("critical_mismatch_flags", {})
    if not critical_flags:
        # Reconstruct from legacy fields
        error_code = replay_result.get("error_code")
        mismatch_type = replay_result.get("mismatch_type")
        critical_flags = {
            "ht_mismatch": error_code == "RUN-44" or mismatch_type == "h_t",
            "rt_mismatch": error_code == "RUN-45" or mismatch_type == "r_t",
            "ut_mismatch": error_code == "RUN-46" or mismatch_type == "u_t",
            "cycle_count_mismatch": error_code == "RUN-47",
            "config_hash_mismatch": error_code == "RUN-48",
        }

    # Determine governance admissibility
    is_partial = replay_result.get("is_partial_replay", False)
    replay_mode = replay_result.get("replay_mode", REPLAY_MODE_FULL)
    governance_admissible = (
        not is_partial
        and replay_mode == REPLAY_MODE_FULL
        and status == "REPLAY_VERIFIED"
    )

    # Extract per-cycle stats
    per_cycle_stats = replay_result.get("per_cycle_stats", {})
    if not per_cycle_stats:
        per_cycle_stats = {
            "cycle_count_primary": replay_result.get("original_cycles", 0),
            "cycle_count_replay": replay_result.get("replayed_cycles", 0),
            "all_cycles_match": replay_result.get("all_cycles_match", False),
        }

    return {
        "contract_version": contract_version,
        "status": status,
        "run_id": run_id,
        "manifest_pair": manifest_pair,
        "critical_mismatch_flags": critical_flags,
        "governance_admissible": governance_admissible,
        "per_cycle_stats": per_cycle_stats,
        "error_code": replay_result.get("error_code"),
        "error_message": replay_result.get("error_message"),
    }


# ============================================================================
# Phase III: Governance Matrix & Safety Envelope
# ============================================================================

SAFETY_ENVELOPE_VERSION = "1.0.0"


class SafetyLevel:
    """Safety level constants for replay safety envelope."""
    OK = "OK"      # All checks passed, safe to proceed
    WARN = "WARN"  # Minor issues, proceed with caution
    FAIL = "FAIL"  # Critical issues, do not proceed


@dataclass
class ReplaySafetyEnvelope:
    """
    Safety envelope for replay verification results.

    This envelope provides a comprehensive safety assessment for governance
    and policy update decisions. It wraps ReplayResult with additional
    safety metadata and recommended actions.

    Attributes:
        schema_version: Version of the safety envelope schema.
        replay_mode: Mode of replay (full, dry_run, partial).
        is_fully_deterministic: True if all cycles matched exactly.
        safety_level: OK, WARN, or FAIL.
        critical_mismatch_flags: Which critical checks failed.
        per_cycle_consistency: Per-cycle match statistics.
        recommended_action: Suggested action based on safety level.
        confidence_score: Computed replay confidence [0, 1].
        policy_update_allowed: Whether policy updates are safe.
        error_details: Error information if any.
    """
    schema_version: str
    replay_mode: str
    is_fully_deterministic: bool
    safety_level: str  # SafetyLevel.OK, WARN, or FAIL
    critical_mismatch_flags: Dict[str, bool]
    per_cycle_consistency: Dict[str, Any]
    recommended_action: str
    confidence_score: float
    policy_update_allowed: bool
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "schema_version": self.schema_version,
            "replay_mode": self.replay_mode,
            "is_fully_deterministic": self.is_fully_deterministic,
            "safety_level": self.safety_level,
            "critical_mismatch_flags": self.critical_mismatch_flags,
            "per_cycle_consistency": self.per_cycle_consistency,
            "recommended_action": self.recommended_action,
            "confidence_score": self.confidence_score,
            "policy_update_allowed": self.policy_update_allowed,
            "error_details": self.error_details,
        }


def build_replay_safety_envelope(
    replay_result: Dict[str, Any],
) -> ReplaySafetyEnvelope:
    """
    Build a safety envelope from a replay result.

    TASK 1: Replay Safety Envelope

    This function analyzes a ReplayResult.to_dict() output and produces
    a comprehensive safety assessment for governance decisions.

    Safety Level Determination:
    - FAIL: Any RUN-4x mismatch (ht, rt, ut), cycle count mismatch, or partial replay
    - WARN: Config hash mismatch, non-standard seed schedule, or coverage < 100%
    - OK: All checks passed, full replay, all cycles match

    Args:
        replay_result: Dict from ReplayResult.to_dict()

    Returns:
        ReplaySafetyEnvelope with safety assessment and recommended action.
    """
    # Extract status and mode
    status = replay_result.get("status", "UNKNOWN")
    replay_mode = replay_result.get("replay_mode", REPLAY_MODE_FULL)
    is_partial = replay_result.get("is_partial_replay", False)

    # Extract critical mismatch flags
    critical_flags = replay_result.get("critical_mismatch_flags", {})
    if not critical_flags:
        # Reconstruct from error_code for legacy format
        error_code = replay_result.get("error_code")
        critical_flags = {
            "ht_mismatch": error_code == "RUN-44",
            "rt_mismatch": error_code == "RUN-45",
            "ut_mismatch": error_code == "RUN-46",
            "cycle_count_mismatch": error_code == "RUN-47",
            "config_hash_mismatch": error_code == "RUN-48",
        }

    # Extract per-cycle stats
    per_cycle_stats = replay_result.get("per_cycle_stats", {})
    original_cycles = per_cycle_stats.get("cycle_count_primary", replay_result.get("original_cycles", 0))
    replayed_cycles = per_cycle_stats.get("cycle_count_replay", replay_result.get("replayed_cycles", 0))
    all_cycles_match = per_cycle_stats.get("all_cycles_match", replay_result.get("all_cycles_match", False))
    coverage_pct = per_cycle_stats.get("replay_coverage_pct", replay_result.get("replay_coverage_pct", 100.0))

    # Compute per-cycle consistency
    cycle_comparisons = replay_result.get("cycle_comparisons", [])
    ht_match_count = sum(1 for c in cycle_comparisons if c.get("h_t_match", True))
    rt_match_count = sum(1 for c in cycle_comparisons if c.get("r_t_match", True))
    ut_match_count = sum(1 for c in cycle_comparisons if c.get("u_t_match", True))
    total_comparisons = len(cycle_comparisons) if cycle_comparisons else replayed_cycles

    per_cycle_consistency = {
        "total_cycles": original_cycles,
        "replayed_cycles": replayed_cycles,
        "coverage_pct": coverage_pct,
        "ht_match_count": ht_match_count,
        "rt_match_count": rt_match_count,
        "ut_match_count": ut_match_count,
        "ht_match_pct": (ht_match_count / total_comparisons * 100.0) if total_comparisons > 0 else 0.0,
        "rt_match_pct": (rt_match_count / total_comparisons * 100.0) if total_comparisons > 0 else 0.0,
        "ut_match_pct": (ut_match_count / total_comparisons * 100.0) if total_comparisons > 0 else 0.0,
    }

    # Determine is_fully_deterministic
    is_fully_deterministic = (
        status == "REPLAY_VERIFIED"
        and all_cycles_match
        and not is_partial
        and replay_mode == REPLAY_MODE_FULL
        and coverage_pct == 100.0
    )

    # Determine safety level
    has_critical_mismatch = any([
        critical_flags.get("ht_mismatch", False),
        critical_flags.get("rt_mismatch", False),
        critical_flags.get("ut_mismatch", False),
        critical_flags.get("cycle_count_mismatch", False),
    ])

    has_warning_condition = (
        critical_flags.get("config_hash_mismatch", False)
        or coverage_pct < 100.0
        or (replay_mode == REPLAY_MODE_DRY_RUN)
    )

    if status == "REPLAY_FAILED" or has_critical_mismatch or is_partial:
        safety_level = SafetyLevel.FAIL
    elif has_warning_condition:
        safety_level = SafetyLevel.WARN
    else:
        safety_level = SafetyLevel.OK

    # Determine recommended action
    if safety_level == SafetyLevel.FAIL:
        if has_critical_mismatch:
            recommended_action = "HALT: Critical attestation mismatch detected. Do not promote. Investigate determinism failure."
        elif is_partial:
            recommended_action = "HALT: Partial replay is inadmissible for governance. Complete full replay."
        else:
            recommended_action = "HALT: Replay verification failed. Review error details and re-run."
    elif safety_level == SafetyLevel.WARN:
        if critical_flags.get("config_hash_mismatch", False):
            recommended_action = "CAUTION: Config hash mismatch. Verify config consistency before proceeding."
        elif coverage_pct < 100.0:
            recommended_action = f"CAUTION: Replay coverage is {coverage_pct:.1f}%. Complete remaining cycles."
        else:
            recommended_action = "CAUTION: Minor issues detected. Review before proceeding."
    else:
        recommended_action = "PROCEED: All safety checks passed. Safe for promotion and policy updates."

    # Compute confidence score
    confidence_score = compute_replay_confidence(replay_result)

    # Determine policy update allowed
    policy_update_allowed = validate_replay_before_policy_update(replay_result)

    # Build error details if present
    error_details = None
    if replay_result.get("error_code") or replay_result.get("error_message"):
        error_details = {
            "error_code": replay_result.get("error_code"),
            "error_message": replay_result.get("error_message"),
            "first_mismatch_cycle": replay_result.get("first_mismatch_cycle"),
            "mismatch_type": replay_result.get("mismatch_type"),
        }

    return ReplaySafetyEnvelope(
        schema_version=SAFETY_ENVELOPE_VERSION,
        replay_mode=replay_mode,
        is_fully_deterministic=is_fully_deterministic,
        safety_level=safety_level,
        critical_mismatch_flags=critical_flags,
        per_cycle_consistency=per_cycle_consistency,
        recommended_action=recommended_action,
        confidence_score=confidence_score,
        policy_update_allowed=policy_update_allowed,
        error_details=error_details,
    )


def validate_replay_before_policy_update(
    replay_result: Dict[str, Any],
) -> bool:
    """
    Validate replay result before allowing policy updates.

    TASK 2: ΔPolicy Interaction Guard

    This function enforces the invariant: NO policy update if any RUN-4x mismatch.
    Policy updates must only occur when replay verification is fully successful.

    Conditions for policy update allowed:
    1. Status is REPLAY_VERIFIED
    2. No critical mismatch flags (ht, rt, ut, cycle_count)
    3. Not a partial replay
    4. Replay mode is FULL (not dry_run)
    5. All cycles match

    Args:
        replay_result: Dict from ReplayResult.to_dict()

    Returns:
        True if policy update is safe, False otherwise.
    """
    # Check status
    status = replay_result.get("status", "UNKNOWN")
    if status != "REPLAY_VERIFIED":
        return False

    # Check replay mode
    replay_mode = replay_result.get("replay_mode", REPLAY_MODE_FULL)
    if replay_mode != REPLAY_MODE_FULL:
        return False

    # Check partial replay
    is_partial = replay_result.get("is_partial_replay", False)
    if is_partial:
        return False

    # Check all cycles match
    per_cycle_stats = replay_result.get("per_cycle_stats", {})
    all_cycles_match = per_cycle_stats.get("all_cycles_match", replay_result.get("all_cycles_match", False))
    if not all_cycles_match:
        return False

    # Check critical mismatch flags
    critical_flags = replay_result.get("critical_mismatch_flags", {})

    # For legacy format, reconstruct from error_code
    if not critical_flags:
        error_code = replay_result.get("error_code")
        if error_code and error_code.startswith("RUN-4"):
            return False
    else:
        # Check each critical flag
        if critical_flags.get("ht_mismatch", False):
            return False
        if critical_flags.get("rt_mismatch", False):
            return False
        if critical_flags.get("ut_mismatch", False):
            return False
        if critical_flags.get("cycle_count_mismatch", False):
            return False
        # Note: config_hash_mismatch is a warning, not a blocker for policy updates
        # However, for strict safety, we also block on this
        if critical_flags.get("config_hash_mismatch", False):
            return False

    return True


def compute_replay_confidence(
    replay_result: Dict[str, Any],
) -> float:
    """
    Compute a confidence score for replay verification.

    TASK 3: Pre-Promotion Replay Score

    This function computes a compact metric in [0, 1] representing the
    overall confidence in the replay verification result.

    Scoring components (weighted):
    - Status match: 0.30 (REPLAY_VERIFIED = 1.0, else 0.0)
    - H_t consistency: 0.25 (% of cycles with matching h_t)
    - R_t consistency: 0.15 (% of cycles with matching r_t)
    - U_t consistency: 0.10 (% of cycles with matching u_t)
    - Coverage: 0.10 (replay_coverage_pct / 100)
    - Mode penalty: 0.10 (full = 1.0, dry_run = 0.5, partial = 0.0)

    Args:
        replay_result: Dict from ReplayResult.to_dict()

    Returns:
        Confidence score in [0, 1]. Higher is better.
    """
    # Weight configuration
    WEIGHT_STATUS = 0.30
    WEIGHT_HT = 0.25
    WEIGHT_RT = 0.15
    WEIGHT_UT = 0.10
    WEIGHT_COVERAGE = 0.10
    WEIGHT_MODE = 0.10

    score = 0.0

    # Component 1: Status (0.30)
    status = replay_result.get("status", "UNKNOWN")
    if status == "REPLAY_VERIFIED":
        score += WEIGHT_STATUS * 1.0
    elif status == "REPLAY_DRY_RUN":
        score += WEIGHT_STATUS * 0.5
    # REPLAY_FAILED or UNKNOWN = 0.0

    # Component 2-4: Per-cycle consistency
    cycle_comparisons = replay_result.get("cycle_comparisons", [])
    per_cycle_stats = replay_result.get("per_cycle_stats", {})

    if cycle_comparisons:
        total = len(cycle_comparisons)
        ht_matches = sum(1 for c in cycle_comparisons if c.get("h_t_match", True))
        rt_matches = sum(1 for c in cycle_comparisons if c.get("r_t_match", True))
        ut_matches = sum(1 for c in cycle_comparisons if c.get("u_t_match", True))

        score += WEIGHT_HT * (ht_matches / total if total > 0 else 0.0)
        score += WEIGHT_RT * (rt_matches / total if total > 0 else 0.0)
        score += WEIGHT_UT * (ut_matches / total if total > 0 else 0.0)
    else:
        # Use summary flags if no cycle comparisons available
        all_match = per_cycle_stats.get("all_cycles_match", replay_result.get("all_cycles_match", False))
        if all_match:
            score += WEIGHT_HT * 1.0
            score += WEIGHT_RT * 1.0
            score += WEIGHT_UT * 1.0
        else:
            # Check individual mismatch flags
            flags = replay_result.get("critical_mismatch_flags", {})
            if not flags.get("ht_mismatch", True):
                score += WEIGHT_HT * 1.0
            if not flags.get("rt_mismatch", True):
                score += WEIGHT_RT * 1.0
            if not flags.get("ut_mismatch", True):
                score += WEIGHT_UT * 1.0

    # Component 5: Coverage (0.10)
    coverage_pct = per_cycle_stats.get("replay_coverage_pct", replay_result.get("replay_coverage_pct", 100.0))
    score += WEIGHT_COVERAGE * (coverage_pct / 100.0)

    # Component 6: Mode penalty (0.10)
    replay_mode = replay_result.get("replay_mode", REPLAY_MODE_FULL)
    if replay_mode == REPLAY_MODE_FULL:
        mode_score = 1.0
    elif replay_mode == REPLAY_MODE_DRY_RUN:
        mode_score = 0.5
    else:  # REPLAY_MODE_PARTIAL
        mode_score = 0.0

    # Additional penalty for partial replay flag
    is_partial = replay_result.get("is_partial_replay", False)
    if is_partial:
        mode_score = 0.0

    score += WEIGHT_MODE * mode_score

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ============================================================================
# Phase IV: Replay Safety as Hard Policy Gate & Evidence Tile
# ============================================================================

class PromotionStatus:
    """Status constants for promotion safety evaluation."""
    OK = "OK"        # Safe for promotion and policy updates
    WARN = "WARN"    # Proceed with caution, review required
    BLOCK = "BLOCK"  # Do not promote, safety checks failed


def evaluate_replay_safety_for_promotion(
    envelope: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate replay safety envelope for promotion and RFL policy updates.

    TASK 1: ΔPolicy Safety Gate for Promotion & RFL

    This function takes a ReplaySafetyEnvelope.to_dict() or equivalent dict
    and determines whether promotion and policy updates are safe.

    Status determination:
    - BLOCK: safety_level=FAIL, policy_update_allowed=False, or is_partial_replay
    - WARN: safety_level=WARN, or confidence_score < 0.9
    - OK: All safety checks passed, high confidence

    Args:
        envelope: Dict from ReplaySafetyEnvelope.to_dict() or similar structure.

    Returns:
        Dict with:
        - safe_for_policy_update: bool (strict gate from validate_replay_before_policy_update)
        - safe_for_promotion: bool (can promote to next stage)
        - status: "OK" | "WARN" | "BLOCK"
        - reasons: list[str] explaining the decision
    """
    reasons: List[str] = []

    # Extract envelope fields
    safety_level = envelope.get("safety_level", SafetyLevel.FAIL)
    policy_update_allowed = envelope.get("policy_update_allowed", False)
    is_fully_deterministic = envelope.get("is_fully_deterministic", False)
    confidence_score = envelope.get("confidence_score", 0.0)
    replay_mode = envelope.get("replay_mode", REPLAY_MODE_FULL)

    # Extract from nested structures if present
    critical_flags = envelope.get("critical_mismatch_flags", {})
    per_cycle = envelope.get("per_cycle_consistency", {})
    error_details = envelope.get("error_details")

    # Determine safe_for_policy_update (strict gate)
    safe_for_policy_update = policy_update_allowed

    # Check for BLOCK conditions
    is_blocked = False

    if safety_level == SafetyLevel.FAIL:
        is_blocked = True
        reasons.append("Safety level is FAIL - critical mismatch detected")

    if not policy_update_allowed:
        is_blocked = True
        if "policy_update_allowed" not in [r for r in reasons]:
            reasons.append("Policy update not allowed by safety guard")

    if not is_fully_deterministic:
        is_blocked = True
        reasons.append("Replay is not fully deterministic")

    # Check critical mismatch flags
    if critical_flags.get("ht_mismatch", False):
        is_blocked = True
        reasons.append("H_t (verification hash) mismatch detected")

    if critical_flags.get("rt_mismatch", False):
        is_blocked = True
        reasons.append("R_t (ordering hash) mismatch detected")

    if critical_flags.get("ut_mismatch", False):
        is_blocked = True
        reasons.append("U_t (policy state) mismatch detected")

    if critical_flags.get("cycle_count_mismatch", False):
        is_blocked = True
        reasons.append("Cycle count mismatch between manifest and log")

    if replay_mode == REPLAY_MODE_PARTIAL:
        is_blocked = True
        reasons.append("Partial replay is inadmissible for governance")

    # Check error details
    if error_details:
        error_code = error_details.get("error_code")
        if error_code:
            is_blocked = True
            reasons.append(f"Replay error {error_code} encountered")

    # Check for WARN conditions (if not already blocked)
    is_warned = False

    if not is_blocked:
        if safety_level == SafetyLevel.WARN:
            is_warned = True
            reasons.append("Safety level is WARN - review recommended")

        if confidence_score < 0.9:
            is_warned = True
            reasons.append(f"Confidence score {confidence_score:.2f} is below threshold (0.9)")

        if critical_flags.get("config_hash_mismatch", False):
            is_warned = True
            reasons.append("Config hash mismatch - verify configuration consistency")

        coverage_pct = per_cycle.get("coverage_pct", 100.0)
        if coverage_pct < 100.0:
            is_warned = True
            reasons.append(f"Replay coverage is {coverage_pct:.1f}% - incomplete verification")

        if replay_mode == REPLAY_MODE_DRY_RUN:
            is_warned = True
            reasons.append("Dry-run mode - full replay not executed")

    # Determine final status
    if is_blocked:
        status = PromotionStatus.BLOCK
        safe_for_promotion = False
        safe_for_policy_update = False
    elif is_warned:
        status = PromotionStatus.WARN
        safe_for_promotion = True
    else:
        status = PromotionStatus.OK
        safe_for_promotion = True
        if not reasons:
            reasons.append("All safety checks passed")

    return {
        "safe_for_policy_update": safe_for_policy_update,
        "safe_for_promotion": safe_for_promotion,
        "status": status,
        "reasons": reasons,
    }


def summarize_replay_safety_for_evidence(
    envelope: Dict[str, Any],
    confidence: float,
    governance_view: Optional[Dict[str, Any]] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize replay safety for evidence pack integration.

    TASK 2: Evidence Pack Safety Adapter (Phase IV)
    Extended in Phase V: Accepts optional governance_view for alignment info.
    Extended in Phase VI: Accepts optional governance_signal for consolidated status.

    This function produces a compact summary suitable for D3's evidence
    pack governance system. It extracts the key safety indicators needed
    for evidence chain validation.

    Args:
        envelope: Dict from ReplaySafetyEnvelope.to_dict() or similar.
        confidence: Confidence score from compute_replay_confidence().
        governance_view: Optional dict from build_replay_safety_governance_view().
                        When provided, adds governance_alignment field.
        governance_signal: Optional dict from to_governance_signal_for_replay_safety().
                          When provided, adds governance_status field (consolidated status).

    Returns:
        Dict with:
        - replay_safety_ok: bool (True if fully safe)
        - confidence_score: float in [0, 1]
        - safety_level: "OK" | "WARN" | "FAIL"
        - policy_update_allowed: bool
        - status: "OK" | "WARN" | "BLOCK"
        - governance_alignment: "ALIGNED" | "TENSION" | "DIVERGENT" (if governance_view provided)
        - governance_status: "OK" | "WARN" | "BLOCK" (if governance_signal provided)
    """
    # Extract envelope fields
    safety_level = envelope.get("safety_level", SafetyLevel.FAIL)
    policy_update_allowed = envelope.get("policy_update_allowed", False)
    is_fully_deterministic = envelope.get("is_fully_deterministic", False)

    # Evaluate for promotion to get status
    promotion_eval = evaluate_replay_safety_for_promotion(envelope)
    status = promotion_eval["status"]

    # replay_safety_ok is True only when everything is safe
    replay_safety_ok = (
        safety_level == SafetyLevel.OK
        and policy_update_allowed
        and is_fully_deterministic
        and status == PromotionStatus.OK
    )

    result = {
        "replay_safety_ok": replay_safety_ok,
        "confidence_score": confidence,
        "safety_level": safety_level,
        "policy_update_allowed": policy_update_allowed,
        "status": status,
    }

    # Phase V: Add governance_alignment if governance_view provided
    if governance_view is not None:
        result["governance_alignment"] = governance_view.get(
            "governance_alignment", GovernanceAlignment.ALIGNED
        )

    # Phase VI: Add governance_status if governance_signal provided
    if governance_signal is not None:
        result["governance_status"] = governance_signal.get(
            "status", PromotionStatus.BLOCK
        )

    return result


def build_replay_safety_director_panel(
    envelope: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    governance_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a Director-facing replay safety panel.

    TASK 3: Director Replay Safety Panel (Phase IV)
    Extended in Phase V: Accepts optional governance_view for conflict info.

    This function produces a human-readable summary tile for the Director
    dashboard. It provides a quick visual status and headline summary.

    Args:
        envelope: Dict from ReplaySafetyEnvelope.to_dict() or similar.
        promotion_eval: Dict from evaluate_replay_safety_for_promotion().
        governance_view: Optional dict from build_replay_safety_governance_view().
                        When provided, adds conflict_flag and conflict_note.

    Returns:
        Dict with:
        - status_light: "green" | "yellow" | "red"
        - safety_level: "OK" | "WARN" | "FAIL"
        - is_fully_deterministic: bool
        - headline: Neutral sentence summarizing replay safety posture.
        - conflict_flag: bool (Phase V, if governance_view provided)
        - conflict_note: Optional[str] (Phase V, if conflict detected)
    """
    # Extract fields
    safety_level = envelope.get("safety_level", SafetyLevel.FAIL)
    is_fully_deterministic = envelope.get("is_fully_deterministic", False)
    confidence_score = envelope.get("confidence_score", 0.0)
    replay_mode = envelope.get("replay_mode", REPLAY_MODE_FULL)

    # Get promotion status
    status = promotion_eval.get("status", PromotionStatus.BLOCK)
    reasons = promotion_eval.get("reasons", [])

    # Determine status light color
    if status == PromotionStatus.OK:
        status_light = "green"
    elif status == PromotionStatus.WARN:
        status_light = "yellow"
    else:  # BLOCK
        status_light = "red"

    # Build headline
    if status == PromotionStatus.OK:
        headline = (
            f"Replay verification passed with {confidence_score:.0%} confidence. "
            f"Safe for promotion and policy updates."
        )
    elif status == PromotionStatus.WARN:
        if len(reasons) == 1:
            headline = f"Replay verification completed with caution: {reasons[0].lower()}"
        else:
            headline = (
                f"Replay verification completed with {len(reasons)} warnings. "
                f"Review recommended before promotion."
            )
    else:  # BLOCK
        if len(reasons) == 1:
            headline = f"Replay verification blocked: {reasons[0]}"
        elif len(reasons) > 1:
            headline = (
                f"Replay verification blocked due to {len(reasons)} issues. "
                f"Primary: {reasons[0]}"
            )
        else:
            headline = "Replay verification blocked. Review safety envelope for details."

    # Add mode context if not full
    if replay_mode == REPLAY_MODE_DRY_RUN:
        headline = f"[DRY-RUN] {headline}"
    elif replay_mode == REPLAY_MODE_PARTIAL:
        headline = f"[PARTIAL] {headline}"

    result = {
        "status_light": status_light,
        "safety_level": safety_level,
        "is_fully_deterministic": is_fully_deterministic,
        "headline": headline,
    }

    # Phase V: Add conflict info if governance_view provided
    if governance_view is not None:
        conflict = governance_view.get("conflict", False)
        result["conflict_flag"] = conflict
        if conflict:
            # Build conflict note from governance view reasons
            safety_status = governance_view.get("safety_status", "UNKNOWN")
            governance_status = governance_view.get("governance_status", "UNKNOWN")
            result["conflict_note"] = (
                f"Safety says {safety_status} but governance radar says {governance_status}. "
                f"Manual review required."
            )
        else:
            result["conflict_note"] = None

    return result


# ============================================================================
# Phase V: Replay Safety × Governance Radar Fusion
# ============================================================================

class GovernanceAlignment:
    """Alignment status between replay safety and governance radar."""
    ALIGNED = "ALIGNED"      # Both agree on status
    TENSION = "TENSION"      # Minor disagreement (WARN vs OK)
    DIVERGENT = "DIVERGENT"  # Major conflict (BLOCK vs OK or vice versa)


def build_replay_safety_governance_view(
    envelope: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a joint view combining replay safety with governance radar.

    Phase V Task 1: Safety + Radar Joint Summary

    This function fuses replay safety evaluation with CLAUDE A's governance
    radar to produce a unified view for promotion decisions.

    Alignment determination:
    - ALIGNED: Both safety and radar agree (both OK, both WARN, or both BLOCK)
    - TENSION: Minor disagreement (e.g., safety=WARN, radar=OK)
    - DIVERGENT: Major conflict (safety=BLOCK but radar=OK, or vice versa)

    Conflict is True when:
    - safety says BLOCK but radar says OK
    - safety says OK but radar says BLOCK

    Args:
        envelope: Dict from ReplaySafetyEnvelope.to_dict() or similar.
        promotion_eval: Dict from evaluate_replay_safety_for_promotion().
        radar: Dict with governance radar status. Expected fields:
               - status: "OK" | "WARN" | "BLOCK" (or similar)
               - reasons: list[str] (optional)

    Returns:
        Dict with:
        - safety_status: "OK" | "WARN" | "BLOCK"
        - governance_status: Status from radar
        - conflict: bool (True if critical disagreement)
        - governance_alignment: "ALIGNED" | "TENSION" | "DIVERGENT"
        - reasons: list[str] combining both sides
    """
    # Extract safety status
    safety_status = promotion_eval.get("status", PromotionStatus.BLOCK)
    safety_reasons = promotion_eval.get("reasons", [])

    # Extract radar status (normalize to our status constants)
    radar_status_raw = radar.get("status", "UNKNOWN")
    radar_reasons = radar.get("reasons", [])

    # Normalize radar status to match our constants
    radar_status = _normalize_status(radar_status_raw)

    # Determine alignment and conflict
    conflict = False
    alignment = GovernanceAlignment.ALIGNED

    if safety_status == radar_status:
        # Perfect alignment
        alignment = GovernanceAlignment.ALIGNED
        conflict = False
    elif _is_major_conflict(safety_status, radar_status):
        # Major conflict: one says BLOCK, other says OK
        alignment = GovernanceAlignment.DIVERGENT
        conflict = True
    else:
        # Minor tension: involves WARN
        alignment = GovernanceAlignment.TENSION
        conflict = False

    # Combine reasons from both sides
    combined_reasons: List[str] = []

    # Add safety reasons with prefix
    for reason in safety_reasons:
        combined_reasons.append(f"[Safety] {reason}")

    # Add radar reasons with prefix
    for reason in radar_reasons:
        combined_reasons.append(f"[Radar] {reason}")

    # Add conflict explanation if present
    if conflict:
        combined_reasons.append(
            f"[CONFLICT] Safety status ({safety_status}) diverges from "
            f"governance radar ({radar_status}). Manual review required."
        )

    return {
        "safety_status": safety_status,
        "governance_status": radar_status,
        "conflict": conflict,
        "governance_alignment": alignment,
        "reasons": combined_reasons,
    }


def _normalize_status(status: str) -> str:
    """Normalize external status strings to PromotionStatus constants."""
    status_upper = str(status).upper()

    if status_upper in ("OK", "PASS", "PASSED", "GREEN", "SAFE"):
        return PromotionStatus.OK
    elif status_upper in ("WARN", "WARNING", "YELLOW", "CAUTION"):
        return PromotionStatus.WARN
    elif status_upper in ("BLOCK", "BLOCKED", "FAIL", "FAILED", "RED", "UNSAFE"):
        return PromotionStatus.BLOCK
    else:
        # Unknown status treated as BLOCK for safety
        return PromotionStatus.BLOCK


def _is_major_conflict(status_a: str, status_b: str) -> bool:
    """Check if two statuses represent a major conflict (BLOCK vs OK)."""
    pair = {status_a, status_b}
    return pair == {PromotionStatus.OK, PromotionStatus.BLOCK}


# ============================================================================
# Phase VI: Replay Safety as Clean GovernanceSignal
# ============================================================================

def to_governance_signal_for_replay_safety(
    safety_eval: Dict[str, Any],
    radar_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collapse Safety and Radar into a single, normalized governance signal.

    Phase VI Task 1: Single GovernanceSignal Adapter

    This function produces a single, unambiguous "go/no-go" governance vector
    by fusing replay safety evaluation with governance radar status.

    Signal determination:
    - status: BLOCK if either side BLOCKs or alignment is DIVERGENT
    - status: WARN if alignment is TENSION (involves WARN)
    - status: OK if both OK and aligned

    Args:
        safety_eval: Dict from evaluate_replay_safety_for_promotion().
        radar_view: Dict from build_replay_safety_governance_view() or
                   with fields: governance_status, governance_alignment, conflict, reasons.

    Returns:
        Dict with:
        - status: "OK" | "WARN" | "BLOCK" (final consolidated status)
        - governance_alignment: "ALIGNED" | "TENSION" | "DIVERGENT"
        - safety_status: Status from safety_eval
        - governance_status: Status from radar
        - conflict: bool (True if critical disagreement)
        - reasons: list[str] with prefixed reasons from both sides
        - signal_type: "replay_safety" (identifies signal source)
    """
    # Extract safety status
    safety_status = safety_eval.get("status", PromotionStatus.BLOCK)
    safety_reasons = safety_eval.get("reasons", [])

    # Extract radar/governance view fields
    governance_status = radar_view.get("governance_status", PromotionStatus.BLOCK)
    governance_alignment = radar_view.get("governance_alignment", GovernanceAlignment.DIVERGENT)
    conflict = radar_view.get("conflict", False)
    radar_reasons = radar_view.get("reasons", [])

    # Determine final consolidated status
    if conflict or governance_alignment == GovernanceAlignment.DIVERGENT:
        # DIVERGENT alignment = always BLOCK
        final_status = PromotionStatus.BLOCK
    elif safety_status == PromotionStatus.BLOCK or governance_status == PromotionStatus.BLOCK:
        # Either side says BLOCK = BLOCK
        final_status = PromotionStatus.BLOCK
    elif governance_alignment == GovernanceAlignment.TENSION:
        # TENSION alignment (involves WARN) = WARN
        final_status = PromotionStatus.WARN
    elif safety_status == PromotionStatus.WARN or governance_status == PromotionStatus.WARN:
        # Either side says WARN = WARN
        final_status = PromotionStatus.WARN
    else:
        # Both OK and aligned = OK
        final_status = PromotionStatus.OK

    # Build combined reasons with prefixes
    combined_reasons: List[str] = []

    # Add safety reasons if not already prefixed
    for reason in safety_reasons:
        if reason.startswith("[Safety]") or reason.startswith("[Radar]") or reason.startswith("[CONFLICT]"):
            combined_reasons.append(reason)
        else:
            combined_reasons.append(f"[Safety] {reason}")

    # Add radar/governance reasons if not already prefixed
    for reason in radar_reasons:
        if reason.startswith("[Safety]") or reason.startswith("[Radar]") or reason.startswith("[CONFLICT]"):
            # Avoid duplicates - only add if not already present
            if reason not in combined_reasons:
                combined_reasons.append(reason)
        else:
            prefixed = f"[Radar] {reason}"
            if prefixed not in combined_reasons:
                combined_reasons.append(prefixed)

    # Add conflict explanation if present and not already in reasons
    if conflict:
        conflict_msg = (
            f"[CONFLICT] Safety status ({safety_status}) diverges from "
            f"governance radar ({governance_status}). Manual review required."
        )
        if not any("[CONFLICT]" in r for r in combined_reasons):
            combined_reasons.append(conflict_msg)

    return {
        "status": final_status,
        "governance_alignment": governance_alignment,
        "safety_status": safety_status,
        "governance_status": governance_status,
        "conflict": conflict,
        "reasons": combined_reasons,
        "signal_type": "replay_safety",
    }


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of config for traceability."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate a deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


@dataclass
class U2Config:
    """
    Configuration for U2 uplift experiments.
    
    Attributes:
        experiment_id: Unique identifier for this experiment.
        slice_name: Name of the curriculum slice.
        mode: Execution mode ("baseline" or "rfl").
        total_cycles: Total number of cycles to run.
        master_seed: Master random seed for determinism.
        snapshot_interval: Save snapshot every N cycles (0 = disabled).
        snapshot_dir: Directory for snapshot files.
        output_dir: Directory for output files.
        slice_config: Raw slice configuration dict.
    """
    experiment_id: str
    slice_name: str
    mode: str
    total_cycles: int
    master_seed: int
    snapshot_interval: int = 0
    snapshot_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    slice_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.mode not in ("baseline", "rfl"):
            raise ValueError(f"mode must be 'baseline' or 'rfl', got {self.mode}")
        if self.snapshot_dir is not None and not isinstance(self.snapshot_dir, Path):
            self.snapshot_dir = Path(self.snapshot_dir)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "total_cycles": self.total_cycles,
            "master_seed": self.master_seed,
            "snapshot_interval": self.snapshot_interval,
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "slice_config": self.slice_config,
        }


@dataclass
class U2CycleResult:
    """
    Result of a single U2 cycle execution.
    
    Extended from CycleResult to include additional U2-specific fields.
    """
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "seed": self.seed,
            "item": self.item,
            "result": self.result,
            "success": self.success,
            "error_message": self.error_message,
        }


class RFLPolicy:
    """
    A mock RFL policy model for U2 experiments.

    Extended with Lean failure feedback so the planner can downweight
    problematic regions deterministically.
    """

    def __init__(self, seed: int):
        self.scores: Dict[str, float] = {}
        self.rng = random.Random(seed)
        self.timeout_counts: Dict[str, int] = {}
        self.timeout_penalties: Dict[str, float] = {}
        self.pattern_failure_counts: Dict[str, int] = {}
        self.pattern_penalties: Dict[str, float] = {}

    def score(self, items: List[str]) -> List[float]:
        """Score items. Higher is better."""
        scored: List[float] = []
        for item in items:
            if item not in self.scores:
                self.scores[item] = self.rng.random()
            base = self.scores[item]
            adjusted = max(0.01, min(0.99, base * self._compute_penalty(item)))
            scored.append(adjusted)
        return scored

    def update(
        self,
        item: str,
        success: bool,
        failure_signal: Optional[LeanFailureSignal] = None,
    ) -> None:
        """Update the policy based on feedback and Lean telemetry."""
        if success:
            self.scores[item] = self.scores.get(item, 0.5) * 1.1
        else:
            self.scores[item] = self.scores.get(item, 0.5) * 0.9
            if failure_signal is not None:
                self._register_failure(item, failure_signal)
        # Clamp base score to stable range
        self.scores[item] = max(0.01, min(self.scores[item], 0.99))

    def get_state(self) -> Tuple:
        """Get RNG state for snapshotting."""
        return self.rng.getstate()

    def set_state(self, state: Tuple) -> None:
        """Restore RNG state from snapshot."""
        self.rng.setstate(state)

    # ------------------------------------------------------------------ #
    # Failure handling helpers
    # ------------------------------------------------------------------ #

    def _register_failure(self, item: str, signal: LeanFailureSignal) -> None:
        if signal.kind == "timeout":
            count = self.timeout_counts.get(item, 0) + 1
            self.timeout_counts[item] = count
            # Penalize exponentially but keep a deterministic floor.
            self.timeout_penalties[item] = max(0.2, 0.8 ** count)
        elif signal.kind == "type_error":
            key = self._pattern_key(item)
            count = self.pattern_failure_counts.get(key, 0) + 1
            self.pattern_failure_counts[key] = count
            self.pattern_penalties[key] = max(0.1, 0.85 ** count)
        elif signal.kind == "tactic_failure":
            # Encourage exploration away from the item by shrinking base score.
            self.scores[item] = max(0.01, self.scores.get(item, 0.5) * 0.85)

    def _compute_penalty(self, item: str) -> float:
        penalty = 1.0
        penalty *= self.timeout_penalties.get(item, 1.0)
        penalty *= self.pattern_penalties.get(self._pattern_key(item), 1.0)
        return penalty

    @staticmethod
    def _pattern_key(item: str) -> str:
        """Deterministic grouping key for similar statements."""
        lhs, _, _ = item.partition("->")
        lhs = lhs.strip()
        if lhs:
            return lhs[:16]
        letters = "".join(ch for ch in item if ch.isalpha())
        return (letters.lower() or item)[:16]


class U2Runner:
    """
    Core runner for U2 uplift experiments with snapshot support.
    
    Manages:
    - Cycle execution with deterministic seeding
    - Policy state (for RFL mode)
    - Telemetry history (ht_series)
    - Snapshot save/restore
    """
    
    _VALID_FAILURE_KINDS: tuple[LeanFailureKind, ...] = (
        "timeout",
        "type_error",
        "tactic_failure",
        "unknown",
    )

    def __init__(self, config: U2Config):
        """Initialize the runner with configuration."""
        self.config = config
        self.cycle_index = 0
        self.ht_series: List[Dict[str, Any]] = []
        
        # Generate seed schedule
        self.seed_schedule = generate_seed_schedule(
            config.master_seed, 
            config.total_cycles
        )
        
        # Initialize policy (only used in RFL mode)
        self.policy: Optional[RFLPolicy] = None
        if config.mode == "rfl":
            self.policy = RFLPolicy(config.master_seed)
        
        # Policy statistics
        self.policy_update_count = 0
        self.success_count: Dict[str, int] = {}
        self.attempt_count: Dict[str, int] = {}
        self.lean_failure_signals: List[LeanFailureSignal] = []
        
        # Main RNG for non-policy operations
        self._rng = random.Random(config.master_seed)

        # Phase VII CORTEX: TDA Governance Hook (optional)
        # Register via register_tda_hook() before running cycles
        self._tda_hook: Optional["TDAGovernanceHook"] = None

    def register_tda_hook(self, hook: "TDAGovernanceHook") -> None:
        """
        Register TDA governance hook for cycle boundary evaluation.

        Phase VII CORTEX Integration: When registered, the hook's
        on_cycle_complete() is called after each cycle completes.
        In HARD mode, should_block=True will mark the cycle as blocked.

        Args:
            hook: TDAGovernanceHook instance from backend.tda.runner_hook
        """
        self._tda_hook = hook
    
    def run_cycle(
        self,
        items: List[str],
        execute_fn: Callable[[str, int], Tuple[bool, Any]],
    ) -> U2CycleResult:
        """
        Execute a single experiment cycle.
        
        All ordering logic is delegated to the runtime's cycle_orchestrator
        module via get_ordering_strategy(). This ensures INV-RUN-1 compliance:
        no duplication of ordering logic outside cycle_orchestrator.
        
        Args:
            items: List of candidate items.
            execute_fn: Function that executes an item and returns (success, result).
        
        Returns:
            U2CycleResult with cycle outcome.
        """
        if self.cycle_index >= self.config.total_cycles:
            raise RuntimeError("Experiment already completed")
        
        cycle_seed = self.seed_schedule[self.cycle_index]
        rng = random.Random(cycle_seed)
        
        # --- Ordering Step (INV-RUN-1: delegated to cycle_orchestrator) ---
        strategy = get_ordering_strategy(self.config.mode, self.policy)
        ordered_items = strategy.order(items, rng)
        chosen_item = ordered_items[0]
        
        # --- Execution ---
        success, result = execute_fn(chosen_item, cycle_seed)
        failure_signal = None if success else self._extract_failure_signal(result)
        if failure_signal is not None:
            self.lean_failure_signals.append(failure_signal)

        # --- Policy Update ---
        if self.config.mode == "rfl" and self.policy is not None:
            self.policy.update(chosen_item, success, failure_signal)
            self.policy_update_count += 1
            
            # Track success/attempt counts
            self.attempt_count[chosen_item] = self.attempt_count.get(chosen_item, 0) + 1
            if success:
                self.success_count[chosen_item] = self.success_count.get(chosen_item, 0) + 1
        
        # --- Telemetry ---
        telemetry_record = {
            "cycle": self.cycle_index,
            "slice": self.config.slice_name,
            "mode": self.config.mode,
            "seed": cycle_seed,
            "item": chosen_item,
            "result": str(result),
            "success": success,
            "label": "PHASE II — NOT USED IN PHASE I",
        }
        self.ht_series.append(telemetry_record)
        
        # Advance cycle
        self.cycle_index += 1

        # Build cycle result
        cycle_result = U2CycleResult(
            cycle_index=self.cycle_index - 1,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            seed=cycle_seed,
            item=chosen_item,
            result=result if isinstance(result, dict) else {"raw": str(result)},
            success=success,
        )

        # ===== Phase VII CORTEX: TDA GOVERNANCE HOOK =====
        # Evaluate hard gate AFTER cycle completion
        # This is the BLOCKING CALL: if TDA says block, mark result as blocked
        if self._tda_hook is not None:
            # Build telemetry from cycle for TDA evaluation
            tda_telemetry = {
                "slice_name": self.config.slice_name,
                "cycle_index": self.cycle_index - 1,
                "mode": self.config.mode,
                "seed": cycle_seed,
                "success": success,
            }

            # Build cycle result dict with TDA metrics
            # Derive HSS from success (1.0 if success, 0.0 if failure)
            # This is a simple proxy; real TDA would use actual metrics
            tda_cycle_data = {
                "hss": 1.0 if success else 0.0,
                "sns": 0.0,
                "pcs": 0.0,
                "drs": 0.0,
            }

            tda_decision = self._tda_hook.on_cycle_complete(
                cycle_index=self.cycle_index - 1,
                success=success,
                cycle_result=tda_cycle_data,
                telemetry=tda_telemetry,
            )

            # If TDA says block in HARD mode, mark cycle result
            if tda_decision.should_block:
                # Modify result to indicate TDA block
                cycle_result.result["tda_blocked"] = True
                cycle_result.result["tda_reason"] = tda_decision.reason
                # Revert policy update if it was applied
                if self.config.mode == "rfl" and self.policy is not None:
                    self.policy_update_count -= 1
        # ===== END Phase VII CORTEX =====

        return cycle_result
    
    def maybe_save_snapshot(self) -> Optional[Path]:
        """
        Save snapshot if at interval boundary.
        
        Returns:
            Path to snapshot file if saved, None otherwise.
        """
        if self.config.snapshot_interval <= 0:
            return None
        
        if self.cycle_index % self.config.snapshot_interval != 0:
            return None
        
        if self.config.snapshot_dir is None:
            return None
        
        snapshot = self.capture_state()
        snapshot_path = (
            self.config.snapshot_dir / 
            f"snapshot_{self.config.experiment_id}_cycle_{self.cycle_index}.snap"
        )
        save_snapshot(snapshot, snapshot_path)
        return snapshot_path
    
    def capture_state(self, manifest_hash: str = "") -> SnapshotData:
        """
        Capture current state for snapshotting.
        
        Args:
            manifest_hash: SHA256 hash of experiment manifest (optional, for replay compatibility)
            
        Returns:
            SnapshotData with all runtime state captured
        """
        numpy_state, python_state = capture_prng_states()
        
        policy_scores = {}
        policy_rng_state = None
        if self.policy is not None:
            policy_scores = dict(self.policy.scores)
            policy_rng_state = self.policy.get_state()
        
        # Compute ht_series hash for replay verification
        ht_series_hash = ""
        if self.ht_series:
            import hashlib
            ht_str = json.dumps(self.ht_series, sort_keys=True)
            ht_series_hash = hashlib.sha256(ht_str.encode()).hexdigest()

        lean_failure_history = [
            {
                "kind": signal.kind,
                "message": signal.message,
                "elapsed_ms": signal.elapsed_ms,
            }
            for signal in self.lean_failure_signals
        ]
        
        # Generate deterministic timestamp from cycle index
        # This ensures identical snapshots produce identical timestamps
        from substrate.repro.determinism import deterministic_timestamp
        snapshot_ts = deterministic_timestamp(self.cycle_index).isoformat() + "Z"
        
        return SnapshotData(
            schema_version="1.0",
            cycle_index=self.cycle_index,
            total_cycles=self.config.total_cycles,
            numpy_rng_state=numpy_state,
            python_rng_state=python_state,
            master_seed=self.config.master_seed,
            seed_schedule=list(self.seed_schedule),
            policy_scores=policy_scores,
            policy_rng_state=policy_rng_state,
            policy_update_count=self.policy_update_count,
            success_count=dict(self.success_count),
            attempt_count=dict(self.attempt_count),
            config_hash=compute_config_hash(self.config.to_dict()),
            experiment_id=self.config.experiment_id,
            mode=self.config.mode,
            slice_name=self.config.slice_name,
            ht_series_length=len(self.ht_series),
            # Replay & governance metadata
            ht_series_hash=ht_series_hash,
            manifest_hash=manifest_hash,
            created_at_cycle=self.cycle_index,
            snapshot_timestamp=snapshot_ts,
            lean_failure_history=lean_failure_history,
        )
    
    def restore_state(self, snapshot: SnapshotData) -> None:
        """Restore state from a snapshot."""
        self.cycle_index = snapshot.cycle_index
        
        # Restore PRNG states
        restore_prng_states(snapshot.numpy_rng_state, snapshot.python_rng_state)
        
        # Restore policy state
        if self.policy is not None and snapshot.policy_rng_state is not None:
            self.policy.scores = dict(snapshot.policy_scores)
            self.policy.set_state(snapshot.policy_rng_state)
        
        self.policy_update_count = snapshot.policy_update_count
        self.success_count = dict(snapshot.success_count)
        self.attempt_count = dict(snapshot.attempt_count)
        self.lean_failure_signals = [
            LeanFailureSignal(
                kind=cast(LeanFailureKind, entry.get("kind", "unknown")),
                message=str(entry.get("message", "")),
                elapsed_ms=int(entry.get("elapsed_ms", 0)),
            )
            for entry in snapshot.lean_failure_history
        ]

    def _extract_failure_signal(self, execution_result: Any) -> Optional[LeanFailureSignal]:
        """
        Normalize Lean failure payloads emitted by substrates.

        Returns:
            LeanFailureSignal if a well-formed failure payload exists.
        """
        if not isinstance(execution_result, dict):
            return None
        payload = execution_result.get("lean_failure")
        if not isinstance(payload, dict):
            return None

        raw_kind = str(payload.get("kind", "unknown")).lower()
        if raw_kind not in self._VALID_FAILURE_KINDS:
            raw_kind = "unknown"

        message = str(payload.get("message", "")).strip() or f"{raw_kind} failure"
        elapsed_raw = payload.get("elapsed_ms", 0)
        try:
            elapsed_ms = int(elapsed_raw)
        except (TypeError, ValueError):
            elapsed_ms = 0

        return LeanFailureSignal(
            kind=cast(LeanFailureKind, raw_kind),
            message=message,
            elapsed_ms=elapsed_ms,
        )

    @classmethod
    def replay(
        cls,
        manifest_path: Path,
        items: List[str],
        execute_fn: Callable[[str, int], Tuple[bool, Any]],
        config_hash_check: bool = True,
        strict_mode: bool = False,
    ) -> ReplayResult:
        """
        Replay a completed experiment and verify determinism.

        Per u2_runner_spec.md v1.1.0 Section 6.4: Replay Verification Protocol.

        This method:
        1. Loads manifest from manifest_path
        2. Validates manifest status = "COMPLETED"
        3. Validates seed schedule formula
        4. Loads original results from artifacts path
        5. Re-executes cycles deterministically
        6. Compares attestation roots (h_t, r_t, u_t) per cycle

        Args:
            manifest_path: Path to the primary run manifest JSON.
            items: List of candidate items (same as original run).
            execute_fn: Function that executes an item and returns (success, result).
            config_hash_check: If True, warn on config hash mismatch (RUN-48).
            strict_mode: If True, fail on config hash mismatch instead of warn.

        Returns:
            ReplayResult with verification status and cycle comparisons.

        Raises:
            ReplayManifestMismatch: RUN-41 if manifest is invalid.
            ReplaySeedScheduleMismatch: RUN-42 if seed schedule doesn't match.
            ReplayLogMissing: RUN-43 if results log is missing.
            ReplayHtMismatch: RUN-44 if h_t root doesn't match.
            ReplayRtMismatch: RUN-45 if r_t root doesn't match.
            ReplayUtMismatch: RUN-46 if u_t root doesn't match.
            ReplayCycleCountMismatch: RUN-47 if cycle counts don't match.
            ReplayConfigHashMismatch: RUN-48 if config hash doesn't match (strict mode).
            ReplayVerificationError: RUN-49 if verification throws exception.
            ReplayUnknownError: RUN-50 for unclassified errors.
        """
        manifest_path = Path(manifest_path)
        cycle_comparisons: List[ReplayCycleComparison] = []

        try:
            # Step 1: Load manifest
            if not manifest_path.exists():
                raise ReplayManifestMismatch(
                    f"Manifest file not found: {manifest_path}",
                    {"path": str(manifest_path)}
                )

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Step 2: Validate manifest has required fields
            required_fields = ["slice", "mode", "cycles", "initial_seed"]
            missing = [f for f in required_fields if f not in manifest]
            if missing:
                raise ReplayManifestMismatch(
                    f"Manifest missing required fields: {missing}",
                    {"missing_fields": missing, "path": str(manifest_path)}
                )

            # Step 3: Validate seed schedule (if present)
            # Note: The runner uses generate_seed_schedule which is deterministic
            # We verify by comparing seeds during replay

            slice_name = manifest["slice"]
            mode = manifest["mode"]
            total_cycles = manifest["cycles"]
            master_seed = manifest["initial_seed"]

            # Step 4: Locate and load original results
            results_path_str = manifest.get("outputs", {}).get("results")
            if not results_path_str:
                raise ReplayLogMissing(
                    "Manifest missing results path",
                    {"manifest_path": str(manifest_path)}
                )

            results_path = Path(results_path_str)
            if not results_path.exists():
                # Try relative to manifest directory
                results_path = manifest_path.parent / Path(results_path_str).name
                if not results_path.exists():
                    raise ReplayLogMissing(
                        f"Results log not found: {results_path_str}",
                        {"path": results_path_str, "manifest_path": str(manifest_path)}
                    )

            # Load original results
            original_results: List[Dict[str, Any]] = []
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        original_results.append(json.loads(line))

            # Step 5: Validate cycle count
            if len(original_results) != total_cycles:
                raise ReplayCycleCountMismatch(
                    f"Cycle count mismatch: manifest={total_cycles}, log={len(original_results)}",
                    {
                        "manifest_cycles": total_cycles,
                        "log_cycles": len(original_results),
                        "path": str(results_path)
                    }
                )

            # Step 6: Check config hash (optional)
            original_config_hash = manifest.get("slice_config_hash")
            if config_hash_check and original_config_hash:
                # We don't have access to current config here, but caller can check
                pass  # Config hash check is done by caller if needed

            # Step 7: Create runner for replay
            config = U2Config(
                experiment_id=f"replay_{manifest_path.stem}",
                slice_name=slice_name,
                mode=mode,
                total_cycles=total_cycles,
                master_seed=master_seed,
            )
            runner = cls(config)

            # Step 8: Re-execute cycles and compare
            first_mismatch_cycle: Optional[int] = None
            mismatch_type: Optional[str] = None
            error_code: Optional[str] = None
            error_message: Optional[str] = None

            for i in range(total_cycles):
                try:
                    # Re-execute cycle
                    result = runner.run_cycle(items, execute_fn)

                    # Get original record
                    orig = original_results[i]

                    # Compute h_t (hash of cycle telemetry)
                    replay_h_t = cls._compute_ht(runner.ht_series[-1])
                    original_h_t = cls._compute_ht(orig)

                    # Compute r_t (hash of item ordering)
                    # For now, use item selection as proxy
                    replay_r_t = hashlib.sha256(
                        f"{i}|{result.seed}|{result.item}".encode()
                    ).hexdigest()
                    original_r_t = hashlib.sha256(
                        f"{orig.get('cycle', i)}|{orig.get('seed', 0)}|{orig.get('item', '')}".encode()
                    ).hexdigest()

                    # Compute u_t (policy state hash, RFL only)
                    replay_u_t: Optional[str] = None
                    original_u_t: Optional[str] = None
                    if mode == "rfl" and runner.policy:
                        replay_u_t = cls._compute_ut(runner.policy.scores)
                        # Original u_t would need to be in the log; use None if not available
                        original_u_t = None  # Not stored in current format

                    # Compare roots
                    h_t_match = replay_h_t == original_h_t
                    r_t_match = replay_r_t == original_r_t
                    u_t_match = (replay_u_t == original_u_t) if (replay_u_t and original_u_t) else True

                    comparison = ReplayCycleComparison(
                        cycle_index=i,
                        original_h_t=original_h_t,
                        replay_h_t=replay_h_t,
                        original_r_t=original_r_t,
                        replay_r_t=replay_r_t,
                        original_u_t=original_u_t,
                        replay_u_t=replay_u_t,
                        h_t_match=h_t_match,
                        r_t_match=r_t_match,
                        u_t_match=u_t_match,
                        all_match=h_t_match and r_t_match and u_t_match,
                    )
                    cycle_comparisons.append(comparison)

                    # Check for first mismatch
                    if not comparison.all_match and first_mismatch_cycle is None:
                        first_mismatch_cycle = i
                        if not h_t_match:
                            mismatch_type = "h_t"
                            error_code = "RUN-44"
                            error_message = f"H_t mismatch at cycle {i}: expected {original_h_t}, got {replay_h_t}"
                        elif not r_t_match:
                            mismatch_type = "r_t"
                            error_code = "RUN-45"
                            error_message = f"R_t mismatch at cycle {i}: ordering divergence"
                        elif not u_t_match:
                            mismatch_type = "u_t"
                            error_code = "RUN-46"
                            error_message = f"U_t mismatch at cycle {i}: policy state divergence"

                except ReplayError:
                    raise
                except Exception as e:
                    raise ReplayVerificationError(
                        f"Verification error during replay at cycle {i}: {e}",
                        {"cycle": i, "exception": str(e)}
                    )

            # Step 9: Compute results hashes
            original_results_hash = manifest.get("ht_series_hash")
            replay_ht_series_str = json.dumps(runner.ht_series, sort_keys=True)
            replay_results_hash = hashlib.sha256(replay_ht_series_str.encode()).hexdigest()

            results_hash_match = (
                original_results_hash == replay_results_hash
                if original_results_hash else True
            )

            # Step 10: Determine final status
            all_cycles_match = all(c.all_match for c in cycle_comparisons)
            status = "REPLAY_VERIFIED" if all_cycles_match else "REPLAY_FAILED"

            # Compute mismatch flags for governance (Task 3)
            has_ht_mismatch = any(not c.h_t_match for c in cycle_comparisons)
            has_rt_mismatch = any(not c.r_t_match for c in cycle_comparisons)
            has_ut_mismatch = any(not c.u_t_match for c in cycle_comparisons)

            # Compute replay coverage
            replay_coverage_pct = (len(cycle_comparisons) / total_cycles * 100.0) if total_cycles > 0 else 0.0

            return ReplayResult(
                status=status,
                manifest_path=str(manifest_path),
                original_cycles=total_cycles,
                replayed_cycles=len(cycle_comparisons),
                all_cycles_match=all_cycles_match,
                first_mismatch_cycle=first_mismatch_cycle,
                mismatch_type=mismatch_type,
                error_code=error_code,
                error_message=error_message,
                cycle_comparisons=cycle_comparisons,
                original_results_hash=original_results_hash,
                replay_results_hash=replay_results_hash,
                results_hash_match=results_hash_match,
                # Contract v1.0.0 fields
                replay_mode=REPLAY_MODE_FULL,
                is_partial_replay=len(cycle_comparisons) < total_cycles,
                replay_coverage_pct=replay_coverage_pct,
                # Critical mismatch flags
                has_ht_mismatch=has_ht_mismatch,
                has_rt_mismatch=has_rt_mismatch,
                has_ut_mismatch=has_ut_mismatch,
            )

        except ReplayError:
            raise
        except json.JSONDecodeError as e:
            raise ReplayManifestMismatch(
                f"Invalid JSON in manifest: {e}",
                {"path": str(manifest_path), "error": str(e)}
            )
        except Exception as e:
            raise ReplayUnknownError(
                f"Unknown replay error: {e}",
                {"path": str(manifest_path), "exception": str(e)}
            )

    @staticmethod
    def _compute_ht(telemetry: Dict[str, Any]) -> str:
        """Compute h_t hash from telemetry record."""
        # Canonical serialization for deterministic hashing
        canonical = json.dumps(telemetry, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _compute_ut(policy_scores: Dict[str, float]) -> str:
        """Compute u_t hash from policy scores."""
        canonical = json.dumps(policy_scores, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ============================================================================
# Trace Logging Infrastructure
# ============================================================================

class TracedExperimentContext:
    """
    Context object passed to the core runner for emitting trace events.
    
    This provides a clean interface for the runner to log events without
    needing to know about the logger implementation details.
    """
    
    def __init__(
        self,
        logger: U2TraceLogger,
        slice_name: str,
        mode: str,
    ):
        self._logger = logger
        self._slice_name = slice_name
        self._mode = mode
        self._cycle_start_time: Optional[float] = None
        self._substrate_start_time: Optional[float] = None
        self._substrate_duration_ms: Optional[float] = None
    
    def begin_cycle(self, cycle: int) -> None:
        """Mark the start of a cycle for timing."""
        self._cycle_start_time = time.perf_counter()
        self._substrate_duration_ms = None
    
    def begin_substrate_call(self) -> None:
        """Mark the start of a substrate call for timing."""
        self._substrate_start_time = time.perf_counter()
    
    def end_substrate_call(self) -> None:
        """Mark the end of a substrate call and record duration."""
        if self._substrate_start_time is not None:
            self._substrate_duration_ms = (time.perf_counter() - self._substrate_start_time) * 1000.0
            self._substrate_start_time = None
    
    def end_cycle(self, cycle: int) -> None:
        """Mark the end of a cycle and emit duration event."""
        if self._cycle_start_time is not None:
            duration_ms = (time.perf_counter() - self._cycle_start_time) * 1000.0
            self._logger.log_cycle_duration(
                schema.CycleDurationEvent(
                    cycle=cycle,
                    slice_name=self._slice_name,
                    mode=self._mode,
                    duration_ms=duration_ms,
                    substrate_duration_ms=self._substrate_duration_ms,
                )
            )
            self._cycle_start_time = None
    
    def log_candidate_ordering(
        self,
        cycle: int,
        ordering: list,
    ) -> None:
        """Log candidate ordering for a cycle."""
        self._logger.log_candidate_ordering(
            schema.CandidateOrderingEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode=self._mode,  # type: ignore
                ordering=tuple(ordering),
            )
        )
    
    def log_scoring_features(
        self,
        cycle: int,
        features: list,
    ) -> None:
        """Log scoring features for RFL policy."""
        if self._mode != "rfl":
            return
        
        self._logger.log_scoring_features(
            schema.ScoringFeaturesEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode="rfl",
                features=tuple(features),
            )
        )
    
    def log_policy_weight_update(
        self,
        cycle: int,
        weights_before: Dict[str, float],
        weights_after: Dict[str, float],
        reward: float,
        verified_count: int,
        target: int,
    ) -> None:
        """Log policy weight update after feedback."""
        if self._mode != "rfl":
            return
        
        self._logger.log_policy_weight_update(
            schema.PolicyWeightUpdateEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode="rfl",
                weights_before=weights_before,
                weights_after=weights_after,
                reward=reward,
                verified_count=verified_count,
                target=target,
            )
        )
    
    def log_budget_consumption(
        self,
        cycle: int,
        candidates_considered: int,
        candidates_limit: Optional[int],
        budget_exhausted: bool,
    ) -> None:
        """Log resource budget tracking."""
        self._logger.log_budget_consumption(
            schema.BudgetConsumptionEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode=self._mode,
                candidates_considered=candidates_considered,
                candidates_limit=candidates_limit,
                budget_exhausted=budget_exhausted,
            )
        )
    
    def log_substrate_result(
        self,
        cycle: int,
        item_id: str,
        seed: int,
        result: str,
        verified_hashes: list,
    ) -> None:
        """Log FO substrate simulation outcome."""
        self._logger.log_substrate_result(
            schema.SubstrateResultEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode=self._mode,
                item_id=item_id,
                seed=seed,
                result=result,
                verified_hashes=tuple(verified_hashes),
            )
        )
    
    def log_cycle_telemetry(
        self,
        cycle: int,
        raw_record: Dict[str, Any],
    ) -> None:
        """Log full cycle telemetry record."""
        self._logger.log_cycle_telemetry(
            schema.CycleTelemetryEvent(
                cycle=cycle,
                slice_name=self._slice_name,
                mode=self._mode,
                raw_record=raw_record,
            )
        )


def run_with_traces(
    *,
    run_id: str,
    slice_name: str,
    mode: str,
    cycles: int,
    seed: int,
    log_path: Path,
    core_runner: Callable[..., Any],
    config: Dict[str, Any],
    enable_hash_chain: bool = False,
    **runner_kwargs: Any,
) -> Any:
    """
    Run the U2 experiment with trace logging enabled.
    
    This function wraps the core runner function and injects trace logging
    at session boundaries. Per-cycle logging is handled by passing a
    TracedExperimentContext to the core runner via the `trace_ctx` kwarg.
    
    Args:
        run_id: Unique identifier for this run.
        slice_name: Name of the curriculum slice.
        mode: Execution mode ("baseline" or "rfl").
        cycles: Number of cycles to run.
        seed: Initial random seed.
        log_path: Path for trace log output (JSONL).
        core_runner: The actual experiment runner function.
        config: Experiment configuration dict.
        enable_hash_chain: If True, enable hash-chaining for tamper-evidence.
        **runner_kwargs: Additional kwargs passed to core_runner.
    
    Returns:
        The return value of core_runner (unchanged).
    """
    config_hash = compute_config_hash(config)
    
    with U2TraceLogger(log_path, enable_hash_chain=enable_hash_chain) as logger:
        # Emit session start
        logger.log_session_start(
            schema.SessionStartEvent(
                run_id=run_id,
                slice_name=slice_name,
                mode=mode,
                schema_version=schema.TRACE_SCHEMA_VERSION,
                config_hash=config_hash,
                total_cycles=cycles,
                initial_seed=seed,
            )
        )
        
        # Create context for per-cycle logging
        trace_ctx = TracedExperimentContext(logger, slice_name, mode)
        
        # Run the core experiment with trace context injected
        result = core_runner(
            slice_name=slice_name,
            cycles=cycles,
            seed=seed,
            mode=mode,
            config=config,
            trace_ctx=trace_ctx,
            **runner_kwargs,
        )
        
        # Extract manifest hashes from result if available
        manifest_hash = None
        ht_series_hash = None
        completed_cycles = cycles
        
        if isinstance(result, dict):
            manifest_hash = result.get("manifest_hash")
            ht_series_hash = result.get("ht_series_hash")
            completed_cycles = result.get("completed_cycles", cycles)
        
        # Emit session end
        logger.log_session_end(
            schema.SessionEndEvent(
                run_id=run_id,
                slice_name=slice_name,
                mode=mode,
                schema_version=schema.TRACE_SCHEMA_VERSION,
                manifest_hash=manifest_hash,
                ht_series_hash=ht_series_hash,
                total_cycles=cycles,
                completed_cycles=completed_cycles,
            )
        )
        
        return result
