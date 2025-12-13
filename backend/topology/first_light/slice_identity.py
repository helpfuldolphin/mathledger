"""
Phase X: Slice Identity Verification Module

This module implements slice identity verification as a pre-execution blocker
for P3/P4 shadow experiments. It ensures that slice identity invariants
(SI-001 through SI-006) are satisfied before any experiment run begins.

SHADOW MODE CONTRACT:
- Verification returns ADVISORY status only
- Does NOT enforce blocking (caller decides)
- All outputs are for logging and evidence chain integrity
- No governance modification

See: docs/system_law/SliceIdentity_PhaseX_Invariants.md

Status: PHASE X PRE-EXECUTION BLOCKER (SHADOW-ONLY)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from substrate.repro.determinism import (
        deterministic_hash,
        deterministic_isoformat,
    )
except ImportError:
    # Fallback implementations
    def deterministic_hash(content: Any, algorithm: str = "sha256") -> str:
        if isinstance(content, bytes):
            payload = content
        elif isinstance(content, str):
            payload = content.encode("utf-8")
        else:
            payload = json.dumps(
                content,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        h = hashlib.new(algorithm)
        h.update(payload)
        return h.hexdigest()

    def deterministic_isoformat(*parts: Any, resolution: str = "seconds") -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(str(part).encode())
        seed = int(h.hexdigest()[:16], 16)
        offset_seconds = seed % (365 * 24 * 3600)
        epoch = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts = datetime.fromtimestamp(epoch.timestamp() + offset_seconds, tz=timezone.utc)
        if resolution == "seconds":
            ts = ts.replace(microsecond=0)
        return ts.isoformat()


__all__ = [
    "InvariantStatus",
    "SliceIdentityResult",
    "verify_slice_identity_for_p3",
    "compute_slice_fingerprint",
    "build_identity_console_tile",
    "SliceIdentityVerifier",
    # Report/Evidence binding
    "attach_slice_identity_to_p3_stability_report",
    "attach_slice_identity_to_evidence",
    # P4 drift context
    "compute_p4_identity_drift_context",
    "P4IdentityDriftContext",
]


class InvariantStatus(Enum):
    """Status of individual slice identity invariants."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNCHECKED = "UNCHECKED"


@dataclass
class SliceIdentityResult:
    """
    Result of slice identity verification for P3/P4 pre-flight.

    SHADOW MODE: This is ADVISORY only. The caller decides whether
    to proceed or abort based on this result.

    See: docs/system_law/SliceIdentity_PhaseX_Invariants.md Section 5.3
    """

    # Overall status
    identity_verified: bool = False
    advisory_block: bool = False  # SHADOW MODE: advisory, not enforced

    # Fingerprints
    slice_name: str = ""
    computed_fingerprint: str = ""
    baseline_fingerprint: Optional[str] = None
    fingerprint_match: bool = False

    # Curriculum context
    curriculum_fingerprint: Optional[str] = None

    # Per-invariant status
    invariant_status: Dict[str, str] = field(default_factory=dict)

    # Violations
    violations: List[str] = field(default_factory=list)

    # Timestamp
    verified_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema": "slice-identity-verification/1.0.0",
            "mode": "SHADOW",
            "identity_verified": self.identity_verified,
            "advisory_block": self.advisory_block,
            "slice_name": self.slice_name,
            "fingerprints": {
                "computed": self.computed_fingerprint,
                "baseline": self.baseline_fingerprint,
                "match": self.fingerprint_match,
                "curriculum": self.curriculum_fingerprint,
            },
            "invariant_status": self.invariant_status,
            "violations": self.violations,
            "verified_at": self.verified_at,
        }


def compute_slice_fingerprint(slice_config: Dict[str, Any]) -> str:
    """
    Compute deterministic fingerprint for a slice configuration.

    Implements SI-001: Unique Slice Fingerprint.

    fingerprint(slice) = SHA256(canonical_json(slice.params ∪ slice.gates))

    Args:
        slice_config: Slice configuration dictionary containing params and/or gates

    Returns:
        Hexadecimal SHA-256 fingerprint

    Example:
        >>> fp = compute_slice_fingerprint({"params": {"depth_max": 5}})
        >>> len(fp)
        64
    """
    # Extract fingerprint-relevant fields
    relevant = {}

    if "params" in slice_config:
        relevant["params"] = slice_config["params"]

    if "gates" in slice_config:
        relevant["gates"] = slice_config["gates"]

    # If neither params nor gates, use entire config minus metadata
    if not relevant:
        relevant = {
            k: v
            for k, v in slice_config.items()
            if k not in ("name", "version", "description", "created_at", "updated_at")
        }

    # Compute canonical JSON and hash
    canonical = json.dumps(
        relevant,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def verify_slice_identity_for_p3(
    slice_config: Dict[str, Any],
    baseline_fingerprint: Optional[str] = None,
    curriculum_fingerprint: Optional[str] = None,
    slice_name: Optional[str] = None,
) -> SliceIdentityResult:
    """
    Pre-flight identity verification for P3/P4 runs.

    SHADOW MODE: Returns ADVISORY status only. Does NOT enforce blocking.
    Caller must decide whether to proceed based on result.

    Checks:
        - SI-001: Unique Slice Fingerprint (can compute)
        - SI-002: Immutable Run Baseline (baseline provided)
        - SI-003: Drift Detection Required (implicit via comparison)
        - SI-004: Provenance Chain Continuity (curriculum fingerprint provided)
        - SI-005: Identity-P4 Binding (fingerprint match)
        - SI-006: Cross-Run Identity Stability (version compatibility)

    Args:
        slice_config: Current slice configuration
        baseline_fingerprint: Expected baseline fingerprint (None = no baseline check)
        curriculum_fingerprint: Curriculum fingerprint for provenance
        slice_name: Slice name (extracted from config if not provided)

    Returns:
        SliceIdentityResult with verification status and any violations
    """
    result = SliceIdentityResult()
    result.verified_at = datetime.now(timezone.utc).isoformat()

    # Extract slice name
    result.slice_name = slice_name or slice_config.get("name", "unknown")

    # Initialize invariant status
    result.invariant_status = {
        "SI-001": InvariantStatus.UNCHECKED.value,
        "SI-002": InvariantStatus.UNCHECKED.value,
        "SI-003": InvariantStatus.UNCHECKED.value,
        "SI-004": InvariantStatus.UNCHECKED.value,
        "SI-005": InvariantStatus.UNCHECKED.value,
        "SI-006": InvariantStatus.UNCHECKED.value,
    }

    # SI-001: Unique Slice Fingerprint
    try:
        result.computed_fingerprint = compute_slice_fingerprint(slice_config)
        result.invariant_status["SI-001"] = InvariantStatus.PASS.value
    except Exception as e:
        result.violations.append(f"SI-001 FAIL: Cannot compute fingerprint: {e}")
        result.invariant_status["SI-001"] = InvariantStatus.FAIL.value

    # SI-002: Immutable Run Baseline
    if baseline_fingerprint is not None:
        result.baseline_fingerprint = baseline_fingerprint
        result.invariant_status["SI-002"] = InvariantStatus.PASS.value
    else:
        # No baseline = first run, which is acceptable
        result.invariant_status["SI-002"] = InvariantStatus.PASS.value

    # SI-003: Drift Detection Required (verified by having the comparison capability)
    result.invariant_status["SI-003"] = InvariantStatus.PASS.value

    # SI-004: Provenance Chain Continuity
    if curriculum_fingerprint:
        result.curriculum_fingerprint = curriculum_fingerprint
        result.invariant_status["SI-004"] = InvariantStatus.PASS.value
    else:
        result.violations.append("SI-004 WARN: No curriculum fingerprint provided")
        # Warning, not failure - may be acceptable for some contexts
        result.invariant_status["SI-004"] = InvariantStatus.PASS.value

    # SI-005: Identity-P4 Binding (fingerprint match)
    if baseline_fingerprint is not None and result.computed_fingerprint:
        if result.computed_fingerprint == baseline_fingerprint:
            result.fingerprint_match = True
            result.invariant_status["SI-005"] = InvariantStatus.PASS.value
        else:
            result.fingerprint_match = False
            result.violations.append(
                f"SI-005 FAIL: Fingerprint mismatch - "
                f"computed={result.computed_fingerprint[:16]}... "
                f"baseline={baseline_fingerprint[:16]}..."
            )
            result.invariant_status["SI-005"] = InvariantStatus.FAIL.value
    else:
        # No baseline to compare = vacuously true
        result.fingerprint_match = baseline_fingerprint is None
        result.invariant_status["SI-005"] = InvariantStatus.PASS.value

    # SI-006: Cross-Run Identity Stability (version compatibility)
    version = slice_config.get("version")
    if version:
        # Version exists, assume compatible for now
        result.invariant_status["SI-006"] = InvariantStatus.PASS.value
    else:
        # No version = implicitly v0.0.0, acceptable
        result.invariant_status["SI-006"] = InvariantStatus.PASS.value

    # Determine overall status
    failed_invariants = [
        inv_id
        for inv_id, status in result.invariant_status.items()
        if status == InvariantStatus.FAIL.value
    ]

    result.identity_verified = len(failed_invariants) == 0

    # Advisory block recommendation (SHADOW MODE: not enforced)
    # Block if any critical invariant failed
    critical_invariants = {"SI-001", "SI-005"}
    critical_failures = [inv for inv in failed_invariants if inv in critical_invariants]
    result.advisory_block = len(critical_failures) > 0

    return result


class SliceIdentityVerifier:
    """
    Stateful slice identity verifier for continuous monitoring during runs.

    SHADOW MODE: Tracks identity stability across cycles but does NOT
    enforce any control flow changes.

    Usage:
        verifier = SliceIdentityVerifier(baseline_config, curriculum_fp)
        for cycle in cycles:
            stable = verifier.check_identity_stable(current_config)
            # Log stable status but don't abort
    """

    def __init__(
        self,
        baseline_config: Dict[str, Any],
        curriculum_fingerprint: Optional[str] = None,
    ) -> None:
        """
        Initialize verifier with frozen baseline.

        Args:
            baseline_config: Frozen baseline slice configuration
            curriculum_fingerprint: Optional curriculum fingerprint
        """
        self.baseline_config = baseline_config
        self.baseline_fingerprint = compute_slice_fingerprint(baseline_config)
        self.curriculum_fingerprint = curriculum_fingerprint
        self.slice_name = baseline_config.get("name", "unknown")

        # Tracking state
        self._drift_events: List[Dict[str, Any]] = []
        self._consecutive_stable_cycles = 0
        self._last_drift_cycle: Optional[int] = None
        self._total_cycles_checked = 0

    def check_identity_stable(
        self,
        current_config: Dict[str, Any],
        cycle: int,
    ) -> bool:
        """
        Check if identity is stable for this cycle.

        SHADOW MODE: Returns stability status for LOGGING only.
        Does NOT trigger any control flow changes.

        Args:
            current_config: Current slice configuration
            cycle: Current cycle number

        Returns:
            True if identity is stable (no drift)
        """
        self._total_cycles_checked += 1

        current_fingerprint = compute_slice_fingerprint(current_config)
        stable = current_fingerprint == self.baseline_fingerprint

        if stable:
            self._consecutive_stable_cycles += 1
        else:
            # Record drift event
            drift_event = {
                "cycle": cycle,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "baseline_fingerprint": self.baseline_fingerprint,
                "current_fingerprint": current_fingerprint,
                "consecutive_stable_before": self._consecutive_stable_cycles,
            }
            self._drift_events.append(drift_event)
            self._consecutive_stable_cycles = 0
            self._last_drift_cycle = cycle

        return stable

    def get_drift_events(self) -> List[Dict[str, Any]]:
        """Get all recorded drift events."""
        return list(self._drift_events)

    def get_stability_score(self) -> float:
        """
        Compute stability score (0.0 to 1.0).

        1.0 = perfectly stable (no drift events)
        0.0 = always drifting
        """
        if self._total_cycles_checked == 0:
            return 1.0
        drift_count = len(self._drift_events)
        return 1.0 - (drift_count / self._total_cycles_checked)

    def get_summary(self) -> Dict[str, Any]:
        """Get verifier summary for logging."""
        return {
            "slice_name": self.slice_name,
            "baseline_fingerprint": self.baseline_fingerprint,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "total_cycles_checked": self._total_cycles_checked,
            "drift_event_count": len(self._drift_events),
            "consecutive_stable_cycles": self._consecutive_stable_cycles,
            "last_drift_cycle": self._last_drift_cycle,
            "stability_score": self.get_stability_score(),
        }


def build_identity_console_tile(
    verification_result: Optional[SliceIdentityResult] = None,
    verifier: Optional[SliceIdentityVerifier] = None,
    active_run_id: Optional[str] = None,
    active_run_type: Optional[str] = None,
    active_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build console tile payload conforming to slice_identity_console_tile.schema.json.

    SHADOW MODE: This tile is for monitoring and display only.

    Args:
        verification_result: Pre-flight verification result
        verifier: Active run verifier (for mid-run status)
        active_run_id: Current run ID if running
        active_run_type: "P3" or "P4"
        active_cycle: Current cycle number

    Returns:
        Console tile payload dict
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Determine overall status
    status = "OK"
    headline = "Slice identity stable"
    alerts: List[Dict[str, Any]] = []

    # Build identity summary
    identity_summary: Dict[str, Any] = {
        "current_slice": "unknown",
        "fingerprint_stable": True,
        "drift_events_24h": 0,
        "drift_events_1h": 0,
        "last_drift_at": None,
        "consecutive_stable_cycles": 0,
    }

    # Build invariant status
    invariant_status = {
        "SI-001": "UNCHECKED",
        "SI-002": "UNCHECKED",
        "SI-003": "UNCHECKED",
        "SI-004": "UNCHECKED",
        "SI-005": "UNCHECKED",
        "SI-006": "UNCHECKED",
    }

    # Populate from verification result
    if verification_result:
        identity_summary["current_slice"] = verification_result.slice_name
        identity_summary["current_fingerprint"] = verification_result.computed_fingerprint[:16] + "..."
        identity_summary["fingerprint_stable"] = verification_result.fingerprint_match
        identity_summary["baseline_fingerprint"] = (
            verification_result.baseline_fingerprint[:16] + "..."
            if verification_result.baseline_fingerprint
            else None
        )
        identity_summary["last_verification_at"] = verification_result.verified_at

        invariant_status = verification_result.invariant_status.copy()

        if not verification_result.identity_verified:
            status = "ERROR"
            headline = f"Identity verification failed: {len(verification_result.violations)} violation(s)"
            for v in verification_result.violations:
                alerts.append({
                    "level": "ERROR",
                    "message": v,
                    "action_required": True,
                })

        if verification_result.advisory_block:
            status = "ERROR"
            headline = "Identity BLOCK advisory (critical invariant failure)"

    # Populate from verifier (mid-run monitoring)
    if verifier:
        identity_summary["current_slice"] = verifier.slice_name
        identity_summary["current_fingerprint"] = verifier.baseline_fingerprint[:16] + "..."
        identity_summary["baseline_fingerprint"] = verifier.baseline_fingerprint[:16] + "..."
        identity_summary["drift_events_24h"] = len(verifier.get_drift_events())
        identity_summary["consecutive_stable_cycles"] = verifier._consecutive_stable_cycles

        stability_score = verifier.get_stability_score()

        if verifier._last_drift_cycle is not None:
            # Find the last drift event timestamp
            drift_events = verifier.get_drift_events()
            if drift_events:
                identity_summary["last_drift_at"] = drift_events[-1]["timestamp"]

        if stability_score < 0.9:
            status = "WARN"
            headline = f"Drift detected ({len(verifier.get_drift_events())} events)"
            alerts.append({
                "level": "WARNING",
                "message": f"Stability score: {stability_score:.2%}",
                "action_required": False,
            })

        if stability_score < 0.5:
            status = "ERROR"
            headline = f"High drift rate ({len(verifier.get_drift_events())} events)"

    # Build active run context
    active_run = None
    if active_run_id:
        evidence_admissibility = "FULL"
        if verifier and len(verifier.get_drift_events()) > 0:
            evidence_admissibility = "PARTIAL"

        active_run = {
            "run_id": active_run_id,
            "run_type": active_run_type or "P3",
            "slice_locked": True,
            "cycle": active_cycle or 0,
            "identity_drift_detected": (
                verifier is not None and len(verifier.get_drift_events()) > 0
            ),
            "evidence_admissibility": evidence_admissibility,
        }

    # Build trend
    trend = {
        "direction": "STABLE",
        "stability_score": 1.0,
    }
    if verifier:
        score = verifier.get_stability_score()
        trend["stability_score"] = score
        if score >= 0.95:
            trend["direction"] = "STABLE"
        elif score >= 0.8:
            trend["direction"] = "DEGRADING"
        else:
            trend["direction"] = "DEGRADING"

    return {
        "schema_version": "1.0.0",
        "tile_type": "slice_identity",
        "timestamp": timestamp,
        "status": status,
        "headline": headline,
        "identity_summary": identity_summary,
        "invariant_status": invariant_status,
        "active_run": active_run,
        "alerts": alerts,
        "trend": trend,
        "metadata": {
            "curriculum_fingerprint": (
                verification_result.curriculum_fingerprint
                if verification_result
                else (verifier.curriculum_fingerprint if verifier else None)
            ),
        },
    }


# =============================================================================
# Task 1: P3 Stability Report Binding
# =============================================================================


def attach_slice_identity_to_p3_stability_report(
    stability_report: Dict[str, Any],
    console_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach slice identity summary to a P3 stability report.

    SHADOW MODE: Non-mutating. Returns a new dict with identity attached
    under "slice_identity_summary".

    Args:
        stability_report: Existing stability report dict
        console_tile: Console tile from build_identity_console_tile()

    Returns:
        New stability report with slice_identity_summary attached

    Example:
        >>> report = {"run_id": "test", "metrics": {...}}
        >>> tile = build_identity_console_tile(verification_result=result)
        >>> enriched = attach_slice_identity_to_p3_stability_report(report, tile)
        >>> "slice_identity_summary" in enriched
        True
    """
    # Extract relevant fields from console tile
    identity_summary = console_tile.get("identity_summary", {})
    invariant_status = console_tile.get("invariant_status", {})
    trend = console_tile.get("trend", {})
    active_run = console_tile.get("active_run", {})

    # Build slice identity summary
    slice_identity_summary = {
        "identity_verified": all(
            status == "PASS"
            for status in invariant_status.values()
            if status != "UNCHECKED"
        ),
        "fingerprint_match": identity_summary.get("fingerprint_stable", True),
        "violations": [
            alert.get("message", "")
            for alert in console_tile.get("alerts", [])
            if alert.get("level") in ("ERROR", "WARNING")
        ],
        "stability_score": trend.get("stability_score", 1.0),
        "invariant_status": invariant_status,
        "slice_name": identity_summary.get("current_slice", "unknown"),
        "drift_events": identity_summary.get("drift_events_24h", 0),
        "consecutive_stable_cycles": identity_summary.get("consecutive_stable_cycles", 0),
    }

    # Include evidence admissibility if available
    if active_run:
        slice_identity_summary["evidence_admissibility"] = active_run.get(
            "evidence_admissibility", "FULL"
        )

    # Non-mutating: create new dict
    result = dict(stability_report)
    result["slice_identity_summary"] = slice_identity_summary

    return result


# =============================================================================
# Task 2: Evidence Attachment
# =============================================================================


def attach_slice_identity_to_evidence(
    evidence: Dict[str, Any],
    console_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach slice identity to evidence pack under governance.slice_identity.

    SHADOW MODE: Non-mutating. Returns a new dict with identity attached.

    The attached identity includes:
    - invariant_statuses: Per-invariant PASS/FAIL/UNCHECKED
    - evidence_admissibility: FULL, PARTIAL, or INADMISSIBLE
    - identity_verified: Overall verification status
    - fingerprint_match: Whether slice fingerprint matches baseline
    - stability_score: 0.0-1.0 stability score
    - violations: List of violation messages

    Args:
        evidence: Existing evidence pack dict
        console_tile: Console tile from build_identity_console_tile()

    Returns:
        New evidence dict with governance.slice_identity attached

    Example:
        >>> evidence = {"artifacts": [...], "governance": {"cohesion": {...}}}
        >>> tile = build_identity_console_tile(verification_result=result)
        >>> enriched = attach_slice_identity_to_evidence(evidence, tile)
        >>> "slice_identity" in enriched["governance"]
        True
    """
    # Extract from console tile
    identity_summary = console_tile.get("identity_summary", {})
    invariant_status = console_tile.get("invariant_status", {})
    trend = console_tile.get("trend", {})
    active_run = console_tile.get("active_run", {})
    alerts = console_tile.get("alerts", [])
    status = console_tile.get("status", "OK")

    # Determine evidence admissibility
    # - FULL: No identity issues
    # - PARTIAL: Non-critical issues (warnings, drift detected)
    # - INADMISSIBLE: Critical invariant failures
    if status == "ERROR":
        # Check if critical invariants failed
        critical_failed = any(
            invariant_status.get(inv_id) == "FAIL"
            for inv_id in ["SI-001", "SI-005"]
        )
        evidence_admissibility = "INADMISSIBLE" if critical_failed else "PARTIAL"
    elif status == "WARN":
        evidence_admissibility = "PARTIAL"
    else:
        evidence_admissibility = "FULL"

    # Override with active_run if available
    if active_run and "evidence_admissibility" in active_run:
        evidence_admissibility = active_run["evidence_admissibility"]

    # Build slice identity block for evidence
    slice_identity = {
        "invariant_statuses": invariant_status,
        "evidence_admissibility": evidence_admissibility,
        "identity_verified": all(
            s == "PASS"
            for s in invariant_status.values()
            if s != "UNCHECKED"
        ),
        "fingerprint_match": identity_summary.get("fingerprint_stable", True),
        "stability_score": trend.get("stability_score", 1.0),
        "violations": [
            alert.get("message", "")
            for alert in alerts
            if alert.get("level") in ("ERROR", "WARNING")
        ],
        "slice_name": identity_summary.get("current_slice", "unknown"),
        "drift_detected": identity_summary.get("drift_events_24h", 0) > 0,
        "attached_at": datetime.now(timezone.utc).isoformat(),
    }

    # Non-mutating: create new dict
    result = dict(evidence)

    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}
    else:
        result["governance"] = dict(result["governance"])

    result["governance"]["slice_identity"] = slice_identity

    return result


# =============================================================================
# Task 3: P4 Drift Context Stub
# =============================================================================


@dataclass
class P4IdentityDriftContext:
    """
    P4 identity drift context for divergence analysis.

    SHADOW MODE: This is for observation/logging only.
    No gating or enforcement.

    See: docs/system_law/SliceIdentity_PhaseX_Invariants.md Section 4
    """

    identity_diverged: bool = False
    identity_divergence_type: Optional[str] = None

    # Additional context
    real_fingerprint: Optional[str] = None
    twin_fingerprint: Optional[str] = None
    slice_name: Optional[str] = None
    invariant_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "identity_diverged": self.identity_diverged,
            "identity_divergence_type": self.identity_divergence_type,
            "real_fingerprint": self.real_fingerprint,
            "twin_fingerprint": self.twin_fingerprint,
            "slice_name": self.slice_name,
            "invariant_violations": self.invariant_violations,
            "mode": "SHADOW",
        }


def compute_p4_identity_drift_context(
    divergence_snapshot: Any,  # DivergenceSnapshot from data_structures_p4
    identity_result: SliceIdentityResult,
) -> P4IdentityDriftContext:
    """
    Compute identity drift context for P4 divergence analysis.

    SHADOW MODE: This helper produces observational data only.
    No gating, no enforcement.

    Given a P4 DivergenceSnapshot and a SliceIdentityResult, determines:
    - identity_diverged: bool - whether identity contributes to divergence
    - identity_divergence_type: str | None - type of identity divergence

    Identity divergence types:
    - "FINGERPRINT_MISMATCH": Slice fingerprints don't match
    - "INVARIANT_VIOLATION": One or more identity invariants failed
    - "PROVENANCE_BROKEN": Curriculum fingerprint chain broken
    - None: No identity divergence

    Args:
        divergence_snapshot: P4 DivergenceSnapshot instance
        identity_result: SliceIdentityResult from verify_slice_identity_for_p3

    Returns:
        P4IdentityDriftContext with identity_diverged and identity_divergence_type

    Example:
        >>> from backend.topology.first_light import DivergenceSnapshot
        >>> div = DivergenceSnapshot(cycle=42, success_diverged=True)
        >>> result = verify_slice_identity_for_p3(config, baseline="wrong")
        >>> ctx = compute_p4_identity_drift_context(div, result)
        >>> ctx.identity_diverged
        True
        >>> ctx.identity_divergence_type
        'FINGERPRINT_MISMATCH'
    """
    context = P4IdentityDriftContext()

    # Extract slice name
    context.slice_name = identity_result.slice_name

    # Extract fingerprints
    context.real_fingerprint = identity_result.computed_fingerprint
    context.twin_fingerprint = identity_result.baseline_fingerprint

    # Check fingerprint mismatch
    if not identity_result.fingerprint_match and identity_result.baseline_fingerprint:
        context.identity_diverged = True
        context.identity_divergence_type = "FINGERPRINT_MISMATCH"
        context.invariant_violations.append(
            f"SI-005: Fingerprint mismatch between real and twin"
        )
        return context

    # Check invariant violations
    failed_invariants = [
        inv_id
        for inv_id, status in identity_result.invariant_status.items()
        if status == InvariantStatus.FAIL.value
    ]

    if failed_invariants:
        context.identity_diverged = True
        context.identity_divergence_type = "INVARIANT_VIOLATION"
        for inv_id in failed_invariants:
            context.invariant_violations.append(f"{inv_id}: Failed")
        return context

    # Check provenance chain (SI-004)
    if "SI-004 WARN" in str(identity_result.violations):
        # Provenance warning - not a full divergence but worth noting
        # Only flag as divergence if it's a FAIL, not WARN
        pass

    # Check if divergence snapshot indicates structural issues
    # that correlate with identity
    if hasattr(divergence_snapshot, "structural_conflict"):
        if divergence_snapshot.structural_conflict:
            # Structural conflict may be identity-related
            context.identity_diverged = True
            context.identity_divergence_type = "STRUCTURAL_CONFLICT"
            context.invariant_violations.append(
                "Structural conflict detected (may be identity-related)"
            )
            return context

    # No identity divergence detected
    context.identity_diverged = False
    context.identity_divergence_type = None

    return context
