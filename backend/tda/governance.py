"""
TDA Governance Module — Phase III/IV Hard Gate Integration

Operation CORTEX: Phase III + Phase IV Operator Guardrails
===========================================================

This module provides governance-level functions for TDA integration:
1. Pipeline hash computation for attestation binding
2. Global health summarization for governance layer
3. Drift detection and reporting
4. [Phase IV] Golden set calibration regression
5. [Phase IV] Governance alignment evaluation
6. [Phase IV] Exception window management
7. [Phase IV] Evidence tile building

Usage:
    from backend.tda.governance import (
        compute_tda_pipeline_hash,
        summarize_tda_for_global_health,
        generate_drift_report,
        # Phase IV
        evaluate_hard_gate_calibration,
        evaluate_tda_governance_alignment,
        build_tda_hard_gate_evidence_tile,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.tda.runtime_monitor import TDAMonitorConfig, TDAMonitorResult
    from backend.tda.reference_profile import ReferenceTDAProfile

logger = logging.getLogger(__name__)

TDA_GOVERNANCE_SCHEMA_VERSION = "tda-governance-1.0.0"
TDA_DRIFT_REPORT_SCHEMA_VERSION = "tda-drift-report-v1"
TDA_CALIBRATION_SCHEMA_VERSION = "tda-calibration-1.0.0"
TDA_EVIDENCE_TILE_SCHEMA_VERSION = "tda-evidence-tile-1.0.0"


# ============================================================================
# Phase IV: Hard Gate Mode Enumeration
# ============================================================================

class TDAHardGateMode(Enum):
    """
    Hard gate operational mode.

    Phase IV introduces fine-grained control over hard gate behavior:
    - OFF: TDA disabled entirely
    - SHADOW: Phase I behavior (logging only)
    - DRY_RUN: Log what would be blocked, but don't block
    - HARD: Full Phase III enforcement
    """
    OFF = "off"
    SHADOW = "shadow"
    DRY_RUN = "dry_run"
    HARD = "hard"

    @classmethod
    def from_env(cls) -> "TDAHardGateMode":
        """Read mode from environment variable."""
        mode_str = os.getenv("MATHLEDGER_TDA_HARD_GATE_MODE", "hard")
        try:
            return cls(mode_str.lower())
        except ValueError:
            logger.warning(f"[TDA] Unknown hard gate mode '{mode_str}', defaulting to HARD")
            return cls.HARD


# ============================================================================
# Phase IV: Labeled Golden Run for Calibration
# ============================================================================

@dataclass
class LabeledTDAResult:
    """
    TDA result with expected label for calibration testing.

    The expected_label indicates what the result SHOULD be classified as
    based on ground truth or expert annotation.
    """
    hss: float
    sns: float
    pcs: float
    drs: float
    expected_label: Literal["OK", "BLOCK"]
    # Optional: actual TDAMonitorResult for richer testing
    actual_result: Optional["TDAMonitorResult"] = None

    def would_be_blocked(self, block_threshold: float = 0.2) -> bool:
        """Check if this result would be blocked at given threshold."""
        return self.hss < block_threshold


@dataclass
class CalibrationResult:
    """Result of hard gate calibration evaluation."""
    schema_version: str
    n_runs: int
    n_expected_ok: int
    n_expected_block: int
    actual_ok: int
    actual_block: int
    false_block_count: int  # Expected OK but blocked
    false_pass_count: int   # Expected BLOCK but allowed
    false_block_rate: float
    false_pass_rate: float
    calibration_status: Literal["OK", "DRIFTING", "BROKEN"]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Pipeline Hash Computation
# ============================================================================

def compute_tda_pipeline_hash(
    config: "TDAMonitorConfig",
    profiles: Dict[str, "ReferenceTDAProfile"],
) -> str:
    """
    Compute deterministic hash of TDA pipeline configuration.

    This hash provides cryptographic binding between:
    - TDA module configuration (thresholds, weights, mode)
    - Reference profile content (per-slice calibration)

    Used for:
    - Attestation binding (Phase III)
    - Drift detection baseline
    - Audit trail

    Args:
        config: TDAMonitorConfig with thresholds and parameters.
        profiles: Dict mapping slice_name to ReferenceTDAProfile.

    Returns:
        64-character hex SHA-256 hash of pipeline configuration.
    """
    payload = {
        "schema_version": "tda-pipeline-1.0.0",
        "config": {
            "hss_block_threshold": config.hss_block_threshold,
            "hss_warn_threshold": config.hss_warn_threshold,
            "mode": config.mode.value,
            "lifetime_threshold": config.lifetime_threshold,
            "deviation_max": config.deviation_max,
            "max_simplex_dim": config.max_simplex_dim,
            "max_homology_dim": config.max_homology_dim,
            "fail_open": config.fail_open,
        },
        "profiles": {},
    }

    # Add profile summaries (sorted for determinism)
    for name in sorted(profiles.keys()):
        profile = profiles[name]
        payload["profiles"][name] = {
            "version": getattr(profile, 'version', '1.0.0'),
            "n_ref": getattr(profile, 'n_ref', 0),
            "mean_betti_0": getattr(profile, 'mean_betti_0', 0.0),
            "mean_betti_1": getattr(profile, 'mean_betti_1', 0.0),
        }

    # Canonical JSON serialization
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================================
# Global Health Summarization
# ============================================================================

@dataclass
class GovernanceSummary:
    """Aggregate TDA governance metrics."""
    cycle_count: int
    block_count: int
    warn_count: int
    ok_count: int
    block_rate: float
    mean_hss: float
    hss_trend: float  # Slope of HSS over cycles
    structural_health: float  # [0, 1] composite
    governance_signal: str  # "HEALTHY", "DEGRADED", "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_tda_for_global_health(
    tda_results: List["TDAMonitorResult"],
    config: "TDAMonitorConfig",
) -> Dict[str, Any]:
    """
    Aggregate TDA results into global health metrics.

    Used by governance layer for system-wide health assessment.
    This function is called at session end or periodically during
    long-running experiments.

    Args:
        tda_results: List of TDAMonitorResult from evaluations.
        config: TDAMonitorConfig for threshold context.

    Returns:
        Dictionary with governance metrics:
        {
            "cycle_count": int,
            "block_count": int,
            "warn_count": int,
            "ok_count": int,
            "block_rate": float,
            "mean_hss": float,
            "hss_trend": float,
            "structural_health": float,
            "governance_signal": str,
        }
    """
    if not tda_results:
        return {
            "cycle_count": 0,
            "block_count": 0,
            "warn_count": 0,
            "ok_count": 0,
            "block_rate": 0.0,
            "mean_hss": 0.0,
            "hss_trend": 0.0,
            "structural_health": 1.0,
            "governance_signal": "HEALTHY",
        }

    # Extract values
    hss_values = [r.hss for r in tda_results]
    block_count = sum(1 for r in tda_results if r.block)
    warn_count = sum(1 for r in tda_results if r.warn and not r.block)
    ok_count = len(tda_results) - block_count - warn_count

    # Compute HSS trend (linear regression slope)
    x = np.arange(len(hss_values))
    if len(hss_values) > 1:
        slope = float(np.polyfit(x, hss_values, 1)[0])
    else:
        slope = 0.0

    # Compute aggregate metrics
    mean_hss = float(np.mean(hss_values))
    block_rate = block_count / len(tda_results)

    # Structural health formula:
    # Penalize high block rate and low mean HSS
    # health = (1 - block_rate) * mean_hss
    structural_health = (1 - block_rate) * mean_hss

    # Determine governance signal
    if block_rate > 0.2 or mean_hss < 0.3:
        governance_signal = "CRITICAL"
    elif block_rate > 0.1 or mean_hss < 0.5:
        governance_signal = "DEGRADED"
    else:
        governance_signal = "HEALTHY"

    return {
        "cycle_count": len(tda_results),
        "block_count": block_count,
        "warn_count": warn_count,
        "ok_count": ok_count,
        "block_rate": block_rate,
        "mean_hss": mean_hss,
        "hss_trend": slope,
        "structural_health": structural_health,
        "governance_signal": governance_signal,
    }


# ============================================================================
# Drift Detection and Reporting
# ============================================================================

@dataclass
class DriftMetrics:
    """Drift detection metrics between baseline and current period."""
    hss_delta: float
    block_rate_delta: float
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool
    drift_severity: str  # "none", "minor", "major", "critical"


def compute_drift_metrics(
    baseline_hss: List[float],
    current_hss: List[float],
    significance_level: float = 0.05,
) -> DriftMetrics:
    """
    Compute drift metrics between baseline and current HSS distributions.

    Uses Kolmogorov-Smirnov test for distribution comparison.

    Args:
        baseline_hss: HSS values from baseline period.
        current_hss: HSS values from current period.
        significance_level: p-value threshold for drift detection.

    Returns:
        DriftMetrics with KS statistics and severity classification.
    """
    from scipy import stats

    if not baseline_hss or not current_hss:
        return DriftMetrics(
            hss_delta=0.0,
            block_rate_delta=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            drift_detected=False,
            drift_severity="none",
        )

    # Compute deltas
    baseline_mean = np.mean(baseline_hss)
    current_mean = np.mean(current_hss)
    hss_delta = current_mean - baseline_mean

    # Block rate delta (assuming threshold of 0.2)
    baseline_block_rate = sum(1 for h in baseline_hss if h < 0.2) / len(baseline_hss)
    current_block_rate = sum(1 for h in current_hss if h < 0.2) / len(current_hss)
    block_rate_delta = current_block_rate - baseline_block_rate

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(baseline_hss, current_hss)

    # Determine drift detection and severity
    drift_detected = ks_pvalue < significance_level

    if not drift_detected:
        drift_severity = "none"
    elif abs(hss_delta) < 0.1 and abs(block_rate_delta) < 0.05:
        drift_severity = "minor"
    elif abs(hss_delta) < 0.2 and abs(block_rate_delta) < 0.1:
        drift_severity = "major"
    else:
        drift_severity = "critical"

    return DriftMetrics(
        hss_delta=float(hss_delta),
        block_rate_delta=float(block_rate_delta),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        drift_detected=drift_detected,
        drift_severity=drift_severity,
    )


def generate_drift_report(
    baseline_results: List["TDAMonitorResult"],
    current_results: List["TDAMonitorResult"],
    pipeline_hash: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate TDA drift report comparing baseline and current periods.

    Args:
        baseline_results: TDA results from baseline period.
        current_results: TDA results from current period.
        pipeline_hash: TDA pipeline configuration hash.
        output_path: Optional path to write JSON report.

    Returns:
        Drift report dictionary conforming to tda-drift-report-v1 schema.
    """
    baseline_hss = [r.hss for r in baseline_results] if baseline_results else []
    current_hss = [r.hss for r in current_results] if current_results else []

    drift_metrics = compute_drift_metrics(baseline_hss, current_hss)

    # Compute period summaries
    baseline_summary = {
        "start_cycle": 0,
        "end_cycle": len(baseline_results) - 1 if baseline_results else 0,
        "mean_hss": float(np.mean(baseline_hss)) if baseline_hss else 0.0,
        "block_rate": sum(1 for h in baseline_hss if h < 0.2) / len(baseline_hss) if baseline_hss else 0.0,
    }

    current_summary = {
        "start_cycle": len(baseline_results) if baseline_results else 0,
        "end_cycle": len(baseline_results) + len(current_results) - 1 if current_results else 0,
        "mean_hss": float(np.mean(current_hss)) if current_hss else 0.0,
        "block_rate": sum(1 for h in current_hss if h < 0.2) / len(current_hss) if current_hss else 0.0,
    }

    # Generate recommendations
    recommendations = []
    if drift_metrics.drift_severity == "critical":
        recommendations.append("CRITICAL: Consider immediate rollback to Phase II")
        recommendations.append("Investigate HSS distribution shift root cause")
        recommendations.append("Review reference profile calibration")
    elif drift_metrics.drift_severity == "major":
        recommendations.append("WARNING: Monitor closely for further degradation")
        recommendations.append("Consider recalibrating reference profiles")
    elif drift_metrics.drift_severity == "minor":
        recommendations.append("INFO: Minor drift detected, continue monitoring")

    if drift_metrics.block_rate_delta > 0.1:
        recommendations.append("Block rate increased significantly - check input quality")
    if drift_metrics.hss_delta < -0.15:
        recommendations.append("Mean HSS decreased - investigate structural changes")

    report = {
        "schema_version": TDA_DRIFT_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_hash": pipeline_hash,
        "baseline_period": baseline_summary,
        "current_period": current_summary,
        "drift_metrics": {
            "hss_delta": drift_metrics.hss_delta,
            "block_rate_delta": drift_metrics.block_rate_delta,
            "ks_statistic": drift_metrics.ks_statistic,
            "ks_pvalue": drift_metrics.ks_pvalue,
            "drift_detected": drift_metrics.drift_detected,
            "drift_severity": drift_metrics.drift_severity,
        },
        "recommendations": recommendations,
    }

    # Write to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[TDA] Drift report written to {output_path}")

    return report


# ============================================================================
# Attestation Integration
# ============================================================================

def extend_attestation_with_tda(
    attestation_metadata: Dict[str, Any],
    tda_results: List["TDAMonitorResult"],
    config: "TDAMonitorConfig",
    pipeline_hash: str,
) -> Dict[str, Any]:
    """
    Extend attestation metadata with Phase III TDA governance fields.

    Args:
        attestation_metadata: Existing attestation metadata dict.
        tda_results: TDA evaluation results from session.
        config: TDA monitor configuration.
        pipeline_hash: Computed pipeline hash.

    Returns:
        Extended attestation metadata with tda_governance section.
    """
    governance_summary = summarize_tda_for_global_health(tda_results, config)

    tda_governance = {
        "phase": "III",
        "mode": config.mode.value,
        "pipeline_hash": pipeline_hash,
        "schema_version": TDA_GOVERNANCE_SCHEMA_VERSION,
        "summary": governance_summary,
        "thresholds": {
            "hss_block": config.hss_block_threshold,
            "hss_warn": config.hss_warn_threshold,
        },
    }

    attestation_metadata["tda_governance"] = tda_governance
    return attestation_metadata


# ============================================================================
# Phase IV: Golden Set Calibration
# ============================================================================

def evaluate_hard_gate_calibration(
    golden_runs: Sequence[LabeledTDAResult],
    block_threshold: float = 0.2,
    false_block_threshold_ok: float = 0.05,
    false_block_threshold_drifting: float = 0.15,
    false_pass_threshold_ok: float = 0.05,
    false_pass_threshold_drifting: float = 0.15,
) -> CalibrationResult:
    """
    Evaluate hard gate calibration against a labeled golden set.

    This function compares expected labels against what the hard gate
    would decide, computing false block and false pass rates.

    Args:
        golden_runs: Sequence of LabeledTDAResult with expected labels.
        block_threshold: HSS threshold for BLOCK decision.
        false_block_threshold_ok: Max false block rate for OK status.
        false_block_threshold_drifting: Max false block rate for DRIFTING.
        false_pass_threshold_ok: Max false pass rate for OK status.
        false_pass_threshold_drifting: Max false pass rate for DRIFTING.

    Returns:
        CalibrationResult with:
        - n_runs: Total number of runs evaluated
        - n_expected_ok: Runs labeled as OK
        - n_expected_block: Runs labeled as BLOCK
        - actual_ok: Runs that would be allowed
        - actual_block: Runs that would be blocked
        - false_block_rate: Expected OK but blocked
        - false_pass_rate: Expected BLOCK but allowed
        - calibration_status: "OK", "DRIFTING", or "BROKEN"
        - notes: Explanatory notes

    Example:
        >>> golden = [
        ...     LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
        ...     LabeledTDAResult(hss=0.1, sns=0.2, pcs=0.1, drs=0.5, expected_label="BLOCK"),
        ... ]
        >>> result = evaluate_hard_gate_calibration(golden)
        >>> assert result.calibration_status == "OK"
    """
    if not golden_runs:
        return CalibrationResult(
            schema_version=TDA_CALIBRATION_SCHEMA_VERSION,
            n_runs=0,
            n_expected_ok=0,
            n_expected_block=0,
            actual_ok=0,
            actual_block=0,
            false_block_count=0,
            false_pass_count=0,
            false_block_rate=0.0,
            false_pass_rate=0.0,
            calibration_status="OK",
            notes=["No golden runs provided"],
        )

    n_expected_ok = sum(1 for r in golden_runs if r.expected_label == "OK")
    n_expected_block = sum(1 for r in golden_runs if r.expected_label == "BLOCK")

    # Compute what the hard gate would decide
    false_block_count = 0  # Expected OK but would be blocked
    false_pass_count = 0   # Expected BLOCK but would be allowed

    actual_block = 0
    actual_ok = 0

    for run in golden_runs:
        would_block = run.would_be_blocked(block_threshold)

        if would_block:
            actual_block += 1
            if run.expected_label == "OK":
                false_block_count += 1
        else:
            actual_ok += 1
            if run.expected_label == "BLOCK":
                false_pass_count += 1

    # Compute rates
    false_block_rate = false_block_count / n_expected_ok if n_expected_ok > 0 else 0.0
    false_pass_rate = false_pass_count / n_expected_block if n_expected_block > 0 else 0.0

    # Determine calibration status
    notes = []

    if false_block_rate <= false_block_threshold_ok and false_pass_rate <= false_pass_threshold_ok:
        calibration_status = "OK"
        notes.append(f"Calibration within acceptable bounds (false_block={false_block_rate:.1%}, false_pass={false_pass_rate:.1%})")
    elif false_block_rate <= false_block_threshold_drifting and false_pass_rate <= false_pass_threshold_drifting:
        calibration_status = "DRIFTING"
        notes.append(f"Calibration drifting (false_block={false_block_rate:.1%}, false_pass={false_pass_rate:.1%})")
        if false_block_rate > false_block_threshold_ok:
            notes.append(f"False block rate ({false_block_rate:.1%}) above OK threshold ({false_block_threshold_ok:.1%})")
        if false_pass_rate > false_pass_threshold_ok:
            notes.append(f"False pass rate ({false_pass_rate:.1%}) above OK threshold ({false_pass_threshold_ok:.1%})")
    else:
        calibration_status = "BROKEN"
        notes.append(f"Calibration broken (false_block={false_block_rate:.1%}, false_pass={false_pass_rate:.1%})")
        if false_block_rate > false_block_threshold_drifting:
            notes.append(f"CRITICAL: False block rate ({false_block_rate:.1%}) exceeds DRIFTING threshold ({false_block_threshold_drifting:.1%})")
        if false_pass_rate > false_pass_threshold_drifting:
            notes.append(f"CRITICAL: False pass rate ({false_pass_rate:.1%}) exceeds DRIFTING threshold ({false_pass_threshold_drifting:.1%})")

    return CalibrationResult(
        schema_version=TDA_CALIBRATION_SCHEMA_VERSION,
        n_runs=len(golden_runs),
        n_expected_ok=n_expected_ok,
        n_expected_block=n_expected_block,
        actual_ok=actual_ok,
        actual_block=actual_block,
        false_block_count=false_block_count,
        false_pass_count=false_pass_count,
        false_block_rate=false_block_rate,
        false_pass_rate=false_pass_rate,
        calibration_status=calibration_status,
        notes=notes,
    )


# ============================================================================
# Phase IV: Governance Alignment Evaluation
# ============================================================================

@dataclass
class GovernanceAlignmentResult:
    """Result of TDA vs global health alignment evaluation."""
    alignment_status: Literal["ALIGNED", "TENSION", "DIVERGENT"]
    tda_block_rate: float
    global_ok_fraction: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_tda_governance_alignment(
    tda_summary: Dict[str, Any],
    global_health_snapshot: Dict[str, Any],
    tension_block_rate_threshold: float = 0.1,
    divergent_block_rate_threshold: float = 0.2,
    global_ok_threshold: float = 0.8,
) -> GovernanceAlignmentResult:
    """
    Evaluate alignment between TDA hard gate and global health signals.

    Detects when TDA is blocking while other system layers report OK,
    indicating potential over-blocking or miscalibration.

    Args:
        tda_summary: Output from summarize_tda_for_global_health().
        global_health_snapshot: Dict with other layer health signals:
            - preflight_ok: bool or fraction
            - bundle_ok: bool or fraction
            - replay_ok: bool or fraction
            (or a simple "global_ok_fraction" float)
        tension_block_rate_threshold: Block rate above which we check alignment.
        divergent_block_rate_threshold: Block rate indicating divergence.
        global_ok_threshold: Fraction of global layers reporting OK for divergence.

    Returns:
        GovernanceAlignmentResult with:
        - alignment_status: "ALIGNED", "TENSION", or "DIVERGENT"
        - tda_block_rate: TDA block rate
        - global_ok_fraction: Fraction of other layers reporting OK
        - notes: Explanatory notes

    Semantics:
        - ALIGNED: TDA blocking AND other layers also reporting issues
        - TENSION: Moderate TDA blocking while others mostly OK
        - DIVERGENT: High TDA blocking while other layers report OK
    """
    tda_block_rate = tda_summary.get("block_rate", 0.0)

    # Compute global OK fraction from snapshot
    if "global_ok_fraction" in global_health_snapshot:
        global_ok_fraction = global_health_snapshot["global_ok_fraction"]
    else:
        # Compute from individual layer signals
        layers = ["preflight_ok", "bundle_ok", "replay_ok", "attestation_ok"]
        ok_count = 0
        total_count = 0
        for layer in layers:
            if layer in global_health_snapshot:
                total_count += 1
                val = global_health_snapshot[layer]
                if isinstance(val, bool):
                    ok_count += 1 if val else 0
                elif isinstance(val, (int, float)):
                    ok_count += val
        global_ok_fraction = ok_count / total_count if total_count > 0 else 1.0

    notes = []

    # Determine alignment status
    if tda_block_rate <= tension_block_rate_threshold:
        # Low block rate - always aligned
        alignment_status = "ALIGNED"
        notes.append(f"TDA block rate ({tda_block_rate:.1%}) within normal range")
    elif tda_block_rate > divergent_block_rate_threshold and global_ok_fraction >= global_ok_threshold:
        # High TDA blocking but other layers OK - divergent
        alignment_status = "DIVERGENT"
        notes.append(f"TDA block rate ({tda_block_rate:.1%}) high while global health OK ({global_ok_fraction:.1%})")
        notes.append("Consider reviewing TDA thresholds or reference profiles")
        notes.append("Exception window may be appropriate")
    elif tda_block_rate > tension_block_rate_threshold:
        # Moderate blocking - check global health
        if global_ok_fraction >= global_ok_threshold:
            alignment_status = "TENSION"
            notes.append(f"TDA blocking ({tda_block_rate:.1%}) while global mostly OK ({global_ok_fraction:.1%})")
            notes.append("Monitor for escalation to DIVERGENT")
        else:
            alignment_status = "ALIGNED"
            notes.append(f"TDA and global health both reporting issues")
    else:
        alignment_status = "ALIGNED"

    return GovernanceAlignmentResult(
        alignment_status=alignment_status,
        tda_block_rate=tda_block_rate,
        global_ok_fraction=global_ok_fraction,
        notes=notes,
    )


# ============================================================================
# Phase IV: Exception Window Management
# ============================================================================

@dataclass
class ExceptionWindowState:
    """State of the TDA exception window."""
    active: bool
    runs_remaining: int
    total_runs: int
    activation_reason: Optional[str]
    activated_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExceptionWindowManager:
    """
    Manages TDA hard gate exception windows.

    When governance alignment is DIVERGENT, operators can enable a
    time-bounded exception window where the hard gate behaves as dry_run.

    The window is explicitly tracked and logged, ensuring no silent bypasses.
    """

    def __init__(self, max_runs: Optional[int] = None):
        """
        Initialize exception window manager.

        Args:
            max_runs: Maximum runs for exception window.
                      Read from MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS if not provided.
        """
        if max_runs is None:
            max_runs = int(os.getenv("MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS", "0"))

        self._max_runs = max_runs
        self._runs_remaining = 0
        self._active = False
        self._activation_reason: Optional[str] = None
        self._activated_at: Optional[str] = None

    @property
    def active(self) -> bool:
        """Check if exception window is currently active."""
        return self._active and self._runs_remaining > 0

    @property
    def runs_remaining(self) -> int:
        """Get remaining runs in exception window."""
        return self._runs_remaining if self._active else 0

    def activate(self, reason: str) -> bool:
        """
        Activate exception window.

        Args:
            reason: Reason for activation (e.g., alignment report reference).

        Returns:
            True if activated, False if already active or max_runs is 0.
        """
        if self._max_runs <= 0:
            logger.warning("[TDA] Cannot activate exception window: max_runs is 0")
            return False

        if self._active:
            logger.warning("[TDA] Exception window already active")
            return False

        self._active = True
        self._runs_remaining = self._max_runs
        self._activation_reason = reason
        self._activated_at = datetime.utcnow().isoformat() + "Z"

        logger.info(f"[TDA] Exception window activated: {self._max_runs} runs, reason: {reason}")
        return True

    def consume_run(self) -> ExceptionWindowState:
        """
        Consume one run from exception window.

        Returns:
            Current exception window state after consumption.
        """
        if self._active and self._runs_remaining > 0:
            self._runs_remaining -= 1
            if self._runs_remaining == 0:
                logger.info("[TDA] Exception window exhausted, returning to hard mode")
                self._active = False

        return self.get_state()

    def get_state(self) -> ExceptionWindowState:
        """Get current exception window state."""
        return ExceptionWindowState(
            active=self._active,
            runs_remaining=self._runs_remaining,
            total_runs=self._max_runs,
            activation_reason=self._activation_reason,
            activated_at=self._activated_at,
        )

    def reset(self) -> None:
        """Reset exception window (for testing or manual intervention)."""
        self._active = False
        self._runs_remaining = 0
        self._activation_reason = None
        self._activated_at = None


# Global exception window manager (singleton pattern)
_exception_window_manager: Optional[ExceptionWindowManager] = None


def get_exception_window_manager() -> ExceptionWindowManager:
    """Get the global exception window manager."""
    global _exception_window_manager
    if _exception_window_manager is None:
        _exception_window_manager = ExceptionWindowManager()
    return _exception_window_manager


# ============================================================================
# Phase IV: Extended Health Summary
# ============================================================================

def summarize_tda_for_global_health_v2(
    tda_results: List["TDAMonitorResult"],
    config: "TDAMonitorConfig",
    hard_gate_mode: TDAHardGateMode = TDAHardGateMode.HARD,
    exception_manager: Optional[ExceptionWindowManager] = None,
    hypothetical_blocks: int = 0,
) -> Dict[str, Any]:
    """
    Extended global health summary with Phase IV fields.

    Includes hard gate mode, exception window state, and dry-run statistics.

    Args:
        tda_results: List of TDAMonitorResult from evaluations.
        config: TDAMonitorConfig for threshold context.
        hard_gate_mode: Current hard gate operational mode.
        exception_manager: Optional exception window manager.
        hypothetical_blocks: Count of blocks in dry_run mode.

    Returns:
        Extended dictionary with Phase IV fields.
    """
    # Start with Phase III summary
    base_summary = summarize_tda_for_global_health(tda_results, config)

    # Add Phase IV fields
    exception_state = exception_manager.get_state() if exception_manager else ExceptionWindowState(
        active=False, runs_remaining=0, total_runs=0, activation_reason=None, activated_at=None
    )

    base_summary.update({
        # Phase IV: Hard gate mode
        "hard_gate_mode": hard_gate_mode.value,
        "hard_gate_block_rate": base_summary["block_rate"],

        # Phase IV: Exception window
        "hard_gate_exception_window_active": exception_state.active,
        "hard_gate_exception_runs_remaining": exception_state.runs_remaining if exception_state.active else None,

        # Phase IV: Dry-run statistics
        "hypothetical_block_count": hypothetical_blocks,
        "hypothetical_block_rate": hypothetical_blocks / len(tda_results) if tda_results else 0.0,
    })

    return base_summary


# ============================================================================
# Phase IV: Evidence Tile Builder
# ============================================================================

def build_tda_hard_gate_evidence_tile(
    tda_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build compact evidence tile for TDA hard gate status.

    This tile is designed for inclusion in attestation/evidence packs.
    It uses neutral, factual language with no normative statements.

    Args:
        tda_summary: Output from summarize_tda_for_global_health or _v2.

    Returns:
        Compact JSON object with:
        - schema_version
        - mode
        - block_rate
        - mean_hss
        - hss_trend
        - structural_health
        - cycle_count
        - exception_active (if applicable)

    Notes:
        - Output is deterministic for same input
        - No normative language (e.g., "good", "bad")
        - Suitable for external audit consumption
    """
    return {
        "schema_version": TDA_EVIDENCE_TILE_SCHEMA_VERSION,
        "mode": tda_summary.get("hard_gate_mode", "unknown"),
        "block_rate": round(tda_summary.get("block_rate", 0.0), 4),
        "mean_hss": round(tda_summary.get("mean_hss", 0.0), 4),
        "hss_trend": round(tda_summary.get("hss_trend", 0.0), 6),
        "structural_health": round(tda_summary.get("structural_health", 0.0), 4),
        "cycle_count": tda_summary.get("cycle_count", 0),
        "block_count": tda_summary.get("block_count", 0),
        "warn_count": tda_summary.get("warn_count", 0),
        "ok_count": tda_summary.get("ok_count", 0),
        "exception_active": tda_summary.get("hard_gate_exception_window_active", False),
        "hypothetical_block_count": tda_summary.get("hypothetical_block_count", 0),
    }


# ============================================================================
# Phase IV: Hard Gate Decision Helper
# ============================================================================

@dataclass
class HardGateDecision:
    """Result of hard gate decision evaluation."""
    should_block: bool
    should_log_as_would_block: bool
    mode: TDAHardGateMode
    exception_window_active: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "should_block": self.should_block,
            "should_log_as_would_block": self.should_log_as_would_block,
            "mode": self.mode.value,  # Convert enum to string
            "exception_window_active": self.exception_window_active,
            "reason": self.reason,
        }


def evaluate_hard_gate_decision(
    tda_result: "TDAMonitorResult",
    mode: TDAHardGateMode,
    exception_manager: Optional[ExceptionWindowManager] = None,
) -> HardGateDecision:
    """
    Centralized hard gate decision logic.

    This function consolidates the decision logic for whether to block,
    taking into account the current mode and exception window state.

    Args:
        tda_result: TDA evaluation result.
        mode: Current hard gate operational mode.
        exception_manager: Optional exception window manager.

    Returns:
        HardGateDecision indicating:
        - should_block: Whether to actually block (return ABANDONED_TDA)
        - should_log_as_would_block: Whether to log as "would have blocked"
        - mode: The mode that was applied
        - exception_window_active: Whether exception window affected decision
        - reason: Human-readable explanation

    Behavior by mode:
        OFF: Never block, never log
        SHADOW: Never block, log TDA scores only
        DRY_RUN: Never block, log "would have blocked"
        HARD: Block when HSS < threshold (unless exception window active)
    """
    would_block = tda_result.block
    exception_active = exception_manager.active if exception_manager else False

    if mode == TDAHardGateMode.OFF:
        return HardGateDecision(
            should_block=False,
            should_log_as_would_block=False,
            mode=mode,
            exception_window_active=False,
            reason="TDA hard gate disabled (OFF mode)",
        )

    if mode == TDAHardGateMode.SHADOW:
        return HardGateDecision(
            should_block=False,
            should_log_as_would_block=False,
            mode=mode,
            exception_window_active=False,
            reason="TDA in shadow mode (logging only)",
        )

    if mode == TDAHardGateMode.DRY_RUN:
        return HardGateDecision(
            should_block=False,
            should_log_as_would_block=would_block,
            mode=mode,
            exception_window_active=False,
            reason=f"TDA dry-run: would_block={would_block}, HSS={tda_result.hss:.3f}",
        )

    # HARD mode
    if exception_active:
        # Exception window overrides to dry-run behavior
        if exception_manager:
            exception_manager.consume_run()
        return HardGateDecision(
            should_block=False,
            should_log_as_would_block=would_block,
            mode=mode,
            exception_window_active=True,
            reason=f"Exception window active: would_block={would_block}, HSS={tda_result.hss:.3f}",
        )

    # Normal HARD mode behavior
    return HardGateDecision(
        should_block=would_block,
        should_log_as_would_block=False,
        mode=mode,
        exception_window_active=False,
        reason=f"Hard gate: block={would_block}, HSS={tda_result.hss:.3f}",
    )


__all__ = [
    # Phase III
    "compute_tda_pipeline_hash",
    "summarize_tda_for_global_health",
    "compute_drift_metrics",
    "generate_drift_report",
    "extend_attestation_with_tda",
    "GovernanceSummary",
    "DriftMetrics",
    "TDA_GOVERNANCE_SCHEMA_VERSION",
    "TDA_DRIFT_REPORT_SCHEMA_VERSION",
    # Phase IV
    "TDAHardGateMode",
    "LabeledTDAResult",
    "CalibrationResult",
    "evaluate_hard_gate_calibration",
    "GovernanceAlignmentResult",
    "evaluate_tda_governance_alignment",
    "ExceptionWindowState",
    "ExceptionWindowManager",
    "get_exception_window_manager",
    "summarize_tda_for_global_health_v2",
    "build_tda_hard_gate_evidence_tile",
    "HardGateDecision",
    "evaluate_hard_gate_decision",
    "TDA_CALIBRATION_SCHEMA_VERSION",
    "TDA_EVIDENCE_TILE_SCHEMA_VERSION",
]
