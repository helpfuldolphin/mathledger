#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
# File: experiments/curriculum_health.py
"""
Curriculum Health Dashboard CLI for Phase II Uplift Slices.

This module provides comprehensive health monitoring, scoring, drift detection,
and snapshot capabilities for the Phase II curriculum. All operations are
read-only and produce deterministic, JSON-serializable outputs.

=============================================================================
CI / PRE-FLIGHT GATE USAGE
=============================================================================

RECOMMENDED CI JOB:
    uv run python -m experiments.curriculum_health --preflight --json

EXIT CODE SEMANTICS:
    0 - OK or WARN: Curriculum is healthy enough to proceed
    1 - FAIL: Critical issues detected, experiments should NOT proceed
    2 - ERROR: Internal error (config not found, parse error, etc.)

WHAT OTHER AGENTS CAN RELY ON:
    - C6 (Ledger): Uses CurriculumSnapshot.to_ledger_entry() for audit trails
    - B1 (Runner): Can gate experiment execution on exit code 0
    - Governance: --preflight --json output is stable and versioned

IMPORTANT: This is a PHASE II gate. It does NOT touch or validate
the Phase I curriculum (config/curriculum.yaml). Phase I has its own
validation mechanisms.

=============================================================================

Features:
---------
1. Curriculum Health Dashboard CLI
   - List slices with metadata
   - Describe individual slices
   - Display success metrics
   - Compute and display config hashes
   - Validate curriculum integrity
   - Check non-degeneracy constraints

2. Slice Health Scoring
   - Formula pool integrity (30%)
   - Success metric completeness (30%)
   - Monotonicity position (20%)
   - Parameter plausibility (20%)

3. Curriculum Drift Detection
   - Per-field comparison between slices
   - Drift magnitude calculation
   - Severity classification (cosmetic/parametric/semantic)

4. Curriculum Snapshot
   - Version, hashes, metrics, health scores
   - Plugs into C6 ledger integration via to_ledger_entry()

5. Pre-Flight Gate (--preflight)
   - Runs validation + non-degeneracy + health scoring
   - Per-slice OK | WARN | FAIL verdicts
   - Global verdict with CI-friendly exit codes
   - Deterministic JSON output for automation

Usage:
------
    # Pre-flight gate (recommended for CI)
    python -m experiments.curriculum_health --preflight
    python -m experiments.curriculum_health --preflight --json

    # Individual commands
    python -m experiments.curriculum_health --list
    python -m experiments.curriculum_health --describe slice_uplift_goal
    python -m experiments.curriculum_health --metrics
    python -m experiments.curriculum_health --hashes
    python -m experiments.curriculum_health --validate
    python -m experiments.curriculum_health --nondegenerate
    python -m experiments.curriculum_health --score slice_uplift_goal
    python -m experiments.curriculum_health --compare slice_uplift_goal slice_uplift_sparse
    python -m experiments.curriculum_health --snapshot

Reference Documents:
--------------------
- docs/PHASE2_RFL_UPLIFT_PLAN.md
- experiments/slice_success_metrics.py
- config/curriculum_uplift_phase2.yaml
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    FormulaPoolIntegrityResult,
    SuccessMetricValidationResult,
    REQUIRED_PARAMETER_FIELDS,
    SUCCESS_METRIC_PARAM_SCHEMA,
)


# =============================================================================
# HEALTH SCORE BANDS
# =============================================================================
#
# THRESHOLD RATIONALE:
# --------------------
# These thresholds are DIAGNOSTIC, not promotion gates. They help developers
# and CI systems quickly understand curriculum health at a glance.
#
#   >= 0.95 (EXCELLENT): All checks pass with no issues. Formula pool is clean,
#       metrics are complete, parameters are within plausible bounds, and
#       monotonicity is respected. Ideal state for production experiments.
#
#   [0.85, 0.95) (GOOD): Minor issues detected but curriculum is usable.
#       Typical cause: monotonicity warnings (cross-family slice ordering).
#       Experiments can proceed but issues should be reviewed.
#
#   [0.70, 0.85) (BORDERLINE): Significant issues warrant attention before
#       running experiments. May have parameter edge cases, missing metric
#       params, or multiple warnings. Human review recommended.
#
#   < 0.70 (POOR): Curriculum has critical issues. Missing required fields,
#       hash collisions, normalization errors, or implausible parameters.
#       Experiments should NOT proceed until issues are resolved.
#
# These bands are used for reporting only. The pre-flight gate uses separate
# thresholds (PREFLIGHT_OK_THRESHOLD, PREFLIGHT_WARN_THRESHOLD) to determine
# exit codes and verdicts.
# =============================================================================

class HealthBand(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Classification bands for health scores.

    These bands are DIAGNOSTIC indicators, not promotion logic.
    They provide human-readable categorization of slice health
    for CI reporting and developer awareness.

    See module-level THRESHOLD RATIONALE comment for justification.
    """
    EXCELLENT = "excellent"    # >= 0.95
    GOOD = "good"              # [0.85, 0.95)
    BORDERLINE = "borderline"  # [0.70, 0.85)
    POOR = "poor"              # < 0.70


# Health band threshold constants (for test coverage and clarity)
BAND_EXCELLENT_THRESHOLD = 0.95
BAND_GOOD_THRESHOLD = 0.85
BAND_BORDERLINE_THRESHOLD = 0.70


def classify_health_band(score: float) -> HealthBand:
    """
    PHASE II — NOT USED IN PHASE I

    Classify a health score into a diagnostic band.

    This is the CANONICAL function for health band classification.
    All band assignments must go through this function to ensure
    consistent categorization across CLI output, JSON, and tests.

    IMPORTANT: These bands are DIAGNOSTIC, not promotion logic.
    They help developers understand curriculum health at a glance.
    Pre-flight gate verdicts use separate thresholds.

    Thresholds:
        >= 0.95 -> EXCELLENT (production-ready)
        [0.85, 0.95) -> GOOD (usable with minor issues)
        [0.70, 0.85) -> BORDERLINE (review recommended)
        < 0.70 -> POOR (experiments should not proceed)

    Pure function. Deterministic. No side effects.

    Args:
        score: Health score between 0.0 and 1.0

    Returns:
        HealthBand classification

    Examples:
        >>> classify_health_band(1.0)
        HealthBand.EXCELLENT
        >>> classify_health_band(0.95)
        HealthBand.EXCELLENT
        >>> classify_health_band(0.9499)
        HealthBand.GOOD
        >>> classify_health_band(0.70)
        HealthBand.BORDERLINE
        >>> classify_health_band(0.69)
        HealthBand.POOR
    """
    if score >= BAND_EXCELLENT_THRESHOLD:
        return HealthBand.EXCELLENT
    elif score >= BAND_GOOD_THRESHOLD:
        return HealthBand.GOOD
    elif score >= BAND_BORDERLINE_THRESHOLD:
        return HealthBand.BORDERLINE
    else:
        return HealthBand.POOR


# =============================================================================
# PRE-FLIGHT VERDICTS
# =============================================================================

class PreflightVerdict(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Pre-flight check verdict for a slice.
    """
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


class PreflightGlobalVerdict(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Global pre-flight verdict for entire curriculum.
    """
    OK = "OK"      # All slices >= 0.9, no FAILs
    WARN = "WARN"  # Any slice 0.7-0.9 or minor issues
    FAIL = "FAIL"  # Any slice < 0.7 or critical issues


# Pre-flight thresholds
PREFLIGHT_OK_THRESHOLD = 0.90
PREFLIGHT_WARN_THRESHOLD = 0.70


@dataclass
class SlicePreflightResult:
    """
    PHASE II — NOT USED IN PHASE I

    Pre-flight check result for a single slice.

    This dataclass captures the complete pre-flight status of an individual
    slice, including health scoring, validation status, and any issues found.

    Attributes:
        slice_name: Unique identifier for the slice
        verdict: OK | WARN | FAIL based on health and validation
        health_score: Weighted health score (0.0 to 1.0)
        health_band: Diagnostic band classification
        issues_count: Number of issues detected
        issues: List of human-readable issue descriptions
        validation_passed: True if schema validation succeeded
        nondegenerate_passed: True if non-degeneracy checks passed
    """
    slice_name: str
    verdict: PreflightVerdict
    health_score: float
    health_band: HealthBand
    issues_count: int
    issues: List[str]
    validation_passed: bool
    nondegenerate_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'slice_name': self.slice_name,
            'verdict': self.verdict.value,
            'health_score': round(self.health_score, 4),
            'health_band': self.health_band.value,
            'issues_count': self.issues_count,
            'issues': self.issues,
            'validation_passed': self.validation_passed,
            'nondegenerate_passed': self.nondegenerate_passed,
        }


@dataclass
class PreflightReport:
    """
    PHASE II — NOT USED IN PHASE I

    Complete pre-flight report for curriculum.

    =========================================================================
    PRE-FLIGHT GATE CONTRACT
    =========================================================================

    This dataclass defines the CANONICAL shape of pre-flight results.
    CI systems, the ledger (C6), and other agents can rely on this contract.

    VERDICT SEMANTICS:
        OK:   All slices have health >= PREFLIGHT_OK_THRESHOLD (0.90),
              validation passed, and non-degeneracy checks passed.
              Exit code: 0

        WARN: At least one slice has health in [PREFLIGHT_WARN_THRESHOLD, OK)
              (0.70 to 0.90), or has minor issues that don't block execution.
              Exit code: 0 (experiments may proceed with caution)

        FAIL: At least one slice has health < PREFLIGHT_WARN_THRESHOLD (0.70),
              or has critical validation failures.
              Exit code: 1 (experiments should NOT proceed)

    JSON OUTPUT:
        The --preflight --json command returns a JSON object that exactly
        mirrors the to_dict() output of this dataclass. This is guaranteed
        to be deterministic for identical curricula (except timestamp).

    STABILITY GUARANTEE:
        Fields in this contract will not be removed or have their semantics
        changed without a major version bump. New fields may be added.

    Attributes:
        timestamp: ISO 8601 UTC timestamp of the check
        curriculum_version: Version string from curriculum YAML
        global_verdict: Overall OK | WARN | FAIL verdict
        slice_count: Total number of slices checked
        ok_count: Number of slices with OK verdict
        warn_count: Number of slices with WARN verdict
        fail_count: Number of slices with FAIL verdict
        overall_health: Average health score across all slices
        slices: Per-slice results keyed by slice name
        monotonicity_warnings: List of monotonicity violation messages
    """
    timestamp: str
    curriculum_version: str
    global_verdict: PreflightGlobalVerdict
    slice_count: int
    ok_count: int
    warn_count: int
    fail_count: int
    overall_health: float
    slices: Dict[str, SlicePreflightResult]
    monotonicity_warnings: List[str]

    @property
    def issues_total(self) -> int:
        """Total number of issues across all slices."""
        return sum(s.issues_count for s in self.slices.values())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        This is the CANONICAL JSON shape for --preflight --json output.
        CI and other agents can parse this structure reliably.
        """
        return {
            'timestamp': self.timestamp,
            'curriculum_version': self.curriculum_version,
            'global_verdict': self.global_verdict.value,
            'slice_count': self.slice_count,
            'ok_count': self.ok_count,
            'warn_count': self.warn_count,
            'fail_count': self.fail_count,
            'issues_total': self.issues_total,
            'overall_health': round(self.overall_health, 4),
            'slices': {k: v.to_dict() for k, v in self.slices.items()},
            'monotonicity_warnings': self.monotonicity_warnings,
        }


# =============================================================================
# DATACLASSES FOR STRUCTURED OUTPUTS
# =============================================================================

class DriftSeverity(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Classification of parameter drift severity.
    """
    COSMETIC = "cosmetic"       # Description changes, no semantic impact
    PARAMETRIC = "parametric"   # Numeric parameter changes
    SEMANTIC = "semantic"       # Metric kind or structural changes


@dataclass
class FieldDrift:
    """
    PHASE II — NOT USED IN PHASE I

    Represents drift in a single field between two slice configurations.
    """
    field_path: str           # e.g., "parameters.atoms" or "success_metric.kind"
    old_value: Any
    new_value: Any
    drift_magnitude: float    # 0.0 to 1.0 normalized magnitude
    severity: DriftSeverity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'field_path': self.field_path,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'drift_magnitude': self.drift_magnitude,
            'severity': self.severity.value,
        }


@dataclass
class DriftReport:
    """
    PHASE II — NOT USED IN PHASE I

    Complete drift report between two slice configurations.
    """
    slice_a: str
    slice_b: str
    changed_fields: List[FieldDrift]
    total_drift_magnitude: float
    max_severity: DriftSeverity
    is_compatible: bool       # True if drift is cosmetic only

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'slice_a': self.slice_a,
            'slice_b': self.slice_b,
            'changed_fields': [f.to_dict() for f in self.changed_fields],
            'total_drift_magnitude': self.total_drift_magnitude,
            'max_severity': self.max_severity.value,
            'is_compatible': self.is_compatible,
        }


@dataclass
class SliceHealthScore:
    """
    PHASE II — NOT USED IN PHASE I

    Health score breakdown for a single slice.
    """
    slice_name: str
    total_score: float                    # 0.0 to 1.0
    formula_pool_integrity: float         # 30% weight
    success_metric_completeness: float    # 30% weight
    monotonicity_position: float          # 20% weight
    parameter_plausibility: float         # 20% weight
    issues: List[str]

    @property
    def band(self) -> HealthBand:
        """Get the health band classification for this score."""
        return classify_health_band(self.total_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'slice_name': self.slice_name,
            'total_score': round(self.total_score, 4),
            'band': self.band.value,
            'components': {
                'formula_pool_integrity': round(self.formula_pool_integrity, 4),
                'success_metric_completeness': round(self.success_metric_completeness, 4),
                'monotonicity_position': round(self.monotonicity_position, 4),
                'parameter_plausibility': round(self.parameter_plausibility, 4),
            },
            'issues': self.issues,
        }


@dataclass
class CurriculumSnapshot:
    """
    PHASE II — NOT USED IN PHASE I

    Complete snapshot of curriculum state for ledger integration.
    """
    timestamp: str
    curriculum_version: str
    slice_count: int
    slice_hashes: Dict[str, str]
    metric_kinds: Dict[str, str]
    formula_pool_counts: Dict[str, int]
    health_scores: Dict[str, float]
    monotonicity_warnings: List[str]
    overall_health: float
    snapshot_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'timestamp': self.timestamp,
            'curriculum_version': self.curriculum_version,
            'slice_count': self.slice_count,
            'slice_hashes': self.slice_hashes,
            'metric_kinds': self.metric_kinds,
            'formula_pool_counts': self.formula_pool_counts,
            'health_scores': {k: round(v, 4) for k, v in self.health_scores.items()},
            'monotonicity_warnings': self.monotonicity_warnings,
            'overall_health': round(self.overall_health, 4),
            'snapshot_hash': self.snapshot_hash,
        }

    def to_ledger_entry(self) -> Dict[str, Any]:
        """
        PHASE II — NOT USED IN PHASE I

        =====================================================================
        CANONICAL LEDGER INTEGRATION INTERFACE
        =====================================================================

        This method is the ONLY API the ledger (C6) should use for obtaining
        curriculum health data. Any ledger changes must preserve this shape
        or extend it with backward compatibility.

        CONTRACT GUARANTEES:
        --------------------
        1. DETERMINISM: Identical curricula produce identical outputs
           (except timestamp, which is not used for hashing)

        2. SORTED KEYS: All dictionary keys are sorted for consistent
           JSON serialization and hash computation

        3. NO LEDGER IMPORTS: This method is a pure data provider.
           It does NOT import or interact with ledger code directly.
           The ledger consumes this output; this module does not know
           about the ledger's existence.

        4. FIELD STABILITY: These fields will not be removed:
           - snapshot_timestamp
           - snapshot_hash
           - slice_hashes
           - overall_health
           - slice_health

        5. HASH INTEGRITY: The snapshot_hash is computed from slice_hashes
           and will change if ANY slice configuration changes.

        USAGE BY C6 (LEDGER):
        ---------------------
        The ledger should call:
            snapshot = create_snapshot(loader)
            entry = snapshot.to_ledger_entry()
            # entry is now ready for ledger recording

        The snapshot_hash field provides a single value that uniquely
        identifies the curriculum state and can be used for audit trails.

        Returns:
            Dictionary suitable for ledger recording with:
            - snapshot_timestamp: ISO 8601 UTC timestamp
            - snapshot_hash: SHA256 hash of all slice configurations
            - slice_hashes: Per-slice config hashes (sorted by name)
            - overall_health: Average health score (0.0-1.0)
            - slice_health: Per-slice health scores (sorted by name)
        """
        return {
            'snapshot_timestamp': self.timestamp,
            'snapshot_hash': self.snapshot_hash,
            'slice_hashes': dict(sorted(self.slice_hashes.items())),
            'overall_health': round(self.overall_health, 4),
            'slice_health': {k: round(v, 4) for k, v in sorted(self.health_scores.items())},
        }


# =============================================================================
# HEALTH SCORING FUNCTIONS (PURE, DETERMINISTIC)
# =============================================================================

# Scoring weights
WEIGHT_FORMULA_POOL_INTEGRITY = 0.30
WEIGHT_SUCCESS_METRIC_COMPLETENESS = 0.30
WEIGHT_MONOTONICITY_POSITION = 0.20
WEIGHT_PARAMETER_PLAUSIBILITY = 0.20

# Parameter plausibility bounds
PLAUSIBLE_ATOMS_RANGE = (2, 10)
PLAUSIBLE_DEPTH_MIN_RANGE = (1, 5)
PLAUSIBLE_DEPTH_MAX_RANGE = (3, 15)
PLAUSIBLE_BREADTH_MAX_RANGE = (10, 200)
PLAUSIBLE_TOTAL_MAX_RANGE = (50, 1000)
PLAUSIBLE_FORMULA_POOL_RANGE = (5, 100)
PLAUSIBLE_AXIOM_INSTANCES_RANGE = (5, 100)


def compute_formula_pool_integrity_score(
    pool_result: FormulaPoolIntegrityResult,
) -> Tuple[float, List[str]]:
    """
    PHASE II — NOT USED IN PHASE I

    Compute integrity score for formula pool (0.0 to 1.0).

    Pure function. Deterministic.

    Args:
        pool_result: Result from validate_formula_pool_integrity()

    Returns:
        Tuple of (score, list_of_issues)
    """
    issues: List[str] = []
    score = 1.0

    # Penalize duplicates (-0.2 per duplicate, max -0.4)
    if pool_result.duplicate_formulas:
        penalty = min(0.4, len(pool_result.duplicate_formulas) * 0.2)
        score -= penalty
        issues.append(f"Duplicate formulas: {len(pool_result.duplicate_formulas)}")

    # Penalize normalization errors (-0.3 per error, max -0.5)
    if pool_result.normalization_errors:
        penalty = min(0.5, len(pool_result.normalization_errors) * 0.3)
        score -= penalty
        issues.append(f"Normalization errors: {len(pool_result.normalization_errors)}")

    # Penalize hash collisions (-0.5 per collision, fatal)
    if pool_result.hash_collisions:
        score = 0.0
        issues.append(f"Hash collisions detected: {len(pool_result.hash_collisions)}")

    return max(0.0, score), issues


def compute_success_metric_completeness_score(
    metric_result: SuccessMetricValidationResult,
) -> Tuple[float, List[str]]:
    """
    PHASE II — NOT USED IN PHASE I

    Compute completeness score for success metric configuration (0.0 to 1.0).

    Pure function. Deterministic.

    Args:
        metric_result: Result from validate_success_metric()

    Returns:
        Tuple of (score, list_of_issues)
    """
    issues: List[str] = []

    if metric_result.valid:
        return 1.0, []

    score = 1.0

    # Missing required params: -0.5 per param
    if metric_result.missing_params:
        penalty = min(1.0, len(metric_result.missing_params) * 0.5)
        score -= penalty
        issues.append(f"Missing metric params: {sorted(metric_result.missing_params)}")

    # Unknown params: -0.2 per param (less severe)
    if metric_result.unknown_params:
        penalty = min(0.4, len(metric_result.unknown_params) * 0.2)
        score -= penalty
        issues.append(f"Unknown metric params: {sorted(metric_result.unknown_params)}")

    return max(0.0, score), issues


def compute_monotonicity_position_score(
    slice_name: str,
    all_slices: List[str],
    monotonicity_warnings: List[str],
) -> Tuple[float, List[str]]:
    """
    PHASE II — NOT USED IN PHASE I

    Compute monotonicity position score (0.0 to 1.0).

    A slice scores higher if it's not involved in any monotonicity violation.

    Pure function. Deterministic.

    Args:
        slice_name: The slice being scored
        all_slices: List of all slice names in order
        monotonicity_warnings: List of warning messages

    Returns:
        Tuple of (score, list_of_issues)
    """
    issues: List[str] = []

    # Check if this slice is mentioned in any warning
    involved_in_violation = any(
        slice_name in warning for warning in monotonicity_warnings
    )

    if involved_in_violation:
        # Count how many violations involve this slice
        violation_count = sum(
            1 for w in monotonicity_warnings if slice_name in w
        )
        penalty = min(0.8, violation_count * 0.4)
        score = 1.0 - penalty
        issues.append(f"Involved in {violation_count} monotonicity violation(s)")
    else:
        score = 1.0

    return score, issues


def compute_parameter_plausibility_score(
    params: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    PHASE II — NOT USED IN PHASE I

    Compute parameter plausibility score (0.0 to 1.0).

    Checks that parameters are within reasonable bounds for uplift experiments.

    Pure function. Deterministic.

    Args:
        params: Parameters dictionary from slice config

    Returns:
        Tuple of (score, list_of_issues)
    """
    issues: List[str] = []
    penalties = 0.0

    def check_range(value: Any, bounds: Tuple[int, int], name: str) -> float:
        """Return penalty (0.0 to 0.15) if value outside bounds."""
        if not isinstance(value, (int, float)):
            return 0.15
        if value < bounds[0] or value > bounds[1]:
            issues.append(f"{name}={value} outside plausible range {bounds}")
            return 0.15
        return 0.0

    penalties += check_range(params.get('atoms', 0), PLAUSIBLE_ATOMS_RANGE, 'atoms')
    penalties += check_range(params.get('depth_min', 0), PLAUSIBLE_DEPTH_MIN_RANGE, 'depth_min')
    penalties += check_range(params.get('depth_max', 0), PLAUSIBLE_DEPTH_MAX_RANGE, 'depth_max')
    penalties += check_range(params.get('breadth_max', 0), PLAUSIBLE_BREADTH_MAX_RANGE, 'breadth_max')
    penalties += check_range(params.get('total_max', 0), PLAUSIBLE_TOTAL_MAX_RANGE, 'total_max')
    penalties += check_range(params.get('formula_pool', 0), PLAUSIBLE_FORMULA_POOL_RANGE, 'formula_pool')
    penalties += check_range(params.get('axiom_instances', 0), PLAUSIBLE_AXIOM_INSTANCES_RANGE, 'axiom_instances')

    # Check depth_min < depth_max
    depth_min = params.get('depth_min', 0)
    depth_max = params.get('depth_max', 0)
    if depth_min >= depth_max:
        penalties += 0.2
        issues.append(f"depth_min ({depth_min}) >= depth_max ({depth_max})")

    return max(0.0, 1.0 - penalties), issues


def compute_slice_health(
    loader: CurriculumLoaderV2,
    slice_name: str,
) -> SliceHealthScore:
    """
    PHASE II — NOT USED IN PHASE I

    Compute comprehensive health score for a slice.

    Scoring components:
    - formula_pool_integrity:      30%
    - success_metric_completeness: 30%
    - monotonicity_position:       20%
    - parameter_plausibility:      20%

    Pure function. Deterministic. Zero side effects.

    Args:
        loader: CurriculumLoaderV2 instance
        slice_name: Name of the slice to score

    Returns:
        SliceHealthScore with breakdown and total score.
    """
    all_issues: List[str] = []
    all_slices = loader.list_slices()
    monotonicity_warnings = loader.validate_monotonicity()

    # 1. Formula pool integrity (30%)
    pool_result = loader.validate_formula_pool_integrity(slice_name)
    pool_score, pool_issues = compute_formula_pool_integrity_score(pool_result)
    all_issues.extend(pool_issues)

    # 2. Success metric completeness (30%)
    metric_result = loader.validate_success_metric(slice_name)
    metric_score, metric_issues = compute_success_metric_completeness_score(metric_result)
    all_issues.extend(metric_issues)

    # 3. Monotonicity position (20%)
    mono_score, mono_issues = compute_monotonicity_position_score(
        slice_name, all_slices, monotonicity_warnings
    )
    all_issues.extend(mono_issues)

    # 4. Parameter plausibility (20%)
    params = loader.get_parameters(slice_name)
    plaus_score, plaus_issues = compute_parameter_plausibility_score(params)
    all_issues.extend(plaus_issues)

    # Weighted total
    total_score = (
        pool_score * WEIGHT_FORMULA_POOL_INTEGRITY
        + metric_score * WEIGHT_SUCCESS_METRIC_COMPLETENESS
        + mono_score * WEIGHT_MONOTONICITY_POSITION
        + plaus_score * WEIGHT_PARAMETER_PLAUSIBILITY
    )

    return SliceHealthScore(
        slice_name=slice_name,
        total_score=total_score,
        formula_pool_integrity=pool_score,
        success_metric_completeness=metric_score,
        monotonicity_position=mono_score,
        parameter_plausibility=plaus_score,
        issues=all_issues,
    )


# =============================================================================
# DRIFT DETECTION FUNCTIONS (PURE, DETERMINISTIC)
# =============================================================================

def _compute_numeric_drift(old: Any, new: Any) -> float:
    """Compute normalized drift magnitude for numeric values."""
    if old == new:
        return 0.0
    if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
        return 1.0  # Non-numeric change = max drift
    if old == 0:
        return 1.0 if new != 0 else 0.0
    return min(1.0, abs(new - old) / abs(old))


def _classify_severity(field_path: str, old: Any, new: Any) -> DriftSeverity:
    """Classify the severity of a field change."""
    # Semantic changes: metric kind, uplift phase
    if 'success_metric.kind' in field_path:
        return DriftSeverity.SEMANTIC
    if 'uplift.phase' in field_path:
        return DriftSeverity.SEMANTIC

    # Cosmetic changes: description only
    if field_path == 'description':
        return DriftSeverity.COSMETIC

    # Everything else is parametric
    return DriftSeverity.PARAMETRIC


def _flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary into dot-separated keys."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            result[key] = v  # Keep lists as-is
        else:
            result[key] = v
    return result


def detect_param_drift(
    loader: CurriculumLoaderV2,
    slice_a: str,
    slice_b: str,
) -> DriftReport:
    """
    PHASE II — NOT USED IN PHASE I

    Detect parameter drift between two slice configurations.

    Compares all fields and computes drift magnitude and severity.

    Pure function. Deterministic. Zero side effects.

    Args:
        loader: CurriculumLoaderV2 instance
        slice_a: Name of first slice
        slice_b: Name of second slice

    Returns:
        DriftReport with detailed field-by-field comparison.
    """
    config_a = loader.get_slice_config(slice_a)
    config_b = loader.get_slice_config(slice_b)

    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)

    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    changed_fields: List[FieldDrift] = []
    max_severity = DriftSeverity.COSMETIC

    for key in sorted(all_keys):
        old_val = flat_a.get(key)
        new_val = flat_b.get(key)

        if old_val != new_val:
            magnitude = _compute_numeric_drift(old_val, new_val)
            severity = _classify_severity(key, old_val, new_val)

            # Track max severity
            if severity == DriftSeverity.SEMANTIC:
                max_severity = DriftSeverity.SEMANTIC
            elif severity == DriftSeverity.PARAMETRIC and max_severity != DriftSeverity.SEMANTIC:
                max_severity = DriftSeverity.PARAMETRIC

            changed_fields.append(FieldDrift(
                field_path=key,
                old_value=old_val,
                new_value=new_val,
                drift_magnitude=magnitude,
                severity=severity,
            ))

    # Total drift is average of all field drifts
    total_drift = (
        sum(f.drift_magnitude for f in changed_fields) / len(changed_fields)
        if changed_fields else 0.0
    )

    return DriftReport(
        slice_a=slice_a,
        slice_b=slice_b,
        changed_fields=changed_fields,
        total_drift_magnitude=total_drift,
        max_severity=max_severity,
        is_compatible=(max_severity == DriftSeverity.COSMETIC),
    )


# =============================================================================
# CURRICULUM SNAPSHOT FUNCTIONS (PURE, DETERMINISTIC)
# =============================================================================

def create_snapshot(loader: CurriculumLoaderV2) -> CurriculumSnapshot:
    """
    PHASE II — NOT USED IN PHASE I

    Create a complete snapshot of the curriculum state.

    This snapshot can be used for:
    - Preregistration hashing
    - Ledger integration (C6)
    - Drift detection over time
    - Audit trails

    Pure function. Deterministic. Zero side effects.

    Args:
        loader: CurriculumLoaderV2 instance

    Returns:
        CurriculumSnapshot with all curriculum metadata.
    """
    slices = loader.list_slices()

    # Collect all slice data
    slice_hashes: Dict[str, str] = {}
    metric_kinds: Dict[str, str] = {}
    formula_pool_counts: Dict[str, int] = {}
    health_scores: Dict[str, float] = {}

    for slice_name in slices:
        slice_hashes[slice_name] = loader.hash_slice_config(slice_name)
        metric = loader.get_success_metric_config(slice_name)
        metric_kinds[slice_name] = metric['kind']
        formula_pool_counts[slice_name] = len(loader.get_formula_pool(slice_name))

        health = compute_slice_health(loader, slice_name)
        health_scores[slice_name] = health.total_score

    # Compute overall health
    overall_health = (
        sum(health_scores.values()) / len(health_scores)
        if health_scores else 0.0
    )

    # Monotonicity warnings
    monotonicity_warnings = loader.validate_monotonicity()

    # Generate deterministic timestamp (use fixed format)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Compute snapshot hash (hash of all slice hashes for integrity)
    hash_content = json.dumps(slice_hashes, sort_keys=True)
    snapshot_hash = hashlib.sha256(hash_content.encode('utf-8')).hexdigest()

    return CurriculumSnapshot(
        timestamp=timestamp,
        curriculum_version=loader.get_version(),
        slice_count=len(slices),
        slice_hashes=slice_hashes,
        metric_kinds=metric_kinds,
        formula_pool_counts=formula_pool_counts,
        health_scores=health_scores,
        monotonicity_warnings=monotonicity_warnings,
        overall_health=overall_health,
        snapshot_hash=snapshot_hash,
    )


# =============================================================================
# PRE-FLIGHT GATE FUNCTIONS (PURE, DETERMINISTIC)
# =============================================================================

def _check_slice_nondegenerate(loader: CurriculumLoaderV2, slice_name: str) -> Tuple[bool, List[str]]:
    """
    Check non-degeneracy constraints for a single slice.

    Returns tuple of (passed, issues).
    """
    issues: List[str] = []
    params = loader.get_parameters(slice_name)
    budget = loader.get_budget(slice_name)
    pool = loader.get_formula_pool(slice_name)

    # Budget should be constraining
    if budget['max_candidates_per_cycle'] >= params['total_max']:
        issues.append("Budget does not constrain exploration")

    # Pool should have variety
    if len(pool) < 5:
        issues.append(f"Formula pool too small ({len(pool)} entries)")

    # Depth range should allow complexity
    if params['depth_max'] - params['depth_min'] < 2:
        issues.append("Depth range too narrow")

    return len(issues) == 0, issues


def run_preflight(loader: CurriculumLoaderV2) -> PreflightReport:
    """
    PHASE II — NOT USED IN PHASE I

    Run complete pre-flight checks for curriculum.

    This executes:
    - Validation checks (schema, required fields)
    - Non-degeneracy checks (budget, pool, depth constraints)
    - Health scoring for each slice

    Pre-flight verdict logic:
    - OK: health >= 0.90, no validation issues
    - WARN: health 0.70-0.90, or minor issues
    - FAIL: health < 0.70, or critical validation failure

    Global verdict:
    - OK: All slices OK
    - WARN: Any slice WARN but no FAIL
    - FAIL: Any slice FAIL

    Pure function. Deterministic. Zero side effects.

    Args:
        loader: CurriculumLoaderV2 instance

    Returns:
        PreflightReport with per-slice and global verdicts.
    """
    slices = loader.list_slices()
    validation_result = loader.validate_all()
    monotonicity_warnings = loader.validate_monotonicity()

    slice_results: Dict[str, SlicePreflightResult] = {}
    ok_count = 0
    warn_count = 0
    fail_count = 0
    total_health = 0.0

    for slice_name in slices:
        # Get health score
        health = compute_slice_health(loader, slice_name)
        total_health += health.total_score

        # Get validation status from validate_all result
        slice_validation = validation_result['slices'].get(slice_name, {})
        validation_passed = (
            slice_validation.get('success_metric_valid', False) and
            slice_validation.get('formula_pool_valid', False)
        )

        # Check non-degeneracy
        nondegen_passed, nondegen_issues = _check_slice_nondegenerate(loader, slice_name)

        # Collect all issues
        all_issues = list(health.issues)
        all_issues.extend(slice_validation.get('issues', []))
        all_issues.extend(nondegen_issues)

        # Determine verdict
        if health.total_score >= PREFLIGHT_OK_THRESHOLD and validation_passed and nondegen_passed:
            verdict = PreflightVerdict.OK
            ok_count += 1
        elif health.total_score >= PREFLIGHT_WARN_THRESHOLD:
            verdict = PreflightVerdict.WARN
            warn_count += 1
        else:
            verdict = PreflightVerdict.FAIL
            fail_count += 1

        slice_results[slice_name] = SlicePreflightResult(
            slice_name=slice_name,
            verdict=verdict,
            health_score=health.total_score,
            health_band=health.band,
            issues_count=len(all_issues),
            issues=all_issues,
            validation_passed=validation_passed,
            nondegenerate_passed=nondegen_passed,
        )

    # Determine global verdict
    if fail_count > 0:
        global_verdict = PreflightGlobalVerdict.FAIL
    elif warn_count > 0:
        global_verdict = PreflightGlobalVerdict.WARN
    else:
        global_verdict = PreflightGlobalVerdict.OK

    overall_health = total_health / len(slices) if slices else 0.0
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    return PreflightReport(
        timestamp=timestamp,
        curriculum_version=loader.get_version(),
        global_verdict=global_verdict,
        slice_count=len(slices),
        ok_count=ok_count,
        warn_count=warn_count,
        fail_count=fail_count,
        overall_health=overall_health,
        slices=slice_results,
        monotonicity_warnings=monotonicity_warnings,
    )


# =============================================================================
# CURRICULUM MANIFEST CONTRACT v1.0
# =============================================================================
#
# The Curriculum Manifest is the SINGLE CANONICAL JSON representation of
# curriculum configuration, health metadata, and inference-ready fields.
#
# All subsystems (ledger, runner, governance, CI) should consume this manifest
# rather than parsing the YAML directly.
# =============================================================================

# Manifest schema version - bump when structure changes
MANIFEST_SCHEMA_VERSION = "1.0.0"

# Float precision for manifest (6 decimals for determinism)
MANIFEST_FLOAT_PRECISION = 6


@dataclass
class CurriculumManifest:
    """
    PHASE II — NOT USED IN PHASE I

    Canonical curriculum manifest for cross-subsystem consumption.

    This is the SINGLE SOURCE OF TRUTH for curriculum metadata that other
    agents and systems should rely on. The manifest includes:

    - Schema version for compatibility checking
    - Slice configuration summaries
    - Health bands and scores
    - Pre-flight gate results
    - Canonical hash for integrity verification

    STABILITY GUARANTEE:
        Fields will not be removed without a schema version bump.
        New fields may be added with backward compatibility.
    """
    schema_version: str
    generated_at: str
    curriculum_version: str
    curriculum_hash: str
    slice_count: int
    slice_names: List[str]
    slices: Dict[str, Dict[str, Any]]
    global_preflight_verdict: str
    overall_health: float
    issues_total: int
    monotonicity_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to deterministic JSON-serializable dictionary.

        All keys are sorted, floats are rounded to MANIFEST_FLOAT_PRECISION.
        """
        return {
            'curriculum_hash': self.curriculum_hash,
            'curriculum_version': self.curriculum_version,
            'generated_at': self.generated_at,
            'global_preflight_verdict': self.global_preflight_verdict,
            'issues_total': self.issues_total,
            'monotonicity_warnings': self.monotonicity_warnings,
            'overall_health': round(self.overall_health, MANIFEST_FLOAT_PRECISION),
            'schema_version': self.schema_version,
            'slice_count': self.slice_count,
            'slice_names': self.slice_names,
            'slices': {
                k: {
                    sk: (round(sv, MANIFEST_FLOAT_PRECISION) if isinstance(sv, float) else sv)
                    for sk, sv in sorted(v.items())
                }
                for k, v in sorted(self.slices.items())
            },
        }


def create_curriculum_manifest(loader: CurriculumLoaderV2) -> CurriculumManifest:
    """
    PHASE II — NOT USED IN PHASE I

    Create a canonical curriculum manifest from loader.

    The manifest is deterministic: identical YAML produces identical manifest
    (except for generated_at timestamp, which is not included in hash computation).

    Pure function. Deterministic. No side effects.

    Args:
        loader: CurriculumLoaderV2 instance

    Returns:
        CurriculumManifest with all curriculum metadata
    """
    # Run preflight to get health data
    preflight = run_preflight(loader)

    # Get sorted slice names
    slice_names = sorted(loader.list_slices())

    # Build per-slice metadata
    slices_data: Dict[str, Dict[str, Any]] = {}
    for slice_name in slice_names:
        health = compute_slice_health(loader, slice_name)
        metric_config = loader.get_success_metric_config(slice_name)

        slices_data[slice_name] = {
            'health_band': health.band.value,
            'health_score': health.total_score,
            'issues_count': len(health.issues),
            'success_metric_kind': metric_config['kind'],
        }

    # Compute curriculum hash (SHA-256 over canonical YAML representation)
    # We use the snapshot hash as the curriculum identity
    snapshot = create_snapshot(loader)
    curriculum_hash = snapshot.snapshot_hash

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    return CurriculumManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at=timestamp,
        curriculum_version=loader.get_version(),
        curriculum_hash=curriculum_hash,
        slice_count=len(slice_names),
        slice_names=slice_names,
        slices=slices_data,
        global_preflight_verdict=preflight.global_verdict.value,
        overall_health=preflight.overall_health,
        issues_total=preflight.issues_total,
        monotonicity_warnings=preflight.monotonicity_warnings,
    )


def export_curriculum_manifest(loader: CurriculumLoaderV2, path: str) -> None:
    """
    PHASE II — NOT USED IN PHASE I

    Export curriculum manifest to a JSON file.

    The output is deterministic: identical curricula produce identical
    manifest files (except for generated_at timestamp).

    Args:
        loader: CurriculumLoaderV2 instance
        path: Output file path for the JSON manifest

    Raises:
        IOError: If file cannot be written
        ValueError: If curriculum is invalid
    """
    manifest = create_curriculum_manifest(loader)
    manifest_dict = manifest.to_dict()

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest_dict, f, indent=2, sort_keys=True)


def load_curriculum_manifest(path: str) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Load a curriculum manifest from JSON file.

    Args:
        path: Path to manifest JSON file

    Returns:
        Parsed manifest dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If manifest is missing required fields
    """
    with open(path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Validate required fields
    required_fields = {
        'schema_version', 'curriculum_hash', 'slice_count',
        'slice_names', 'slices', 'global_preflight_verdict',
        'overall_health', 'issues_total',
    }
    missing = required_fields - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")

    return manifest


# =============================================================================
# LONGITUDINAL DRIFT SURFACE v0.1
# =============================================================================
#
# Compare two curriculum manifests to detect drift between versions.
# This enables tracking curriculum evolution over time.
# =============================================================================

@dataclass
class ManifestDriftReport:
    """
    PHASE II — NOT USED IN PHASE I

    Report of drift between two curriculum manifests.

    Used for longitudinal tracking of curriculum evolution.
    """
    old_manifest_hash: str
    new_manifest_hash: str
    has_material_drift: bool
    slices_added: List[str]
    slices_removed: List[str]
    metric_changes: Dict[str, Dict[str, str]]  # slice -> {old, new}
    band_changes: Dict[str, Dict[str, str]]    # slice -> {old, new}
    verdict_change: Optional[Dict[str, str]]   # {old, new} or None
    health_delta: float
    summary: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'band_changes': dict(sorted(self.band_changes.items())),
            'has_material_drift': self.has_material_drift,
            'health_delta': round(self.health_delta, MANIFEST_FLOAT_PRECISION),
            'metric_changes': dict(sorted(self.metric_changes.items())),
            'new_manifest_hash': self.new_manifest_hash,
            'old_manifest_hash': self.old_manifest_hash,
            'slices_added': self.slices_added,
            'slices_removed': self.slices_removed,
            'summary': self.summary,
            'verdict_change': self.verdict_change,
        }


def compare_manifests(old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> ManifestDriftReport:
    """
    PHASE II — NOT USED IN PHASE I

    Compare two curriculum manifests and produce a drift report.

    Material drift is defined as:
    - Slices added or removed
    - Success metric kind changed
    - Health band changed (Excellent→Good, etc.)
    - Global verdict changed

    Pure function. Deterministic. No side effects.

    Args:
        old_manifest: Previous curriculum manifest dict
        new_manifest: Current curriculum manifest dict

    Returns:
        ManifestDriftReport with detailed drift analysis
    """
    old_slices = set(old_manifest.get('slice_names', []))
    new_slices = set(new_manifest.get('slice_names', []))

    slices_added = sorted(new_slices - old_slices)
    slices_removed = sorted(old_slices - new_slices)

    # Check metric changes for common slices
    metric_changes: Dict[str, Dict[str, str]] = {}
    band_changes: Dict[str, Dict[str, str]] = {}
    common_slices = old_slices & new_slices

    old_slices_data = old_manifest.get('slices', {})
    new_slices_data = new_manifest.get('slices', {})

    for slice_name in sorted(common_slices):
        old_data = old_slices_data.get(slice_name, {})
        new_data = new_slices_data.get(slice_name, {})

        old_metric = old_data.get('success_metric_kind', '')
        new_metric = new_data.get('success_metric_kind', '')
        if old_metric != new_metric:
            metric_changes[slice_name] = {'old': old_metric, 'new': new_metric}

        old_band = old_data.get('health_band', '')
        new_band = new_data.get('health_band', '')
        if old_band != new_band:
            band_changes[slice_name] = {'old': old_band, 'new': new_band}

    # Check verdict change
    old_verdict = old_manifest.get('global_preflight_verdict', '')
    new_verdict = new_manifest.get('global_preflight_verdict', '')
    verdict_change = None
    if old_verdict != new_verdict:
        verdict_change = {'old': old_verdict, 'new': new_verdict}

    # Health delta
    old_health = old_manifest.get('overall_health', 0.0)
    new_health = new_manifest.get('overall_health', 0.0)
    health_delta = new_health - old_health

    # Determine if material drift occurred
    has_material_drift = bool(
        slices_added or slices_removed or
        metric_changes or band_changes or verdict_change
    )

    # Generate summary
    summary: List[str] = []
    if slices_added:
        summary.append(f"Slices added: {', '.join(slices_added)}")
    if slices_removed:
        summary.append(f"Slices removed: {', '.join(slices_removed)}")
    for slice_name, change in sorted(metric_changes.items()):
        summary.append(f"Metric changed for {slice_name}: {change['old']} → {change['new']}")
    for slice_name, change in sorted(band_changes.items()):
        summary.append(f"Band changed for {slice_name}: {change['old']} → {change['new']}")
    if verdict_change:
        summary.append(f"Global verdict changed: {verdict_change['old']} → {verdict_change['new']}")
    if abs(health_delta) > 0.01:
        direction = "improved" if health_delta > 0 else "degraded"
        summary.append(f"Overall health {direction} by {abs(health_delta):.2%}")

    if not summary:
        summary.append("No material drift detected")

    return ManifestDriftReport(
        old_manifest_hash=old_manifest.get('curriculum_hash', ''),
        new_manifest_hash=new_manifest.get('curriculum_hash', ''),
        has_material_drift=has_material_drift,
        slices_added=slices_added,
        slices_removed=slices_removed,
        metric_changes=metric_changes,
        band_changes=band_changes,
        verdict_change=verdict_change,
        health_delta=health_delta,
        summary=summary,
    )


def format_manifest_drift_report(report: ManifestDriftReport) -> str:
    """Format manifest drift report as human-readable text."""
    status = "DRIFT DETECTED" if report.has_material_drift else "NO DRIFT"

    lines = [
        "=" * 70,
        f"LONGITUDINAL CURRICULUM DRIFT REPORT — {status}",
        "=" * 70,
        "",
        f"Old Manifest Hash: {report.old_manifest_hash[:16]}...",
        f"New Manifest Hash: {report.new_manifest_hash[:16]}...",
        f"Health Delta: {report.health_delta:+.2%}",
        "",
        "-" * 70,
        "Summary:",
    ]

    for item in report.summary:
        lines.append(f"  • {item}")

    if report.slices_added:
        lines.append("")
        lines.append("Slices Added:")
        for s in report.slices_added:
            lines.append(f"  + {s}")

    if report.slices_removed:
        lines.append("")
        lines.append("Slices Removed:")
        for s in report.slices_removed:
            lines.append(f"  - {s}")

    if report.band_changes:
        lines.append("")
        lines.append("Band Changes:")
        for slice_name, change in sorted(report.band_changes.items()):
            lines.append(f"  {slice_name}: {change['old']} → {change['new']}")

    if report.metric_changes:
        lines.append("")
        lines.append("Metric Changes:")
        for slice_name, change in sorted(report.metric_changes.items()):
            lines.append(f"  {slice_name}: {change['old']} → {change['new']}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# CURRICULUM HEALTH HINTS LAYER (Advisory Only)
# =============================================================================
#
# Generates non-binding, deterministic, diagnostic hints based on curriculum
# health. These hints are purely advisory and NEVER affect promotion logic.
#
# FORBIDDEN LANGUAGE (to avoid implying normative judgment):
#   - "improve", "better", "worse", "should", "must", "optimize"
#
# Hints are observations, not recommendations.
# =============================================================================

# Threshold for "near borderline" warning
NEAR_BORDERLINE_THRESHOLD = 0.78  # 8% above borderline
NEAR_GOOD_THRESHOLD = 0.88        # 3% above good threshold


def generate_curriculum_hints(preflight: PreflightReport) -> List[str]:
    """
    PHASE II — NOT USED IN PHASE I

    Generate advisory hints from preflight report.

    IMPORTANT: Hints are:
    - Non-binding (purely informational)
    - Deterministic (same input → same output)
    - Diagnostic (observations, not recommendations)
    - Advisory (no normative language)

    This function NEVER uses language like "improve", "better", "should", etc.

    Pure function. Deterministic. No side effects.

    Args:
        preflight: PreflightReport from run_preflight()

    Returns:
        List of hint strings, deterministically ordered
    """
    hints: List[str] = []

    # Check for slices near borderline threshold
    for slice_name in sorted(preflight.slices.keys()):
        result = preflight.slices[slice_name]
        score = result.health_score

        if BAND_BORDERLINE_THRESHOLD <= score < NEAR_BORDERLINE_THRESHOLD:
            hints.append(
                f"Slice '{slice_name}' health ({score:.2%}) is near BORDERLINE threshold; "
                f"metric parameters may warrant review."
            )
        elif BAND_GOOD_THRESHOLD <= score < NEAR_GOOD_THRESHOLD:
            hints.append(
                f"Slice '{slice_name}' health ({score:.2%}) is in lower GOOD range."
            )

    # Check for validation issues
    for slice_name in sorted(preflight.slices.keys()):
        result = preflight.slices[slice_name]
        if not result.validation_passed:
            hints.append(
                f"Slice '{slice_name}' has validation issues; "
                f"schema conformance may need verification."
            )
        if not result.nondegenerate_passed:
            hints.append(
                f"Slice '{slice_name}' non-degeneracy checks flagged; "
                f"experimental parameters may be at edge cases."
            )

    # Global health observations
    if preflight.overall_health >= BAND_EXCELLENT_THRESHOLD:
        all_excellent = all(
            s.health_band == HealthBand.EXCELLENT
            for s in preflight.slices.values()
        )
        if all_excellent:
            hints.append(
                "All slices are in EXCELLENT health band; "
                "curriculum is in stable state."
            )

    # Monotonicity observations
    if preflight.monotonicity_warnings:
        hints.append(
            f"{len(preflight.monotonicity_warnings)} monotonicity warning(s) detected; "
            f"slice ordering may reflect cross-family design constraints."
        )

    # Global verdict observations
    if preflight.global_verdict == PreflightGlobalVerdict.WARN:
        hints.append(
            "Global verdict is WARN; experiments may proceed with heightened monitoring."
        )
    elif preflight.global_verdict == PreflightGlobalVerdict.FAIL:
        hints.append(
            "Global verdict is FAIL; experiment execution is not advised "
            "until issues are addressed."
        )

    # Issue count observation
    if preflight.issues_total == 0:
        hints.append("No issues detected across all slices.")
    elif preflight.issues_total > 5:
        hints.append(
            f"Total of {preflight.issues_total} issues detected across slices; "
            f"aggregate review may be warranted."
        )

    return hints


def format_hints(hints: List[str]) -> str:
    """Format hints as human-readable text."""
    if not hints:
        return "No advisory hints generated."

    lines = [
        "=" * 70,
        "CURRICULUM HEALTH HINTS (Advisory Only)",
        "=" * 70,
        "",
        "The following observations are non-binding and purely diagnostic:",
        "",
    ]

    for i, hint in enumerate(hints, 1):
        lines.append(f"  {i}. {hint}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# CURRICULUM MANIFEST TIMELINE & HISTORY (Phase III)
# =============================================================================
#
# Build longitudinal views of curriculum evolution over multiple runs.
# Enables governance dashboards and drift trend analysis.
# =============================================================================

class HealthTrend(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Health trend classification over time.
    """
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"


class DriftStatus(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Material drift classification for CI gating.
    """
    NONE = "NONE"
    MINOR = "MINOR"
    MAJOR = "MAJOR"


def build_curriculum_manifest_timeline(
    manifest_paths: List[str],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build a timeline from multiple curriculum manifests.

    This function loads multiple manifests, sorts them by generation time,
    and computes longitudinal metrics for governance dashboards.

    Pure function. Deterministic. No side effects.

    Args:
        manifest_paths: List of paths to manifest JSON files

    Returns:
        Dictionary with timeline data:
        - total_versions: Number of manifests in timeline
        - initial_manifest_hash: Hash of first manifest
        - latest_manifest_hash: Hash of most recent manifest
        - global_health_series: List of overall_health values in time order
        - global_verdict_series: List of verdict strings in time order
        - slice_band_transitions: Per-slice sequence of health bands
        - timestamps: List of generated_at values in order

    Raises:
        FileNotFoundError: If a manifest file doesn't exist
        json.JSONDecodeError: If a manifest is invalid JSON
        ValueError: If a manifest is missing required fields or list is empty
    """
    if not manifest_paths:
        raise ValueError("At least one manifest path required")

    # Load all manifests
    manifests: List[Dict[str, Any]] = []
    for path in manifest_paths:
        manifest = load_curriculum_manifest(path)
        manifests.append(manifest)

    # Sort by generated_at (ISO 8601 strings sort lexicographically)
    manifests.sort(key=lambda m: m.get('generated_at', ''))

    # Extract series data
    timestamps: List[str] = []
    health_series: List[float] = []
    verdict_series: List[str] = []

    # Track per-slice band transitions
    slice_band_transitions: Dict[str, List[str]] = {}

    for manifest in manifests:
        timestamps.append(manifest.get('generated_at', ''))
        health_series.append(manifest.get('overall_health', 0.0))
        verdict_series.append(manifest.get('global_preflight_verdict', ''))

        # Track band transitions per slice
        slices_data = manifest.get('slices', {})
        for slice_name in sorted(slices_data.keys()):
            slice_info = slices_data[slice_name]
            band = slice_info.get('health_band', 'unknown')

            if slice_name not in slice_band_transitions:
                slice_band_transitions[slice_name] = []
            slice_band_transitions[slice_name].append(band)

    # Sort slice_band_transitions by slice name for determinism
    slice_band_transitions = dict(sorted(slice_band_transitions.items()))

    return {
        'initial_manifest_hash': manifests[0].get('curriculum_hash', ''),
        'latest_manifest_hash': manifests[-1].get('curriculum_hash', ''),
        'global_health_series': health_series,
        'global_verdict_series': verdict_series,
        'slice_band_transitions': slice_band_transitions,
        'timestamps': timestamps,
        'total_versions': len(manifests),
    }


# =============================================================================
# MATERIAL DRIFT CLASSIFIER FOR CI (Phase III)
# =============================================================================
#
# Classifies drift as NONE/MINOR/MAJOR for CI decision-making.
# This is advisory; actual gating decisions are made by consuming systems.
# =============================================================================

def classify_curriculum_drift(report: ManifestDriftReport) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Classify curriculum drift severity for CI consumption.

    Classification rules:
    - NONE: No material drift detected
    - MINOR: Only band changes within GOOD/EXCELLENT, no verdict change
    - MAJOR: Slices added/removed, metric kind change, or verdict change

    This classification is ADVISORY. Consuming systems (CI, MAAS, governance)
    decide how to act on these classifications.

    Pure function. Deterministic. No side effects.

    Args:
        report: ManifestDriftReport from compare_manifests()

    Returns:
        Dictionary with:
        - drift_status: "NONE" | "MINOR" | "MAJOR"
        - blocking: bool (True if MAJOR drift)
        - reasons: List of neutral explanation strings
    """
    reasons: List[str] = []

    # Check for no drift
    if not report.has_material_drift:
        return {
            'drift_status': DriftStatus.NONE.value,
            'blocking': False,
            'reasons': ['No material drift detected'],
        }

    # Check for MAJOR drift conditions
    is_major = False

    # Slices added or removed is MAJOR
    if report.slices_added:
        is_major = True
        reasons.append(f"Slices added: {', '.join(report.slices_added)}")

    if report.slices_removed:
        is_major = True
        reasons.append(f"Slices removed: {', '.join(report.slices_removed)}")

    # Metric kind change is MAJOR
    if report.metric_changes:
        is_major = True
        for slice_name, change in sorted(report.metric_changes.items()):
            reasons.append(
                f"Metric kind changed for {slice_name}: "
                f"{change['old']} -> {change['new']}"
            )

    # Verdict change is MAJOR
    if report.verdict_change:
        is_major = True
        reasons.append(
            f"Global verdict changed: "
            f"{report.verdict_change['old']} -> {report.verdict_change['new']}"
        )

    if is_major:
        return {
            'drift_status': DriftStatus.MAJOR.value,
            'blocking': True,
            'reasons': reasons,
        }

    # Check for MINOR drift (band changes only, within acceptable bounds)
    if report.band_changes:
        acceptable_bands = {'excellent', 'good'}
        all_acceptable = True

        for slice_name, change in sorted(report.band_changes.items()):
            old_band = change['old'].lower()
            new_band = change['new'].lower()

            # Check if transition is within acceptable bands
            if old_band not in acceptable_bands or new_band not in acceptable_bands:
                all_acceptable = False

            reasons.append(
                f"Band changed for {slice_name}: {change['old']} -> {change['new']}"
            )

        if all_acceptable:
            return {
                'drift_status': DriftStatus.MINOR.value,
                'blocking': False,
                'reasons': reasons,
            }
        else:
            # Band change involves borderline or poor -> MAJOR
            return {
                'drift_status': DriftStatus.MAJOR.value,
                'blocking': True,
                'reasons': reasons,
            }

    # Default: MINOR if we got here with material drift but no specific triggers
    return {
        'drift_status': DriftStatus.MINOR.value,
        'blocking': False,
        'reasons': reasons if reasons else ['Unclassified drift detected'],
    }


# =============================================================================
# GLOBAL HEALTH & MAAS ADAPTERS (Phase III)
# =============================================================================
#
# Adapters for external systems (Global Health Dashboard, MAAS) to consume
# curriculum health data in standardized formats.
# =============================================================================

def _compute_health_trend(health_series: List[float]) -> HealthTrend:
    """
    Compute health trend from a series of health values.

    Uses simple comparison of first and last values with tolerance.
    """
    if len(health_series) < 2:
        return HealthTrend.STABLE

    first = health_series[0]
    last = health_series[-1]
    delta = last - first

    # Tolerance of 2% for stability
    if delta > 0.02:
        return HealthTrend.IMPROVING
    elif delta < -0.02:
        return HealthTrend.DEGRADING
    else:
        return HealthTrend.STABLE


def _find_slices_with_major_band_changes(
    slice_band_transitions: Dict[str, List[str]],
) -> List[str]:
    """
    Find slices that have experienced major band changes.

    A major band change is defined as any transition involving
    'borderline' or 'poor' bands.
    """
    major_change_slices: List[str] = []
    concerning_bands = {'borderline', 'poor'}

    for slice_name, bands in sorted(slice_band_transitions.items()):
        # Check if any band in the sequence is concerning
        has_concerning = any(b.lower() in concerning_bands for b in bands)

        # Check for transitions (different consecutive values)
        has_transition = False
        for i in range(1, len(bands)):
            if bands[i] != bands[i - 1]:
                has_transition = True
                break

        if has_concerning and has_transition:
            major_change_slices.append(slice_name)

    return major_change_slices


def summarize_curriculum_for_global_health(
    timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Summarize curriculum timeline for Global Health Dashboard.

    Produces a compact summary suitable for dashboard display and
    health aggregation across multiple subsystems.

    Pure function. Deterministic. No side effects.

    Args:
        timeline: Output from build_curriculum_manifest_timeline()

    Returns:
        Dictionary with:
        - curriculum_ok: bool (True if latest verdict is OK)
        - latest_verdict: str (most recent verdict)
        - health_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - slices_with_major_band_changes: List of slice names
    """
    health_series = timeline.get('global_health_series', [])
    verdict_series = timeline.get('global_verdict_series', [])
    slice_transitions = timeline.get('slice_band_transitions', {})

    latest_verdict = verdict_series[-1] if verdict_series else 'UNKNOWN'
    curriculum_ok = latest_verdict == 'OK'

    health_trend = _compute_health_trend(health_series)
    major_change_slices = _find_slices_with_major_band_changes(slice_transitions)

    return {
        'curriculum_ok': curriculum_ok,
        'health_trend': health_trend.value,
        'latest_verdict': latest_verdict,
        'slices_with_major_band_changes': major_change_slices,
    }


def summarize_curriculum_for_maas(
    drift_classification: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Summarize drift classification for MAAS (Multi-Agent Assurance System).

    Produces a compact admissibility assessment suitable for MAAS
    decision gates and audit logs.

    Pure function. Deterministic. No side effects.

    Args:
        drift_classification: Output from classify_curriculum_drift()

    Returns:
        Dictionary with:
        - is_curriculum_admissible: bool (True if not blocking)
        - drift_status: "NONE" | "MINOR" | "MAJOR"
        - blocking_reasons: List of reasons if blocking, empty otherwise
    """
    drift_status = drift_classification.get('drift_status', 'UNKNOWN')
    is_blocking = drift_classification.get('blocking', False)
    reasons = drift_classification.get('reasons', [])

    return {
        'blocking_reasons': reasons if is_blocking else [],
        'drift_status': drift_status,
        'is_curriculum_admissible': not is_blocking,
    }


def format_timeline_summary(timeline: Dict[str, Any]) -> str:
    """Format timeline summary as human-readable text."""
    lines = [
        "=" * 70,
        "CURRICULUM MANIFEST TIMELINE",
        "=" * 70,
        "",
        f"Total Versions: {timeline.get('total_versions', 0)}",
        f"Initial Hash: {timeline.get('initial_manifest_hash', '')[:16]}...",
        f"Latest Hash: {timeline.get('latest_manifest_hash', '')[:16]}...",
        "",
        "-" * 70,
        "Health Series:",
    ]

    health_series = timeline.get('global_health_series', [])
    timestamps = timeline.get('timestamps', [])

    for i, (ts, health) in enumerate(zip(timestamps, health_series)):
        lines.append(f"  {i + 1}. {ts}: {health:.2%}")

    verdict_series = timeline.get('global_verdict_series', [])
    lines.append("")
    lines.append("Verdict Series:")
    for i, (ts, verdict) in enumerate(zip(timestamps, verdict_series)):
        lines.append(f"  {i + 1}. {ts}: {verdict}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_drift_classification(classification: Dict[str, Any]) -> str:
    """Format drift classification as human-readable text."""
    status = classification.get('drift_status', 'UNKNOWN')
    blocking = classification.get('blocking', False)
    reasons = classification.get('reasons', [])

    status_symbol = "✗" if blocking else "✓"

    lines = [
        "=" * 70,
        f"DRIFT CLASSIFICATION: {status_symbol} {status}",
        "=" * 70,
        "",
        f"Blocking: {'Yes' if blocking else 'No'}",
        "",
        "Reasons:",
    ]

    for reason in reasons:
        lines.append(f"  • {reason}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# PHASE IV — CROSS-SYSTEM CURRICULUM SPINE & ACQUISITION-GRADE CHRONICLE
# =============================================================================
#
# These functions integrate curriculum health with other system components
# (metrics, topology, confusability) to provide unified governance views
# and acquisition-ready summaries.
# =============================================================================

class AlignmentStatus(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Global alignment status across curriculum, metrics, topology, and confusability.
    """
    ALIGNED = "ALIGNED"
    PARTIAL = "PARTIAL"
    MISALIGNED = "MISALIGNED"


class StatusLight(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Status light for Director Console visualization.
    """
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ChangeFrequencyBand(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Change frequency classification for acquisition chronicle.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskProfile(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Risk profile classification based on MAJOR drift frequency.
    """
    CONSERVATIVE = "CONSERVATIVE"
    ACTIVE = "ACTIVE"
    AGGRESSIVE = "AGGRESSIVE"


def build_curriculum_alignment_view(
    manifest_timeline: Dict[str, Any],
    metric_conformance: Dict[str, Any],
    confusability_risk: Dict[str, Any],
    topology_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build cross-system curriculum alignment view.

    This function integrates curriculum health with:
    - Metric conformance (from B1/B2/B5-style snapshots)
    - Confusability risk (from C2's family risk snapshot)
    - Topology health (from C5's ledger analytics)

    The alignment view identifies slices where all systems are structurally
    aligned vs. slices with tension between systems.

    Pure function. Deterministic. No side effects.

    Args:
        manifest_timeline: Output from build_curriculum_manifest_timeline()
        metric_conformance: Dict with per-slice conformance status
        confusability_risk: Dict with per-slice risk assessments
        topology_health: Dict with per-slice topology health

    Returns:
        Dictionary with:
        - slices_structurally_stable: List of slice names with full alignment
        - slices_with_metric_structural_tension: List of slice names with misalignment
        - global_alignment_status: "ALIGNED" | "PARTIAL" | "MISALIGNED"
        - per_slice_alignment: Dict mapping slice names to alignment details
    """
    slice_band_transitions = manifest_timeline.get('slice_band_transitions', {})
    all_slices = set(slice_band_transitions.keys())

    # Extract per-slice status from external systems
    # Expected structure (flexible to accommodate different formats):
    # metric_conformance: {'slices': {slice_name: {'status': 'OK'|'WARN'|'FAIL'}}}
    # confusability_risk: {'slices': {slice_name: {'risk_level': 'LOW'|'MEDIUM'|'HIGH'}}}
    # topology_health: {'slices': {slice_name: {'health': 'HEALTHY'|'DEGRADED'|'CRITICAL'}}}

    metric_slices = set(metric_conformance.get('slices', {}).keys())
    confusability_slices = set(confusability_risk.get('slices', {}).keys())
    topology_slices = set(topology_health.get('slices', {}).keys())

    # Determine alignment per slice
    per_slice_alignment: Dict[str, Dict[str, Any]] = {}
    slices_structurally_stable: List[str] = []
    slices_with_metric_structural_tension: List[str] = []

    for slice_name in sorted(all_slices):
        # Get current health band (latest in transition sequence)
        band_transitions = slice_band_transitions.get(slice_name, [])
        current_band = band_transitions[-1] if band_transitions else 'unknown'

        # Check metric conformance
        metric_data = metric_conformance.get('slices', {}).get(slice_name, {})
        metric_status = metric_data.get('status', 'UNKNOWN')
        metric_aligned = metric_status in ('OK', 'PASS', 'HEALTHY')

        # Check confusability risk
        confusability_data = confusability_risk.get('slices', {}).get(slice_name, {})
        risk_level = confusability_data.get('risk_level', 'UNKNOWN')
        confusability_aligned = risk_level in ('LOW', 'NONE', 'MINIMAL')

        # Check topology health
        topology_data = topology_health.get('slices', {}).get(slice_name, {})
        topology_status = topology_data.get('health', 'UNKNOWN')
        topology_aligned = topology_status in ('HEALTHY', 'OK', 'STABLE')

        # Determine if slice is fully aligned
        is_aligned = metric_aligned and confusability_aligned and topology_aligned

        per_slice_alignment[slice_name] = {
            'current_health_band': current_band,
            'metric_aligned': metric_aligned,
            'confusability_aligned': confusability_aligned,
            'topology_aligned': topology_aligned,
            'is_fully_aligned': is_aligned,
        }

        if is_aligned:
            slices_structurally_stable.append(slice_name)
        else:
            slices_with_metric_structural_tension.append(slice_name)

    # Determine global alignment status
    total_slices = len(all_slices)
    aligned_count = len(slices_structurally_stable)

    if aligned_count == total_slices and total_slices > 0:
        global_status = AlignmentStatus.ALIGNED
    elif aligned_count > 0:
        global_status = AlignmentStatus.PARTIAL
    else:
        global_status = AlignmentStatus.MISALIGNED

    return {
        'global_alignment_status': global_status.value,
        'per_slice_alignment': per_slice_alignment,
        'slices_structurally_stable': slices_structurally_stable,
        'slices_with_metric_structural_tension': slices_with_metric_structural_tension,
    }


def build_curriculum_director_panel(
    manifest_timeline: Dict[str, Any],
    drift_classification: Dict[str, Any],
    alignment_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build curriculum governance panel for Director Console.

    This produces a single-tile summary suitable for executive dashboard
    visualization. All text is neutral and descriptive.

    Pure function. Deterministic. No side effects.

    Args:
        manifest_timeline: Output from build_curriculum_manifest_timeline()
        drift_classification: Output from classify_curriculum_drift()
        alignment_view: Output from build_curriculum_alignment_view()

    Returns:
        Dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - latest_verdict: Most recent preflight verdict
        - health_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - alignment_status: "ALIGNED" | "PARTIAL" | "MISALIGNED"
        - slices_of_concern: List of slice names requiring attention
        - headline: Short neutral descriptive sentence
    """
    # Get latest verdict from timeline
    verdict_series = manifest_timeline.get('global_verdict_series', [])
    latest_verdict = verdict_series[-1] if verdict_series else 'UNKNOWN'

    # Get health trend
    health_series = manifest_timeline.get('global_health_series', [])
    health_trend = _compute_health_trend(health_series)

    # Get alignment status
    alignment_status = alignment_view.get('global_alignment_status', 'UNKNOWN')

    # Get drift status
    drift_status = drift_classification.get('drift_status', 'UNKNOWN')
    is_blocking = drift_classification.get('blocking', False)

    # Collect slices of concern
    slices_of_concern: List[str] = []
    slices_of_concern.extend(
        alignment_view.get('slices_with_metric_structural_tension', [])
    )

    # Add slices with major band changes
    slice_transitions = manifest_timeline.get('slice_band_transitions', {})
    for slice_name, bands in sorted(slice_transitions.items()):
        if len(bands) > 1:
            # Check for transitions involving concerning bands
            concerning_bands = {'borderline', 'poor'}
            if any(b.lower() in concerning_bands for b in bands):
                if slice_name not in slices_of_concern:
                    slices_of_concern.append(slice_name)

    # Determine status light
    if (
        latest_verdict == 'OK' and
        alignment_status == 'ALIGNED' and
        drift_status != 'MAJOR' and
        health_trend != HealthTrend.DEGRADING
    ):
        status_light = StatusLight.GREEN
    elif (
        latest_verdict == 'FAIL' or
        alignment_status == 'MISALIGNED' or
        is_blocking
    ):
        status_light = StatusLight.RED
    else:
        status_light = StatusLight.YELLOW

    # Generate neutral headline
    headline_parts = []
    if latest_verdict:
        headline_parts.append(f"Verdict: {latest_verdict}")
    if health_trend != HealthTrend.STABLE:
        headline_parts.append(f"Health trend: {health_trend.value.lower()}")
    if alignment_status != 'ALIGNED':
        headline_parts.append(f"Alignment: {alignment_status.lower()}")

    if not headline_parts:
        headline = "Curriculum status: stable across all systems"
    else:
        headline = "Curriculum status: " + ", ".join(headline_parts)

    return {
        'alignment_status': alignment_status,
        'headline': headline,
        'health_trend': health_trend.value,
        'latest_verdict': latest_verdict,
        'slices_of_concern': sorted(slices_of_concern),
        'status_light': status_light.value,
    }


def build_curriculum_chronicle_for_acquisition(
    manifest_timeline: Dict[str, Any],
    drift_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build acquisition-facing curriculum chronicle.

    This produces a high-level, executive-friendly summary of curriculum
    evolution suitable for technical due diligence and acquisition discussions.

    The chronicle is designed to be:
    - Human-readable (technical executive audience)
    - Machine-readable (fully deterministic JSON)
    - Descriptive (no normative judgments)

    Pure function. Deterministic. No side effects.

    Args:
        manifest_timeline: Output from build_curriculum_manifest_timeline()
        drift_events: List of drift classification dicts from classify_curriculum_drift()

    Returns:
        Dictionary with:
        - total_versions: Number of curriculum versions tracked
        - change_frequency_band: "LOW" | "MEDIUM" | "HIGH"
        - risk_profile: "CONSERVATIVE" | "ACTIVE" | "AGGRESSIVE"
        - overall_health_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - drift_summary: Counts of NONE/MINOR/MAJOR drift events
        - stability_indicators: Key stability metrics
    """
    total_versions = manifest_timeline.get('total_versions', 0)
    health_series = manifest_timeline.get('global_health_series', [])
    overall_trend = _compute_health_trend(health_series)

    # Analyze drift events
    drift_counts = {'NONE': 0, 'MINOR': 0, 'MAJOR': 0}
    blocking_count = 0

    for event in drift_events:
        status = event.get('drift_status', 'UNKNOWN')
        if status in drift_counts:
            drift_counts[status] += 1
        if event.get('blocking', False):
            blocking_count += 1

    total_drift_events = len(drift_events)
    major_drift_ratio = (
        drift_counts['MAJOR'] / total_drift_events
        if total_drift_events > 0 else 0.0
    )

    # Determine change frequency band
    # Based on number of versions relative to time period
    # For simplicity, we use version count as proxy
    if total_versions <= 3:
        change_frequency = ChangeFrequencyBand.LOW
    elif total_versions <= 10:
        change_frequency = ChangeFrequencyBand.MEDIUM
    else:
        change_frequency = ChangeFrequencyBand.HIGH

    # Determine risk profile based on MAJOR drift frequency
    if major_drift_ratio < 0.2:
        risk_profile = RiskProfile.CONSERVATIVE
    elif major_drift_ratio < 0.5:
        risk_profile = RiskProfile.ACTIVE
    else:
        risk_profile = RiskProfile.AGGRESSIVE

    # Calculate stability indicators
    initial_health = health_series[0] if health_series else 0.0
    latest_health = health_series[-1] if health_series else 0.0
    health_delta = latest_health - initial_health

    # Count slices with stable bands (no transitions)
    slice_transitions = manifest_timeline.get('slice_band_transitions', {})
    stable_slices = sum(
        1 for bands in slice_transitions.values()
        if len(set(bands)) == 1  # All bands are the same
    )

    return {
        'change_frequency_band': change_frequency.value,
        'drift_summary': {
            'total_events': total_drift_events,
            'none_count': drift_counts['NONE'],
            'minor_count': drift_counts['MINOR'],
            'major_count': drift_counts['MAJOR'],
            'blocking_events': blocking_count,
        },
        'overall_health_trend': overall_trend.value,
        'risk_profile': risk_profile.value,
        'stability_indicators': {
            'health_delta': round(health_delta, 4),
            'initial_health': round(initial_health, 4),
            'latest_health': round(latest_health, 4),
            'slices_with_stable_bands': stable_slices,
            'total_slices': len(slice_transitions),
        },
        'total_versions': total_versions,
    }


# =============================================================================
# PHASE IV FOLLOW-UP — CONVERGENCE MAP & PHASE-BOUNDARY FORECASTER
# =============================================================================
#
# Predictive control across slices and systems for curriculum governance.
# =============================================================================

class ConvergenceStatus(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Convergence status across curriculum, metrics, topology, and confusability.
    """
    CONVERGING = "CONVERGING"
    STABLE = "STABLE"
    DIVERGING = "DIVERGING"


def build_curriculum_convergence_map(
    alignment_view: Dict[str, Any],
    drift_timeline: Dict[str, Any],
    metric_trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build cross-system curriculum convergence map.

    Analyzes whether curriculum, metrics, topology, and confusability systems
    are converging (improving alignment), stable, or diverging (increasing tension).

    Convergence criteria:
    - CONVERGING: Alignment improving AND drift decreasing
    - STABLE: Alignment stable AND drift stable
    - DIVERGING: Alignment degrading OR multi-system tension detected

    Pure function. Deterministic. No side effects.

    Args:
        alignment_view: Output from build_curriculum_alignment_view()
        drift_timeline: Dict with drift events over time (e.g., {'events': [...]})
        metric_trajectory: Dict with metric conformance trajectory (e.g., {'trend': 'IMPROVING'})

    Returns:
        Dictionary with:
        - convergence_status: "CONVERGING" | "STABLE" | "DIVERGING"
        - slices_converging: List of slice names showing convergence
        - slices_diverging: List of slice names showing divergence
        - cross_signal_correlations: Dict mapping signal pairs to correlation scores
        - summary: Neutral descriptive summary string
    """
    # Analyze alignment trend
    alignment_status = alignment_view.get('global_alignment_status', 'UNKNOWN')
    stable_slices = set(alignment_view.get('slices_structurally_stable', []))
    tension_slices = set(alignment_view.get('slices_with_metric_structural_tension', []))

    # Analyze drift trend from timeline
    drift_events = drift_timeline.get('events', [])
    recent_drift = drift_events[-5:] if len(drift_events) >= 5 else drift_events
    older_drift = drift_events[:-5] if len(drift_events) >= 10 else drift_events[:len(drift_events)//2]

    # Count MAJOR drift in recent vs older periods
    recent_major = sum(1 for e in recent_drift if e.get('drift_status') == 'MAJOR')
    older_major = sum(1 for e in older_drift if e.get('drift_status') == 'MAJOR')

    drift_decreasing = recent_major < older_major if older_major > 0 else (recent_major == 0)
    drift_stable = recent_major == older_major if older_major > 0 else (recent_major == 0)

    # Analyze metric trajectory
    metric_trend = metric_trajectory.get('trend', 'UNKNOWN')
    metric_improving = metric_trend in ('IMPROVING', 'STABLE')

    # Determine per-slice convergence
    per_slice_alignment = alignment_view.get('per_slice_alignment', {})
    slices_converging: List[str] = []
    slices_diverging: List[str] = []

    for slice_name, alignment_data in sorted(per_slice_alignment.items()):
        is_aligned = alignment_data.get('is_fully_aligned', False)

        # Check for multi-system tension
        metric_aligned = alignment_data.get('metric_aligned', False)
        confusability_aligned = alignment_data.get('confusability_aligned', False)
        topology_aligned = alignment_data.get('topology_aligned', False)

        # Diverging: not aligned AND multiple systems misaligned
        if not is_aligned:
            misaligned_count = sum([
                not metric_aligned,
                not confusability_aligned,
                not topology_aligned,
            ])
            if misaligned_count >= 2:
                slices_diverging.append(slice_name)
            elif is_aligned:  # Was aligned, now not
                slices_diverging.append(slice_name)
        elif is_aligned and slice_name in tension_slices:
            # Was in tension, now aligned
            slices_converging.append(slice_name)

    # Determine global convergence status
    alignment_improving = (
        alignment_status == 'ALIGNED' or
        (alignment_status == 'PARTIAL' and len(stable_slices) > len(tension_slices))
    )
    alignment_degrading = (
        alignment_status == 'MISALIGNED' or
        (alignment_status == 'PARTIAL' and len(tension_slices) > len(stable_slices))
    )

    if alignment_improving and drift_decreasing and metric_improving:
        convergence_status = ConvergenceStatus.CONVERGING
    elif alignment_degrading or len(slices_diverging) > len(slices_converging):
        convergence_status = ConvergenceStatus.DIVERGING
    else:
        convergence_status = ConvergenceStatus.STABLE

    # Compute cross-signal correlations (simplified correlation scores)
    # Correlation between alignment and drift: negative correlation expected
    # (better alignment = less drift)
    alignment_score = (
        1.0 if alignment_status == 'ALIGNED' else
        0.5 if alignment_status == 'PARTIAL' else
        0.0
    )
    drift_score = 1.0 - (recent_major / max(len(recent_drift), 1))
    metrics_alignment_correlation = (alignment_score + drift_score) / 2.0

    # Correlation between topology and confusability
    # Count aligned slices for each system
    topology_aligned_count = sum(
        1 for a in per_slice_alignment.values()
        if a.get('topology_aligned', False)
    )
    confusability_aligned_count = sum(
        1 for a in per_slice_alignment.values()
        if a.get('confusability_aligned', False)
    )
    total_slices = len(per_slice_alignment) if per_slice_alignment else 1
    topology_ratio = topology_aligned_count / total_slices
    confusability_ratio = confusability_aligned_count / total_slices
    topology_confusability_correlation = (topology_ratio + confusability_ratio) / 2.0

    cross_signal_correlations = {
        'metrics↔topology': round(metrics_alignment_correlation, 4),
        'topology↔confusability': round(topology_confusability_correlation, 4),
    }

    # Generate neutral summary
    summary_parts = []
    summary_parts.append(f"Convergence status: {convergence_status.value.lower()}")
    if slices_converging:
        summary_parts.append(f"{len(slices_converging)} slice(s) converging")
    if slices_diverging:
        summary_parts.append(f"{len(slices_diverging)} slice(s) diverging")
    summary_parts.append(f"Alignment: {alignment_status.lower()}")

    summary = ". ".join(summary_parts) + "."

    return {
        'convergence_status': convergence_status.value,
        'cross_signal_correlations': cross_signal_correlations,
        'slices_converging': sorted(slices_converging),
        'slices_diverging': sorted(slices_diverging),
        'summary': summary,
    }


def forecast_curriculum_phase_boundary(
    convergence_map: Dict[str, Any],
    horizon: int = 10,
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Forecast curriculum phase boundary transitions.

    Predicts when curriculum is likely to cross alignment phase boundaries:
    - STABLE → PARTIAL
    - PARTIAL → MISALIGNED
    - MISALIGNED → RECOVERING (back to PARTIAL or ALIGNED)

    This is a structural prediction based on convergence trends, not a
    statistical forecast. Confidence is based on signal strength.

    Pure function. Deterministic. No side effects.

    Args:
        convergence_map: Output from build_curriculum_convergence_map()
        horizon: Number of versions ahead to forecast (default: 10)

    Returns:
        Dictionary with:
        - predicted_boundary: "STABLE→PARTIAL" | "PARTIAL→MISALIGNED" | "MISALIGNED→RECOVERING" | None
        - confidence: float (0.0 to 1.0)
        - reasons: List of neutral explanation strings
        - estimated_versions_until: int or None
    """
    convergence_status = convergence_map.get('convergence_status', 'UNKNOWN')
    slices_converging = convergence_map.get('slices_converging', [])
    slices_diverging = convergence_map.get('slices_diverging', [])
    correlations = convergence_map.get('cross_signal_correlations', {})

    reasons: List[str] = []
    predicted_boundary = None
    confidence = 0.0
    estimated_versions_until = None

    # Analyze convergence signals
    converging_count = len(slices_converging)
    diverging_count = len(slices_diverging)
    total_slices = converging_count + diverging_count

    if total_slices == 0:
        # No slices to analyze
        return {
            'confidence': 0.0,
            'estimated_versions_until': None,
            'predicted_boundary': None,
            'reasons': ['Insufficient slice data for boundary prediction'],
        }

    divergence_ratio = diverging_count / total_slices
    convergence_ratio = converging_count / total_slices

    # Check correlation strength
    avg_correlation = (
        sum(correlations.values()) / len(correlations)
        if correlations else 0.5
    )

    # Predict boundary transitions based on convergence status
    if convergence_status == 'DIVERGING':
        # Likely moving toward MISALIGNED
        if divergence_ratio > 0.5:
            predicted_boundary = 'PARTIAL→MISALIGNED'
            confidence = min(0.9, 0.5 + divergence_ratio)
            reasons.append(f"Divergence ratio ({divergence_ratio:.1%}) indicates potential misalignment")
            reasons.append(f"{diverging_count} slice(s) showing divergence signals")
            # Estimate: faster divergence = fewer versions
            estimated_versions_until = max(1, int(horizon * (1 - divergence_ratio)))
        else:
            predicted_boundary = 'STABLE→PARTIAL'
            confidence = 0.4 + (divergence_ratio * 0.3)
            reasons.append("Moderate divergence signals detected")
            estimated_versions_until = horizon

    elif convergence_status == 'CONVERGING':
        # Likely recovering toward ALIGNED
        if convergence_ratio > 0.6:
            predicted_boundary = 'MISALIGNED→RECOVERING'
            confidence = min(0.85, 0.5 + convergence_ratio)
            reasons.append(f"Convergence ratio ({convergence_ratio:.1%}) indicates recovery potential")
            reasons.append(f"{converging_count} slice(s) showing convergence signals")
            estimated_versions_until = max(1, int(horizon * (1 - convergence_ratio)))
        else:
            predicted_boundary = 'PARTIAL→ALIGNED'
            confidence = 0.4 + (convergence_ratio * 0.3)
            reasons.append("Moderate convergence signals detected")
            estimated_versions_until = horizon

    else:  # STABLE
        # Check for weak signals that might indicate future change
        if divergence_ratio > 0.3:
            predicted_boundary = 'STABLE→PARTIAL'
            confidence = 0.3 + (divergence_ratio * 0.2)
            reasons.append("Stable state with emerging divergence signals")
            estimated_versions_until = horizon * 2  # Longer horizon for stable state
        elif convergence_ratio > 0.3:
            predicted_boundary = 'PARTIAL→ALIGNED'
            confidence = 0.3 + (convergence_ratio * 0.2)
            reasons.append("Stable state with emerging convergence signals")
            estimated_versions_until = horizon * 2
        else:
            # No clear prediction
            reasons.append("Stable state with no strong transition signals")
            confidence = 0.2

    # Adjust confidence based on correlation strength
    if avg_correlation < 0.3:
        confidence *= 0.7  # Lower confidence if signals are weak
        reasons.append("Weak cross-signal correlations reduce prediction confidence")
    elif avg_correlation > 0.7:
        confidence = min(1.0, confidence * 1.1)  # Slightly boost if strong correlations
        reasons.append("Strong cross-signal correlations support prediction")

    # Cap confidence
    confidence = min(1.0, max(0.0, confidence))

    if not reasons:
        reasons.append("Insufficient signal strength for boundary prediction")

    return {
        'confidence': round(confidence, 4),
        'estimated_versions_until': estimated_versions_until,
        'predicted_boundary': predicted_boundary,
        'reasons': reasons,
    }


# =============================================================================
# PHASE V — CONVERGENCE PRESSURE GRID & EARLY-WARNING RADAR
# =============================================================================
#
# Governance-grade pressure tensor and phase-transition early-warning system
# for A-series, B-series, and TDA-level curriculum gates.
# =============================================================================

class TransitionLikelihoodBand(str, Enum):
    """
    PHASE II — NOT USED IN PHASE I

    Likelihood band for phase transitions.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Pressure tensor schema version
PRESSURE_TENSOR_SCHEMA_VERSION = "1.0.0"


def build_convergence_pressure_tensor(
    convergence_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build convergence pressure tensor from convergence map.

    The pressure tensor quantifies structural pressure on each slice across
    three dimensions: alignment, drift, and metric conformance.

    Pressure vectors are normalized to [0, 1] where:
    - 0.0 = no pressure (fully aligned, no drift, metrics OK)
    - 1.0 = maximum pressure (misaligned, high drift, metrics failing)

    Global pressure norm is computed via L2 norm (Euclidean distance).

    Pure function. Deterministic. No side effects.

    Args:
        convergence_map: Output from build_curriculum_convergence_map()

    Returns:
        Dictionary with:
        - slice_pressure_vectors: Per-slice 3D pressure vectors
        - global_pressure_norm: L2 norm of all pressure vectors
        - pressure_ranked_slices: Slices ranked by descending pressure norm
        - schema_version: "1.0.0"
    """
    convergence_status = convergence_map.get('convergence_status', 'UNKNOWN')
    slices_converging = set(convergence_map.get('slices_converging', []))
    slices_diverging = set(convergence_map.get('slices_diverging', []))
    correlations = convergence_map.get('cross_signal_correlations', {})

    # Collect all slices from convergence and divergence sets
    all_slices = sorted(slices_converging | slices_diverging)

    # If no slices in convergence map, return empty tensor
    if not all_slices:
        return {
            'global_pressure_norm': 0.0,
            'pressure_ranked_slices': [],
            'schema_version': PRESSURE_TENSOR_SCHEMA_VERSION,
            'slice_pressure_vectors': {},
        }

    slice_pressure_vectors: Dict[str, Dict[str, float]] = {}

    # Compute pressure for each slice
    for slice_name in all_slices:
        # Alignment pressure: 0.0 if converging, 1.0 if diverging, 0.5 if neither
        if slice_name in slices_converging:
            alignment_pressure = 0.0
        elif slice_name in slices_diverging:
            alignment_pressure = 1.0
        else:
            alignment_pressure = 0.5

        # Drift pressure: inverse of correlation strength
        # Higher correlation = lower drift pressure
        avg_correlation = (
            sum(correlations.values()) / len(correlations)
            if correlations else 0.5
        )
        drift_pressure = 1.0 - avg_correlation

        # Metric pressure: based on convergence status
        # CONVERGING = low pressure, DIVERGING = high pressure, STABLE = medium
        if convergence_status == 'CONVERGING':
            metric_pressure = 0.2
        elif convergence_status == 'DIVERGING':
            metric_pressure = 0.8
        else:  # STABLE
            metric_pressure = 0.5

        # Store normalized pressure vector
        slice_pressure_vectors[slice_name] = {
            'alignment': round(alignment_pressure, 4),
            'drift': round(drift_pressure, 4),
            'metric': round(metric_pressure, 4),
        }

    # Compute L2 norm for each slice (Euclidean distance from origin)
    slice_norms: Dict[str, float] = {}
    for slice_name, vector in slice_pressure_vectors.items():
        norm = (
            vector['alignment'] ** 2 +
            vector['drift'] ** 2 +
            vector['metric'] ** 2
        ) ** 0.5
        slice_norms[slice_name] = round(norm, 4)

    # Compute global pressure norm (L2 norm of all vectors combined)
    # This is the Euclidean norm of the concatenated vector
    all_pressures = []
    for vector in slice_pressure_vectors.values():
        all_pressures.extend([vector['alignment'], vector['drift'], vector['metric']])

    if all_pressures:
        global_norm = (sum(p ** 2 for p in all_pressures)) ** 0.5
        global_norm = round(global_norm, 4)
    else:
        global_norm = 0.0

    # Rank slices by descending pressure norm
    pressure_ranked_slices = sorted(
        slice_norms.keys(),
        key=lambda s: slice_norms[s],
        reverse=True
    )

    return {
        'global_pressure_norm': global_norm,
        'pressure_ranked_slices': pressure_ranked_slices,
        'schema_version': PRESSURE_TENSOR_SCHEMA_VERSION,
        'slice_pressure_vectors': {
            k: dict(sorted(v.items()))
            for k, v in sorted(slice_pressure_vectors.items())
        },
    }


def build_phase_transition_early_warning_radar(
    pressure_tensor: Dict[str, Any],
    phase_forecast: Dict[str, Any],
    drift_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build phase-transition early-warning radar.

    Combines pressure tensor, phase-boundary forecast, and drift trends
    to produce a comprehensive early-warning assessment.

    Pure function. Deterministic. No side effects.

    Args:
        pressure_tensor: Output from build_convergence_pressure_tensor()
        phase_forecast: Output from forecast_curriculum_phase_boundary()
        drift_timeline: Dict with drift events over time

    Returns:
        Dictionary with:
        - transition_likelihood_band: "LOW" | "MEDIUM" | "HIGH"
        - root_drivers: List of neutral descriptive strings
        - first_slices_at_risk: List of slice names (sorted)
        - time_to_inflection_estimate: int or None
    """
    global_pressure = pressure_tensor.get('global_pressure_norm', 0.0)
    forecast_confidence = phase_forecast.get('confidence', 0.0)
    predicted_boundary = phase_forecast.get('predicted_boundary')
    estimated_versions = phase_forecast.get('estimated_versions_until')

    # Analyze drift trend
    drift_events = drift_timeline.get('events', [])
    recent_drift = drift_events[-5:] if len(drift_events) >= 5 else drift_events
    recent_major = sum(1 for e in recent_drift if e.get('drift_status') == 'MAJOR')

    # Determine transition likelihood band
    # Factors: pressure norm, forecast confidence, drift trend
    pressure_factor = min(1.0, global_pressure / 2.0)  # Normalize to [0, 1]
    confidence_factor = forecast_confidence
    drift_factor = min(1.0, recent_major / 3.0)  # 3+ MAJOR events = high

    combined_score = (pressure_factor * 0.4 + confidence_factor * 0.4 + drift_factor * 0.2)

    if combined_score >= 0.7:
        likelihood_band = TransitionLikelihoodBand.HIGH
    elif combined_score >= 0.4:
        likelihood_band = TransitionLikelihoodBand.MEDIUM
    else:
        likelihood_band = TransitionLikelihoodBand.LOW

    # Identify root drivers (neutral descriptions)
    root_drivers: List[str] = []

    if global_pressure > 1.5:
        root_drivers.append(f"Global pressure norm elevated ({global_pressure:.2f})")
    if forecast_confidence > 0.7:
        root_drivers.append(f"Phase boundary forecast confidence high ({forecast_confidence:.1%})")
    if recent_major >= 2:
        root_drivers.append(f"Recent drift events include {recent_major} MAJOR classification(s)")
    if predicted_boundary:
        root_drivers.append(f"Predicted boundary transition: {predicted_boundary}")

    if not root_drivers:
        root_drivers.append("No strong transition signals detected")

    # Identify first slices at risk (top pressure-ranked slices)
    pressure_ranked = pressure_tensor.get('pressure_ranked_slices', [])
    pressure_vectors = pressure_tensor.get('slice_pressure_vectors', {})

    # Filter to slices with pressure norm > 0.5
    slices_at_risk = [
        s for s in pressure_ranked
        if s in pressure_vectors
    ]

    # Take top 3 highest pressure slices
    first_slices_at_risk = slices_at_risk[:3]

    # Time to inflection estimate
    time_to_inflection = estimated_versions if estimated_versions is not None else None

    return {
        'first_slices_at_risk': sorted(first_slices_at_risk),
        'root_drivers': root_drivers,
        'time_to_inflection_estimate': time_to_inflection,
        'transition_likelihood_band': likelihood_band.value,
    }


def build_convergence_director_tile(
    pressure_tensor: Dict[str, Any],
    early_warning: Dict[str, Any],
) -> Dict[str, Any]:
    """
    PHASE II — NOT USED IN PHASE I

    Build director-level tile for convergence pressure visualization.

    This produces a single-tile summary suitable for executive dashboard
    display, combining pressure tensor and early-warning radar data.

    Pure function. Deterministic. No side effects.

    Args:
        pressure_tensor: Output from build_convergence_pressure_tensor()
        early_warning: Output from build_phase_transition_early_warning_radar()

    Returns:
        Dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - transition_band: "LOW" | "MEDIUM" | "HIGH"
        - global_pressure_norm: float
        - headline: Neutral descriptive string
        - pressure_drivers: List of neutral descriptive strings
    """
    global_pressure = pressure_tensor.get('global_pressure_norm', 0.0)
    transition_band = early_warning.get('transition_likelihood_band', 'UNKNOWN')
    root_drivers = early_warning.get('root_drivers', [])
    first_slices = early_warning.get('first_slices_at_risk', [])

    # Determine status light
    if global_pressure < 1.0 and transition_band == 'LOW':
        status_light = StatusLight.GREEN
    elif global_pressure > 2.0 or transition_band == 'HIGH':
        status_light = StatusLight.RED
    else:
        status_light = StatusLight.YELLOW

    # Generate neutral headline
    headline_parts = []
    headline_parts.append(f"Pressure norm: {global_pressure:.2f}")
    headline_parts.append(f"Transition likelihood: {transition_band.lower()}")
    if first_slices:
        headline_parts.append(f"{len(first_slices)} slice(s) at elevated risk")

    headline = "Convergence status: " + ", ".join(headline_parts)

    # Extract pressure drivers from root drivers (limit to 3)
    pressure_drivers = root_drivers[:3]

    return {
        'global_pressure_norm': round(global_pressure, 4),
        'headline': headline,
        'pressure_drivers': pressure_drivers,
        'status_light': status_light.value,
        'transition_band': transition_band,
    }


# =============================================================================
# CLI OUTPUT FORMATTERS
# =============================================================================

def format_slice_list(loader: CurriculumLoaderV2) -> str:
    """Format slice list as human-readable text."""
    slices = loader.list_slices()
    lines = [
        "=" * 70,
        "PHASE II CURRICULUM SLICES",
        "=" * 70,
        f"Version: {loader.get_version()}",
        f"Total: {len(slices)} slices",
        "",
        "-" * 70,
        f"{'Slice Name':<30} {'Atoms':<6} {'Depth':<10} {'Metric':<20}",
        "-" * 70,
    ]

    for slice_name in slices:
        params = loader.get_parameters(slice_name)
        metric = loader.get_success_metric_config(slice_name)
        depth_range = f"{params['depth_min']}-{params['depth_max']}"
        lines.append(
            f"{slice_name:<30} {params['atoms']:<6} {depth_range:<10} {metric['kind']:<20}"
        )

    lines.append("-" * 70)
    return "\n".join(lines)


def format_metrics_summary(loader: CurriculumLoaderV2) -> str:
    """Format success metrics summary."""
    lines = [
        "=" * 70,
        "SUCCESS METRICS SUMMARY",
        "=" * 70,
        "",
    ]

    for slice_name in loader.list_slices():
        metric = loader.get_success_metric_config(slice_name)
        lines.append(f"Slice: {slice_name}")
        lines.append(f"  Kind: {metric['kind']}")
        if 'parameters' in metric:
            for k, v in sorted(metric['parameters'].items()):
                lines.append(f"  {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def format_hashes(loader: CurriculumLoaderV2) -> str:
    """Format config hashes."""
    lines = [
        "=" * 70,
        "CONFIG HASHES (SHA256)",
        "=" * 70,
        "",
    ]

    for slice_name in loader.list_slices():
        h = loader.hash_slice_config(slice_name)
        lines.append(f"{slice_name}:")
        lines.append(f"  {h}")
        lines.append("")

    return "\n".join(lines)


def format_validation_report(loader: CurriculumLoaderV2) -> str:
    """Format validation report."""
    result = loader.validate_all()
    lines = [
        "=" * 70,
        "CURRICULUM VALIDATION REPORT",
        "=" * 70,
        "",
        f"Version: {result['version']}",
        f"Total Slices: {result['slice_count']}",
        f"Overall Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}",
        "",
    ]

    if result['monotonicity_warnings']:
        lines.append("Monotonicity Warnings:")
        for w in result['monotonicity_warnings']:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    lines.append("-" * 70)
    for slice_name, slice_result in result['slices'].items():
        status = "✓" if (slice_result['success_metric_valid'] and
                         slice_result['formula_pool_valid']) else "✗"
        lines.append(f"\n{status} {slice_name}")
        lines.append(f"  Success Metric: {'✓' if slice_result['success_metric_valid'] else '✗'}")
        lines.append(f"  Formula Pool: {'✓' if slice_result['formula_pool_valid'] else '✗'}")
        if slice_result['issues']:
            for issue in slice_result['issues']:
                lines.append(f"  - {issue}")

    return "\n".join(lines)


def format_nondegenerate_check(loader: CurriculumLoaderV2) -> str:
    """Format non-degeneracy constraint check."""
    lines = [
        "=" * 70,
        "NON-DEGENERACY CONSTRAINT CHECK",
        "=" * 70,
        "",
        "A slice is non-degenerate if:",
        "  - Baseline success rate is neither 0% nor 100%",
        "  - Budget constraints are meaningful",
        "  - Formula pool enables asymmetric exploration",
        "",
        "-" * 70,
    ]

    for slice_name in loader.list_slices():
        params = loader.get_parameters(slice_name)
        budget = loader.get_budget(slice_name)
        pool = loader.get_formula_pool(slice_name)

        # Check constraints
        checks = []

        # Budget should be constraining
        if budget['max_candidates_per_cycle'] >= params['total_max']:
            checks.append("⚠ Budget may not constrain exploration")
        else:
            checks.append("✓ Budget constrains exploration")

        # Pool should have variety
        if len(pool) < 5:
            checks.append("⚠ Formula pool may be too small")
        else:
            checks.append(f"✓ Formula pool has {len(pool)} entries")

        # Depth range should allow complexity
        if params['depth_max'] - params['depth_min'] < 2:
            checks.append("⚠ Depth range may be too narrow")
        else:
            checks.append("✓ Depth range allows complexity")

        lines.append(f"\n{slice_name}:")
        for c in checks:
            lines.append(f"  {c}")

    return "\n".join(lines)


def format_health_score(health: SliceHealthScore) -> str:
    """Format health score for a single slice."""
    band_display = health.band.value.upper()
    lines = [
        "=" * 70,
        f"HEALTH SCORE: {health.slice_name}",
        "=" * 70,
        "",
        f"Total Score: {health.total_score:.2%}",
        f"Band: {band_display}",
        "",
        "Component Breakdown:",
        f"  Formula Pool Integrity (30%):      {health.formula_pool_integrity:.2%}",
        f"  Success Metric Completeness (30%): {health.success_metric_completeness:.2%}",
        f"  Monotonicity Position (20%):       {health.monotonicity_position:.2%}",
        f"  Parameter Plausibility (20%):      {health.parameter_plausibility:.2%}",
    ]

    if health.issues:
        lines.append("")
        lines.append("Issues:")
        for issue in health.issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


def format_drift_report(report: DriftReport) -> str:
    """Format drift report."""
    lines = [
        "=" * 70,
        f"DRIFT REPORT: {report.slice_a} → {report.slice_b}",
        "=" * 70,
        "",
        f"Total Drift Magnitude: {report.total_drift_magnitude:.2%}",
        f"Max Severity: {report.max_severity.value.upper()}",
        f"Compatible: {'Yes' if report.is_compatible else 'No'}",
        "",
    ]

    if report.changed_fields:
        lines.append("-" * 70)
        lines.append("Changed Fields:")
        for field in report.changed_fields:
            lines.append(f"\n  {field.field_path}")
            lines.append(f"    Old: {field.old_value}")
            lines.append(f"    New: {field.new_value}")
            lines.append(f"    Magnitude: {field.drift_magnitude:.2%}")
            lines.append(f"    Severity: {field.severity.value}")
    else:
        lines.append("No changes detected.")

    return "\n".join(lines)


def format_snapshot(snapshot: CurriculumSnapshot) -> str:
    """Format curriculum snapshot."""
    lines = [
        "=" * 70,
        "CURRICULUM SNAPSHOT",
        "=" * 70,
        "",
        f"Timestamp: {snapshot.timestamp}",
        f"Version: {snapshot.curriculum_version}",
        f"Slice Count: {snapshot.slice_count}",
        f"Overall Health: {snapshot.overall_health:.2%}",
        f"Snapshot Hash: {snapshot.snapshot_hash[:16]}...",
        "",
        "-" * 70,
        "Per-Slice Summary:",
    ]

    for slice_name in sorted(snapshot.slice_hashes.keys()):
        lines.append(f"\n  {slice_name}")
        lines.append(f"    Hash: {snapshot.slice_hashes[slice_name][:16]}...")
        lines.append(f"    Metric: {snapshot.metric_kinds[slice_name]}")
        lines.append(f"    Pool Size: {snapshot.formula_pool_counts[slice_name]}")
        lines.append(f"    Health: {snapshot.health_scores[slice_name]:.2%}")

    if snapshot.monotonicity_warnings:
        lines.append("")
        lines.append("Monotonicity Warnings:")
        for w in snapshot.monotonicity_warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def format_preflight_report(report: PreflightReport) -> str:
    """Format pre-flight report as human-readable text."""
    verdict_symbol = {
        PreflightVerdict.OK: "✓",
        PreflightVerdict.WARN: "⚠",
        PreflightVerdict.FAIL: "✗",
    }
    global_symbol = {
        PreflightGlobalVerdict.OK: "✓",
        PreflightGlobalVerdict.WARN: "⚠",
        PreflightGlobalVerdict.FAIL: "✗",
    }

    lines = [
        "=" * 70,
        "PRE-FLIGHT CURRICULUM CHECK",
        "=" * 70,
        "",
        f"Timestamp: {report.timestamp}",
        f"Version: {report.curriculum_version}",
        "",
        f"Global Verdict: {global_symbol[report.global_verdict]} {report.global_verdict.value}",
        f"Overall Health: {report.overall_health:.2%}",
        "",
        f"Slice Summary: {report.ok_count} OK, {report.warn_count} WARN, {report.fail_count} FAIL",
        "",
        "-" * 70,
        f"{'Slice':<30} {'Verdict':<8} {'Health':<10} {'Band':<12} {'Issues':<6}",
        "-" * 70,
    ]

    for slice_name, result in report.slices.items():
        symbol = verdict_symbol[result.verdict]
        lines.append(
            f"{slice_name:<30} {symbol} {result.verdict.value:<5} "
            f"{result.health_score:.2%}    {result.health_band.value:<12} {result.issues_count}"
        )

    # Show monotonicity warnings if any
    if report.monotonicity_warnings:
        lines.append("")
        lines.append("Monotonicity Warnings:")
        for w in report.monotonicity_warnings:
            lines.append(f"  ⚠ {w}")

    # Show issues for non-OK slices
    non_ok_slices = [
        (name, r) for name, r in report.slices.items()
        if r.verdict != PreflightVerdict.OK
    ]
    if non_ok_slices:
        lines.append("")
        lines.append("-" * 70)
        lines.append("Issues Detail:")
        for name, result in non_ok_slices:
            lines.append(f"\n  {name} ({result.verdict.value}):")
            for issue in result.issues[:5]:  # Limit to first 5 issues
                lines.append(f"    - {issue}")
            if len(result.issues) > 5:
                lines.append(f"    ... and {len(result.issues) - 5} more")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# CLI MAIN
# =============================================================================

# Exit codes for CLI
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def run_health_dashboard(args: argparse.Namespace) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Main entry point for curriculum health dashboard.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: EXIT_OK (0), EXIT_FAIL (1), or EXIT_ERROR (2).
    """
    try:
        loader = CurriculumLoaderV2(filepath=args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return EXIT_ERROR
    except ValueError as e:
        print(f"Error: Invalid config file: {e}", file=sys.stderr)
        return EXIT_ERROR

    output_json = args.json

    # --list
    if args.list:
        if output_json:
            data = {
                'version': loader.get_version(),
                'slices': [
                    {
                        'name': s,
                        'atoms': loader.get_parameters(s)['atoms'],
                        'depth_range': f"{loader.get_parameters(s)['depth_min']}-{loader.get_parameters(s)['depth_max']}",
                        'metric_kind': loader.get_success_metric_config(s)['kind'],
                    }
                    for s in loader.list_slices()
                ],
            }
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(format_slice_list(loader))
        return EXIT_OK

    # --describe
    if args.describe:
        try:
            if output_json:
                config = loader.get_slice_config(args.describe)
                data = {
                    'slice_name': args.describe,
                    'config': config,
                    'config_hash': loader.hash_slice_config(args.describe),
                }
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                print(loader.describe_slice(args.describe))
        except KeyError:
            print(f"Error: Slice not found: {args.describe}", file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK

    # --metrics
    if args.metrics:
        if output_json:
            data = {
                'slices': {
                    s: loader.get_success_metric_config(s)
                    for s in loader.list_slices()
                }
            }
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(format_metrics_summary(loader))
        return EXIT_OK

    # --hashes
    if args.hashes:
        if output_json:
            data = {
                'hashes': {
                    s: loader.hash_slice_config(s)
                    for s in loader.list_slices()
                }
            }
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(format_hashes(loader))
        return EXIT_OK

    # --validate
    if args.validate:
        result = loader.validate_all()
        if output_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(format_validation_report(loader))
        return EXIT_OK if result['valid'] else EXIT_FAIL

    # --nondegenerate
    if args.nondegenerate:
        if output_json:
            data = {
                'slices': {}
            }
            for s in loader.list_slices():
                params = loader.get_parameters(s)
                budget = loader.get_budget(s)
                pool = loader.get_formula_pool(s)
                data['slices'][s] = {
                    'budget_constraining': budget['max_candidates_per_cycle'] < params['total_max'],
                    'pool_size': len(pool),
                    'depth_range': params['depth_max'] - params['depth_min'],
                }
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(format_nondegenerate_check(loader))
        return EXIT_OK

    # --score
    if args.score:
        try:
            health = compute_slice_health(loader, args.score)
            if output_json:
                print(json.dumps(health.to_dict(), indent=2, sort_keys=True))
            else:
                print(format_health_score(health))
        except KeyError:
            print(f"Error: Slice not found: {args.score}", file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK

    # --compare
    if args.compare:
        if len(args.compare) != 2:
            print("Error: --compare requires exactly two slice names", file=sys.stderr)
            return EXIT_ERROR
        try:
            report = detect_param_drift(loader, args.compare[0], args.compare[1])
            if output_json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                print(format_drift_report(report))
        except KeyError as e:
            print(f"Error: Slice not found: {e}", file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK

    # --snapshot
    if args.snapshot:
        snapshot = create_snapshot(loader)
        if output_json:
            print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_snapshot(snapshot))
        return EXIT_OK

    # --preflight
    if args.preflight:
        report = run_preflight(loader)
        if output_json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_preflight_report(report))

        # Return appropriate exit code
        if report.global_verdict == PreflightGlobalVerdict.OK:
            return EXIT_OK
        elif report.global_verdict == PreflightGlobalVerdict.FAIL:
            return EXIT_FAIL
        else:
            # WARN still returns OK exit code (0) but signals issues in output
            return EXIT_OK

    # --manifest
    if args.manifest:
        manifest = create_curriculum_manifest(loader)
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return EXIT_OK

    # --export-manifest
    if args.export_manifest:
        try:
            export_curriculum_manifest(loader, args.export_manifest)
            print(f"Manifest exported to: {args.export_manifest}")
            return EXIT_OK
        except IOError as e:
            print(f"Error writing manifest: {e}", file=sys.stderr)
            return EXIT_ERROR

    # --compare-manifests (longitudinal drift)
    if args.compare_manifests:
        old_path, new_path = args.compare_manifests
        try:
            old_manifest = load_curriculum_manifest(old_path)
            new_manifest = load_curriculum_manifest(new_path)
        except FileNotFoundError as e:
            print(f"Error: Manifest file not found: {e}", file=sys.stderr)
            return EXIT_ERROR
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in manifest: {e}", file=sys.stderr)
            return EXIT_ERROR
        except ValueError as e:
            print(f"Error: Invalid manifest format: {e}", file=sys.stderr)
            return EXIT_ERROR

        drift_report = compare_manifests(old_manifest, new_manifest)
        if output_json:
            print(json.dumps(drift_report.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_manifest_drift_report(drift_report))

        # Exit code: 0 = no drift, 1 = drift detected
        return EXIT_FAIL if drift_report.has_material_drift else EXIT_OK

    # --classify-drift
    if args.classify_drift:
        old_path, new_path = args.classify_drift
        try:
            old_manifest = load_curriculum_manifest(old_path)
            new_manifest = load_curriculum_manifest(new_path)
        except FileNotFoundError as e:
            print(f"Error: Manifest file not found: {e}", file=sys.stderr)
            return EXIT_ERROR
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in manifest: {e}", file=sys.stderr)
            return EXIT_ERROR
        except ValueError as e:
            print(f"Error: Invalid manifest format: {e}", file=sys.stderr)
            return EXIT_ERROR

        drift_report = compare_manifests(old_manifest, new_manifest)
        classification = classify_curriculum_drift(drift_report)

        if output_json:
            print(json.dumps(classification, indent=2, sort_keys=True))
        else:
            print(format_drift_classification(classification))

        # Exit code: 0 = not blocking, 1 = blocking
        return EXIT_FAIL if classification['blocking'] else EXIT_OK

    # --timeline
    if args.timeline:
        try:
            timeline = build_curriculum_manifest_timeline(args.timeline)
        except FileNotFoundError as e:
            print(f"Error: Manifest file not found: {e}", file=sys.stderr)
            return EXIT_ERROR
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in manifest: {e}", file=sys.stderr)
            return EXIT_ERROR
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_ERROR

        if output_json:
            # Include global health summary in JSON output
            global_summary = summarize_curriculum_for_global_health(timeline)
            output = {
                'timeline': timeline,
                'global_health_summary': global_summary,
            }
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print(format_timeline_summary(timeline))
            # Also print global health summary
            global_summary = summarize_curriculum_for_global_health(timeline)
            print("\nGlobal Health Summary:")
            print(f"  Curriculum OK: {global_summary['curriculum_ok']}")
            print(f"  Latest Verdict: {global_summary['latest_verdict']}")
            print(f"  Health Trend: {global_summary['health_trend']}")
            if global_summary['slices_with_major_band_changes']:
                print(f"  Slices with Major Band Changes: {', '.join(global_summary['slices_with_major_band_changes'])}")

        return EXIT_OK

    # --hints
    if args.hints:
        report = run_preflight(loader)
        hints = generate_curriculum_hints(report)
        if output_json:
            print(json.dumps({'hints': hints}, indent=2, sort_keys=True))
        else:
            print(format_hints(hints))
        return EXIT_OK

    # Default: show help
    print("Use --help for usage information.", file=sys.stderr)
    return EXIT_FAIL


def main() -> int:
    """
    PHASE II — NOT USED IN PHASE I

    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Curriculum Health Dashboard CLI for Phase II Uplift Slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all slices
  python -m experiments.curriculum_health --list

  # Describe a specific slice
  python -m experiments.curriculum_health --describe slice_uplift_goal

  # View success metrics
  python -m experiments.curriculum_health --metrics

  # View config hashes
  python -m experiments.curriculum_health --hashes

  # Validate curriculum
  python -m experiments.curriculum_health --validate

  # Check non-degeneracy
  python -m experiments.curriculum_health --nondegenerate

  # Compute health score
  python -m experiments.curriculum_health --score slice_uplift_goal

  # Compare two slices (drift detection)
  python -m experiments.curriculum_health --compare slice_uplift_goal slice_uplift_sparse

  # Create snapshot
  python -m experiments.curriculum_health --snapshot

  # Pre-flight check (validates + scores + non-degeneracy)
  python -m experiments.curriculum_health --preflight

  # Export curriculum manifest (canonical JSON)
  python -m experiments.curriculum_health --manifest
  python -m experiments.curriculum_health --export-manifest ./manifest.json

  # Compare two manifests (longitudinal drift detection)
  python -m experiments.curriculum_health --compare-manifests old.json new.json

  # Classify drift severity (NONE/MINOR/MAJOR)
  python -m experiments.curriculum_health --classify-drift old.json new.json

  # Build timeline from multiple manifests
  python -m experiments.curriculum_health --timeline v1.json v2.json v3.json

  # Generate advisory hints
  python -m experiments.curriculum_health --hints

  # Output as JSON
  python -m experiments.curriculum_health --list --json
""",
    )

    # Config file
    parser.add_argument(
        '--config', '-c',
        default='config/curriculum_uplift_phase2.yaml',
        help='Path to curriculum YAML file',
    )

    # Output format
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output in JSON format',
    )

    # Actions
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all slices with metadata',
    )
    parser.add_argument(
        '--describe', '-d',
        metavar='SLICE',
        help='Describe a specific slice',
    )
    parser.add_argument(
        '--metrics', '-m',
        action='store_true',
        help='Show success metric specifications',
    )
    parser.add_argument(
        '--hashes',
        action='store_true',
        help='Show config hashes for all slices',
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate curriculum integrity',
    )
    parser.add_argument(
        '--nondegenerate', '-n',
        action='store_true',
        help='Check non-degeneracy constraints',
    )
    parser.add_argument(
        '--score', '-s',
        metavar='SLICE',
        help='Compute health score for a slice',
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar='SLICE',
        help='Compare two slices for drift detection',
    )
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='Create curriculum snapshot',
    )
    parser.add_argument(
        '--preflight', '-p',
        action='store_true',
        help='Run pre-flight check (validate + score + non-degeneracy)',
    )
    parser.add_argument(
        '--manifest',
        action='store_true',
        help='Generate curriculum manifest (canonical JSON)',
    )
    parser.add_argument(
        '--export-manifest',
        metavar='PATH',
        help='Export curriculum manifest to JSON file',
    )
    parser.add_argument(
        '--compare-manifests',
        nargs=2,
        metavar='MANIFEST',
        help='Compare two manifest files for longitudinal drift',
    )
    parser.add_argument(
        '--classify-drift',
        nargs=2,
        metavar='MANIFEST',
        help='Classify drift severity between two manifests (NONE/MINOR/MAJOR)',
    )
    parser.add_argument(
        '--timeline',
        nargs='+',
        metavar='MANIFEST',
        help='Build timeline from multiple manifest files',
    )
    parser.add_argument(
        '--hints',
        action='store_true',
        help='Generate advisory health hints',
    )

    args = parser.parse_args()
    return run_health_dashboard(args)


if __name__ == '__main__':
    sys.exit(main())

