"""
Budget Admissibility Evaluator

==============================================================================
STATUS: PHASE II — NOT RUN IN PHASE I
==============================================================================

This module implements the Budget-Induced Abstention Law and admissibility
classification from docs/BUDGET_ADMISSIBILITY_SPEC.md.

Classification is ONLY about budget behavior:
- No Δp computation
- No uplift significance decisions
- Pure budget metrics → SAFE / SUSPICIOUS / INVALID

v2 additions (Budget Sentinel Grid):
- Snapshot contract for single-run admissibility
- Multi-run sentinel grid aggregation
- MAAS/governance-compatible summary helpers
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class AdmissibilityClassification(str, Enum):
    """Admissibility classification regions."""
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    INVALID = "INVALID"


# =============================================================================
# Thresholds from BUDGET_ADMISSIBILITY_SPEC.md Appendix A
# =============================================================================

TAU_MAX = 0.15          # Max exhaustion (safe) - upper bound for SAFE
TAU_SUSPICIOUS = 0.30   # Max exhaustion (suspicious) - upper bound for SUSPICIOUS
TAU_REJECT = 0.50       # Hard rejection threshold (R1)

DELTA_SYM = 0.05        # Max asymmetry (safe) - symmetric bound
DELTA_REJECT = 0.20     # Hard asymmetry rejection (R2)

KAPPA_MIN = 0.50        # Min cycle coverage (A4)
COMPLETE_SKIP_MAX = 0.10  # Max complete-skip cycles (R4)


@dataclass
class BudgetStats:
    """
    Budget statistics for a single condition (baseline or RFL).

    Attributes:
        total_candidates: Total candidates attempted
        total_skipped: Candidates skipped due to budget exhaustion
        exhausted_cycles: Number of cycles where budget was exhausted
        complete_skip_cycles: Cycles with 100% skip rate
        total_cycles: Total number of cycles
        min_cycle_coverage: Minimum (V + R + A) / N across cycles
        skip_trend_significant: Whether skip rate has significant monotonic trend
    """
    total_candidates: int
    total_skipped: int
    exhausted_cycles: int = 0
    complete_skip_cycles: int = 0
    total_cycles: int = 1
    min_cycle_coverage: float = 1.0
    skip_trend_significant: bool = False

    @property
    def exhaustion_rate(self) -> float:
        """B_rate = S_total / N_total"""
        if self.total_candidates == 0:
            return 0.0
        return self.total_skipped / self.total_candidates

    @property
    def complete_skip_fraction(self) -> float:
        """Fraction of cycles with 100% skip rate."""
        if self.total_cycles == 0:
            return 0.0
        return self.complete_skip_cycles / self.total_cycles


@dataclass
class BudgetAdmissibilityResult:
    """
    Result of budget admissibility classification.

    Attributes:
        classification: SAFE, SUSPICIOUS, or INVALID
        baseline_exhaustion_rate: B_rate for baseline condition
        rfl_exhaustion_rate: B_rate for RFL condition
        max_exhaustion_rate: max(baseline, rfl) exhaustion rates
        asymmetry: |B_rate(baseline) - B_rate(rfl)|
        min_cycle_coverage: Minimum coverage across both conditions
        complete_skip_fraction: Max complete-skip fraction across conditions
        skip_trend_detected: Whether systematic trend was detected

        a1_bounded_rate: Whether A1 condition is satisfied
        a2_symmetry: Whether A2 condition is satisfied
        a3_independence: Whether A3 condition is satisfied (proxy)
        a4_cycle_coverage: Whether A4 condition is satisfied

        r1_excessive_rate: Whether R1 rejection triggered
        r2_severe_asymmetry: Whether R2 rejection triggered
        r3_systematic_bias: Whether R3 rejection triggered
        r4_complete_failures: Whether R4 rejection triggered

        reasons: Human-readable list of classification reasons
    """
    classification: AdmissibilityClassification

    # Rates
    baseline_exhaustion_rate: float
    rfl_exhaustion_rate: float
    max_exhaustion_rate: float
    asymmetry: float
    min_cycle_coverage: float
    complete_skip_fraction: float
    skip_trend_detected: bool

    # Admissibility conditions (A1-A4)
    a1_bounded_rate: bool
    a2_symmetry: bool
    a3_independence: bool  # Proxy: no trend detected
    a4_cycle_coverage: bool

    # Rejection conditions (R1-R4)
    r1_excessive_rate: bool
    r2_severe_asymmetry: bool
    r3_systematic_bias: bool
    r4_complete_failures: bool

    # Human-readable reasons
    reasons: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if classification is not INVALID."""
        return self.classification != AdmissibilityClassification.INVALID

    @property
    def any_rejection(self) -> bool:
        """True if any R1-R4 rejection condition is triggered."""
        return (
            self.r1_excessive_rate or
            self.r2_severe_asymmetry or
            self.r3_systematic_bias or
            self.r4_complete_failures
        )

    @property
    def all_admissible(self) -> bool:
        """True if all A1-A4 admissibility conditions are satisfied."""
        return (
            self.a1_bounded_rate and
            self.a2_symmetry and
            self.a3_independence and
            self.a4_cycle_coverage
        )


def classify_budget_admissibility(
    baseline_stats: BudgetStats,
    rfl_stats: BudgetStats,
    effect_magnitude: Optional[float] = None,
) -> BudgetAdmissibilityResult:
    """
    Classify run admissibility based on budget metrics.

    Implements the Budget-Induced Abstention Law from
    docs/BUDGET_ADMISSIBILITY_SPEC.md.

    Args:
        baseline_stats: Budget statistics for baseline condition
        rfl_stats: Budget statistics for RFL condition
        effect_magnitude: Optional |Δp| for region refinement.
                         If None, assumes small effect (conservative).

    Returns:
        BudgetAdmissibilityResult with classification and detailed flags.

    Note:
        This function does NOT compute Δp or make uplift significance decisions.
        It only classifies budget behavior.
    """
    reasons: List[str] = []

    # Extract rates
    b_rate_base = baseline_stats.exhaustion_rate
    b_rate_rfl = rfl_stats.exhaustion_rate
    b_rate_max = max(b_rate_base, b_rate_rfl)
    asymmetry = abs(b_rate_base - b_rate_rfl)

    min_coverage = min(
        baseline_stats.min_cycle_coverage,
        rfl_stats.min_cycle_coverage
    )

    complete_skip_frac = max(
        baseline_stats.complete_skip_fraction,
        rfl_stats.complete_skip_fraction
    )

    skip_trend = (
        baseline_stats.skip_trend_significant or
        rfl_stats.skip_trend_significant
    )

    # Default effect magnitude to 0 (conservative - assumes small effect)
    if effect_magnitude is None:
        effect_magnitude = 0.0
    effect_mag = abs(effect_magnitude)

    # ==========================================================================
    # Evaluate Rejection Conditions (R1-R4)
    # ==========================================================================

    # R1: Excessive Rate (B_rate > 50%)
    r1 = b_rate_max > TAU_REJECT
    if r1:
        reasons.append(
            f"R1: Excessive exhaustion rate ({b_rate_max:.1%} > {TAU_REJECT:.0%})"
        )

    # R2: Severe Asymmetry (asymmetry > 20%)
    r2 = asymmetry > DELTA_REJECT
    if r2:
        reasons.append(
            f"R2: Severe asymmetry ({asymmetry:.1%} > {DELTA_REJECT:.0%})"
        )

    # R3: Systematic Bias Pattern (significant trend in skip rate)
    r3 = skip_trend
    if r3:
        reasons.append("R3: Systematic bias pattern detected (skip trend)")

    # R4: Complete Cycle Failures (>10% cycles with 100% skip)
    r4 = complete_skip_frac > COMPLETE_SKIP_MAX
    if r4:
        reasons.append(
            f"R4: Complete-skip cycles ({complete_skip_frac:.1%} > {COMPLETE_SKIP_MAX:.0%})"
        )

    # ==========================================================================
    # Evaluate Admissibility Conditions (A1-A4)
    # ==========================================================================

    # A1: Bounded Rate (B_rate <= τ_max = 15%)
    a1 = b_rate_max <= TAU_MAX
    if not a1 and not r1:
        reasons.append(
            f"A1 violation: Exhaustion rate ({b_rate_max:.1%} > {TAU_MAX:.0%})"
        )

    # A2: Symmetry (asymmetry <= δ_sym = 5%)
    a2 = asymmetry <= DELTA_SYM
    if not a2 and not r2:
        reasons.append(
            f"A2 violation: Asymmetry ({asymmetry:.1%} > {DELTA_SYM:.0%})"
        )

    # A3: Independence (proxy: no systematic trend)
    a3 = not skip_trend
    # Already captured in R3 if violated

    # A4: Cycle Coverage (min coverage >= κ_min = 50%)
    a4 = min_coverage >= KAPPA_MIN
    if not a4:
        reasons.append(
            f"A4 violation: Cycle coverage ({min_coverage:.1%} < {KAPPA_MIN:.0%})"
        )

    # ==========================================================================
    # Classification Logic (from spec §4.3 and Appendix B decision tree)
    # ==========================================================================

    classification: AdmissibilityClassification

    # Hard rejection conditions → INVALID
    if r1 or r2 or r3 or r4:
        classification = AdmissibilityClassification.INVALID

    # 30-50% exhaustion with small effect → INVALID
    elif b_rate_max > TAU_SUSPICIOUS and effect_mag < 0.15:
        classification = AdmissibilityClassification.INVALID
        reasons.append(
            f"Exhaustion {b_rate_max:.1%} > 30% with small effect ({effect_mag:.1%})"
        )

    # 30-50% exhaustion with large effect → SUSPICIOUS
    elif b_rate_max > TAU_SUSPICIOUS:
        classification = AdmissibilityClassification.SUSPICIOUS
        reasons.append(
            f"Exhaustion {b_rate_max:.1%} > 30% (large effect mitigates)"
        )

    # 15-30% exhaustion with small effect → SUSPICIOUS
    elif b_rate_max > TAU_MAX and effect_mag < 0.15:
        classification = AdmissibilityClassification.SUSPICIOUS
        reasons.append(
            f"Exhaustion {b_rate_max:.1%} > 15% with small effect ({effect_mag:.1%})"
        )

    # 15-30% exhaustion with large effect → SAFE
    elif b_rate_max > TAU_MAX and effect_mag >= 0.15:
        classification = AdmissibilityClassification.SAFE
        reasons.append(
            f"Exhaustion {b_rate_max:.1%} > 15% but large effect ({effect_mag:.1%})"
        )

    # Asymmetry 10-20% → SUSPICIOUS
    elif asymmetry > 0.10:
        classification = AdmissibilityClassification.SUSPICIOUS
        reasons.append(f"Elevated asymmetry ({asymmetry:.1%} > 10%)")

    # Low coverage → SUSPICIOUS
    elif not a4:
        classification = AdmissibilityClassification.SUSPICIOUS
        reasons.append(f"Low cycle coverage ({min_coverage:.1%})")

    # Mild asymmetry 5-10% with elevated rate → SUSPICIOUS
    elif asymmetry > DELTA_SYM and b_rate_max > 0.10:
        classification = AdmissibilityClassification.SUSPICIOUS
        reasons.append(
            f"Asymmetry ({asymmetry:.1%}) with elevated rate ({b_rate_max:.1%})"
        )

    # All conditions satisfied → SAFE
    else:
        classification = AdmissibilityClassification.SAFE
        if not reasons:
            reasons.append("All admissibility conditions satisfied")

    return BudgetAdmissibilityResult(
        classification=classification,
        baseline_exhaustion_rate=b_rate_base,
        rfl_exhaustion_rate=b_rate_rfl,
        max_exhaustion_rate=b_rate_max,
        asymmetry=asymmetry,
        min_cycle_coverage=min_coverage,
        complete_skip_fraction=complete_skip_frac,
        skip_trend_detected=skip_trend,
        a1_bounded_rate=a1,
        a2_symmetry=a2,
        a3_independence=a3,
        a4_cycle_coverage=a4,
        r1_excessive_rate=r1,
        r2_severe_asymmetry=r2,
        r3_systematic_bias=r3,
        r4_complete_failures=r4,
        reasons=reasons,
    )


def format_admissibility_report(result: BudgetAdmissibilityResult) -> str:
    """
    Format a human-readable admissibility report.

    Matches the report templates from BUDGET_ADMISSIBILITY_SPEC.md §7.
    """
    lines = [
        f"BUDGET ADMISSIBILITY: {result.classification.value}",
        "=" * 62,
        "",
        "Budget Exhaustion Analysis",
        "-" * 61,
    ]

    def check_mark(condition: bool) -> str:
        return "✓" if condition else "✗" if not condition else "⚠"

    lines.extend([
        f"  Baseline exhaustion rate:  {result.baseline_exhaustion_rate:5.1%}  "
        f"(threshold: {TAU_MAX:.0%})  {check_mark(result.baseline_exhaustion_rate <= TAU_MAX)}",

        f"  RFL exhaustion rate:       {result.rfl_exhaustion_rate:5.1%}  "
        f"(threshold: {TAU_MAX:.0%})  {check_mark(result.rfl_exhaustion_rate <= TAU_MAX)}",

        f"  Asymmetry:                 {result.asymmetry:5.1%}  "
        f"(threshold: {DELTA_SYM:.0%})   {check_mark(result.a2_symmetry)}",

        f"  Min cycle coverage:        {result.min_cycle_coverage:5.0%}   "
        f"(threshold: {KAPPA_MIN:.0%})  {check_mark(result.a4_cycle_coverage)}",

        f"  Complete-skip cycles:      {result.complete_skip_fraction:5.0%}   "
        f"(threshold: {COMPLETE_SKIP_MAX:.0%})  {check_mark(not result.r4_complete_failures)}",
    ])

    lines.extend([
        "",
        f"Classification: {result.classification.value}",
        "-" * 61,
    ])

    for reason in result.reasons:
        lines.append(f"  {reason}")

    lines.append("")

    if result.classification == AdmissibilityClassification.SAFE:
        lines.append("Conclusion: Results are scientifically valid for comparison.")
    elif result.classification == AdmissibilityClassification.SUSPICIOUS:
        lines.append("Conclusion: Results require additional scrutiny.")
        lines.append("            Sensitivity analysis recommended.")
    else:  # INVALID
        lines.append("Conclusion: Results are NOT scientifically valid.")
        lines.append("            Run must be repeated with adjusted budget parameters.")

    return "\n".join(lines)


# =============================================================================
# BUDGET SENTINEL GRID v2 — Snapshot Contract
# =============================================================================

BUDGET_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def build_budget_admissibility_snapshot(
    result: BudgetAdmissibilityResult,
    slice_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a stable snapshot object for a single run's budget admissibility.

    This snapshot is suitable for storage, aggregation, and governance consumption.
    All fields are primitive JSON-serializable types.

    CONTRACT:
        - schema_version: always BUDGET_SNAPSHOT_SCHEMA_VERSION
        - Keys are sorted alphabetically in output
        - Deterministic: same input → same snapshot
        - Pure function with no side effects

    Parameters
    ----------
    result : BudgetAdmissibilityResult
        Classification result from classify_budget_admissibility().

    slice_name : str, optional
        Name of the slice this run belongs to.

    Returns
    -------
    dict
        Snapshot with:
        - schema_version
        - slice_name (if provided)
        - classification
        - exhaustion_rate_baseline, exhaustion_rate_rfl
        - asymmetry_score
        - A_flags (A1-A4) and R_flags (R1-R4)
    """
    return {
        "A_flags": {
            "a1_bounded_rate": result.a1_bounded_rate,
            "a2_symmetry": result.a2_symmetry,
            "a3_independence": result.a3_independence,
            "a4_cycle_coverage": result.a4_cycle_coverage,
        },
        "R_flags": {
            "r1_excessive_rate": result.r1_excessive_rate,
            "r2_severe_asymmetry": result.r2_severe_asymmetry,
            "r3_systematic_bias": result.r3_systematic_bias,
            "r4_complete_failures": result.r4_complete_failures,
        },
        "asymmetry_score": result.asymmetry,
        "classification": result.classification.value,
        "exhaustion_rate_baseline": result.baseline_exhaustion_rate,
        "exhaustion_rate_rfl": result.rfl_exhaustion_rate,
        "max_exhaustion_rate": result.max_exhaustion_rate,
        "min_cycle_coverage": result.min_cycle_coverage,
        "schema_version": BUDGET_SNAPSHOT_SCHEMA_VERSION,
        "slice_name": slice_name,
    }


def save_budget_admissibility_snapshot(
    path: str,
    snapshot: Dict[str, Any],
) -> None:
    """
    Save a budget admissibility snapshot to a JSON file.

    Parameters
    ----------
    path : str
        File path to write the snapshot.

    snapshot : dict
        Snapshot from build_budget_admissibility_snapshot().
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)


def load_budget_admissibility_snapshot(
    path: str,
) -> Dict[str, Any]:
    """
    Load a budget admissibility snapshot from a JSON file.

    Parameters
    ----------
    path : str
        File path to read the snapshot from.

    Returns
    -------
    dict
        Loaded snapshot.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# BUDGET SENTINEL GRID v2 — Multi-Run Aggregation
# =============================================================================

BUDGET_SENTINEL_GRID_SCHEMA_VERSION = "1.0.0"


def build_budget_sentinel_grid(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate multiple budget admissibility snapshots into a sentinel grid.

    This grid provides a summary view of budget health across multiple runs,
    suitable for CI dashboards and MAAS consumption.

    CONTRACT:
        - schema_version: always BUDGET_SENTINEL_GRID_SCHEMA_VERSION
        - Deterministic ordering: per_slice sorted alphabetically by slice_name
        - Pure function with no side effects

    Parameters
    ----------
    snapshots : Sequence[dict]
        List of snapshots from build_budget_admissibility_snapshot().

    Returns
    -------
    dict
        Sentinel grid with:
        - schema_version
        - total_runs
        - safe_count, suspicious_count, invalid_count
        - per_slice: dict mapping slice_name → last classification
    """
    safe_count = 0
    suspicious_count = 0
    invalid_count = 0

    # Track last classification per slice (by order of appearance)
    per_slice: Dict[str, str] = {}

    for snap in snapshots:
        classification = snap.get("classification", "UNKNOWN")

        if classification == "SAFE":
            safe_count += 1
        elif classification == "SUSPICIOUS":
            suspicious_count += 1
        elif classification == "INVALID":
            invalid_count += 1

        slice_name = snap.get("slice_name")
        if slice_name is not None:
            per_slice[slice_name] = classification

    # Sort per_slice by key for deterministic output
    per_slice_sorted = {k: per_slice[k] for k in sorted(per_slice.keys())}

    return {
        "invalid_count": invalid_count,
        "per_slice": per_slice_sorted,
        "safe_count": safe_count,
        "schema_version": BUDGET_SENTINEL_GRID_SCHEMA_VERSION,
        "suspicious_count": suspicious_count,
        "total_runs": len(snapshots),
    }


# =============================================================================
# BUDGET SENTINEL GRID v2 — Governance/MAAS Summary
# =============================================================================

def summarize_budget_for_governance(
    result: BudgetAdmissibilityResult,
) -> Dict[str, Any]:
    """
    Produce a minimal, MAAS-compatible signal for budget admissibility.

    This summary is suitable for direct consumption by MAAS (CLAUDE O)
    and Governance Verifier (CLAUDE I).

    CONTRACT:
        - is_budget_admissible: True if classification is SAFE or SUSPICIOUS
        - classification: string value of classification
        - rejection_reasons: subset of reasons when any R-conditions triggered
        - Deterministic and side-effect free

    Parameters
    ----------
    result : BudgetAdmissibilityResult
        Classification result from classify_budget_admissibility().

    Returns
    -------
    dict
        Governance summary with:
        - is_budget_admissible: bool (True unless INVALID)
        - classification: str
        - rejection_reasons: list of str (only R-condition reasons)
    """
    # Extract only rejection reasons (those starting with "R1:", "R2:", etc.)
    rejection_reasons: List[str] = []
    if result.any_rejection:
        for reason in result.reasons:
            if reason.startswith(("R1:", "R2:", "R3:", "R4:")):
                rejection_reasons.append(reason)

    return {
        "classification": result.classification.value,
        "is_budget_admissible": result.classification != AdmissibilityClassification.INVALID,
        "rejection_reasons": rejection_reasons,
    }


# =============================================================================
# BUDGET SENTINEL GRID v3 — Phase III Cross-Run Intelligence
# =============================================================================

BUDGET_DRIFT_LEDGER_SCHEMA_VERSION = "1.0.0"

# Drift detection thresholds
DRIFT_RATE_THRESHOLD = 0.05      # 5% change in exhaustion rate triggers drift
DRIFT_ASYMMETRY_THRESHOLD = 0.03 # 3% change in asymmetry triggers drift
REPEATED_INVALID_THRESHOLD = 2   # 2+ consecutive INVALIDs is concerning


class UpliftBudgetGate(str, Enum):
    """Uplift budget gate decisions."""
    OK = "OK"           # Budget is healthy, proceed with uplift
    WARN = "WARN"       # Budget concerns, proceed with caution
    BLOCK = "BLOCK"     # Budget issues, do not proceed with uplift


@dataclass
class DriftDetection:
    """Result of drift detection for a single metric."""
    metric_name: str
    has_drift: bool
    direction: str  # "increasing", "decreasing", "stable"
    delta: float    # Change from first to last
    values: List[float]


def _detect_drift(
    values: List[float],
    threshold: float,
    metric_name: str,
) -> DriftDetection:
    """
    Detect drift in a sequence of values.

    Drift is detected if the change from first to last exceeds threshold.
    """
    if len(values) < 2:
        return DriftDetection(
            metric_name=metric_name,
            has_drift=False,
            direction="stable",
            delta=0.0,
            values=values,
        )

    first = values[0]
    last = values[-1]
    delta = last - first

    has_drift = abs(delta) > threshold

    if delta > threshold:
        direction = "increasing"
    elif delta < -threshold:
        direction = "decreasing"
    else:
        direction = "stable"

    return DriftDetection(
        metric_name=metric_name,
        has_drift=has_drift,
        direction=direction,
        delta=delta,
        values=values,
    )


def _count_consecutive_invalids(classifications: List[str]) -> int:
    """Count maximum consecutive INVALID classifications."""
    if not classifications:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for c in classifications:
        if c == "INVALID":
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def build_budget_drift_ledger(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze budget snapshots for drift patterns across multiple runs.

    Detects:
    - Drift in exhaustion rates (baseline and RFL)
    - Drift in asymmetry scores
    - Repeated INVALID classifications
    - Per-slice health trends

    CONTRACT:
        - schema_version: always BUDGET_DRIFT_LEDGER_SCHEMA_VERSION
        - Deterministic: same snapshots → same ledger
        - Pure function with no side effects

    Parameters
    ----------
    snapshots : Sequence[dict]
        List of snapshots from build_budget_admissibility_snapshot(),
        ordered chronologically (oldest first).

    Returns
    -------
    dict
        Drift ledger with:
        - schema_version
        - run_count: total runs analyzed
        - exhaustion_rate_drift: DriftDetection for max exhaustion rate
        - asymmetry_drift: DriftDetection for asymmetry
        - consecutive_invalids: max consecutive INVALID count
        - has_concerning_drift: bool summary flag
        - per_slice_health: dict of slice → health summary
    """
    if not snapshots:
        return {
            "schema_version": BUDGET_DRIFT_LEDGER_SCHEMA_VERSION,
            "run_count": 0,
            "exhaustion_rate_drift": {
                "metric_name": "max_exhaustion_rate",
                "has_drift": False,
                "direction": "stable",
                "delta": 0.0,
                "values": [],
            },
            "asymmetry_drift": {
                "metric_name": "asymmetry_score",
                "has_drift": False,
                "direction": "stable",
                "delta": 0.0,
                "values": [],
            },
            "consecutive_invalids": 0,
            "has_concerning_drift": False,
            "per_slice_health": {},
        }

    # Extract time series
    exhaustion_rates: List[float] = []
    asymmetry_scores: List[float] = []
    classifications: List[str] = []

    # Per-slice tracking
    slice_data: Dict[str, Dict[str, List[Any]]] = {}

    for snap in snapshots:
        # Global metrics
        exhaustion_rates.append(snap.get("max_exhaustion_rate", 0.0))
        asymmetry_scores.append(snap.get("asymmetry_score", 0.0))
        classifications.append(snap.get("classification", "UNKNOWN"))

        # Per-slice tracking
        slice_name = snap.get("slice_name")
        if slice_name is not None:
            if slice_name not in slice_data:
                slice_data[slice_name] = {
                    "exhaustion_rates": [],
                    "classifications": [],
                }
            slice_data[slice_name]["exhaustion_rates"].append(
                snap.get("max_exhaustion_rate", 0.0)
            )
            slice_data[slice_name]["classifications"].append(
                snap.get("classification", "UNKNOWN")
            )

    # Detect drift
    exhaustion_drift = _detect_drift(
        exhaustion_rates, DRIFT_RATE_THRESHOLD, "max_exhaustion_rate"
    )
    asymmetry_drift = _detect_drift(
        asymmetry_scores, DRIFT_ASYMMETRY_THRESHOLD, "asymmetry_score"
    )

    # Count consecutive invalids
    consecutive_invalids = _count_consecutive_invalids(classifications)

    # Per-slice health
    per_slice_health: Dict[str, Dict[str, Any]] = {}
    for slice_name, data in sorted(slice_data.items()):
        slice_drift = _detect_drift(
            data["exhaustion_rates"], DRIFT_RATE_THRESHOLD, "exhaustion_rate"
        )
        slice_invalids = _count_consecutive_invalids(data["classifications"])
        last_classification = data["classifications"][-1] if data["classifications"] else "UNKNOWN"

        per_slice_health[slice_name] = {
            "run_count": len(data["classifications"]),
            "last_classification": last_classification,
            "has_drift": slice_drift.has_drift,
            "drift_direction": slice_drift.direction,
            "consecutive_invalids": slice_invalids,
        }

    # Summary flag
    has_concerning_drift = (
        exhaustion_drift.has_drift and exhaustion_drift.direction == "increasing"
    ) or (
        asymmetry_drift.has_drift and asymmetry_drift.direction == "increasing"
    ) or (
        consecutive_invalids >= REPEATED_INVALID_THRESHOLD
    )

    return {
        "schema_version": BUDGET_DRIFT_LEDGER_SCHEMA_VERSION,
        "run_count": len(snapshots),
        "exhaustion_rate_drift": {
            "metric_name": exhaustion_drift.metric_name,
            "has_drift": exhaustion_drift.has_drift,
            "direction": exhaustion_drift.direction,
            "delta": exhaustion_drift.delta,
            "values": exhaustion_drift.values,
        },
        "asymmetry_drift": {
            "metric_name": asymmetry_drift.metric_name,
            "has_drift": asymmetry_drift.has_drift,
            "direction": asymmetry_drift.direction,
            "delta": asymmetry_drift.delta,
            "values": asymmetry_drift.values,
        },
        "consecutive_invalids": consecutive_invalids,
        "has_concerning_drift": has_concerning_drift,
        "per_slice_health": per_slice_health,
    }


# =============================================================================
# BUDGET SENTINEL GRID v3 — Uplift Budget Gate
# =============================================================================

def evaluate_budget_for_uplift(
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether a budget snapshot permits proceeding with uplift analysis.

    Gate decisions:
    - OK: Budget is healthy, proceed with uplift
    - WARN: Budget has concerns, proceed with caution (sensitivity analysis recommended)
    - BLOCK: Budget issues invalidate uplift evidence

    CONTRACT:
        - gate: UpliftBudgetGate enum value as string
        - Deterministic decision based on snapshot contents
        - Pure function with no side effects

    Parameters
    ----------
    snapshot : dict
        Budget admissibility snapshot from build_budget_admissibility_snapshot().

    Returns
    -------
    dict
        Gate decision with:
        - gate: "OK" | "WARN" | "BLOCK"
        - reasons: list of reasons for the decision
        - can_proceed: bool (True if OK or WARN)
        - requires_sensitivity_analysis: bool
    """
    classification = snapshot.get("classification", "UNKNOWN")
    exhaustion_baseline = snapshot.get("exhaustion_rate_baseline", 0.0)
    exhaustion_rfl = snapshot.get("exhaustion_rate_rfl", 0.0)
    max_exhaustion = snapshot.get("max_exhaustion_rate", 0.0)
    asymmetry = snapshot.get("asymmetry_score", 0.0)

    r_flags = snapshot.get("R_flags", {})
    a_flags = snapshot.get("A_flags", {})

    reasons: List[str] = []
    gate: UpliftBudgetGate

    # BLOCK conditions
    if classification == "INVALID":
        gate = UpliftBudgetGate.BLOCK
        reasons.append("Classification is INVALID")

        # Add specific R-flag reasons
        if r_flags.get("r1_excessive_rate"):
            reasons.append("R1: Excessive exhaustion rate")
        if r_flags.get("r2_severe_asymmetry"):
            reasons.append("R2: Severe asymmetry between conditions")
        if r_flags.get("r3_systematic_bias"):
            reasons.append("R3: Systematic bias pattern detected")
        if r_flags.get("r4_complete_failures"):
            reasons.append("R4: Too many complete-skip cycles")

    # WARN conditions
    elif classification == "SUSPICIOUS":
        gate = UpliftBudgetGate.WARN
        reasons.append("Classification is SUSPICIOUS")

        if max_exhaustion > TAU_MAX:
            reasons.append(f"Exhaustion rate ({max_exhaustion:.1%}) exceeds safe threshold ({TAU_MAX:.0%})")
        if asymmetry > DELTA_SYM:
            reasons.append(f"Asymmetry ({asymmetry:.1%}) exceeds symmetric bound ({DELTA_SYM:.0%})")
        if not a_flags.get("a4_cycle_coverage", True):
            reasons.append("Low cycle coverage detected")

    # Additional WARN checks for SAFE with borderline metrics
    elif classification == "SAFE":
        # Check for borderline conditions that warrant caution
        borderline_concerns = []

        if max_exhaustion > 0.10:  # >10% but <=15%
            borderline_concerns.append(f"Exhaustion rate ({max_exhaustion:.1%}) approaching threshold")

        if asymmetry > 0.03:  # >3% but <=5%
            borderline_concerns.append(f"Asymmetry ({asymmetry:.1%}) is elevated")

        if borderline_concerns:
            gate = UpliftBudgetGate.WARN
            reasons.extend(borderline_concerns)
        else:
            gate = UpliftBudgetGate.OK
            reasons.append("All budget metrics within healthy bounds")

    else:
        # Unknown classification
        gate = UpliftBudgetGate.BLOCK
        reasons.append(f"Unknown classification: {classification}")

    return {
        "gate": gate.value,
        "reasons": reasons,
        "can_proceed": gate in (UpliftBudgetGate.OK, UpliftBudgetGate.WARN),
        "requires_sensitivity_analysis": gate == UpliftBudgetGate.WARN,
    }


# =============================================================================
# BUDGET SENTINEL GRID v3 — Global Health Signal
# =============================================================================

BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION = "1.0.0"


class BudgetStability(str, Enum):
    """Global budget stability classification."""
    STABLE = "STABLE"           # Budget metrics are consistent and healthy
    DEGRADING = "DEGRADING"     # Budget health is declining
    UNSTABLE = "UNSTABLE"       # Budget metrics are volatile or unhealthy


def summarize_budget_for_global_health(
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a global health signal from a budget drift ledger.

    Suitable for system-wide dashboards and automated monitoring.

    CONTRACT:
        - schema_version: always BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION
        - Deterministic and side-effect free
        - Non-interpretive (no "good"/"bad" judgments)

    Parameters
    ----------
    ledger : dict
        Budget drift ledger from build_budget_drift_ledger().

    Returns
    -------
    dict
        Global health signal with:
        - schema_version
        - budget_stability: "STABLE" | "DEGRADING" | "UNSTABLE"
        - unsafe_slices: list of slice names with concerning health
        - uplift_budget_ready: bool (True if system is ready for uplift)
        - summary_metrics: aggregated metrics
    """
    run_count = ledger.get("run_count", 0)

    if run_count == 0:
        return {
            "schema_version": BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION,
            "budget_stability": BudgetStability.STABLE.value,
            "unsafe_slices": [],
            "uplift_budget_ready": True,
            "summary_metrics": {
                "run_count": 0,
                "has_drift": False,
                "consecutive_invalids": 0,
                "unsafe_slice_count": 0,
            },
        }

    # Extract ledger data
    exhaustion_drift = ledger.get("exhaustion_rate_drift", {})
    asymmetry_drift = ledger.get("asymmetry_drift", {})
    consecutive_invalids = ledger.get("consecutive_invalids", 0)
    has_concerning_drift = ledger.get("has_concerning_drift", False)
    per_slice_health = ledger.get("per_slice_health", {})

    # Identify unsafe slices
    unsafe_slices: List[str] = []
    for slice_name, health in per_slice_health.items():
        is_unsafe = (
            health.get("last_classification") == "INVALID" or
            health.get("consecutive_invalids", 0) >= REPEATED_INVALID_THRESHOLD or
            (health.get("has_drift") and health.get("drift_direction") == "increasing")
        )
        if is_unsafe:
            unsafe_slices.append(slice_name)

    # Sort for determinism
    unsafe_slices = sorted(unsafe_slices)

    # Determine stability
    stability: BudgetStability

    if consecutive_invalids >= 3 or len(unsafe_slices) > len(per_slice_health) / 2:
        # Majority slices unsafe or many consecutive invalids
        stability = BudgetStability.UNSTABLE
    elif has_concerning_drift or len(unsafe_slices) > 0:
        # Some drift or some unsafe slices
        stability = BudgetStability.DEGRADING
    else:
        stability = BudgetStability.STABLE

    # Uplift readiness
    uplift_budget_ready = (
        stability != BudgetStability.UNSTABLE and
        consecutive_invalids < REPEATED_INVALID_THRESHOLD and
        len(unsafe_slices) == 0
    )

    return {
        "schema_version": BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION,
        "budget_stability": stability.value,
        "unsafe_slices": unsafe_slices,
        "uplift_budget_ready": uplift_budget_ready,
        "summary_metrics": {
            "run_count": run_count,
            "has_drift": has_concerning_drift,
            "consecutive_invalids": consecutive_invalids,
            "unsafe_slice_count": len(unsafe_slices),
        },
    }


# =============================================================================
# BUDGET SENTINEL GRID v4 — Cross-Slice Budget Risk Map
# =============================================================================

BUDGET_RISK_MAP_SCHEMA_VERSION = "1.0.0"


class RiskBand(str, Enum):
    """Overall risk band classification."""
    LOW = "LOW"       # No concerning patterns
    MEDIUM = "MEDIUM" # Some concerning patterns but manageable
    HIGH = "HIGH"     # Significant risk requiring attention


def build_budget_risk_map(
    budget_drift_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a cross-slice budget risk map from a drift ledger.

    Analyzes per-slice health to identify:
    - Slices with concerning drift (exhaustion rate increasing)
    - Slices with repeated INVALID classifications
    - Overall risk band based on aggregate patterns

    CONTRACT:
        - schema_version: always BUDGET_RISK_MAP_SCHEMA_VERSION
        - Deterministic: same ledger → same risk map
        - Pure function with no side effects
        - Lists are sorted alphabetically for determinism

    Parameters
    ----------
    budget_drift_ledger : dict
        Budget drift ledger from build_budget_drift_ledger().

    Returns
    -------
    dict
        Risk map with:
        - schema_version
        - slices_with_concerning_drift: list of slice names
        - slices_with_repeated_invalid: list of slice names
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - summary: aggregated risk metrics
    """
    per_slice_health = budget_drift_ledger.get("per_slice_health", {})
    has_concerning_drift = budget_drift_ledger.get("has_concerning_drift", False)
    consecutive_invalids = budget_drift_ledger.get("consecutive_invalids", 0)

    slices_with_concerning_drift: List[str] = []
    slices_with_repeated_invalid: List[str] = []

    for slice_name, health in per_slice_health.items():
        # Check for concerning drift (increasing exhaustion)
        if health.get("has_drift") and health.get("drift_direction") == "increasing":
            slices_with_concerning_drift.append(slice_name)

        # Check for repeated INVALIDs
        if health.get("consecutive_invalids", 0) >= REPEATED_INVALID_THRESHOLD:
            slices_with_repeated_invalid.append(slice_name)

    # Sort for determinism
    slices_with_concerning_drift = sorted(slices_with_concerning_drift)
    slices_with_repeated_invalid = sorted(slices_with_repeated_invalid)

    # Determine risk band
    total_slices = len(per_slice_health)
    problem_slices = len(set(slices_with_concerning_drift) | set(slices_with_repeated_invalid))

    risk_band: RiskBand
    if total_slices == 0:
        # No slices to assess
        risk_band = RiskBand.LOW
    elif consecutive_invalids >= 3 or problem_slices > total_slices / 2:
        # Many consecutive invalids or majority slices have issues
        risk_band = RiskBand.HIGH
    elif has_concerning_drift or problem_slices > 0:
        # Some drift or some problem slices
        risk_band = RiskBand.MEDIUM
    else:
        risk_band = RiskBand.LOW

    return {
        "schema_version": BUDGET_RISK_MAP_SCHEMA_VERSION,
        "slices_with_concerning_drift": slices_with_concerning_drift,
        "slices_with_repeated_invalid": slices_with_repeated_invalid,
        "risk_band": risk_band.value,
        "summary": {
            "total_slices": total_slices,
            "drift_slice_count": len(slices_with_concerning_drift),
            "invalid_slice_count": len(slices_with_repeated_invalid),
            "problem_slice_count": problem_slices,
        },
    }


# =============================================================================
# BUDGET SENTINEL GRID v4 — Joint Budget × Metric Readiness Adapter
# =============================================================================

BUDGET_METRIC_UPLIFT_SCHEMA_VERSION = "1.0.0"


class JointUpliftStatus(str, Enum):
    """Joint uplift status for budget + metrics."""
    OK = "OK"       # Both budget and metrics are ready
    WARN = "WARN"   # Minor concerns, proceed with caution
    BLOCK = "BLOCK" # Significant issues blocking uplift


def summarize_budget_and_metrics_for_uplift(
    budget_risk_map: Dict[str, Any],
    metric_readiness_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a joint summary of budget risk and metric readiness for uplift decisions.

    Combines budget risk assessment with metric readiness to provide
    a unified uplift gate decision.

    CONTRACT:
        - schema_version: always BUDGET_METRIC_UPLIFT_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects
        - Non-interpretive (neutral language)

    Parameters
    ----------
    budget_risk_map : dict
        Budget risk map from build_budget_risk_map().
    metric_readiness_summary : dict
        Metric readiness summary (expected fields: ready, blocking_metrics).

    Returns
    -------
    dict
        Joint uplift summary with:
        - schema_version
        - uplift_ready: bool
        - blocking_slices: list of slices blocking uplift
        - status: "OK" | "WARN" | "BLOCK"
        - notes: list of neutral rationale strings
    """
    # Extract budget risk data
    budget_risk_band = budget_risk_map.get("risk_band", "LOW")
    slices_drift = budget_risk_map.get("slices_with_concerning_drift", [])
    slices_invalid = budget_risk_map.get("slices_with_repeated_invalid", [])

    # Combine blocking slices (union, sorted)
    blocking_slices = sorted(set(slices_drift) | set(slices_invalid))

    # Extract metric readiness data
    metrics_ready = metric_readiness_summary.get("ready", True)
    blocking_metrics = metric_readiness_summary.get("blocking_metrics", [])

    notes: List[str] = []
    status: JointUpliftStatus

    # Determine joint status
    if budget_risk_band == "HIGH":
        status = JointUpliftStatus.BLOCK
        notes.append("Budget risk band is HIGH")
        if slices_drift:
            notes.append(f"{len(slices_drift)} slice(s) show concerning drift")
        if slices_invalid:
            notes.append(f"{len(slices_invalid)} slice(s) have repeated INVALID classifications")
    elif not metrics_ready:
        status = JointUpliftStatus.BLOCK
        notes.append("Metrics are not ready for uplift")
        if blocking_metrics:
            notes.append(f"Blocking metrics: {', '.join(blocking_metrics)}")
    elif budget_risk_band == "MEDIUM":
        status = JointUpliftStatus.WARN
        notes.append("Budget risk band is MEDIUM")
        if slices_drift:
            notes.append(f"Drift detected in: {', '.join(slices_drift)}")
        if slices_invalid:
            notes.append(f"Repeated INVALIDs in: {', '.join(slices_invalid)}")
    else:
        status = JointUpliftStatus.OK
        notes.append("Budget and metrics are within acceptable bounds")

    # Determine overall uplift readiness
    uplift_ready = status in (JointUpliftStatus.OK, JointUpliftStatus.WARN) and metrics_ready

    return {
        "schema_version": BUDGET_METRIC_UPLIFT_SCHEMA_VERSION,
        "uplift_ready": uplift_ready,
        "blocking_slices": blocking_slices,
        "status": status.value,
        "notes": notes,
    }


# =============================================================================
# BUDGET SENTINEL GRID v4 — Director Budget Panel
# =============================================================================

BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION = "1.0.0"


class StatusLight(str, Enum):
    """Traffic light status for director panel."""
    GREEN = "GREEN"   # All systems healthy
    YELLOW = "YELLOW" # Caution, some concerns
    RED = "RED"       # Significant issues


def build_budget_director_panel(
    global_health_budget: Dict[str, Any],
    budget_risk_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director-level budget panel for executive dashboards.

    Provides a high-level summary of budget posture suitable for
    non-technical stakeholders.

    CONTRACT:
        - schema_version: always BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects
        - Non-interpretive headline (neutral language)

    Parameters
    ----------
    global_health_budget : dict
        Global health signal from summarize_budget_for_global_health().
    budget_risk_map : dict
        Budget risk map from build_budget_risk_map().

    Returns
    -------
    dict
        Director panel with:
        - schema_version
        - status_light: "GREEN" | "YELLOW" | "RED"
        - budget_stability: "STABLE" | "DEGRADING" | "UNSTABLE"
        - uplift_budget_ready: bool
        - headline: neutral summary of budget posture
        - details: additional metrics for drill-down
    """
    # Extract data
    budget_stability = global_health_budget.get("budget_stability", "STABLE")
    uplift_budget_ready = global_health_budget.get("uplift_budget_ready", True)
    unsafe_slices = global_health_budget.get("unsafe_slices", [])
    summary_metrics = global_health_budget.get("summary_metrics", {})

    risk_band = budget_risk_map.get("risk_band", "LOW")
    risk_summary = budget_risk_map.get("summary", {})

    # Determine status light
    status_light: StatusLight
    if budget_stability == "UNSTABLE" or risk_band == "HIGH":
        status_light = StatusLight.RED
    elif budget_stability == "DEGRADING" or risk_band == "MEDIUM":
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Build neutral headline
    headline: str
    total_slices = risk_summary.get("total_slices", 0)
    problem_slices = risk_summary.get("problem_slice_count", 0)

    if status_light == StatusLight.GREEN:
        headline = "Budget metrics are within expected bounds across all slices."
    elif status_light == StatusLight.YELLOW:
        if problem_slices > 0:
            headline = f"Budget monitoring indicates {problem_slices} of {total_slices} slice(s) require attention."
        else:
            headline = "Budget drift patterns detected; monitoring continues."
    else:  # RED
        if len(unsafe_slices) > 0:
            headline = f"Budget constraints exceeded in {len(unsafe_slices)} slice(s); uplift gating is active."
        else:
            headline = "Budget stability is outside acceptable parameters; review required."

    return {
        "schema_version": BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION,
        "status_light": status_light.value,
        "budget_stability": budget_stability,
        "uplift_budget_ready": uplift_budget_ready,
        "headline": headline,
        "details": {
            "run_count": summary_metrics.get("run_count", 0),
            "unsafe_slice_count": len(unsafe_slices),
            "risk_band": risk_band,
            "has_drift": summary_metrics.get("has_drift", False),
            "consecutive_invalids": summary_metrics.get("consecutive_invalids", 0),
        },
    }


# =============================================================================
# BUDGET SENTINEL GRID v5 — Temporal Budget Radar
# =============================================================================

BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION = "1.0.0"


class RiskTrend(str, Enum):
    """Temporal trend of risk over runs."""
    IMPROVING = "IMPROVING"   # Risk is decreasing over time
    STABLE = "STABLE"         # Risk is consistent
    DEGRADING = "DEGRADING"   # Risk is increasing over time
    UNKNOWN = "UNKNOWN"       # Insufficient data to determine trend


def _compute_invalid_streaks(classifications: List[str]) -> Dict[str, Any]:
    """
    Compute distribution of consecutive INVALID streaks.

    Returns:
        - max_streak: longest consecutive INVALID run
        - streak_count: number of distinct INVALID streaks
        - total_invalids: total INVALID classifications
    """
    if not classifications:
        return {
            "max_streak": 0,
            "streak_count": 0,
            "total_invalids": 0,
        }

    max_streak = 0
    streak_count = 0
    current_streak = 0
    total_invalids = 0

    for c in classifications:
        if c == "INVALID":
            current_streak += 1
            total_invalids += 1
            max_streak = max(max_streak, current_streak)
        else:
            if current_streak > 0:
                streak_count += 1
            current_streak = 0

    # Count final streak if it ended with INVALID
    if current_streak > 0:
        streak_count += 1

    return {
        "max_streak": max_streak,
        "streak_count": streak_count,
        "total_invalids": total_invalids,
    }


def _determine_risk_trend(risk_bands: List[str]) -> RiskTrend:
    """
    Determine the trend of risk bands over time.

    Compares first half to second half of the series.
    """
    if len(risk_bands) < 2:
        return RiskTrend.UNKNOWN

    # Map risk bands to numeric values
    risk_values = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    # Convert to numeric
    numeric = [risk_values.get(r, 1) for r in risk_bands]

    # Compare first half vs second half
    mid = len(numeric) // 2
    first_half = numeric[:mid] if mid > 0 else numeric[:1]
    second_half = numeric[mid:]

    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0

    # Threshold for determining trend (0.5 = half a level)
    threshold = 0.3

    if second_avg < first_avg - threshold:
        return RiskTrend.IMPROVING
    elif second_avg > first_avg + threshold:
        return RiskTrend.DEGRADING
    else:
        return RiskTrend.STABLE


def build_budget_risk_trajectory(
    budget_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a temporal trajectory of budget risk over multiple runs.

    Tracks how risk evolves over time, identifying trends and
    chronic invalid streaks that may indicate systemic issues.

    CONTRACT:
        - schema_version: always BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION
        - Deterministic: same history → same trajectory
        - Pure function with no side effects

    Parameters
    ----------
    budget_history : Sequence[dict]
        List of budget snapshots or risk maps, ordered chronologically.
        Each entry should have 'classification' and optionally 'risk_band'.

    Returns
    -------
    dict
        Risk trajectory with:
        - schema_version
        - run_count: number of runs analyzed
        - risk_band_series: list of risk bands over time
        - invalid_run_streaks: distribution of INVALID streaks
        - trend: "IMPROVING" | "STABLE" | "DEGRADING" | "UNKNOWN"
        - summary: aggregated trajectory metrics
    """
    if not budget_history:
        return {
            "schema_version": BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION,
            "run_count": 0,
            "risk_band_series": [],
            "invalid_run_streaks": {
                "max_streak": 0,
                "streak_count": 0,
                "total_invalids": 0,
            },
            "trend": RiskTrend.UNKNOWN.value,
            "summary": {
                "improving_runs": 0,
                "stable_runs": 0,
                "degrading_runs": 0,
            },
        }

    # Extract classifications and risk bands
    classifications: List[str] = []
    risk_bands: List[str] = []

    for entry in budget_history:
        # Get classification
        classification = entry.get("classification", "UNKNOWN")
        classifications.append(classification)

        # Get or infer risk band
        risk_band = entry.get("risk_band")
        if risk_band is None:
            # Infer from classification
            if classification == "INVALID":
                risk_band = "HIGH"
            elif classification == "SUSPICIOUS":
                risk_band = "MEDIUM"
            else:
                risk_band = "LOW"
        risk_bands.append(risk_band)

    # Compute invalid streaks
    invalid_streaks = _compute_invalid_streaks(classifications)

    # Determine trend
    trend = _determine_risk_trend(risk_bands)

    # Count runs by trend direction (comparing each to previous)
    improving_runs = 0
    stable_runs = 0
    degrading_runs = 0
    risk_values = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    for i in range(1, len(risk_bands)):
        prev_val = risk_values.get(risk_bands[i - 1], 1)
        curr_val = risk_values.get(risk_bands[i], 1)
        if curr_val < prev_val:
            improving_runs += 1
        elif curr_val > prev_val:
            degrading_runs += 1
        else:
            stable_runs += 1

    return {
        "schema_version": BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION,
        "run_count": len(budget_history),
        "risk_band_series": risk_bands,
        "invalid_run_streaks": invalid_streaks,
        "trend": trend.value,
        "summary": {
            "improving_runs": improving_runs,
            "stable_runs": stable_runs,
            "degrading_runs": degrading_runs,
        },
    }


# =============================================================================
# BUDGET SENTINEL GRID v5 — Policy Interaction View
# =============================================================================

BUDGET_POLICY_IMPACT_SCHEMA_VERSION = "1.0.0"


def summarize_budget_impact_on_policy(
    budget_trajectory: Dict[str, Any],
    policy_drift_radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze the correlation between budget risk and policy drift.

    Identifies slices where high budget risk coincides with high
    policy drift, suggesting budget constraints may be limiting
    policy learning.

    CONTRACT:
        - schema_version: always BUDGET_POLICY_IMPACT_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects
        - Non-interpretive (neutral language)

    Parameters
    ----------
    budget_trajectory : dict
        Budget risk trajectory from build_budget_risk_trajectory().
    policy_drift_radar : dict
        Policy drift radar (expected fields: slices_with_drift, high_drift_slices).

    Returns
    -------
    dict
        Policy impact summary with:
        - schema_version
        - correlated_slices: slices with both high budget risk and high policy drift
        - suspected_budget_limited_learning: bool
        - notes: list of neutral observation strings
        - correlation_strength: "NONE" | "WEAK" | "MODERATE" | "STRONG"
    """
    # Extract budget trajectory data
    trend = budget_trajectory.get("trend", "UNKNOWN")
    invalid_streaks = budget_trajectory.get("invalid_run_streaks", {})
    max_streak = invalid_streaks.get("max_streak", 0)
    risk_band_series = budget_trajectory.get("risk_band_series", [])

    # Count high-risk runs
    high_risk_runs = sum(1 for r in risk_band_series if r == "HIGH")
    total_runs = len(risk_band_series)

    # Extract policy drift data
    slices_with_drift = set(policy_drift_radar.get("slices_with_drift", []))
    high_drift_slices = set(policy_drift_radar.get("high_drift_slices", []))

    # For budget, we track slices with concerning patterns
    # We'll use the trajectory's pattern of high risk as a proxy
    # In practice, this would come from per-slice budget tracking
    budget_high_risk_slices = set(policy_drift_radar.get("budget_high_risk_slices", []))

    # Find correlated slices (intersection of high budget risk and high policy drift)
    correlated_slices = sorted(budget_high_risk_slices & high_drift_slices)

    # If no explicit budget slice data, check for systemic correlation
    # (high budget risk + high policy drift globally)
    systemic_correlation = False
    if not correlated_slices and total_runs > 0:
        # Check if high budget risk coincides with high policy drift
        high_risk_ratio = high_risk_runs / total_runs if total_runs > 0 else 0
        has_high_drift = len(high_drift_slices) > 0

        if high_risk_ratio > 0.3 and has_high_drift:
            systemic_correlation = True

    # Determine if budget may be limiting learning
    suspected_budget_limited_learning = (
        len(correlated_slices) > 0 or
        systemic_correlation or
        (max_streak >= 3 and len(high_drift_slices) > 0)
    )

    # Determine correlation strength
    if len(correlated_slices) > 2 or (max_streak >= 3 and len(high_drift_slices) > 2):
        correlation_strength = "STRONG"
    elif len(correlated_slices) > 0 or systemic_correlation:
        correlation_strength = "MODERATE"
    elif max_streak >= 2 and len(slices_with_drift) > 0:
        correlation_strength = "WEAK"
    else:
        correlation_strength = "NONE"

    # Build notes
    notes: List[str] = []

    if trend == "DEGRADING":
        notes.append("Budget risk trend is degrading over time")
    elif trend == "IMPROVING":
        notes.append("Budget risk trend is improving over time")

    if max_streak >= 3:
        notes.append(f"Chronic INVALID streak detected ({max_streak} consecutive runs)")
    elif max_streak >= 2:
        notes.append(f"Repeated INVALID classifications detected ({max_streak} consecutive runs)")

    if len(correlated_slices) > 0:
        notes.append(f"{len(correlated_slices)} slice(s) show both high budget risk and high policy drift")

    if systemic_correlation and not correlated_slices:
        notes.append("Systemic correlation detected between budget constraints and policy drift")

    if suspected_budget_limited_learning:
        notes.append("Budget constraints may be limiting policy learning")
    elif correlation_strength == "NONE":
        notes.append("No significant correlation detected between budget and policy drift")

    return {
        "schema_version": BUDGET_POLICY_IMPACT_SCHEMA_VERSION,
        "correlated_slices": correlated_slices,
        "suspected_budget_limited_learning": suspected_budget_limited_learning,
        "notes": notes,
        "correlation_strength": correlation_strength,
    }


# =============================================================================
# BUDGET SENTINEL GRID v5 — Extended Director Panel
# =============================================================================

BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION = "1.0.0"


def build_budget_director_panel_v2(
    global_health_budget: Dict[str, Any],
    budget_risk_map: Dict[str, Any],
    budget_trajectory: Optional[Dict[str, Any]] = None,
    budget_policy_impact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an extended director-level budget panel with temporal and policy context.

    Enriches the base director panel with:
    - Risk trend over time
    - Maximum invalid streak length
    - Optional budget-policy correlation flag

    CONTRACT:
        - schema_version: always BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects
        - Non-interpretive headline (neutral language)

    Parameters
    ----------
    global_health_budget : dict
        Global health signal from summarize_budget_for_global_health().
    budget_risk_map : dict
        Budget risk map from build_budget_risk_map().
    budget_trajectory : dict, optional
        Budget risk trajectory from build_budget_risk_trajectory().
    budget_policy_impact : dict, optional
        Budget-policy impact from summarize_budget_impact_on_policy().

    Returns
    -------
    dict
        Extended director panel with:
        - schema_version
        - status_light: "GREEN" | "YELLOW" | "RED"
        - budget_stability
        - uplift_budget_ready
        - headline
        - trend: "IMPROVING" | "STABLE" | "DEGRADING" | "UNKNOWN"
        - invalid_streaks_max: maximum consecutive INVALID count
        - budget_policy_correlation_flag: optional bool
        - details: additional metrics
    """
    # Get base panel data
    budget_stability = global_health_budget.get("budget_stability", "STABLE")
    uplift_budget_ready = global_health_budget.get("uplift_budget_ready", True)
    unsafe_slices = global_health_budget.get("unsafe_slices", [])
    summary_metrics = global_health_budget.get("summary_metrics", {})

    risk_band = budget_risk_map.get("risk_band", "LOW")
    risk_summary = budget_risk_map.get("summary", {})

    # Extract trajectory data
    trend = RiskTrend.UNKNOWN.value
    invalid_streaks_max = 0

    if budget_trajectory:
        trend = budget_trajectory.get("trend", RiskTrend.UNKNOWN.value)
        invalid_streaks = budget_trajectory.get("invalid_run_streaks", {})
        invalid_streaks_max = invalid_streaks.get("max_streak", 0)

    # Extract policy correlation data
    budget_policy_correlation_flag: Optional[bool] = None
    if budget_policy_impact:
        budget_policy_correlation_flag = budget_policy_impact.get(
            "suspected_budget_limited_learning", False
        )

    # Determine status light (enhanced with trend)
    status_light: StatusLight
    if budget_stability == "UNSTABLE" or risk_band == "HIGH":
        status_light = StatusLight.RED
    elif budget_stability == "DEGRADING" or risk_band == "MEDIUM":
        status_light = StatusLight.YELLOW
    elif trend == "DEGRADING" and invalid_streaks_max >= 2:
        # Upgrade to YELLOW if degrading trend with repeated invalids
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Build neutral headline (enhanced with trend context)
    headline: str
    total_slices = risk_summary.get("total_slices", 0)
    problem_slices = risk_summary.get("problem_slice_count", 0)

    if status_light == StatusLight.GREEN:
        if trend == "IMPROVING":
            headline = "Budget metrics are within expected bounds; risk trend is improving."
        else:
            headline = "Budget metrics are within expected bounds across all slices."
    elif status_light == StatusLight.YELLOW:
        if trend == "DEGRADING":
            headline = f"Budget risk is increasing; {problem_slices} of {total_slices} slice(s) require attention."
        elif problem_slices > 0:
            headline = f"Budget monitoring indicates {problem_slices} of {total_slices} slice(s) require attention."
        else:
            headline = "Budget drift patterns detected; monitoring continues."
    else:  # RED
        if invalid_streaks_max >= 3:
            headline = f"Chronic budget constraints detected ({invalid_streaks_max} consecutive INVALID runs); review required."
        elif len(unsafe_slices) > 0:
            headline = f"Budget constraints exceeded in {len(unsafe_slices)} slice(s); uplift gating is active."
        else:
            headline = "Budget stability is outside acceptable parameters; review required."

    # Build result
    result: Dict[str, Any] = {
        "schema_version": BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION,
        "status_light": status_light.value,
        "budget_stability": budget_stability,
        "uplift_budget_ready": uplift_budget_ready,
        "headline": headline,
        "trend": trend,
        "invalid_streaks_max": invalid_streaks_max,
        "details": {
            "run_count": summary_metrics.get("run_count", 0),
            "unsafe_slice_count": len(unsafe_slices),
            "risk_band": risk_band,
            "has_drift": summary_metrics.get("has_drift", False),
            "consecutive_invalids": summary_metrics.get("consecutive_invalids", 0),
        },
    }

    # Add optional policy correlation flag
    if budget_policy_correlation_flag is not None:
        result["budget_policy_correlation_flag"] = budget_policy_correlation_flag

    return result


# =============================================================================
# BUDGET SENTINEL GRID v5 — Global Console Adapter
# =============================================================================

BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION = "1.0.0"


def summarize_budget_trajectory_for_global_console(
    trajectory: Dict[str, Any],
    director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize budget trajectory for the global console dashboard.

    Provides a condensed view of budget health suitable for
    system-wide monitoring and alerting.

    CONTRACT:
        - schema_version: always BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects

    Parameters
    ----------
    trajectory : dict
        Budget risk trajectory from build_budget_risk_trajectory().
    director_panel : dict
        Director panel from build_budget_director_panel_v2().

    Returns
    -------
    dict
        Global console summary with:
        - schema_version
        - budget_ok: bool (True if status is GREEN or YELLOW without blocks)
        - status_light: "GREEN" | "YELLOW" | "RED"
        - trend: "IMPROVING" | "STABLE" | "DEGRADING" | "UNKNOWN"
        - invalid_run_streaks_max: maximum consecutive INVALID count
        - summary: additional context metrics
    """
    # Extract from director panel
    status_light = director_panel.get("status_light", "GREEN")
    uplift_budget_ready = director_panel.get("uplift_budget_ready", True)

    # Extract from trajectory
    trend = trajectory.get("trend", "UNKNOWN")
    invalid_streaks = trajectory.get("invalid_run_streaks", {})
    invalid_run_streaks_max = invalid_streaks.get("max_streak", 0)
    run_count = trajectory.get("run_count", 0)

    # Determine budget_ok
    # Budget is OK if:
    # - Status is GREEN, or
    # - Status is YELLOW and uplift is still ready (minor concerns)
    budget_ok = (
        status_light == "GREEN" or
        (status_light == "YELLOW" and uplift_budget_ready)
    )

    return {
        "schema_version": BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION,
        "budget_ok": budget_ok,
        "status_light": status_light,
        "trend": trend,
        "invalid_run_streaks_max": invalid_run_streaks_max,
        "summary": {
            "run_count": run_count,
            "uplift_budget_ready": uplift_budget_ready,
            "has_chronic_invalids": invalid_run_streaks_max >= 3,
        },
    }


# =============================================================================
# BUDGET SENTINEL GRID v5 — Governance Signal Adapter
# =============================================================================

BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION = "1.0.0"


class GovernanceSignal(str, Enum):
    """Governance signal for budget risk."""
    OK = "OK"       # Proceed with uplift
    WARN = "WARN"   # Proceed with caution
    BLOCK = "BLOCK" # Do not proceed


def to_governance_signal_for_budget(
    trajectory: Dict[str, Any],
    policy_impact: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert budget trajectory and policy impact to a governance signal.

    Maps temporal budget risk patterns to governance decisions:
    - DEGRADING + long INVALID streaks → BLOCK
    - STABLE + high invalids → WARN
    - IMPROVING → OK

    CONTRACT:
        - schema_version: always BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION
        - Deterministic: same inputs → same output
        - Pure function with no side effects

    Parameters
    ----------
    trajectory : dict
        Budget risk trajectory from build_budget_risk_trajectory().
    policy_impact : dict
        Budget-policy impact from summarize_budget_impact_on_policy().

    Returns
    -------
    dict
        Governance signal with:
        - schema_version
        - signal: "OK" | "WARN" | "BLOCK"
        - reasons: list of reasons for the signal
        - can_proceed: bool (True if OK or WARN)
        - requires_review: bool (True if WARN or BLOCK)
    """
    # Extract trajectory data
    trend = trajectory.get("trend", "UNKNOWN")
    invalid_streaks = trajectory.get("invalid_run_streaks", {})
    max_streak = invalid_streaks.get("max_streak", 0)
    total_invalids = invalid_streaks.get("total_invalids", 0)
    run_count = trajectory.get("run_count", 0)

    # Extract policy impact data
    suspected_limited_learning = policy_impact.get("suspected_budget_limited_learning", False)
    correlation_strength = policy_impact.get("correlation_strength", "NONE")

    reasons: List[str] = []
    signal: GovernanceSignal

    # BLOCK conditions
    if trend == "DEGRADING" and max_streak >= 3:
        signal = GovernanceSignal.BLOCK
        reasons.append("Budget trend is DEGRADING with chronic INVALID streaks")
        reasons.append(f"Maximum consecutive INVALIDs: {max_streak}")
    elif trend == "DEGRADING" and suspected_limited_learning:
        signal = GovernanceSignal.BLOCK
        reasons.append("Budget trend is DEGRADING and may be limiting policy learning")
    elif max_streak >= 4:
        # Very long INVALID streak is always concerning
        signal = GovernanceSignal.BLOCK
        reasons.append(f"Severe INVALID streak detected ({max_streak} consecutive runs)")
    elif correlation_strength == "STRONG":
        signal = GovernanceSignal.BLOCK
        reasons.append("Strong correlation between budget constraints and policy drift")

    # WARN conditions
    elif trend == "STABLE" and max_streak >= 2:
        signal = GovernanceSignal.WARN
        reasons.append("Budget trend is STABLE but has repeated INVALID classifications")
        reasons.append(f"Maximum consecutive INVALIDs: {max_streak}")
    elif trend == "DEGRADING":
        signal = GovernanceSignal.WARN
        reasons.append("Budget trend is DEGRADING")
    elif run_count > 0 and total_invalids / run_count > 0.3:
        signal = GovernanceSignal.WARN
        reasons.append(f"High INVALID rate ({total_invalids}/{run_count} runs)")
    elif suspected_limited_learning:
        signal = GovernanceSignal.WARN
        reasons.append("Budget constraints may be limiting policy learning")
    elif correlation_strength == "MODERATE":
        signal = GovernanceSignal.WARN
        reasons.append("Moderate correlation between budget constraints and policy drift")

    # OK conditions
    elif trend == "IMPROVING":
        signal = GovernanceSignal.OK
        reasons.append("Budget trend is IMPROVING")
    elif trend == "STABLE" and max_streak < 2:
        signal = GovernanceSignal.OK
        reasons.append("Budget trend is STABLE with no concerning patterns")
    else:
        # Default to OK for UNKNOWN or other cases without issues
        signal = GovernanceSignal.OK
        reasons.append("No concerning budget patterns detected")

    return {
        "schema_version": BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION,
        "signal": signal.value,
        "reasons": reasons,
        "can_proceed": signal in (GovernanceSignal.OK, GovernanceSignal.WARN),
        "requires_review": signal in (GovernanceSignal.WARN, GovernanceSignal.BLOCK),
    }
