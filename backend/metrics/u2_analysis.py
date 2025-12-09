
"""
U2 Uplift Analytics

==============================================================================
STATUS: PHASE II — IMPLEMENTATION
==============================================================================

This module contains the implementation for the U2 uplift analytics pipeline.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, ClassVar

import numpy as np
from scipy.stats import norm


# ===========================================================================
# PHASE II — ANALYTICS API (DESIGN CONTRACT)
# ===========================================================================

@dataclass
class U2ExperimentData:
    """
    Represents the complete data for a U2 uplift experiment, typically for a single slice.
    This is the primary input to compute_uplift_metrics().
    """
    slice_id: str
    baseline_records: List[Dict[str, Any]] = field(default_factory=list)
    rfl_records: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class U2UpliftResult:
    """
    Contains the full analysis results for a U2 experiment slice.
    This is the output of compute_uplift_metrics().
    """
    slice_id: str
    n_baseline: int
    n_rfl: int
    passes_governance: bool
    bootstrap_seed: int
    n_bootstrap: int
    
    # Uplift metrics (e.g., success rate, throughput)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Governance evaluation
    governance_details: Dict[str, bool] = field(default_factory=dict)


def load_u2_experiment(
    baseline_log_path: str, 
    rfl_log_path: str
) -> U2ExperimentData:
    """
    Loads, parses, and combines baseline and RFL JSONL logs for one experiment slice.
    """
    baseline_records = []
    with open(baseline_log_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                baseline_records.append(record)

    rfl_records = []
    with open(rfl_log_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                rfl_records.append(record)

    slice_id = ""
    if baseline_records:
        slice_id = baseline_records[0].get("slice", "")
    elif rfl_records:
        slice_id = rfl_records[0].get("slice", "")

    return U2ExperimentData(
        slice_id=slice_id,
        baseline_records=baseline_records,
        rfl_records=rfl_records,
        metadata={"baseline_log_path": baseline_log_path, "rfl_log_path": rfl_log_path}
    )


def compute_uplift_metrics(
    data: U2ExperimentData, 
    bootstrap_seed: int = 42,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> U2UpliftResult:
    """
    Computes success rates, throughput, and bootstrap CIs for the delta.
    """
    baseline_records = data.baseline_records
    rfl_records = data.rfl_records

    n_baseline = len(baseline_records)
    n_rfl = len(rfl_records)

    # --- Success Rate ---
    baseline_successes = sum(1 for r in baseline_records if r.get("success"))
    rfl_successes = sum(1 for r in rfl_records if r.get("success"))
    
    baseline_success_rate = baseline_successes / n_baseline if n_baseline > 0 else 0
    rfl_success_rate = rfl_successes / n_rfl if n_rfl > 0 else 0

    # This is a simplification. A CI on the delta of proportions is more complex.
    # Here we just calculate the CI on the RFL rate.
    success_rate_ci = compute_wilson_ci(rfl_successes, n_rfl, confidence)

    # --- Abstention Rate ---
    baseline_abstentions = sum(1 for r in baseline_records if r.get("abstention_count", 0) > 0)
    rfl_abstentions = sum(1 for r in rfl_records if r.get("abstention_count", 0) > 0)

    baseline_abstention_rate = baseline_abstentions / n_baseline if n_baseline > 0 else 0
    rfl_abstention_rate = rfl_abstentions / n_rfl if n_rfl > 0 else 0
    
    abstention_rate_ci = compute_wilson_ci(rfl_abstentions, n_rfl, confidence)

    # --- Throughput ---
    baseline_durations = [r["duration_seconds"] for r in baseline_records if "duration_seconds" in r]
    rfl_durations = [r["duration_seconds"] for r in rfl_records if "duration_seconds" in r]

    throughput_metrics = bootstrap_delta(
        baseline_durations,
        rfl_durations,
        statistic="throughput",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=bootstrap_seed,
    )

    metrics = {
        "success_rate": {
            "baseline": baseline_success_rate,
            "rfl": rfl_success_rate,
            "delta": rfl_success_rate - baseline_success_rate,
            "ci": success_rate_ci,
        },
        "abstention_rate": {
            "baseline": baseline_abstention_rate,
            "rfl": rfl_abstention_rate,
            "delta": rfl_abstention_rate - baseline_abstention_rate,
            "ci": abstention_rate_ci,
        },
        "throughput": throughput_metrics,
    }

    # --- Governance ---
    criteria = SLICE_SUCCESS_CRITERIA.get(data.slice_id, {})
    governance_details = {}
    if criteria:
        governance_details["sample_size_passed"] = (n_baseline >= criteria.get("min_samples", 0) and n_rfl >= criteria.get("min_samples", 0))
        governance_details["success_rate_passed"] = rfl_success_rate >= criteria.get("min_success_rate", 0)
        governance_details["abstention_rate_passed"] = rfl_abstention_rate <= criteria.get("max_abstention_rate", 1)
        governance_details["throughput_uplift_passed"] = throughput_metrics.get("delta_pct", 0) >= criteria.get("min_throughput_uplift_pct", 0)
    
    passes_governance = all(governance_details.values())

    return U2UpliftResult(
        slice_id=data.slice_id,
        n_baseline=n_baseline,
        n_rfl=n_rfl,
        metrics=metrics,
        passes_governance=passes_governance,
        governance_details=governance_details,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )


def render_u2_summary(
    result: U2UpliftResult
) -> Dict[str, Any]:
    """
    Renders the uplift result into a human-readable summary dictionary.
    """
    summary = {
        "slice_id": result.slice_id,
        "sample_size": {
            "baseline": result.n_baseline,
            "rfl": result.n_rfl,
        },
        "metrics": result.metrics,
        "governance": {
            "passed": result.passes_governance,
            "details": result.governance_details,
        },
        "reproducibility": {
            "bootstrap_seed": result.bootstrap_seed,
            "n_bootstrap": result.n_bootstrap,
        }
    }
    return summary


# ===========================================================================
# STATISTICAL IMPLEMENTATIONS
# ===========================================================================

def compute_wilson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.
    """
    if trials <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > trials:
        raise ValueError("Successes must be between 0 and trials.")

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    
    n = trials
    
    center = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))

    ci_low = center - margin
    ci_high = center + margin
    
    return (max(0.0, ci_low), min(1.0, ci_high))


def bootstrap_delta(
    baseline_values: List[float],
    treatment_values: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for the difference (delta)
    between treatment and baseline statistics.
    """
    if not baseline_values or not treatment_values:
        raise ValueError("Input lists cannot be empty.")

    rng = np.random.default_rng(seed)
    
    baseline = np.array(baseline_values)
    treatment = np.array(treatment_values)

    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    elif statistic == "throughput":
        stat_fn = lambda x: 1.0 / np.mean(x) if np.mean(x) != 0 else 0
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    baseline_stat = stat_fn(baseline)
    treatment_stat = stat_fn(treatment)
    delta_obs = treatment_stat - baseline_stat

    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        baseline_sample = rng.choice(baseline, size=len(baseline), replace=True)
        treatment_sample = rng.choice(treatment, size=len(treatment), replace=True)
        
        baseline_sample_stat = stat_fn(baseline_sample)
        treatment_sample_stat = stat_fn(treatment_sample)
        
        bootstrap_deltas.append(treatment_sample_stat - baseline_sample_stat)

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_deltas, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2))
    
    significant = ci_low > 0 or ci_high < 0

    delta_pct = (delta_obs / abs(baseline_stat)) * 100 if baseline_stat != 0 else 0
    
    # This is a simplification. A proper CI for the percentage would require bootstrapping the ratio.
    delta_pct_ci_low = (ci_low / abs(baseline_stat)) * 100 if baseline_stat != 0 else 0
    delta_pct_ci_high = (ci_high / abs(baseline_stat)) * 100 if baseline_stat != 0 else 0

    return {
        "baseline_stat": baseline_stat,
        "treatment_stat": treatment_stat,
        "delta": delta_obs,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "delta_pct": delta_pct,
        "delta_pct_ci_low": delta_pct_ci_low,
        "delta_pct_ci_high": delta_pct_ci_high,
        "significant": significant,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence": confidence,
        "statistic": statistic,
    }


# ===========================================================================
# PAIRED BOOTSTRAP HELPER (D3 Integration)
# ===========================================================================

import hashlib
from pathlib import Path

# Canonical fields for PairedDeltaResult (alphabetically sorted)
# This defines the contract for to_dict() output
# LOCKED: Any changes require updating bootstrap_contract.json
PAIRED_DELTA_RESULT_FIELDS = (
    "analysis_id",
    "ci_lower",
    "ci_upper",
    "delta",
    "method",
    "metric_path",
    "n_baseline",
    "n_bootstrap",
    "n_rfl",
    "seed",
)

# Schema version for backwards compatibility tracking
PAIRED_DELTA_RESULT_SCHEMA_VERSION = "1.0.0"


def compute_analysis_id(
    baseline_path: str,
    rfl_path: str,
    metric_path: str,
    seed: int,
    n_bootstrap: int,
) -> str:
    """
    Compute deterministic analysis ID as SHA-256 hash of input parameters.
    
    This provides a unique, reproducible identifier for each analysis run.
    The same inputs will always produce the same analysis_id.
    
    Parameters
    ----------
    baseline_path : str
        Path to baseline log file.
    rfl_path : str
        Path to RFL log file.
    metric_path : str
        Dot-notation metric path.
    seed : int
        Random seed.
    n_bootstrap : int
        Number of bootstrap resamples.
    
    Returns
    -------
    str
        64-character hex SHA-256 hash.
    """
    # Normalize paths to use forward slashes for cross-platform consistency
    normalized_baseline = str(Path(baseline_path).as_posix())
    normalized_rfl = str(Path(rfl_path).as_posix())
    
    # Create canonical string representation
    canonical = f"{normalized_baseline}|{normalized_rfl}|{metric_path}|{seed}|{n_bootstrap}"
    
    # Compute SHA-256
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


@dataclass
class PairedDeltaResult:
    """
    Result of paired bootstrap delta computation from logs.
    
    This is the CANONICAL return type for bootstrap delta computations.
    It provides a stable, deterministic interface for U2 analysis workflows.
    
    EVIDENCE CONTRACT (LOCKED):
        - Exactly 10 fields, alphabetically sorted in to_dict()
        - analysis_id: deterministic SHA-256 of inputs
        - All fields are primitive types (no nested randomness)
        - Field set is frozen: PAIRED_DELTA_RESULT_FIELDS
        - Changes require updating bootstrap_contract.json
    
    SCHEMA:
        {
            "analysis_id": "string",  # SHA-256 of input parameters
            "ci_lower": float,        # Lower bound of 95% CI
            "ci_upper": float,        # Upper bound of 95% CI
            "delta": float,           # Point estimate: mean(rfl) - mean(baseline)
            "method": "string",       # Bootstrap method (percentile, BCa, etc.)
            "metric_path": "string",  # Dot-notation path to metric
            "n_baseline": int,        # Number of baseline observations
            "n_bootstrap": int,       # Number of bootstrap resamples
            "n_rfl": int,             # Number of RFL observations
            "seed": int               # Random seed used
        }
    """
    delta: float  # Point estimate: mean(rfl) - mean(baseline)
    ci_lower: float  # Lower bound of 95% CI
    ci_upper: float  # Upper bound of 95% CI
    n_baseline: int  # Number of baseline observations
    n_rfl: int  # Number of RFL observations
    metric_path: str  # Path to extracted metric
    seed: int  # Random seed used
    n_bootstrap: int  # Number of resamples
    method: str  # Bootstrap method used (percentile, BCa, etc.)
    analysis_id: str  # Deterministic SHA-256 of input parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        CONTRACT:
            - Exactly 10 keys, sorted alphabetically
            - All values are JSON-serializable primitives
            - No nested randomness or non-deterministic fields
            - Output is stable and can be used directly by governance verifier
        """
        # Return fields in canonical alphabetical order (LOCKED)
        return {
            "analysis_id": str(self.analysis_id),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "delta": float(self.delta),
            "method": str(self.method),
            "metric_path": str(self.metric_path),
            "n_baseline": int(self.n_baseline),
            "n_bootstrap": int(self.n_bootstrap),
            "n_rfl": int(self.n_rfl),
            "seed": int(self.seed),
        }
    
    def to_json(self) -> str:
        """
        Serialize to JSON string with sorted keys.
        
        Guaranteed deterministic: same data → same string.
        """
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def get_field_names(cls) -> Tuple[str, ...]:
        """Return canonical field names in alphabetical order."""
        return PAIRED_DELTA_RESULT_FIELDS
    
    @classmethod
    def get_schema_version(cls) -> str:
        """Return schema version for backwards compatibility tracking."""
        return PAIRED_DELTA_RESULT_SCHEMA_VERSION
    
    @classmethod
    def validate_dict(cls, d: Dict[str, Any]) -> bool:
        """
        Validate that a dictionary conforms to PairedDeltaResult schema.
        
        Returns True if valid, raises ValueError if invalid.
        """
        expected_fields = set(PAIRED_DELTA_RESULT_FIELDS)
        actual_fields = set(d.keys())
        
        missing = expected_fields - actual_fields
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        
        extra = actual_fields - expected_fields
        if extra:
            raise ValueError(f"Unexpected fields: {sorted(extra)}")
        
        # Type checks
        type_checks = {
            "analysis_id": str,
            "ci_lower": (int, float),
            "ci_upper": (int, float),
            "delta": (int, float),
            "method": str,
            "metric_path": str,
            "n_baseline": int,
            "n_bootstrap": int,
            "n_rfl": int,
            "seed": int,
        }
        
        for field, expected_type in type_checks.items():
            if not isinstance(d[field], expected_type):
                raise ValueError(
                    f"Field '{field}' has wrong type: expected {expected_type}, "
                    f"got {type(d[field])}"
                )
        
        return True


def _extract_metric_from_records(
    records: List[Dict[str, Any]],
    metric_path: str,
) -> List[float]:
    """
    Extract metric values from records using dot-notation path.
    
    Parameters
    ----------
    records : list of dict
        Log records from JSONL.
        
    metric_path : str
        Dot-separated path to metric (e.g., "success", "derivation.verified").
    
    Returns
    -------
    list of float
        Extracted numeric values. Booleans converted to 0/1.
    """
    path_parts = metric_path.split('.')
    values = []
    
    for record in records:
        value = record
        try:
            for part in path_parts:
                if isinstance(value, dict):
                    value = value[part]
                else:
                    raise KeyError(part)
            
            # Convert booleans to numeric
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                continue
            
            values.append(value)
        except (KeyError, TypeError):
            continue
    
    return values


# ===========================================================================
# EVIDENCE PACK CONTRACT (D3 v1.1)
# ===========================================================================

# Evidence Pack schema version - tracks multi-analysis pack format
EVIDENCE_PACK_SCHEMA_VERSION = "1.0.0"

# Forbidden interpretive words in governance summaries
# These words imply judgment; D3 produces evidence only
GOVERNANCE_SUMMARY_FORBIDDEN_WORDS = frozenset({
    "better", "worse", "improve", "improvement", "degrade", "degradation",
    "positive", "negative", "significant", "insignificant", "meaningful",
    "promising", "concerning", "increase", "decrease", "higher", "lower",
    "good", "bad", "successful", "failure", "fail", "success",
})


def build_evidence_pack(
    results: "Sequence[PairedDeltaResult]",
    pack_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate multiple PairedDeltaResult objects into a single evidence pack.
    
    This creates a deterministic, schema-compliant JSON structure that can
    be consumed by governance systems and auditors.
    
    CONTRACT:
        - schema_version: always EVIDENCE_PACK_SCHEMA_VERSION
        - analyses: list of PairedDeltaResult.to_dict(), sorted by (metric_path, analysis_id)
        - analysis_count: exact count of analyses
        - pack_id: optional identifier for the pack (SHA-256 if not provided)
    
    Parameters
    ----------
    results : Sequence[PairedDeltaResult]
        List of bootstrap results to aggregate.
        
    pack_id : str, optional
        Identifier for this evidence pack. If not provided, computed as
        SHA-256 of all analysis_ids concatenated.
    
    Returns
    -------
    dict
        Evidence pack JSON structure with deterministic ordering.
    
    Examples
    --------
    >>> from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
    >>> result1 = PairedDeltaResult(delta=0.1, ci_lower=0.05, ci_upper=0.15, ...)
    >>> result2 = PairedDeltaResult(delta=0.2, ci_lower=0.12, ci_upper=0.28, ...)
    >>> pack = build_evidence_pack([result1, result2])
    >>> pack["schema_version"]
    '1.0.0'
    >>> len(pack["analyses"]) == pack["analysis_count"]
    True
    """
    # Convert results to dicts
    analyses_raw = [r.to_dict() for r in results]
    
    # Sort by (metric_path, analysis_id) for deterministic ordering
    analyses_sorted = sorted(
        analyses_raw,
        key=lambda a: (a.get("metric_path", ""), a.get("analysis_id", ""))
    )
    
    # Compute pack_id if not provided
    if pack_id is None:
        # Concatenate all analysis_ids in sorted order
        ids_concat = "|".join(a.get("analysis_id", "") for a in analyses_sorted)
        pack_id = hashlib.sha256(ids_concat.encode("utf-8")).hexdigest()
    
    return {
        "analysis_count": len(analyses_sorted),
        "analyses": analyses_sorted,
        "pack_id": pack_id,
        "schema_version": EVIDENCE_PACK_SCHEMA_VERSION,
    }


def summarize_evidence_for_governance(
    pack: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a non-interpretive summary of an evidence pack for governance dashboards.
    
    This function extracts purely descriptive statistics from an evidence pack.
    It NEVER produces interpretive statements (e.g., "positive", "better", "worse").
    
    CONTRACT:
        - Output contains only numeric and structural facts
        - No sign interpretation or judgment words
        - Suitable for governance consumption without transformation
    
    Parameters
    ----------
    pack : dict
        Evidence pack as returned by build_evidence_pack().
    
    Returns
    -------
    dict
        Non-interpretive summary with:
        - analysis_count: number of analyses
        - metric_paths: list of unique metric paths (sorted)
        - methods_used: list of unique bootstrap methods (sorted)
        - sample_sizes: min/max for baseline and RFL
        - pack_id: identifier from the pack
        - schema_version: pack schema version
    
    Examples
    --------
    >>> pack = build_evidence_pack([result1, result2])
    >>> summary = summarize_evidence_for_governance(pack)
    >>> summary["analysis_count"]
    2
    >>> "better" in str(summary)  # No interpretive words
    False
    """
    analyses = pack.get("analyses", [])
    
    if not analyses:
        return {
            "analysis_count": 0,
            "metric_paths": [],
            "methods_used": [],
            "pack_id": pack.get("pack_id", ""),
            "sample_sizes": {
                "max_baseline": 0,
                "max_rfl": 0,
                "min_baseline": 0,
                "min_rfl": 0,
            },
            "schema_version": pack.get("schema_version", EVIDENCE_PACK_SCHEMA_VERSION),
        }
    
    # Extract unique metric paths and methods
    metric_paths = sorted(set(a.get("metric_path", "") for a in analyses))
    methods_used = sorted(set(a.get("method", "") for a in analyses))
    
    # Extract sample sizes
    baseline_sizes = [a.get("n_baseline", 0) for a in analyses]
    rfl_sizes = [a.get("n_rfl", 0) for a in analyses]
    
    return {
        "analysis_count": pack.get("analysis_count", len(analyses)),
        "metric_paths": metric_paths,
        "methods_used": methods_used,
        "pack_id": pack.get("pack_id", ""),
        "sample_sizes": {
            "max_baseline": max(baseline_sizes) if baseline_sizes else 0,
            "max_rfl": max(rfl_sizes) if rfl_sizes else 0,
            "min_baseline": min(baseline_sizes) if baseline_sizes else 0,
            "min_rfl": min(rfl_sizes) if rfl_sizes else 0,
        },
        "schema_version": pack.get("schema_version", EVIDENCE_PACK_SCHEMA_VERSION),
    }


def format_evidence_summary_line(
    result: "PairedDeltaResult",
) -> str:
    """
    Format a single evidence summary line for CI logs and human grepping.
    
    Format (single line, no newlines):
        BootstrapEvidence: metric=<metric_path> delta=<delta> ci=[<ci_lower>,<ci_upper>] n_base=<n_baseline> n_rfl=<n_rfl> method=<method>
    
    CONTRACT:
        - Exactly one line, no trailing newline
        - Deterministic format with fixed decimal precision
        - No interpretive language
    
    Parameters
    ----------
    result : PairedDeltaResult
        Bootstrap result to format.
    
    Returns
    -------
    str
        Single-line summary string.
    """
    return (
        f"BootstrapEvidence: "
        f"metric={result.metric_path} "
        f"delta={result.delta:.6f} "
        f"ci=[{result.ci_lower:.6f},{result.ci_upper:.6f}] "
        f"n_base={result.n_baseline} "
        f"n_rfl={result.n_rfl} "
        f"method={result.method}"
    )


# ===========================================================================
# PHASE III — GOVERNANCE LAYER FOR EVIDENCE PACKS (D3 v1.1)
# ===========================================================================

# Thresholds for evidence quality assessment
# These are structural thresholds, NOT effect-size thresholds
EVIDENCE_QUALITY_THRESHOLDS = {
    "min_sample_size_weak": 10,      # Below this is a weak point
    "min_sample_size_block": 2,       # Below this should block
    "min_analyses_for_review": 1,     # Need at least this many analyses
    # Quality tier thresholds
    "tier_1_max_sample_size": 20,     # TIER_1: low sample sizes
    "tier_2_min_sample_size": 20,     # TIER_2: adequate sample sizes
    "tier_3_min_sample_size": 50,     # TIER_3: strong sample sizes
    "tier_2_min_metrics": 2,          # TIER_2: multiple metrics
    "tier_3_min_metrics": 2,          # TIER_3: multiple metrics
}


def build_evidence_quality_snapshot(
    pack: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a quality snapshot of an evidence pack for governance review.
    
    This function derives quality metrics strictly from the pack's analyses.
    It does NOT interpret effect sizes or make uplift claims.
    
    CONTRACT:
        - All fields derived from pack analyses
        - No interpretive language
        - No effect-size judgments
    
    Parameters
    ----------
    pack : dict
        Evidence pack as returned by build_evidence_pack().
    
    Returns
    -------
    dict
        Quality snapshot with:
        - schema_version: pack schema version
        - analysis_count: number of analyses
        - metrics_with_multiple_methods: list of metric_paths with >1 method
        - methods_by_metric: dict mapping metric_path to list of methods
        - min_n_baseline, max_n_baseline: sample size bounds
        - min_n_rfl, max_n_rfl: sample size bounds
    
    Examples
    --------
    >>> pack = build_evidence_pack([result1, result2])
    >>> snapshot = build_evidence_quality_snapshot(pack)
    >>> snapshot["analysis_count"]
    2
    """
    analyses = pack.get("analyses", [])
    
    if not analyses:
        return {
            "analysis_count": 0,
            "max_n_baseline": 0,
            "max_n_rfl": 0,
            "methods_by_metric": {},
            "metrics_with_multiple_methods": [],
            "min_n_baseline": 0,
            "min_n_rfl": 0,
            "schema_version": pack.get("schema_version", EVIDENCE_PACK_SCHEMA_VERSION),
        }
    
    # Build methods_by_metric: metric_path -> sorted list of unique methods
    methods_by_metric: Dict[str, List[str]] = {}
    for analysis in analyses:
        metric_path = analysis.get("metric_path", "")
        method = analysis.get("method", "")
        if metric_path not in methods_by_metric:
            methods_by_metric[metric_path] = []
        if method and method not in methods_by_metric[metric_path]:
            methods_by_metric[metric_path].append(method)
    
    # Sort methods within each metric for determinism
    for metric_path in methods_by_metric:
        methods_by_metric[metric_path] = sorted(methods_by_metric[metric_path])
    
    # Find metrics with multiple methods
    metrics_with_multiple_methods = sorted([
        mp for mp, methods in methods_by_metric.items()
        if len(methods) > 1
    ])
    
    # Extract sample sizes
    baseline_sizes = [a.get("n_baseline", 0) for a in analyses]
    rfl_sizes = [a.get("n_rfl", 0) for a in analyses]
    
    # Sort methods_by_metric keys for deterministic output
    methods_by_metric_sorted = {k: methods_by_metric[k] for k in sorted(methods_by_metric.keys())}
    
    return {
        "analysis_count": len(analyses),
        "max_n_baseline": max(baseline_sizes) if baseline_sizes else 0,
        "max_n_rfl": max(rfl_sizes) if rfl_sizes else 0,
        "methods_by_metric": methods_by_metric_sorted,
        "metrics_with_multiple_methods": metrics_with_multiple_methods,
        "min_n_baseline": min(baseline_sizes) if baseline_sizes else 0,
        "min_n_rfl": min(rfl_sizes) if rfl_sizes else 0,
        "schema_version": pack.get("schema_version", EVIDENCE_PACK_SCHEMA_VERSION),
    }


def evaluate_evidence_readiness(
    quality_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether an evidence pack is ready for governance review.
    
    This evaluates EVIDENCE QUALITY, not effect sizes. It checks structural
    properties like sample sizes and analysis count.
    
    CONTRACT:
        - No uplift semantics
        - Weak points are neutral, descriptive strings
        - Status is about evidence quality, not effect direction
    
    Parameters
    ----------
    quality_snapshot : dict
        Quality snapshot as returned by build_evidence_quality_snapshot().
    
    Returns
    -------
    dict
        Readiness evaluation with:
        - ready_for_governance_review: bool
        - weak_points: list of descriptive strings
        - status: "OK" | "ATTENTION" | "WEAK"
    
    Examples
    --------
    >>> snapshot = build_evidence_quality_snapshot(pack)
    >>> readiness = evaluate_evidence_readiness(snapshot)
    >>> readiness["status"]
    'OK'
    """
    weak_points: List[str] = []
    
    analysis_count = quality_snapshot.get("analysis_count", 0)
    min_n_baseline = quality_snapshot.get("min_n_baseline", 0)
    min_n_rfl = quality_snapshot.get("min_n_rfl", 0)
    
    # Check for blocking conditions
    blocking = False
    
    if analysis_count < EVIDENCE_QUALITY_THRESHOLDS["min_analyses_for_review"]:
        weak_points.append("no analyses present in evidence pack")
        blocking = True
    
    if min_n_baseline < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_block"]:
        weak_points.append("some baseline sample sizes are critically low")
        blocking = True
    
    if min_n_rfl < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_block"]:
        weak_points.append("some RFL sample sizes are critically low")
        blocking = True
    
    # Check for weak conditions (not blocking, but attention needed)
    if min_n_baseline < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_weak"]:
        weak_points.append("some metrics have low baseline sample sizes")
    
    if min_n_rfl < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_weak"]:
        weak_points.append("some metrics have low RFL sample sizes")
    
    # Determine status
    if blocking:
        status = "WEAK"
        ready_for_governance_review = False
    elif weak_points:
        status = "ATTENTION"
        ready_for_governance_review = True
    else:
        status = "OK"
        ready_for_governance_review = True
    
    return {
        "ready_for_governance_review": ready_for_governance_review,
        "status": status,
        "weak_points": sorted(weak_points),  # Sorted for determinism
    }


def summarize_evidence_for_global_health(
    quality_snapshot: Dict[str, Any],
    readiness_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a summary for integration with Director Console's global health view.
    
    This is intended to be consumed by a global health dashboard. It provides
    a high-level signal about evidence quality across the pack.
    
    CONTRACT:
        - No uplift interpretation
        - Status is structural (OK / WARN / BLOCK)
        - Weak metric paths are identified by sample size, not effect size
    
    Parameters
    ----------
    quality_snapshot : dict
        Quality snapshot as returned by build_evidence_quality_snapshot().
        
    readiness_eval : dict
        Readiness evaluation as returned by evaluate_evidence_readiness().
    
    Returns
    -------
    dict
        Global health summary with:
        - evidence_ok: bool (True if ready for governance review)
        - analysis_count: number of analyses
        - weak_metric_paths: list of metric paths with quality issues
        - status: "OK" | "WARN" | "BLOCK"
    
    Examples
    --------
    >>> snapshot = build_evidence_quality_snapshot(pack)
    >>> readiness = evaluate_evidence_readiness(snapshot)
    >>> health = summarize_evidence_for_global_health(snapshot, readiness)
    >>> health["status"]
    'OK'
    """
    # Map readiness status to global health status
    readiness_status = readiness_eval.get("status", "WEAK")
    status_mapping = {
        "OK": "OK",
        "ATTENTION": "WARN",
        "WEAK": "BLOCK",
    }
    status = status_mapping.get(readiness_status, "BLOCK")
    
    # Identify weak metric paths based on sample sizes
    # A metric is weak if it has sample sizes below threshold
    weak_metric_paths: List[str] = []
    methods_by_metric = quality_snapshot.get("methods_by_metric", {})
    
    # We need to check the original pack for sample sizes per metric
    # But we only have the snapshot, so we flag based on overall min
    min_n_baseline = quality_snapshot.get("min_n_baseline", 0)
    min_n_rfl = quality_snapshot.get("min_n_rfl", 0)
    
    # If any sample size is below threshold, we can't identify which metric
    # from the snapshot alone. This is intentional: the snapshot is aggregate.
    # For now, we just indicate if there are weak points at all.
    if min_n_baseline < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_weak"] or \
       min_n_rfl < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_weak"]:
        # Mark all metrics as potentially weak (conservative)
        weak_metric_paths = sorted(methods_by_metric.keys())
    
    return {
        "analysis_count": quality_snapshot.get("analysis_count", 0),
        "evidence_ok": readiness_eval.get("ready_for_governance_review", False),
        "status": status,
        "weak_metric_paths": weak_metric_paths,
    }


# ===========================================================================
# PHASE IV — EVIDENCE QUALITY LADDER & RELEASE GOVERNANCE FEED (D3 v1.1)
# ===========================================================================

def classify_evidence_quality_level(
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify evidence quality into a tiered ladder (TIER_1, TIER_2, TIER_3).
    
    This function evaluates structural properties of the evidence pack:
    - Sample sizes
    - Number of metrics
    - Method diversity (multiple methods per metric)
    
    CONTRACT:
        - No uplift semantics or effect-size interpretation
        - Classification based on structural quality only
        - Requirements are descriptive, not interpretive
    
    Parameters
    ----------
    snapshot : dict
        Quality snapshot as returned by build_evidence_quality_snapshot().
    
    Returns
    -------
    dict
        Quality tier classification with:
        - quality_tier: "TIER_1" | "TIER_2" | "TIER_3"
        - requirements_met: list of requirement strings that were met
        - requirements_missing: list of requirement strings that were missing
    
    Tier Definitions:
        - TIER_1 (minimal): single method, low sample sizes, single metric
        - TIER_2 (standard): multiple metrics, adequate sample sizes, single method
        - TIER_3 (strong): multiple metrics, multiple methods per metric, strong sample sizes
    
    Examples
    --------
    >>> snapshot = build_evidence_quality_snapshot(pack)
    >>> tier_info = classify_evidence_quality_level(snapshot)
    >>> tier_info["quality_tier"]
    'TIER_2'
    """
    analysis_count = snapshot.get("analysis_count", 0)
    min_n_baseline = snapshot.get("min_n_baseline", 0)
    min_n_rfl = snapshot.get("min_n_rfl", 0)
    methods_by_metric = snapshot.get("methods_by_metric", {})
    metrics_with_multiple_methods = snapshot.get("metrics_with_multiple_methods", [])
    
    # Count unique metrics
    num_metrics = len(methods_by_metric)
    
    # Check requirements for each tier
    requirements_met: List[str] = []
    requirements_missing: List[str] = []
    
    # TIER_3 requirements (strongest)
    tier_3_requirements = {
        "multiple_metrics": num_metrics >= EVIDENCE_QUALITY_THRESHOLDS["tier_3_min_metrics"],
        "strong_sample_sizes": (
            min_n_baseline >= EVIDENCE_QUALITY_THRESHOLDS["tier_3_min_sample_size"] and
            min_n_rfl >= EVIDENCE_QUALITY_THRESHOLDS["tier_3_min_sample_size"]
        ),
        "multiple_methods_per_metric": len(metrics_with_multiple_methods) > 0,
    }
    
    # TIER_2 requirements
    tier_2_requirements = {
        "multiple_metrics": num_metrics >= EVIDENCE_QUALITY_THRESHOLDS["tier_2_min_metrics"],
        "adequate_sample_sizes": (
            min_n_baseline >= EVIDENCE_QUALITY_THRESHOLDS["tier_2_min_sample_size"] and
            min_n_rfl >= EVIDENCE_QUALITY_THRESHOLDS["tier_2_min_sample_size"]
        ),
    }
    
    # TIER_1 requirements (minimal - always achievable if we have at least one analysis)
    tier_1_requirements = {
        "at_least_one_analysis": analysis_count >= 1,
    }
    
    # Determine tier by checking from highest to lowest
    if all(tier_3_requirements.values()):
        quality_tier = "TIER_3"
        requirements_met = [
            f"multiple metrics ({num_metrics})",
            f"strong sample sizes (min: {min_n_baseline})",
            f"multiple methods per metric ({len(metrics_with_multiple_methods)} metrics)",
        ]
        requirements_missing = []
    elif all(tier_2_requirements.values()):
        quality_tier = "TIER_2"
        requirements_met = [
            f"multiple metrics ({num_metrics})",
            f"adequate sample sizes (min: {min_n_baseline})",
        ]
        requirements_missing = []
        if not tier_3_requirements["multiple_methods_per_metric"]:
            requirements_missing.append("multiple methods per metric")
        if not tier_3_requirements["strong_sample_sizes"]:
            requirements_missing.append("strong sample sizes")
    else:
        quality_tier = "TIER_1"
        requirements_met = [
            f"at least one analysis ({analysis_count})",
        ]
        requirements_missing = []
        if not tier_2_requirements["multiple_metrics"]:
            requirements_missing.append(f"multiple metrics (have {num_metrics}, need {EVIDENCE_QUALITY_THRESHOLDS['tier_2_min_metrics']})")
        if not tier_2_requirements["adequate_sample_sizes"]:
            requirements_missing.append(f"adequate sample sizes (have {min_n_baseline}, need {EVIDENCE_QUALITY_THRESHOLDS['tier_2_min_sample_size']})")
    
    return {
        "quality_tier": quality_tier,
        "requirements_met": sorted(requirements_met),  # Sorted for determinism
        "requirements_missing": sorted(requirements_missing),  # Sorted for determinism
    }


def evaluate_evidence_for_promotion(
    quality_snapshot: Dict[str, Any],
    readiness_eval: Dict[str, Any],
    adversarial_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether evidence is ready for promotion/release.
    
    This function combines evidence quality, readiness, and adversarial health
    (from D2) to determine promotion readiness.
    
    CONTRACT:
        - No uplift semantics
        - Blocking reasons are descriptive, not interpretive
        - Notes are neutral explanatory strings
    
    Parameters
    ----------
    quality_snapshot : dict
        Quality snapshot as returned by build_evidence_quality_snapshot().
        
    readiness_eval : dict
        Readiness evaluation as returned by evaluate_evidence_readiness().
        
    adversarial_health : dict
        Adversarial health from D2. Expected to have:
        - status: "OK" | "WARN" | "BLOCK" (or similar)
        - Any other fields D2 provides
    
    Returns
    -------
    dict
        Promotion evaluation with:
        - promotion_ok: bool
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: list of descriptive strings
        - notes: list of neutral explanatory strings
    
    Examples
    --------
    >>> snapshot = build_evidence_quality_snapshot(pack)
    >>> readiness = evaluate_evidence_readiness(snapshot)
    >>> adversarial = {"status": "OK"}
    >>> promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
    >>> promotion["promotion_ok"]
    True
    """
    blocking_reasons: List[str] = []
    notes: List[str] = []
    
    # Check evidence readiness
    if not readiness_eval.get("ready_for_governance_review", False):
        blocking_reasons.append("evidence not ready for governance review")
    
    readiness_status = readiness_eval.get("status", "WEAK")
    if readiness_status == "WEAK":
        blocking_reasons.append("evidence quality is weak")
        notes.append("evidence pack has structural quality issues")
    elif readiness_status == "ATTENTION":
        notes.append("evidence pack requires attention but is reviewable")
    
    # Check adversarial health (from D2)
    adversarial_status = adversarial_health.get("status", "BLOCK")
    if adversarial_status == "BLOCK":
        blocking_reasons.append("adversarial health check failed")
    elif adversarial_status == "WARN":
        notes.append("adversarial health shows warnings")
    
    # Check sample sizes
    min_n_baseline = quality_snapshot.get("min_n_baseline", 0)
    min_n_rfl = quality_snapshot.get("min_n_rfl", 0)
    if min_n_baseline < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_block"]:
        blocking_reasons.append("baseline sample sizes are critically low")
    if min_n_rfl < EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_block"]:
        blocking_reasons.append("RFL sample sizes are critically low")
    
    # Determine promotion status
    if blocking_reasons:
        promotion_ok = False
        status = "BLOCK"
    elif readiness_status == "ATTENTION" or adversarial_status == "WARN":
        promotion_ok = True
        status = "WARN"
        if not notes:
            notes.append("evidence is reviewable with caution")
    else:
        promotion_ok = True
        status = "OK"
        if not notes:
            notes.append("evidence meets promotion requirements")
    
    return {
        "blocking_reasons": sorted(blocking_reasons),  # Sorted for determinism
        "notes": sorted(notes),  # Sorted for determinism
        "promotion_ok": promotion_ok,
        "status": status,
    }


def build_evidence_director_panel(
    quality_tier_info: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a single evidence tile/panel for the Director Console.
    
    This provides a high-level, executive summary of evidence quality and
    promotion readiness in a format suitable for dashboard display.
    
    CONTRACT:
        - No uplift interpretation
        - Headline is neutral and descriptive
        - Status light is based on structural quality, not effect size
    
    Parameters
    ----------
    quality_tier_info : dict
        Quality tier classification as returned by classify_evidence_quality_level().
        
    promotion_eval : dict
        Promotion evaluation as returned by evaluate_evidence_for_promotion().
    
    Returns
    -------
    dict
        Director panel with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - quality_tier: "TIER_1" | "TIER_2" | "TIER_3"
        - analysis_count: number of analyses (from promotion_eval context if available)
        - evidence_ok: bool (from promotion_eval)
        - headline: neutral sentence summarizing evidence state
    
    Examples
    --------
    >>> tier_info = classify_evidence_quality_level(snapshot)
    >>> promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
    >>> panel = build_evidence_director_panel(tier_info, promotion)
    >>> panel["status_light"]
    'GREEN'
    """
    quality_tier = quality_tier_info.get("quality_tier", "TIER_1")
    promotion_ok = promotion_eval.get("promotion_ok", False)
    promotion_status = promotion_eval.get("status", "BLOCK")
    blocking_reasons = promotion_eval.get("blocking_reasons", [])
    requirements_met = quality_tier_info.get("requirements_met", [])
    requirements_missing = quality_tier_info.get("requirements_missing", [])
    
    # Determine status light based on promotion status
    if promotion_status == "OK" and promotion_ok:
        status_light = "GREEN"
    elif promotion_status == "WARN" and promotion_ok:
        status_light = "YELLOW"
    else:
        status_light = "RED"
    
    # Build headline (neutral, descriptive)
    if blocking_reasons:
        headline = f"Evidence pack has {len(blocking_reasons)} blocking issue(s) preventing promotion"
    elif promotion_status == "WARN":
        headline = f"Evidence pack is reviewable with {quality_tier} quality; caution advised"
    elif quality_tier == "TIER_3":
        headline = f"Evidence pack meets {quality_tier} quality standards and is ready for promotion"
    elif quality_tier == "TIER_2":
        headline = f"Evidence pack meets {quality_tier} quality standards and is ready for review"
    else:
        headline = f"Evidence pack has {quality_tier} quality; meets minimal requirements"
    
    # Get analysis count from context if available
    # (We don't have direct access, but promotion_eval might have it in context)
    # For now, we'll note that it should be provided separately or extracted
    # from the original snapshot/readiness_eval if needed
    
    return {
        "analysis_count": promotion_eval.get("analysis_count", 0),  # May be 0 if not provided
        "evidence_ok": promotion_ok,
        "headline": headline,
        "quality_tier": quality_tier,
        "status_light": status_light,
    }


# ===========================================================================
# PHASE V — EVIDENCE EVOLUTION TIMELINE & REGRESSION WATCHDOG (D3 v1.1)
# ===========================================================================

def build_evidence_quality_timeline(
    snapshots: "Sequence[Dict[str, Any]]",
) -> Dict[str, Any]:
    """
    Build a timeline of evidence quality over multiple runs.
    
    This function tracks quality tier transitions and identifies trends
    in evidence quality across time.
    
    CONTRACT:
        - No uplift semantics
        - Timeline entries are sorted by run_id for determinism
        - Quality trend is based on tier transitions, not effect sizes
    
    Parameters
    ----------
    snapshots : Sequence[Dict[str, Any]]
        Sequence of quality snapshots, each expected to have:
        - run_id: str (identifier for the run)
        - quality_tier: "TIER_1" | "TIER_2" | "TIER_3" (from classify_evidence_quality_level)
        - analysis_count: int (from quality snapshot)
    
    Returns
    -------
    dict
        Quality timeline with:
        - schema_version: "1.0.0"
        - timeline: list of dicts with run_id, quality_tier, analysis_count
        - tier_transition_counts: dict mapping transition types to counts
        - quality_trend: "IMPROVING" | "STABLE" | "DEGRADING"
    
    Examples
    --------
    >>> snapshots = [
    ...     {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
    ...     {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
    ...     {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
    ... ]
    >>> timeline = build_evidence_quality_timeline(snapshots)
    >>> timeline["quality_trend"]
    'IMPROVING'
    """
    if not snapshots:
        return {
            "quality_trend": "STABLE",
            "schema_version": "1.0.0",
            "tier_transition_counts": {},
            "timeline": [],
        }
    
    # Sort by run_id for deterministic ordering
    sorted_snapshots = sorted(snapshots, key=lambda s: s.get("run_id", ""))
    
    # Build timeline entries
    timeline = []
    for snapshot in sorted_snapshots:
        timeline.append({
            "analysis_count": snapshot.get("analysis_count", 0),
            "quality_tier": snapshot.get("quality_tier", "TIER_1"),
            "run_id": snapshot.get("run_id", ""),
        })
    
    # Count tier transitions
    tier_transition_counts: Dict[str, int] = {}
    improving_count = 0
    degrading_count = 0
    stable_count = 0
    
    # Tier ordering for comparison (higher number = better)
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    
    for i in range(len(timeline) - 1):
        current_tier = timeline[i]["quality_tier"]
        next_tier = timeline[i + 1]["quality_tier"]
        
        current_order = tier_order.get(current_tier, 0)
        next_order = tier_order.get(next_tier, 0)
        
        if next_order > current_order:
            transition = f"{current_tier}→{next_tier}"
            tier_transition_counts[transition] = tier_transition_counts.get(transition, 0) + 1
            improving_count += 1
        elif next_order < current_order:
            transition = f"{current_tier}→{next_tier}"
            tier_transition_counts[transition] = tier_transition_counts.get(transition, 0) + 1
            degrading_count += 1
        else:
            stable_count += 1
    
    # Determine quality trend
    total_transitions = improving_count + degrading_count + stable_count
    if total_transitions == 0:
        quality_trend = "STABLE"
    else:
        improving_ratio = improving_count / total_transitions
        degrading_ratio = degrading_count / total_transitions
        
        # If improving transitions are more common, trend is IMPROVING
        # If degrading transitions are more common, trend is DEGRADING
        # Otherwise, STABLE
        if improving_ratio > 0.5:
            quality_trend = "IMPROVING"
        elif degrading_ratio > 0.5:
            quality_trend = "DEGRADING"
        else:
            quality_trend = "STABLE"
    
    # Sort transition counts for determinism
    tier_transition_counts_sorted = {k: tier_transition_counts[k] for k in sorted(tier_transition_counts.keys())}
    
    return {
        "quality_trend": quality_trend,
        "schema_version": "1.0.0",
        "tier_transition_counts": tier_transition_counts_sorted,
        "timeline": timeline,
    }


def evaluate_evidence_quality_regression(
    quality_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate evidence quality timeline for regressions.
    
    This function detects when evidence quality degrades over time,
    particularly focusing on drops from higher tiers to lower tiers.
    
    CONTRACT:
        - No uplift semantics
        - Regression detection is based on tier transitions, not effect sizes
        - Reasons are neutral and descriptive
    
    Parameters
    ----------
    quality_timeline : dict
        Quality timeline as returned by build_evidence_quality_timeline().
    
    Returns
    -------
    dict
        Regression evaluation with:
        - regression_detected: bool
        - status: "OK" | "ATTENTION" | "BLOCK"
        - neutral_reasons: list of descriptive strings
    
    Examples
    --------
    >>> timeline = build_evidence_quality_timeline(snapshots)
    >>> regression = evaluate_evidence_quality_regression(timeline)
    >>> regression["regression_detected"]
    False
    """
    neutral_reasons: List[str] = []
    timeline = quality_timeline.get("timeline", [])
    tier_transition_counts = quality_timeline.get("tier_transition_counts", {})
    quality_trend = quality_timeline.get("quality_trend", "STABLE")
    
    if not timeline:
        return {
            "neutral_reasons": ["no timeline data available"],
            "regression_detected": False,
            "status": "OK",
        }
    
    # Check for significant tier drops (TIER_3 → TIER_1 or TIER_3 → TIER_2)
    tier_3_to_lower = (
        tier_transition_counts.get("TIER_3→TIER_1", 0) +
        tier_transition_counts.get("TIER_3→TIER_2", 0)
    )
    
    # Check for repeated degradation patterns
    # Tier ordering for comparison (higher number = better)
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    degrading_transitions = 0
    for transition, count in tier_transition_counts.items():
        if "→" in transition:
            parts = transition.split("→")
            if len(parts) == 2:
                from_tier = parts[0]
                to_tier = parts[1]
                from_order = tier_order.get(from_tier, 0)
                to_order = tier_order.get(to_tier, 0)
                if to_order < from_order:
                    degrading_transitions += count
    
    # Check recent runs (last 3) for quality drops
    recent_runs = timeline[-3:] if len(timeline) >= 3 else timeline
    recent_tiers = [run.get("quality_tier", "TIER_1") for run in recent_runs]
    
    # Count how many recent runs are below TIER_3
    recent_below_tier_3 = sum(1 for tier in recent_tiers if tier != "TIER_3")
    
    # Determine regression status
    regression_detected = False
    
    # Check for improving transitions (recovery patterns)
    improving_transitions = 0
    for transition, count in tier_transition_counts.items():
        if "→" in transition:
            parts = transition.split("→")
            if len(parts) == 2:
                from_tier = parts[0]
                to_tier = parts[1]
                from_order = tier_order.get(from_tier, 0)
                to_order = tier_order.get(to_tier, 0)
                if to_order > from_order:
                    improving_transitions += count
    
    # Check for multiple degrading transitions first (alternating patterns are less severe)
    # Block if: multiple TIER_3 → lower transitions (>= 2) OR recent runs consistently below TIER_3
    # But if there are improving transitions (recovery), treat alternating patterns as ATTENTION
    if degrading_transitions >= 3 and improving_transitions > 0:
        # Multiple degrading transitions with recovery (alternating pattern)
        regression_detected = True
        neutral_reasons.append(f"multiple quality degrading transitions detected ({degrading_transitions} occurrences)")
        status = "ATTENTION"
    elif recent_below_tier_3 == len(recent_runs) and len(recent_runs) >= 2:
        # All recent runs are below TIER_3 (check before TIER_3 drop checks)
        regression_detected = True
        neutral_reasons.append(f"recent runs show consistent quality below TIER_3 ({recent_below_tier_3}/{len(recent_runs)} runs)")
        status = "BLOCK"
    elif tier_3_to_lower >= 2:
        # Multiple TIER_3 drops (consecutive or concerning pattern)
        regression_detected = True
        neutral_reasons.append(f"multiple quality drops from TIER_3 to lower tiers ({tier_3_to_lower} occurrences)")
        status = "BLOCK"
    elif tier_3_to_lower >= 1:
        # Single TIER_3 drop
        regression_detected = True
        neutral_reasons.append("quality drop from TIER_3 detected")
        status = "ATTENTION"
    elif degrading_transitions >= 3:
        # Multiple degrading transitions (no recovery pattern)
        regression_detected = True
        neutral_reasons.append(f"multiple quality degrading transitions detected ({degrading_transitions} occurrences)")
        status = "ATTENTION"
    elif quality_trend == "DEGRADING" and degrading_transitions >= 1:
        # Degrading trend with at least one degrading transition
        regression_detected = True
        neutral_reasons.append("evidence quality trend is degrading")
        status = "ATTENTION"
    else:
        status = "OK"
    
    return {
        "neutral_reasons": sorted(neutral_reasons),  # Sorted for determinism
        "regression_detected": regression_detected,
        "status": status,
    }


# ===========================================================================
# PHASE VI — EVIDENCE QUALITY PHASE-PORTRAIT ENGINE (D3 v1.1)
# ===========================================================================

def build_evidence_phase_portrait(
    quality_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a geometric phase-portrait of evidence quality evolution.
    
    This function creates a phase-space representation of quality tiers over time,
    enabling trajectory analysis and pattern recognition.
    
    CONTRACT:
        - No uplift semantics
        - Phase points are geometric coordinates [run_index, tier_value]
        - Trajectory classes are structural, not interpretive
    
    Parameters
    ----------
    quality_timeline : dict
        Quality timeline as returned by build_evidence_quality_timeline().
    
    Returns
    -------
    dict
        Phase portrait with:
        - phase_points: list of [run_index, tier_value] pairs
        - trajectory_class: "IMPROVING" | "STABLE" | "OSCILLATING" | "DEGRADING"
        - neutral_notes: list of descriptive strings
    
    Examples
    --------
    >>> timeline = build_evidence_quality_timeline(snapshots)
    >>> portrait = build_evidence_phase_portrait(timeline)
    >>> portrait["trajectory_class"]
    'IMPROVING'
    """
    timeline = quality_timeline.get("timeline", [])
    tier_transition_counts = quality_timeline.get("tier_transition_counts", {})
    quality_trend = quality_timeline.get("quality_trend", "STABLE")
    
    if not timeline:
        return {
            "neutral_notes": ["no timeline data available for phase portrait"],
            "phase_points": [],
            "trajectory_class": "STABLE",
        }
    
    # Tier value mapping for phase-space coordinates
    tier_values = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    
    # Build phase points: [run_index, tier_value]
    phase_points: List[List[int]] = []
    for idx, run in enumerate(timeline):
        tier = run.get("quality_tier", "TIER_1")
        tier_value = tier_values.get(tier, 1)
        phase_points.append([idx, tier_value])
    
    # Classify trajectory
    neutral_notes: List[str] = []
    
    # Check for oscillating pattern (alternating improve/degrade cycles)
    improving_count = 0
    degrading_count = 0
    
    for transition, count in tier_transition_counts.items():
        if "→" in transition:
            parts = transition.split("→")
            if len(parts) == 2:
                from_tier = parts[0]
                to_tier = parts[1]
                from_val = tier_values.get(from_tier, 0)
                to_val = tier_values.get(to_tier, 0)
                if to_val > from_val:
                    improving_count += count
                elif to_val < from_val:
                    degrading_count += count
    
    # Oscillating: both improving and degrading transitions present
    if improving_count > 0 and degrading_count > 0:
        # Check if pattern is truly oscillating (not just noise)
        if improving_count >= 2 and degrading_count >= 2:
            trajectory_class = "OSCILLATING"
            neutral_notes.append(f"evidence quality shows alternating improve/degrade cycles ({improving_count} improving, {degrading_count} degrading transitions)")
        elif quality_trend == "DEGRADING":
            trajectory_class = "DEGRADING"
            neutral_notes.append("evidence quality trend is degrading despite some recovery attempts")
        else:
            trajectory_class = quality_trend
    else:
        # Use quality trend from timeline
        trajectory_class = quality_trend
        if trajectory_class == "IMPROVING":
            neutral_notes.append("evidence quality shows consistent improvement over time")
        elif trajectory_class == "DEGRADING":
            neutral_notes.append("evidence quality shows consistent degradation over time")
        else:
            neutral_notes.append("evidence quality remains stable over time")
    
    return {
        "neutral_notes": sorted(neutral_notes),  # Sorted for determinism
        "phase_points": phase_points,
        "trajectory_class": trajectory_class,
    }


def forecast_evidence_envelope(
    phase_portrait: Dict[str, Any],
    regression_watchdog: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Forecast future evidence quality envelope based on phase portrait and regression patterns.
    
    This function predicts the likely quality band and estimates risk cycles,
    providing forward-looking guidance for evidence quality management.
    
    CONTRACT:
        - No uplift semantics
        - Predictions are structural/statistical, not interpretive
        - Confidence is a numeric measure, not a judgment
    
    Parameters
    ----------
    phase_portrait : dict
        Phase portrait as returned by build_evidence_phase_portrait().
        
    regression_watchdog : dict
        Regression evaluation as returned by evaluate_evidence_quality_regression().
    
    Returns
    -------
    dict
        Forecast with:
        - predicted_band: "LOW" | "MEDIUM" | "HIGH"
        - confidence: float (0.0 to 1.0)
        - cycles_until_risk: int (estimated cycles until quality risk)
        - neutral_explanation: list of descriptive strings
    
    Examples
    --------
    >>> portrait = build_evidence_phase_portrait(timeline)
    >>> regression = evaluate_evidence_quality_regression(timeline)
    >>> forecast = forecast_evidence_envelope(portrait, regression)
    >>> forecast["predicted_band"]
    'HIGH'
    """
    phase_points = phase_portrait.get("phase_points", [])
    trajectory_class = phase_portrait.get("trajectory_class", "STABLE")
    regression_status = regression_watchdog.get("status", "OK")
    regression_detected = regression_watchdog.get("regression_detected", False)
    
    neutral_explanation: List[str] = []
    
    if not phase_points:
        return {
            "confidence": 0.0,
            "cycles_until_risk": 0,
            "neutral_explanation": ["insufficient data for envelope prediction"],
            "predicted_band": "LOW",
        }
    
    # Extract recent tier values for trend analysis
    recent_points = phase_points[-3:] if len(phase_points) >= 3 else phase_points
    recent_tier_values = [point[1] for point in recent_points]
    avg_recent_tier = sum(recent_tier_values) / len(recent_tier_values) if recent_tier_values else 1.0
    
    # Predict band based on trajectory and current state
    if regression_status == "BLOCK":
        predicted_band = "LOW"
        confidence = 0.8
        cycles_until_risk = 0
        neutral_explanation.append("regression watchdog indicates blocking conditions")
    elif regression_status == "ATTENTION":
        if trajectory_class == "DEGRADING":
            predicted_band = "LOW"
            confidence = 0.7
            cycles_until_risk = 1
            neutral_explanation.append("degrading trajectory with attention status suggests low quality band")
        elif trajectory_class == "OSCILLATING":
            predicted_band = "MEDIUM"
            confidence = 0.6
            cycles_until_risk = 2
            neutral_explanation.append("oscillating pattern with attention status suggests medium quality band")
        else:
            predicted_band = "MEDIUM"
            confidence = 0.65
            cycles_until_risk = 2
            neutral_explanation.append("attention status with stable/improving trajectory suggests medium quality band")
    else:
        # OK status
        if trajectory_class == "IMPROVING":
            if avg_recent_tier >= 2.5:
                predicted_band = "HIGH"
                confidence = 0.75
                cycles_until_risk = 5
                neutral_explanation.append("improving trajectory with high recent quality suggests high quality band")
            else:
                predicted_band = "MEDIUM"
                confidence = 0.7
                cycles_until_risk = 3
                neutral_explanation.append("improving trajectory with moderate recent quality suggests medium quality band")
        elif trajectory_class == "STABLE":
            if avg_recent_tier >= 2.5:
                predicted_band = "HIGH"
                confidence = 0.8
                cycles_until_risk = 4
                neutral_explanation.append("stable trajectory with high recent quality suggests high quality band")
            else:
                predicted_band = "MEDIUM"
                confidence = 0.75
                cycles_until_risk = 2
                neutral_explanation.append("stable trajectory with moderate recent quality suggests medium quality band")
        elif trajectory_class == "OSCILLATING":
            predicted_band = "MEDIUM"
            confidence = 0.6
            cycles_until_risk = 2
            neutral_explanation.append("oscillating pattern suggests medium quality band with variable risk")
        else:  # DEGRADING but OK status (early stage)
            predicted_band = "MEDIUM"
            confidence = 0.65
            cycles_until_risk = 1
            neutral_explanation.append("degrading trajectory with OK status suggests medium quality band")
    
    # Adjust confidence based on data points
    if len(phase_points) < 3:
        confidence *= 0.7  # Lower confidence with few data points
        neutral_explanation.append("limited data points reduce prediction confidence")
    
    return {
        "confidence": round(confidence, 2),  # Round to 2 decimal places for determinism
        "cycles_until_risk": cycles_until_risk,
        "neutral_explanation": sorted(neutral_explanation),  # Sorted for determinism
        "predicted_band": predicted_band,
    }


def build_evidence_director_panel_v2(
    quality_tier_info: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    phase_portrait: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    regression_watchdog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build enhanced evidence tile/panel for Director Console with phase-portrait integration.
    
    This is an enhanced version that incorporates phase-portrait analysis and envelope forecasting.
    
    CONTRACT:
        - No uplift interpretation
        - Headline is neutral and descriptive
        - Status light is based on structural quality, not effect size
    
    Parameters
    ----------
    quality_tier_info : dict
        Quality tier classification as returned by classify_evidence_quality_level().
        
    promotion_eval : dict
        Promotion evaluation as returned by evaluate_evidence_for_promotion().
        
    phase_portrait : dict, optional
        Phase portrait as returned by build_evidence_phase_portrait().
        
    forecast : dict, optional
        Forecast as returned by forecast_evidence_envelope().
        
    regression_watchdog : dict, optional
        Regression evaluation as returned by evaluate_evidence_quality_regression().
    
    Returns
    -------
    dict
        Enhanced director panel with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - quality_tier: "TIER_1" | "TIER_2" | "TIER_3"
        - trajectory_class: trajectory from phase portrait
        - regression_status: status from regression watchdog
        - analysis_count: number of analyses
        - evidence_ok: bool
        - headline: neutral sentence summarizing evidence state
        - flags: list of notable conditions
    
    Examples
    --------
    >>> tier_info = classify_evidence_quality_level(snapshot)
    >>> promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
    >>> portrait = build_evidence_phase_portrait(timeline)
    >>> forecast = forecast_evidence_envelope(portrait, regression)
    >>> panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
    >>> panel["status_light"]
    'GREEN'
    """
    quality_tier = quality_tier_info.get("quality_tier", "TIER_1")
    promotion_ok = promotion_eval.get("promotion_ok", False)
    promotion_status = promotion_eval.get("status", "BLOCK")
    blocking_reasons = promotion_eval.get("blocking_reasons", [])
    
    # Extract phase portrait data
    trajectory_class = phase_portrait.get("trajectory_class", "STABLE") if phase_portrait else "STABLE"
    
    # Extract regression watchdog data
    regression_status = regression_watchdog.get("status", "OK") if regression_watchdog else "OK"
    regression_detected = regression_watchdog.get("regression_detected", False) if regression_watchdog else False
    
    # Extract forecast data
    predicted_band = forecast.get("predicted_band", "MEDIUM") if forecast else "MEDIUM"
    cycles_until_risk = forecast.get("cycles_until_risk", 0) if forecast else 0
    
    # Determine status light based on promotion status and regression
    if promotion_status == "OK" and promotion_ok and not regression_detected:
        status_light = "GREEN"
    elif promotion_status == "WARN" or regression_status == "ATTENTION":
        status_light = "YELLOW"
    else:
        status_light = "RED"
    
    # Build flags list
    flags: List[str] = []
    if regression_detected:
        flags.append("regression_detected")
    if trajectory_class == "OSCILLATING":
        flags.append("oscillating_trajectory")
    if predicted_band == "LOW":
        flags.append("low_predicted_band")
    if cycles_until_risk <= 1:
        flags.append("imminent_risk")
    
    # Build headline (neutral, descriptive)
    if blocking_reasons:
        headline = f"Evidence pack has {len(blocking_reasons)} blocking issue(s) preventing promotion"
    elif regression_detected:
        headline = f"Evidence pack shows {trajectory_class.lower()} trajectory with regression detected"
    elif promotion_status == "WARN":
        headline = f"Evidence pack is reviewable with {quality_tier} quality; {trajectory_class.lower()} trajectory"
    elif quality_tier == "TIER_3" and trajectory_class == "IMPROVING":
        headline = f"Evidence pack meets {quality_tier} quality standards with improving trajectory"
    elif quality_tier == "TIER_3":
        headline = f"Evidence pack meets {quality_tier} quality standards and is ready for promotion"
    elif trajectory_class == "DEGRADING":
        headline = f"Evidence pack has {quality_tier} quality with degrading trajectory; monitor closely"
    else:
        headline = f"Evidence pack has {quality_tier} quality with {trajectory_class.lower()} trajectory"
    
    return {
        "analysis_count": promotion_eval.get("analysis_count", 0),
        "evidence_ok": promotion_ok,
        "flags": sorted(flags),  # Sorted for determinism
        "headline": headline,
        "quality_tier": quality_tier,
        "regression_status": regression_status,
        "status_light": status_light,
        "trajectory_class": trajectory_class,
    }


def compute_paired_delta_from_logs(
    baseline_path: str,
    rfl_path: str,
    metric_path: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> PairedDeltaResult:
    """
    Compute paired bootstrap delta from experiment log files.
    
    This function loads baseline and RFL logs, extracts the specified metric,
    and computes a bootstrap confidence interval for the mean difference.
    
    Uses the deterministic paired bootstrap engine from statistical.bootstrap.
    
    Parameters
    ----------
    baseline_path : str
        Path to baseline experiment JSONL log.
        
    rfl_path : str
        Path to RFL experiment JSONL log.
        
    metric_path : str
        Dot-notation path to the metric field.
        Examples: "success", "derivation.verified", "timing.duration_ms"
        
    n_bootstrap : int, default=10000
        Number of bootstrap resamples. Must be in [1000, 100000].
        
    seed : int, default=42
        Random seed for deterministic results.
    
    Returns
    -------
    PairedDeltaResult
        Result containing delta, CI bounds, and metadata.
    
    Raises
    ------
    ValueError
        If insufficient data or invalid parameters.
    FileNotFoundError
        If log files don't exist.
    
    Examples
    --------
    >>> result = compute_paired_delta_from_logs(
    ...     "baseline.jsonl",
    ...     "rfl.jsonl",
    ...     "success",
    ...     n_bootstrap=10000,
    ...     seed=42
    ... )
    >>> print(f"Delta: {result.delta:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    Notes
    -----
    When baseline and RFL have different numbers of observations, the function
    truncates to the minimum length to maintain paired structure. Data is sorted
    before pairing to ensure determinism.
    """
    # Import bootstrap engine (late import to avoid circular dependencies)
    from statistical.bootstrap import paired_bootstrap_delta
    
    # Load logs
    baseline_records = []
    with open(baseline_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                baseline_records.append(json.loads(stripped))
    
    rfl_records = []
    with open(rfl_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rfl_records.append(json.loads(stripped))
    
    # Extract metric values
    baseline_values = _extract_metric_from_records(baseline_records, metric_path)
    rfl_values = _extract_metric_from_records(rfl_records, metric_path)
    
    n_baseline = len(baseline_values)
    n_rfl = len(rfl_values)
    
    if n_baseline < 2:
        raise ValueError(f"Insufficient baseline data: {n_baseline} values (need >= 2)")
    
    if n_rfl < 2:
        raise ValueError(f"Insufficient RFL data: {n_rfl} values (need >= 2)")
    
    # Convert to numpy arrays and sort for determinism
    baseline_arr = np.array(sorted(baseline_values), dtype=np.float64)
    rfl_arr = np.array(sorted(rfl_values), dtype=np.float64)
    
    # Truncate to minimum length for pairing
    if len(baseline_arr) != len(rfl_arr):
        min_n = min(len(baseline_arr), len(rfl_arr))
        baseline_arr = baseline_arr[:min_n]
        rfl_arr = rfl_arr[:min_n]
    
    # Compute analysis_id for traceability
    analysis_id = compute_analysis_id(
        baseline_path=baseline_path,
        rfl_path=rfl_path,
        metric_path=metric_path,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    
    # Compute bootstrap CI
    result = paired_bootstrap_delta(
        baseline_arr,
        rfl_arr,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    
    return PairedDeltaResult(
        delta=result.delta_mean,
        ci_lower=result.CI_low,
        ci_upper=result.CI_high,
        n_baseline=n_baseline,
        n_rfl=n_rfl,
        metric_path=metric_path,
        seed=seed,
        n_bootstrap=n_bootstrap,
        method=result.method,
        analysis_id=analysis_id,
    )


# ===========================================================================
# SLICE-SPECIFIC SUCCESS METRICS
# ===========================================================================

SLICE_SUCCESS_CRITERIA = {
    "prop_depth4": {
        "min_success_rate": 0.95,
        "max_abstention_rate": 0.02,
        "min_throughput_uplift_pct": 5.0,
        "min_samples": 500,
    },
    "fol_eq_group": {
        "min_success_rate": 0.85,
        "max_abstention_rate": 0.10,
        "min_throughput_uplift_pct": 3.0,
        "min_samples": 300,
    },
    "fol_eq_ring": {
        "min_success_rate": 0.80,
        "max_abstention_rate": 0.15,
        "min_throughput_uplift_pct": 2.0,
        "min_samples": 300,
    },
    "linear_arith": {
        "min_success_rate": 0.70,
        "max_abstention_rate": 0.20,
        "min_throughput_uplift_pct": 0.0,
        "min_samples": 200,
    },
}


# ===========================================================================
# BUDGET ADMISSIBILITY INTEGRATION (PHASE II)
# ===========================================================================

def compute_budget_comparison_from_logs(
    baseline_log_path: str,
    rfl_log_path: str,
    effect_magnitude: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute budget admissibility classification from experiment log files.

    This function loads baseline and RFL logs, extracts budget-related fields,
    computes BudgetStats for each condition, and classifies admissibility.

    STATUS: PHASE II — NOT RUN IN PHASE I

    Parameters
    ----------
    baseline_log_path : str
        Path to baseline experiment JSONL log.

    rfl_log_path : str
        Path to RFL experiment JSONL log.

    effect_magnitude : float, optional
        Optional |Δp| for region refinement.
        If None, assumes small effect (conservative classification).

    Returns
    -------
    dict
        Dictionary containing:
        - classification: "SAFE" | "SUSPICIOUS" | "INVALID"
        - baseline: dict with baseline budget metrics
        - rfl: dict with RFL budget metrics
        - asymmetry: dict with symmetry analysis
        - admissibility: full BudgetAdmissibilityResult as dict

    Raises
    ------
    FileNotFoundError
        If log files don't exist.

    Notes
    -----
    Expected log record fields for budget extraction:
    - outcome: "V" | "R" | "A_timeout" | "A_crash" | "S_budget"
    - cycle_id: identifier for grouping by cycle
    - candidates_attempted: int (per cycle)
    - candidates_skipped: int (per cycle)

    If budget fields are missing, conservative defaults are used.
    """
    from backend.metrics.budget_admissibility import (
        BudgetStats,
        classify_budget_admissibility,
    )

    def load_records(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def extract_budget_stats(records: List[Dict[str, Any]]) -> BudgetStats:
        """Extract BudgetStats from a list of log records."""
        total_candidates = 0
        total_skipped = 0
        exhausted_cycles = 0
        complete_skip_cycles = 0
        cycle_coverages = []

        # Group records by cycle_id if available
        cycles: Dict[Any, List[Dict[str, Any]]] = {}
        for r in records:
            cycle_id = r.get("cycle_id", 0)
            if cycle_id not in cycles:
                cycles[cycle_id] = []
            cycles[cycle_id].append(r)

        for cycle_id, cycle_records in cycles.items():
            cycle_candidates = 0
            cycle_skipped = 0
            cycle_observed = 0  # V + R + A

            for r in cycle_records:
                # Check for per-record outcome
                outcome = r.get("outcome", "")

                if outcome == "S_budget":
                    cycle_skipped += 1
                    cycle_candidates += 1
                elif outcome in ("V", "R", "A_timeout", "A_crash"):
                    cycle_observed += 1
                    cycle_candidates += 1
                else:
                    # Legacy format: check candidates_attempted/skipped
                    cycle_candidates += r.get("candidates_attempted", 1)
                    cycle_skipped += r.get("candidates_skipped", 0)
                    # Estimate observed from success/failure fields
                    if r.get("success") is not None or r.get("verified") is not None:
                        cycle_observed += 1

            total_candidates += cycle_candidates
            total_skipped += cycle_skipped

            # Check if budget was exhausted (any skips in cycle)
            if cycle_skipped > 0:
                exhausted_cycles += 1

            # Check for complete-skip cycle
            if cycle_candidates > 0 and cycle_skipped == cycle_candidates:
                complete_skip_cycles += 1

            # Compute cycle coverage: (V + R + A) / N
            if cycle_candidates > 0:
                # If we don't have explicit observed count, estimate from non-skipped
                if cycle_observed == 0:
                    cycle_observed = cycle_candidates - cycle_skipped
                coverage = cycle_observed / cycle_candidates
                cycle_coverages.append(coverage)

        total_cycles = len(cycles) if cycles else 1
        min_coverage = min(cycle_coverages) if cycle_coverages else 1.0

        # Detect skip trend (simplified: check if skips increase monotonically)
        skip_trend = False
        if len(cycles) >= 3:
            cycle_skip_rates = []
            for cycle_id in sorted(cycles.keys()):
                cycle_records = cycles[cycle_id]
                n = len(cycle_records)
                skipped = sum(1 for r in cycle_records if r.get("outcome") == "S_budget")
                if n > 0:
                    cycle_skip_rates.append(skipped / n)

            if len(cycle_skip_rates) >= 3:
                # Simple monotonicity check
                increasing = all(
                    cycle_skip_rates[i] <= cycle_skip_rates[i + 1]
                    for i in range(len(cycle_skip_rates) - 1)
                )
                decreasing = all(
                    cycle_skip_rates[i] >= cycle_skip_rates[i + 1]
                    for i in range(len(cycle_skip_rates) - 1)
                )
                # Only flag if strictly monotonic and significant range
                if increasing or decreasing:
                    rate_range = max(cycle_skip_rates) - min(cycle_skip_rates)
                    if rate_range > 0.10:
                        skip_trend = True

        return BudgetStats(
            total_candidates=total_candidates,
            total_skipped=total_skipped,
            exhausted_cycles=exhausted_cycles,
            complete_skip_cycles=complete_skip_cycles,
            total_cycles=total_cycles,
            min_cycle_coverage=min_coverage,
            skip_trend_significant=skip_trend,
        )

    # Load logs
    baseline_records = load_records(baseline_log_path)
    rfl_records = load_records(rfl_log_path)

    # Extract stats
    baseline_stats = extract_budget_stats(baseline_records)
    rfl_stats = extract_budget_stats(rfl_records)

    # Classify admissibility
    result = classify_budget_admissibility(
        baseline_stats,
        rfl_stats,
        effect_magnitude=effect_magnitude,
    )

    # Build summary dict matching BUDGET_ADMISSIBILITY_SPEC.md §5.2
    return {
        "classification": result.classification.value,
        "baseline": {
            "exhaustion_rate": result.baseline_exhaustion_rate,
            "exhausted_cycles": baseline_stats.exhausted_cycles,
            "mean_cycle_coverage": baseline_stats.min_cycle_coverage,  # approx
        },
        "rfl": {
            "exhaustion_rate": result.rfl_exhaustion_rate,
            "exhausted_cycles": rfl_stats.exhausted_cycles,
            "mean_cycle_coverage": rfl_stats.min_cycle_coverage,  # approx
        },
        "asymmetry": {
            "exhaustion_rate_diff": result.asymmetry,
            "within_symmetric_bound": result.a2_symmetry,
            "symmetric_bound": 0.05,
        },
        "admissibility": {
            "joint_classification": result.classification.value,
            "comparison_valid": result.is_valid,
            "a1_bounded_rate": result.a1_bounded_rate,
            "a2_symmetry": result.a2_symmetry,
            "a3_independence": result.a3_independence,
            "a4_cycle_coverage": result.a4_cycle_coverage,
            "r1_excessive_rate": result.r1_excessive_rate,
            "r2_severe_asymmetry": result.r2_severe_asymmetry,
            "r3_systematic_bias": result.r3_systematic_bias,
            "r4_complete_failures": result.r4_complete_failures,
            "reasons": result.reasons,
        },
    }
