"""
Conjecture Engine Contract Implementation.

Evaluates experimental data against theoretical conjectures as specified in
CONJECTURE_ENGINE_CONTRACT.md. This engine is deterministic and interpretive,
operating within strict binding rules defined by RFL_UPLIFT_THEORY.md.

CONSTRAINTS (from contract):
- No invention of new conjectures
- No reinterpretation of conjecture meanings
- No modification of thresholds
- Deterministic: same input → same output
- All conclusions must cite specific data fields

STATUS: PHASE II — NOT RUN IN PHASE I
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports for statistical tests
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# CONSTANTS FROM THEORY (DO NOT MODIFY)
# =============================================================================

# From RFL_UPLIFT_THEORY.md Table 19.1
THRESHOLD_P_VALUE = 0.05
THRESHOLD_LOGISTIC_R2 = 0.80
THRESHOLD_LOGISTIC_R2_PARTIAL = 0.60
THRESHOLD_R2_DIFFERENCE = 0.10
THRESHOLD_PSI_CONVERGED = 0.01
THRESHOLD_PSI_NEARLY_CONVERGED = 0.05
THRESHOLD_PSI_DIVERGED = 0.10
THRESHOLD_OSCILLATION_HEALTHY = 0.20
THRESHOLD_OSCILLATION_WARNING = 0.35
THRESHOLD_ABSTENTION_CONVERGED = 0.10
THRESHOLD_TAU_FLAT = 0.05
THRESHOLD_VARIANCE_REDUCTION = 0.10

ENGINE_VERSION = "2.0.0"
THEORY_VERSION = "RFL_UPLIFT_THEORY.md v2025-12-06"
SNAPSHOT_SCHEMA_VERSION = "1.0"

# Conjecture IDs considered "key" for convergence analysis
KEY_CONVERGENCE_CONJECTURES = [
    "conjecture_3_1",   # Supermartingale Property
    "conjecture_6_1",   # Almost Sure Convergence
    "theorem_13_2",     # Multi-Goal RFL Convergence
    "theorem_15_1",     # Local Stability Criterion
]

# Governance status levels
class GovernanceStatus(Enum):
    """Governance health status levels."""
    OK = "OK"
    WARN = "WARN"
    ATTENTION = "ATTENTION"


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class EvidenceStatus(Enum):
    """Evidence status for conjecture evaluation."""
    SUPPORTS = "SUPPORTS"
    CONSISTENT = "CONSISTENT"
    CONTRADICTS = "CONTRADICTS"
    INCONCLUSIVE = "INCONCLUSIVE"


class ValidationStatus(Enum):
    """Input validation status."""
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass
class Observation:
    """A single observation used in conjecture evaluation."""
    metric: str
    value: Any
    source: str


@dataclass
class Evaluation:
    """Evaluation details for a conjecture."""
    rule_applied: str
    threshold: Optional[float]
    threshold_source: str
    comparison: str
    result: bool


@dataclass
class DiagnosticUsed:
    """A diagnostic value used in evaluation."""
    diagnostic_id: str
    value: Any
    interpretation: str


@dataclass
class ConjectureResult:
    """Result of evaluating a single conjecture."""
    conjecture_id: str
    name: str
    theory_reference: str
    applicable: bool
    applicability_reason: str
    observations: Dict[str, Any]
    evaluation: Optional[Evaluation]
    evidence_status: EvidenceStatus
    evidence_rationale: str
    diagnostics_used: List[DiagnosticUsed]
    caveats: List[str]


@dataclass
class InputValidationResult:
    """Result of input validation."""
    status: ValidationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ReportSummary:
    """Summary of conjecture evaluation."""
    total_evaluated: int
    supports_count: int
    consistent_count: int
    contradicts_count: int
    inconclusive_count: int
    overall_assessment: str


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _validate_jsonl_record(record: Dict[str, Any], index: int, mode: str) -> List[str]:
    """Validate a single JSONL record."""
    errors = []

    # Required top-level fields
    required_fields = ["cycle", "timestamp_utc", "slice_name", "mode", "H_t"]
    for fld in required_fields:
        if fld not in record:
            errors.append(f"Record {index}: missing required field '{fld}'")

    # Validate candidates
    if "candidates" not in record:
        errors.append(f"Record {index}: missing 'candidates' object")
    elif "total" not in record.get("candidates", {}):
        errors.append(f"Record {index}: missing 'candidates.total'")

    # Validate verified
    if "verified" not in record:
        errors.append(f"Record {index}: missing 'verified' object")
    elif "count" not in record.get("verified", {}):
        errors.append(f"Record {index}: missing 'verified.count'")

    # Validate abstained
    if "abstained" not in record:
        errors.append(f"Record {index}: missing 'abstained' object")
    elif "count" not in record.get("abstained", {}):
        errors.append(f"Record {index}: missing 'abstained.count'")

    # Validate metrics
    if "metrics" not in record:
        errors.append(f"Record {index}: missing 'metrics' object")
    else:
        metrics = record["metrics"]
        if "abstention_rate" not in metrics:
            errors.append(f"Record {index}: missing 'metrics.abstention_rate'")
        if "verification_density" not in metrics:
            errors.append(f"Record {index}: missing 'metrics.verification_density'")

    # RFL mode requires policy fields
    if mode == "rfl" and "policy" in record:
        policy = record["policy"]
        if "theta" not in policy:
            errors.append(f"Record {index}: RFL mode missing 'policy.theta'")
        if "gradient_norm" not in policy:
            errors.append(f"Record {index}: RFL mode missing 'policy.gradient_norm'")

    return errors


def _validate_jsonl_file(path: Path, mode: str) -> Tuple[List[Dict], List[str], List[str]]:
    """Validate a JSONL log file."""
    errors = []
    warnings = []
    records = []

    if not path.exists():
        errors.append(f"File not found: {path}")
        return records, errors, warnings

    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_errors = _validate_jsonl_record(record, i, mode)
                    errors.extend(record_errors)
                    records.append(record)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {e}")
    except Exception as e:
        errors.append(f"Error reading {path}: {e}")

    if not records:
        errors.append(f"Empty log file: {path}")

    return records, errors, warnings


def _validate_summary_file(path: Path, mode: str) -> Tuple[Optional[Dict], List[str], List[str]]:
    """Validate a summary JSON file."""
    errors = []
    warnings = []
    summary = None

    if not path.exists():
        errors.append(f"File not found: {path}")
        return summary, errors, warnings

    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {path}: {e}")
        return summary, errors, warnings
    except Exception as e:
        errors.append(f"Error reading {path}: {e}")
        return summary, errors, warnings

    # Required fields
    required = ["experiment_id", "slice_name", "mode", "total_cycles"]
    for fld in required:
        if fld not in summary:
            errors.append(f"Summary missing required field: '{fld}'")

    # Required metrics
    if "metrics" not in summary:
        errors.append("Summary missing 'metrics' object")
    else:
        if "mean_abstention_rate" not in summary["metrics"]:
            errors.append("Summary missing 'metrics.mean_abstention_rate'")
        if "primary_metric" not in summary["metrics"]:
            errors.append("Summary missing 'metrics.primary_metric'")

    # Required time series
    if "time_series" not in summary:
        errors.append("Summary missing 'time_series' object")
    else:
        if "abstention_rates" not in summary["time_series"]:
            errors.append("Summary missing 'time_series.abstention_rates'")
        if "success_rates" not in summary["time_series"]:
            errors.append("Summary missing 'time_series.success_rates'")

    return summary, errors, warnings


def _validate_telemetry_file(path: Path) -> Tuple[Optional[Dict], List[str], List[str]]:
    """Validate a telemetry aggregate JSON file."""
    errors = []
    warnings = []
    telemetry = None

    if not path.exists():
        errors.append(f"File not found: {path}")
        return telemetry, errors, warnings

    try:
        with open(path, "r", encoding="utf-8") as f:
            telemetry = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {path}: {e}")
        return telemetry, errors, warnings
    except Exception as e:
        errors.append(f"Error reading {path}: {e}")
        return telemetry, errors, warnings

    # Required fields
    if "experiment_id" not in telemetry:
        errors.append("Telemetry missing 'experiment_id'")

    # Required comparison
    if "comparison" not in telemetry:
        errors.append("Telemetry missing 'comparison' object")
    else:
        comp = telemetry["comparison"]
        required_comp = ["delta", "ci_95_lower", "ci_95_upper", "ci_excludes_zero"]
        for fld in required_comp:
            if fld not in comp:
                errors.append(f"Telemetry missing 'comparison.{fld}'")

    # Required diagnostics
    if "diagnostics" not in telemetry:
        errors.append("Telemetry missing 'diagnostics' object")
    else:
        diag = telemetry["diagnostics"]
        required_diag = [
            "policy_stability_index", "oscillation_index", "metric_stationary",
            "abstention_trend_tau", "abstention_trend_p"
        ]
        for fld in required_diag:
            if fld not in diag:
                errors.append(f"Telemetry missing 'diagnostics.{fld}'")

    # Required validity
    if "validity" not in telemetry:
        errors.append("Telemetry missing 'validity' object")

    return telemetry, errors, warnings


def validate_inputs(
    baseline_log: Path,
    rfl_log: Path,
    baseline_summary: Path,
    rfl_summary: Path,
    telemetry: Path
) -> Tuple[InputValidationResult, Dict[str, Any]]:
    """
    Validate all inputs conform to schema.

    Returns:
        Tuple of (validation_result, parsed_data)
    """
    all_errors = []
    all_warnings = []
    parsed = {}

    # Validate JSONL logs
    baseline_records, errors, warnings = _validate_jsonl_file(baseline_log, "baseline")
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    parsed["baseline_records"] = baseline_records

    rfl_records, errors, warnings = _validate_jsonl_file(rfl_log, "rfl")
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    parsed["rfl_records"] = rfl_records

    # Validate summaries
    baseline_sum, errors, warnings = _validate_summary_file(baseline_summary, "baseline")
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    parsed["baseline_summary"] = baseline_sum

    rfl_sum, errors, warnings = _validate_summary_file(rfl_summary, "rfl")
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    parsed["rfl_summary"] = rfl_sum

    # Validate telemetry
    telem, errors, warnings = _validate_telemetry_file(telemetry)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    parsed["telemetry"] = telem

    # Determine status
    status = ValidationStatus.INVALID if all_errors else ValidationStatus.VALID

    return InputValidationResult(status, all_errors, all_warnings), parsed


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def _mann_kendall_trend(series: List[float]) -> Tuple[float, float]:
    """
    Compute Mann-Kendall trend test.

    Returns:
        Tuple of (tau, p_value)
    """
    n = len(series)
    if n < 3:
        return 0.0, 1.0

    # Count concordant and discordant pairs
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Compute tau
    tau = s / (n * (n - 1) / 2)

    # Compute p-value using normal approximation
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    if SCIPY_AVAILABLE:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        # Approximate using standard normal
        p_value = 2 * (1 - _approx_normal_cdf(abs(z)))

    return tau, p_value


def _approx_normal_cdf(z: float) -> float:
    """Approximate normal CDF without scipy."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


def _logistic_function(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logistic decay function: a / (1 + exp(b * (t - c)))"""
    return a / (1 + np.exp(b * (t - c)))


def _fit_logistic(series: List[float]) -> Tuple[float, bool]:
    """
    Fit logistic decay to time series.

    Returns:
        Tuple of (r_squared, fit_successful)
    """
    if not SCIPY_AVAILABLE or len(series) < 5:
        return 0.0, False

    try:
        t = np.arange(len(series))
        y = np.array(series)

        # Initial guesses
        p0 = [max(y), -0.1, len(y) / 2]

        # Fit
        popt, _ = curve_fit(_logistic_function, t, y, p0=p0, maxfev=5000)

        # Compute R²
        y_pred = _logistic_function(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0, False

        r2 = 1 - (ss_res / ss_tot)
        return max(0.0, r2), True
    except Exception:
        return 0.0, False


def _fit_linear(series: List[float]) -> float:
    """Fit linear model and return R²."""
    if len(series) < 3:
        return 0.0

    t = np.arange(len(series))
    y = np.array(series)

    # Linear regression
    n = len(t)
    sum_t = np.sum(t)
    sum_y = np.sum(y)
    sum_ty = np.sum(t * y)
    sum_t2 = np.sum(t ** 2)

    denom = n * sum_t2 - sum_t ** 2
    if denom == 0:
        return 0.0

    slope = (n * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / n

    y_pred = slope * t + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 0.0

    return max(0.0, 1 - (ss_res / ss_tot))


# =============================================================================
# BINDING RULE IMPLEMENTATIONS
# =============================================================================

def _check_R3_1_supermartingale(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any]
) -> ConjectureResult:
    """
    RULE R3.1: Conjecture 3.1 (Supermartingale Property)

    Evaluates whether abstention rate shows negative drift.
    """
    diagnostics = telemetry.get("diagnostics", {})
    tau = diagnostics.get("abstention_trend_tau", 0.0)
    p_value = diagnostics.get("abstention_trend_p", 1.0)

    observations = {
        "primary": {
            "metric": "abstention_trend_tau",
            "value": tau,
            "source": "telemetry.diagnostics.abstention_trend_tau"
        },
        "secondary": [
            {
                "metric": "abstention_trend_p",
                "value": p_value,
                "source": "telemetry.diagnostics.abstention_trend_p"
            }
        ]
    }

    # Evaluation logic
    if tau < 0 and p_value < THRESHOLD_P_VALUE:
        status = EvidenceStatus.SUPPORTS
        rationale = "Abstention rate shows statistically significant decreasing trend"
        comparison = f"tau={tau:.4f} < 0 AND p={p_value:.4f} < {THRESHOLD_P_VALUE}"
        result = True
    elif tau < 0 and p_value >= THRESHOLD_P_VALUE:
        status = EvidenceStatus.CONSISTENT
        rationale = "Abstention rate decreasing but not statistically significant"
        comparison = f"tau={tau:.4f} < 0 AND p={p_value:.4f} >= {THRESHOLD_P_VALUE}"
        result = False
    elif tau > 0 and p_value < THRESHOLD_P_VALUE:
        status = EvidenceStatus.CONTRADICTS
        rationale = "Abstention rate shows statistically significant INCREASING trend"
        comparison = f"tau={tau:.4f} > 0 AND p={p_value:.4f} < {THRESHOLD_P_VALUE}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Abstention rate flat; insufficient signal for trend determination"
        comparison = f"|tau|={abs(tau):.4f} < {THRESHOLD_TAU_FLAT}"
        result = False

    return ConjectureResult(
        conjecture_id="conjecture_3_1",
        name="Supermartingale Property",
        theory_reference="RFL_UPLIFT_THEORY.md §3.1, Theorem 3.1",
        applicable=True,
        applicability_reason="Applies to all slices",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R3.1",
            threshold=THRESHOLD_P_VALUE,
            threshold_source="Table 19.1 (RFL_UPLIFT_THEORY.md §19.4)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("abstention_trend_tau", tau, f"Kendall's tau = {tau:.4f}"),
            DiagnosticUsed("abstention_trend_p", p_value, f"p-value = {p_value:.4f}")
        ],
        caveats=["Trend test assumes sufficient cycles for statistical power"]
    )


def _check_R4_1_logistic_decay(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any],
    slice_name: str
) -> ConjectureResult:
    """
    RULE R4.1: Conjecture 4.1 (Logistic Decay)

    Evaluates whether abstention curve fits logistic decay model.
    """
    # Get abstention time series
    time_series = rfl_summary.get("time_series", {})
    abstention_rates = time_series.get("abstention_rates", [])

    # Fit models
    logistic_r2, fit_ok = _fit_logistic(abstention_rates)
    linear_r2 = _fit_linear(abstention_rates)

    # Primary applicability for Slice A
    is_primary = slice_name in ["slice_uplift_goal", "slice_a", "A"]

    observations = {
        "primary": {
            "metric": "logistic_r2",
            "value": logistic_r2,
            "source": "computed from time_series.abstention_rates"
        },
        "secondary": [
            {
                "metric": "linear_r2",
                "value": linear_r2,
                "source": "computed from time_series.abstention_rates"
            },
            {
                "metric": "fit_successful",
                "value": fit_ok,
                "source": "logistic curve_fit result"
            }
        ]
    }

    if not fit_ok:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Logistic fit failed; insufficient data or numerical issues"
        comparison = "fit_successful = False"
        result = False
    elif logistic_r2 > THRESHOLD_LOGISTIC_R2:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Abstention curve fits logistic decay model (R² = {logistic_r2:.3f} > {THRESHOLD_LOGISTIC_R2})"
        comparison = f"logistic_r2={logistic_r2:.3f} > {THRESHOLD_LOGISTIC_R2}"
        result = True
    elif logistic_r2 > THRESHOLD_LOGISTIC_R2_PARTIAL and logistic_r2 > linear_r2:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Abstention curve partially fits logistic model (R² = {logistic_r2:.3f})"
        comparison = f"logistic_r2={logistic_r2:.3f} > {THRESHOLD_LOGISTIC_R2_PARTIAL} AND > linear_r2"
        result = False
    elif linear_r2 > logistic_r2 + THRESHOLD_R2_DIFFERENCE:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Linear model fits better than logistic (linear R² = {linear_r2:.3f})"
        comparison = f"linear_r2={linear_r2:.3f} > logistic_r2 + {THRESHOLD_R2_DIFFERENCE}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "No model fits well; data may be noisy or insufficient"
        comparison = f"logistic_r2={logistic_r2:.3f}, linear_r2={linear_r2:.3f}"
        result = False

    return ConjectureResult(
        conjecture_id="conjecture_4_1",
        name="Logistic Decay",
        theory_reference="RFL_UPLIFT_THEORY.md §4.2",
        applicable=True,
        applicability_reason="Primary for Slice A, secondary for others" if is_primary else "Secondary applicability",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R4.1",
            threshold=THRESHOLD_LOGISTIC_R2,
            threshold_source="Table 19.1 (RFL_UPLIFT_THEORY.md §19.4)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("logistic_r2", logistic_r2, f"Logistic fit R² = {logistic_r2:.3f}"),
            DiagnosticUsed("linear_r2", linear_r2, f"Linear fit R² = {linear_r2:.3f}")
        ],
        caveats=["Requires scipy for curve fitting", "Sensitive to initial conditions"]
    )


def _check_R6_1_convergence(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any]
) -> ConjectureResult:
    """
    RULE R6.1: Conjecture 6.1 (Almost Sure Convergence)

    Evaluates whether abstention converges to zero.
    """
    diagnostics = telemetry.get("diagnostics", {})
    metric_stationary = diagnostics.get("metric_stationary", False)
    psi = diagnostics.get("policy_stability_index", 1.0)

    metrics = rfl_summary.get("metrics", {})
    final_abstention = metrics.get("mean_abstention_rate", 1.0)

    observations = {
        "primary": {
            "metric": "metric_stationary",
            "value": metric_stationary,
            "source": "telemetry.diagnostics.metric_stationary"
        },
        "secondary": [
            {
                "metric": "final_abstention_rate",
                "value": final_abstention,
                "source": "rfl_summary.metrics.mean_abstention_rate"
            },
            {
                "metric": "policy_stability_index",
                "value": psi,
                "source": "telemetry.diagnostics.policy_stability_index"
            }
        ]
    }

    if metric_stationary and final_abstention < THRESHOLD_ABSTENTION_CONVERGED:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Abstention converged to near-zero level (α = {final_abstention:.3f})"
        comparison = f"stationary=True AND α={final_abstention:.3f} < {THRESHOLD_ABSTENTION_CONVERGED}"
        result = True
    elif metric_stationary and final_abstention >= THRESHOLD_ABSTENTION_CONVERGED:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Metric stationary but not at zero (α = {final_abstention:.3f}); may be local optimum"
        comparison = f"stationary=True AND α={final_abstention:.3f} >= {THRESHOLD_ABSTENTION_CONVERGED}"
        result = False
    elif not metric_stationary and psi > THRESHOLD_PSI_NEARLY_CONVERGED:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Neither metric nor policy converged (Ψ = {psi:.4f})"
        comparison = f"stationary=False AND Ψ={psi:.4f} > {THRESHOLD_PSI_NEARLY_CONVERGED}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Convergence status unclear; may need more cycles"
        comparison = f"stationary={metric_stationary}, Ψ={psi:.4f}"
        result = False

    return ConjectureResult(
        conjecture_id="conjecture_6_1",
        name="Almost Sure Convergence",
        theory_reference="RFL_UPLIFT_THEORY.md §6.1",
        applicable=True,
        applicability_reason="Applies to all slices after T_max cycles",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R6.1",
            threshold=THRESHOLD_ABSTENTION_CONVERGED,
            threshold_source="Definition 14.2 (RFL_UPLIFT_THEORY.md §14.2)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("metric_stationary", metric_stationary,
                          "ADF test: stationary" if metric_stationary else "ADF test: non-stationary"),
            DiagnosticUsed("policy_stability_index", psi, f"Ψ = {psi:.4f}")
        ],
        caveats=["Stationarity test requires sufficient data points"]
    )


def _check_R13_2_multi_goal_convergence(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any],
    slice_name: str
) -> ConjectureResult:
    """
    RULE R13.2: Theorem 13.2 (Multi-Goal RFL Convergence)

    Evaluates whether policy converges in multi-goal setting.
    """
    diagnostics = telemetry.get("diagnostics", {})
    psi = diagnostics.get("policy_stability_index", 1.0)

    # Check if primary metric is improving (crude check)
    time_series = rfl_summary.get("time_series", {})
    success_rates = time_series.get("success_rates", [])
    metric_improving = False
    if len(success_rates) >= 2:
        early_mean = np.mean(success_rates[:len(success_rates)//3]) if success_rates else 0
        late_mean = np.mean(success_rates[-len(success_rates)//3:]) if success_rates else 0
        metric_improving = late_mean > early_mean

    # Check for theta norm (divergence)
    policy_final = rfl_summary.get("policy_final", {})
    theta_norm = policy_final.get("theta_norm", 0.0)
    theta_diverged = theta_norm > 1e6 or np.isnan(theta_norm) or np.isinf(theta_norm)

    # Primary applicability for Slices C and D
    is_primary = slice_name in ["slice_uplift_tree", "slice_uplift_dependency", "slice_c", "slice_d", "C", "D"]

    observations = {
        "primary": {
            "metric": "policy_stability_index",
            "value": psi,
            "source": "telemetry.diagnostics.policy_stability_index"
        },
        "secondary": [
            {
                "metric": "metric_improving",
                "value": metric_improving,
                "source": "computed from time_series.success_rates"
            },
            {
                "metric": "theta_norm",
                "value": theta_norm,
                "source": "rfl_summary.policy_final.theta_norm"
            }
        ]
    }

    if theta_diverged:
        status = EvidenceStatus.CONTRADICTS
        rationale = "Policy parameters diverged"
        comparison = f"theta_norm={theta_norm} → ∞"
        result = False
    elif psi < THRESHOLD_PSI_CONVERGED and metric_improving:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Policy converged (Ψ = {psi:.4f} < {THRESHOLD_PSI_CONVERGED}) with improving metric"
        comparison = f"Ψ={psi:.4f} < {THRESHOLD_PSI_CONVERGED} AND metric_improving=True"
        result = True
    elif psi < THRESHOLD_PSI_NEARLY_CONVERGED:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Policy nearly converged (Ψ = {psi:.4f}); may be at local optimum"
        comparison = f"Ψ={psi:.4f} < {THRESHOLD_PSI_NEARLY_CONVERGED}"
        result = False
    elif psi > THRESHOLD_PSI_DIVERGED:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Policy failed to converge (Ψ = {psi:.4f} > {THRESHOLD_PSI_DIVERGED})"
        comparison = f"Ψ={psi:.4f} > {THRESHOLD_PSI_DIVERGED}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Convergence status unclear"
        comparison = f"Ψ={psi:.4f}"
        result = False

    return ConjectureResult(
        conjecture_id="theorem_13_2",
        name="Multi-Goal RFL Convergence",
        theory_reference="RFL_UPLIFT_THEORY.md §13.4",
        applicable=True,
        applicability_reason="Primary for Slices C, D" if is_primary else "Secondary applicability",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R13.2",
            threshold=THRESHOLD_PSI_CONVERGED,
            threshold_source="Convergence Rule 14.1 (RFL_UPLIFT_THEORY.md §14.1)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("policy_stability_index", psi, f"Ψ = {psi:.4f}"),
            DiagnosticUsed("theta_norm", theta_norm, f"||θ|| = {theta_norm:.2f}")
        ],
        caveats=["Requires policy parameters to be tracked over time"]
    )


def _check_R15_1_local_stability(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any]
) -> ConjectureResult:
    """
    RULE R15.1: Theorem 15.1 (Local Stability Criterion)

    Evaluates whether policy is locally stable.
    """
    diagnostics = telemetry.get("diagnostics", {})
    oscillation = diagnostics.get("oscillation_index", 1.0)

    policy_final = rfl_summary.get("policy_final", {})
    theta_norm = policy_final.get("theta_norm", 0.0)
    theta_bounded = not (np.isnan(theta_norm) or np.isinf(theta_norm) or theta_norm > 1e6)

    observations = {
        "primary": {
            "metric": "theta_bounded",
            "value": theta_bounded,
            "source": "computed from rfl_summary.policy_final.theta_norm"
        },
        "secondary": [
            {
                "metric": "oscillation_index",
                "value": oscillation,
                "source": "telemetry.diagnostics.oscillation_index"
            },
            {
                "metric": "theta_norm",
                "value": theta_norm,
                "source": "rfl_summary.policy_final.theta_norm"
            }
        ]
    }

    if not theta_bounded:
        status = EvidenceStatus.CONTRADICTS
        rationale = "Policy diverged (unbounded growth)"
        comparison = f"theta_bounded=False (||θ||={theta_norm})"
        result = False
    elif theta_bounded and oscillation < THRESHOLD_OSCILLATION_HEALTHY:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Policy stable: bounded parameters, low oscillation (O = {oscillation:.3f})"
        comparison = f"bounded=True AND O={oscillation:.3f} < {THRESHOLD_OSCILLATION_HEALTHY}"
        result = True
    elif theta_bounded and oscillation >= THRESHOLD_OSCILLATION_HEALTHY:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Policy bounded but oscillating (O = {oscillation:.3f}); may need momentum"
        comparison = f"bounded=True AND O={oscillation:.3f} >= {THRESHOLD_OSCILLATION_HEALTHY}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Stability status unclear"
        comparison = f"bounded={theta_bounded}, O={oscillation:.3f}"
        result = False

    return ConjectureResult(
        conjecture_id="theorem_15_1",
        name="Local Stability Criterion",
        theory_reference="RFL_UPLIFT_THEORY.md §15.2",
        applicable=True,
        applicability_reason="Applies to all slices with RFL mode",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R15.1",
            threshold=THRESHOLD_OSCILLATION_HEALTHY,
            threshold_source="Theorem 15.1 (RFL_UPLIFT_THEORY.md §15.2)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("oscillation_index", oscillation, f"O = {oscillation:.3f}"),
            DiagnosticUsed("theta_norm", theta_norm, f"||θ|| = {theta_norm:.2f}")
        ],
        caveats=["Oscillation detection requires consecutive theta_delta values"]
    )


def _check_R15_4_basin_structure(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any],
    slice_name: str
) -> ConjectureResult:
    """
    RULE R15.4: Conjecture 15.4 (Basin Structure for U2 Slices)

    Evaluates whether observed pattern matches predicted basin structure.
    """
    patterns = telemetry.get("patterns", {})
    detected_pattern = patterns.get("detected_pattern", "")
    pattern_confidence = patterns.get("pattern_confidence", 0.0)

    # Predicted patterns per slice
    predictions = {
        "slice_uplift_goal": ["A.1", "A.2"],
        "slice_a": ["A.1", "A.2"],
        "A": ["A.1", "A.2"],
        "slice_uplift_sparse": ["B.1", "B.4"],
        "slice_b": ["B.1", "B.4"],
        "B": ["B.1", "B.4"],
        "slice_uplift_tree": ["C.1"],
        "slice_c": ["C.1"],
        "C": ["C.1"],
        "slice_uplift_dependency": ["D.3", "D.4"],
        "slice_d": ["D.3", "D.4"],
        "D": ["D.3", "D.4"],
    }

    expected = predictions.get(slice_name, [])
    expected_str = ", ".join(expected) if expected else "unknown"

    observations = {
        "primary": {
            "metric": "detected_pattern",
            "value": detected_pattern,
            "source": "telemetry.patterns.detected_pattern"
        },
        "secondary": [
            {
                "metric": "pattern_confidence",
                "value": pattern_confidence,
                "source": "telemetry.patterns.pattern_confidence"
            },
            {
                "metric": "expected_patterns",
                "value": expected,
                "source": "Conjecture 15.4 predictions"
            }
        ]
    }

    if not detected_pattern:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "No pattern detected; insufficient data to assess basin structure"
        comparison = "detected_pattern is empty"
        result = False
    elif detected_pattern in expected:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Basin structure matches predicted pattern ({detected_pattern} in {expected_str})"
        comparison = f"'{detected_pattern}' in {expected}"
        result = True
    elif detected_pattern and len(detected_pattern) > 0 and slice_name and len(slice_name) > 0:
        # Check if same family
        pattern_family = detected_pattern[0].upper() if detected_pattern else ""
        slice_family = slice_name[-1].upper() if slice_name.startswith("slice_") else slice_name[0].upper()
        if pattern_family == slice_family:
            status = EvidenceStatus.CONSISTENT
            rationale = f"Basin structure partially consistent ({detected_pattern}, expected {expected_str})"
            comparison = f"'{detected_pattern}' same family as {expected_str}"
            result = False
        else:
            status = EvidenceStatus.CONTRADICTS
            rationale = f"Basin structure contradicts prediction (got {detected_pattern}, expected {expected_str})"
            comparison = f"'{detected_pattern}' not in {expected}"
            result = False
    else:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Basin structure contradicts prediction (got {detected_pattern}, expected {expected_str})"
        comparison = f"'{detected_pattern}' not in {expected}"
        result = False

    return ConjectureResult(
        conjecture_id="conjecture_15_4",
        name="Basin Structure for U2 Slices",
        theory_reference="RFL_UPLIFT_THEORY.md §15.5",
        applicable=True,
        applicability_reason=f"Slice-specific prediction for {slice_name}",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R15.4",
            threshold=None,
            threshold_source="Conjecture 15.4 pattern predictions",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("detected_pattern", detected_pattern, f"Pattern: {detected_pattern or 'none'}"),
            DiagnosticUsed("pattern_confidence", pattern_confidence, f"Confidence: {pattern_confidence:.2f}")
        ],
        caveats=["Pattern detection is heuristic", "Requires sufficient cycles for reliable detection"]
    )


def _check_R2_1_variance_amplification(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any],
    slice_name: str
) -> ConjectureResult:
    """
    RULE R2.1: Lemma 2.1 (Variance Under Wide Slice)

    Evaluates whether variance reduced as policy learned.
    """
    time_series = rfl_summary.get("time_series", {})
    densities = time_series.get("densities", [])

    # Primary for Slice B
    is_primary = slice_name in ["slice_uplift_sparse", "slice_b", "B"]

    # Compute early vs late variance
    if len(densities) >= 6:
        n = len(densities)
        third = n // 3
        early_var = float(np.var(densities[:third]))
        late_var = float(np.var(densities[-third:]))
        variance_reduction = early_var - late_var
        has_data = True
    else:
        early_var = 0.0
        late_var = 0.0
        variance_reduction = 0.0
        has_data = False

    observations = {
        "primary": {
            "metric": "variance_reduction",
            "value": variance_reduction,
            "source": "computed from time_series.densities"
        },
        "secondary": [
            {
                "metric": "early_density_variance",
                "value": early_var,
                "source": "first third of densities"
            },
            {
                "metric": "late_density_variance",
                "value": late_var,
                "source": "last third of densities"
            }
        ]
    }

    if not has_data:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Insufficient density data for variance analysis"
        comparison = f"len(densities)={len(densities)} < 6"
        result = False
    elif early_var > late_var and variance_reduction > THRESHOLD_VARIANCE_REDUCTION:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Variance reduced as policy specialized (Δvar = {variance_reduction:.4f})"
        comparison = f"early_var={early_var:.4f} > late_var={late_var:.4f}, reduction > {THRESHOLD_VARIANCE_REDUCTION}"
        result = True
    elif abs(variance_reduction) < THRESHOLD_VARIANCE_REDUCTION:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Variance stable; policy may not be learning density (Δvar = {variance_reduction:.4f})"
        comparison = f"|variance_reduction|={abs(variance_reduction):.4f} < {THRESHOLD_VARIANCE_REDUCTION}"
        result = False
    elif early_var < late_var:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Variance increased; policy drifting away from optima (Δvar = {variance_reduction:.4f})"
        comparison = f"early_var={early_var:.4f} < late_var={late_var:.4f}"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = "Variance pattern unclear"
        comparison = f"early_var={early_var:.4f}, late_var={late_var:.4f}"
        result = False

    return ConjectureResult(
        conjecture_id="lemma_2_1",
        name="Variance Under Wide Slice",
        theory_reference="RFL_UPLIFT_THEORY.md §2.2",
        applicable=True,
        applicability_reason="Primary for Slice B" if is_primary else "Secondary applicability",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R2.1",
            threshold=THRESHOLD_VARIANCE_REDUCTION,
            threshold_source="Lemma 2.1 proof sketch",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("early_variance", early_var, f"Early σ² = {early_var:.4f}"),
            DiagnosticUsed("late_variance", late_var, f"Late σ² = {late_var:.4f}")
        ],
        caveats=["Requires density tracking over time"]
    )


def _check_R2_2_learning_signal(
    telemetry: Dict[str, Any],
    rfl_summary: Dict[str, Any]
) -> ConjectureResult:
    """
    RULE R2.2: Proposition 2.2 (Entropy-Signal Correspondence)

    Evaluates whether higher variance yielded learning signal (uplift).
    """
    comparison_data = telemetry.get("comparison", {})
    delta = comparison_data.get("delta", 0.0)
    ci_excludes_zero = comparison_data.get("ci_excludes_zero", False)
    ci_lower = comparison_data.get("ci_95_lower", 0.0)
    ci_upper = comparison_data.get("ci_95_upper", 0.0)

    observations = {
        "primary": {
            "metric": "delta",
            "value": delta,
            "source": "telemetry.comparison.delta"
        },
        "secondary": [
            {
                "metric": "ci_excludes_zero",
                "value": ci_excludes_zero,
                "source": "telemetry.comparison.ci_excludes_zero"
            },
            {
                "metric": "ci_95",
                "value": [ci_lower, ci_upper],
                "source": "telemetry.comparison.ci_95_*"
            }
        ]
    }

    if delta > 0 and ci_excludes_zero:
        status = EvidenceStatus.SUPPORTS
        rationale = f"Positive uplift detected (Δ = {delta:.4f}, CI excludes 0)"
        comparison = f"Δ={delta:.4f} > 0 AND ci_excludes_zero=True"
        result = True
    elif delta > 0 and not ci_excludes_zero:
        status = EvidenceStatus.CONSISTENT
        rationale = f"Positive trend but not statistically significant (Δ = {delta:.4f})"
        comparison = f"Δ={delta:.4f} > 0 AND ci_excludes_zero=False"
        result = False
    elif delta <= 0 and ci_excludes_zero:
        status = EvidenceStatus.CONTRADICTS
        rationale = f"Negative uplift; learning signal did not translate to improvement (Δ = {delta:.4f})"
        comparison = f"Δ={delta:.4f} <= 0 AND ci_excludes_zero=True"
        result = False
    else:
        status = EvidenceStatus.INCONCLUSIVE
        rationale = f"Uplift near zero; signal-variance relationship unclear (Δ = {delta:.4f})"
        comparison = f"Δ={delta:.4f} ≈ 0"
        result = False

    return ConjectureResult(
        conjecture_id="proposition_2_2",
        name="Entropy-Signal Correspondence",
        theory_reference="RFL_UPLIFT_THEORY.md §2.3",
        applicable=True,
        applicability_reason="Applies to all slices",
        observations=observations,
        evaluation=Evaluation(
            rule_applied="R2.2",
            threshold=0.0,
            threshold_source="Definition 17.1 (RFL_UPLIFT_THEORY.md §17.4)",
            comparison=comparison,
            result=result
        ),
        evidence_status=status,
        evidence_rationale=rationale,
        diagnostics_used=[
            DiagnosticUsed("delta", delta, f"Uplift Δ = {delta:.4f}"),
            DiagnosticUsed("ci_95", f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                          "Excludes 0" if ci_excludes_zero else "Includes 0")
        ],
        caveats=["Uplift calculation requires paired baseline/RFL runs"]
    )


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-JSON types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _conjecture_result_to_dict(result: ConjectureResult) -> Dict[str, Any]:
    """Convert ConjectureResult to dictionary for JSON serialization."""
    raw = {
        "conjecture_id": result.conjecture_id,
        "name": result.name,
        "theory_reference": result.theory_reference,
        "applicable": result.applicable,
        "applicability_reason": result.applicability_reason,
        "observations": result.observations,
        "evaluation": {
            "rule_applied": result.evaluation.rule_applied,
            "threshold": result.evaluation.threshold,
            "threshold_source": result.evaluation.threshold_source,
            "comparison": result.evaluation.comparison,
            "result": result.evaluation.result
        } if result.evaluation else None,
        "evidence_status": result.evidence_status.value,
        "evidence_rationale": result.evidence_rationale,
        "diagnostics_used": [
            {
                "diagnostic_id": d.diagnostic_id,
                "value": d.value,
                "interpretation": d.interpretation
            }
            for d in result.diagnostics_used
        ],
        "caveats": result.caveats
    }
    return _make_json_serializable(raw)


def evaluate_conjectures(
    baseline_log: Path,
    rfl_log: Path,
    baseline_summary: Path,
    rfl_summary: Path,
    telemetry_aggregate: Path,
    output_path: Path,
    *,
    engine_version: str = ENGINE_VERSION,
    theory_version: str = THEORY_VERSION
) -> Dict[str, Any]:
    """
    Evaluate experimental data against theoretical conjectures.

    Args:
        baseline_log: Path to baseline JSONL log
        rfl_log: Path to RFL JSONL log
        baseline_summary: Path to baseline summary JSON
        rfl_summary: Path to RFL summary JSON
        telemetry_aggregate: Path to telemetry aggregate JSON
        output_path: Path to write conjecture_report.json
        engine_version: Version of this engine
        theory_version: Version of theory document used

    Returns:
        Dictionary report (also written to output_path)
    """
    # Convert to Path objects
    baseline_log = Path(baseline_log)
    rfl_log = Path(rfl_log)
    baseline_summary = Path(baseline_summary)
    rfl_summary = Path(rfl_summary)
    telemetry_aggregate = Path(telemetry_aggregate)
    output_path = Path(output_path)

    # Validate inputs
    validation_result, parsed = validate_inputs(
        baseline_log, rfl_log, baseline_summary, rfl_summary, telemetry_aggregate
    )

    # Generate timestamp
    generated_at = datetime.now(timezone.utc).isoformat()

    # Compute file hashes
    input_hashes = {}
    for name, path in [
        ("baseline_log_sha256", baseline_log),
        ("rfl_log_sha256", rfl_log),
        ("baseline_summary_sha256", baseline_summary),
        ("rfl_summary_sha256", rfl_summary),
        ("telemetry_sha256", telemetry_aggregate)
    ]:
        if path.exists():
            input_hashes[name] = _compute_file_hash(path)
        else:
            input_hashes[name] = "FILE_NOT_FOUND"

    # If validation failed, return early with no conjecture evaluation
    if validation_result.status == ValidationStatus.INVALID:
        report = {
            "report_version": "1.0",
            "generated_at": generated_at,
            "experiment_id": "UNKNOWN",
            "slice_name": "UNKNOWN",
            "input_validation": {
                "status": validation_result.status.value,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings
            },
            "conjectures": {},
            "summary": {
                "total_evaluated": 0,
                "supports_count": 0,
                "consistent_count": 0,
                "contradicts_count": 0,
                "inconclusive_count": 0,
                "overall_assessment": "INPUT_INVALID - No conjectures evaluated"
            },
            "provenance": {
                "input_files": {
                    "baseline_log": str(baseline_log),
                    "rfl_log": str(rfl_log),
                    "baseline_summary": str(baseline_summary),
                    "rfl_summary": str(rfl_summary),
                    "telemetry": str(telemetry_aggregate)
                },
                "input_hashes": input_hashes,
                "engine_version": engine_version,
                "theory_document_version": theory_version
            }
        }

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    # Extract parsed data
    telemetry = parsed["telemetry"]
    rfl_sum = parsed["rfl_summary"]

    # Get experiment metadata
    experiment_id = telemetry.get("experiment_id", rfl_sum.get("experiment_id", "UNKNOWN"))
    slice_name = rfl_sum.get("slice_name", "UNKNOWN")

    # Evaluate all conjectures
    conjecture_results = []

    # R3.1: Supermartingale
    conjecture_results.append(_check_R3_1_supermartingale(telemetry, rfl_sum))

    # R4.1: Logistic Decay
    conjecture_results.append(_check_R4_1_logistic_decay(telemetry, rfl_sum, slice_name))

    # R6.1: Almost Sure Convergence
    conjecture_results.append(_check_R6_1_convergence(telemetry, rfl_sum))

    # R13.2: Multi-Goal Convergence
    conjecture_results.append(_check_R13_2_multi_goal_convergence(telemetry, rfl_sum, slice_name))

    # R15.1: Local Stability
    conjecture_results.append(_check_R15_1_local_stability(telemetry, rfl_sum))

    # R15.4: Basin Structure
    conjecture_results.append(_check_R15_4_basin_structure(telemetry, rfl_sum, slice_name))

    # R2.1: Variance Amplification
    conjecture_results.append(_check_R2_1_variance_amplification(telemetry, rfl_sum, slice_name))

    # R2.2: Learning Signal
    conjecture_results.append(_check_R2_2_learning_signal(telemetry, rfl_sum))

    # Convert to dict
    conjectures_dict = {r.conjecture_id: _conjecture_result_to_dict(r) for r in conjecture_results}

    # Compute summary
    supports = sum(1 for r in conjecture_results if r.evidence_status == EvidenceStatus.SUPPORTS)
    consistent = sum(1 for r in conjecture_results if r.evidence_status == EvidenceStatus.CONSISTENT)
    contradicts = sum(1 for r in conjecture_results if r.evidence_status == EvidenceStatus.CONTRADICTS)
    inconclusive = sum(1 for r in conjecture_results if r.evidence_status == EvidenceStatus.INCONCLUSIVE)

    # Overall assessment
    if contradicts > 0:
        overall = f"MIXED - {supports} support, {contradicts} contradict"
    elif supports > 0:
        overall = f"POSITIVE - {supports} conjectures supported"
    elif consistent > 0:
        overall = f"TENTATIVE - {consistent} conjectures consistent, none strongly supported"
    else:
        overall = f"INCONCLUSIVE - Insufficient evidence for {inconclusive} conjectures"

    # Build report
    report = {
        "report_version": "1.0",
        "generated_at": generated_at,
        "experiment_id": experiment_id,
        "slice_name": slice_name,
        "input_validation": {
            "status": validation_result.status.value,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings
        },
        "conjectures": conjectures_dict,
        "summary": {
            "total_evaluated": len(conjecture_results),
            "supports_count": supports,
            "consistent_count": consistent,
            "contradicts_count": contradicts,
            "inconclusive_count": inconclusive,
            "overall_assessment": overall
        },
        "provenance": {
            "input_files": {
                "baseline_log": str(baseline_log),
                "rfl_log": str(rfl_log),
                "baseline_summary": str(baseline_summary),
                "rfl_summary": str(rfl_summary),
                "telemetry": str(telemetry_aggregate)
            },
            "input_hashes": input_hashes,
            "engine_version": engine_version,
            "theory_document_version": theory_version
        }
    }

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# =============================================================================
# TASK 1: CONJECTURE TRAJECTORY SNAPSHOT & DELTA
# =============================================================================

def build_conjecture_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact per-run snapshot of conjecture evidence statuses.

    This snapshot is designed for trajectory tracking across multiple runs,
    enabling detection of how evidence evolves over time.

    Args:
        report: A conjecture report dictionary from evaluate_conjectures().

    Returns:
        A compact snapshot dictionary containing:
        - schema_version: Version of the snapshot schema
        - experiment_id: ID of the experiment
        - slice_name: Name of the slice
        - generated_at: Timestamp from the original report
        - statuses: Dict mapping conjecture_id → evidence status string
        - counts: Summary counts (supports, consistent, contradicts, inconclusive)
    """
    conjectures = report.get("conjectures", {})

    statuses = {}
    for conj_id, conj_data in conjectures.items():
        statuses[conj_id] = conj_data.get("evidence_status", "UNKNOWN")

    summary = report.get("summary", {})

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "experiment_id": report.get("experiment_id", "UNKNOWN"),
        "slice_name": report.get("slice_name", "UNKNOWN"),
        "generated_at": report.get("generated_at", ""),
        "statuses": statuses,
        "counts": {
            "supports": summary.get("supports_count", 0),
            "consistent": summary.get("consistent_count", 0),
            "contradicts": summary.get("contradicts_count", 0),
            "inconclusive": summary.get("inconclusive_count", 0),
        },
    }


def compare_conjecture_snapshots(
    old: Dict[str, Any],
    new: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two conjecture snapshots to identify status transitions.

    This function is pure and deterministic: same inputs produce same outputs.

    Args:
        old: The earlier snapshot (from build_conjecture_snapshot).
        new: The later snapshot (from build_conjecture_snapshot).

    Returns:
        A delta dictionary containing:
        - old_experiment_id, new_experiment_id: Experiment IDs
        - old_generated_at, new_generated_at: Timestamps
        - transitions: List of {conjecture_id, from_status, to_status}
        - improved: List of conjecture IDs that moved toward SUPPORTS
        - degraded: List of conjecture IDs that moved toward CONTRADICTS
        - unchanged: List of conjecture IDs with same status
        - net_change: Summary of overall direction (+supports, -contradicts)
    """
    old_statuses = old.get("statuses", {})
    new_statuses = new.get("statuses", {})

    # Define status ordering for improvement/degradation detection
    # SUPPORTS > CONSISTENT > INCONCLUSIVE > CONTRADICTS
    status_rank = {
        "SUPPORTS": 3,
        "CONSISTENT": 2,
        "INCONCLUSIVE": 1,
        "CONTRADICTS": 0,
        "UNKNOWN": -1,
    }

    transitions = []
    improved = []
    degraded = []
    unchanged = []

    # Get union of all conjecture IDs
    all_ids = set(old_statuses.keys()) | set(new_statuses.keys())

    for conj_id in sorted(all_ids):
        old_status = old_statuses.get(conj_id, "UNKNOWN")
        new_status = new_statuses.get(conj_id, "UNKNOWN")

        if old_status != new_status:
            transitions.append({
                "conjecture_id": conj_id,
                "from_status": old_status,
                "to_status": new_status,
            })

            old_rank = status_rank.get(old_status, -1)
            new_rank = status_rank.get(new_status, -1)

            if new_rank > old_rank:
                improved.append(conj_id)
            elif new_rank < old_rank:
                degraded.append(conj_id)
        else:
            unchanged.append(conj_id)

    # Compute net change
    old_counts = old.get("counts", {})
    new_counts = new.get("counts", {})

    net_change = {
        "supports_delta": new_counts.get("supports", 0) - old_counts.get("supports", 0),
        "contradicts_delta": new_counts.get("contradicts", 0) - old_counts.get("contradicts", 0),
        "consistent_delta": new_counts.get("consistent", 0) - old_counts.get("consistent", 0),
        "inconclusive_delta": new_counts.get("inconclusive", 0) - old_counts.get("inconclusive", 0),
    }

    return {
        "old_experiment_id": old.get("experiment_id", "UNKNOWN"),
        "new_experiment_id": new.get("experiment_id", "UNKNOWN"),
        "old_generated_at": old.get("generated_at", ""),
        "new_generated_at": new.get("generated_at", ""),
        "transitions": transitions,
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "net_change": net_change,
    }


# =============================================================================
# TASK 2: RFL INTEGRATION HOOK - EPISTEMIC TREND SIGNAL
# =============================================================================

def summarize_conjectures_for_rfl(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide a simple signal RFL (or meta-controller) can consume.

    This summary is small, JSON-friendly, and focuses on information
    relevant to understanding learning dynamics.

    Args:
        snapshot: A conjecture snapshot from build_conjecture_snapshot().

    Returns:
        A compact RFL-friendly summary containing:
        - num_supports: Count of conjectures with SUPPORTS status
        - num_contradicts: Count of conjectures with CONTRADICTS status
        - num_consistent: Count of conjectures with CONSISTENT status
        - num_inconclusive: Count of conjectures with INCONCLUSIVE status
        - key_conjectures: Status of convergence-related conjectures
        - learning_health: Simple assessment (HEALTHY/MIXED/UNHEALTHY)
    """
    counts = snapshot.get("counts", {})
    statuses = snapshot.get("statuses", {})

    num_supports = counts.get("supports", 0)
    num_contradicts = counts.get("contradicts", 0)
    num_consistent = counts.get("consistent", 0)
    num_inconclusive = counts.get("inconclusive", 0)

    # Extract key convergence conjectures
    key_conjectures = {}
    for conj_id in KEY_CONVERGENCE_CONJECTURES:
        key_conjectures[conj_id] = statuses.get(conj_id, "NOT_EVALUATED")

    # Compute learning health assessment
    # HEALTHY: majority supports/consistent, no contradictions
    # MIXED: some supports, some contradicts
    # UNHEALTHY: majority contradicts or all inconclusive
    if num_contradicts == 0 and num_supports > 0:
        learning_health = "HEALTHY"
    elif num_contradicts > num_supports:
        learning_health = "UNHEALTHY"
    elif num_contradicts > 0:
        learning_health = "MIXED"
    elif num_supports == 0 and num_consistent == 0:
        learning_health = "INCONCLUSIVE"
    else:
        learning_health = "HEALTHY"

    return {
        "num_supports": num_supports,
        "num_contradicts": num_contradicts,
        "num_consistent": num_consistent,
        "num_inconclusive": num_inconclusive,
        "key_conjectures": key_conjectures,
        "learning_health": learning_health,
    }


# =============================================================================
# TASK 3: GOVERNANCE / GLOBAL HEALTH SIGNAL
# =============================================================================

def summarize_conjectures_for_governance(delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide a high-level stability indicator for governance dashboards.

    This function analyzes conjecture transitions to determine if the
    system is improving, stable, or showing warning signs.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().

    Returns:
        A governance-friendly summary containing:
        - increasing_support: True if supports count increased
        - emerging_contradictions: True if contradicts count increased
        - status: GovernanceStatus value (OK/WARN/ATTENTION)
        - status_reason: Human-readable explanation
        - transition_count: Number of status transitions
    """
    net_change = delta.get("net_change", {})
    transitions = delta.get("transitions", [])
    degraded = delta.get("degraded", [])
    improved = delta.get("improved", [])

    supports_delta = net_change.get("supports_delta", 0)
    contradicts_delta = net_change.get("contradicts_delta", 0)

    increasing_support = supports_delta > 0
    emerging_contradictions = contradicts_delta > 0

    # Determine status based on transitions
    # OK: No degradation, or net improvement
    # WARN: Some degradation but also improvement
    # ATTENTION: Net degradation or new contradictions

    if emerging_contradictions and len(degraded) > len(improved):
        status = GovernanceStatus.ATTENTION
        status_reason = f"{len(degraded)} conjectures degraded, {contradicts_delta} new contradictions"
    elif emerging_contradictions or len(degraded) > 0:
        if increasing_support or len(improved) >= len(degraded):
            status = GovernanceStatus.WARN
            status_reason = f"Mixed signals: {len(improved)} improved, {len(degraded)} degraded"
        else:
            status = GovernanceStatus.ATTENTION
            status_reason = f"Net degradation: {len(degraded)} conjectures moved toward contradiction"
    elif increasing_support or len(improved) > 0:
        status = GovernanceStatus.OK
        status_reason = f"Positive trajectory: {len(improved)} conjectures improved"
    elif len(transitions) == 0:
        status = GovernanceStatus.OK
        status_reason = "Stable: no conjecture status changes"
    else:
        status = GovernanceStatus.OK
        status_reason = f"{len(transitions)} transitions, no degradation"

    return {
        "increasing_support": increasing_support,
        "emerging_contradictions": emerging_contradictions,
        "status": status.value,
        "status_reason": status_reason,
        "transition_count": len(transitions),
        "improved_count": len(improved),
        "degraded_count": len(degraded),
    }


# =============================================================================
# PHASE III TASK 1: CONJECTURE HISTORY LEDGER
# =============================================================================

def build_conjecture_history(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a history ledger tracking how each conjecture evolved over time.

    This function analyzes a sequence of snapshots to produce trajectory
    information, stability metrics, and regression counts for each conjecture.

    Args:
        snapshots: List of snapshots from build_conjecture_snapshot(), ordered
                   chronologically (oldest first).

    Returns:
        A history dictionary containing:
        - trajectory_per_conjecture: Dict mapping conjecture_id → list of statuses over time
        - stability_index: Float 0.0-1.0 indicating overall stability (1.0 = never changed)
        - number_of_regressions: Total count of status degradations across all conjectures
        - first_snapshot_at: Timestamp of first snapshot
        - last_snapshot_at: Timestamp of last snapshot
        - num_snapshots: Number of snapshots analyzed
        - per_conjecture_stats: Dict with per-conjecture stability and regression info
    """
    if not snapshots:
        return {
            "trajectory_per_conjecture": {},
            "stability_index": 1.0,
            "number_of_regressions": 0,
            "first_snapshot_at": "",
            "last_snapshot_at": "",
            "num_snapshots": 0,
            "per_conjecture_stats": {},
        }

    # Status ranking for regression detection
    status_rank = {
        "SUPPORTS": 3,
        "CONSISTENT": 2,
        "INCONCLUSIVE": 1,
        "CONTRADICTS": 0,
        "UNKNOWN": -1,
        "NOT_EVALUATED": -1,
    }

    # Collect all conjecture IDs across all snapshots
    all_conj_ids: set = set()
    for snap in snapshots:
        all_conj_ids.update(snap.get("statuses", {}).keys())

    # Build trajectories
    trajectory_per_conjecture: Dict[str, List[str]] = {}
    for conj_id in sorted(all_conj_ids):
        trajectory = []
        for snap in snapshots:
            status = snap.get("statuses", {}).get(conj_id, "NOT_EVALUATED")
            trajectory.append(status)
        trajectory_per_conjecture[conj_id] = trajectory

    # Compute per-conjecture stats
    per_conjecture_stats: Dict[str, Dict[str, Any]] = {}
    total_transitions = 0
    total_regressions = 0

    for conj_id, trajectory in trajectory_per_conjecture.items():
        transitions = 0
        regressions = 0
        for i in range(1, len(trajectory)):
            prev_status = trajectory[i - 1]
            curr_status = trajectory[i]
            if prev_status != curr_status:
                transitions += 1
                prev_rank = status_rank.get(prev_status, -1)
                curr_rank = status_rank.get(curr_status, -1)
                if curr_rank < prev_rank:
                    regressions += 1

        # Stability for this conjecture: 1.0 if no transitions
        stability = 1.0 if len(trajectory) <= 1 else 1.0 - (transitions / (len(trajectory) - 1))

        per_conjecture_stats[conj_id] = {
            "transitions": transitions,
            "regressions": regressions,
            "stability": round(stability, 4),
            "first_status": trajectory[0] if trajectory else "UNKNOWN",
            "last_status": trajectory[-1] if trajectory else "UNKNOWN",
        }

        total_transitions += transitions
        total_regressions += regressions

    # Overall stability index
    num_conjectures = len(trajectory_per_conjecture)
    num_possible_transitions = num_conjectures * max(0, len(snapshots) - 1)
    if num_possible_transitions == 0:
        stability_index = 1.0
    else:
        stability_index = 1.0 - (total_transitions / num_possible_transitions)

    return {
        "trajectory_per_conjecture": trajectory_per_conjecture,
        "stability_index": round(stability_index, 4),
        "number_of_regressions": total_regressions,
        "first_snapshot_at": snapshots[0].get("generated_at", ""),
        "last_snapshot_at": snapshots[-1].get("generated_at", ""),
        "num_snapshots": len(snapshots),
        "per_conjecture_stats": per_conjecture_stats,
    }


# =============================================================================
# PHASE III TASK 2: GOVERNANCE COUPLING
# =============================================================================

class UpliftReadiness(Enum):
    """Uplift readiness levels based on conjecture evidence."""
    READY = "READY"
    CAUTION = "CAUTION"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def combine_conjecture_delta_with_governance(
    delta: Dict[str, Any],
    governance_posture: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine conjecture delta with governance posture for uplift decisions.

    This function integrates the epistemic state (from conjectures) with
    the operational state (from governance) to produce uplift readiness.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().
        governance_posture: A dictionary containing governance state, expected fields:
            - governance_passed: bool (overall governance check)
            - sample_size_passed: bool (optional)
            - success_rate_passed: bool (optional)
            - abstention_rate_passed: bool (optional)

    Returns:
        A coupling result containing:
        - uplift_readiness: UpliftReadiness value (READY/CAUTION/BLOCKED/INSUFFICIENT_DATA)
        - epistemic_stability: Float 0.0-1.0 based on delta
        - contradictions_of_interest: List of conjecture IDs that contradict
        - readiness_reason: Human-readable explanation
        - governance_alignment: How well conjecture state aligns with governance
    """
    # Extract delta information
    net_change = delta.get("net_change", {})
    transitions = delta.get("transitions", [])
    degraded = delta.get("degraded", [])
    improved = delta.get("improved", [])

    supports_delta = net_change.get("supports_delta", 0)
    contradicts_delta = net_change.get("contradicts_delta", 0)

    # Extract governance information
    governance_passed = governance_posture.get("governance_passed", False)

    # Compute epistemic stability
    # 1.0 = no changes, decreases with more transitions, heavily penalized for degradation
    num_transitions = len(transitions)
    num_degraded = len(degraded)

    if num_transitions == 0:
        epistemic_stability = 1.0
    else:
        # Base penalty for any transitions
        transition_penalty = min(0.3, num_transitions * 0.05)
        # Additional penalty for degradations
        degradation_penalty = min(0.5, num_degraded * 0.15)
        epistemic_stability = max(0.0, 1.0 - transition_penalty - degradation_penalty)

    epistemic_stability = round(epistemic_stability, 4)

    # Find contradictions of interest (new or existing)
    contradictions_of_interest = []
    for t in transitions:
        if t.get("to_status") == "CONTRADICTS":
            contradictions_of_interest.append(t.get("conjecture_id", "unknown"))

    # Determine uplift readiness
    if not governance_passed:
        if contradicts_delta > 0 or num_degraded > 0:
            readiness = UpliftReadiness.BLOCKED
            reason = "Governance failed and epistemic state degraded"
        elif epistemic_stability < 0.5:
            readiness = UpliftReadiness.BLOCKED
            reason = "Governance failed with unstable epistemic state"
        else:
            readiness = UpliftReadiness.CAUTION
            reason = "Governance failed but epistemic state is stable"
    else:
        # Governance passed
        if contradicts_delta > 0:
            readiness = UpliftReadiness.CAUTION
            reason = f"New contradictions emerged ({contradicts_delta})"
        elif num_degraded > len(improved):
            readiness = UpliftReadiness.CAUTION
            reason = f"Net epistemic degradation: {num_degraded} degraded vs {len(improved)} improved"
        elif supports_delta > 0 and epistemic_stability >= 0.7:
            readiness = UpliftReadiness.READY
            reason = f"Positive trajectory: {supports_delta} new supports, stable state"
        elif epistemic_stability >= 0.8:
            readiness = UpliftReadiness.READY
            reason = "Stable epistemic state with governance passed"
        elif num_transitions == 0:
            # Check if we have any data
            if supports_delta == 0 and contradicts_delta == 0:
                readiness = UpliftReadiness.READY
                reason = "Stable: no conjecture changes, governance passed"
            else:
                readiness = UpliftReadiness.READY
                reason = "Governance passed with positive signals"
        else:
            readiness = UpliftReadiness.CAUTION
            reason = f"Epistemic volatility detected ({num_transitions} transitions)"

    # Governance alignment: how well do conjectures support the governance decision
    if governance_passed and supports_delta >= 0 and contradicts_delta <= 0:
        alignment = "ALIGNED"
    elif governance_passed and contradicts_delta > 0:
        alignment = "TENSION"
    elif not governance_passed and contradicts_delta > 0:
        alignment = "ALIGNED"  # Both indicate problems
    else:
        alignment = "NEUTRAL"

    return {
        "uplift_readiness": readiness.value,
        "epistemic_stability": epistemic_stability,
        "contradictions_of_interest": contradictions_of_interest,
        "readiness_reason": reason,
        "governance_alignment": alignment,
        "governance_passed": governance_passed,
        "supports_delta": supports_delta,
        "contradicts_delta": contradicts_delta,
    }


# =============================================================================
# PHASE III TASK 3: GLOBAL HEALTH (GH) SUMMARY
# =============================================================================

def summarize_conjecture_delta_for_global_health(delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize conjecture delta for global health dashboard consumption.

    This function produces a compact summary suitable for aggregation
    across multiple experiments or slices.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().

    Returns:
        A GH summary containing:
        - uplift_signal: String (POSITIVE/NEUTRAL/NEGATIVE) indicating overall direction
        - contradictions: List of conjecture IDs that moved to CONTRADICTS
        - changed_conjectures: List of all conjecture IDs that changed status
        - signal_strength: Float 0.0-1.0 indicating confidence in signal
        - summary_text: Human-readable one-line summary
    """
    transitions = delta.get("transitions", [])
    improved = delta.get("improved", [])
    degraded = delta.get("degraded", [])
    net_change = delta.get("net_change", {})

    supports_delta = net_change.get("supports_delta", 0)
    contradicts_delta = net_change.get("contradicts_delta", 0)

    # Extract changed conjectures
    changed_conjectures = [t.get("conjecture_id", "unknown") for t in transitions]

    # Extract contradictions (conjectures that moved TO CONTRADICTS)
    contradictions = []
    for t in transitions:
        if t.get("to_status") == "CONTRADICTS":
            contradictions.append(t.get("conjecture_id", "unknown"))

    # Determine uplift signal
    if supports_delta > 0 and contradicts_delta == 0:
        uplift_signal = "POSITIVE"
    elif supports_delta > contradicts_delta and len(improved) > len(degraded):
        uplift_signal = "POSITIVE"
    elif contradicts_delta > 0 and contradicts_delta >= supports_delta:
        uplift_signal = "NEGATIVE"
    elif len(degraded) > len(improved):
        uplift_signal = "NEGATIVE"
    else:
        uplift_signal = "NEUTRAL"

    # Signal strength based on number of changes and their direction
    if len(transitions) == 0:
        signal_strength = 0.5  # Neutral - no data
    else:
        # More changes = stronger signal
        change_factor = min(1.0, len(transitions) / 4.0)
        # Consistency factor: all improved or all degraded = stronger
        if len(degraded) == 0 and len(improved) > 0:
            consistency = 1.0
        elif len(improved) == 0 and len(degraded) > 0:
            consistency = 1.0
        elif len(improved) > 0 and len(degraded) > 0:
            # Mixed signals reduce strength
            consistency = 0.5
        else:
            consistency = 0.5

        signal_strength = round(change_factor * consistency, 4)

    # Generate summary text
    if len(transitions) == 0:
        summary_text = "No conjecture status changes"
    elif uplift_signal == "POSITIVE":
        summary_text = f"{len(improved)} conjecture(s) improved, {supports_delta:+d} supports"
    elif uplift_signal == "NEGATIVE":
        summary_text = f"{len(degraded)} conjecture(s) degraded, {len(contradictions)} contradiction(s)"
    else:
        summary_text = f"{len(transitions)} transition(s): {len(improved)} improved, {len(degraded)} degraded"

    return {
        "uplift_signal": uplift_signal,
        "contradictions": contradictions,
        "changed_conjectures": changed_conjectures,
        "signal_strength": signal_strength,
        "summary_text": summary_text,
        "improved_count": len(improved),
        "degraded_count": len(degraded),
    }


# =============================================================================
# PHASE IV TASK 1: CONJECTURE-UPLIFT DECISION HELPER
# =============================================================================

class UpliftStatus(Enum):
    """Uplift status levels for decision helper."""
    OK = "OK"
    CAUTION = "CAUTION"
    BLOCK = "BLOCK"


DEFAULT_MIN_EPISTEMIC_HEALTH = 0.5


def _compute_epistemic_health_score(
    history: Dict[str, Any],
    delta: Dict[str, Any]
) -> float:
    """
    Compute an epistemic health score from history and delta.

    The score combines:
    - Stability index (40% weight): Historical conjecture consistency
    - Non-contradiction rate (30% weight): Proportion without contradictions
    - Improvement signal (30% weight): Net improvement direction

    Returns:
        Float 0.0-1.0 representing overall epistemic health.
    """
    # Stability contribution (0.4 weight)
    stability_index = history.get("stability_index", 0.5)
    stability_contribution = stability_index * 0.4

    # Non-contradiction rate (0.3 weight)
    per_conjecture_stats = history.get("per_conjecture_stats", {})
    if per_conjecture_stats:
        contradicting = sum(
            1 for stats in per_conjecture_stats.values()
            if stats.get("last_status") == "CONTRADICTS"
        )
        total = len(per_conjecture_stats)
        non_contradiction_rate = 1.0 - (contradicting / total) if total > 0 else 1.0
    else:
        non_contradiction_rate = 1.0
    non_contradiction_contribution = non_contradiction_rate * 0.3

    # Improvement signal (0.3 weight)
    improved = delta.get("improved", [])
    degraded = delta.get("degraded", [])
    total_changes = len(improved) + len(degraded)
    if total_changes > 0:
        improvement_rate = len(improved) / total_changes
    else:
        improvement_rate = 0.5  # Neutral when no changes
    improvement_contribution = improvement_rate * 0.3

    epistemic_health = round(
        stability_contribution + non_contradiction_contribution + improvement_contribution,
        4
    )
    return min(1.0, max(0.0, epistemic_health))


def evaluate_conjectures_for_uplift(
    history: Dict[str, Any],
    delta: Dict[str, Any],
    min_epistemic_health: float = DEFAULT_MIN_EPISTEMIC_HEALTH
) -> Dict[str, Any]:
    """
    Evaluate conjecture dynamics to produce an uplift decision signal.

    This function integrates historical trajectory data with recent delta
    to determine whether uplift should proceed, proceed with caution, or block.

    Args:
        history: A history dictionary from build_conjecture_history().
        delta: A delta dictionary from compare_conjecture_snapshots().
        min_epistemic_health: Minimum epistemic health score required (default 0.5).
            Uplift will BLOCK if epistemic_health_score < this threshold.

    Returns:
        An uplift evaluation containing:
        - uplift_ok: Boolean indicating if uplift can proceed
        - status: UpliftStatus value (OK/CAUTION/BLOCK)
        - blocking_conjectures: List of conjecture IDs causing blocks/caution
        - notes: Neutral rationale for the decision
        - epistemic_health_score: Float 0.0-1.0 representing epistemic health
    """
    # Compute epistemic health score
    epistemic_health_score = _compute_epistemic_health_score(history, delta)

    # Extract history metrics
    stability_index = history.get("stability_index", 1.0)
    num_regressions = history.get("number_of_regressions", 0)
    per_conjecture_stats = history.get("per_conjecture_stats", {})
    num_snapshots = history.get("num_snapshots", 0)

    # Extract delta metrics
    transitions = delta.get("transitions", [])
    degraded = delta.get("degraded", [])
    improved = delta.get("improved", [])
    net_change = delta.get("net_change", {})
    contradicts_delta = net_change.get("contradicts_delta", 0)

    # Identify blocking conjectures
    blocking_conjectures = []

    # 1. Conjectures that moved to CONTRADICTS in this delta
    for t in transitions:
        if t.get("to_status") == "CONTRADICTS":
            blocking_conjectures.append(t.get("conjecture_id", "unknown"))

    # 2. Conjectures with repeated regressions in history
    for conj_id, stats in per_conjecture_stats.items():
        if stats.get("regressions", 0) >= 2:
            if conj_id not in blocking_conjectures:
                blocking_conjectures.append(conj_id)

    # 3. Key convergence conjectures that are currently CONTRADICTS
    for conj_id in KEY_CONVERGENCE_CONJECTURES:
        stats = per_conjecture_stats.get(conj_id, {})
        if stats.get("last_status") == "CONTRADICTS":
            if conj_id not in blocking_conjectures:
                blocking_conjectures.append(conj_id)

    # Sort for determinism
    blocking_conjectures = sorted(set(blocking_conjectures))

    # Decision logic
    notes_parts = []

    # BLOCK conditions (epistemic health check is FIRST priority)
    if epistemic_health_score < min_epistemic_health and num_snapshots >= 2:
        status = UpliftStatus.BLOCK
        uplift_ok = False
        notes_parts.append(
            f"Epistemic health below threshold "
            f"({epistemic_health_score:.2f} < {min_epistemic_health:.2f})"
        )
    elif contradicts_delta >= 2:
        status = UpliftStatus.BLOCK
        uplift_ok = False
        notes_parts.append(f"{contradicts_delta} new contradictions emerged")
    elif len(blocking_conjectures) >= 3:
        status = UpliftStatus.BLOCK
        uplift_ok = False
        notes_parts.append(f"{len(blocking_conjectures)} conjectures flagged as blocking")
    elif stability_index < 0.3 and num_snapshots >= 3:
        status = UpliftStatus.BLOCK
        uplift_ok = False
        notes_parts.append(f"Highly unstable conjecture history (stability={stability_index:.2f})")
    # CAUTION conditions
    elif contradicts_delta > 0:
        status = UpliftStatus.CAUTION
        uplift_ok = True
        notes_parts.append(f"{contradicts_delta} new contradiction(s) require attention")
    elif len(degraded) > len(improved):
        status = UpliftStatus.CAUTION
        uplift_ok = True
        notes_parts.append(f"Net degradation: {len(degraded)} degraded vs {len(improved)} improved")
    elif num_regressions > 0 and stability_index < 0.6:
        status = UpliftStatus.CAUTION
        uplift_ok = True
        notes_parts.append(f"{num_regressions} historical regression(s) with moderate instability")
    elif len(blocking_conjectures) > 0:
        status = UpliftStatus.CAUTION
        uplift_ok = True
        notes_parts.append(f"{len(blocking_conjectures)} conjecture(s) require monitoring")
    elif epistemic_health_score < min_epistemic_health + 0.1:
        # Near threshold - caution
        status = UpliftStatus.CAUTION
        uplift_ok = True
        notes_parts.append(
            f"Epistemic health near threshold ({epistemic_health_score:.2f})"
        )
    # OK conditions
    else:
        status = UpliftStatus.OK
        uplift_ok = True
        if len(improved) > 0:
            notes_parts.append(f"{len(improved)} conjecture(s) improved")
        if stability_index >= 0.8:
            notes_parts.append("High stability in conjecture history")
        if epistemic_health_score >= 0.8:
            notes_parts.append(f"Strong epistemic health ({epistemic_health_score:.2f})")
        if not notes_parts:
            notes_parts.append("No blocking conditions detected")

    notes = "; ".join(notes_parts) if notes_parts else "Stable conjecture dynamics"

    return {
        "uplift_ok": uplift_ok,
        "status": status.value,
        "blocking_conjectures": blocking_conjectures,
        "notes": notes,
        "stability_index": stability_index,
        "num_regressions": num_regressions,
        "contradicts_delta": contradicts_delta,
        "epistemic_health_score": epistemic_health_score,
        "min_epistemic_health": min_epistemic_health,
    }


# =============================================================================
# PHASE IV TASK 2: MAAS EPISTEMIC ADAPTER
# =============================================================================

class MaasStatus(Enum):
    """MAAS status levels."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


def summarize_conjectures_for_maas(
    delta: Dict[str, Any],
    uplift_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize conjecture state for MAAS (Meta-Agent Awareness System) consumption.

    This function produces a compact epistemic signal suitable for
    meta-controller decision making.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().
        uplift_eval: An evaluation dictionary from evaluate_conjectures_for_uplift().

    Returns:
        A MAAS-friendly summary containing:
        - epistemic_signal: POSITIVE/NEUTRAL/NEGATIVE
        - uplift_ready: Boolean
        - contradictions_of_interest: List of conjecture IDs
        - status: MaasStatus value (OK/ATTENTION/BLOCK)
    """
    # Extract delta information
    net_change = delta.get("net_change", {})
    transitions = delta.get("transitions", [])
    improved = delta.get("improved", [])
    degraded = delta.get("degraded", [])

    supports_delta = net_change.get("supports_delta", 0)
    contradicts_delta = net_change.get("contradicts_delta", 0)

    # Extract uplift evaluation
    uplift_ok = uplift_eval.get("uplift_ok", False)
    uplift_status = uplift_eval.get("status", "CAUTION")
    blocking_conjectures = uplift_eval.get("blocking_conjectures", [])

    # Determine epistemic signal
    if supports_delta > 0 and contradicts_delta == 0 and len(degraded) == 0:
        epistemic_signal = "POSITIVE"
    elif contradicts_delta > 0 or len(degraded) > len(improved):
        epistemic_signal = "NEGATIVE"
    else:
        epistemic_signal = "NEUTRAL"

    # Extract contradictions of interest
    contradictions_of_interest = []
    for t in transitions:
        if t.get("to_status") == "CONTRADICTS":
            contradictions_of_interest.append(t.get("conjecture_id", "unknown"))
    # Also include blocking conjectures that are contradictions
    for conj_id in blocking_conjectures:
        if conj_id not in contradictions_of_interest:
            contradictions_of_interest.append(conj_id)
    contradictions_of_interest = sorted(set(contradictions_of_interest))

    # Determine MAAS status based on uplift evaluation
    if uplift_status == "BLOCK":
        maas_status = MaasStatus.BLOCK
    elif uplift_status == "CAUTION" or len(contradictions_of_interest) > 0:
        maas_status = MaasStatus.ATTENTION
    else:
        maas_status = MaasStatus.OK

    return {
        "epistemic_signal": epistemic_signal,
        "uplift_ready": uplift_ok,
        "contradictions_of_interest": contradictions_of_interest,
        "status": maas_status.value,
        "supports_delta": supports_delta,
        "contradicts_delta": contradicts_delta,
        "improved_count": len(improved),
        "degraded_count": len(degraded),
    }


# =============================================================================
# PHASE IV TASK 3: DIRECTOR CONJECTURE PANEL
# =============================================================================

class StatusLight(Enum):
    """Status light indicators for Director panel."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    GRAY = "GRAY"


def build_conjecture_director_panel(
    global_health_conjectures: Dict[str, Any],
    uplift_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a conjecture panel for the Director dashboard.

    This function produces a visual-ready summary of conjecture dynamics
    suitable for display as a Director tile.

    Args:
        global_health_conjectures: A summary from summarize_conjecture_delta_for_global_health().
        uplift_eval: An evaluation dictionary from evaluate_conjectures_for_uplift().

    Returns:
        A Director panel containing:
        - status_light: StatusLight value (GREEN/YELLOW/RED/GRAY)
        - uplift_signal: POSITIVE/NEUTRAL/NEGATIVE
        - signal_strength: Float 0.0-1.0
        - headline: Neutral summary of current conjecture dynamics
    """
    # Extract global health info
    uplift_signal = global_health_conjectures.get("uplift_signal", "NEUTRAL")
    signal_strength = global_health_conjectures.get("signal_strength", 0.5)
    summary_text = global_health_conjectures.get("summary_text", "")
    improved_count = global_health_conjectures.get("improved_count", 0)
    degraded_count = global_health_conjectures.get("degraded_count", 0)
    contradictions = global_health_conjectures.get("contradictions", [])

    # Extract uplift evaluation
    uplift_status = uplift_eval.get("status", "OK")
    uplift_ok = uplift_eval.get("uplift_ok", True)
    blocking_count = len(uplift_eval.get("blocking_conjectures", []))
    stability_index = uplift_eval.get("stability_index", 1.0)

    # Determine status light
    if uplift_status == "BLOCK":
        status_light = StatusLight.RED
    elif uplift_status == "CAUTION":
        status_light = StatusLight.YELLOW
    elif uplift_signal == "POSITIVE" and uplift_ok:
        status_light = StatusLight.GREEN
    elif uplift_signal == "NEGATIVE":
        status_light = StatusLight.YELLOW
    elif signal_strength < 0.3:
        status_light = StatusLight.GRAY  # Insufficient signal
    else:
        status_light = StatusLight.GREEN

    # Build headline
    headline_parts = []

    if uplift_status == "BLOCK":
        headline_parts.append(f"Blocked: {blocking_count} issue(s)")
    elif len(contradictions) > 0:
        headline_parts.append(f"{len(contradictions)} contradiction(s)")

    if improved_count > 0 and degraded_count == 0:
        headline_parts.append(f"{improved_count} improved")
    elif degraded_count > 0 and improved_count == 0:
        headline_parts.append(f"{degraded_count} degraded")
    elif improved_count > 0 and degraded_count > 0:
        headline_parts.append(f"{improved_count}↑ {degraded_count}↓")

    if stability_index >= 0.9:
        headline_parts.append("Highly stable")
    elif stability_index < 0.5:
        headline_parts.append("Unstable history")

    if not headline_parts:
        if uplift_signal == "POSITIVE":
            headline = "Positive trajectory"
        elif uplift_signal == "NEGATIVE":
            headline = "Negative trajectory"
        else:
            headline = "Stable - no significant changes"
    else:
        headline = "; ".join(headline_parts)

    return {
        "status_light": status_light.value,
        "uplift_signal": uplift_signal,
        "signal_strength": signal_strength,
        "headline": headline,
        "uplift_ok": uplift_ok,
        "blocking_count": blocking_count,
        "improved_count": improved_count,
        "degraded_count": degraded_count,
    }


# =============================================================================
# PHASE V TASK 1: GLOBAL CONSOLE ADAPTER
# =============================================================================

# Schema version for Global Console integration
GLOBAL_CONSOLE_SCHEMA_VERSION = "1.0.0"

# Required fields for Global Console tile
CONSOLE_TILE_REQUIRED_FIELDS = frozenset([
    "status_light",
    "headline",
    "epistemic_ok",
])

# Signal values (canonical)
SIGNAL_POSITIVE = "POSITIVE"
SIGNAL_NEUTRAL = "NEUTRAL"
SIGNAL_NEGATIVE = "NEGATIVE"
VALID_SIGNALS = frozenset([SIGNAL_POSITIVE, SIGNAL_NEUTRAL, SIGNAL_NEGATIVE])


def summarize_conjectures_for_global_console(
    delta: Dict[str, Any],
    uplift_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize conjecture state for Global Console display.

    This function produces a compact summary suitable for the global
    console dashboard with clear epistemic health indicators.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().
        uplift_eval: An evaluation dictionary from evaluate_conjectures_for_uplift().

    Returns:
        A Global Console summary containing:
        - epistemic_ok: Boolean indicating overall epistemic health
        - signal: "POSITIVE" | "NEUTRAL" | "NEGATIVE"
        - status_light: StatusLight value (GREEN/YELLOW/RED/GRAY)
        - headline: Human-readable summary
    """
    # Extract delta information
    net_change = delta.get("net_change", {})
    transitions = delta.get("transitions", [])
    improved = delta.get("improved", [])
    degraded = delta.get("degraded", [])

    supports_delta = net_change.get("supports_delta", 0)
    contradicts_delta = net_change.get("contradicts_delta", 0)

    # Extract uplift evaluation
    uplift_ok = uplift_eval.get("uplift_ok", False)
    uplift_status = uplift_eval.get("status", "CAUTION")
    epistemic_health_score = uplift_eval.get("epistemic_health_score", 0.5)
    min_epistemic_health = uplift_eval.get("min_epistemic_health", DEFAULT_MIN_EPISTEMIC_HEALTH)
    blocking_conjectures = uplift_eval.get("blocking_conjectures", [])

    # Determine epistemic_ok
    epistemic_ok = (
        epistemic_health_score >= min_epistemic_health
        and uplift_status != "BLOCK"
        and contradicts_delta < 2
    )

    # Determine signal
    if supports_delta > 0 and contradicts_delta == 0 and len(degraded) == 0:
        signal = "POSITIVE"
    elif contradicts_delta > 0 or len(degraded) > len(improved):
        signal = "NEGATIVE"
    else:
        signal = "NEUTRAL"

    # Determine status light
    if uplift_status == "BLOCK":
        status_light = StatusLight.RED
    elif uplift_status == "CAUTION":
        status_light = StatusLight.YELLOW
    elif signal == "POSITIVE" and epistemic_ok:
        status_light = StatusLight.GREEN
    elif signal == "NEGATIVE":
        status_light = StatusLight.YELLOW
    elif len(transitions) == 0:
        status_light = StatusLight.GRAY  # No data
    else:
        status_light = StatusLight.GREEN

    # Build headline
    headline_parts = []

    if not epistemic_ok:
        if epistemic_health_score < min_epistemic_health:
            headline_parts.append(f"Epistemic health low ({epistemic_health_score:.0%})")
        elif uplift_status == "BLOCK":
            headline_parts.append("Uplift blocked")

    if contradicts_delta > 0:
        headline_parts.append(f"{contradicts_delta} contradiction(s)")

    if len(blocking_conjectures) > 0 and not headline_parts:
        headline_parts.append(f"{len(blocking_conjectures)} issue(s) flagged")

    if len(improved) > 0 and signal == "POSITIVE":
        headline_parts.append(f"{len(improved)} improved")

    if not headline_parts:
        if signal == "POSITIVE":
            headline = "Epistemic health good; positive trajectory"
        elif signal == "NEGATIVE":
            headline = "Negative trajectory observed"
        else:
            headline = "Stable epistemic state"
    else:
        headline = "; ".join(headline_parts)

    return {
        "epistemic_ok": epistemic_ok,
        "signal": signal,
        "status_light": status_light.value,
        "headline": headline,
        "epistemic_health_score": epistemic_health_score,
        "supports_delta": supports_delta,
        "contradicts_delta": contradicts_delta,
        "improved_count": len(improved),
        "degraded_count": len(degraded),
        "_schema_version": GLOBAL_CONSOLE_SCHEMA_VERSION,
    }


def build_epistemic_console_tile(
    delta: Dict[str, Any],
    uplift_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a slim epistemic tile for Global Console direct display.

    This is the CANONICAL tile format for displaying epistemic health
    on the Global Console dashboard. It contains only the fields required
    for visual display with no extraneous diagnostic data.

    SCHEMA CONTRACT (v1.0.0):
    -------------------------
    This tile is designed to be directly renderable by the Global Console
    without transformation. All consumers should validate against
    CONSOLE_TILE_REQUIRED_FIELDS.

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().
        uplift_eval: An evaluation dictionary from evaluate_conjectures_for_uplift().

    Returns:
        A minimal tile dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED" | "GRAY"
        - headline: Short human-readable summary (max ~50 chars)
        - epistemic_ok: Boolean for quick pass/fail checks
        - health_pct: Integer 0-100 for progress bar display
        - tile_id: Canonical identifier "epistemic_health"
    """
    # Get full summary and extract minimal fields
    full_summary = summarize_conjectures_for_global_console(delta, uplift_eval)

    # Convert health score to percentage for display
    health_pct = int(round(full_summary["epistemic_health_score"] * 100))

    # Truncate headline if too long
    headline = full_summary["headline"]
    if len(headline) > 50:
        headline = headline[:47] + "..."

    return {
        "tile_id": "epistemic_health",
        "status_light": full_summary["status_light"],
        "headline": headline,
        "epistemic_ok": full_summary["epistemic_ok"],
        "health_pct": health_pct,
        "_schema_version": GLOBAL_CONSOLE_SCHEMA_VERSION,
    }


# =============================================================================
# PHASE V TASK 2: GOVERNANCE SIGNAL FOR CLAUDE I (CANONICAL ADAPTER)
# =============================================================================

# Schema version for CLAUDE I governance signal
CLAUDE_I_SCHEMA_VERSION = "1.0.0"

# Governance signal level priorities (for sorting/comparison)
GOVERNANCE_LEVEL_PRIORITY = {
    "CLEAR": 0,
    "ADVISORY": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}

# Required fields for CLAUDE I governance signal
CLAUDE_I_REQUIRED_FIELDS = frozenset([
    "level",
    "uplift_status",
    "epistemic_health_score",
    "contradictions_of_interest",
    "recommended_action",
    "uplift_ok",
])


class GovernanceSignalLevel(Enum):
    """
    Governance signal levels for CLAUDE I.

    These levels form a strict hierarchy for epistemic governance:
    - CLEAR: All systems nominal, proceed with normal operations
    - ADVISORY: Minor concerns noted, proceed with awareness
    - WARNING: Significant concerns, exercise caution before proceeding
    - CRITICAL: Major issues detected, recommend halt until resolved

    Level transitions should follow the hierarchy - a system cannot
    return CLEAR if any blocking condition exists.
    """
    CLEAR = "CLEAR"
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str) -> "GovernanceSignalLevel":
        """Parse a string to GovernanceSignalLevel."""
        try:
            return cls(value)
        except ValueError:
            return cls.WARNING  # Default to WARNING on unknown

    @property
    def priority(self) -> int:
        """Return priority level (higher = more severe)."""
        return GOVERNANCE_LEVEL_PRIORITY.get(self.value, 2)

    def __lt__(self, other: "GovernanceSignalLevel") -> bool:
        return self.priority < other.priority

    def __le__(self, other: "GovernanceSignalLevel") -> bool:
        return self.priority <= other.priority


def get_governance_signal_for_claude_i(
    delta: Dict[str, Any],
    uplift_eval: Dict[str, Any],
    global_console: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a normalized governance signal for CLAUDE I consumption.

    CANONICAL ADAPTER (v1.0.0)
    ==========================
    This is the CANONICAL governance signal interface for CLAUDE I.
    All epistemic governance decisions should flow through this adapter.

    The signal provides a structured, schema-validated output that CLAUDE I
    can use to make informed decisions about uplift governance.

    SCHEMA CONTRACT:
    ----------------
    Required fields are defined in CLAUDE_I_REQUIRED_FIELDS. All signals
    MUST include these fields. Consumers should validate using:

        assert CLAUDE_I_REQUIRED_FIELDS.issubset(signal.keys())

    SIGNAL HIERARCHY:
    -----------------
    Levels follow strict priority ordering (CLEAR < ADVISORY < WARNING < CRITICAL).
    The signal level is determined by the most severe condition present:

    | Condition                          | Level     |
    |------------------------------------|-----------|
    | BLOCK + key conjecture contradict  | CRITICAL  |
    | BLOCK (any reason)                 | WARNING   |
    | Key conjecture contradiction       | WARNING   |
    | CAUTION + low epistemic health     | WARNING   |
    | CAUTION + good epistemic health    | ADVISORY  |
    | Any non-key contradiction          | ADVISORY  |
    | All clear                          | CLEAR     |

    Args:
        delta: A delta dictionary from compare_conjecture_snapshots().
        uplift_eval: An evaluation dictionary from evaluate_conjectures_for_uplift().
        global_console: Optional global console summary (computed if not provided).

    Returns:
        A governance signal containing:
        - level: GovernanceSignalLevel value (CLEAR/ADVISORY/WARNING/CRITICAL)
        - uplift_status: Current uplift status (OK/CAUTION/BLOCK)
        - uplift_ok: Boolean indicating if uplift can proceed
        - epistemic_health_score: Float 0.0-1.0
        - contradictions_of_interest: List of key contradicting conjectures
        - key_conjecture_status: Status of each KEY_CONVERGENCE_CONJECTURE
        - recommended_action: Human-readable action guidance
        - details: Diagnostic information for debugging
        - _schema_version: Schema version identifier
    """
    # Compute global console if not provided
    if global_console is None:
        global_console = summarize_conjectures_for_global_console(delta, uplift_eval)

    # Extract core metrics
    uplift_status = uplift_eval.get("status", "CAUTION")
    uplift_ok = uplift_eval.get("uplift_ok", False)
    epistemic_health_score = uplift_eval.get("epistemic_health_score", 0.5)
    blocking_conjectures = uplift_eval.get("blocking_conjectures", [])
    stability_index = uplift_eval.get("stability_index", 1.0)

    net_change = delta.get("net_change", {})
    contradicts_delta = net_change.get("contradicts_delta", 0)
    transitions = delta.get("transitions", [])

    # Extract contradictions of interest (focusing on key conjectures)
    contradictions_of_interest = []
    all_contradictions = []

    for t in transitions:
        if t.get("to_status") == "CONTRADICTS":
            conj_id = t.get("conjecture_id", "unknown")
            all_contradictions.append(conj_id)
            # Prioritize key convergence conjectures
            if conj_id in KEY_CONVERGENCE_CONJECTURES:
                contradictions_of_interest.append(conj_id)

    # Add blocking key conjectures
    for conj_id in blocking_conjectures:
        if conj_id in KEY_CONVERGENCE_CONJECTURES and conj_id not in contradictions_of_interest:
            contradictions_of_interest.append(conj_id)

    contradictions_of_interest = sorted(set(contradictions_of_interest))

    # Build key conjecture status
    key_conjecture_status = {}
    per_conjecture_stats = uplift_eval.get("per_conjecture_stats", {})
    # Try to get from history if available in the function context
    # For now, check blocking conjectures
    for conj_id in KEY_CONVERGENCE_CONJECTURES:
        if conj_id in blocking_conjectures:
            key_conjecture_status[conj_id] = "BLOCKED"
        elif conj_id in contradictions_of_interest:
            key_conjecture_status[conj_id] = "CONTRADICTS"
        else:
            key_conjecture_status[conj_id] = "OK"

    # Determine governance signal level
    has_key_contradiction = len(contradictions_of_interest) > 0
    has_any_contradiction = contradicts_delta > 0

    if uplift_status == "BLOCK":
        if has_key_contradiction:
            level = GovernanceSignalLevel.CRITICAL
            recommended_action = "Halt uplift; investigate key conjecture contradictions"
        else:
            level = GovernanceSignalLevel.WARNING
            recommended_action = "Halt uplift; review blocking conditions"
    elif has_key_contradiction:
        level = GovernanceSignalLevel.WARNING
        recommended_action = "Review contradictions in key convergence conjectures before proceeding"
    elif uplift_status == "CAUTION":
        if epistemic_health_score < 0.5:
            level = GovernanceSignalLevel.WARNING
            recommended_action = "Exercise caution; epistemic health below optimal"
        else:
            level = GovernanceSignalLevel.ADVISORY
            recommended_action = "Proceed with awareness of flagged issues"
    elif has_any_contradiction:
        level = GovernanceSignalLevel.ADVISORY
        recommended_action = "Monitor non-key conjecture contradictions"
    else:
        level = GovernanceSignalLevel.CLEAR
        recommended_action = "Proceed with normal operations"

    # Build details
    details = {
        "total_contradictions": contradicts_delta,
        "key_contradictions": len(contradictions_of_interest),
        "non_key_contradictions": len(all_contradictions) - len(contradictions_of_interest),
        "blocking_conjectures_count": len(blocking_conjectures),
        "stability_index": stability_index,
        "signal_from_console": global_console.get("signal", "NEUTRAL"),
    }

    return {
        "level": level.value,
        "uplift_status": uplift_status,
        "epistemic_health_score": epistemic_health_score,
        "contradictions_of_interest": contradictions_of_interest,
        "key_conjecture_status": key_conjecture_status,
        "recommended_action": recommended_action,
        "details": details,
        "uplift_ok": uplift_ok,
        "_schema_version": CLAUDE_I_SCHEMA_VERSION,
        "_adapter_id": "conjecture_engine.governance_signal",
    }


def validate_governance_signal(signal: Dict[str, Any]) -> bool:
    """
    Validate that a governance signal conforms to the CLAUDE I schema.

    Args:
        signal: A dictionary to validate.

    Returns:
        True if signal is valid, False otherwise.
    """
    return CLAUDE_I_REQUIRED_FIELDS.issubset(signal.keys())


def validate_console_tile(tile: Dict[str, Any]) -> bool:
    """
    Validate that a console tile conforms to the Global Console schema.

    Args:
        tile: A dictionary to validate.

    Returns:
        True if tile is valid, False otherwise.
    """
    return CONSOLE_TILE_REQUIRED_FIELDS.issubset(tile.keys())
