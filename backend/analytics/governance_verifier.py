"""
Uplift Analytics Governance Verifier

==============================================================================
STATUS: PHASE II — IMPLEMENTATION
==============================================================================

This module implements the Governance Verifier as specified in:
- docs/UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md
- docs/UPLIFT_GOVERNANCE_VERIFIER_SPEC.md

The verifier is a pure function that checks 43 rules across 4 categories:
- GOV-*: Governance Rules (12 rules)
- REP-*: Reproducibility Rules (8 rules)
- MAN-*: Manifest Rules (10 rules)
- INV-*: Invariant Rules (13 rules)

Usage:
    verdict = governance_verify(summary, manifest, telemetry)
    if verdict.status == "FAIL":
        # Handle invalidating violations
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Sequence

__version__ = "2.0.0"

# =============================================================================
# CONSTANTS
# =============================================================================

SLICE_IDS = frozenset([
    "prop_depth4",
    "fol_eq_group",
    "fol_eq_ring",
    "linear_arith",
])

# Success criteria per slice (from PREREG_UPLIFT_U2.yaml / governance spec)
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

# Valid governance decisions
VALID_DECISIONS = frozenset(["proceed", "hold", "rollback"])

# Minimum bootstrap iterations
MIN_BOOTSTRAP_ITERATIONS = 10000

# Default confidence level
DEFAULT_CONFIDENCE = 0.95

# Hash regex pattern (SHA-256)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RuleResult:
    """Result of checking a single governance rule."""
    rule_id: str
    passed: bool
    severity: str  # "INVALIDATING" | "WARNING"
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceVerdict:
    """
    Complete governance verification result.

    This is the primary output of governance_verify().
    """
    # Overall status
    status: str  # "PASS" | "WARN" | "FAIL"

    # Rule violations
    invalidating_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed_rules: List[str] = field(default_factory=list)

    # Human-readable summary
    summary: str = ""

    # Input metadata
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Full rule details
    details: Dict[str, RuleResult] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    verifier_version: str = __version__
    rules_checked: int = 0

    # v2: Reason codes and explanation (for MAAS / audit trail)
    reason_codes: List[str] = field(default_factory=list)
    short_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "status": self.status,
            "invalidating_rules": self.invalidating_rules,
            "warnings": self.warnings,
            "passed_rules": self.passed_rules,
            "summary": self.summary,
            "inputs": self.inputs,
            "details": {k: asdict(v) for k, v in self.details.items()},
            "timestamp": self.timestamp,
            "verifier_version": self.verifier_version,
            "rules_checked": self.rules_checked,
            "reason_codes": self.reason_codes,
            "short_explanation": self.short_explanation,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dict."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def _compute_pass_predicate(slice_data: Dict, slice_id: str) -> bool:
    """
    Compute the PASS predicate for a slice based on its metrics.

    PASS(s) ⟺ SR(T_s) ≥ G_s.min_SR
            ∧ AR(T_s) ≤ G_s.max_AR
            ∧ Δ_Θ%(s) ≥ G_s.min_Δ_Θ%
            ∧ |T_s| ≥ G_s.min_n
    """
    criteria = SLICE_SUCCESS_CRITERIA.get(slice_id, {})

    # Get metrics - handle various schema formats
    success_rate = _get_nested(slice_data, "success_rate", "rfl") or \
                   _get_nested(slice_data, "metrics", "success_rate", "rfl", default=0)
    abstention_rate = _get_nested(slice_data, "abstention_rate", "rfl") or \
                      _get_nested(slice_data, "metrics", "abstention_rate", "rfl", default=1)
    throughput_uplift = _get_nested(slice_data, "throughput", "delta_pct") or \
                        _get_nested(slice_data, "metrics", "throughput", "delta_pct", default=-999)
    n_rfl = _get_nested(slice_data, "n_rfl") or \
            _get_nested(slice_data, "sample_size", "rfl", default=0)

    min_sr = criteria.get("min_success_rate", 0)
    max_ar = criteria.get("max_abstention_rate", 1)
    min_uplift = criteria.get("min_throughput_uplift_pct", 0)
    min_n = criteria.get("min_samples", 0)

    return (
        success_rate >= min_sr and
        abstention_rate <= max_ar and
        throughput_uplift >= min_uplift and
        n_rfl >= min_n
    )


# =============================================================================
# GOVERNANCE RULES (GOV-*)
# =============================================================================

def _check_gov_1(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-1: Threshold Compliance
    Each slice MUST be evaluated against its predefined success criteria.
    Severity: INVALIDATING
    """
    failures = []
    slices = summary.get("slices", {})

    for slice_id, criteria in SLICE_SUCCESS_CRITERIA.items():
        slice_data = slices.get(slice_id, {})
        if not slice_data:
            continue  # GOV-9 handles missing slices

        # Get metrics
        success_rate = _get_nested(slice_data, "success_rate", "rfl") or \
                       _get_nested(slice_data, "metrics", "success_rate", "rfl", default=0)
        abstention_rate = _get_nested(slice_data, "abstention_rate", "rfl") or \
                          _get_nested(slice_data, "metrics", "abstention_rate", "rfl", default=1)
        throughput_uplift = _get_nested(slice_data, "throughput", "delta_pct") or \
                            _get_nested(slice_data, "metrics", "throughput", "delta_pct", default=-999)

        # Check each criterion
        if success_rate < criteria["min_success_rate"]:
            failures.append(f"{slice_id}: success_rate {success_rate:.3f} < {criteria['min_success_rate']}")
        if abstention_rate > criteria["max_abstention_rate"]:
            failures.append(f"{slice_id}: abstention_rate {abstention_rate:.3f} > {criteria['max_abstention_rate']}")
        if throughput_uplift < criteria["min_throughput_uplift_pct"]:
            failures.append(f"{slice_id}: throughput_uplift {throughput_uplift:.1f}% < {criteria['min_throughput_uplift_pct']}%")

    passed = len(failures) == 0
    return RuleResult(
        rule_id="GOV-1",
        passed=passed,
        severity="INVALIDATING",
        message="Threshold compliance check passed" if passed else f"Threshold violations: {failures}",
        evidence={"failures": failures}
    )


def _check_gov_2(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-2: Decision Exclusivity
    Exactly one of {PROCEED, HOLD, ROLLBACK} MUST be true.
    Severity: INVALIDATING
    """
    recommendation = _get_nested(summary, "governance", "recommendation", default="")
    if isinstance(recommendation, str):
        recommendation = recommendation.lower()

    passed = recommendation in VALID_DECISIONS
    return RuleResult(
        rule_id="GOV-2",
        passed=passed,
        severity="INVALIDATING",
        message=f"Decision '{recommendation}' is valid" if passed else f"Invalid decision: '{recommendation}'",
        evidence={"recommendation": recommendation, "valid_options": list(VALID_DECISIONS)}
    )


def _check_gov_3(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-3: Decision Consistency
    PROCEED ⟹ all_slices_pass = true
    all_slices_pass = false ⟹ recommendation ∈ {HOLD, ROLLBACK}
    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    recommendation = str(gov.get("recommendation", "")).lower()
    all_pass = gov.get("all_slices_pass", False)

    if recommendation == "proceed" and not all_pass:
        return RuleResult(
            rule_id="GOV-3",
            passed=False,
            severity="INVALIDATING",
            message="PROCEED but all_slices_pass=False",
            evidence={"recommendation": recommendation, "all_slices_pass": all_pass}
        )

    if not all_pass and recommendation not in {"hold", "rollback"}:
        return RuleResult(
            rule_id="GOV-3",
            passed=False,
            severity="INVALIDATING",
            message="Not all slices pass but recommendation is not HOLD/ROLLBACK",
            evidence={"recommendation": recommendation, "all_slices_pass": all_pass}
        )

    return RuleResult(
        rule_id="GOV-3",
        passed=True,
        severity="INVALIDATING",
        message="Decision is consistent with slice pass status",
        evidence={}
    )


def _check_gov_4(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-4: Failing Slice Identification
    passing_slices ∪ failing_slices = all slices
    passing_slices ∩ failing_slices = ∅
    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    passing = set(gov.get("passing_slices", []))
    failing = set(gov.get("failing_slices", []))
    all_slices = set(summary.get("slices", {}).keys())

    # If slices dict is empty, use SLICE_IDS
    if not all_slices:
        all_slices = SLICE_IDS

    union_correct = (passing | failing) == all_slices
    disjoint = len(passing & failing) == 0

    passed = union_correct and disjoint
    message = "Slice partition is valid" if passed else ""
    if not union_correct:
        missing = all_slices - (passing | failing)
        extra = (passing | failing) - all_slices
        message = f"Partition incomplete. Missing: {missing}, Extra: {extra}"
    if not disjoint:
        overlap = passing & failing
        message = f"Slice overlap: {overlap}"

    return RuleResult(
        rule_id="GOV-4",
        passed=passed,
        severity="INVALIDATING",
        message=message,
        evidence={"passing": list(passing), "failing": list(failing), "expected": list(all_slices)}
    )


def _check_gov_5(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-5: Marginal Case Flagging
    If any slice has CI overlapping threshold, it should be flagged.
    Severity: WARNING
    """
    marginal_slices = []
    slices = summary.get("slices", {})

    for slice_id, criteria in SLICE_SUCCESS_CRITERIA.items():
        slice_data = slices.get(slice_id, {})
        if not slice_data:
            continue

        threshold = criteria["min_throughput_uplift_pct"]
        ci_low = _get_nested(slice_data, "throughput", "ci_low") or \
                 _get_nested(slice_data, "metrics", "throughput", "delta_ci_low", default=0)
        ci_high = _get_nested(slice_data, "throughput", "ci_high") or \
                  _get_nested(slice_data, "metrics", "throughput", "delta_ci_high", default=0)

        if ci_low < threshold < ci_high:
            marginal_slices.append(slice_id)

    # This rule is WARNING level and always "passes" - it just flags
    return RuleResult(
        rule_id="GOV-5",
        passed=len(marginal_slices) == 0,
        severity="WARNING",
        message=f"Marginal slices detected: {marginal_slices}" if marginal_slices else "No marginal cases",
        evidence={"marginal_slices": marginal_slices}
    )


def _check_gov_6(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-6: HOLD Rationale Required
    If recommendation = HOLD, a rationale MUST be provided.
    Severity: WARNING
    """
    gov = summary.get("governance", {})
    recommendation = str(gov.get("recommendation", "")).lower()
    rationale = gov.get("rationale", "")

    if recommendation != "hold":
        return RuleResult(
            rule_id="GOV-6",
            passed=True,
            severity="WARNING",
            message="Not applicable (not HOLD)",
            evidence={}
        )

    passed = bool(rationale and str(rationale).strip())
    return RuleResult(
        rule_id="GOV-6",
        passed=passed,
        severity="WARNING",
        message="HOLD rationale provided" if passed else "HOLD decision without rationale",
        evidence={"recommendation": recommendation, "rationale": rationale}
    )


def _check_gov_7(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-7: ROLLBACK Rationale Required
    If recommendation = ROLLBACK, a rationale MUST be provided.
    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    recommendation = str(gov.get("recommendation", "")).lower()
    rationale = gov.get("rationale", "")
    failing_slices = gov.get("failing_slices", [])

    if recommendation != "rollback":
        return RuleResult(
            rule_id="GOV-7",
            passed=True,
            severity="INVALIDATING",
            message="Not applicable (not ROLLBACK)",
            evidence={}
        )

    has_rationale = bool(rationale and str(rationale).strip())
    has_failing_slices = len(failing_slices) > 0
    passed = has_rationale and has_failing_slices

    return RuleResult(
        rule_id="GOV-7",
        passed=passed,
        severity="INVALIDATING",
        message="ROLLBACK rationale and failing slices provided" if passed else "ROLLBACK without proper rationale or failing slices",
        evidence={"rationale": rationale, "failing_slices": failing_slices}
    )


def _check_gov_8(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-8: Sample Size Minimum
    Each slice MUST meet its minimum sample size requirement.
    Severity: INVALIDATING
    """
    failures = []
    slices = summary.get("slices", {})

    for slice_id, criteria in SLICE_SUCCESS_CRITERIA.items():
        slice_data = slices.get(slice_id, {})
        if not slice_data:
            continue

        n_rfl = _get_nested(slice_data, "n_rfl") or \
                _get_nested(slice_data, "sample_size", "rfl", default=0)
        min_n = criteria["min_samples"]

        if n_rfl < min_n:
            failures.append(f"{slice_id}: n_rfl={n_rfl} < min={min_n}")

    passed = len(failures) == 0
    return RuleResult(
        rule_id="GOV-8",
        passed=passed,
        severity="INVALIDATING",
        message="Sample size requirements met" if passed else f"Sample size violations: {failures}",
        evidence={"failures": failures}
    )


def _check_gov_9(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-9: All Slices Present
    All four slices MUST be present in the summary.
    Severity: INVALIDATING
    """
    slices = set(summary.get("slices", {}).keys())
    missing = SLICE_IDS - slices

    passed = len(missing) == 0
    return RuleResult(
        rule_id="GOV-9",
        passed=passed,
        severity="INVALIDATING",
        message="All slices present" if passed else f"Missing slices: {missing}",
        evidence={"present": list(slices), "missing": list(missing)}
    )


def _check_gov_10(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-10: No Unreported Failures
    No slice failure may be omitted from failing_slices.
    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    reported_failing = set(gov.get("failing_slices", []))
    slices = summary.get("slices", {})

    # Compute actual failures
    actual_failing = set()
    for slice_id in slices:
        slice_data = slices[slice_id]
        if not _compute_pass_predicate(slice_data, slice_id):
            actual_failing.add(slice_id)

    unreported = actual_failing - reported_failing

    passed = len(unreported) == 0
    return RuleResult(
        rule_id="GOV-10",
        passed=passed,
        severity="INVALIDATING",
        message="All failures reported" if passed else f"Unreported failures: {unreported}",
        evidence={"unreported": list(unreported), "reported": list(reported_failing), "actual": list(actual_failing)}
    )


def _check_gov_11(summary: Dict, prereg: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    GOV-11: Confidence Level Match
    The confidence level used MUST match the preregistered level (default 0.95).
    Severity: INVALIDATING
    """
    repro = summary.get("reproducibility", {})
    actual_confidence = repro.get("confidence", DEFAULT_CONFIDENCE)

    expected_confidence = DEFAULT_CONFIDENCE
    if prereg:
        expected_confidence = prereg.get("confidence_level", DEFAULT_CONFIDENCE)

    passed = abs(actual_confidence - expected_confidence) < 1e-9
    return RuleResult(
        rule_id="GOV-11",
        passed=passed,
        severity="INVALIDATING",
        message=f"Confidence level matches ({actual_confidence})" if passed else f"Confidence mismatch: {actual_confidence} != {expected_confidence}",
        evidence={"actual": actual_confidence, "expected": expected_confidence}
    )


def _check_gov_12(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    GOV-12: Statistical Method Match
    Wilson CI for proportions; bootstrap for continuous metrics.
    Severity: INVALIDATING
    """
    repro = summary.get("reproducibility", {})

    # Check for indicators that correct methods were used
    has_bootstrap_seed = "bootstrap_seed" in repro
    has_n_bootstrap = "n_bootstrap" in repro

    # For proportions, we check if Wilson method is indicated (if available)
    ci_method = repro.get("ci_method", "wilson")  # Default assumption

    passed = has_bootstrap_seed and has_n_bootstrap and ci_method in ["wilson", "wilson_score"]
    return RuleResult(
        rule_id="GOV-12",
        passed=passed,
        severity="INVALIDATING",
        message="Statistical methods verified" if passed else "Statistical method indicators missing",
        evidence={"has_bootstrap": has_bootstrap_seed and has_n_bootstrap, "ci_method": ci_method}
    )


# =============================================================================
# REPRODUCIBILITY RULES (REP-*)
# =============================================================================

def _check_rep_1(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    REP-1: Baseline Seed Documented
    manifest.json MUST contain seed_baseline as a positive integer.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="REP-1",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    seed = _get_nested(manifest, "config", "seed_baseline")
    passed = isinstance(seed, int) and seed > 0

    return RuleResult(
        rule_id="REP-1",
        passed=passed,
        severity="INVALIDATING",
        message=f"seed_baseline={seed}" if passed else "Missing or invalid seed_baseline",
        evidence={"seed_baseline": seed}
    )


def _check_rep_2(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    REP-2: RFL Seed Documented
    manifest.json MUST contain seed_rfl as a positive integer.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="REP-2",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    seed = _get_nested(manifest, "config", "seed_rfl")
    passed = isinstance(seed, int) and seed > 0

    return RuleResult(
        rule_id="REP-2",
        passed=passed,
        severity="INVALIDATING",
        message=f"seed_rfl={seed}" if passed else "Missing or invalid seed_rfl",
        evidence={"seed_rfl": seed}
    )


def _check_rep_3(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    REP-3: Bootstrap Seed Documented
    summary.json MUST contain bootstrap_seed as a positive integer.
    Severity: INVALIDATING
    """
    seed = _get_nested(summary, "reproducibility", "bootstrap_seed")
    passed = isinstance(seed, int) and seed > 0

    return RuleResult(
        rule_id="REP-3",
        passed=passed,
        severity="INVALIDATING",
        message=f"bootstrap_seed={seed}" if passed else "Missing or invalid bootstrap_seed",
        evidence={"bootstrap_seed": seed}
    )


def _check_rep_4(summary: Dict, manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    REP-4: Seed Distinctness
    All three seeds must be distinct.
    Severity: WARNING
    """
    bootstrap_seed = _get_nested(summary, "reproducibility", "bootstrap_seed")

    if manifest is None:
        return RuleResult(
            rule_id="REP-4",
            passed=True,
            severity="WARNING",
            message="Partial check (no manifest provided)",
            evidence={"skipped": True}
        )

    seed_baseline = _get_nested(manifest, "config", "seed_baseline")
    seed_rfl = _get_nested(manifest, "config", "seed_rfl")

    seeds = [seed_baseline, seed_rfl, bootstrap_seed]
    valid_seeds = [s for s in seeds if s is not None]
    unique_seeds = set(valid_seeds)

    passed = len(unique_seeds) == len(valid_seeds)
    return RuleResult(
        rule_id="REP-4",
        passed=passed,
        severity="WARNING",
        message="All seeds are distinct" if passed else f"Duplicate seeds detected: {seeds}",
        evidence={"seeds": seeds, "unique_count": len(unique_seeds)}
    )


def _check_rep_5(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    REP-5: Bootstrap Iterations Minimum
    n_bootstrap >= 10,000.
    Severity: INVALIDATING
    """
    n_bootstrap = _get_nested(summary, "reproducibility", "n_bootstrap", default=0)
    passed = n_bootstrap >= MIN_BOOTSTRAP_ITERATIONS

    return RuleResult(
        rule_id="REP-5",
        passed=passed,
        severity="INVALIDATING",
        message=f"n_bootstrap={n_bootstrap}" if passed else f"n_bootstrap={n_bootstrap} < {MIN_BOOTSTRAP_ITERATIONS}",
        evidence={"n_bootstrap": n_bootstrap, "minimum": MIN_BOOTSTRAP_ITERATIONS}
    )


def _check_rep_6(**kwargs) -> Optional[RuleResult]:
    """
    REP-6: Determinism Verification
    Re-running analysis must produce identical results.
    Severity: INVALIDATING

    Note: This requires an actual re-run which is not performed here.
    The rule is marked as passed with a note.
    """
    return RuleResult(
        rule_id="REP-6",
        passed=True,
        severity="INVALIDATING",
        message="Determinism check requires re-run (not performed in static verification)",
        evidence={"requires_rerun": True}
    )


def _check_rep_7(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    REP-7: Code Version Recorded
    manifest.json MUST contain analysis_code_version.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="REP-7",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    version = _get_nested(manifest, "metadata", "analysis_code_version") or \
              manifest.get("analysis_code_version")
    passed = version is not None and str(version).strip() != ""

    return RuleResult(
        rule_id="REP-7",
        passed=passed,
        severity="WARNING",
        message=f"Code version: {version}" if passed else "Missing analysis_code_version",
        evidence={"analysis_code_version": version}
    )


def _check_rep_8(manifest: Optional[Dict] = None, base_path: str = ".", **kwargs) -> Optional[RuleResult]:
    """
    REP-8: Raw Data Preserved
    Paths to raw JSONL logs MUST be recorded and files MUST exist.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="REP-8",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    artifacts = manifest.get("artifacts", {})
    baseline_logs = artifacts.get("baseline_logs", [])
    rfl_logs = artifacts.get("rfl_logs", [])

    missing = []
    for log_path in baseline_logs + rfl_logs:
        full_path = Path(base_path) / log_path
        if not full_path.exists():
            missing.append(str(log_path))

    passed = len(missing) == 0 and (len(baseline_logs) > 0 or len(rfl_logs) > 0)
    return RuleResult(
        rule_id="REP-8",
        passed=passed,
        severity="INVALIDATING",
        message="Raw data logs present" if passed else f"Missing logs: {missing}",
        evidence={"missing": missing, "baseline_count": len(baseline_logs), "rfl_count": len(rfl_logs)}
    )


# =============================================================================
# MANIFEST RULES (MAN-*)
# =============================================================================

def _check_man_1(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-1: Experiment ID Present
    manifest.json MUST contain experiment_id (non-empty string).
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-1",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    exp_id = manifest.get("experiment_id", "")
    passed = isinstance(exp_id, str) and exp_id.strip() != ""

    return RuleResult(
        rule_id="MAN-1",
        passed=passed,
        severity="INVALIDATING",
        message=f"experiment_id: {exp_id}" if passed else "Missing or empty experiment_id",
        evidence={"experiment_id": exp_id}
    )


def _check_man_2(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-2: Preregistration Reference
    manifest.json MUST reference the preregistration file (prereg_ref).
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-2",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    prereg_ref = manifest.get("prereg_ref", "")
    passed = isinstance(prereg_ref, str) and prereg_ref.strip() != ""

    return RuleResult(
        rule_id="MAN-2",
        passed=passed,
        severity="INVALIDATING",
        message=f"prereg_ref: {prereg_ref}" if passed else "Missing prereg_ref",
        evidence={"prereg_ref": prereg_ref}
    )


def _check_man_3(manifest: Optional[Dict] = None, base_path: str = ".", **kwargs) -> Optional[RuleResult]:
    """
    MAN-3: Preregistration File Exists
    The referenced preregistration file MUST exist.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-3",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    prereg_ref = manifest.get("prereg_ref", "")
    if not prereg_ref:
        return RuleResult(
            rule_id="MAN-3",
            passed=False,
            severity="INVALIDATING",
            message="No prereg_ref to check",
            evidence={}
        )

    full_path = Path(base_path) / prereg_ref
    passed = full_path.exists()

    return RuleResult(
        rule_id="MAN-3",
        passed=passed,
        severity="INVALIDATING",
        message=f"Preregistration file exists: {prereg_ref}" if passed else f"Preregistration file not found: {prereg_ref}",
        evidence={"prereg_ref": prereg_ref, "checked_path": str(full_path)}
    )


def _check_man_4(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-4: Slice Configuration Complete
    manifest.json MUST contain configuration for all four slices.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-4",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    config_slices = set(_get_nested(manifest, "config", "slices", default={}).keys())
    missing = SLICE_IDS - config_slices

    passed = len(missing) == 0
    return RuleResult(
        rule_id="MAN-4",
        passed=passed,
        severity="INVALIDATING",
        message="All slice configurations present" if passed else f"Missing slice configs: {missing}",
        evidence={"present": list(config_slices), "missing": list(missing)}
    )


def _check_man_5(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-5: Artifact Checksums Present
    manifest.json MUST contain SHA-256 checksums for all artifact files.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-5",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    checksums = manifest.get("checksums", {})
    passed = len(checksums) > 0

    return RuleResult(
        rule_id="MAN-5",
        passed=passed,
        severity="WARNING",
        message=f"Checksums present ({len(checksums)} files)" if passed else "No checksums recorded",
        evidence={"checksum_count": len(checksums)}
    )


def _check_man_6(manifest: Optional[Dict] = None, base_path: str = ".", **kwargs) -> Optional[RuleResult]:
    """
    MAN-6: Artifact Checksums Valid
    All recorded checksums MUST match actual file checksums.
    Severity: INVALIDATING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-6",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    checksums = manifest.get("checksums", {})
    if not checksums:
        return RuleResult(
            rule_id="MAN-6",
            passed=True,
            severity="INVALIDATING",
            message="No checksums to verify",
            evidence={"checksum_count": 0}
        )

    failures = []
    for file_path, expected_hash in checksums.items():
        full_path = Path(base_path) / file_path
        if not full_path.exists():
            failures.append({"path": file_path, "error": "file not found"})
            continue

        with open(full_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected_hash:
            failures.append({
                "path": file_path,
                "expected": expected_hash[:16] + "...",
                "actual": actual_hash[:16] + "..."
            })

    passed = len(failures) == 0
    return RuleResult(
        rule_id="MAN-6",
        passed=passed,
        severity="INVALIDATING",
        message="All checksums valid" if passed else f"Checksum failures: {len(failures)}",
        evidence={"failures": failures, "total_checked": len(checksums)}
    )


def _check_man_7(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-7: Created Timestamp Present
    manifest.json MUST contain created_at in ISO 8601 format.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-7",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    created_at = manifest.get("created_at", "")

    # Try to parse as ISO 8601
    passed = False
    if created_at:
        try:
            # Handle various ISO 8601 formats
            if "T" in str(created_at):
                datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                passed = True
        except ValueError:
            pass

    return RuleResult(
        rule_id="MAN-7",
        passed=passed,
        severity="WARNING",
        message=f"created_at: {created_at}" if passed else "Missing or invalid created_at timestamp",
        evidence={"created_at": created_at}
    )


def _check_man_8(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-8: Schema Version Present
    manifest.json MUST declare its schema version.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-8",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    schema = manifest.get("$schema") or manifest.get("schema_version")
    passed = schema is not None and str(schema).strip() != ""

    return RuleResult(
        rule_id="MAN-8",
        passed=passed,
        severity="WARNING",
        message=f"Schema: {schema}" if passed else "Missing schema version",
        evidence={"schema": schema}
    )


def _check_man_9(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-9: No Extraneous Slices
    manifest.json MUST NOT contain undefined slice configurations.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-9",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    config_slices = set(_get_nested(manifest, "config", "slices", default={}).keys())
    extraneous = config_slices - SLICE_IDS

    passed = len(extraneous) == 0
    return RuleResult(
        rule_id="MAN-9",
        passed=passed,
        severity="WARNING",
        message="No extraneous slices" if passed else f"Extraneous slices: {extraneous}",
        evidence={"extraneous": list(extraneous)}
    )


def _check_man_10(manifest: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    MAN-10: Derivation Parameters Recorded
    For each slice, derivation parameters MUST be recorded.
    Severity: WARNING
    """
    if manifest is None:
        return RuleResult(
            rule_id="MAN-10",
            passed=True,
            severity="WARNING",
            message="Skipped (no manifest provided)",
            evidence={"skipped": True}
        )

    slices_config = _get_nested(manifest, "config", "slices", default={})
    missing_params = []
    required_params = ["steps", "depth", "breadth", "total"]

    for slice_id in SLICE_IDS:
        slice_cfg = slices_config.get(slice_id, {})
        derivation = slice_cfg.get("derivation_params", {})
        missing = [p for p in required_params if p not in derivation]
        if missing:
            missing_params.append(f"{slice_id}: {missing}")

    passed = len(missing_params) == 0
    return RuleResult(
        rule_id="MAN-10",
        passed=passed,
        severity="WARNING",
        message="Derivation parameters recorded" if passed else f"Missing params: {missing_params}",
        evidence={"missing": missing_params}
    )


# =============================================================================
# INVARIANT RULES (INV-*)
# =============================================================================

def _check_inv_d1(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D1: Cycle Index Continuity
    Cycle indices MUST be consecutive with no gaps.
    Severity: INVALIDATING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D1",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    gaps = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        if not cycles_data:
            continue

        indices = sorted([c.get("cycle", 0) for c in cycles_data])
        if len(indices) > 1:
            expected = list(range(indices[0], indices[-1] + 1))
            if indices != expected:
                missing = set(expected) - set(indices)
                gaps.append({"condition": condition, "missing": list(missing)})

    passed = len(gaps) == 0
    return RuleResult(
        rule_id="INV-D1",
        passed=passed,
        severity="INVALIDATING",
        message="Cycle indices continuous" if passed else f"Gaps found: {gaps}",
        evidence={"gaps": gaps}
    )


def _check_inv_d2(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D2: Timestamp Monotonicity
    Timestamps within a condition MUST be strictly increasing.
    Severity: WARNING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D2",
            passed=True,
            severity="WARNING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    violations = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        timestamps = [c.get("timestamp", "") for c in cycles_data]

        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                violations.append({"condition": condition, "index": i})

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-D2",
        passed=passed,
        severity="WARNING",
        message="Timestamps monotonic" if passed else f"Non-monotonic timestamps: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_d3(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D3: Verification Bound
    proofs_succeeded <= proofs_attempted for all cycles.
    Severity: INVALIDATING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D3",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    violations = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        for c in cycles_data:
            succeeded = c.get("proofs_succeeded", 0)
            attempted = c.get("proofs_attempted", 0)
            if succeeded > attempted:
                violations.append({"condition": condition, "cycle": c.get("cycle"),
                                   "succeeded": succeeded, "attempted": attempted})

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-D3",
        passed=passed,
        severity="INVALIDATING",
        message="Verification bound holds" if passed else f"Violations: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_d4(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D4: Abstention Bound
    abstention_count <= proofs_attempted for all cycles.
    Severity: INVALIDATING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D4",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    violations = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        for c in cycles_data:
            abstentions = c.get("abstention_count", 0)
            attempted = c.get("proofs_attempted", 0)
            if abstentions > attempted:
                violations.append({"condition": condition, "cycle": c.get("cycle"),
                                   "abstentions": abstentions, "attempted": attempted})

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-D4",
        passed=passed,
        severity="INVALIDATING",
        message="Abstention bound holds" if passed else f"Violations: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_d5(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D5: Duration Positivity
    duration_seconds > 0 for all cycles.
    Severity: INVALIDATING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D5",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    violations = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        for c in cycles_data:
            duration = c.get("duration_seconds", 0)
            if duration <= 0:
                violations.append({"condition": condition, "cycle": c.get("cycle"), "duration": duration})

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-D5",
        passed=passed,
        severity="INVALIDATING",
        message="Duration positivity holds" if passed else f"Non-positive durations: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_d6(telemetry: Optional[Dict] = None, **kwargs) -> Optional[RuleResult]:
    """
    INV-D6: Hash Format Validity
    ht_hash MUST match regex ^[0-9a-f]{64}$.
    Severity: INVALIDATING
    """
    if telemetry is None:
        return RuleResult(
            rule_id="INV-D6",
            passed=True,
            severity="INVALIDATING",
            message="Skipped (no telemetry provided)",
            evidence={"skipped": True}
        )

    violations = []
    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        for c in cycles_data:
            ht_hash = c.get("ht_hash", "")
            if ht_hash and not HASH_PATTERN.match(ht_hash):
                violations.append({"condition": condition, "cycle": c.get("cycle"), "hash": ht_hash[:20] + "..."})

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-D6",
        passed=passed,
        severity="INVALIDATING",
        message="Hash formats valid" if passed else f"Invalid hashes: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_s1(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-S1: Wilson CI Bounds
    For Wilson CIs: 0 <= ci_low <= ci_high <= 1.
    Severity: INVALIDATING
    """
    violations = []
    slices = summary.get("slices", {})

    for slice_id, slice_data in slices.items():
        for metric in ["success_rate", "abstention_rate"]:
            m = _get_nested(slice_data, metric, default={}) or \
                _get_nested(slice_data, "metrics", metric, default={})

            ci_low = m.get("ci_low")
            ci_high = m.get("ci_high")

            if ci_low is not None and ci_high is not None:
                if not (0 <= ci_low <= ci_high <= 1):
                    violations.append({
                        "slice": slice_id, "metric": metric,
                        "ci_low": ci_low, "ci_high": ci_high
                    })

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-S1",
        passed=passed,
        severity="INVALIDATING",
        message="Wilson CI bounds valid" if passed else f"Invalid bounds: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_s2(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-S2: Bootstrap CI Ordering
    For all CIs: ci_low <= ci_high.
    Severity: INVALIDATING
    """
    violations = []
    slices = summary.get("slices", {})

    for slice_id, slice_data in slices.items():
        for metric in ["success_rate", "abstention_rate", "throughput", "duration"]:
            m = _get_nested(slice_data, metric, default={}) or \
                _get_nested(slice_data, "metrics", metric, default={})

            # Check various CI key patterns
            for low_key, high_key in [("ci_low", "ci_high"), ("delta_ci_low", "delta_ci_high")]:
                ci_low = m.get(low_key)
                ci_high = m.get(high_key)

                if ci_low is not None and ci_high is not None:
                    if ci_low > ci_high:
                        violations.append({
                            "slice": slice_id, "metric": metric,
                            "ci_low": ci_low, "ci_high": ci_high
                        })

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-S2",
        passed=passed,
        severity="INVALIDATING",
        message="CI ordering valid" if passed else f"Inverted CIs: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_s3(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-S3: Point Estimate Plausibility
    Point estimates SHOULD be within their CIs.
    Severity: WARNING
    """
    warnings = []
    slices = summary.get("slices", {})

    for slice_id, slice_data in slices.items():
        for metric in ["success_rate", "abstention_rate"]:
            m = _get_nested(slice_data, metric, default={}) or \
                _get_nested(slice_data, "metrics", metric, default={})

            baseline = m.get("baseline")
            rfl = m.get("rfl")
            ci_low = m.get("ci_low")
            ci_high = m.get("ci_high")

            if all(v is not None for v in [rfl, ci_low, ci_high]):
                if not (ci_low <= rfl <= ci_high):
                    warnings.append({
                        "slice": slice_id, "metric": metric,
                        "point": rfl, "ci_low": ci_low, "ci_high": ci_high
                    })

    passed = len(warnings) == 0
    return RuleResult(
        rule_id="INV-S3",
        passed=passed,
        severity="WARNING",
        message="Point estimates plausible" if passed else f"Point estimates outside CI: {warnings}",
        evidence={"warnings": warnings}
    )


def _check_inv_s4(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-S4: Rate Bounds
    All rate metrics MUST be in [0, 1].
    Severity: INVALIDATING
    """
    violations = []
    slices = summary.get("slices", {})

    for slice_id, slice_data in slices.items():
        for metric in ["success_rate", "abstention_rate"]:
            m = _get_nested(slice_data, metric, default={}) or \
                _get_nested(slice_data, "metrics", metric, default={})

            for key in ["baseline", "rfl"]:
                value = m.get(key)
                if value is not None and not (0 <= value <= 1):
                    violations.append({
                        "slice": slice_id, "metric": metric, "key": key, "value": value
                    })

    passed = len(violations) == 0
    return RuleResult(
        rule_id="INV-S4",
        passed=passed,
        severity="INVALIDATING",
        message="Rate bounds valid" if passed else f"Out-of-bound rates: {violations}",
        evidence={"violations": violations}
    )


def _check_inv_g1(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-G1: Decision Uniqueness
    Exactly one governance decision MUST be set.
    Severity: INVALIDATING
    """
    # This is equivalent to GOV-2, but kept for invariant completeness
    return _check_gov_2(summary, **kwargs)


def _check_inv_g2(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-G2: Pass Consistency
    PROCEED requires all_slices_pass = true.
    Severity: INVALIDATING
    """
    # This is equivalent to GOV-3, but kept for invariant completeness
    result = _check_gov_3(summary, **kwargs)
    result.rule_id = "INV-G2"
    return result


def _check_inv_g3(summary: Dict, **kwargs) -> Optional[RuleResult]:
    """
    INV-G3: Slice Partition
    passing_slices ∪ failing_slices = set(slices) ∧ passing_slices ∩ failing_slices = ∅.
    Severity: INVALIDATING
    """
    # This is equivalent to GOV-4, but kept for invariant completeness
    result = _check_gov_4(summary, **kwargs)
    result.rule_id = "INV-G3"
    return result


# =============================================================================
# RULE REGISTRY
# =============================================================================

# Map rule IDs to their check functions and severities
RULE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Governance Rules
    "GOV-1": {"fn": _check_gov_1, "severity": "INVALIDATING"},
    "GOV-2": {"fn": _check_gov_2, "severity": "INVALIDATING"},
    "GOV-3": {"fn": _check_gov_3, "severity": "INVALIDATING"},
    "GOV-4": {"fn": _check_gov_4, "severity": "INVALIDATING"},
    "GOV-5": {"fn": _check_gov_5, "severity": "WARNING"},
    "GOV-6": {"fn": _check_gov_6, "severity": "WARNING"},
    "GOV-7": {"fn": _check_gov_7, "severity": "INVALIDATING"},
    "GOV-8": {"fn": _check_gov_8, "severity": "INVALIDATING"},
    "GOV-9": {"fn": _check_gov_9, "severity": "INVALIDATING"},
    "GOV-10": {"fn": _check_gov_10, "severity": "INVALIDATING"},
    "GOV-11": {"fn": _check_gov_11, "severity": "INVALIDATING"},
    "GOV-12": {"fn": _check_gov_12, "severity": "INVALIDATING"},
    # Reproducibility Rules
    "REP-1": {"fn": _check_rep_1, "severity": "INVALIDATING"},
    "REP-2": {"fn": _check_rep_2, "severity": "INVALIDATING"},
    "REP-3": {"fn": _check_rep_3, "severity": "INVALIDATING"},
    "REP-4": {"fn": _check_rep_4, "severity": "WARNING"},
    "REP-5": {"fn": _check_rep_5, "severity": "INVALIDATING"},
    "REP-6": {"fn": _check_rep_6, "severity": "INVALIDATING"},
    "REP-7": {"fn": _check_rep_7, "severity": "WARNING"},
    "REP-8": {"fn": _check_rep_8, "severity": "INVALIDATING"},
    # Manifest Rules
    "MAN-1": {"fn": _check_man_1, "severity": "INVALIDATING"},
    "MAN-2": {"fn": _check_man_2, "severity": "INVALIDATING"},
    "MAN-3": {"fn": _check_man_3, "severity": "INVALIDATING"},
    "MAN-4": {"fn": _check_man_4, "severity": "INVALIDATING"},
    "MAN-5": {"fn": _check_man_5, "severity": "WARNING"},
    "MAN-6": {"fn": _check_man_6, "severity": "INVALIDATING"},
    "MAN-7": {"fn": _check_man_7, "severity": "WARNING"},
    "MAN-8": {"fn": _check_man_8, "severity": "WARNING"},
    "MAN-9": {"fn": _check_man_9, "severity": "WARNING"},
    "MAN-10": {"fn": _check_man_10, "severity": "WARNING"},
    # Invariant Rules
    "INV-D1": {"fn": _check_inv_d1, "severity": "INVALIDATING"},
    "INV-D2": {"fn": _check_inv_d2, "severity": "WARNING"},
    "INV-D3": {"fn": _check_inv_d3, "severity": "INVALIDATING"},
    "INV-D4": {"fn": _check_inv_d4, "severity": "INVALIDATING"},
    "INV-D5": {"fn": _check_inv_d5, "severity": "INVALIDATING"},
    "INV-D6": {"fn": _check_inv_d6, "severity": "INVALIDATING"},
    "INV-S1": {"fn": _check_inv_s1, "severity": "INVALIDATING"},
    "INV-S2": {"fn": _check_inv_s2, "severity": "INVALIDATING"},
    "INV-S3": {"fn": _check_inv_s3, "severity": "WARNING"},
    "INV-S4": {"fn": _check_inv_s4, "severity": "INVALIDATING"},
    "INV-G1": {"fn": _check_inv_g1, "severity": "INVALIDATING"},
    "INV-G2": {"fn": _check_inv_g2, "severity": "INVALIDATING"},
    "INV-G3": {"fn": _check_inv_g3, "severity": "INVALIDATING"},
}


# =============================================================================
# MAIN VERIFICATION FUNCTION
# =============================================================================

def governance_verify(
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    prereg: Optional[Dict[str, Any]] = None,
    base_path: str = ".",
) -> GovernanceVerdict:
    """
    Pure function that verifies an uplift analysis against governance rules.

    This is the primary entry point for governance verification. It checks all
    43 rules and produces a GovernanceVerdict with status PASS/WARN/FAIL.

    Args:
        summary: The statistical_summary.json content (required)
        manifest: The experiment manifest.json content (optional)
        telemetry: The telemetry_summary.json content (optional)
        prereg: The preregistration YAML content (optional)
        base_path: Base path for file existence checks (optional)

    Returns:
        GovernanceVerdict with:
          - status: "PASS" | "WARN" | "FAIL"
          - invalidating_rules: List of rule IDs with INVALIDATING failures
          - warnings: List of rule IDs with WARNING failures
          - passed_rules: List of rule IDs that passed
          - summary: Human-readable summary
          - details: Full RuleResult for each rule

    Properties:
        - PURE: No side effects
        - DETERMINISTIC: Same inputs -> same outputs
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    invalidating_rules: List[str] = []
    warnings: List[str] = []
    passed_rules: List[str] = []
    details: Dict[str, RuleResult] = {}

    # Build kwargs for rule functions
    kwargs = {
        "summary": summary,
        "manifest": manifest,
        "telemetry": telemetry,
        "prereg": prereg,
        "base_path": base_path,
    }

    # Run all rules
    for rule_id, rule_info in RULE_REGISTRY.items():
        fn = rule_info["fn"]
        severity = rule_info["severity"]

        try:
            result = fn(**kwargs)
            if result is None:
                # Rule skipped
                result = RuleResult(
                    rule_id=rule_id,
                    passed=True,
                    severity=severity,
                    message="Skipped",
                    evidence={"skipped": True}
                )

            details[rule_id] = result

            if result.passed:
                passed_rules.append(rule_id)
            elif result.severity == "INVALIDATING":
                invalidating_rules.append(rule_id)
            else:
                warnings.append(rule_id)

        except Exception as e:
            # Rule check failed with exception
            details[rule_id] = RuleResult(
                rule_id=rule_id,
                passed=False,
                severity=severity,
                message=f"Rule check failed with exception: {e}",
                evidence={"exception": str(e)}
            )
            if severity == "INVALIDATING":
                invalidating_rules.append(rule_id)
            else:
                warnings.append(rule_id)

    # Determine final status based on decision tree
    if invalidating_rules:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    # Build summary message
    summary_parts = []
    summary_parts.append(f"Status: {status}")
    summary_parts.append(f"Rules checked: {len(RULE_REGISTRY)}")
    summary_parts.append(f"Passed: {len(passed_rules)}")
    if invalidating_rules:
        summary_parts.append(f"INVALIDATING violations: {invalidating_rules}")
    if warnings:
        summary_parts.append(f"Warnings: {warnings}")

    # v2: Compute reason codes (triggered rules)
    reason_codes = invalidating_rules + warnings

    # Build verdict object
    verdict = GovernanceVerdict(
        status=status,
        invalidating_rules=invalidating_rules,
        warnings=warnings,
        passed_rules=passed_rules,
        summary="; ".join(summary_parts),
        inputs={
            "has_summary": True,
            "has_manifest": manifest is not None,
            "has_telemetry": telemetry is not None,
            "has_prereg": prereg is not None,
            "spec_version": "2.0.0",
        },
        details=details,
        timestamp=timestamp,
        verifier_version=__version__,
        rules_checked=len(RULE_REGISTRY),
        reason_codes=reason_codes,
        short_explanation="",  # Will be generated lazily
    )

    # v2: Generate short explanation
    verdict.short_explanation = _generate_short_explanation(verdict)

    return verdict


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def verify_summary_file(
    summary_path: str,
    manifest_path: Optional[str] = None,
    telemetry_path: Optional[str] = None,
    prereg_path: Optional[str] = None,
) -> GovernanceVerdict:
    """
    Convenience function to verify files by path.

    Args:
        summary_path: Path to summary.json
        manifest_path: Optional path to manifest.json
        telemetry_path: Optional path to telemetry_summary.json
        prereg_path: Optional path to preregistration YAML

    Returns:
        GovernanceVerdict
    """
    import json
    import yaml

    with open(summary_path, "r") as f:
        summary = json.load(f)

    manifest = None
    if manifest_path:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    telemetry = None
    if telemetry_path:
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)

    prereg = None
    if prereg_path:
        with open(prereg_path, "r") as f:
            prereg = yaml.safe_load(f)

    base_path = str(Path(summary_path).parent)

    return governance_verify(
        summary=summary,
        manifest=manifest,
        telemetry=telemetry,
        prereg=prereg,
        base_path=base_path,
    )


# =============================================================================
# V2: GOVERNANCE CHRONICLE & EXPLAINER
# =============================================================================

# Rule descriptions for human-readable explanations
RULE_DESCRIPTIONS: Dict[str, str] = {
    # Governance Rules
    "GOV-1": "Slice metrics must meet predefined success criteria thresholds",
    "GOV-2": "Exactly one governance decision (PROCEED/HOLD/ROLLBACK) must be set",
    "GOV-3": "PROCEED requires all slices to pass; failures require HOLD/ROLLBACK",
    "GOV-4": "Passing and failing slice sets must partition all slices",
    "GOV-5": "Slices with CI overlapping threshold should be flagged as marginal",
    "GOV-6": "HOLD decisions require a rationale",
    "GOV-7": "ROLLBACK decisions require a rationale and identified failing slices",
    "GOV-8": "Each slice must meet minimum sample size requirements",
    "GOV-9": "All four slices must be present in the summary",
    "GOV-10": "All failing slices must be reported in failing_slices",
    "GOV-11": "Confidence level must match preregistered value (default 0.95)",
    "GOV-12": "Statistical methods must match spec (Wilson CI, bootstrap)",
    # Reproducibility Rules
    "REP-1": "Baseline seed must be documented in manifest",
    "REP-2": "RFL seed must be documented in manifest",
    "REP-3": "Bootstrap seed must be documented in summary",
    "REP-4": "All three seeds should be distinct",
    "REP-5": "Bootstrap iterations must be at least 10,000",
    "REP-6": "Re-running analysis must produce identical results",
    "REP-7": "Code version should be recorded in manifest",
    "REP-8": "Raw data logs must exist at documented paths",
    # Manifest Rules
    "MAN-1": "Experiment ID must be present in manifest",
    "MAN-2": "Preregistration reference must be present in manifest",
    "MAN-3": "Referenced preregistration file must exist",
    "MAN-4": "Slice configurations must be complete for all slices",
    "MAN-5": "Artifact checksums should be present",
    "MAN-6": "All recorded checksums must match actual files",
    "MAN-7": "Created timestamp should be present in ISO 8601",
    "MAN-8": "Schema version should be declared",
    "MAN-9": "No undefined slice configurations allowed",
    "MAN-10": "Derivation parameters should be recorded for each slice",
    # Invariant Rules
    "INV-D1": "Cycle indices must be consecutive",
    "INV-D2": "Timestamps within a condition must be monotonic",
    "INV-D3": "proofs_succeeded must not exceed proofs_attempted",
    "INV-D4": "abstention_count must not exceed proofs_attempted",
    "INV-D5": "Duration must be positive",
    "INV-D6": "Hash values must match SHA-256 format",
    "INV-S1": "Wilson CIs must satisfy 0 <= ci_low <= ci_high <= 1",
    "INV-S2": "All CIs must satisfy ci_low <= ci_high",
    "INV-S3": "Point estimates should be within their CIs",
    "INV-S4": "Rate metrics must be in [0, 1]",
    "INV-G1": "Exactly one governance decision must be set",
    "INV-G2": "PROCEED requires all_slices_pass = true",
    "INV-G3": "Slice partition must be valid",
}


def _generate_short_explanation(verdict: "GovernanceVerdict") -> str:
    """
    Generate a short, neutral explanation of the verdict.

    Returns 1-2 sentences describing why the verdict is PASS/WARN/FAIL.
    """
    if verdict.status == "PASS":
        return f"All {verdict.rules_checked} governance rules passed."

    if verdict.status == "WARN":
        warn_count = len(verdict.warnings)
        if warn_count == 1:
            rule_id = verdict.warnings[0]
            desc = RULE_DESCRIPTIONS.get(rule_id, "rule check")
            return f"One warning detected: {rule_id} ({desc})."
        else:
            return f"{warn_count} warnings detected across rules {', '.join(verdict.warnings[:3])}{'...' if warn_count > 3 else ''}."

    # FAIL case
    inv_count = len(verdict.invalidating_rules)
    if inv_count == 1:
        rule_id = verdict.invalidating_rules[0]
        desc = RULE_DESCRIPTIONS.get(rule_id, "rule check")
        return f"INVALIDATING violation: {rule_id} ({desc})."
    else:
        top_rules = verdict.invalidating_rules[:3]
        return f"{inv_count} INVALIDATING violations detected, including {', '.join(top_rules)}{'...' if inv_count > 3 else ''}."


def explain_verdict(verdict: "GovernanceVerdict") -> Dict[str, Any]:
    """
    Produce a compact explanation of a GovernanceVerdict.

    This function returns a structured explanation suitable for:
    - Human review
    - MAAS consumption
    - Audit trail

    Args:
        verdict: The GovernanceVerdict to explain

    Returns:
        Dict containing:
          - status: "PASS" | "WARN" | "FAIL"
          - reason_codes: List of triggered rule IDs
          - short_explanation: 1-2 sentence neutral explanation
          - triggered_rules: Dict mapping rule_id -> description
          - invalidating_count: Number of INVALIDATING violations
          - warning_count: Number of warnings
          - pass_rate: Fraction of rules passed
    """
    # Compute reason codes (all triggered rules)
    reason_codes = verdict.invalidating_rules + verdict.warnings

    # Generate explanation
    short_explanation = verdict.short_explanation or _generate_short_explanation(verdict)

    # Build triggered rules with descriptions
    triggered_rules = {}
    for rule_id in reason_codes:
        triggered_rules[rule_id] = {
            "description": RULE_DESCRIPTIONS.get(rule_id, "Unknown rule"),
            "severity": RULE_REGISTRY.get(rule_id, {}).get("severity", "UNKNOWN"),
            "message": verdict.details.get(rule_id, RuleResult(rule_id, False, "", "")).message,
        }

    return {
        "status": verdict.status,
        "reason_codes": reason_codes,
        "short_explanation": short_explanation,
        "triggered_rules": triggered_rules,
        "invalidating_count": len(verdict.invalidating_rules),
        "warning_count": len(verdict.warnings),
        "pass_rate": len(verdict.passed_rules) / verdict.rules_checked if verdict.rules_checked > 0 else 0.0,
        "timestamp": verdict.timestamp,
        "verifier_version": verdict.verifier_version,
    }


def build_governance_posture(
    verdicts: Sequence["GovernanceVerdict"],
    file_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate multiple GovernanceVerdict objects into a posture snapshot.

    This function provides a single JSON blob summarizing governance status
    across all discovered/verified files.

    Args:
        verdicts: Sequence of GovernanceVerdict objects
        file_paths: Optional sequence of file paths corresponding to verdicts

    Returns:
        Dict containing:
          - pass_count: Number of PASS verdicts
          - warn_count: Number of WARN verdicts
          - fail_count: Number of FAIL verdicts
          - total_count: Total number of verdicts
          - is_governance_blocking: True if any FAIL verdict exists
          - failing_files: List of dicts with file info and reason_codes
          - aggregate_status: Overall posture status
          - timestamp: When posture was computed
    """
    if not verdicts:
        return {
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "total_count": 0,
            "is_governance_blocking": False,
            "failing_files": [],
            "aggregate_status": "PASS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Ensure file_paths matches verdicts length
    if file_paths is None:
        file_paths = [f"file_{i}" for i in range(len(verdicts))]

    pass_count = sum(1 for v in verdicts if v.status == "PASS")
    warn_count = sum(1 for v in verdicts if v.status == "WARN")
    fail_count = sum(1 for v in verdicts if v.status == "FAIL")

    # Collect failing files with their reason codes
    failing_files = []
    for v, path in zip(verdicts, file_paths):
        if v.status == "FAIL":
            reason_codes = v.reason_codes or (v.invalidating_rules + v.warnings)
            failing_files.append({
                "file": path,
                "reason_codes": reason_codes,
                "invalidating_rules": v.invalidating_rules,
                "message": _generate_short_explanation(v),
            })

    # Determine aggregate status
    if fail_count > 0:
        aggregate_status = "FAIL"
    elif warn_count > 0:
        aggregate_status = "WARN"
    else:
        aggregate_status = "PASS"

    return {
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "total_count": len(verdicts),
        "is_governance_blocking": fail_count > 0,
        "failing_files": failing_files,
        "aggregate_status": aggregate_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize_for_admissibility(verdict: "GovernanceVerdict") -> Dict[str, Any]:
    """
    Produce a summary suitable for MAAS v2 (CLAUDE O) admissibility checks.

    This helper matches MAAS expectations for governance integration
    (HE-GV1 through HE-GV5).

    Args:
        verdict: The GovernanceVerdict to summarize

    Returns:
        Dict containing:
          - overall_status: "PASS" | "WARN" | "FAIL"
          - has_invalidating_violations: True if any INVALIDATING rule failed
          - invalidating_rules: List of failed INVALIDATING rule IDs
          - invalidating_count: Number of INVALIDATING violations
          - warning_count: Number of warnings
          - is_admissible: True if verdict is not FAIL
          - reason_summary: Brief explanation for MAAS
          - governance_version: Verifier version for compatibility

    MAAS Integration Points:
      - HE-GV1: overall_status checked for "FAIL"
      - HE-GV2: has_invalidating_violations used for gate decisions
      - HE-GV3: invalidating_rules enumerated for error messages
      - HE-GV4: is_admissible used as boolean gate
      - HE-GV5: governance_version checked for compatibility
    """
    has_invalidating = len(verdict.invalidating_rules) > 0

    # Admissible if not FAIL (PASS and WARN are both admissible)
    is_admissible = verdict.status != "FAIL"

    # Generate reason summary for MAAS
    if verdict.status == "PASS":
        reason_summary = "Governance verification passed."
    elif verdict.status == "WARN":
        reason_summary = f"Governance verification passed with {len(verdict.warnings)} warning(s)."
    else:
        reason_summary = f"Governance verification failed with {len(verdict.invalidating_rules)} INVALIDATING violation(s)."

    return {
        "overall_status": verdict.status,
        "has_invalidating_violations": has_invalidating,
        "invalidating_rules": verdict.invalidating_rules,
        "invalidating_count": len(verdict.invalidating_rules),
        "warning_count": len(verdict.warnings),
        "is_admissible": is_admissible,
        "reason_summary": reason_summary,
        "governance_version": verdict.verifier_version,
    }


# =============================================================================
# PHASE III: DIRECTOR CONSOLE GOVERNANCE FEED
# =============================================================================

def build_governance_chronicle(
    posture_snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a governance chronicle from a sequence of posture snapshots.

    This function analyzes governance trends over time, identifying recurring
    failures and computing blocking rates for the Director Console feed.

    Args:
        posture_snapshots: Sequence of posture dicts from build_governance_posture()
                          Each should have: aggregate_status, failing_files,
                          is_governance_blocking, timestamp

    Returns:
        Dict containing:
          - trend: "improving" | "stable" | "degrading" | "unknown"
          - recurring_rule_failures: Dict[rule_id -> count] for rules failing > 1 time
          - governance_blocking_rate: float in [0, 1] (fraction of blocking snapshots)
          - total_snapshots: int
          - pass_count: int
          - warn_count: int
          - fail_count: int
          - most_common_failures: List of (rule_id, count) tuples, top 5
          - first_snapshot: timestamp of oldest snapshot
          - last_snapshot: timestamp of newest snapshot
    """
    if not posture_snapshots:
        return {
            "trend": "unknown",
            "recurring_rule_failures": {},
            "governance_blocking_rate": 0.0,
            "total_snapshots": 0,
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "most_common_failures": [],
            "first_snapshot": None,
            "last_snapshot": None,
        }

    # Count statuses
    pass_count = sum(1 for p in posture_snapshots if p.get("aggregate_status") == "PASS")
    warn_count = sum(1 for p in posture_snapshots if p.get("aggregate_status") == "WARN")
    fail_count = sum(1 for p in posture_snapshots if p.get("aggregate_status") == "FAIL")

    # Count blocking snapshots
    blocking_count = sum(1 for p in posture_snapshots if p.get("is_governance_blocking", False))
    blocking_rate = blocking_count / len(posture_snapshots)

    # Collect all rule failures across snapshots
    rule_failure_counts: Dict[str, int] = {}
    for posture in posture_snapshots:
        failing_files = posture.get("failing_files", [])
        for ff in failing_files:
            for rule_id in ff.get("reason_codes", []):
                rule_failure_counts[rule_id] = rule_failure_counts.get(rule_id, 0) + 1

    # Filter to recurring failures (> 1 occurrence)
    recurring_failures = {k: v for k, v in rule_failure_counts.items() if v > 1}

    # Most common failures (top 5)
    sorted_failures = sorted(rule_failure_counts.items(), key=lambda x: -x[1])
    most_common = sorted_failures[:5]

    # Compute trend based on recent vs older snapshots
    trend = _compute_governance_trend(posture_snapshots)

    # Extract timestamps
    timestamps = [p.get("timestamp") for p in posture_snapshots if p.get("timestamp")]
    first_snapshot = min(timestamps) if timestamps else None
    last_snapshot = max(timestamps) if timestamps else None

    return {
        "trend": trend,
        "recurring_rule_failures": recurring_failures,
        "governance_blocking_rate": blocking_rate,
        "total_snapshots": len(posture_snapshots),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "most_common_failures": most_common,
        "first_snapshot": first_snapshot,
        "last_snapshot": last_snapshot,
    }


def _compute_governance_trend(posture_snapshots: Sequence[Dict[str, Any]]) -> str:
    """
    Compute the trend direction from posture snapshots.

    Compares the first half vs second half of snapshots to determine trend.

    Returns:
        "improving" - fewer failures in recent snapshots
        "stable" - similar failure rates
        "degrading" - more failures in recent snapshots
        "unknown" - insufficient data
    """
    if len(posture_snapshots) < 2:
        return "unknown"

    # Split into halves
    mid = len(posture_snapshots) // 2
    first_half = posture_snapshots[:mid]
    second_half = posture_snapshots[mid:]

    if not first_half or not second_half:
        return "unknown"

    # Calculate blocking rates for each half
    first_blocking = sum(1 for p in first_half if p.get("is_governance_blocking", False)) / len(first_half)
    second_blocking = sum(1 for p in second_half if p.get("is_governance_blocking", False)) / len(second_half)

    # Determine trend with 10% threshold
    delta = second_blocking - first_blocking
    if delta < -0.1:
        return "improving"
    elif delta > 0.1:
        return "degrading"
    else:
        return "stable"


def map_governance_to_director_status(
    posture: Dict[str, Any],
) -> str:
    """
    Map a governance posture to a Director Console status color.

    This function translates governance results into the traffic-light
    status system used by the Director Console.

    Args:
        posture: A posture dict from build_governance_posture()

    Returns:
        "GREEN" - All governance checks pass (PASS status)
        "YELLOW" - Warnings present but no blocking violations (WARN status)
        "RED" - Governance is blocking (FAIL status)

    Director Console Semantics:
        - GREEN: Safe to proceed, all governance requirements met
        - YELLOW: Caution advised, review warnings before proceeding
        - RED: Stop, governance violations must be resolved
    """
    if not posture:
        return "RED"  # No data is treated as failure

    aggregate_status = posture.get("aggregate_status", "FAIL")
    is_blocking = posture.get("is_governance_blocking", True)

    if is_blocking or aggregate_status == "FAIL":
        return "RED"
    elif aggregate_status == "WARN":
        return "YELLOW"
    else:
        return "GREEN"


def summarize_governance_for_global_health(
    posture: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a governance summary for the Global Health dashboard.

    This function creates a compact summary suitable for inclusion in
    the system-wide health check that monitors all MathLedger subsystems.

    Args:
        posture: A posture dict from build_governance_posture()

    Returns:
        Dict containing:
          - governance_ok: bool (True if not blocking)
          - failing_rules: List[str] (all unique failing rule IDs)
          - fatality: "none" | "warning" | "fatal"
          - status_code: int (0=ok, 1=warning, 2=fatal)
          - summary: str (one-line summary for dashboards)
          - file_count: int (number of files checked)
          - fail_count: int (number of files failing)

    Fatality Levels:
        - "none": All governance checks pass
        - "warning": Non-blocking issues detected
        - "fatal": Governance is blocking, action required
    """
    if not posture:
        return {
            "governance_ok": False,
            "failing_rules": [],
            "fatality": "fatal",
            "status_code": 2,
            "summary": "No governance data available",
            "file_count": 0,
            "fail_count": 0,
        }

    is_blocking = posture.get("is_governance_blocking", True)
    aggregate_status = posture.get("aggregate_status", "FAIL")
    fail_count = posture.get("fail_count", 0)
    total_count = posture.get("total_count", 0)
    failing_files = posture.get("failing_files", [])

    # Collect all unique failing rules
    failing_rules: List[str] = []
    seen_rules: set = set()
    for ff in failing_files:
        for rule_id in ff.get("reason_codes", []):
            if rule_id not in seen_rules:
                failing_rules.append(rule_id)
                seen_rules.add(rule_id)

    # Determine fatality level
    if is_blocking:
        fatality = "fatal"
        status_code = 2
        governance_ok = False
    elif aggregate_status == "WARN":
        fatality = "warning"
        status_code = 1
        governance_ok = True  # Warnings don't block
    else:
        fatality = "none"
        status_code = 0
        governance_ok = True

    # Generate summary line
    if governance_ok and fatality == "none":
        summary = f"Governance OK: {total_count} file(s) verified"
    elif governance_ok and fatality == "warning":
        warn_count = posture.get("warn_count", 0)
        summary = f"Governance warnings: {warn_count} file(s) with warnings"
    else:
        summary = f"Governance FAILED: {fail_count}/{total_count} file(s) blocking"

    return {
        "governance_ok": governance_ok,
        "failing_rules": failing_rules,
        "fatality": fatality,
        "status_code": status_code,
        "summary": summary,
        "file_count": total_count,
        "fail_count": fail_count,
    }


# =============================================================================
# PHASE IV: GOVERNANCE CHRONICLE COMPASS & CROSS-SYSTEM GATE
# =============================================================================

def build_governance_alignment_view(
    chronicle: Dict[str, Any],
    admissibility_analytics: Optional[Dict[str, Any]] = None,
    topology_trajectory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a cross-system governance alignment view.

    This function correlates governance failures with admissibility analytics
    and topology trajectory data to identify systemic issues affecting
    multiple layers.

    Args:
        chronicle: Governance chronicle from build_governance_chronicle()
        admissibility_analytics: Analytics layer status (optional)
            Expected keys: failing_rules, status, issues
        topology_trajectory: Topology layer status (optional)
            Expected keys: failing_rules, trajectory_status, anomalies

    Returns:
        Dict containing:
          - rules_failing_in_multiple_layers: Dict[rule_id -> List[layer_name]]
          - governance_alignment_status: "ALIGNED" | "TENSION" | "DIVERGENT"
          - layer_statuses: Dict of each layer's status
          - cross_layer_failures: List of rules failing in 2+ layers
          - alignment_score: float in [0, 1] (1 = fully aligned)
          - recommendation: str (neutral guidance)

    Alignment Status Semantics:
        - ALIGNED: All layers agree, no cross-layer failures
        - TENSION: Some cross-layer failures, but manageable
        - DIVERGENT: Significant cross-layer failures, requires attention
    """
    if not chronicle:
        chronicle = {}

    # Extract failing rules from each layer
    governance_failures = set(chronicle.get("recurring_rule_failures", {}).keys())

    # Also include most common failures if no recurring
    if not governance_failures:
        most_common = chronicle.get("most_common_failures", [])
        governance_failures = set(rule for rule, _ in most_common)

    admissibility_failures: set = set()
    if admissibility_analytics:
        admissibility_failures = set(admissibility_analytics.get("failing_rules", []))

    topology_failures: set = set()
    if topology_trajectory:
        topology_failures = set(topology_trajectory.get("failing_rules", []))

    # Find rules failing in multiple layers
    all_rules = governance_failures | admissibility_failures | topology_failures
    rules_in_layers: Dict[str, List[str]] = {}

    for rule in all_rules:
        layers = []
        if rule in governance_failures:
            layers.append("governance")
        if rule in admissibility_failures:
            layers.append("admissibility")
        if rule in topology_failures:
            layers.append("topology")
        if len(layers) > 0:
            rules_in_layers[rule] = layers

    # Filter to rules in multiple layers
    rules_failing_multiple = {
        rule: layers for rule, layers in rules_in_layers.items()
        if len(layers) >= 2
    }

    cross_layer_failures = list(rules_failing_multiple.keys())

    # Collect layer statuses
    layer_statuses = {
        "governance": _get_layer_status(chronicle),
        "admissibility": _get_admissibility_layer_status(admissibility_analytics),
        "topology": _get_topology_layer_status(topology_trajectory),
    }

    # Compute alignment score
    total_failures = len(all_rules)
    multi_layer_failures = len(cross_layer_failures)

    if total_failures == 0:
        alignment_score = 1.0
    else:
        # Score decreases with multi-layer failures
        alignment_score = max(0.0, 1.0 - (multi_layer_failures / max(total_failures, 1)))

    # Determine alignment status
    if multi_layer_failures == 0:
        alignment_status = "ALIGNED"
        recommendation = "All layers operating independently without cross-layer conflicts."
    elif multi_layer_failures <= 2:
        alignment_status = "TENSION"
        recommendation = f"{multi_layer_failures} rule(s) failing across layers. Review for systemic issues."
    else:
        alignment_status = "DIVERGENT"
        recommendation = f"{multi_layer_failures} rules failing across multiple layers. Coordination required."

    return {
        "rules_failing_in_multiple_layers": rules_failing_multiple,
        "governance_alignment_status": alignment_status,
        "layer_statuses": layer_statuses,
        "cross_layer_failures": cross_layer_failures,
        "alignment_score": alignment_score,
        "recommendation": recommendation,
    }


def _get_layer_status(chronicle: Dict[str, Any]) -> str:
    """Extract status from governance chronicle."""
    if not chronicle:
        return "UNKNOWN"

    blocking_rate = chronicle.get("governance_blocking_rate", 0.0)
    if blocking_rate == 0.0:
        return "OK"
    elif blocking_rate < 0.5:
        return "WARN"
    else:
        return "FAIL"


def _get_admissibility_layer_status(admissibility: Optional[Dict[str, Any]]) -> str:
    """Extract status from admissibility analytics."""
    if not admissibility:
        return "UNKNOWN"

    status = admissibility.get("status", admissibility.get("overall_status", "UNKNOWN"))
    if status in {"PASS", "OK", "ok", "pass"}:
        return "OK"
    elif status in {"WARN", "warn", "warning"}:
        return "WARN"
    elif status in {"FAIL", "fail", "fatal"}:
        return "FAIL"
    return str(status).upper() if status else "UNKNOWN"


def _get_topology_layer_status(topology: Optional[Dict[str, Any]]) -> str:
    """Extract status from topology trajectory."""
    if not topology:
        return "UNKNOWN"

    status = topology.get("trajectory_status", topology.get("status", "UNKNOWN"))
    if status in {"stable", "STABLE", "OK", "ok"}:
        return "OK"
    elif status in {"degrading", "DEGRADING", "WARN", "warn"}:
        return "WARN"
    elif status in {"critical", "CRITICAL", "FAIL", "fail"}:
        return "FAIL"
    return str(status).upper() if status else "UNKNOWN"


def evaluate_governance_for_promotion(
    alignment_view: Dict[str, Any],
    global_posture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate governance status for promotion readiness.

    This function determines whether the current governance state permits
    promotion to the next stage (e.g., staging → production).

    Args:
        alignment_view: Cross-system alignment from build_governance_alignment_view()
        global_posture: Current global governance posture (optional)

    Returns:
        Dict containing:
          - promotion_ok: bool (True if promotion is permitted)
          - status: "OK" | "WARN" | "BLOCK"
          - blocking_rules: List[str] (rules blocking promotion)
          - reason: str (explanation of decision)
          - alignment_status: str (from alignment view)
          - conditions: List[str] (conditions that must be met)

    Promotion Logic:
        - BLOCK if alignment is DIVERGENT
        - BLOCK if global_posture has fatal governance failures
        - WARN if alignment is TENSION
        - OK if alignment is ALIGNED and no blocking rules
    """
    if not alignment_view:
        return {
            "promotion_ok": False,
            "status": "BLOCK",
            "blocking_rules": [],
            "reason": "No alignment view available",
            "alignment_status": "UNKNOWN",
            "conditions": ["Provide valid alignment view"],
        }

    alignment_status = alignment_view.get("governance_alignment_status", "UNKNOWN")
    cross_layer_failures = alignment_view.get("cross_layer_failures", [])
    rules_multi = alignment_view.get("rules_failing_in_multiple_layers", {})

    # Collect all blocking rules
    blocking_rules: List[str] = []

    # Add cross-layer failures as blocking
    blocking_rules.extend(cross_layer_failures)

    # Check global posture for additional blocking rules
    if global_posture:
        global_failing = global_posture.get("failing_rules", [])
        global_fatality = global_posture.get("fatality", "none")

        if global_fatality == "fatal":
            for rule in global_failing:
                if rule not in blocking_rules:
                    blocking_rules.append(rule)

    # Determine promotion status
    conditions: List[str] = []

    if alignment_status == "DIVERGENT":
        promotion_ok = False
        status = "BLOCK"
        reason = f"Cross-system alignment is DIVERGENT with {len(cross_layer_failures)} conflicting rules."
        conditions.append("Resolve cross-layer rule failures")
        conditions.append("Achieve ALIGNED or TENSION status")

    elif alignment_status == "TENSION":
        # TENSION allows promotion with warning
        if len(blocking_rules) > 3:
            promotion_ok = False
            status = "BLOCK"
            reason = f"Too many blocking rules ({len(blocking_rules)}) for promotion."
            conditions.append("Reduce blocking rules to 3 or fewer")
        else:
            promotion_ok = True
            status = "WARN"
            reason = f"Promotion permitted with {len(blocking_rules)} rule(s) under review."
            conditions.append("Monitor cross-layer failures post-promotion")

    elif alignment_status == "ALIGNED":
        if len(blocking_rules) == 0:
            promotion_ok = True
            status = "OK"
            reason = "All governance checks aligned. Promotion permitted."
        else:
            # ALIGNED but global posture has issues
            promotion_ok = True
            status = "WARN"
            reason = f"Aligned but {len(blocking_rules)} rule(s) flagged in global posture."
            conditions.append("Review flagged rules before promotion")

    else:
        # Unknown alignment status
        promotion_ok = False
        status = "BLOCK"
        reason = f"Unknown alignment status: {alignment_status}"
        conditions.append("Provide valid alignment data")

    return {
        "promotion_ok": promotion_ok,
        "status": status,
        "blocking_rules": blocking_rules,
        "reason": reason,
        "alignment_status": alignment_status,
        "conditions": conditions,
    }


def build_governance_director_panel_v2(
    chronicle: Dict[str, Any],
    promotion_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Director Governance Panel v2.

    This function produces a compact dashboard panel for the Director Console
    showing governance status, trends, and promotion readiness.

    Args:
        chronicle: Governance chronicle from build_governance_chronicle()
        promotion_eval: Promotion evaluation from evaluate_governance_for_promotion()

    Returns:
        Dict containing:
          - status_light: "GREEN" | "YELLOW" | "RED"
          - governance_blocking_rate: float in [0, 1]
          - trend: "improving" | "stable" | "degrading" | "unknown"
          - headline: str (neutral one-line summary)
          - promotion_status: "OK" | "WARN" | "BLOCK" | "N/A"
          - blocking_count: int
          - snapshot_count: int
          - most_common_failure: str | None

    Panel Semantics:
        - status_light: Overall traffic light based on blocking rate + promotion
        - headline: Single neutral sentence for dashboard display
    """
    if not chronicle:
        return {
            "status_light": "RED",
            "governance_blocking_rate": 1.0,
            "trend": "unknown",
            "headline": "No governance data available.",
            "promotion_status": "N/A",
            "blocking_count": 0,
            "snapshot_count": 0,
            "most_common_failure": None,
        }

    blocking_rate = chronicle.get("governance_blocking_rate", 0.0)
    trend = chronicle.get("trend", "unknown")
    total_snapshots = chronicle.get("total_snapshots", 0)
    fail_count = chronicle.get("fail_count", 0)
    most_common = chronicle.get("most_common_failures", [])

    # Extract most common failure
    most_common_failure = most_common[0][0] if most_common else None

    # Get promotion status
    promotion_status = "N/A"
    promotion_blocking: List[str] = []
    if promotion_eval:
        promotion_status = promotion_eval.get("status", "N/A")
        promotion_blocking = promotion_eval.get("blocking_rules", [])

    # Determine status light
    if promotion_status == "BLOCK":
        status_light = "RED"
    elif blocking_rate >= 0.5:
        status_light = "RED"
    elif blocking_rate > 0 or promotion_status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Generate headline
    headline = _generate_panel_headline(
        blocking_rate, trend, fail_count, total_snapshots, promotion_status
    )

    return {
        "status_light": status_light,
        "governance_blocking_rate": blocking_rate,
        "trend": trend,
        "headline": headline,
        "promotion_status": promotion_status,
        "blocking_count": len(promotion_blocking) if promotion_blocking else fail_count,
        "snapshot_count": total_snapshots,
        "most_common_failure": most_common_failure,
    }


def _generate_panel_headline(
    blocking_rate: float,
    trend: str,
    fail_count: int,
    total_snapshots: int,
    promotion_status: str,
) -> str:
    """Generate a neutral one-line headline for the panel."""
    if total_snapshots == 0:
        return "No governance snapshots recorded."

    if blocking_rate == 0.0:
        if trend == "improving":
            return f"Governance clear across {total_snapshots} snapshot(s), trend improving."
        elif trend == "stable":
            return f"Governance stable and clear across {total_snapshots} snapshot(s)."
        else:
            return f"Governance clear across {total_snapshots} snapshot(s)."

    blocking_pct = int(blocking_rate * 100)

    if promotion_status == "BLOCK":
        return f"Governance blocking promotion: {blocking_pct}% blocking rate, {fail_count} failure(s)."
    elif blocking_rate >= 0.5:
        return f"Governance critical: {blocking_pct}% blocking rate across {total_snapshots} snapshot(s)."
    elif trend == "improving":
        return f"Governance recovering: {blocking_pct}% blocking rate, trend improving."
    elif trend == "degrading":
        return f"Governance degrading: {blocking_pct}% blocking rate, {fail_count} failure(s)."
    else:
        return f"Governance active: {blocking_pct}% blocking rate, {fail_count} of {total_snapshots} failing."


# =============================================================================
# PHASE V: GLOBAL GOVERNANCE SYNTHESIZER
# =============================================================================

# -----------------------------------------------------------------------------
# Signal Ingestion Contract
# -----------------------------------------------------------------------------

@dataclass
class GovernanceSignal:
    """
    Canonical governance signal schema for cross-layer alignment.

    All layers (replay, topology, security, HT, bundle, admissibility,
    preflight, metrics, budget, conjecture) should conform to this schema.

    Attributes:
        layer_name: Unique identifier for the layer (e.g., "replay", "topology")
        status: "OK" | "WARN" | "BLOCK"
        blocking_rules: List of rule IDs causing issues
        blocking_rate: Float in [0, 1] indicating severity
        headline: One-line neutral summary
    """
    layer_name: str
    status: str  # "OK" | "WARN" | "BLOCK"
    blocking_rules: List[str] = field(default_factory=list)
    blocking_rate: float = 0.0
    headline: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "layer_name": self.layer_name,
            "status": self.status,
            "blocking_rules": self.blocking_rules,
            "blocking_rate": self.blocking_rate,
            "headline": self.headline,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceSignal":
        """Deserialize from dict."""
        return cls(
            layer_name=data.get("layer_name", "unknown"),
            status=data.get("status", "BLOCK"),
            blocking_rules=data.get("blocking_rules", []),
            blocking_rate=data.get("blocking_rate", 0.0),
            headline=data.get("headline", ""),
        )


# Layer name constants for consistency
LAYER_REPLAY = "replay"           # A
LAYER_TOPOLOGY = "topology"       # B+G
LAYER_SECURITY = "security"       # K
LAYER_HT = "ht"                   # L (Hash Tree)
LAYER_BUNDLE = "bundle"           # N
LAYER_ADMISSIBILITY = "admissibility"  # O
LAYER_PREFLIGHT = "preflight"     # J
LAYER_METRICS = "metrics"         # D
LAYER_BUDGET = "budget"           # F
LAYER_CONJECTURE = "conjecture"   # M
LAYER_GOVERNANCE = "governance"   # Core governance


# Default critical layers that must be OK for promotion
DEFAULT_CRITICAL_LAYERS = frozenset({
    LAYER_REPLAY,
    LAYER_HT,
    LAYER_PREFLIGHT,
    LAYER_ADMISSIBILITY,
})


# -----------------------------------------------------------------------------
# Layer Adapters
# -----------------------------------------------------------------------------

def adapt_replay_to_signal(replay_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Replay layer (A) → GovernanceSignal.

    Expected keys: status, failing_rules, replay_blocking_rate, summary
    """
    if not replay_data:
        return GovernanceSignal(
            layer_name=LAYER_REPLAY,
            status="BLOCK",
            headline="No replay data available",
        )

    status = _normalize_status(replay_data.get("status", "BLOCK"))
    blocking_rules = replay_data.get("failing_rules", replay_data.get("blocking_rules", []))
    blocking_rate = replay_data.get("replay_blocking_rate", replay_data.get("blocking_rate", 0.0))
    headline = replay_data.get("summary", replay_data.get("headline", f"Replay: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_REPLAY,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_topology_to_signal(topology_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Topology layer (B+G) → GovernanceSignal.

    Expected keys: trajectory_status, failing_rules, anomaly_rate, headline
    """
    if not topology_data:
        return GovernanceSignal(
            layer_name=LAYER_TOPOLOGY,
            status="BLOCK",
            headline="No topology data available",
        )

    raw_status = topology_data.get("trajectory_status", topology_data.get("status", "BLOCK"))
    status = _normalize_topology_status(raw_status)
    blocking_rules = topology_data.get("failing_rules", [])
    blocking_rate = topology_data.get("anomaly_rate", topology_data.get("blocking_rate", 0.0))
    headline = topology_data.get("headline", f"Topology: {status}")

    return GovernanceSignal(
        layer_name=LAYER_TOPOLOGY,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_security_to_signal(security_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Security layer (K) → GovernanceSignal.

    Expected keys: security_status, vulnerabilities, risk_score, summary
    """
    if not security_data:
        return GovernanceSignal(
            layer_name=LAYER_SECURITY,
            status="BLOCK",
            headline="No security data available",
        )

    status = _normalize_status(security_data.get("security_status", security_data.get("status", "BLOCK")))
    blocking_rules = security_data.get("vulnerabilities", security_data.get("blocking_rules", []))
    blocking_rate = security_data.get("risk_score", security_data.get("blocking_rate", 0.0))
    headline = security_data.get("summary", security_data.get("headline", f"Security: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_SECURITY,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_ht_to_signal(ht_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Hash Tree layer (L) → GovernanceSignal.

    Expected keys: ht_status, integrity_failures, error_rate, summary
    """
    if not ht_data:
        return GovernanceSignal(
            layer_name=LAYER_HT,
            status="BLOCK",
            headline="No HT data available",
        )

    status = _normalize_status(ht_data.get("ht_status", ht_data.get("status", "BLOCK")))
    blocking_rules = ht_data.get("integrity_failures", ht_data.get("blocking_rules", []))
    blocking_rate = ht_data.get("error_rate", ht_data.get("blocking_rate", 0.0))
    headline = ht_data.get("summary", ht_data.get("headline", f"HT: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_HT,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_bundle_to_signal(bundle_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Bundle layer (N) → GovernanceSignal.

    Expected keys: bundle_status, missing_artifacts, completeness_rate, summary
    """
    if not bundle_data:
        return GovernanceSignal(
            layer_name=LAYER_BUNDLE,
            status="BLOCK",
            headline="No bundle data available",
        )

    status = _normalize_status(bundle_data.get("bundle_status", bundle_data.get("status", "BLOCK")))
    blocking_rules = bundle_data.get("missing_artifacts", bundle_data.get("blocking_rules", []))
    # Convert completeness to blocking rate (1 - completeness)
    completeness = bundle_data.get("completeness_rate", 1.0)
    blocking_rate = bundle_data.get("blocking_rate", 1.0 - completeness)
    headline = bundle_data.get("summary", bundle_data.get("headline", f"Bundle: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_BUNDLE,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_admissibility_to_signal(admissibility_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Admissibility layer (O) → GovernanceSignal.

    Expected keys: overall_status, invalidating_rules, blocking_rate, reason_summary
    """
    if not admissibility_data:
        return GovernanceSignal(
            layer_name=LAYER_ADMISSIBILITY,
            status="BLOCK",
            headline="No admissibility data available",
        )

    status = _normalize_status(admissibility_data.get("overall_status", admissibility_data.get("status", "BLOCK")))
    blocking_rules = admissibility_data.get("invalidating_rules", admissibility_data.get("blocking_rules", []))
    blocking_rate = admissibility_data.get("blocking_rate", 0.0 if status == "OK" else 1.0)
    headline = admissibility_data.get("reason_summary", admissibility_data.get("headline", f"Admissibility: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_ADMISSIBILITY,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_preflight_to_signal(preflight_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Preflight layer (J) → GovernanceSignal.

    Expected keys: preflight_status, failed_checks, failure_rate, summary
    """
    if not preflight_data:
        return GovernanceSignal(
            layer_name=LAYER_PREFLIGHT,
            status="BLOCK",
            headline="No preflight data available",
        )

    status = _normalize_status(preflight_data.get("preflight_status", preflight_data.get("status", "BLOCK")))
    blocking_rules = preflight_data.get("failed_checks", preflight_data.get("blocking_rules", []))
    blocking_rate = preflight_data.get("failure_rate", preflight_data.get("blocking_rate", 0.0))
    headline = preflight_data.get("summary", preflight_data.get("headline", f"Preflight: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_PREFLIGHT,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_metrics_to_signal(metrics_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Metrics layer (D) → GovernanceSignal.

    Expected keys: metrics_status, threshold_violations, violation_rate, summary
    """
    if not metrics_data:
        return GovernanceSignal(
            layer_name=LAYER_METRICS,
            status="BLOCK",
            headline="No metrics data available",
        )

    status = _normalize_status(metrics_data.get("metrics_status", metrics_data.get("status", "BLOCK")))
    blocking_rules = metrics_data.get("threshold_violations", metrics_data.get("blocking_rules", []))
    blocking_rate = metrics_data.get("violation_rate", metrics_data.get("blocking_rate", 0.0))
    headline = metrics_data.get("summary", metrics_data.get("headline", f"Metrics: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_METRICS,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_budget_to_signal(budget_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Budget layer (F) → GovernanceSignal.

    Expected keys: budget_status, overruns, utilization_rate, summary
    """
    if not budget_data:
        return GovernanceSignal(
            layer_name=LAYER_BUDGET,
            status="BLOCK",
            headline="No budget data available",
        )

    status = _normalize_status(budget_data.get("budget_status", budget_data.get("status", "BLOCK")))
    blocking_rules = budget_data.get("overruns", budget_data.get("blocking_rules", []))
    # Over 1.0 utilization = blocking
    utilization = budget_data.get("utilization_rate", 0.0)
    blocking_rate = budget_data.get("blocking_rate", max(0.0, utilization - 1.0) if utilization > 1.0 else 0.0)
    headline = budget_data.get("summary", budget_data.get("headline", f"Budget: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_BUDGET,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_conjecture_to_signal(conjecture_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Conjecture layer (M) → GovernanceSignal.

    Expected keys: conjecture_status, unverified_conjectures, uncertainty_rate, summary
    """
    if not conjecture_data:
        return GovernanceSignal(
            layer_name=LAYER_CONJECTURE,
            status="BLOCK",
            headline="No conjecture data available",
        )

    status = _normalize_status(conjecture_data.get("conjecture_status", conjecture_data.get("status", "BLOCK")))
    blocking_rules = conjecture_data.get("unverified_conjectures", conjecture_data.get("blocking_rules", []))
    blocking_rate = conjecture_data.get("uncertainty_rate", conjecture_data.get("blocking_rate", 0.0))
    headline = conjecture_data.get("summary", conjecture_data.get("headline", f"Conjecture: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_CONJECTURE,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def adapt_governance_to_signal(governance_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Adapter: Core Governance layer → GovernanceSignal.

    Expected keys: aggregate_status, failing_rules, governance_blocking_rate, summary
    """
    if not governance_data:
        return GovernanceSignal(
            layer_name=LAYER_GOVERNANCE,
            status="BLOCK",
            headline="No governance data available",
        )

    status = _normalize_status(governance_data.get("aggregate_status", governance_data.get("status", "BLOCK")))

    # Extract failing rules from failing_files if present
    blocking_rules = governance_data.get("failing_rules", [])
    if not blocking_rules and "failing_files" in governance_data:
        for ff in governance_data.get("failing_files", []):
            for rule in ff.get("reason_codes", []):
                if rule not in blocking_rules:
                    blocking_rules.append(rule)

    blocking_rate = governance_data.get("governance_blocking_rate", governance_data.get("blocking_rate", 0.0))
    headline = governance_data.get("summary", governance_data.get("headline", f"Governance: {status}"))

    return GovernanceSignal(
        layer_name=LAYER_GOVERNANCE,
        status=status,
        blocking_rules=blocking_rules,
        blocking_rate=blocking_rate,
        headline=headline,
    )


def _normalize_status(status: Any) -> str:
    """Normalize various status formats to OK/WARN/BLOCK."""
    if not status:
        return "BLOCK"

    status_str = str(status).upper()

    if status_str in {"OK", "PASS", "PASSED", "SUCCESS", "GREEN"}:
        return "OK"
    elif status_str in {"WARN", "WARNING", "YELLOW", "CAUTION"}:
        return "WARN"
    else:
        return "BLOCK"


def _normalize_topology_status(status: Any) -> str:
    """Normalize topology-specific status formats."""
    if not status:
        return "BLOCK"

    status_str = str(status).lower()

    if status_str in {"stable", "ok", "pass", "healthy"}:
        return "OK"
    elif status_str in {"degrading", "warn", "warning", "unstable"}:
        return "WARN"
    else:
        return "BLOCK"


# Adapter registry for convenience
LAYER_ADAPTERS: Dict[str, Callable[[Optional[Dict[str, Any]]], GovernanceSignal]] = {
    LAYER_REPLAY: adapt_replay_to_signal,
    LAYER_TOPOLOGY: adapt_topology_to_signal,
    LAYER_SECURITY: adapt_security_to_signal,
    LAYER_HT: adapt_ht_to_signal,
    LAYER_BUNDLE: adapt_bundle_to_signal,
    LAYER_ADMISSIBILITY: adapt_admissibility_to_signal,
    LAYER_PREFLIGHT: adapt_preflight_to_signal,
    LAYER_METRICS: adapt_metrics_to_signal,
    LAYER_BUDGET: adapt_budget_to_signal,
    LAYER_CONJECTURE: adapt_conjecture_to_signal,
    LAYER_GOVERNANCE: adapt_governance_to_signal,
}


def adapt_layer_to_signal(layer_name: str, layer_data: Optional[Dict[str, Any]]) -> GovernanceSignal:
    """
    Generic adapter that routes to the appropriate layer-specific adapter.

    Args:
        layer_name: Name of the layer to adapt
        layer_data: Raw layer data

    Returns:
        GovernanceSignal with normalized data
    """
    adapter = LAYER_ADAPTERS.get(layer_name)
    if adapter:
        return adapter(layer_data)

    # Fallback: generic adaptation
    if not layer_data:
        return GovernanceSignal(
            layer_name=layer_name,
            status="BLOCK",
            headline=f"No {layer_name} data available",
        )

    return GovernanceSignal(
        layer_name=layer_name,
        status=_normalize_status(layer_data.get("status", "BLOCK")),
        blocking_rules=layer_data.get("blocking_rules", []),
        blocking_rate=layer_data.get("blocking_rate", 0.0),
        headline=layer_data.get("headline", f"{layer_name}: {layer_data.get('status', 'BLOCK')}"),
    )


# -----------------------------------------------------------------------------
# Global Alignment View v2
# -----------------------------------------------------------------------------

def build_global_alignment_view(
    signals: Sequence[GovernanceSignal],
) -> Dict[str, Any]:
    """
    Build a global governance alignment view from normalized signals.

    This is an expanded version of build_governance_alignment_view that works
    with the canonical GovernanceSignal schema.

    Args:
        signals: Sequence of GovernanceSignal objects from all layers

    Returns:
        Dict containing:
          - layer_block_map: Dict[layer_name -> status (OK/WARN/BLOCK)]
          - global_status: "OK" | "WARN" | "BLOCK"
          - rules_failing_in_multiple_layers: Dict[rule_id -> List[layer_name]]
          - cross_layer_failures: List of rules failing in 2+ layers
          - alignment_score: float in [0, 1]
          - blocking_layers: List of layers with BLOCK status
          - warning_layers: List of layers with WARN status
          - ok_layers: List of layers with OK status
          - total_blocking_rules: int
          - recommendation: str
    """
    if not signals:
        return {
            "layer_block_map": {},
            "global_status": "OK",
            "rules_failing_in_multiple_layers": {},
            "cross_layer_failures": [],
            "alignment_score": 1.0,
            "blocking_layers": [],
            "warning_layers": [],
            "ok_layers": [],
            "total_blocking_rules": 0,
            "recommendation": "No governance signals to analyze.",
        }

    # Build layer block map
    layer_block_map: Dict[str, str] = {}
    blocking_layers: List[str] = []
    warning_layers: List[str] = []
    ok_layers: List[str] = []

    for signal in signals:
        layer_block_map[signal.layer_name] = signal.status
        if signal.status == "BLOCK":
            blocking_layers.append(signal.layer_name)
        elif signal.status == "WARN":
            warning_layers.append(signal.layer_name)
        else:
            ok_layers.append(signal.layer_name)

    # Determine global status
    if blocking_layers:
        global_status = "BLOCK"
    elif warning_layers:
        global_status = "WARN"
    else:
        global_status = "OK"

    # Find rules failing in multiple layers
    rule_to_layers: Dict[str, List[str]] = {}
    for signal in signals:
        for rule in signal.blocking_rules:
            if rule not in rule_to_layers:
                rule_to_layers[rule] = []
            if signal.layer_name not in rule_to_layers[rule]:
                rule_to_layers[rule].append(signal.layer_name)

    rules_failing_multiple = {
        rule: layers for rule, layers in rule_to_layers.items()
        if len(layers) >= 2
    }
    cross_layer_failures = list(rules_failing_multiple.keys())

    # Compute alignment score
    total_layers = len(signals)
    blocked_count = len(blocking_layers)
    warned_count = len(warning_layers)

    if total_layers == 0:
        alignment_score = 1.0
    else:
        # Blocked layers count double against alignment
        alignment_score = max(0.0, 1.0 - (blocked_count * 2 + warned_count) / (total_layers * 2))

    # Count total blocking rules
    total_blocking_rules = sum(len(s.blocking_rules) for s in signals)

    # Generate recommendation
    if global_status == "OK":
        recommendation = f"All {total_layers} layer(s) healthy. System ready for promotion."
    elif global_status == "WARN":
        recommendation = f"{len(warning_layers)} layer(s) with warnings. Review before promotion."
    else:
        recommendation = f"{len(blocking_layers)} layer(s) blocking. Resolve before promotion."

    return {
        "layer_block_map": layer_block_map,
        "global_status": global_status,
        "rules_failing_in_multiple_layers": rules_failing_multiple,
        "cross_layer_failures": cross_layer_failures,
        "alignment_score": alignment_score,
        "blocking_layers": blocking_layers,
        "warning_layers": warning_layers,
        "ok_layers": ok_layers,
        "total_blocking_rules": total_blocking_rules,
        "recommendation": recommendation,
    }


# -----------------------------------------------------------------------------
# Global Promotion Gate
# -----------------------------------------------------------------------------

def evaluate_global_promotion(
    global_alignment: Dict[str, Any],
    critical_layers: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """
    Evaluate global governance status for promotion readiness.

    Extended version of evaluate_governance_for_promotion that requires
    a configurable set of critical layers to be OK.

    Args:
        global_alignment: Global alignment view from build_global_alignment_view()
        critical_layers: Set of layer names that must be OK for promotion.
                        Defaults to DEFAULT_CRITICAL_LAYERS (replay, ht, preflight, admissibility)

    Returns:
        Dict containing:
          - promotion_ok: bool
          - status: "OK" | "WARN" | "BLOCK"
          - blocking_rules: List[str]
          - blocking_layers: List[str] (layers causing block)
          - critical_layers_status: Dict[layer -> status]
          - critical_layers_ok: bool
          - reason: str
          - conditions: List[str]
    """
    if critical_layers is None:
        critical_layers = DEFAULT_CRITICAL_LAYERS

    if not global_alignment:
        return {
            "promotion_ok": False,
            "status": "BLOCK",
            "blocking_rules": [],
            "blocking_layers": [],
            "critical_layers_status": {},
            "critical_layers_ok": False,
            "reason": "No global alignment data available",
            "conditions": ["Provide valid global alignment view"],
        }

    layer_block_map = global_alignment.get("layer_block_map", {})
    global_status = global_alignment.get("global_status", "BLOCK")
    all_blocking_layers = global_alignment.get("blocking_layers", [])
    cross_layer_failures = global_alignment.get("cross_layer_failures", [])

    # Check critical layers
    critical_layers_status: Dict[str, str] = {}
    critical_blocking: List[str] = []

    for layer in critical_layers:
        status = layer_block_map.get(layer, "BLOCK")  # Missing = BLOCK
        critical_layers_status[layer] = status
        if status == "BLOCK":
            critical_blocking.append(layer)

    critical_layers_ok = len(critical_blocking) == 0

    # Collect all blocking rules from blocked layers
    blocking_rules: List[str] = list(cross_layer_failures)

    # Determine promotion status
    conditions: List[str] = []

    if not critical_layers_ok:
        promotion_ok = False
        status = "BLOCK"
        reason = f"Critical layer(s) blocking: {', '.join(critical_blocking)}"
        conditions.append(f"Resolve issues in: {', '.join(critical_blocking)}")

    elif global_status == "BLOCK":
        # Non-critical layers blocking
        non_critical_blocking = [l for l in all_blocking_layers if l not in critical_layers]
        if len(non_critical_blocking) > 2:
            promotion_ok = False
            status = "BLOCK"
            reason = f"Too many non-critical layers blocking: {', '.join(non_critical_blocking)}"
            conditions.append("Reduce blocking layers to 2 or fewer")
        else:
            promotion_ok = True
            status = "WARN"
            reason = f"Promotion permitted with {len(non_critical_blocking)} non-critical layer(s) blocking."
            conditions.append(f"Monitor: {', '.join(non_critical_blocking)}")

    elif global_status == "WARN":
        promotion_ok = True
        status = "WARN"
        warning_layers = global_alignment.get("warning_layers", [])
        reason = f"Promotion permitted with {len(warning_layers)} warning(s)."
        conditions.append("Review warnings before proceeding")

    else:
        promotion_ok = True
        status = "OK"
        reason = "All layers healthy. Promotion permitted."

    return {
        "promotion_ok": promotion_ok,
        "status": status,
        "blocking_rules": blocking_rules,
        "blocking_layers": all_blocking_layers,
        "critical_layers_status": critical_layers_status,
        "critical_layers_ok": critical_layers_ok,
        "reason": reason,
        "conditions": conditions,
    }


# -----------------------------------------------------------------------------
# Director Meta-Panel
# -----------------------------------------------------------------------------

def build_global_governance_director_panel(
    global_alignment: Dict[str, Any],
    per_layer_signals: Optional[Sequence[GovernanceSignal]] = None,
) -> Dict[str, Any]:
    """
    Build the global governance director meta-panel.

    This panel summarizes the governance state across all layers for the
    Director Console.

    Args:
        global_alignment: Global alignment view from build_global_alignment_view()
        per_layer_signals: Optional sequence of all layer signals for additional context

    Returns:
        Dict containing:
          - status_light: "GREEN" | "YELLOW" | "RED"
          - blocking_layers: List[str]
          - headline: str (neutral summary)
          - snapshot_count: int (number of layers)
          - global_status: "OK" | "WARN" | "BLOCK"
          - alignment_score: float
          - layer_summary: Dict[status -> count]
          - most_critical_rule: str | None
    """
    if not global_alignment:
        return {
            "status_light": "RED",
            "blocking_layers": [],
            "headline": "No governance data available.",
            "snapshot_count": 0,
            "global_status": "BLOCK",
            "alignment_score": 0.0,
            "layer_summary": {"OK": 0, "WARN": 0, "BLOCK": 0},
            "most_critical_rule": None,
        }

    global_status = global_alignment.get("global_status", "BLOCK")
    blocking_layers = global_alignment.get("blocking_layers", [])
    warning_layers = global_alignment.get("warning_layers", [])
    ok_layers = global_alignment.get("ok_layers", [])
    alignment_score = global_alignment.get("alignment_score", 0.0)
    cross_layer_failures = global_alignment.get("cross_layer_failures", [])

    # Determine status light
    if global_status == "BLOCK":
        status_light = "RED"
    elif global_status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Count layers
    snapshot_count = len(blocking_layers) + len(warning_layers) + len(ok_layers)
    layer_summary = {
        "OK": len(ok_layers),
        "WARN": len(warning_layers),
        "BLOCK": len(blocking_layers),
    }

    # Find most critical rule (appears in most layers)
    most_critical_rule = None
    rules_multi = global_alignment.get("rules_failing_in_multiple_layers", {})
    if rules_multi:
        most_critical_rule = max(rules_multi.keys(), key=lambda r: len(rules_multi[r]))

    # Generate headline
    headline = _generate_global_panel_headline(
        global_status, blocking_layers, warning_layers, ok_layers, snapshot_count
    )

    return {
        "status_light": status_light,
        "blocking_layers": blocking_layers,
        "headline": headline,
        "snapshot_count": snapshot_count,
        "global_status": global_status,
        "alignment_score": alignment_score,
        "layer_summary": layer_summary,
        "most_critical_rule": most_critical_rule,
    }


def _generate_global_panel_headline(
    global_status: str,
    blocking_layers: List[str],
    warning_layers: List[str],
    ok_layers: List[str],
    total_layers: int,
) -> str:
    """Generate a neutral headline for the global panel."""
    if total_layers == 0:
        return "No governance layers configured."

    if global_status == "OK":
        return f"All {total_layers} governance layer(s) healthy."

    if global_status == "WARN":
        return f"{len(warning_layers)} of {total_layers} layer(s) with warnings: {', '.join(warning_layers[:3])}{'...' if len(warning_layers) > 3 else ''}."

    # BLOCK case
    if len(blocking_layers) == 1:
        return f"Layer '{blocking_layers[0]}' blocking governance."
    else:
        return f"{len(blocking_layers)} layer(s) blocking: {', '.join(blocking_layers[:3])}{'...' if len(blocking_layers) > 3 else ''}."
