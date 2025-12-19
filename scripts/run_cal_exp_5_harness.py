#!/usr/bin/env python3
"""
CAL-EXP-5: Variance Alignment — FAIL-CLOSE Avoidance Test (Harness)

BINDING REFERENCE: docs/system_law/calibration/CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md
IMPLEMENTATION REFERENCE: docs/system_law/calibration/CAL_EXP_5_IMPLEMENTATION_PLAN.md
FREEZE REFERENCE: docs/system_law/calibration/CAL_EXP_5_FREEZE.md

Extends CAL-EXP-4 harness with:
- experiment_id = "CAL-EXP-5"
- Output directory: results/cal_exp_5/
- Same variance profile generation as CAL-EXP-4
- Binary verdict (PASS/FAIL only, no PARTIAL)

Uses CAL-EXP-4 verifier without modification.

Execution without interpretation.
SHADOW MODE only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CONSTANTS (from frozen spec)
# =============================================================================

SPEC_REFERENCE = "CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md"
IMPL_REFERENCE = "CAL_EXP_5_IMPLEMENTATION_PLAN.md"
FREEZE_REFERENCE = "CAL_EXP_5_FREEZE.md"
SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "CAL-EXP-5"  # Changed from CAL-EXP-4

# Window bounds per CAL-EXP-3 (inherited)
WARM_UP_START = 0
WARM_UP_END = 200
EVAL_START = 201
EVAL_END = 1000

# Sub-windows per CAL-EXP-3 (inherited)
SUB_WINDOWS = [
    ("W1_early", 201, 400),
    ("W2_mid", 401, 600),
    ("W3_late", 601, 800),
    ("W4_final", 801, 1000),
]

# Frozen thresholds per CAL_EXP_4_FREEZE.md §2 (inherited, NO MODIFICATION)
VARIANCE_RATIO_MAX = 2.0
VARIANCE_RATIO_MIN = 0.5
IQR_RATIO_MAX = 2.0
WINDOWED_DRIFT_MAX = 0.05
CLAIM_CAP_THRESHOLD = 3.0
MIN_COVERAGE_RATIO = 1.0
MAX_GAP_RATIO_DIVERGENCE = 0.1

# FAIL-CLOSE codes per CAL_EXP_5_FREEZE.md §4
FAIL_CLOSE_CODES = {"F5.1", "F5.2", "F5.4", "F5.5", "F5.6"}
WARN_CODES = {"F5.3", "F5.7"}


# =============================================================================
# TOOLCHAIN FINGERPRINT (inherited from CAL-EXP-4)
# =============================================================================

def compute_toolchain_fingerprint() -> str:
    """Compute SHA-256 of runtime environment."""
    components = [
        f"python:{platform.python_version()}",
        f"platform:{platform.system()}:{platform.release()}",
        f"schema:{SCHEMA_VERSION}",
        f"experiment:{EXPERIMENT_ID}",
    ]
    uv_lock = Path("uv.lock")
    if uv_lock.exists():
        with open(uv_lock, "rb") as f:
            uv_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            components.append(f"uv_lock:{uv_hash}")

    content = "|".join(sorted(components))
    return hashlib.sha256(content.encode()).hexdigest()


def compute_uv_lock_hash() -> Optional[str]:
    """Compute SHA-256 of uv.lock file."""
    uv_lock = Path("uv.lock")
    if uv_lock.exists():
        with open(uv_lock, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return None


def compute_verifier_hash() -> Optional[str]:
    """Compute SHA-256 of verifier script for provenance."""
    verifier_path = Path("scripts/verify_cal_exp_4_run.py")
    if verifier_path.exists():
        with open(verifier_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return None


# =============================================================================
# SYNTHETIC CORPUS (inherited from CAL-EXP-4, identical variance profile)
# =============================================================================

class SyntheticCorpus:
    """Deterministic corpus generator with explicit variance profile."""

    def __init__(
        self,
        seed: int,
        total_cycles: int,
        noise_scale: float = 0.03,
        drift_rate: float = 0.002,
        spike_probability: float = 0.05,
    ):
        self._seed = seed
        self._total_cycles = total_cycles
        self._noise_scale = noise_scale
        self._drift_rate = drift_rate
        self._spike_probability = spike_probability
        self._rng = random.Random(seed)
        self._corpus: List[Dict[str, float]] = []
        self._generate()

    def _generate(self) -> None:
        """Pre-generate deterministic problem set."""
        H = 0.5
        rho = 0.7
        tau = 0.20
        beta = 0.1

        for cycle in range(1, self._total_cycles + 1):
            # Evolve state deterministically with parameterized variance
            H += self._rng.gauss(self._drift_rate, self._noise_scale)
            H = max(0.0, min(1.0, H))

            rho += self._rng.gauss(self._drift_rate * 0.5, self._noise_scale * 0.67)
            rho = max(0.0, min(1.0, rho))

            tau += self._rng.gauss(0, self._noise_scale * 0.33)
            tau = max(0.1, min(0.3, tau))

            if self._rng.random() < self._spike_probability:
                beta = min(1.0, beta + 0.1)
            else:
                beta = max(0.0, beta - 0.01)

            self._corpus.append({
                "cycle": cycle,
                "H": H,
                "rho": rho,
                "tau": tau,
                "beta": beta,
            })

    def get_cycle(self, cycle: int) -> Dict[str, float]:
        """Get corpus entry for cycle (1-indexed)."""
        return self._corpus[cycle - 1].copy()

    def compute_manifest_hash(self) -> str:
        """Compute hash of corpus for validity check."""
        content = json.dumps(self._corpus, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def variance_profile(self) -> Dict[str, float]:
        """Return declared variance profile."""
        return {
            "noise_scale": self._noise_scale,
            "drift_rate": self._drift_rate,
            "spike_probability": self._spike_probability,
        }


# =============================================================================
# ARM EXECUTION (inherited from CAL-EXP-4, no modification)
# =============================================================================

def execute_arm(
    corpus: SyntheticCorpus,
    seed: int,
    learning_enabled: bool,
    total_cycles: int,
) -> List[Dict[str, Any]]:
    """
    Execute one arm of CAL-EXP-5.

    Returns list of {"cycle": int, "delta_p": float} entries.
    """
    rng = random.Random(seed + (1000 if learning_enabled else 0))

    agent_H = 0.5
    agent_rho = 0.7
    agent_tau = 0.20
    agent_beta = 0.1

    lr = 0.1 if learning_enabled else 0.0
    noise = 0.02

    cycles_output = []

    for cycle in range(1, total_cycles + 1):
        real = corpus.get_cycle(cycle)

        H_alignment = 1.0 - abs(real["H"] - agent_H)
        rho_alignment = 1.0 - abs(real["rho"] - agent_rho)
        tau_alignment = 1.0 - abs(real["tau"] - agent_tau)
        beta_alignment = 1.0 - abs(real["beta"] - agent_beta)

        base_prob = 0.5
        alignment_bonus = 0.3 * (H_alignment + rho_alignment + tau_alignment + beta_alignment) / 4.0
        delta_p = base_prob + alignment_bonus
        delta_p = max(0.1, min(0.95, delta_p))

        cycles_output.append({
            "cycle": cycle,
            "delta_p": delta_p,
        })

        if learning_enabled:
            agent_H = agent_H * (1 - lr) + real["H"] * lr
            agent_H += rng.gauss(0, noise)
            agent_H = max(0.0, min(1.0, agent_H))

            agent_rho = agent_rho * (1 - lr) + real["rho"] * lr
            agent_rho += rng.gauss(0, noise)
            agent_rho = max(0.0, min(1.0, agent_rho))

            lr_tau = lr * 0.5
            agent_tau = agent_tau * (1 - lr_tau) + real["tau"] * lr_tau
            agent_tau += rng.gauss(0, noise * 0.5)
            agent_tau = max(0.0, min(1.0, agent_tau))

            if real["beta"] > 0.7:
                agent_beta = min(1.0, agent_beta + 0.05)
            else:
                agent_beta = max(0.0, agent_beta - 0.01 + rng.gauss(0, noise))

    return cycles_output


# =============================================================================
# STATISTICS HELPERS (inherited from CAL-EXP-4, no modification)
# =============================================================================

def compute_variance(values: List[float]) -> float:
    """Compute sample variance."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)


def compute_iqr(values: List[float]) -> Tuple[float, float, float]:
    """Compute Q1, median, Q3 and return (Q1, median, Q3)."""
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    q1 = percentile(0.25)
    median = percentile(0.50)
    q3 = percentile(0.75)
    return (q1, median, q3)


def check_pathology(values: List[float]) -> Tuple[bool, bool]:
    """Check for NaN and Inf values. Returns (has_nan, has_inf)."""
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    return (has_nan, has_inf)


# =============================================================================
# TEMPORAL STRUCTURE AUDIT (inherited from CAL-EXP-4, no modification)
# =============================================================================

def compute_arm_temporal_profile(
    cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    """Compute temporal structure profile for one arm."""
    eval_cycles = [c for c in cycles if eval_start <= c["cycle"] <= eval_end]

    if not eval_cycles:
        return {
            "cycle_count": 0,
            "cycle_min": 0,
            "cycle_max": 0,
            "cycle_gap_max": 0,
            "cycle_gap_mean": 0.0,
            "monotonic_cycle_indices": False,
            "timestamp_monotonic": True,
            "temporal_coverage_ratio": 0.0,
            "missing_cycles": list(range(eval_start, eval_end + 1)),
            "duplicate_cycles": None,
        }

    cycle_indices = [c["cycle"] for c in eval_cycles]
    sorted_indices = sorted(cycle_indices)
    expected_count = eval_end - eval_start + 1

    # Compute gaps
    gaps = [sorted_indices[i + 1] - sorted_indices[i] for i in range(len(sorted_indices) - 1)]
    gap_max = max(gaps) if gaps else 1
    gap_mean = sum(gaps) / len(gaps) if gaps else 1.0

    # Check monotonicity
    monotonic = all(cycle_indices[i] < cycle_indices[i + 1] for i in range(len(cycle_indices) - 1))

    # Coverage
    coverage_ratio = len(eval_cycles) / expected_count

    # Missing cycles
    expected_set = set(range(eval_start, eval_end + 1))
    actual_set = set(cycle_indices)
    missing = sorted(expected_set - actual_set) if expected_set != actual_set else None

    # Duplicates
    seen = set()
    duplicates = []
    for c in cycle_indices:
        if c in seen:
            duplicates.append(c)
        seen.add(c)
    duplicates = sorted(set(duplicates)) if duplicates else None

    return {
        "cycle_count": len(eval_cycles),
        "cycle_min": min(cycle_indices),
        "cycle_max": max(cycle_indices),
        "cycle_gap_max": gap_max,
        "cycle_gap_mean": gap_mean,
        "monotonic_cycle_indices": monotonic,
        "timestamp_monotonic": True,  # Synthetic harness guarantees this
        "temporal_coverage_ratio": coverage_ratio,
        "missing_cycles": missing,
        "duplicate_cycles": duplicates,
    }


def generate_temporal_structure_audit(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    """Generate temporal_structure_audit.json per frozen schema."""
    baseline_profile = compute_arm_temporal_profile(baseline_cycles, eval_start, eval_end)
    treatment_profile = compute_arm_temporal_profile(treatment_cycles, eval_start, eval_end)

    # Comparability checks per F5.1
    cycle_count_match = baseline_profile["cycle_count"] == treatment_profile["cycle_count"]

    baseline_indices = {c["cycle"] for c in baseline_cycles if eval_start <= c["cycle"] <= eval_end}
    treatment_indices = {c["cycle"] for c in treatment_cycles if eval_start <= c["cycle"] <= eval_end}
    cycle_indices_identical = baseline_indices == treatment_indices

    coverage_b = baseline_profile["temporal_coverage_ratio"]
    coverage_t = treatment_profile["temporal_coverage_ratio"]
    coverage_ratio_match = coverage_b >= MIN_COVERAGE_RATIO and coverage_t >= MIN_COVERAGE_RATIO

    gap_divergence = abs(baseline_profile["cycle_gap_mean"] - treatment_profile["cycle_gap_mean"])
    gap_structure_compatible = gap_divergence <= MAX_GAP_RATIO_DIVERGENCE

    # Composite check
    temporal_structure_compatible = (
        cycle_count_match
        and cycle_indices_identical
        and coverage_ratio_match
        and gap_structure_compatible
        and baseline_profile["monotonic_cycle_indices"]
        and treatment_profile["monotonic_cycle_indices"]
    )

    temporal_structure_pass = temporal_structure_compatible

    # Mismatch details (neutral description)
    mismatch_details = None
    if not temporal_structure_pass:
        issues = []
        if not cycle_count_match:
            issues.append(f"cycle_count: baseline={baseline_profile['cycle_count']}, treatment={treatment_profile['cycle_count']}")
        if not cycle_indices_identical:
            issues.append("cycle_indices differ")
        if not coverage_ratio_match:
            issues.append(f"coverage: baseline={coverage_b:.4f}, treatment={coverage_t:.4f}")
        if not gap_structure_compatible:
            issues.append(f"gap_divergence={gap_divergence:.4f}")
        if not baseline_profile["monotonic_cycle_indices"]:
            issues.append("baseline non-monotonic")
        if not treatment_profile["monotonic_cycle_indices"]:
            issues.append("treatment non-monotonic")
        mismatch_details = "; ".join(issues)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "baseline_arm": baseline_profile,
        "treatment_arm": treatment_profile,
        "comparability": {
            "cycle_count_match": cycle_count_match,
            "cycle_indices_identical": cycle_indices_identical,
            "coverage_ratio_match": coverage_ratio_match,
            "gap_structure_compatible": gap_structure_compatible,
            "temporal_structure_compatible": temporal_structure_compatible,
            "temporal_structure_pass": temporal_structure_pass,
            "mismatch_details": mismatch_details,
        },
        "thresholds": {
            "min_coverage_ratio": MIN_COVERAGE_RATIO,
            "max_gap_ratio_divergence": MAX_GAP_RATIO_DIVERGENCE,
        },
        "evaluation_window": {
            "start_cycle": eval_start,
            "end_cycle": eval_end,
            "expected_cycle_count": eval_end - eval_start + 1,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "_meta": {
            "generator": "run_cal_exp_5_harness.py",
            "note": "Temporal structure audit per CAL_EXP_4_FREEZE.md (inherited)",
        },
    }


# =============================================================================
# VARIANCE PROFILE AUDIT (inherited from CAL-EXP-4, no modification)
# =============================================================================

def compute_arm_variance_profile(
    cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    """Compute variance profile for one arm."""
    eval_cycles = [c for c in cycles if eval_start <= c["cycle"] <= eval_end]
    delta_p_values = [c["delta_p"] for c in eval_cycles]

    if not delta_p_values:
        return {
            "delta_p_count": 0,
            "delta_p_mean": 0.0,
            "delta_p_variance": 0.0,
            "delta_p_std": 0.0,
            "delta_p_iqr": 0.0,
            "delta_p_range": 0.0,
            "delta_p_min": 0.0,
            "delta_p_max": 0.0,
            "delta_p_median": None,
            "delta_p_q1": None,
            "delta_p_q3": None,
            "has_nan": False,
            "has_inf": False,
            "windowed_variances": None,
        }

    has_nan, has_inf = check_pathology(delta_p_values)

    n = len(delta_p_values)
    mean_dp = sum(delta_p_values) / n
    variance = compute_variance(delta_p_values)
    std = math.sqrt(variance)
    q1, median, q3 = compute_iqr(delta_p_values)
    iqr = q3 - q1
    dp_min = min(delta_p_values)
    dp_max = max(delta_p_values)
    dp_range = dp_max - dp_min

    # Windowed variances
    windowed_variances = []
    for _, w_start, w_end in SUB_WINDOWS:
        w_values = [c["delta_p"] for c in eval_cycles if w_start <= c["cycle"] <= w_end]
        if len(w_values) >= 2:
            windowed_variances.append(compute_variance(w_values))
        else:
            windowed_variances.append(0.0)

    return {
        "delta_p_count": n,
        "delta_p_mean": mean_dp,
        "delta_p_variance": variance,
        "delta_p_std": std,
        "delta_p_iqr": iqr,
        "delta_p_range": dp_range,
        "delta_p_min": dp_min,
        "delta_p_max": dp_max,
        "delta_p_median": median,
        "delta_p_q1": q1,
        "delta_p_q3": q3,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "windowed_variances": windowed_variances,
    }


def generate_variance_profile_audit(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    """Generate variance_profile_audit.json per frozen schema."""
    baseline_profile = compute_arm_variance_profile(baseline_cycles, eval_start, eval_end)
    treatment_profile = compute_arm_variance_profile(treatment_cycles, eval_start, eval_end)

    # Pathology check (F5.6)
    has_pathology = (
        baseline_profile["has_nan"]
        or baseline_profile["has_inf"]
        or treatment_profile["has_nan"]
        or treatment_profile["has_inf"]
    )

    # Variance ratio (F5.2)
    baseline_var = baseline_profile["delta_p_variance"]
    treatment_var = treatment_profile["delta_p_variance"]

    if baseline_var > 0:
        variance_ratio = treatment_var / baseline_var
    elif treatment_var > 0:
        variance_ratio = float("inf")
    else:
        variance_ratio = 1.0  # Both zero

    variance_ratio_acceptable = VARIANCE_RATIO_MIN <= variance_ratio <= VARIANCE_RATIO_MAX

    # Windowed drift (F5.3)
    baseline_windowed = baseline_profile.get("windowed_variances") or []
    treatment_windowed = treatment_profile.get("windowed_variances") or []

    windowed_variance_drift = 0.0
    if baseline_windowed and treatment_windowed:
        drifts = []
        for bw, tw in zip(baseline_windowed, treatment_windowed):
            if bw > 0:
                drifts.append(abs(tw / bw - 1.0))
            elif tw > 0:
                drifts.append(1.0)
            else:
                drifts.append(0.0)
        windowed_variance_drift = max(drifts) if drifts else 0.0

    windowed_drift_acceptable = windowed_variance_drift <= WINDOWED_DRIFT_MAX

    # IQR ratio (F5.7)
    baseline_iqr = baseline_profile["delta_p_iqr"]
    treatment_iqr = treatment_profile["delta_p_iqr"]

    if baseline_iqr > 0:
        iqr_ratio = treatment_iqr / baseline_iqr
    elif treatment_iqr > 0:
        iqr_ratio = float("inf")
    else:
        iqr_ratio = 1.0

    iqr_ratio_acceptable = iqr_ratio <= IQR_RATIO_MAX

    # Profile compatible (composite)
    profile_compatible = (
        variance_ratio_acceptable
        and windowed_drift_acceptable
        and iqr_ratio_acceptable
        and not has_pathology
    )

    variance_profile_pass = profile_compatible

    # Claim capping logic per FREEZE §4.2 (inherited from CAL-EXP-4)
    claim_cap_applied = False
    claim_cap_level = None

    if has_pathology:
        # F5.6: Pathological data -> L0
        claim_cap_applied = True
        claim_cap_level = "L0"
    elif not variance_ratio_acceptable:
        if variance_ratio > CLAIM_CAP_THRESHOLD or variance_ratio < (1 / CLAIM_CAP_THRESHOLD):
            # Severe mismatch -> L0
            claim_cap_applied = True
            claim_cap_level = "L0"
        else:
            # Moderate mismatch -> L3
            claim_cap_applied = True
            claim_cap_level = "L3"
    elif not windowed_drift_acceptable:
        # F5.3 -> L3
        claim_cap_applied = True
        claim_cap_level = "L3"
    elif not iqr_ratio_acceptable:
        # F5.7 -> L3
        claim_cap_applied = True
        claim_cap_level = "L3"

    # Mismatch details
    mismatch_details = None
    if not variance_profile_pass:
        issues = []
        if has_pathology:
            issues.append("pathological_data_detected")
        if not variance_ratio_acceptable:
            issues.append(f"variance_ratio={variance_ratio:.4f}")
        if not windowed_drift_acceptable:
            issues.append(f"windowed_drift={windowed_variance_drift:.4f}")
        if not iqr_ratio_acceptable:
            issues.append(f"iqr_ratio={iqr_ratio:.4f}")
        mismatch_details = "; ".join(issues)

    # Per-window analysis
    per_window_ratios = []
    if baseline_windowed and treatment_windowed:
        for bw, tw in zip(baseline_windowed, treatment_windowed):
            if bw > 0:
                per_window_ratios.append(tw / bw)
            else:
                per_window_ratios.append(0.0 if tw == 0 else float("inf"))

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "baseline_arm": baseline_profile,
        "treatment_arm": treatment_profile,
        "comparability": {
            "variance_ratio": variance_ratio,
            "variance_ratio_acceptable": variance_ratio_acceptable,
            "windowed_variance_drift": windowed_variance_drift,
            "windowed_drift_acceptable": windowed_drift_acceptable,
            "iqr_ratio": iqr_ratio,
            "iqr_ratio_acceptable": iqr_ratio_acceptable,
            "profile_compatible": profile_compatible,
            "variance_profile_pass": variance_profile_pass,
            "claim_cap_applied": claim_cap_applied,
            "claim_cap_level": claim_cap_level,
            "mismatch_details": mismatch_details,
        },
        "thresholds": {
            "variance_ratio_max": VARIANCE_RATIO_MAX,
            "variance_ratio_min": VARIANCE_RATIO_MIN,
            "windowed_drift_max": WINDOWED_DRIFT_MAX,
            "iqr_ratio_max": IQR_RATIO_MAX,
            "claim_cap_threshold": CLAIM_CAP_THRESHOLD,
        },
        "windowed_analysis": {
            "window_count": len(SUB_WINDOWS),
            "baseline_windowed_variances": baseline_windowed,
            "treatment_windowed_variances": treatment_windowed,
            "per_window_ratios": per_window_ratios,
        },
        "evaluation_window": {
            "start_cycle": eval_start,
            "end_cycle": eval_end,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "_meta": {
            "generator": "run_cal_exp_5_harness.py",
            "note": "Variance profile audit per CAL_EXP_4_FREEZE.md (inherited)",
        },
    }


# =============================================================================
# CAL-EXP-3 INHERITED ANALYSIS (no modification)
# =============================================================================

def compute_summary(cycles: List[Dict[str, Any]], eval_start: int, eval_end: int) -> Dict[str, Any]:
    """Compute summary statistics for an arm (inherited from CAL-EXP-3)."""
    eval_cycles = [c for c in cycles if eval_start <= c["cycle"] <= eval_end]

    if not eval_cycles:
        return {"error": "No cycles in evaluation window"}

    delta_p_values = [c["delta_p"] for c in eval_cycles]
    n = len(delta_p_values)
    mean_delta_p = sum(delta_p_values) / n
    variance = sum((x - mean_delta_p) ** 2 for x in delta_p_values) / n
    std_delta_p = math.sqrt(variance)

    return {
        "n_cycles": n,
        "evaluation_window": {
            "start_cycle": eval_start,
            "end_cycle": eval_end,
        },
        "mean_delta_p": mean_delta_p,
        "std_delta_p": std_delta_p,
        "min_delta_p": min(delta_p_values),
        "max_delta_p": max(delta_p_values),
    }


def compute_uplift_report(
    baseline_summary: Dict[str, Any],
    treatment_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute delta-delta-p per CAL-EXP-3 spec (inherited)."""
    baseline_mean = baseline_summary["mean_delta_p"]
    treatment_mean = treatment_summary["mean_delta_p"]
    delta_delta_p = treatment_mean - baseline_mean

    n_baseline = baseline_summary["n_cycles"]
    n_treatment = treatment_summary["n_cycles"]
    se_baseline = baseline_summary["std_delta_p"] / math.sqrt(n_baseline)
    se_treatment = treatment_summary["std_delta_p"] / math.sqrt(n_treatment)
    standard_error = math.sqrt(se_baseline**2 + se_treatment**2)
    noise_floor = 2 * baseline_summary["std_delta_p"] / math.sqrt(n_baseline)

    return {
        "baseline_mean_delta_p": baseline_mean,
        "treatment_mean_delta_p": treatment_mean,
        "delta_delta_p": delta_delta_p,
        "standard_error": standard_error,
        "noise_floor": noise_floor,
        "exceeds_noise_floor": abs(delta_delta_p) > noise_floor,
        "evaluation_window": baseline_summary["evaluation_window"],
        "n_baseline": n_baseline,
        "n_treatment": n_treatment,
    }


def compute_windowed_analysis(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-window breakdown (inherited from CAL-EXP-3)."""
    windows = {}

    for window_name, start, end in SUB_WINDOWS:
        baseline_window = [c for c in baseline_cycles if start <= c["cycle"] <= end]
        treatment_window = [c for c in treatment_cycles if start <= c["cycle"] <= end]

        if not baseline_window or not treatment_window:
            continue

        baseline_mean = sum(c["delta_p"] for c in baseline_window) / len(baseline_window)
        treatment_mean = sum(c["delta_p"] for c in treatment_window) / len(treatment_window)

        windows[window_name] = {
            "start_cycle": start,
            "end_cycle": end,
            "baseline_mean_delta_p": baseline_mean,
            "treatment_mean_delta_p": treatment_mean,
            "delta_delta_p": treatment_mean - baseline_mean,
            "n_cycles": len(baseline_window),
        }

    return {"windows": windows}


def run_validity_checks(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    toolchain_baseline: str,
    toolchain_treatment: str,
    corpus_hash: str,
) -> Dict[str, Any]:
    """Run CAL-EXP-3 validity condition checks (inherited)."""
    checks = {}

    checks["toolchain_parity"] = {
        "passed": toolchain_baseline == toolchain_treatment,
        "baseline_hash": toolchain_baseline,
        "treatment_hash": toolchain_treatment,
    }

    checks["corpus_identity"] = {
        "passed": True,
        "corpus_hash": corpus_hash,
    }

    baseline_cycles_set = {c["cycle"] for c in baseline_cycles if EVAL_START <= c["cycle"] <= EVAL_END}
    treatment_cycles_set = {c["cycle"] for c in treatment_cycles if EVAL_START <= c["cycle"] <= EVAL_END}
    checks["window_alignment"] = {
        "passed": baseline_cycles_set == treatment_cycles_set,
        "baseline_count": len(baseline_cycles_set),
        "treatment_count": len(treatment_cycles_set),
    }

    expected_cycles = set(range(EVAL_START, EVAL_END + 1))
    baseline_missing = expected_cycles - baseline_cycles_set
    treatment_missing = expected_cycles - treatment_cycles_set
    baseline_nan = any(math.isnan(c["delta_p"]) for c in baseline_cycles)
    treatment_nan = any(math.isnan(c["delta_p"]) for c in treatment_cycles)

    checks["no_pathology"] = {
        "passed": not baseline_missing and not treatment_missing and not baseline_nan and not treatment_nan,
        "baseline_missing_cycles": len(baseline_missing),
        "treatment_missing_cycles": len(treatment_missing),
        "baseline_nan": baseline_nan,
        "treatment_nan": treatment_nan,
    }

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "checks": checks,
        "all_passed": all_passed,
    }


def generate_isolation_audit() -> Dict[str, Any]:
    """Generate isolation audit (inherited from CAL-EXP-3)."""
    return {
        "network_calls": [],
        "file_reads_outside_corpus": [],
        "isolation_passed": True,
        "verification_method": "synthetic_harness_design_guarantee",
    }


# =============================================================================
# F5.x STATUS AGGREGATION (inherited from CAL-EXP-4, no modification)
# =============================================================================

def compute_f5_status(
    temporal_audit: Dict[str, Any],
    variance_audit: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate F5.x failure codes per FREEZE Section 3."""
    f5_codes = []

    # F5.1: Temporal structure incompatible
    if not temporal_audit["comparability"]["temporal_structure_pass"]:
        f5_codes.append("F5.1")

    # F5.2: Variance ratio out of bounds
    if not variance_audit["comparability"]["variance_ratio_acceptable"]:
        f5_codes.append("F5.2")

    # F5.3: Windowed drift excessive
    if not variance_audit["comparability"]["windowed_drift_acceptable"]:
        f5_codes.append("F5.3")

    # F5.6: Pathological data
    baseline_arm = variance_audit["baseline_arm"]
    treatment_arm = variance_audit["treatment_arm"]
    if baseline_arm.get("has_nan") or baseline_arm.get("has_inf") or treatment_arm.get("has_nan") or treatment_arm.get("has_inf"):
        f5_codes.append("F5.6")

    # F5.7: IQR ratio out of bounds
    if not variance_audit["comparability"]["iqr_ratio_acceptable"]:
        f5_codes.append("F5.7")

    # Aggregate pass
    temporal_pass = temporal_audit["comparability"]["temporal_structure_pass"]
    variance_pass = variance_audit["comparability"]["variance_profile_pass"]
    all_pass = temporal_pass and variance_pass

    # Claim cap from variance audit
    claim_cap_applied = variance_audit["comparability"]["claim_cap_applied"]
    claim_cap_level = variance_audit["comparability"]["claim_cap_level"]

    # Override cap if temporal fails
    if not temporal_pass:
        claim_cap_applied = True
        claim_cap_level = "L0"

    return {
        "f5_failure_codes": f5_codes,
        "temporal_comparability": temporal_pass,
        "variance_comparability": variance_pass,
        "f5_all_pass": all_pass,
        "claim_cap_applied": claim_cap_applied,
        "claim_cap_level": claim_cap_level,
    }


# =============================================================================
# CAL-EXP-5 SPECIFIC: BINARY VERDICT COMPUTATION
# =============================================================================

def compute_cal_exp_5_verdict(f5_failure_codes: List[str]) -> str:
    """
    Compute CAL-EXP-5 verdict per spec.

    PASS: No FAIL-CLOSE codes triggered
    FAIL: Any FAIL-CLOSE code triggered
    """
    if set(f5_failure_codes) & FAIL_CLOSE_CODES:
        return "FAIL"
    return "PASS"


def assign_claim_level(
    uplift_report: Dict[str, Any],
    validity: Dict[str, Any],
    f5_status: Dict[str, Any],
) -> str:
    """Assign claim level per CAL-EXP-4 rules (inherited)."""
    # F5.x caps take precedence
    if f5_status["claim_cap_applied"] and f5_status["claim_cap_level"]:
        cap = f5_status["claim_cap_level"]
        levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
        cap_idx = levels.index(cap)
    else:
        cap_idx = 5  # No cap

    # Base level from CAL-EXP-3 logic
    if uplift_report.get("error"):
        base_level = "L0"
    elif "baseline_mean_delta_p" not in uplift_report:
        base_level = "L1"
    elif "delta_delta_p" not in uplift_report:
        base_level = "L2"
    elif not uplift_report.get("exceeds_noise_floor", False):
        base_level = "L2"
    elif not validity.get("all_passed", False):
        base_level = "L3"
    elif not f5_status["f5_all_pass"]:
        base_level = "L3"
    else:
        base_level = "L4"

    # Apply cap
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    base_idx = levels.index(base_level)
    final_idx = min(base_idx, cap_idx)

    return levels[final_idx]


# =============================================================================
# ARTIFACT WRITERS (inherited from CAL-EXP-4, no modification)
# =============================================================================

def write_cycles_jsonl(cycles: List[Dict[str, Any]], path: Path) -> None:
    """Write cycles.jsonl per artifact contract."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in cycles:
            line = json.dumps({
                "cycle": c["cycle"],
                "delta_p": c["delta_p"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, sort_keys=True)
            f.write(line + "\n")


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON artifact with canonical formatting."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


# =============================================================================
# MAIN HARNESS
# =============================================================================

def run_experiment(
    seed: int,
    output_dir: Path,
    noise_scale: float,
    drift_rate: float,
    spike_probability: float,
) -> Dict[str, Any]:
    """Execute CAL-EXP-5 for a single pre-registered seed."""
    run_id = f"cal_exp_5_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id

    # Create directory structure per artifact contract
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "treatment").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "validity").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    toolchain_fingerprint = compute_toolchain_fingerprint()
    uv_lock_hash = compute_uv_lock_hash()
    verifier_hash = compute_verifier_hash()

    # Step 1-3: Write run_config.json with variance profile pre-registration
    run_config = {
        "experiment": EXPERIMENT_ID,
        "spec_reference": SPEC_REFERENCE,
        "impl_reference": IMPL_REFERENCE,
        "freeze_reference": FREEZE_REFERENCE,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "cycles": EVAL_END,
        "windows": {
            "warm_up_exclusion": {
                "start_cycle": WARM_UP_START,
                "end_cycle": WARM_UP_END,
                "included_in_analysis": False,
            },
            "evaluation_window": {
                "start_cycle": EVAL_START,
                "end_cycle": EVAL_END,
                "included_in_analysis": True,
            },
        },
        "baseline_config": {
            "learning_enabled": False,
            "rfl_active": False,
        },
        "treatment_config": {
            "learning_enabled": True,
            "rfl_active": True,
        },
        "variance_profile": {
            "noise_scale": noise_scale,
            "drift_rate": drift_rate,
            "spike_probability": spike_probability,
            "registered_at": timestamp,
        },
        "registered_at": timestamp,
        "seed_registered_at": timestamp,
        "seed_source": "pre-registered",
        "window_registered_at": timestamp,
    }
    write_json(run_config, run_dir / "run_config.json")

    # Step 4: Generate corpus with declared variance profile
    corpus = SyntheticCorpus(
        seed,
        EVAL_END,
        noise_scale=noise_scale,
        drift_rate=drift_rate,
        spike_probability=spike_probability,
    )
    corpus_hash = corpus.compute_manifest_hash()

    corpus_manifest = {
        "seed": seed,
        "total_cycles": EVAL_END,
        "corpus_hash": corpus_hash,
        "variance_profile": corpus.variance_profile,
        "generated_at": timestamp,
    }
    write_json(corpus_manifest, run_dir / "validity" / "corpus_manifest.json")

    # Step 5: Record toolchain hash
    with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
        f.write(toolchain_fingerprint)

    # Step 6-7: Execute baseline arm
    baseline_cycles = execute_arm(corpus, seed, learning_enabled=False, total_cycles=EVAL_END)
    write_cycles_jsonl(baseline_cycles, run_dir / "baseline" / "cycles.jsonl")

    baseline_summary = compute_summary(baseline_cycles, EVAL_START, EVAL_END)
    baseline_summary["arm"] = "baseline"
    baseline_summary["learning_enabled"] = False
    write_json(baseline_summary, run_dir / "baseline" / "summary.json")

    # Step 8-9: Execute treatment arm
    treatment_cycles = execute_arm(corpus, seed, learning_enabled=True, total_cycles=EVAL_END)
    write_cycles_jsonl(treatment_cycles, run_dir / "treatment" / "cycles.jsonl")

    treatment_summary = compute_summary(treatment_cycles, EVAL_START, EVAL_END)
    treatment_summary["arm"] = "treatment"
    treatment_summary["learning_enabled"] = True
    write_json(treatment_summary, run_dir / "treatment" / "summary.json")

    # Step 10: CAL-EXP-3 inherited validity checks
    validity = run_validity_checks(
        baseline_cycles,
        treatment_cycles,
        toolchain_fingerprint,
        toolchain_fingerprint,
        corpus_hash,
    )
    write_json(validity, run_dir / "validity" / "validity_checks.json")

    # Isolation audit
    isolation_audit = generate_isolation_audit()
    write_json(isolation_audit, run_dir / "validity" / "isolation_audit.json")

    if not isolation_audit["isolation_passed"]:
        validity["all_passed"] = False
        validity["checks"]["isolation"] = {"passed": False}

    # Step 11: Temporal structure audit
    temporal_audit = generate_temporal_structure_audit(
        baseline_cycles,
        treatment_cycles,
        EVAL_START,
        EVAL_END,
    )
    write_json(temporal_audit, run_dir / "validity" / "temporal_structure_audit.json")

    # Step 12: Variance profile audit
    variance_audit = generate_variance_profile_audit(
        baseline_cycles,
        treatment_cycles,
        EVAL_START,
        EVAL_END,
    )
    write_json(variance_audit, run_dir / "validity" / "variance_profile_audit.json")

    # Step 13: Compute delta-delta-p (inherited)
    uplift_report = compute_uplift_report(baseline_summary, treatment_summary)
    write_json(uplift_report, run_dir / "analysis" / "uplift_report.json")

    # Step 14: Windowed analysis (inherited)
    windowed = compute_windowed_analysis(baseline_cycles, treatment_cycles)
    write_json(windowed, run_dir / "analysis" / "windowed_analysis.json")

    # Step 15: F5.x status and claim level
    f5_status = compute_f5_status(temporal_audit, variance_audit)
    claim_level = assign_claim_level(uplift_report, validity, f5_status)

    # Step 16: CAL-EXP-5 specific binary verdict
    cal_exp_5_verdict = compute_cal_exp_5_verdict(f5_status["f5_failure_codes"])
    fail_close_triggered = cal_exp_5_verdict == "FAIL"

    # Identify WARN codes
    warn_codes_triggered = [code for code in f5_status["f5_failure_codes"] if code in WARN_CODES]

    # RUN_METADATA.json with CAL-EXP-5 verdict
    run_metadata = {
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "seed": seed,
        "mode": "SHADOW",
        "cal_exp_5_verdict": cal_exp_5_verdict,
        "f5_failure_codes": f5_status["f5_failure_codes"],
        "fail_close_triggered": fail_close_triggered,
        "warn_codes_triggered": warn_codes_triggered,
        "claim_cap_level": claim_level,
        "validity_passed": validity["all_passed"],
        "temporal_comparability_passed": f5_status["temporal_comparability"],
        "variance_comparability_passed": f5_status["variance_comparability"],
        "f5_all_pass": f5_status["f5_all_pass"],
        "claim_cap_applied": f5_status["claim_cap_applied"],
        "delta_delta_p": uplift_report["delta_delta_p"],
        "toolchain_fingerprint": toolchain_fingerprint,
        "uv_lock_hash": uv_lock_hash,
        "verifier_hash": verifier_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_metadata, run_dir / "RUN_METADATA.json")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "cal_exp_5_verdict": cal_exp_5_verdict,
        "f5_failure_codes": f5_status["f5_failure_codes"],
        "fail_close_triggered": fail_close_triggered,
        "warn_codes_triggered": warn_codes_triggered,
        "claim_cap_level": claim_level,
        "validity_passed": validity["all_passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAL-EXP-5: Variance Alignment — FAIL-CLOSE Avoidance Test (Harness)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Pre-registered seed for this run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cal_exp_5"),
        help="Output directory (default: results/cal_exp_5)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.03,
        help="Variance profile: noise scale (default: 0.03)",
    )
    parser.add_argument(
        "--drift-rate",
        type=float,
        default=0.002,
        help="Variance profile: drift rate (default: 0.002)",
    )
    parser.add_argument(
        "--spike-probability",
        type=float,
        default=0.05,
        help="Variance profile: spike probability (default: 0.05)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    result = run_experiment(
        args.seed,
        args.output_dir,
        args.noise_scale,
        args.drift_rate,
        args.spike_probability,
    )

    # Output per artifact contract
    print(f"run_id: {result['run_id']}")
    print(f"run_dir: {result['run_dir']}")
    print(f"cal_exp_5_verdict: {result['cal_exp_5_verdict']}")
    print(f"f5_failure_codes: {result['f5_failure_codes']}")
    print(f"fail_close_triggered: {result['fail_close_triggered']}")
    print(f"warn_codes_triggered: {result['warn_codes_triggered']}")
    print(f"claim_cap_level: {result['claim_cap_level']}")
    print(f"validity_passed: {result['validity_passed']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
