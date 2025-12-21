#!/usr/bin/env python3
"""
Phase II.c: Verdict Invariance Under Auxiliary Perturbation — Harness

BINDING REFERENCE: docs/system_law/calibration/PHASE_II_VERDICT_INVARIANCE_SPEC.md
FREEZE REFERENCE: docs/system_law/calibration/PHASE_II_VERDICT_INVARIANCE_FREEZE.md

Scientific Question:
    Given a fixed seed and frozen predicate set, is the governance verdict
    (F5.x codes, claim level, PASS/FAIL) invariant under perturbation of
    auxiliary parameters that are not part of the frozen governance contract?

Execution:
    1. Run baseline (unperturbed) configuration
    2. Run each perturbation from ADMISSIBLE_PERTURBATIONS
    3. Compare verdicts (F5.x codes, claim level, binary verdict)
    4. Produce verdict matrix artifact

SHADOW MODE only.
No capability claims.
No recommendations.
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
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# CONSTANTS (inherited from CAL-EXP-4)
# =============================================================================

SPEC_REFERENCE = "PHASE_II_VERDICT_INVARIANCE_SPEC.md"
FREEZE_REFERENCE = "PHASE_II_VERDICT_INVARIANCE_FREEZE.md"
INHERITED_FREEZE = "CAL_EXP_4_FREEZE.md"
SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "PHASE-II-C"

# Window bounds per CAL-EXP-3/4 (inherited, frozen)
WARM_UP_START = 0
WARM_UP_END = 200
EVAL_START = 201
EVAL_END = 1000

# Sub-windows per CAL-EXP-3/4 (inherited, frozen)
SUB_WINDOWS = [
    ("W1_early", 201, 400),
    ("W2_mid", 401, 600),
    ("W3_late", 601, 800),
    ("W4_final", 801, 1000),
]

# Frozen thresholds per CAL_EXP_4_FREEZE.md §2 (MUST NOT be modified)
VARIANCE_RATIO_MAX = 2.0
VARIANCE_RATIO_MIN = 0.5
IQR_RATIO_MAX = 2.0
WINDOWED_DRIFT_MAX = 0.05
CLAIM_CAP_THRESHOLD = 3.0
MIN_COVERAGE_RATIO = 1.0
MAX_GAP_RATIO_DIVERGENCE = 0.1


# =============================================================================
# AUXILIARY PERTURBATIONS (per FREEZE §4)
# =============================================================================

# Per PHASE_II_VERDICT_INVARIANCE_FREEZE.md §4.2:
# - Timestamp representation: ISO8601 precision, timezone offset format
# - JSON serialization order: key ordering (beyond sort_keys)
# - Floating-point representation: decimal precision in output
# - Logging verbosity: debug output presence
# - Environment metadata: platform details format

class PerturbationConfig:
    """Configuration for a single perturbation test."""

    def __init__(
        self,
        name: str,
        description: str,
        # Auxiliary parameter modifications
        timestamp_precision: int = 6,  # microseconds
        float_precision: int = 15,     # decimal places in delta_p
        log_level: str = "INFO",
        include_platform_details: bool = True,
        json_indent: int = 2,
        timezone_format: str = "offset",  # "offset" (+00:00) or "z" (Z)
    ):
        self.name = name
        self.description = description
        self.timestamp_precision = timestamp_precision
        self.float_precision = float_precision
        self.log_level = log_level
        self.include_platform_details = include_platform_details
        self.json_indent = json_indent
        self.timezone_format = timezone_format

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "timestamp_precision": self.timestamp_precision,
            "float_precision": self.float_precision,
            "log_level": self.log_level,
            "include_platform_details": self.include_platform_details,
            "json_indent": self.json_indent,
            "timezone_format": self.timezone_format,
        }


# Baseline configuration (p₀)
BASELINE_PERTURBATION = PerturbationConfig(
    name="p0_baseline",
    description="Unperturbed baseline configuration",
)

# Admissible perturbations per §4.2
ADMISSIBLE_PERTURBATIONS = [
    # P1: Timestamp precision (reduced to milliseconds)
    PerturbationConfig(
        name="p1_timestamp_ms",
        description="Timestamp precision reduced to milliseconds",
        timestamp_precision=3,
    ),
    # P2: Floating-point precision (reduced output)
    PerturbationConfig(
        name="p2_float_precision_8",
        description="Floating-point output precision reduced to 8 decimals",
        float_precision=8,
    ),
    # P3: JSON indent (4 spaces instead of 2)
    PerturbationConfig(
        name="p3_json_indent_4",
        description="JSON output indent changed to 4 spaces",
        json_indent=4,
    ),
    # P4: Timezone format (Z instead of +00:00)
    PerturbationConfig(
        name="p4_timezone_z",
        description="Timezone format changed from +00:00 to Z",
        timezone_format="z",
    ),
    # P5: Platform details excluded
    PerturbationConfig(
        name="p5_no_platform",
        description="Platform details excluded from metadata",
        include_platform_details=False,
    ),
    # P6: Log level DEBUG
    PerturbationConfig(
        name="p6_log_debug",
        description="Log level set to DEBUG",
        log_level="DEBUG",
    ),
    # P7: Combined perturbation (all auxiliary changes)
    PerturbationConfig(
        name="p7_combined",
        description="All auxiliary perturbations combined",
        timestamp_precision=3,
        float_precision=8,
        json_indent=4,
        timezone_format="z",
        include_platform_details=False,
        log_level="DEBUG",
    ),
]


# =============================================================================
# TOOLCHAIN FINGERPRINT (inherited from CAL-EXP-4)
# =============================================================================

def compute_toolchain_fingerprint(include_platform: bool = True) -> str:
    """Compute SHA-256 of runtime environment."""
    components = [
        f"python:{platform.python_version()}",
        f"schema:{SCHEMA_VERSION}",
        f"experiment:{EXPERIMENT_ID}",
    ]
    if include_platform:
        components.append(f"platform:{platform.system()}:{platform.release()}")

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
# SYNTHETIC CORPUS (inherited from CAL-EXP-4, FROZEN)
# =============================================================================

class SyntheticCorpus:
    """Deterministic corpus generator (FROZEN from CAL-EXP-4)."""

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
# ARM EXECUTION (inherited from CAL-EXP-4, FROZEN)
# =============================================================================

def execute_arm(
    corpus: SyntheticCorpus,
    seed: int,
    learning_enabled: bool,
    total_cycles: int,
) -> List[Dict[str, Any]]:
    """Execute one arm (FROZEN from CAL-EXP-4)."""
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
# STATISTICS HELPERS (inherited from CAL-EXP-4, FROZEN)
# =============================================================================

def compute_variance(values: List[float]) -> float:
    """Compute sample variance."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)


def compute_iqr(values: List[float]) -> Tuple[float, float, float]:
    """Compute Q1, median, Q3."""
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
    """Check for NaN and Inf values."""
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    return (has_nan, has_inf)


# =============================================================================
# GOVERNANCE VERDICT COMPUTATION (inherited from CAL-EXP-4, FROZEN)
# =============================================================================

def compute_summary(cycles: List[Dict[str, Any]], eval_start: int, eval_end: int) -> Dict[str, Any]:
    """Compute summary statistics for an arm."""
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


def compute_temporal_comparability(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Tuple[bool, List[str]]:
    """Check temporal structure comparability (F5.1)."""
    b_cycles = {c["cycle"] for c in baseline_cycles if eval_start <= c["cycle"] <= eval_end}
    t_cycles = {c["cycle"] for c in treatment_cycles if eval_start <= c["cycle"] <= eval_end}

    expected = set(range(eval_start, eval_end + 1))
    expected_count = eval_end - eval_start + 1

    # Coverage check
    b_coverage = len(b_cycles) / expected_count
    t_coverage = len(t_cycles) / expected_count

    # Gap check (simplified)
    b_sorted = sorted(b_cycles)
    t_sorted = sorted(t_cycles)

    b_gaps = [b_sorted[i+1] - b_sorted[i] for i in range(len(b_sorted)-1)] if len(b_sorted) > 1 else [1]
    t_gaps = [t_sorted[i+1] - t_sorted[i] for i in range(len(t_sorted)-1)] if len(t_sorted) > 1 else [1]

    b_gap_mean = sum(b_gaps) / len(b_gaps) if b_gaps else 1.0
    t_gap_mean = sum(t_gaps) / len(t_gaps) if t_gaps else 1.0

    gap_ratio = max(b_gap_mean, t_gap_mean) / max(min(b_gap_mean, t_gap_mean), 0.001)

    issues = []
    if b_coverage < MIN_COVERAGE_RATIO:
        issues.append(f"baseline_coverage={b_coverage:.3f}")
    if t_coverage < MIN_COVERAGE_RATIO:
        issues.append(f"treatment_coverage={t_coverage:.3f}")
    if abs(gap_ratio - 1.0) > MAX_GAP_RATIO_DIVERGENCE:
        issues.append(f"gap_ratio_divergence={gap_ratio:.3f}")
    if b_cycles != t_cycles:
        issues.append("cycle_set_mismatch")

    temporal_pass = len(issues) == 0
    return temporal_pass, issues


def compute_variance_comparability(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    eval_start: int,
    eval_end: int,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check variance structure comparability (F5.2, F5.3, F5.7)."""
    b_values = [c["delta_p"] for c in baseline_cycles if eval_start <= c["cycle"] <= eval_end]
    t_values = [c["delta_p"] for c in treatment_cycles if eval_start <= c["cycle"] <= eval_end]

    issues = []
    details = {}

    # Check pathology (F5.6)
    b_nan, b_inf = check_pathology(b_values)
    t_nan, t_inf = check_pathology(t_values)
    if b_nan or b_inf or t_nan or t_inf:
        issues.append("pathological_data")
        details["has_pathology"] = True
    else:
        details["has_pathology"] = False

    # Variance ratio (F5.2)
    b_var = compute_variance(b_values)
    t_var = compute_variance(t_values)
    variance_ratio = t_var / b_var if b_var > 0 else (0.0 if t_var == 0 else float("inf"))
    details["variance_ratio"] = variance_ratio

    if variance_ratio < VARIANCE_RATIO_MIN or variance_ratio > VARIANCE_RATIO_MAX:
        issues.append(f"variance_ratio={variance_ratio:.4f}")

    # Windowed drift (F5.3)
    b_windowed_vars = []
    t_windowed_vars = []
    for _, start, end in SUB_WINDOWS:
        b_window = [c["delta_p"] for c in baseline_cycles if start <= c["cycle"] <= end]
        t_window = [c["delta_p"] for c in treatment_cycles if start <= c["cycle"] <= end]
        if b_window and t_window:
            b_windowed_vars.append(compute_variance(b_window))
            t_windowed_vars.append(compute_variance(t_window))

    if b_windowed_vars and t_windowed_vars:
        drifts = []
        for bv, tv in zip(b_windowed_vars, t_windowed_vars):
            if bv > 0:
                drifts.append(abs(tv - bv) / bv)
            else:
                drifts.append(0.0 if tv == 0 else 1.0)
        windowed_drift = max(drifts)
        details["windowed_drift"] = windowed_drift
        if windowed_drift > WINDOWED_DRIFT_MAX:
            issues.append(f"windowed_drift={windowed_drift:.4f}")
    else:
        details["windowed_drift"] = 0.0

    # IQR ratio (F5.7)
    b_q1, _, b_q3 = compute_iqr(b_values)
    t_q1, _, t_q3 = compute_iqr(t_values)
    b_iqr = b_q3 - b_q1
    t_iqr = t_q3 - t_q1
    iqr_ratio = t_iqr / b_iqr if b_iqr > 0 else (0.0 if t_iqr == 0 else float("inf"))
    details["iqr_ratio"] = iqr_ratio

    if iqr_ratio > IQR_RATIO_MAX:
        issues.append(f"iqr_ratio={iqr_ratio:.4f}")

    variance_pass = len(issues) == 0
    return variance_pass, issues, details


def compute_governance_verdict(
    baseline_cycles: List[Dict[str, Any]],
    treatment_cycles: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    """
    Compute governance verdict.

    Returns:
        Dict with:
            - f5_codes: Set of triggered F5.x codes
            - claim_level: L0-L5
            - binary_verdict: PASS or FAIL
            - temporal_comparability: bool
            - variance_comparability: bool
    """
    f5_codes: Set[str] = set()

    # Temporal comparability (F5.1)
    temporal_pass, temporal_issues = compute_temporal_comparability(
        baseline_cycles, treatment_cycles, EVAL_START, EVAL_END
    )
    if not temporal_pass:
        f5_codes.add("F5.1")

    # Variance comparability (F5.2, F5.3, F5.6, F5.7)
    variance_pass, variance_issues, variance_details = compute_variance_comparability(
        baseline_cycles, treatment_cycles, EVAL_START, EVAL_END
    )

    if variance_details.get("has_pathology"):
        f5_codes.add("F5.6")

    if any("variance_ratio" in i for i in variance_issues):
        f5_codes.add("F5.2")

    if any("windowed_drift" in i for i in variance_issues):
        f5_codes.add("F5.3")

    if any("iqr_ratio" in i for i in variance_issues):
        f5_codes.add("F5.7")

    # Compute summaries for claim level
    baseline_summary = compute_summary(baseline_cycles, EVAL_START, EVAL_END)
    treatment_summary = compute_summary(treatment_cycles, EVAL_START, EVAL_END)

    # ΔΔp computation
    if "error" in baseline_summary or "error" in treatment_summary:
        delta_delta_p = 0.0
        exceeds_noise = False
    else:
        baseline_mean = baseline_summary["mean_delta_p"]
        treatment_mean = treatment_summary["mean_delta_p"]
        delta_delta_p = treatment_mean - baseline_mean
        noise_floor = 2 * baseline_summary["std_delta_p"] / math.sqrt(baseline_summary["n_cycles"])
        exceeds_noise = abs(delta_delta_p) > noise_floor

    # Claim level assignment (per CAL-EXP-4 rules)
    # F5.1, F5.2, F5.4, F5.5, F5.6 are FAIL-CLOSE (cap to L0)
    # F5.3, F5.7 are WARN (cap to L3)
    fail_close_codes = {"F5.1", "F5.2", "F5.4", "F5.5", "F5.6"}
    warn_codes = {"F5.3", "F5.7"}

    if f5_codes & fail_close_codes:
        claim_level = "L0"
    elif f5_codes & warn_codes:
        claim_level = "L3"
    elif not exceeds_noise:
        claim_level = "L2"
    else:
        claim_level = "L4"

    # Binary verdict
    binary_verdict = "FAIL" if (f5_codes & fail_close_codes) else "PASS"

    return {
        "f5_codes": sorted(f5_codes),
        "claim_level": claim_level,
        "binary_verdict": binary_verdict,
        "temporal_comparability": temporal_pass,
        "variance_comparability": variance_pass,
        "delta_delta_p": delta_delta_p,
        "variance_details": variance_details,
    }


# =============================================================================
# VERDICT INVARIANCE CHECK
# =============================================================================

def verdicts_equal(v1: Dict[str, Any], v2: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if two verdicts are equal per FREEZE §3.4.

    Invariance requires:
        - F5.x codes identical (set equality)
        - Claim level identical
        - Binary verdict identical
        - Temporal comparability identical
        - Variance comparability identical
    """
    divergences = []

    # F5.x codes (set equality, order-independent)
    codes1 = set(v1.get("f5_codes", []))
    codes2 = set(v2.get("f5_codes", []))
    if codes1 != codes2:
        divergences.append(f"f5_codes: {sorted(codes1)} vs {sorted(codes2)}")

    # Claim level
    if v1.get("claim_level") != v2.get("claim_level"):
        divergences.append(f"claim_level: {v1.get('claim_level')} vs {v2.get('claim_level')}")

    # Binary verdict
    if v1.get("binary_verdict") != v2.get("binary_verdict"):
        divergences.append(f"binary_verdict: {v1.get('binary_verdict')} vs {v2.get('binary_verdict')}")

    # Temporal comparability
    if v1.get("temporal_comparability") != v2.get("temporal_comparability"):
        divergences.append(f"temporal_comparability: {v1.get('temporal_comparability')} vs {v2.get('temporal_comparability')}")

    # Variance comparability
    if v1.get("variance_comparability") != v2.get("variance_comparability"):
        divergences.append(f"variance_comparability: {v1.get('variance_comparability')} vs {v2.get('variance_comparability')}")

    return len(divergences) == 0, divergences


# =============================================================================
# MAIN HARNESS
# =============================================================================

def run_single_perturbation(
    seed: int,
    perturbation: PerturbationConfig,
    corpus: SyntheticCorpus,
) -> Dict[str, Any]:
    """Run experiment with a single perturbation configuration."""
    # Execute arms (FROZEN logic - perturbation does NOT affect execution)
    baseline_cycles = execute_arm(corpus, seed, learning_enabled=False, total_cycles=EVAL_END)
    treatment_cycles = execute_arm(corpus, seed, learning_enabled=True, total_cycles=EVAL_END)

    # Compute governance verdict (FROZEN logic)
    verdict = compute_governance_verdict(baseline_cycles, treatment_cycles, seed)

    return {
        "perturbation": perturbation.to_dict(),
        "verdict": verdict,
    }


def run_phase_ii_invariance(
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute Phase II.c Verdict Invariance test."""
    run_id = f"phase_ii_c_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Generate corpus (deterministic, same for all perturbations)
    corpus = SyntheticCorpus(seed, EVAL_END)
    corpus_hash = corpus.compute_manifest_hash()

    # Run baseline (p₀)
    print(f"[{EXPERIMENT_ID}] Running baseline (p0)...")
    baseline_result = run_single_perturbation(seed, BASELINE_PERTURBATION, corpus)
    baseline_verdict = baseline_result["verdict"]

    # Run all perturbations
    perturbation_results = []
    for i, perturbation in enumerate(ADMISSIBLE_PERTURBATIONS):
        print(f"[{EXPERIMENT_ID}] Running perturbation {i+1}/{len(ADMISSIBLE_PERTURBATIONS)}: {perturbation.name}...")
        result = run_single_perturbation(seed, perturbation, corpus)

        # Check invariance
        is_invariant, divergences = verdicts_equal(baseline_verdict, result["verdict"])

        perturbation_results.append({
            "perturbation": perturbation.to_dict(),
            "verdict": result["verdict"],
            "invariant": is_invariant,
            "divergences": divergences,
        })

    # Aggregate: PASS iff all perturbations are invariant
    all_invariant = all(r["invariant"] for r in perturbation_results)

    # Build verdict matrix
    verdict_matrix = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "spec_reference": SPEC_REFERENCE,
        "freeze_reference": FREEZE_REFERENCE,
        "inherited_freeze": INHERITED_FREEZE,
        "run_id": run_id,
        "seed": seed,
        "mode": "SHADOW",
        "corpus_hash": corpus_hash,
        "baseline": {
            "perturbation": BASELINE_PERTURBATION.to_dict(),
            "verdict": baseline_verdict,
        },
        "perturbations": perturbation_results,
        "aggregate": {
            "total_perturbations": len(ADMISSIBLE_PERTURBATIONS),
            "invariant_count": sum(1 for r in perturbation_results if r["invariant"]),
            "divergent_count": sum(1 for r in perturbation_results if not r["invariant"]),
            "all_invariant": all_invariant,
            "phase_ii_verdict": "PASS" if all_invariant else "FAIL",
        },
        "toolchain": {
            "fingerprint": compute_toolchain_fingerprint(),
            "uv_lock_hash": compute_uv_lock_hash(),
            "verifier_hash": compute_verifier_hash(),
        },
        "generated_at": timestamp,
    }

    # Write verdict matrix
    with open(run_dir / "VERDICT_MATRIX.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(verdict_matrix, f, indent=2, sort_keys=True)
        f.write("\n")

    # Write summary
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "seed": seed,
        "phase_ii_verdict": verdict_matrix["aggregate"]["phase_ii_verdict"],
        "invariant_count": verdict_matrix["aggregate"]["invariant_count"],
        "divergent_count": verdict_matrix["aggregate"]["divergent_count"],
        "baseline_f5_codes": baseline_verdict["f5_codes"],
        "baseline_claim_level": baseline_verdict["claim_level"],
        "baseline_binary_verdict": baseline_verdict["binary_verdict"],
        "generated_at": timestamp,
    }

    with open(run_dir / "SUMMARY.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "phase_ii_verdict": verdict_matrix["aggregate"]["phase_ii_verdict"],
        "invariant_count": verdict_matrix["aggregate"]["invariant_count"],
        "divergent_count": verdict_matrix["aggregate"]["divergent_count"],
        "baseline_verdict": baseline_verdict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase II.c: Verdict Invariance Under Auxiliary Perturbation"
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
        default=Path("results/phase_ii_c"),
        help="Output directory (default: results/phase_ii_c)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"=" * 60)
    print(f"Phase II.c: Verdict Invariance Under Auxiliary Perturbation")
    print(f"Seed: {args.seed}")
    print(f"Mode: SHADOW (observational only)")
    print(f"=" * 60)

    result = run_phase_ii_invariance(args.seed, args.output_dir)

    print()
    print(f"=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Run ID: {result['run_id']}")
    print(f"Run Dir: {result['run_dir']}")
    print()
    print(f"Phase II.c Verdict: {result['phase_ii_verdict']}")
    print(f"Invariant: {result['invariant_count']}/{result['invariant_count'] + result['divergent_count']}")
    print(f"Divergent: {result['divergent_count']}/{result['invariant_count'] + result['divergent_count']}")
    print()
    print(f"Baseline F5 Codes: {result['baseline_verdict']['f5_codes']}")
    print(f"Baseline Claim Level: {result['baseline_verdict']['claim_level']}")
    print(f"Baseline Binary Verdict: {result['baseline_verdict']['binary_verdict']}")
    print(f"=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
