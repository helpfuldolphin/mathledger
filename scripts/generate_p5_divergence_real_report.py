#!/usr/bin/env python3
"""
P5 Real Telemetry Divergence Report Generator

Generates p5_divergence_real.json from a P5 run directory containing:
- p4_summary.json (required)
- divergence_log.jsonl (required)
- validation results from RealTelemetryAdapter (optional)
- TDA comparison metrics (optional)

SHADOW MODE CONTRACT:
- All outputs are observational only
- No gating or enforcement logic
- mode="SHADOW" is always set in output

See: docs/system_law/schemas/p5/p5_divergence_real.schema.json

Usage:
    python scripts/generate_p5_divergence_real_report.py --p5-run-dir /path/to/run --output /path/to/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Schema validation (optional)
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "1.1.0"
DEFAULT_OUTPUT_FILENAME = "p5_divergence_real.json"

# Pattern classification thresholds
DRIFT_BIAS_THRESHOLD = 0.05
NOISE_VARIANCE_THRESHOLD = 0.1
PHASE_LAG_TIMING_THRESHOLD = 0.3
STRUCTURAL_BREAK_THRESHOLD = 0.2

# Validation status thresholds
VALIDATION_CONFIDENCE_HIGH = 0.85
VALIDATION_CONFIDENCE_MEDIUM = 0.60


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DivergenceStats:
    """Aggregated divergence statistics from divergence_log.jsonl."""
    total_cycles: int = 0
    divergent_cycles: int = 0
    success_diverged: int = 0
    blocked_diverged: int = 0
    omega_diverged: int = 0

    # State deltas
    H_deltas: List[float] = field(default_factory=list)
    rho_deltas: List[float] = field(default_factory=list)
    tau_deltas: List[float] = field(default_factory=list)

    # Severity distribution
    severity_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def divergence_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.divergent_cycles / self.total_cycles


@dataclass
class ManifoldValidation:
    """RTTS manifold validation results."""
    boundedness_ok: bool = True
    continuity_ok: bool = True
    correlation_ok: bool = True
    violations: List[str] = field(default_factory=list)


@dataclass
class TDAComparison:
    """TDA metric comparison between real and twin."""
    sns_delta: Optional[float] = None
    pcs_delta: Optional[float] = None
    drs_delta: Optional[float] = None
    hss_delta: Optional[float] = None


@dataclass
class DivergenceDecomposition:
    """Divergence decomposition per RTTS Section 3.2."""
    bias: float = 0.0
    variance: float = 0.0
    timing: float = 0.0
    structural: float = 0.0


@dataclass
class TrueDivergenceVectorV1:
    """
    True Divergence Vector v1 — NO METRIC LAUNDERING.

    Explicit, named fields for each divergence component:
    - safety_state_mismatch_rate: Fraction of cycles where safety-critical state differs
    - state_error_mean: Mean absolute state delta (mean_delta_p proxy)
    - outcome_brier_score: Optional Brier score for outcome prediction calibration

    SHADOW MODE: All fields are observational only.
    """
    safety_state_mismatch_rate: float = 0.0
    state_error_mean: Optional[float] = None
    outcome_brier_score: Optional[float] = None


@dataclass
class SourceReference:
    """Reference to an input source file used in report generation."""
    name: str
    path: str
    sha256: str
    present: bool = True


# =============================================================================
# File Loading
# =============================================================================

def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_p4_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load p4_summary.json from run directory."""
    summary_path = run_dir / "p4_summary.json"
    if not summary_path.exists():
        # Also check p4_shadow subdirectory
        summary_path = run_dir / "p4_shadow" / "p4_summary.json"
        if not summary_path.exists():
            return None

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_divergence_log(run_dir: Path) -> List[Dict[str, Any]]:
    """Load divergence_log.jsonl from run directory."""
    log_path = run_dir / "divergence_log.jsonl"
    if not log_path.exists():
        log_path = run_dir / "p4_shadow" / "divergence_log.jsonl"
        if not log_path.exists():
            return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_validation_results(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load RealTelemetryAdapter validation results if available."""
    # Check common locations for validation results
    candidates = [
        run_dir / "validation_results.json",
        run_dir / "p4_shadow" / "validation_results.json",
        run_dir / "real_telemetry_validation.json",
        run_dir / "p4_shadow" / "real_telemetry_validation.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_tda_comparison(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load TDA comparison metrics if available."""
    candidates = [
        run_dir / "tda_comparison.json",
        run_dir / "p4_shadow" / "tda_comparison.json",
        run_dir / "tda_metrics.json",
        run_dir / "p4_shadow" / "tda_metrics.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_calibration_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load twin calibration report if available."""
    candidates = [
        run_dir / "calibration_report.json",
        run_dir / "p4_shadow" / "calibration_report.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_rtts_validation(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load rtts_validation.json from run directory.

    P5.2 VALIDATE STAGE:
    - Loads RTTS validation results emitted by P4 harness
    - Contains mock detection flags, statistical validation, continuity checks
    - All fields are optional for downstream consumers

    SHADOW MODE CONTRACT:
    - This file is for observational analysis only
    - It does NOT gate any operations
    - Mock detection flags are advisory

    Args:
        run_dir: Path to run directory

    Returns:
        RTTS validation dict, or None if not found
    """
    candidates = [
        run_dir / "rtts_validation.json",
        run_dir / "p4_shadow" / "rtts_validation.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def find_source_file(run_dir: Path, candidates: List[Path]) -> Optional[Path]:
    """Find the first existing file from candidate paths."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_sources_block(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Build the sources block listing which optional inputs were present.

    Returns a deterministically ordered list of source references.
    Order is fixed: rtts_validation, tda_comparison, calibration_report.
    Only present sources are included.

    SHADOW MODE: This is purely observational metadata.
    """
    sources = []

    # RTTS Validation - check first (canonical source for real telemetry validation)
    rtts_candidates = [
        run_dir / "rtts_validation.json",
        run_dir / "p4_shadow" / "rtts_validation.json",
    ]
    rtts_path = find_source_file(run_dir, rtts_candidates)
    if rtts_path:
        sources.append({
            "name": "rtts_validation",
            "path": str(rtts_path.relative_to(run_dir)),
            "sha256": compute_file_sha256(rtts_path),
        })

    # TDA Comparison - topological data analysis metrics
    tda_candidates = [
        run_dir / "tda_comparison.json",
        run_dir / "p4_shadow" / "tda_comparison.json",
        run_dir / "tda_metrics.json",
        run_dir / "p4_shadow" / "tda_metrics.json",
    ]
    tda_path = find_source_file(run_dir, tda_candidates)
    if tda_path:
        sources.append({
            "name": "tda_comparison",
            "path": str(tda_path.relative_to(run_dir)),
            "sha256": compute_file_sha256(tda_path),
        })

    # Calibration Report - twin calibration results
    cal_candidates = [
        run_dir / "calibration_report.json",
        run_dir / "p4_shadow" / "calibration_report.json",
    ]
    cal_path = find_source_file(run_dir, cal_candidates)
    if cal_path:
        sources.append({
            "name": "calibration_report",
            "path": str(cal_path.relative_to(run_dir)),
            "sha256": compute_file_sha256(cal_path),
        })

    return sources


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_divergence_stats(entries: List[Dict[str, Any]]) -> DivergenceStats:
    """Compute aggregated divergence statistics from log entries."""
    stats = DivergenceStats()
    stats.total_cycles = len(entries)

    for entry in entries:
        # Count divergent cycles
        is_divergent = (
            entry.get("success_diverged", False) or
            entry.get("blocked_diverged", False) or
            entry.get("omega_diverged", False)
        )
        if is_divergent:
            stats.divergent_cycles += 1

        if entry.get("success_diverged", False):
            stats.success_diverged += 1
        if entry.get("blocked_diverged", False):
            stats.blocked_diverged += 1
        if entry.get("omega_diverged", False):
            stats.omega_diverged += 1

        # Collect deltas
        if "H_delta" in entry:
            stats.H_deltas.append(abs(entry["H_delta"]))
        if "rho_delta" in entry:
            stats.rho_deltas.append(abs(entry["rho_delta"]))
        if "tau_delta" in entry:
            stats.tau_deltas.append(abs(entry["tau_delta"]))

        # Count severities
        severity = entry.get("severity", "NONE")
        stats.severity_counts[severity] = stats.severity_counts.get(severity, 0) + 1

    return stats


def compute_twin_tracking_accuracy(
    summary: Dict[str, Any],
    stats: DivergenceStats
) -> Dict[str, float]:
    """Compute twin tracking accuracy metrics."""
    accuracy = {}

    # Get accuracy from summary if available
    if "twin_success_accuracy" in summary:
        accuracy["success"] = summary["twin_success_accuracy"]
    elif stats.total_cycles > 0:
        accuracy["success"] = 1.0 - (stats.success_diverged / stats.total_cycles)

    if "twin_omega_accuracy" in summary:
        accuracy["omega"] = summary["twin_omega_accuracy"]
    elif stats.total_cycles > 0:
        accuracy["omega"] = 1.0 - (stats.omega_diverged / stats.total_cycles)

    if "twin_blocked_accuracy" in summary:
        accuracy["blocked"] = summary["twin_blocked_accuracy"]
    elif stats.total_cycles > 0:
        accuracy["blocked"] = 1.0 - (stats.blocked_diverged / stats.total_cycles)

    return accuracy


def compute_manifold_validation(
    stats: DivergenceStats,
    validation: Optional[Dict[str, Any]]
) -> ManifoldValidation:
    """Compute RTTS manifold validation results."""
    result = ManifoldValidation()

    # Check boundedness (all values in [0,1])
    # This is typically ensured by the telemetry adapter
    if validation:
        result.boundedness_ok = validation.get("boundedness_ok", True)

    # Check continuity via Lipschitz violations
    if validation and "lipschitz_violations" in validation:
        if validation["lipschitz_violations"] > 0:
            result.continuity_ok = False
            result.violations.append(
                f"V1: {validation['lipschitz_violations']} Lipschitz violations"
            )

    # Check correlation (cross-correlation in expected range)
    if validation and "correlation_ok" in validation:
        result.correlation_ok = validation["correlation_ok"]
        if not result.correlation_ok:
            result.violations.append("V3: Cross-correlation out of range")

    # Check for mock indicators
    if validation:
        mock_indicators = validation.get("mock_indicators", [])
        for indicator in mock_indicators:
            result.violations.append(f"MOCK-001: {indicator}")

    return result


def compute_divergence_decomposition(stats: DivergenceStats) -> DivergenceDecomposition:
    """Compute divergence decomposition per RTTS Section 3.2."""
    decomp = DivergenceDecomposition()

    if not stats.H_deltas:
        return decomp

    import statistics

    # Bias: |mean(p_twin) - mean(p_real)|
    # Approximate via mean absolute delta
    decomp.bias = statistics.mean(stats.H_deltas) if stats.H_deltas else 0.0

    # Variance: |std(p_twin) - std(p_real)|
    # Approximate via standard deviation of deltas
    if len(stats.H_deltas) > 1:
        decomp.variance = statistics.stdev(stats.H_deltas)

    # Timing: 1 - max(xcorr)
    # Placeholder - would need time series for proper calculation
    decomp.timing = 0.0

    # Structural: rate of sign changes
    if len(stats.H_deltas) > 1:
        sign_changes = sum(
            1 for i in range(1, len(stats.H_deltas))
            if (stats.H_deltas[i] > 0) != (stats.H_deltas[i-1] > 0)
        )
        decomp.structural = sign_changes / (len(stats.H_deltas) - 1)

    return decomp


def compute_true_divergence_vector_v1(
    stats: DivergenceStats,
    divergence_entries: List[Dict[str, Any]],
) -> TrueDivergenceVectorV1:
    """
    Compute True Divergence Vector v1 — NO METRIC LAUNDERING.

    Explicit, named fields:
    - safety_state_mismatch_rate: Fraction of cycles where safety state (blocked/omega) differs
    - state_error_mean: Mean absolute state delta (mean_delta_p proxy from H_deltas)
    - outcome_brier_score: Brier score for outcome prediction calibration (if probabilities available)

    SHADOW MODE: All computations are observational only.
    """
    vector = TrueDivergenceVectorV1()

    if stats.total_cycles == 0:
        return vector

    # safety_state_mismatch_rate: cycles where blocked OR omega state differs
    # This is distinct from outcome_mismatch_rate (which includes success divergence)
    safety_mismatches = stats.blocked_diverged + stats.omega_diverged
    vector.safety_state_mismatch_rate = round(safety_mismatches / stats.total_cycles, 6)

    # state_error_mean: mean absolute H delta (proxy for mean_delta_p)
    if stats.H_deltas:
        import statistics
        vector.state_error_mean = round(statistics.mean(stats.H_deltas), 6)

    # outcome_brier_score: Brier score if prediction probabilities are available
    # Brier = (1/N) * Σ(p_predicted - outcome_actual)^2
    # Only compute if entries have prediction probabilities
    brier_terms = []
    for entry in divergence_entries:
        # Check if entry has twin probability predictions
        twin_prob = entry.get("twin_success_prob")
        actual_outcome = entry.get("outcome")

        if twin_prob is not None and actual_outcome is not None:
            # Convert outcome to binary (1 = success, 0 = not success)
            actual_binary = 1.0 if actual_outcome == "success" else 0.0
            brier_term = (twin_prob - actual_binary) ** 2
            brier_terms.append(brier_term)

    if brier_terms:
        vector.outcome_brier_score = round(sum(brier_terms) / len(brier_terms), 6)

    return vector


def classify_divergence_pattern(
    decomp: DivergenceDecomposition,
    stats: DivergenceStats
) -> Tuple[str, float]:
    """Classify divergence pattern per RTTS Section 3.1."""
    # Check for nominal case
    if stats.divergence_rate < 0.01:
        return "NOMINAL", 0.95

    # Check for drift (high bias)
    if decomp.bias > DRIFT_BIAS_THRESHOLD:
        return "DRIFT", min(0.9, decomp.bias / 0.1)

    # Check for noise amplification (high variance)
    if decomp.variance > NOISE_VARIANCE_THRESHOLD:
        return "NOISE_AMPLIFICATION", min(0.9, decomp.variance / 0.2)

    # Check for phase lag (high timing component)
    if decomp.timing > PHASE_LAG_TIMING_THRESHOLD:
        return "PHASE_LAG", min(0.9, decomp.timing / 0.5)

    # Check for structural break (high structural component)
    if decomp.structural > STRUCTURAL_BREAK_THRESHOLD:
        return "STRUCTURAL_BREAK", min(0.9, decomp.structural / 0.4)

    # Default to attractor miss for moderate divergence
    if stats.divergence_rate > 0.1:
        return "ATTRACTOR_MISS", 0.6

    # Transient miss for low divergence
    return "TRANSIENT_MISS", 0.5


def determine_validation_status(
    validation: Optional[Dict[str, Any]],
    manifold: ManifoldValidation
) -> Tuple[str, float]:
    """Determine RTTS validation status."""
    if validation is None:
        return "UNVALIDATED", 0.5

    confidence = validation.get("confidence", 0.5)
    status = validation.get("status", "UNKNOWN")

    # Map adapter status to RTTS status
    if status == "PROVISIONAL_REAL" and confidence >= VALIDATION_CONFIDENCE_HIGH:
        if manifold.boundedness_ok and manifold.continuity_ok:
            return "VALIDATED_REAL", confidence

    if status == "MOCK_LIKE" or len(manifold.violations) > 2:
        return "SUSPECTED_MOCK", max(0.3, 1.0 - confidence)

    return "UNVALIDATED", confidence


def extract_mock_detection_flags(
    validation: Optional[Dict[str, Any]],
    manifold: ManifoldValidation,
    rtts_validation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Extract mock detection flags in MOCK-NNN format.

    P5.2: Prefers flags from rtts_validation.json when available,
    as these are computed by the authoritative RTTS validators.

    Args:
        validation: Legacy validation results
        manifold: ManifoldValidation results
        rtts_validation: RTTS validation from rtts_validation.json

    Returns:
        List of MOCK-NNN flag strings
    """
    flags = []

    # P5.2: Prefer RTTS validation flags when available
    if rtts_validation and "mock_detection_flags" in rtts_validation:
        # Use standardized flags from RTTS validation
        for flag in rtts_validation["mock_detection_flags"]:
            if flag not in flags:
                flags.append(flag)
        return flags

    # Fallback: Extract from legacy validation
    if validation:
        mock_indicators = validation.get("mock_indicators", [])
        for i, indicator in enumerate(mock_indicators[:5], start=1):
            flags.append(f"MOCK-{i:03d}")

    # Add flags from manifold violations
    for v in manifold.violations:
        if v.startswith("MOCK-"):
            flag = v.split(":")[0].strip()
            if flag not in flags:
                flags.append(flag)

    return flags


def compute_noise_envelope(stats: DivergenceStats) -> Optional[Dict[str, Any]]:
    """Compute noise envelope per RTTS Section 1.3."""
    if not stats.H_deltas or len(stats.H_deltas) < 10:
        return None

    import statistics

    envelope = {
        "sigma_H": statistics.stdev(stats.H_deltas) if len(stats.H_deltas) > 1 else 0.0,
    }

    if stats.rho_deltas and len(stats.rho_deltas) > 1:
        envelope["sigma_rho"] = statistics.stdev(stats.rho_deltas)

    # Autocorrelation at lag=1
    if len(stats.H_deltas) > 2:
        n = len(stats.H_deltas)
        mean_h = statistics.mean(stats.H_deltas)
        var_h = statistics.variance(stats.H_deltas)
        if var_h > 0:
            autocorr = sum(
                (stats.H_deltas[i] - mean_h) * (stats.H_deltas[i+1] - mean_h)
                for i in range(n - 1)
            ) / ((n - 1) * var_h)
            envelope["autocorr_lag1"] = autocorr

    # Excess kurtosis (placeholder)
    envelope["kurtosis"] = 0.0

    return envelope


# =============================================================================
# Report Generation
# =============================================================================

def generate_run_id(run_dir: Path) -> str:
    """Generate a P5 run ID from directory or timestamp."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    suffix = run_dir.name[:16] if run_dir.name else "run"
    return f"p5_{timestamp}_{suffix}"


def generate_report(
    run_dir: Path,
    summary: Dict[str, Any],
    divergence_entries: List[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    tda: Optional[Dict[str, Any]],
    calibration: Optional[Dict[str, Any]],
    rtts_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate the P5 divergence report.

    P5.2: Integrates rtts_validation.json when available for:
    - Mock detection flags (MOCK-NNN format)
    - Statistical validation results
    - Continuity tracking results
    - Correlation validation results
    """
    # Compute statistics
    stats = compute_divergence_stats(divergence_entries)

    # Get or generate run ID
    # If summary has a run_id starting with p4_, convert to p5_ format
    source_run_id = summary.get("run_id", "")
    if source_run_id.startswith("p5_"):
        run_id = source_run_id
    elif source_run_id.startswith("p4_"):
        # Convert p4_YYYYMMDD_HHMMSS_suffix to p5_YYYYMMDD_HHMMSS_suffix
        run_id = "p5_" + source_run_id[3:]
    elif source_run_id:
        run_id = f"p5_{source_run_id}"
    else:
        run_id = generate_run_id(run_dir)

    # Determine telemetry source
    telemetry_source = "real" if validation else "mock"
    if validation and validation.get("status") == "MOCK_LIKE":
        telemetry_source = "mock"

    # Compute manifold validation
    manifold = compute_manifold_validation(stats, validation)

    # Determine validation status
    validation_status, validation_confidence = determine_validation_status(
        validation, manifold
    )

    # Build minimal required fields
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "telemetry_source": telemetry_source,
        "validation_status": validation_status,
        "validation_confidence": round(validation_confidence, 4),
        "total_cycles": stats.total_cycles,
        "divergence_rate": round(stats.divergence_rate, 6),
        "mode": "SHADOW",
    }

    # True Divergence Vector v1 — NO METRIC LAUNDERING
    # Explicit, named fields for each divergence component
    # SHADOW MODE: All fields are observational only
    true_div_vector = compute_true_divergence_vector_v1(stats, divergence_entries)
    true_div_block: Dict[str, Any] = {
        "safety_state_mismatch_rate": true_div_vector.safety_state_mismatch_rate,
    }
    # Add optional fields only if computed
    if true_div_vector.state_error_mean is not None:
        true_div_block["state_error_mean"] = true_div_vector.state_error_mean
    if true_div_vector.outcome_brier_score is not None:
        true_div_block["outcome_brier_score"] = true_div_vector.outcome_brier_score
    report["true_divergence_vector_v1"] = true_div_block

    # Metric Versioning Block — NO METRIC LAUNDERING
    # Explicit declaration of which metrics are legacy vs true_vector
    # This block is strings-only, no inference, no computation
    # See docs/system_law/no_metric_laundering.md for rationale
    report["metric_versioning"] = {
        "legacy_metrics": [
            "divergence_rate",
            "mock_baseline_divergence_rate",
            "divergence_delta",
        ],
        "true_vector_v1_metrics": [
            "safety_state_mismatch_rate",
            "state_error_mean",
            "outcome_brier_score",
        ],
        "equivalence_note": "legacy_outcome_mismatch_rate NOT_EQUIVALENT_TO state_error_mean",
        "doc_reference": "docs/system_law/no_metric_laundering.md",
    }

    # Add recommended fields

    # Mock baseline comparison (if available from summary)
    mock_baseline = summary.get("mock_baseline_divergence_rate")
    if mock_baseline is not None:
        report["mock_baseline_divergence_rate"] = mock_baseline
        report["divergence_delta"] = round(
            stats.divergence_rate - mock_baseline, 6
        )

    # Twin tracking accuracy
    twin_accuracy = compute_twin_tracking_accuracy(summary, stats)
    if twin_accuracy:
        report["twin_tracking_accuracy"] = {
            k: round(v, 4) for k, v in twin_accuracy.items()
        }

    # Manifold validation
    report["manifold_validation"] = {
        "boundedness_ok": manifold.boundedness_ok,
        "continuity_ok": manifold.continuity_ok,
        "correlation_ok": manifold.correlation_ok,
        "violations": manifold.violations,
    }

    # TDA comparison (if available)
    if tda:
        tda_comparison = {}
        for key in ["sns_delta", "pcs_delta", "drs_delta", "hss_delta"]:
            if key in tda:
                tda_comparison[key] = tda[key]
        if tda_comparison:
            report["tda_comparison"] = tda_comparison

    # Warm-start calibration (if available)
    if calibration:
        report["warm_start_calibration"] = {
            "calibration_cycles": calibration.get("calibration_cycles", 0),
            "initial_divergence": calibration.get("initial_divergence", 0.0),
            "final_divergence": calibration.get("final_divergence", 0.0),
            "convergence_achieved": calibration.get("convergence_achieved", False),
        }

    # Optional diagnostics

    # Divergence decomposition
    decomp = compute_divergence_decomposition(stats)
    report["divergence_decomposition"] = {
        "bias": round(decomp.bias, 6),
        "variance": round(decomp.variance, 6),
        "timing": round(decomp.timing, 6),
        "structural": round(decomp.structural, 6),
    }

    # Pattern classification
    pattern, pattern_confidence = classify_divergence_pattern(decomp, stats)
    report["pattern_classification"] = pattern
    report["pattern_confidence"] = round(pattern_confidence, 4)

    # Mock detection flags (P5.2: prefers rtts_validation when available)
    mock_flags = extract_mock_detection_flags(validation, manifold, rtts_validation)
    if mock_flags:
        report["mock_detection_flags"] = mock_flags

    # P5.2: Include RTTS validation block if available
    if rtts_validation:
        report["rtts_validation"] = {
            "schema_version": rtts_validation.get("schema_version", "1.0.0"),
            "overall_status": rtts_validation.get("overall_status", "UNKNOWN"),
            "validation_passed": rtts_validation.get("validation_passed", False),
            "warning_count": rtts_validation.get("warning_count", 0),
            "window": rtts_validation.get("window"),
            "source": "rtts_validation.json",
        }

    # Noise envelope
    noise = compute_noise_envelope(stats)
    if noise:
        report["noise_envelope"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in noise.items()
        }

    # Governance signals (placeholder - would come from GGFL)
    report["governance_signals"] = {
        "sig_top_status": "NOMINAL",
        "sig_rpl_status": "NOMINAL",
        "sig_tel_status": "GREEN" if validation_status == "VALIDATED_REAL" else "YELLOW",
        "sig_met_status": "GREEN" if stats.divergence_rate < 0.1 else "YELLOW",
        "sig_bud_status": "NOMINAL",
    }

    # Fusion advisory (SHADOW MODE - advisory only)
    if stats.divergence_rate > 0.3 or validation_status == "SUSPECTED_MOCK":
        fusion_recommendation = "WARN"
    elif stats.divergence_rate > 0.5:
        fusion_recommendation = "BLOCK"
    else:
        fusion_recommendation = "ALLOW"

    report["fusion_advisory"] = {
        "recommendation": fusion_recommendation,
        "conflict_detected": len(manifold.violations) > 0,
    }

    # Recalibration recommendations
    recommendations = []
    if decomp.bias > DRIFT_BIAS_THRESHOLD:
        recommendations.append({
            "parameter": "twin_bias_correction",
            "current_value": 0.0,
            "suggested_value": decomp.bias,
            "rationale": "Observed systematic bias in twin predictions",
        })
    if decomp.variance > NOISE_VARIANCE_THRESHOLD:
        recommendations.append({
            "parameter": "twin_noise_scale",
            "current_value": 1.0,
            "suggested_value": 1.0 + decomp.variance,
            "rationale": "Twin variance underestimates real variance",
        })
    if recommendations:
        report["recalibration_recommendations"] = recommendations

    # Timing metadata
    now = datetime.now(timezone.utc)
    report["timing"] = {
        "start_time": summary.get("start_time", now.isoformat()),
        "end_time": summary.get("end_time", now.isoformat()),
        "duration_seconds": summary.get("duration_seconds", 0.0),
    }

    # Sources block: list of optional input files that were present
    # Deterministic order: rtts_validation, tda_comparison, calibration_report
    # Only includes files that were found; empty list if none present
    sources = build_sources_block(run_dir)
    report["sources"] = sources

    return report


def validate_report(report: Dict[str, Any], schema_path: Optional[Path]) -> bool:
    """Validate report against JSON Schema."""
    if not HAS_JSONSCHEMA:
        print("Warning: jsonschema not available, skipping validation", file=sys.stderr)
        return True

    if schema_path is None:
        # Try default location
        schema_path = (
            Path(__file__).parent.parent
            / "docs"
            / "system_law"
            / "schemas"
            / "p5"
            / "p5_divergence_real.schema.json"
        )

    if not schema_path.exists():
        print(f"Warning: Schema not found at {schema_path}, skipping validation", file=sys.stderr)
        return True

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=report, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Schema validation failed: {e.message}", file=sys.stderr)
        return False


def write_report(report: Dict[str, Any], output_path: Path) -> str:
    """Write report to file and return SHA256 hash."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(report, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Main Entry Point
# =============================================================================

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate P5 real telemetry divergence report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--p5-run-dir",
        type=Path,
        required=True,
        help="Path to P5 run directory containing p4_summary.json and divergence_log.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for report (default: <run-dir>/p4_shadow/p5_divergence_real.json)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to JSON Schema for validation",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages",
    )

    parsed = parser.parse_args(args)
    run_dir = parsed.p5_run_dir.resolve()

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    # Load required files
    summary = load_p4_summary(run_dir)
    if summary is None:
        print(f"Error: p4_summary.json not found in {run_dir}", file=sys.stderr)
        return 1

    divergence_entries = load_divergence_log(run_dir)
    if not divergence_entries:
        print(f"Warning: No divergence log entries found", file=sys.stderr)
        # Continue with empty entries - will produce minimal report

    # Load optional files
    validation = load_validation_results(run_dir)
    tda = load_tda_comparison(run_dir)
    calibration = load_calibration_report(run_dir)

    # P5.2: Load RTTS validation if available
    rtts_validation = load_rtts_validation(run_dir)

    # Generate report
    report = generate_report(
        run_dir=run_dir,
        summary=summary,
        divergence_entries=divergence_entries,
        validation=validation,
        tda=tda,
        calibration=calibration,
        rtts_validation=rtts_validation,
    )

    # Validate against schema
    if not parsed.skip_validation:
        if not validate_report(report, parsed.schema):
            return 1

    # Determine output path
    if parsed.output:
        output_path = parsed.output.resolve()
    else:
        # Default to p4_shadow subdirectory
        p4_shadow = run_dir / "p4_shadow"
        p4_shadow.mkdir(parents=True, exist_ok=True)
        output_path = p4_shadow / DEFAULT_OUTPUT_FILENAME

    # Write report
    sha256 = write_report(report, output_path)

    if not parsed.quiet:
        print(f"Generated: {output_path}")
        print(f"SHA256: {sha256}")
        print(f"Run ID: {report['run_id']}")
        print(f"Validation Status: {report['validation_status']}")
        print(f"Divergence Rate: {report['divergence_rate']:.4%}")
        print(f"Pattern: {report['pattern_classification']}")
        # P5.2: Show RTTS validation info
        if "rtts_validation" in report:
            rtts = report["rtts_validation"]
            print(f"RTTS Status: {rtts.get('overall_status', 'N/A')} (warnings: {rtts.get('warning_count', 0)})")
        if "mock_detection_flags" in report and report["mock_detection_flags"]:
            print(f"Mock Flags: {', '.join(report['mock_detection_flags'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
