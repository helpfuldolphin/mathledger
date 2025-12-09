"""
Statistical Methods Package for MathLedger

This package provides deterministic statistical analysis tools for U2 uplift quantification.
All methods guarantee reproducibility: same seed → identical results.
"""

from statistical.bootstrap import (
    # Core bootstrap
    paired_bootstrap_delta,
    PairedBootstrapResult,
    DistributionSummary,
    # Confidence bands (visualization only)
    compute_confidence_band,
    ConfidenceBand,
    # Leakage detection
    detect_bootstrap_leakage,
    LeakageDetectionResult,
    # Profiling
    profile_bootstrap,
    BootstrapProfile,
    BootstrapHistogram,
    HistogramBin,
    # Contract
    get_bootstrap_contract,
    write_bootstrap_contract,
)

__all__ = [
    # Core bootstrap
    "paired_bootstrap_delta",
    "PairedBootstrapResult",
    "DistributionSummary",
    # Confidence bands
    "compute_confidence_band",
    "ConfidenceBand",
    # Leakage detection
    "detect_bootstrap_leakage",
    "LeakageDetectionResult",
    # Profiling
    "profile_bootstrap",
    "BootstrapProfile",
    "BootstrapHistogram",
    "HistogramBin",
    # Contract
    "get_bootstrap_contract",
    "write_bootstrap_contract",
]
