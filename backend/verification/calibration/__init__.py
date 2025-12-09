"""
Noise Calibration Package

Provides CLI and statistical fitting for noise model calibration.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from .statistical_fitting import (
    fit_bernoulli_rate,
    wilson_confidence_interval,
    fit_timeout_distribution,
)

__all__ = [
    "fit_bernoulli_rate",
    "wilson_confidence_interval",
    "fit_timeout_distribution",
]
