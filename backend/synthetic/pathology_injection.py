"""Helpers to inject simple pathologies into a delta-p time series."""

from __future__ import annotations

import math
from typing import List


def inject_spike(series: List[float], at: int, magnitude: float) -> List[float]:
    """Return a new series with a single-point spike added at the given index."""
    data = list(series)
    if at < 0 or at >= len(data):
        raise IndexError("spike index is out of range")
    data[at] += magnitude
    return data


def inject_drift(series: List[float], slope: float) -> List[float]:
    """Return a new series with a linear drift applied across the horizon."""
    data = list(series)
    return [value + slope * idx for idx, value in enumerate(data)]


def inject_oscillation(series: List[float], period: int, amplitude: float) -> List[float]:
    """Return a new series with a sinusoidal oscillation overlaid."""
    if period <= 0:
        raise ValueError("period must be positive")

    data = list(series)
    angular_frequency = (2 * math.pi) / period
    return [
        value + amplitude * math.sin(angular_frequency * idx)
        for idx, value in enumerate(data)
    ]

