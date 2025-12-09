"""
Noise Models Package

Provides noise injection and sampling for verifier imperfection modeling.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from .noise_sampler import should_inject_noise_decision

__all__ = [
    "should_inject_noise_decision",
]
