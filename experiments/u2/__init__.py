"""
PHASE-II — NOT USED IN PHASE I
===============================

U2 Uplift Experiment Module
---------------------------

This module provides a structured internal implementation for running U2 uplift
experiments. It is designed to be deterministic and self-contained for reproducibility.

**Determinism Notes:**
    - All random operations use seeded RNG instances.
    - Seed schedules are computed deterministically from the initial seed.
    - Hash computations use SHA-256 for reproducibility.

**Module Structure:**
    - ``seed.py``: Deterministic seed schedule generation
    - ``metrics.py``: Success metric functions for different experiment slices
    - ``manifest.py``: Experiment manifest generation and hashing
    - ``audit.py``: Audit trail logging for experiment runs
    - ``runner.py``: Main experiment runner orchestrating all components

Absolute Safeguards:
    - Do NOT reinterpret Phase I logs as uplift evidence.
    - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
    - All code must remain deterministic except random shuffle in the baseline policy.
    - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
    - All new files must be standalone and MUST NOT modify Phase I behavior.
"""

from experiments.u2.seed import generate_seed_schedule
from experiments.u2.metrics import (
    metric_arithmetic_simple,
    metric_algebra_expansion,
    get_metric_function,
    METRIC_DISPATCHER,
)
from experiments.u2.manifest import (
    compute_hash,
    generate_manifest,
    save_manifest,
)
from experiments.u2.audit import AuditLogger
from experiments.u2.runner import RFLPolicy, run_experiment

__all__ = [
    # Seed utilities
    "generate_seed_schedule",
    # Metrics
    "metric_arithmetic_simple",
    "metric_algebra_expansion",
    "get_metric_function",
    "METRIC_DISPATCHER",
    # Manifest
    "compute_hash",
    "generate_manifest",
    "save_manifest",
    # Audit
    "AuditLogger",
    # Runner
    "RFLPolicy",
    "run_experiment",
]
