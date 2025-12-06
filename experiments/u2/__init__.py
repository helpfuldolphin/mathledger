"""
U2 Experiment Framework - Phase III Safety Envelope & Type-Verified Runner

This package provides type-safe, performance-monitored infrastructure for
Phase III U2 uplift experiments with strict safety guarantees.

Core components:
- runner: Type-verified U2Runner with config and result types
- u2_safe_eval: Safe evaluation with lint mode
- safety_envelope: Safety status and performance monitoring
- snapshots: Deterministic state capture and restoration
- logging: Structured trace logging
- schema: Event and result schemas
"""

from .runner import U2Config, CycleResult, U2Runner, TracedExperimentContext, run_with_traces
from .safety_envelope import U2SafetyEnvelope, build_u2_safety_envelope
from .u2_safe_eval import SafeEvalLintResult, safe_eval, lint_expression, batch_lint_expressions
from .entrypoint import run_u2_experiment

__all__ = [
    "U2Config",
    "CycleResult",
    "U2Runner",
    "TracedExperimentContext",
    "run_with_traces",
    "U2SafetyEnvelope",
    "build_u2_safety_envelope",
    "SafeEvalLintResult",
    "safe_eval",
    "lint_expression",
    "batch_lint_expressions",
    "run_u2_experiment",
]
