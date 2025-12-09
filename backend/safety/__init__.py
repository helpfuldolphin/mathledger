"""
Safety SLO Module

Defines Safety Service Level Objectives (SLOs) and the SafetyEnvelope
that wraps TDA decisions with safety guarantees.
"""

from .envelope import SafetyEnvelope, SafetySLO, SLOStatus
from .cortex import evaluate_hard_gate_decision, check_gate_decision

__all__ = [
    "SafetyEnvelope",
    "SafetySLO",
    "SLOStatus",
    "evaluate_hard_gate_decision",
    "check_gate_decision",
]
