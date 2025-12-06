"""
Causal analysis framework for Reflexive Formal Learning (RFL).

Formalizes cause-effect structures linking:
- Policy updates → Abstention shifts → Throughput uplifts

Implements Do-Calculus for interventional analysis and counterfactual simulation.
"""

from backend.causal.graph import CausalGraph, CausalNode, CausalEdge
from backend.causal.variables import (
    extract_run_deltas,
    compute_policy_delta,
    compute_abstention_delta,
    compute_throughput_delta
)
from backend.causal.do_calculus import DoOperator, intervene
from backend.causal.estimator import estimate_causal_effect, CausalCoefficient, compute_stability
from backend.causal.report import generate_causal_report

__all__ = [
    'CausalGraph',
    'CausalNode',
    'CausalEdge',
    'extract_run_deltas',
    'compute_policy_delta',
    'compute_abstention_delta',
    'compute_throughput_delta',
    'DoOperator',
    'intervene',
    'estimate_causal_effect',
    'CausalCoefficient',
    'compute_stability',
    'generate_causal_report',
]
