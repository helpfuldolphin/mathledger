"""
TDA (Topological Decision Analysis) Module

Manages TDA modes and decision-making for hard gates.
"""

from .modes import TDAMode, TDADecision
from .analyzer import evaluate_tda_decision

__all__ = [
    "TDAMode",
    "TDADecision",
    "evaluate_tda_decision",
]
