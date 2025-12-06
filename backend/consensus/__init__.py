"""
Phase IX Celestial Convergence - Consensus Module
Byzantine-resilient consensus with adaptive trust weighting.
"""

from .harmony import (
    HarmonyProtocol,
    ConsensusRound,
    ValidatorSet,
    TrustWeight,
    converge,
)

__all__ = [
    "HarmonyProtocol",
    "ConsensusRound",
    "ValidatorSet",
    "TrustWeight",
    "converge",
]
