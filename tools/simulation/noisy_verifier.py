# EXPERIMENTAL: Imperfect verifier simulation
# SAFETY: NEVER use this in the production pipeline.
# This module is used only by scripts under tools/simulation or experiments/.

from __future__ import annotations
import random
from typing import Optional
from derivation.verification import StatementVerifier, VerificationOutcome

class NoisyVerifierWrapper:
    """
    Imperfect verifier wrapper for simulation of verifier bias.

    SAFETY:
      - Never used in production.
      - Only used inside tools/simulation or experiments scripts.
    """

    def __init__(self,
                 inner: StatementVerifier,
                 epsilon: float,
                 seed: int = 42) -> None:
        if not (0.0 <= epsilon < 0.5):
            raise ValueError("epsilon must be in [0.0, 0.5)")
        self._inner = inner
        self._epsilon = float(epsilon)
        self._rng = random.Random(seed)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def verify(self, normalized: str) -> VerificationOutcome:
        real = self._inner.verify(normalized)

        # Bernoulli(ε) flip
        flip = self._rng.random() < self._epsilon
        if not flip:
            return real

        new_verified = not real.verified
        new_method = f"{real.method}-NOISY-FLIPPED"

        return VerificationOutcome(
            verified=new_verified,
            method=new_method,
            details=(
                f"Imperfect verifier simulation: flip with eps={self._epsilon:.4f}. "
                f"original_verified={real.verified}, original_method={real.method}"
            ),
        )
