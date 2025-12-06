"""
Slice configuration primitives for the derivation pipeline.

These bounds constrain the combinatorial search across axiom instantiations
and Modus Ponens derivations so that every run is deterministic, bounded,
and reproducible.  The defaults reflect the PL curriculum slice described
in the MathLedger whitepaper (atoms ≤4, formula depth ≤4, tight breadth caps).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SliceBounds:
    """
    Deterministic limits for a single derivation slice.

    Attributes:
        max_atoms: Maximum distinct propositional atoms permitted in a formula.
        max_formula_depth: Maximum syntactic depth of generated formulas.
        max_mp_depth: Maximum number of sequential Modus Ponens rounds per step.
        max_breadth: Cap on new statements accepted per derivation step.
        max_total: Global cap on new statements accepted in a run.
        max_axiom_instances: Maximum number of instantiated axioms seeded per run.
        max_formula_pool: Cap on enumerated formulas used for substitution.
        lean_timeout_s: Timeout for Lean fallback verification (seconds).
    """

    max_atoms: int = 4
    max_formula_depth: int = 4
    max_mp_depth: int = 3
    max_breadth: int = 64
    max_total: int = 256
    max_axiom_instances: int = 96
    max_formula_pool: int = 256
    lean_timeout_s: float = 0.5

    @property
    def atom_alphabet(self) -> tuple[str, ...]:
        """Return the deterministic atom alphabet truncated to max_atoms."""
        base = ("p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
        return base[: self.max_atoms]

