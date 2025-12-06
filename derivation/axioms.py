"""
Axiom schemas and deterministic instantiation utilities for the derivation engine.

The generator implements the K/S axioms with bounded substitution according to
the active `SliceBounds`.  Instances are emitted in a stable order (by depth,
then lexicographically) to guarantee idempotent runs.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from normalization.canon import normalize, normalize_pretty

from .bounds import SliceBounds
from .structure import (
    OP_AND,
    OP_IMPLIES,
    OP_OR,
    atom_frozenset,
    formula_depth,
)


@dataclass(frozen=True, slots=True)
class AxiomInstance:
    """Concrete instantiation of an axiom schema."""

    name: str
    pretty: str
    normalized: str
    substitution: Dict[str, str]


@dataclass(frozen=True, slots=True)
class AxiomSchema:
    """Hilbert-style axiom schema."""

    name: str
    template: str

    @property
    def metavariables(self) -> Tuple[str, ...]:
        return tuple(sorted(set(re.findall(r"\b([a-z])\b", self.template))))

    def instantiate(self, binding: Dict[str, str]) -> Tuple[str, str]:
        """
        Instantiate the schema with normalized formula bindings.

        Returns:
            (pretty, normalized) pair for downstream persistence.
        """
        expr = self.template
        for var, replacement in binding.items():
            # Use a lambda to avoid interpreting backslashes in the replacement string
            expr = re.sub(rf"\b{var}\b", lambda _: f"({replacement})", expr)
        pretty = normalize_pretty(expr)
        normalized = normalize(expr)
        return pretty, normalized


AXIOM_SCHEMAS: Tuple[AxiomSchema, ...] = (
    AxiomSchema("K", "p -> (q -> p)"),
    AxiomSchema("S", "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"),
)


class FormulaEnumerator:
    """
    Deterministic formula generator parameterised by atom set and depth.
    """

    def __init__(self, atoms: Sequence[str], max_depth: int, max_pool: int) -> None:
        self._atoms = tuple(sorted(set(atoms)))
        self._max_depth = max(0, max_depth)
        self._max_pool = max_pool

    def enumerate(self) -> List[str]:
        """
        Enumerate normalized formulas ordered by (depth, lexicographic).
        """
        pool: List[str] = []
        by_depth: List[List[str]] = []

        # Depth 0: atoms
        depth0 = [normalize(atom) for atom in self._atoms][: self._max_pool]
        by_depth.append(depth0)
        pool.extend(depth0)

        if len(pool) >= self._max_pool:
            return pool[: self._max_pool]

        # Higher depths via binary connectives
        for depth in range(1, self._max_depth + 1):
            formulas: set[str] = set()
            for left_depth in range(depth):
                for right_depth in range(depth):
                    left_forms = by_depth[left_depth]
                    right_forms = by_depth[right_depth]
                    for left in left_forms:
                        for right in right_forms:
                            if len(formulas) + len(pool) >= self._max_pool:
                                break
                            formulas.add(normalize(f"({left}){OP_IMPLIES}({right})"))
                            formulas.add(normalize(f"({left}){OP_AND}({right})"))
                            formulas.add(normalize(f"({left}){OP_OR}({right})"))
                        if len(formulas) + len(pool) >= self._max_pool:
                            break
                    if len(formulas) + len(pool) >= self._max_pool:
                        break
                if len(formulas) + len(pool) >= self._max_pool:
                    break
            level = sorted(formulas)
            by_depth.append(level)
            pool.extend(level)
            if len(pool) >= self._max_pool:
                break

        return pool[: self._max_pool]


def instantiate_axioms(bounds: SliceBounds) -> List[AxiomInstance]:
    """
    Instantiate K and S axioms under the provided slice bounds.

    Instances are unique by normalized form and respect atom/depth limits.
    """
    enumerator = FormulaEnumerator(bounds.atom_alphabet, bounds.max_formula_depth, bounds.max_formula_pool)
    formula_pool = enumerator.enumerate()
    if not formula_pool:
        return []

    seen: set[str] = set()
    instances: List[AxiomInstance] = []

    for schema in AXIOM_SCHEMAS:
        metavars = schema.metavariables
        if not metavars:
            pretty, normalized = schema.instantiate({})
            if _within_slice(normalized, bounds) and normalized not in seen:
                instances.append(AxiomInstance(schema.name, pretty, normalized, {}))
                seen.add(normalized)
            continue

        products: Iterable[Tuple[str, ...]] = itertools.product(formula_pool, repeat=len(metavars))
        for combo in products:
            binding = dict(zip(metavars, combo))
            pretty, normalized = schema.instantiate(binding)
            if normalized in seen:
                continue
            if not _within_slice(normalized, bounds):
                continue
            instances.append(AxiomInstance(schema.name, pretty, normalized, binding))
            seen.add(normalized)
            if len(instances) >= bounds.max_axiom_instances:
                return instances

    return instances


def _within_slice(normalized: str, bounds: SliceBounds) -> bool:
    """Check atom and depth guards for a candidate formula."""
    if formula_depth(normalized) > bounds.max_formula_depth:
        return False
    return len(atom_frozenset(normalized)) <= bounds.max_atoms

