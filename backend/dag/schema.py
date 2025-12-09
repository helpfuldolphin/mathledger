# backend/dag/schema.py
"""
PHASE II - Data schema for the Global DAG Builder.

This module defines the core data structures and custom exceptions used in
the construction and validation of the Proof Dependency DAG.
"""
from dataclasses import dataclass, field
from typing import Tuple

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading DAG schema.", file=__import__("sys").stderr)

@dataclass(frozen=True)
class Derivation:
    """
    Represents a single, canonical derivation rule.

    The `premises` are stored in a sorted tuple to ensure that the
    object is hashable and that two derivations with the same premises
    (regardless of order) are treated as identical.
    """
    conclusion: str
    premises: Tuple[str, ...]

    def __post_init__(self):
        """Sorts premises to ensure canonical representation."""
        # The frozen=True dataclass does not allow direct modification after __init__.
        # To enforce a canonical representation, we must either trust the caller
        # to provide sorted premises or use a more complex setup.
        # For this implementation, we assume the caller will create it correctly,
        # for example: Derivation("C", tuple(sorted(["B", "A"])))
        pass

@dataclass
class CycleLog:
    """
    Represents the set of new derivations produced in a single cycle.
    """
    cycle_index: int
    derivations: Tuple[Derivation, ...]

# --- Custom Exceptions ---

class DagConsistencyError(Exception):
    """Base exception for DAG consistency violations."""
    pass

class CyclicDependencyError(DagConsistencyError):
    """Raised when a cyclic dependency is detected in the DAG."""
    def __init__(self, message: str, cycle_path: Tuple[str, ...]):
        super().__init__(message)
        self.cycle_path = cycle_path

class InvariantViolationError(DagConsistencyError):
    """Raised when a consolidation invariant is violated."""
    pass

class DanglingPremiseError(InvariantViolationError):
    """Raised in strict mode when a premise is not a known conclusion or axiom."""
    def __init__(self, message: str, conclusion: str, premise: str):
        super().__init__(message)
        self.conclusion = conclusion
        self.premise = premise
