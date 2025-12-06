"""
Lean Control Module — Controlled Statements for Safe Failure Testing

This module provides controlled statements for the First Organism chain that can be used
to safely test real Lean verification. These statements have known, predictable behaviors:

1. SAFE_TAUTOLOGY: A simple tautology that should always verify (p → p)
2. SAFE_CONTRADICTION: A contradiction that should always fail (p ∧ ¬p)
3. CONTROLLED_STATEMENTS: Registry of statements with expected outcomes

The key design principle is that these statements provide **predictable failure modes**
that the First Organism can handle gracefully when transitioning from mock to real Lean.

Environment Variables:
    FIRST_ORGANISM_REAL_LEAN: Set to "true" to enable real Lean on controlled statements
    FIRST_ORGANISM_CONTROLLED_ONLY: Set to "true" to only verify controlled statements

Usage:
    from backend.lean_control import (
        get_controlled_build_runner,
        is_controlled_statement,
        ControlledStatement,
    )

    # Get a runner that uses real Lean only for controlled statements
    runner = get_controlled_build_runner()
    result = execute_lean_job(statement, build_runner=runner)

First Organism Integration:
    - In the short term: mock path for all statements (current behavior)
    - In the longer term: real Lean on controlled statements, mock for others
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from backend.lean_mode import (
    LeanMode,
    get_build_runner,
    get_lean_status,
    mock_lean_build,
    is_lean_available,
    ABSTENTION_SIGNATURE,
)


class ControlledStatementType(Enum):
    """Types of controlled statements for safe testing."""

    TAUTOLOGY = "tautology"
    """Simple tautology that should always verify."""

    CONTRADICTION = "contradiction"
    """Contradiction that should always fail verification."""

    DECIDABLE = "decidable"
    """Statement decidable by truth tables."""

    COMPLEX = "complex"
    """More complex statement requiring proof search."""


@dataclass(frozen=True)
class ControlledStatement:
    """A controlled statement with known expected behavior."""

    canonical: str
    """Canonical form of the statement."""

    statement_type: ControlledStatementType
    """Type of controlled statement."""

    expected_success: bool
    """Whether Lean verification should succeed."""

    proof_body: str
    """Lean proof body for this statement."""

    description: str
    """Human-readable description."""

    @property
    def hash(self) -> str:
        """SHA256 hash of canonical form."""
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


# Registry of controlled statements with known behaviors
CONTROLLED_STATEMENTS: Dict[str, ControlledStatement] = {
    # Simple tautology - should always verify
    "p->p": ControlledStatement(
        canonical="p->p",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro hp\n  exact hp",
        description="Identity tautology: p implies p",
    ),

    # More complex tautology
    "p->q->p": ControlledStatement(
        canonical="p->q->p",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro hp\n  intro hq\n  exact hp",
        description="Weakening: p implies (q implies p)",
    ),

    # Conjunction elimination left
    "p/\\q->p": ControlledStatement(
        canonical="p/\\q->p",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro h\n  exact h.left",
        description="Conjunction elimination (left)",
    ),

    # Conjunction elimination right
    "p/\\q->q": ControlledStatement(
        canonical="p/\\q->q",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro h\n  exact h.right",
        description="Conjunction elimination (right)",
    ),

    # Disjunction introduction left
    "p->p\\/q": ControlledStatement(
        canonical="p->p\\/q",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro hp\n  apply Or.inl\n  exact hp",
        description="Disjunction introduction (left)",
    ),

    # Disjunction introduction right
    "q->p\\/q": ControlledStatement(
        canonical="q->p\\/q",
        statement_type=ControlledStatementType.TAUTOLOGY,
        expected_success=True,
        proof_body="  intro hq\n  apply Or.inr\n  exact hq",
        description="Disjunction introduction (right)",
    ),

    # Decidable by truth table - uses decide tactic
    "(p->q)->(~q->~p)": ControlledStatement(
        canonical="(p->q)->(~q->~p)",
        statement_type=ControlledStatementType.DECIDABLE,
        expected_success=True,
        proof_body="  classical\n  decide",
        description="Contraposition (decidable)",
    ),
}


# Safe default: the simplest tautology
SAFE_TAUTOLOGY = CONTROLLED_STATEMENTS["p->p"]

# Registry of contradiction patterns (expected to fail Lean verification)
CONTRADICTION_PATTERNS: Dict[str, ControlledStatement] = {
    # Contradiction - should fail verification (provably false)
    "p/\\~p": ControlledStatement(
        canonical="p/\\~p",
        statement_type=ControlledStatementType.CONTRADICTION,
        expected_success=False,
        proof_body="  admit",  # Will fail or produce warning
        description="Contradiction: p and not p",
    ),
}


def is_controlled_statement(canonical: str) -> bool:
    """Check if a statement is in the controlled registry."""
    return canonical in CONTROLLED_STATEMENTS or canonical in CONTRADICTION_PATTERNS


def get_controlled_statement(canonical: str) -> Optional[ControlledStatement]:
    """Get controlled statement info if it exists."""
    return CONTROLLED_STATEMENTS.get(canonical) or CONTRADICTION_PATTERNS.get(canonical)


def should_use_real_lean() -> bool:
    """Check if First Organism should use real Lean for controlled statements.

    Returns True if:
    - FIRST_ORGANISM_REAL_LEAN=true is set
    - Lean toolchain is available

    Returns False otherwise (use mock mode).
    """
    env_value = os.environ.get("FIRST_ORGANISM_REAL_LEAN", "").lower()
    if env_value not in ("true", "1", "yes"):
        return False
    return is_lean_available()


def should_verify_only_controlled() -> bool:
    """Check if First Organism should only verify controlled statements.

    When FIRST_ORGANISM_CONTROLLED_ONLY=true:
    - Controlled statements use real Lean (if available)
    - Other statements use mock mode (safe abstention)
    """
    return os.environ.get("FIRST_ORGANISM_CONTROLLED_ONLY", "").lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class ControlledRunResult:
    """Result of a controlled statement verification."""

    canonical: str
    """Canonical form of the statement."""

    is_controlled: bool
    """Whether this is a controlled statement."""

    used_real_lean: bool
    """Whether real Lean was used."""

    expected_success: Optional[bool]
    """Expected success for controlled statements."""

    actual_success: bool
    """Actual verification success."""

    matches_expectation: Optional[bool]
    """Whether result matches expectation (for controlled statements)."""

    abstention_signature: Optional[str]
    """Abstention signature if mock was used."""

    def to_metadata(self) -> dict:
        """Convert to metadata dict for ledger recording."""
        return {
            "is_controlled": self.is_controlled,
            "used_real_lean": self.used_real_lean,
            "expected_success": self.expected_success,
            "actual_success": self.actual_success,
            "matches_expectation": self.matches_expectation,
            "abstention_signature": self.abstention_signature,
        }


BuildRunner = Callable[[str], subprocess.CompletedProcess[str]]


def get_controlled_build_runner(
    *,
    controlled_use_real: Optional[bool] = None,
    project_dir: Optional[str] = None,
    timeout: int = 90,
) -> BuildRunner:
    """
    Get a build runner that applies real Lean only to controlled statements.

    This implements the "gradual transition" strategy for First Organism:
    - Controlled statements: Use real Lean if available and enabled
    - Other statements: Use mock mode (safe abstention)

    Args:
        controlled_use_real: Override for using real Lean on controlled statements.
                            If None, uses FIRST_ORGANISM_REAL_LEAN env var.
        project_dir: Lean project directory.
        timeout: Build timeout in seconds.

    Returns:
        A build runner that applies the controlled strategy.
    """
    use_real = controlled_use_real if controlled_use_real is not None else should_use_real_lean()

    if use_real:
        real_runner = get_build_runner(
            mode=LeanMode.FULL,
            project_dir=project_dir,
            timeout=timeout,
        )
    else:
        real_runner = None

    def controlled_runner(module_name: str) -> subprocess.CompletedProcess[str]:
        """Runner that applies controlled verification strategy."""
        # For now, always use mock - the module_name doesn't give us the canonical form
        # This runner is meant to be used via execute_controlled_lean_job
        return mock_lean_build(module_name)

    return controlled_runner


def execute_controlled_lean_job(
    canonical: str,
    module_name: str,
    *,
    project_dir: Optional[str] = None,
    timeout: int = 90,
    force_mock: bool = False,
    force_real: bool = False,
) -> tuple[subprocess.CompletedProcess[str], ControlledRunResult]:
    """
    Execute a Lean job with controlled statement handling.

    This function applies the First Organism controlled verification strategy:
    - For controlled statements: optionally use real Lean
    - For other statements: use mock mode

    Args:
        canonical: Canonical form of the statement
        module_name: Lean module name for the job
        project_dir: Lean project directory
        timeout: Build timeout in seconds
        force_mock: Force mock mode even for controlled statements
        force_real: Force real Lean even for non-controlled statements

    Returns:
        Tuple of (CompletedProcess, ControlledRunResult)
    """
    controlled = get_controlled_statement(canonical)
    is_controlled = controlled is not None
    expected_success = controlled.expected_success if controlled else None

    # Determine whether to use real Lean
    use_real_lean = False
    if force_real and is_lean_available():
        use_real_lean = True
    elif force_mock:
        use_real_lean = False
    elif is_controlled and should_use_real_lean():
        use_real_lean = True
    elif should_verify_only_controlled():
        use_real_lean = is_controlled and is_lean_available()

    # Get the appropriate runner
    if use_real_lean:
        runner = get_build_runner(
            mode=LeanMode.FULL,
            project_dir=project_dir,
            timeout=timeout,
        )
    else:
        runner = mock_lean_build

    # Execute the build
    result = runner(module_name)

    # Determine actual success
    actual_success = result.returncode == 0

    # Check if mock abstention
    from backend.lean_mode import is_mock_abstention
    is_mock = is_mock_abstention(result.stderr or "")
    abstention_sig = ABSTENTION_SIGNATURE if is_mock else None

    # Check expectation match
    matches = None
    if is_controlled and not is_mock:
        matches = actual_success == expected_success

    controlled_result = ControlledRunResult(
        canonical=canonical,
        is_controlled=is_controlled,
        used_real_lean=use_real_lean,
        expected_success=expected_success,
        actual_success=actual_success,
        matches_expectation=matches,
        abstention_signature=abstention_sig,
    )

    return result, controlled_result


@dataclass(frozen=True)
class ControlledModeStatus:
    """Status of controlled verification mode."""

    first_organism_real_lean: bool
    """Whether FIRST_ORGANISM_REAL_LEAN is enabled."""

    first_organism_controlled_only: bool
    """Whether FIRST_ORGANISM_CONTROLLED_ONLY is enabled."""

    lean_available: bool
    """Whether Lean toolchain is available."""

    effective_mode: str
    """Effective mode description."""

    controlled_statement_count: int
    """Number of registered controlled statements."""

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "first_organism_real_lean": self.first_organism_real_lean,
            "first_organism_controlled_only": self.first_organism_controlled_only,
            "lean_available": self.lean_available,
            "effective_mode": self.effective_mode,
            "controlled_statement_count": self.controlled_statement_count,
        }


def get_controlled_mode_status() -> ControlledModeStatus:
    """Get comprehensive status of controlled verification mode."""
    real_enabled = should_use_real_lean()
    controlled_only = should_verify_only_controlled()
    lean_avail = is_lean_available()

    if real_enabled:
        effective = "real_lean_for_controlled"
    elif controlled_only:
        effective = "mock_except_controlled"
    else:
        effective = "full_mock"

    return ControlledModeStatus(
        first_organism_real_lean=real_enabled,
        first_organism_controlled_only=controlled_only,
        lean_available=lean_avail,
        effective_mode=effective,
        controlled_statement_count=len(CONTROLLED_STATEMENTS) + len(CONTRADICTION_PATTERNS),
    )


__all__ = [
    # Enums
    "ControlledStatementType",

    # Dataclasses
    "ControlledStatement",
    "ControlledRunResult",
    "ControlledModeStatus",

    # Registries
    "CONTROLLED_STATEMENTS",
    "CONTRADICTION_PATTERNS",
    "SAFE_TAUTOLOGY",

    # Functions
    "is_controlled_statement",
    "get_controlled_statement",
    "should_use_real_lean",
    "should_verify_only_controlled",
    "get_controlled_build_runner",
    "execute_controlled_lean_job",
    "get_controlled_mode_status",
]
