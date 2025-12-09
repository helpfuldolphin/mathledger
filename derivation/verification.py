"""
Verification utilities for derived statements.

Primary strategy:
    1. Recognise known tautology schemata.
    2. Deterministic truth-table evaluation.
    3. Optional Lean fallback (disabled by default, opt-in via env).

IMPORTANT: FO Hermetic and Wide Slice Experiments
--------------------------------------------------
First Organism (FO) hermetic runs and Wide Slice experiments do NOT depend on
a live Lean kernel. They exercise the 'lean-disabled' abstention mode for
deterministic behavior.

When ML_ENABLE_LEAN_FALLBACK is not set (default):
    - LeanFallback.verify() returns VerificationOutcome(False, "lean-disabled")
    - No subprocess calls to Lean are made
    - No Lean binary or kernel is required
    - Abstention behavior is deterministic given seeds and slice config

This ensures that:
    - FO cycles used for baseline/RFL logs are fully hermetic
    - Wide Slice experiments are reproducible without external dependencies
    - Abstention metrics are deterministic and testable

To enable Lean verification (not recommended for hermetic runs):
    Set environment variable: ML_ENABLE_LEAN_FALLBACK=1
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from normalization.truthtab import is_tautology as truth_table_is_tautology
from derivation.bounds import SliceBounds
from derivation.structure import atom_frozenset
from derivation.derive_rules import is_known_tautology
from .noise_guard import VerifierNoiseGuard, global_noise_guard


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """
    Result of a verification attempt.

    Attributes:
        verified: True if the statement was proven to be a tautology.
        method: The method that produced this outcome (pattern, truth-table, lean, lean-disabled, etc.).
        details: Optional additional information (e.g., error messages).
    """
    verified: bool
    method: str
    details: Optional[str] = None


class LeanFallback:
    """
    Optional Lean verifier. Disabled unless ML_ENABLE_LEAN_FALLBACK=1.

    When disabled, returns a VerificationOutcome with method="lean-disabled".
    When enabled but times out, returns method="lean-timeout".
    """

    def __init__(self, project_root: Optional[Path], timeout_s: float) -> None:
        self._project_root = project_root
        self._timeout = timeout_s
        self._enabled = os.getenv("ML_ENABLE_LEAN_FALLBACK") == "1"
        self._lean_exe = os.getenv("LEAN_EXE") or "lean"

    def verify(self, normalized: str) -> VerificationOutcome:
        if not self._enabled or not self._project_root:
            return VerificationOutcome(False, "lean-disabled")

        atoms = sorted(atom_frozenset(normalized))
        declarations = " ".join(f"({atom} : Prop)" for atom in atoms) or "(p : Prop)"
        goal = _to_lean(normalized)

        script = f"""import Mathlib

open Classical

theorem auto_proof {declarations} : {goal} := by
  classical
  taut
"""
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(script)
            cmd = [self._lean_exe, str(tmp_path)]
            subprocess.run(
                cmd,
                cwd=self._project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return VerificationOutcome(True, "lean")
        except subprocess.TimeoutExpired:
            return VerificationOutcome(False, "lean-timeout")
        except Exception as exc:  # pragma: no cover - best-effort path
            return VerificationOutcome(False, "lean-error", details=str(exc))
        finally:
            try:
                if "tmp_path" in locals():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


class StatementVerifier:
    """
    Layered verifier orchestrating tautology checks and Lean fallback.

    Verification order:
        1. Pattern matching for known tautology schemata (fastest).
        2. Deterministic truth-table evaluation.
        3. Lean fallback (if enabled via ML_ENABLE_LEAN_FALLBACK=1).

    If all methods fail to prove the statement is a tautology, the verifier
    returns verified=False with the method of the last check attempted.
    """

    def __init__(
        self,
        bounds: SliceBounds,
        lean_project_root: Optional[Path] = None,
        noise_guard: Optional[VerifierNoiseGuard] = None,
    ) -> None:
        self._bounds = bounds
        self._lean = LeanFallback(lean_project_root, bounds.lean_timeout_s)
        self._noise_guard = noise_guard or global_noise_guard()

    def verify(self, normalized: str) -> VerificationOutcome:
        # Layer 1: Known tautology patterns (instant)
        if is_known_tautology(normalized):
            outcome = VerificationOutcome(True, "pattern")
            self._record_noise(normalized, outcome, tier_hint="T0")
            return outcome

        # Layer 2: Truth-table evaluation (deterministic, O(2^n) in atoms)
        try:
            if truth_table_is_tautology(normalized):
                outcome = VerificationOutcome(True, "truth-table")
                self._record_noise(normalized, outcome, tier_hint="T0")
                return outcome
        except Exception as exc:
            outcome = VerificationOutcome(False, "truth-table-error", details=str(exc))
            self._record_noise(normalized, outcome, tier_hint="T0")
            return outcome

        # Layer 3: Lean fallback (optional, may timeout or be disabled)
        outcome = self._lean.verify(normalized)
        self._record_noise(normalized, outcome, tier_hint="T2")
        return outcome

    def _record_noise(
        self,
        normalized: str,
        outcome: VerificationOutcome,
        *,
        tier_hint: Optional[str] = None,
    ) -> None:
        if not self._noise_guard:
            return
        try:
            self._noise_guard.record_verification(
                normalized,
                outcome,
                tier_hint=tier_hint,
            )
        except Exception:
            # Guardrails are best-effort; never block verification on telemetry.
            pass


def _to_lean(normalized: str) -> str:
    """Convert canonical ASCII formula to Lean syntax."""
    return (
        normalized.replace("->", " → ")
        .replace("/\\", " ∧ ")
        .replace("\\/", " ∨ ")
        .replace("~", "¬")
    )


__all__ = ["VerificationOutcome", "LeanFallback", "StatementVerifier"]
