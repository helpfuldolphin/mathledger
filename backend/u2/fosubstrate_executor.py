"""
FOSubstrateExecutor - Complete Implementation

This module provides the complete implementation for integrating the U2 Planner
with the First Organism (FO) derivation substrate.

Author: Manus-F
Date: 2025-12-06
Status: Phase V Implementation
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import skeleton types
from backend.u2.fosubstrate_executor_skeleton import (
    BudgetConfig,
    BudgetConsumption,
    BudgetRemaining,
    CandidateItem,
    ExecutionOutcome,
    ExecutionResult,
    StatementRecord,
)


# ============================================================================
# SIMPLE PROPOSITIONAL LOGIC VERIFIER
# ============================================================================

class PropositionalVerifier:
    """
    Simple truth-table based verifier for propositional logic.
    
    This is a minimal implementation for MVP demonstration.
    In production, this would integrate with Lean or other theorem provers.
    """
    
    def __init__(self, timeout_ms: int = 5000):
        """
        Initialize verifier.
        
        Args:
            timeout_ms: Verification timeout in milliseconds
        """
        self.timeout_ms = timeout_ms
    
    def verify_tautology(self, normalized: str) -> Tuple[bool, str, int]:
        """
        Verify if a normalized statement is a tautology.
        
        Args:
            normalized: Normalized propositional formula
            
        Returns:
            Tuple of (is_tautology, method, time_ms)
        """
        start_time = time.time_ns() // 1_000_000
        
        # Extract propositional variables
        variables = self._extract_variables(normalized)
        
        # Check timeout
        if len(variables) > 10:  # Too many variables for truth table
            return False, "abstained", 0
        
        # Evaluate all truth assignments
        is_tautology = self._check_all_assignments(normalized, variables)
        
        end_time = time.time_ns() // 1_000_000
        elapsed_ms = end_time - start_time
        
        return is_tautology, "truth-table", elapsed_ms
    
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract propositional variables from formula."""
        import re
        # Match single letters or p0, p1, etc.
        variables = set(re.findall(r'[a-z]\d*', formula))
        return sorted(variables)
    
    def _check_all_assignments(self, formula: str, variables: List[str]) -> bool:
        """Check if formula is true under all truth assignments."""
        if not variables:
            # No variables, evaluate directly
            return self._evaluate(formula, {})
        
        # Try all 2^n assignments
        n = len(variables)
        for i in range(2 ** n):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)
            
            if not self._evaluate(formula, assignment):
                return False  # Found counterexample
        
        return True  # All assignments satisfy formula
    
    def _evaluate(self, formula: str, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate formula under truth assignment.
        
        This is a simplified evaluator for demonstration.
        """
        # Replace variables with truth values
        expr = formula
        for var, value in assignment.items():
            expr = expr.replace(var, "True" if value else "False")
        
        # Replace logical operators
        expr = expr.replace("→", "<=")  # Implication
        expr = expr.replace("∧", "and")
        expr = expr.replace("∨", "or")
        expr = expr.replace("¬", "not ")
        
        try:
            return eval(expr)
        except:
            # Parse error, assume false
            return False


# ============================================================================
# MODUS PONENS DERIVATION ENGINE
# ============================================================================

class ModusPonensEngine:
    """
    Simple Modus Ponens derivation engine.
    
    Given a set of statements, applies Modus Ponens to derive new statements.
    """
    
    def __init__(self, prng_seed: int):
        """
        Initialize engine with deterministic PRNG.
        
        Args:
            prng_seed: Seed for deterministic derivation
        """
        self.prng = random.Random(prng_seed)
        self.derived_count = 0
    
    def apply_modus_ponens(
        self,
        statements: List[StatementRecord],
        max_applications: int = 10,
    ) -> List[StatementRecord]:
        """
        Apply Modus Ponens to derive new statements.
        
        Args:
            statements: List of known statements
            max_applications: Maximum MP applications
            
        Returns:
            List of newly derived statements
        """
        new_statements = []
        
        # Find implications (A → B)
        implications = [s for s in statements if "→" in s.normalized]
        antecedents = [s for s in statements if "→" not in s.normalized]
        
        # Shuffle for deterministic randomness
        self.prng.shuffle(implications)
        self.prng.shuffle(antecedents)
        
        for _ in range(max_applications):
            if not implications or not antecedents:
                break
            
            # Pick random implication and antecedent
            impl = implications[self.derived_count % len(implications)]
            ante = antecedents[self.derived_count % len(antecedents)]
            
            # Try to apply MP
            derived = self._try_modus_ponens(impl, ante)
            if derived:
                new_statements.append(derived)
                self.derived_count += 1
        
        return new_statements
    
    def _try_modus_ponens(
        self,
        implication: StatementRecord,
        antecedent: StatementRecord,
    ) -> Optional[StatementRecord]:
        """
        Try to apply Modus Ponens: (A → B), A ⊢ B
        
        Args:
            implication: Statement of form "A → B"
            antecedent: Statement A
            
        Returns:
            Derived statement B, or None if MP doesn't apply
        """
        # Parse implication
        if "→" not in implication.normalized:
            return None
        
        parts = implication.normalized.split("→")
        if len(parts) != 2:
            return None
        
        ante_part = parts[0].strip()
        cons_part = parts[1].strip()
        
        # Check if antecedent matches
        if antecedent.normalized.strip() != ante_part:
            return None
        
        # Derive consequent
        consequent_normalized = cons_part
        consequent_hash = hashlib.sha256(consequent_normalized.encode()).hexdigest()
        
        derived = StatementRecord(
            normalized=consequent_normalized,
            hash=consequent_hash,
            pretty=consequent_normalized,  # Simplified
            rule="modus-ponens",
            is_axiom=False,
            mp_depth=max(implication.mp_depth, antecedent.mp_depth) + 1,
            parents=(implication.hash, antecedent.hash),
            verification_method="modus-ponens",
        )
        
        return derived


# ============================================================================
# FOSUBSTRATE EXECUTOR IMPLEMENTATION
# ============================================================================

class FOSubstrateExecutor:
    """
    Complete implementation of FOSubstrateExecutor.
    
    Integrates:
    - PropositionalVerifier for tautology checking
    - ModusPonensEngine for derivation
    - Budget enforcement
    - Error handling
    """
    
    def __init__(
        self,
        slice_name: str,
        budget_config: Optional[BudgetConfig] = None,
        substrate_version: str = "1.0.0-mvp",
    ):
        """
        Initialize executor.
        
        Args:
            slice_name: Experiment slice name
            budget_config: Budget configuration (uses defaults if None)
            substrate_version: Substrate version string
        """
        self.slice_name = slice_name
        self.substrate_version = substrate_version
        
        # Use provided budget or defaults
        self.budget = budget_config or BudgetConfig(
            max_depth=10,
            max_breadth=100,
            cycle_time_budget_ms=60000,  # 60 seconds
            cycle_memory_budget_kb=512000,  # 512 MB
            experiment_time_budget_ms=3600000,  # 1 hour
            experiment_memory_budget_kb=2048000,  # 2 GB
            verification_timeout_ms=5000,  # 5 seconds
            enable_lean=False,
        )
        
        # Initialize verifier
        self.verifier = PropositionalVerifier(timeout_ms=self.budget.verification_timeout_ms)
        
        # Track experiment-level budget
        self.experiment_time_consumed_ms = 0
        self.experiment_memory_consumed_kb = 0
    
    def execute(
        self,
        item: CandidateItem,
        seed: int,
        cycle_start_time_ms: int,
    ) -> Tuple[bool, ExecutionResult]:
        """
        Execute a candidate with deterministic seeding.
        
        Args:
            item: Candidate to execute
            seed: Deterministic execution seed
            cycle_start_time_ms: Cycle start timestamp
            
        Returns:
            Tuple of (success, result)
        """
        exec_start_time_ms = time.time_ns() // 1_000_000
        
        # Check cycle budget
        cycle_elapsed_ms = exec_start_time_ms - cycle_start_time_ms
        cycle_remaining_ms = self.budget.cycle_time_budget_ms - cycle_elapsed_ms
        
        if cycle_remaining_ms <= 0:
            return self._budget_exceeded_result(
                item, seed, exec_start_time_ms, "cycle"
            )
        
        # Check experiment budget
        experiment_remaining_ms = (
            self.budget.experiment_time_budget_ms - self.experiment_time_consumed_ms
        )
        
        if experiment_remaining_ms <= 0:
            return self._budget_exceeded_result(
                item, seed, exec_start_time_ms, "experiment"
            )
        
        # Execute derivation
        try:
            success, result = self._execute_derivation(
                item, seed, exec_start_time_ms
            )
            
            # Update experiment budget
            self.experiment_time_consumed_ms += result.time_ms
            self.experiment_memory_consumed_kb = max(
                self.experiment_memory_consumed_kb, result.memory_kb
            )
            
            return success, result
            
        except TimeoutError:
            return self._timeout_result(item, seed, exec_start_time_ms)
        
        except Exception as e:
            return self._error_result(item, seed, exec_start_time_ms, e)
    
    def _execute_derivation(
        self,
        item: CandidateItem,
        seed: int,
        exec_start_time_ms: int,
    ) -> Tuple[bool, ExecutionResult]:
        """
        Execute derivation on FO substrate.
        
        Steps:
        1. Verify if statement is a tautology
        2. If tautology, apply Modus Ponens to derive new statements
        3. Return execution result
        
        Args:
            item: Candidate to execute
            seed: Execution seed
            exec_start_time_ms: Execution start timestamp
            
        Returns:
            Tuple of (success, result)
        """
        # Step 1: Verify tautology
        is_tautology, method, verification_time_ms = self.verifier.verify_tautology(
            item.statement.normalized
        )
        
        # Step 2: Apply Modus Ponens if tautology
        new_statements = []
        mp_steps = 0
        
        if is_tautology:
            mp_engine = ModusPonensEngine(prng_seed=seed)
            new_statements = mp_engine.apply_modus_ponens(
                statements=[item.statement],
                max_applications=5,
            )
            mp_steps = len(new_statements)
        
        # Step 3: Compute execution metrics
        exec_end_time_ms = time.time_ns() // 1_000_000
        exec_time_ms = exec_end_time_ms - exec_start_time_ms
        
        # Estimate memory (simplified)
        memory_kb = 1024 + len(new_statements) * 10
        
        # Determine outcome
        if is_tautology:
            outcome = ExecutionOutcome.SUCCESS
        else:
            outcome = ExecutionOutcome.FAILURE
        
        # Build result
        result = ExecutionResult(
            outcome=outcome,
            verification_method=method,
            is_tautology=is_tautology,
            new_statements=new_statements,
            mp_steps=mp_steps,
            axiom_instances=0,
            time_ms=exec_time_ms,
            memory_kb=memory_kb,
            verification_time_ms=verification_time_ms,
            budget_consumed=BudgetConsumption(
                total_time_ms=exec_time_ms,
                mp_applications=mp_steps,
                verification_attempts=1,
            ),
            budget_remaining=self._compute_budget_remaining(
                exec_start_time_ms, exec_time_ms
            ),
            error_type=None,
            error_message=None,
            error_traceback=None,
            execution_seed=hex(seed),
            substrate_version=self.substrate_version,
            timestamp_ms=exec_end_time_ms,
        )
        
        return is_tautology, result
    
    def _budget_exceeded_result(
        self,
        item: CandidateItem,
        seed: int,
        timestamp_ms: int,
        budget_type: str,
    ) -> Tuple[bool, ExecutionResult]:
        """Create result for budget exceeded."""
        result = ExecutionResult(
            outcome=ExecutionOutcome.BUDGET_EXCEEDED,
            verification_method="none",
            is_tautology=False,
            new_statements=[],
            mp_steps=0,
            axiom_instances=0,
            time_ms=0,
            memory_kb=0,
            verification_time_ms=0,
            budget_consumed=BudgetConsumption(
                total_time_ms=0,
                mp_applications=0,
                verification_attempts=0,
            ),
            budget_remaining=self._compute_budget_remaining(timestamp_ms, 0),
            error_type="BudgetExceededError",
            error_message=f"{budget_type} budget exhausted",
            error_traceback=None,
            execution_seed=hex(seed),
            substrate_version=self.substrate_version,
            timestamp_ms=timestamp_ms,
        )
        return False, result
    
    def _timeout_result(
        self,
        item: CandidateItem,
        seed: int,
        timestamp_ms: int,
    ) -> Tuple[bool, ExecutionResult]:
        """Create result for timeout."""
        result = ExecutionResult(
            outcome=ExecutionOutcome.TIMEOUT,
            verification_method="none",
            is_tautology=False,
            new_statements=[],
            mp_steps=0,
            axiom_instances=0,
            time_ms=self.budget.verification_timeout_ms,
            memory_kb=0,
            verification_time_ms=self.budget.verification_timeout_ms,
            budget_consumed=BudgetConsumption(
                total_time_ms=self.budget.verification_timeout_ms,
                mp_applications=0,
                verification_attempts=1,
            ),
            budget_remaining=self._compute_budget_remaining(
                timestamp_ms, self.budget.verification_timeout_ms
            ),
            error_type="VerificationTimeoutError",
            error_message="Verification exceeded timeout",
            error_traceback=None,
            execution_seed=hex(seed),
            substrate_version=self.substrate_version,
            timestamp_ms=timestamp_ms,
        )
        return False, result
    
    def _error_result(
        self,
        item: CandidateItem,
        seed: int,
        timestamp_ms: int,
        error: Exception,
    ) -> Tuple[bool, ExecutionResult]:
        """Create result for unexpected error."""
        import traceback
        
        result = ExecutionResult(
            outcome=ExecutionOutcome.ERROR,
            verification_method="none",
            is_tautology=False,
            new_statements=[],
            mp_steps=0,
            axiom_instances=0,
            time_ms=0,
            memory_kb=0,
            verification_time_ms=0,
            budget_consumed=BudgetConsumption(
                total_time_ms=0,
                mp_applications=0,
                verification_attempts=0,
            ),
            budget_remaining=self._compute_budget_remaining(timestamp_ms, 0),
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            execution_seed=hex(seed),
            substrate_version=self.substrate_version,
            timestamp_ms=timestamp_ms,
        )
        return False, result
    
    def _compute_budget_remaining(
        self, current_time_ms: int, elapsed_ms: int
    ) -> BudgetRemaining:
        """Compute remaining budget."""
        cycle_remaining = self.budget.cycle_time_budget_ms - elapsed_ms
        experiment_remaining = (
            self.budget.experiment_time_budget_ms - self.experiment_time_consumed_ms - elapsed_ms
        )
        
        return BudgetRemaining(
            cycle_time_remaining_ms=max(0, cycle_remaining),
            experiment_time_remaining_ms=max(0, experiment_remaining),
            cycle_budget_exhausted=(cycle_remaining <= 0),
            experiment_budget_exhausted=(experiment_remaining <= 0),
        )
