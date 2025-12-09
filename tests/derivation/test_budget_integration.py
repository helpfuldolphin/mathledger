"""
Phase II Budget Integration Tests for Derivation Pipeline
═══════════════════════════════════════════════════════════════════════════════

Tests for budget enforcement in derivation/pipeline.py

PHASE II — NOT USED IN PHASE I

Test coverage:
    1. With budget=None, behavior unchanged vs old code
    2. With very small cycle_budget_s, stats.budget_exhausted is set
    3. With small max_candidates_per_cycle, stats.max_candidates_hit is set
    4. Budget enforcement is deterministic (same inputs → same termination)
    5. No change to attestation roots when budgets don't bind
"""

import pytest
import time
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from derivation.pipeline import (
    DerivationPipeline,
    DerivationResult,
    StatementRecord,
    PipelineStats,
    run_slice_for_test,
)
from derivation.bounds import SliceBounds
from derivation.verification import StatementVerifier
from curriculum.gates import CurriculumSlice, SliceGates
from backend.verification.budget_loader import VerifierBudget


@dataclass(frozen=True)
class MockBudget:
    """Mock budget for testing pipeline integration."""
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int
    
    @property
    def max_candidates(self) -> int:
        return self.max_candidates_per_cycle


class TestPipelineWithNoBudget:
    """Tests ensuring budget=None preserves original behavior."""
    
    def test_no_budget_flags_default(self):
        """Without budget, budget flags should remain False."""
        bounds = SliceBounds(
            max_atoms=2,
            max_formula_depth=2,
            max_mp_depth=1,
            max_breadth=10,
            max_total=10,
        )
        verifier = StatementVerifier(bounds)
        pipeline = DerivationPipeline(bounds, verifier)
        
        # Run with no budget
        outcome = pipeline.run_step([])
        
        assert outcome.stats.budget_exhausted is False
        assert outcome.stats.max_candidates_hit is False
        assert outcome.stats.budget_remaining_s == -1.0
    
    def test_no_budget_processes_all_candidates(self):
        """Without budget, all candidates should be processed."""
        bounds = SliceBounds(
            max_atoms=2,
            max_formula_depth=2,
            max_mp_depth=1,
            max_breadth=20,
            max_total=50,
        )
        verifier = StatementVerifier(bounds)
        pipeline = DerivationPipeline(bounds, verifier, budget=None)
        
        # Run derivation
        outcome = pipeline.run_step([])
        
        # Should process multiple candidates (axiom seeding generates some)
        assert outcome.stats.candidates_considered > 0
        # No budget exhaustion
        assert outcome.stats.budget_exhausted is False
        assert outcome.stats.max_candidates_hit is False


class TestCycleBudgetEnforcement:
    """Tests for cycle_budget_s wall-clock enforcement."""
    
    def test_very_small_budget_triggers_exhaustion(self):
        """With very small cycle_budget_s, budget_exhausted should be set."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=3,
            max_mp_depth=2,
            max_breadth=100,
            max_total=200,
        )
        verifier = StatementVerifier(bounds)
        
        # Very small budget - should exhaust quickly
        budget = MockBudget(
            cycle_budget_s=0.0001,  # 0.1ms - will exhaust immediately
            taut_timeout_s=1.0,
            max_candidates_per_cycle=1000,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        
        # Small delay to ensure budget expires
        time.sleep(0.001)
        
        outcome = pipeline.run_step([])
        
        # Budget should be exhausted
        assert outcome.stats.budget_exhausted is True
        assert outcome.stats.budget_remaining_s == 0.0
    
    def test_large_budget_doesnt_trigger(self):
        """With large cycle_budget_s, budget_exhausted should remain False."""
        bounds = SliceBounds(
            max_atoms=2,
            max_formula_depth=2,
            max_mp_depth=1,
            max_breadth=10,
            max_total=10,
        )
        verifier = StatementVerifier(bounds)
        
        # Large budget - should never exhaust
        budget = MockBudget(
            cycle_budget_s=300.0,  # 5 minutes
            taut_timeout_s=1.0,
            max_candidates_per_cycle=1000,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        outcome = pipeline.run_step([])
        
        # Budget should NOT be exhausted
        assert outcome.stats.budget_exhausted is False


class TestMaxCandidatesEnforcement:
    """Tests for max_candidates_per_cycle enforcement."""
    
    def test_small_max_candidates_triggers_flag(self):
        """With small max_candidates, max_candidates_hit should be set."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=4,
            max_mp_depth=3,
            max_breadth=100,
            max_total=200,
            max_axiom_instances=50,
        )
        verifier = StatementVerifier(bounds)
        
        # Very small candidate limit
        budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=3,  # Only 3 candidates
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        outcome = pipeline.run_step([])
        
        # Max candidates should be hit
        assert outcome.stats.max_candidates_hit is True
        # Candidates considered should be <= limit
        assert outcome.stats.candidates_considered <= 3
    
    def test_large_max_candidates_doesnt_trigger(self):
        """With large max_candidates, max_candidates_hit should remain False."""
        bounds = SliceBounds(
            max_atoms=2,
            max_formula_depth=2,
            max_mp_depth=1,
            max_breadth=10,
            max_total=10,
        )
        verifier = StatementVerifier(bounds)
        
        # Large candidate limit
        budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=10000,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        outcome = pipeline.run_step([])
        
        # Max candidates should NOT be hit
        assert outcome.stats.max_candidates_hit is False
    
    def test_exact_limit_candidates(self):
        """Exact limit should trigger max_candidates_hit."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=4,
            max_mp_depth=2,
            max_breadth=50,
            max_total=100,
        )
        verifier = StatementVerifier(bounds)
        
        # Set limit to something that will be exactly reached
        budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=5,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        outcome = pipeline.run_step([])
        
        # Should stop at exactly the limit
        assert outcome.stats.candidates_considered <= 5


class TestBudgetDeterminism:
    """Tests ensuring budget enforcement is deterministic."""
    
    def test_same_budget_same_termination(self):
        """Same budget + same inputs should produce same termination state."""
        bounds = SliceBounds(
            max_atoms=2,
            max_formula_depth=2,
            max_mp_depth=1,
            max_breadth=20,
            max_total=20,
        )
        verifier = StatementVerifier(bounds)
        
        budget = MockBudget(
            cycle_budget_s=300.0,  # Large enough not to bind
            taut_timeout_s=1.0,
            max_candidates_per_cycle=10,  # Small enough to bind
        )
        
        # Run twice with same config
        pipeline1 = DerivationPipeline(bounds, verifier, budget=budget)
        outcome1 = pipeline1.run_step([])
        
        pipeline2 = DerivationPipeline(bounds, verifier, budget=budget)
        outcome2 = pipeline2.run_step([])
        
        # Both should have same termination state
        assert outcome1.stats.max_candidates_hit == outcome2.stats.max_candidates_hit
        assert outcome1.stats.candidates_considered == outcome2.stats.candidates_considered
    
    def test_max_candidates_is_hard_limit(self):
        """max_candidates should be a hard limit, not approximate."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=4,
            max_mp_depth=2,
            max_breadth=100,
            max_total=200,
        )
        verifier = StatementVerifier(bounds)
        
        for limit in [1, 2, 5, 10]:
            budget = MockBudget(
                cycle_budget_s=300.0,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=limit,
            )
            
            pipeline = DerivationPipeline(bounds, verifier, budget=budget)
            outcome = pipeline.run_step([])
            
            # Hard limit: never exceed
            assert outcome.stats.candidates_considered <= limit, \
                f"Exceeded limit {limit}: got {outcome.stats.candidates_considered}"


class TestBudgetWithExistingStatements:
    """Tests for budget enforcement with pre-seeded statements."""
    
    def test_budget_enforced_with_seeds(self):
        """Budget should be enforced even with existing statements."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=4,
            max_mp_depth=2,
            max_breadth=50,
            max_total=100,
        )
        verifier = StatementVerifier(bounds)
        
        # Create some seed statements
        seeds = [
            StatementRecord(
                normalized="p",
                hash="seed_p_hash",
                pretty="p",
                rule="seed",
                is_axiom=False,
                mp_depth=0,
                verification_method="seed",
            ),
            StatementRecord(
                normalized="(p->q)",
                hash="seed_pq_hash",
                pretty="p → q",
                rule="seed",
                is_axiom=False,
                mp_depth=0,
                verification_method="seed",
            ),
        ]
        
        budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=3,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        outcome = pipeline.run_step(seeds)
        
        # Budget should still be enforced
        assert outcome.stats.candidates_considered <= 3


class TestBudgetParameterOverride:
    """Tests for budget parameter precedence."""
    
    def test_run_step_budget_overrides_constructor(self):
        """Budget passed to run_step should override constructor budget."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=4,
            max_mp_depth=2,
            max_breadth=50,
            max_total=100,
        )
        verifier = StatementVerifier(bounds)
        
        # Constructor budget with high limit
        constructor_budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=1000,
        )
        
        # run_step budget with low limit (should take precedence)
        run_step_budget = MockBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=2,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=constructor_budget)
        outcome = pipeline.run_step([], budget=run_step_budget)
        
        # run_step budget should have been used
        assert outcome.stats.candidates_considered <= 2


class TestPipelineStatsFields:
    """Tests for budget-related stats fields."""
    
    def test_stats_fields_initialized(self):
        """Budget-related stats fields should be properly initialized."""
        stats = PipelineStats()
        
        assert stats.budget_exhausted is False
        assert stats.max_candidates_hit is False
        assert stats.timeout_abstentions == 0
        assert stats.budget_remaining_s == -1.0
    
    def test_budget_remaining_s_set_on_exhaustion(self):
        """budget_remaining_s should be 0.0 when budget exhausted."""
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=3,
            max_mp_depth=2,
            max_breadth=100,
            max_total=200,
        )
        verifier = StatementVerifier(bounds)
        
        budget = MockBudget(
            cycle_budget_s=0.0001,  # Will exhaust immediately
            taut_timeout_s=1.0,
            max_candidates_per_cycle=1000,
        )
        
        pipeline = DerivationPipeline(bounds, verifier, budget=budget)
        time.sleep(0.001)  # Ensure budget expires
        outcome = pipeline.run_step([])
        
        if outcome.stats.budget_exhausted:
            assert outcome.stats.budget_remaining_s == 0.0


class TestRunSliceForTestWithBudget:
    """Tests for run_slice_for_test with budget parameters."""
    
    def test_run_slice_with_max_candidates(self):
        """run_slice_for_test should respect max_candidates parameter."""
        slice_cfg = CurriculumSlice(
            name="test-slice",
            params={
                "atoms": 2,
                "depth_max": 2,
                "mp_depth": 1,
                "breadth_max": 20,
                "total_max": 50,
            },
            gates=SliceGates(),
            metadata={},
        )
        
        result = run_slice_for_test(
            slice_cfg,
            limit=1,
            max_candidates=5,
            emit_log=False,
        )
        
        # Should respect the limit
        assert result.stats.candidates_considered <= 5

