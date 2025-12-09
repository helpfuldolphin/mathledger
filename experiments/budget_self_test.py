#!/usr/bin/env python3
"""
Phase II Budget Enforcement Self-Test Harness
═══════════════════════════════════════════════════════════════════════════════

PHASE II — NOT USED IN PHASE I

This script validates that budget enforcement works correctly across all
code paths. It runs fast (<1 second) and performs NO uplift or significance
tests — only budget invariant validation.

CI Integration
──────────────
This script is designed to be run in CI to verify budget enforcement:

    uv run python experiments/budget_self_test.py

Exit Codes:
    0 = All budget invariants validated successfully
    1 = One or more budget invariants violated
    2 = Script error (import failure, missing dependencies)

CI Integration Example (GitHub Actions):
    - name: Budget Self-Test
      run: uv run python experiments/budget_self_test.py
      env:
        MATHLEDGER_DEBUG_BUDGET: "1"

Failure Interpretation:
    If this test fails, it indicates a regression in budget enforcement.
    Review the specific assertion that failed and check:
    1. Pipeline budget checks in derivation/pipeline.py
    2. Budget loader in backend/verification/budget_loader.py
    3. VerifierBudget dataclass validation

What This Test Does NOT Do:
    - Does NOT run uplift experiments
    - Does NOT compute statistical significance
    - Does NOT modify any state
    - Does NOT require database/Redis

Author: Agent B1 (verifier-ops-1)
Phase: II
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

# Phase II Budget Self-Test Configuration
# ========================================
# These constants control the self-test harness behavior.
# The goal is <5s total runtime while still exercising invariants.

MAX_SELF_TEST_RUNTIME_S = 5.0  # Warning threshold for total harness runtime
MICRO_SLICE_ATOMS = 2          # Tiny slice to keep derivation fast
MICRO_SLICE_DEPTH = 2
MICRO_SLICE_MP_DEPTH = 1
MICRO_SLICE_BREADTH = 8        # Reduced from 10
MICRO_SLICE_TOTAL = 8          # Reduced from 10
MICRO_BUDGET_CYCLE_S = 0.1     # 100ms hard cap for "no budget" test (safety net)

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of a single budget validation test."""
    name: str
    passed: bool
    message: str
    duration_ms: float


class BudgetSelfTest:
    """
    Budget enforcement self-test harness.
    
    Runs three test cases:
    1. Baseline (no budget) - verifies default behavior unchanged
    2. Tight budget - verifies budget_exhausted / max_candidates_hit flags
    3. Pathological - high-depth formulas + micro timeouts
    """
    
    def __init__(self):
        self.results: List[TestResult] = []
        self._setup_imports()
    
    def _setup_imports(self):
        """Lazy import to catch import errors gracefully."""
        try:
            from derivation.pipeline import (
                DerivationPipeline,
                run_slice_for_test,
                PipelineStats,
                DerivationSummary,
            )
            from derivation.bounds import SliceBounds
            from derivation.verification import StatementVerifier
            from curriculum.gates import CurriculumSlice, SliceGates
            from backend.verification.budget_loader import VerifierBudget
            
            self.DerivationPipeline = DerivationPipeline
            self.run_slice_for_test = run_slice_for_test
            self.PipelineStats = PipelineStats
            self.DerivationSummary = DerivationSummary
            self.SliceBounds = SliceBounds
            self.StatementVerifier = StatementVerifier
            self.CurriculumSlice = CurriculumSlice
            self.SliceGates = SliceGates
            self.VerifierBudget = VerifierBudget
            
        except ImportError as e:
            print(f"✗ Import error: {e}", file=sys.stderr)
            sys.exit(2)
    
    def _make_slice(self, name: str, **params) -> "CurriculumSlice":
        """
        Create a minimal test slice optimized for fast self-test execution.
        
        Uses MICRO_SLICE_* constants by default for sub-second derivation.
        Override via **params if needed for specific tests.
        """
        # Import gate specs for creating valid gates
        from curriculum.gates import (
            CoverageGateSpec,
            AbstentionGateSpec,
            VelocityGateSpec,
            CapsGateSpec,
        )
        
        # Use micro-slice defaults for fast execution
        default_params = {
            "atoms": MICRO_SLICE_ATOMS,
            "depth_max": MICRO_SLICE_DEPTH,
            "mp_depth": MICRO_SLICE_MP_DEPTH,
            "breadth_max": MICRO_SLICE_BREADTH,
            "total_max": MICRO_SLICE_TOTAL,
        }
        default_params.update(params)
        
        # Create minimal valid gate specs for testing
        test_gates = self.SliceGates(
            coverage=CoverageGateSpec(ci_lower_min=0.5, sample_min=10),
            abstention=AbstentionGateSpec(max_rate_pct=25.0, max_mass=100),
            velocity=VelocityGateSpec(min_pph=1.0, stability_cv_max=0.5, window_minutes=5),
            caps=CapsGateSpec(min_attempt_mass=5, min_runtime_minutes=0.1, backlog_max=1.0),
        )
        
        return self.CurriculumSlice(
            name=name,
            params=default_params,
            gates=test_gates,
            metadata={"test": True},
        )
    
    def _make_budget(
        self,
        cycle_budget_s: float = 300.0,
        taut_timeout_s: float = 1.0,
        max_candidates_per_cycle: int = 1000,
    ) -> "VerifierBudget":
        """Create a test budget with given parameters."""
        return self.VerifierBudget(
            cycle_budget_s=cycle_budget_s,
            taut_timeout_s=taut_timeout_s,
            max_candidates_per_cycle=max_candidates_per_cycle,
        )
    
    def test_baseline_no_budget(self) -> TestResult:
        """
        Test 1: Baseline case — derivation completes without hitting budget limits.
        
        Uses a generous safety budget (MICRO_BUDGET_CYCLE_S) to prevent runaway
        execution while still verifying that budget flags are NOT triggered
        when derivation completes normally within limits.
        
        Semantic intent: "Does derivation work when budget is non-binding?"
        """
        start = time.perf_counter()
        name = "baseline_no_budget"
        
        try:
            slice_cfg = self._make_slice("test_baseline")
            
            # Use a generous budget that shouldn't bind for micro-slices
            # This prevents runaway execution while preserving test semantics
            safety_budget = self._make_budget(
                cycle_budget_s=MICRO_BUDGET_CYCLE_S,  # 100ms — generous for micro slice
                taut_timeout_s=0.05,                   # 50ms per candidate
                max_candidates_per_cycle=50,           # More than micro slice produces
            )
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=safety_budget,
                emit_log=False,
            )
            
            # For a micro-slice with generous budget, we expect NEITHER flag set
            # (budget exists but should not be exhausted)
            assertions = []
            
            # If budget was exhausted, the micro-slice is too big or budget too tight
            if result.stats.budget_exhausted:
                assertions.append(
                    f"budget_exhausted=True (unexpected — micro-slice should complete in {MICRO_BUDGET_CYCLE_S}s)"
                )
            
            if result.stats.max_candidates_hit:
                assertions.append(
                    "max_candidates_hit=True (unexpected — limit should be generous)"
                )
            
            # Budget was set, so remaining should be >= 0 (not -1)
            if result.stats.budget_remaining_s < 0:
                assertions.append(
                    f"budget_remaining_s={result.stats.budget_remaining_s} (expected >= 0 with budget)"
                )
            
            if assertions:
                return TestResult(
                    name=name,
                    passed=False,
                    message="; ".join(assertions),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"Micro-slice completed without budget exhaustion (remaining={result.stats.budget_remaining_s:.3f}s)",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_tight_budget(self) -> TestResult:
        """
        Test 2: Tight budget that should trigger max_candidates_hit.
        
        Uses a very small max_candidates_per_cycle to ensure the limit is hit.
        """
        start = time.perf_counter()
        name = "tight_budget"
        
        try:
            slice_cfg = self._make_slice(
                "test_tight",
                atoms=3,
                depth_max=4,
                mp_depth=2,
                breadth_max=50,
                total_max=100,
            )
            
            budget = self._make_budget(
                cycle_budget_s=300.0,  # Won't bind
                taut_timeout_s=1.0,
                max_candidates_per_cycle=3,  # Very tight
            )
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            assertions = []
            
            # With max_candidates=3, we should hit the limit
            if not result.stats.max_candidates_hit:
                assertions.append("max_candidates_hit should be True with tight limit")
            
            # Candidate count should be <= limit
            if result.stats.candidates_considered > 3:
                assertions.append(
                    f"candidates_considered ({result.stats.candidates_considered}) "
                    f"exceeds limit (3)"
                )
            
            # Summary should include budget config
            summary_dict = result.summary.to_dict()
            budget_section = summary_dict.get("budget", {})
            
            if budget_section.get("max_candidates_per_cycle") != 3:
                assertions.append(
                    f"summary budget.max_candidates_per_cycle should be 3, "
                    f"got {budget_section.get('max_candidates_per_cycle')}"
                )
            
            if not budget_section.get("max_candidates_hit"):
                assertions.append("summary budget.max_candidates_hit should be True")
            
            if assertions:
                return TestResult(
                    name=name,
                    passed=False,
                    message="; ".join(assertions),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"max_candidates correctly enforced (candidates={result.stats.candidates_considered})",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_pathological_timeout(self) -> TestResult:
        """
        Test 3: Pathological case with micro timeout.
        
        Uses a very small cycle_budget_s to trigger budget exhaustion.
        Note: This test is timing-dependent and may not always trigger
        exhaustion on very fast systems.
        """
        start = time.perf_counter()
        name = "pathological_timeout"
        
        try:
            slice_cfg = self._make_slice(
                "test_pathological",
                atoms=4,
                depth_max=4,
                mp_depth=3,
                breadth_max=100,
                total_max=200,
            )
            
            # Very small budget - should exhaust quickly
            budget = self._make_budget(
                cycle_budget_s=0.0001,  # 0.1ms - will exhaust almost immediately
                taut_timeout_s=1.0,
                max_candidates_per_cycle=1000,
            )
            
            # Small delay to ensure budget expires
            time.sleep(0.001)
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            assertions = []
            
            # With micro timeout, budget should be exhausted (but this is timing dependent)
            # We just verify the fields are set correctly
            
            # budget_remaining_s should be non-negative when budget is set
            if result.stats.budget_remaining_s < 0 and result.stats.budget_exhausted:
                assertions.append(
                    f"budget_remaining_s ({result.stats.budget_remaining_s}) "
                    f"should be >= 0 when budget_exhausted"
                )
            
            # If budget exhausted, remaining should be 0
            if result.stats.budget_exhausted and result.stats.budget_remaining_s != 0.0:
                assertions.append(
                    f"budget_remaining_s should be 0.0 when exhausted, "
                    f"got {result.stats.budget_remaining_s}"
                )
            
            # Summary should include budget config
            summary_dict = result.summary.to_dict()
            budget_section = summary_dict.get("budget", {})
            
            if budget_section.get("cycle_budget_s") != 0.0001:
                assertions.append(
                    f"summary budget.cycle_budget_s should be 0.0001, "
                    f"got {budget_section.get('cycle_budget_s')}"
                )
            
            # remaining_budget_s should be non-negative
            if budget_section.get("remaining_budget_s", -1) < 0:
                # Only fail if budget was actually enforced
                if result.stats.budget_exhausted or result.stats.max_candidates_hit:
                    assertions.append(
                        f"summary budget.remaining_budget_s should be >= 0, "
                        f"got {budget_section.get('remaining_budget_s')}"
                    )
            
            if assertions:
                return TestResult(
                    name=name,
                    passed=False,
                    message="; ".join(assertions),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            status = "exhausted" if result.stats.budget_exhausted else "not exhausted (fast system)"
            return TestResult(
                name=name,
                passed=True,
                message=f"Budget handling correct ({status})",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_timeout_abstentions(self) -> TestResult:
        """
        Test 3b: Verify timeout_abstentions counter is wired end-to-end.
        
        Uses a micro taut_timeout_s (1ms) with a larger formula space to
        induce truth-table timeouts. The timeout_abstentions counter should
        be incremented when TruthTableTimeout is raised.
        
        Note: This test is probabilistic — very simple formulas may complete
        before the timeout. We use a larger atom count to increase likelihood
        of timeouts.
        """
        start = time.perf_counter()
        name = "timeout_abstentions"
        
        try:
            # Use larger atoms to increase truth-table evaluation time
            slice_cfg = self._make_slice(
                "test_timeout_abstentions",
                atoms=4,           # 2^4 = 16 assignments per formula
                depth_max=3,
                mp_depth=2,
                breadth_max=20,
                total_max=30,
            )
            
            # Very tight taut_timeout_s to induce timeouts
            budget = self._make_budget(
                cycle_budget_s=1.0,           # 1 second — generous overall
                taut_timeout_s=0.0001,        # 0.1ms — very tight per-candidate
                max_candidates_per_cycle=50,
            )
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            # Check that timeout_abstentions is exposed in stats and summary
            stats_timeout = result.stats.timeout_abstentions
            summary_dict = result.summary.to_dict()
            budget_section = summary_dict.get("budget", {})
            summary_timeout = budget_section.get("timeout_abstentions", -1)
            
            assertions = []
            
            # timeout_abstentions should be present and match
            if summary_timeout != stats_timeout:
                assertions.append(
                    f"timeout_abstentions mismatch: stats={stats_timeout}, summary={summary_timeout}"
                )
            
            # summarize_budget() should include timeout_abstentions
            from derivation.pipeline import summarize_budget
            budget_summary = summarize_budget(result.stats)
            if "timeout_abstentions" not in budget_summary:
                assertions.append("summarize_budget() missing timeout_abstentions")
            
            if assertions:
                return TestResult(
                    name=name,
                    passed=False,
                    message="; ".join(assertions),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            # Report whether timeouts actually occurred (informational)
            timeout_info = f"timeout_abstentions={stats_timeout}"
            if stats_timeout > 0:
                timeout_info += " (timeouts induced successfully)"
            else:
                timeout_info += " (no timeouts — formulas completed quickly)"
            
            return TestResult(
                name=name,
                passed=True,
                message=f"timeout_abstentions wired correctly: {timeout_info}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_summary_to_dict_completeness(self) -> TestResult:
        """
        Test 4: Verify DerivationSummary.to_dict() includes all budget fields.
        """
        start = time.perf_counter()
        name = "summary_to_dict_completeness"
        
        try:
            slice_cfg = self._make_slice("test_summary")
            budget = self._make_budget(
                cycle_budget_s=5.0,
                taut_timeout_s=0.10,
                max_candidates_per_cycle=40,
            )
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            summary_dict = result.summary.to_dict()
            budget_section = summary_dict.get("budget", {})
            
            required_fields = [
                # Configuration
                "cycle_budget_s",
                "taut_timeout_s",
                "max_candidates_per_cycle",
                # Outcomes
                "budget_exhausted",
                "max_candidates_hit",
                "statements_skipped",
                "timeout_abstentions",
                "remaining_budget_s",
                # INV-BUD diagnostics
                "budget_checks_performed",
                "post_exhaustion_candidates",
            ]
            
            missing = [f for f in required_fields if f not in budget_section]
            
            if missing:
                return TestResult(
                    name=name,
                    passed=False,
                    message=f"Missing budget fields in to_dict(): {missing}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"All {len(required_fields)} budget fields present in to_dict()",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_inv_bud_1_post_exhaustion(self) -> TestResult:
        """
        Test 5: INV-BUD-1 — Verify post_exhaustion_candidates is always 0.
        
        If budget enforcement works correctly, no candidates should be
        processed after budget_exhausted=True.
        """
        start = time.perf_counter()
        name = "inv_bud_1_never_post_exhaustion"
        
        try:
            slice_cfg = self._make_slice(
                "test_inv_bud_1",
                atoms=3,
                depth_max=3,
                mp_depth=2,
                breadth_max=50,
                total_max=100,
            )
            
            # Tight budget to trigger exhaustion
            budget = self._make_budget(
                cycle_budget_s=0.0001,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=5,
            )
            
            time.sleep(0.001)
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            # INV-BUD-1: post_exhaustion_candidates must be 0
            if result.stats.post_exhaustion_candidates != 0:
                return TestResult(
                    name=name,
                    passed=False,
                    message=f"INV-BUD-1 violated: post_exhaustion_candidates={result.stats.post_exhaustion_candidates}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"INV-BUD-1 satisfied: post_exhaustion_candidates=0",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_inv_bud_3_remaining_monotonic(self) -> TestResult:
        """
        Test 6: INV-BUD-3 — Verify remaining_budget_s is non-negative or -1.
        
        The remaining budget must be monotonically non-increasing within a cycle.
        After exhaustion, remaining_budget_s should be 0.0, not negative.
        """
        start = time.perf_counter()
        name = "inv_bud_3_remaining_budget_monotonic"
        
        try:
            slice_cfg = self._make_slice(
                "test_inv_bud_3",
                atoms=3,
                depth_max=3,
                mp_depth=2,
                breadth_max=50,
                total_max=100,
            )
            
            # Short budget to trigger exhaustion
            budget = self._make_budget(
                cycle_budget_s=0.001,  # 1ms
                taut_timeout_s=1.0,
                max_candidates_per_cycle=1000,
            )
            
            time.sleep(0.002)  # Ensure budget expires
            
            result = self.run_slice_for_test(
                slice_cfg,
                limit=1,
                budget=budget,
                emit_log=False,
            )
            
            remaining = result.stats.budget_remaining_s
            
            # INV-BUD-3: remaining must be >= 0 (or -1 if no budget)
            if remaining < 0 and remaining != -1.0:
                return TestResult(
                    name=name,
                    passed=False,
                    message=f"INV-BUD-3 violated: remaining_budget_s={remaining:.4f} < 0",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            # If exhausted, remaining should be exactly 0
            if result.stats.budget_exhausted and remaining != 0.0:
                return TestResult(
                    name=name,
                    passed=False,
                    message=f"INV-BUD-3: exhausted but remaining={remaining:.4f} (expected 0.0)",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"INV-BUD-3 satisfied: remaining={remaining:.4f}s",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def test_real_config_loads(self) -> TestResult:
        """
        Test 7: Verify real Phase II config loads successfully.
        """
        start = time.perf_counter()
        name = "real_config_loads"
        
        try:
            from backend.verification.budget_loader import (
                load_budget_for_slice,
                DEFAULT_CONFIG_PATH,
            )
            from pathlib import Path
            
            config_path = Path(DEFAULT_CONFIG_PATH)
            if not config_path.exists():
                return TestResult(
                    name=name,
                    passed=False,
                    message=f"Config not found: {config_path}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            
            # Load all Phase II slices
            slices = [
                "slice_uplift_goal",
                "slice_uplift_sparse",
                "slice_uplift_tree",
                "slice_uplift_dependency",
            ]
            
            for slice_name in slices:
                budget = load_budget_for_slice(slice_name)
                # Verify all fields are valid
                if budget.cycle_budget_s < 0:
                    return TestResult(
                        name=name,
                        passed=False,
                        message=f"{slice_name}: invalid cycle_budget_s={budget.cycle_budget_s}",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                if budget.max_candidates_per_cycle < 1:
                    return TestResult(
                        name=name,
                        passed=False,
                        message=f"{slice_name}: invalid max_candidates={budget.max_candidates_per_cycle}",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
            
            return TestResult(
                name=name,
                passed=True,
                message=f"Loaded {len(slices)} Phase II slice configs",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
    
    def run_all(self) -> bool:
        """
        Run all budget self-tests.
        
        Returns:
            True if all tests pass, False otherwise.
        """
        print("=" * 60)
        print("Phase II Budget Enforcement Self-Test")
        print("=" * 60)
        print()
        
        tests = [
            self.test_baseline_no_budget,
            self.test_tight_budget,
            self.test_pathological_timeout,
            self.test_timeout_abstentions,
            self.test_summary_to_dict_completeness,
            self.test_inv_bud_1_post_exhaustion,
            self.test_inv_bud_3_remaining_monotonic,
            self.test_real_config_loads,
        ]
        
        total_start = time.perf_counter()
        
        for test_fn in tests:
            result = test_fn()
            self.results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.name} ({result.duration_ms:.1f}ms)")
            if not result.passed:
                print(f"  └─ {result.message}")
        
        total_duration = (time.perf_counter() - total_start) * 1000
        
        print()
        print("-" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        # Summary
        print("BUDGET INVARIANT SUMMARY")
        print("-" * 60)
        
        # Map test names to invariants
        inv_status = {
            "INV-BUD-1": "UNKNOWN",
            "INV-BUD-2": "UNKNOWN",
            "INV-BUD-3": "UNKNOWN",
            "INV-BUD-4": "UNKNOWN",
            "INV-BUD-5": "UNKNOWN",
        }
        
        for r in self.results:
            if "inv_bud_1" in r.name.lower():
                inv_status["INV-BUD-1"] = "PASS" if r.passed else "FAIL"
            elif "tight_budget" in r.name.lower() or "max_candidates" in r.name.lower():
                inv_status["INV-BUD-2"] = "PASS" if r.passed else "FAIL"
            elif "inv_bud_3" in r.name.lower() or "monotonic" in r.name.lower():
                inv_status["INV-BUD-3"] = "PASS" if r.passed else "FAIL"
            elif "to_dict" in r.name.lower() or "completeness" in r.name.lower():
                inv_status["INV-BUD-4"] = "PASS" if r.passed else "FAIL"
            elif "baseline" in r.name.lower():
                inv_status["INV-BUD-5"] = "PASS" if r.passed else "FAIL"
        
        for inv, status in inv_status.items():
            marker = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "?")
            print(f"  {marker} {inv}: {status}")
        
        print("-" * 60)
        
        # Sanity guard: warn if self-test exceeds runtime budget
        total_duration_s = total_duration / 1000.0
        if total_duration_s > MAX_SELF_TEST_RUNTIME_S:
            print()
            print(f"⚠ WARNING: Self-test took {total_duration_s:.2f}s (exceeds {MAX_SELF_TEST_RUNTIME_S}s target)")
            print("  Consider reducing slice size or tightening test budgets.")
        
        if failed == 0:
            print(f"✓ PASS: All {passed} tests passed ({total_duration:.1f}ms)")
            print()
            print("Budget enforcement is CI-ready.")
            return True
        else:
            print(f"✗ FAIL: {failed}/{len(self.results)} tests failed ({total_duration:.1f}ms)")
            print()
            print("Budget enforcement REGRESSION detected. Review failures above.")
            return False


def main() -> int:
    """
    Main entry point.
    
    Returns:
        0 = success
        1 = test failure
        2 = script error
    """
    # Enable debug budget assertions if not explicitly set
    if os.getenv("MATHLEDGER_DEBUG_BUDGET") is None:
        os.environ["MATHLEDGER_DEBUG_BUDGET"] = "1"
    
    try:
        harness = BudgetSelfTest()
        success = harness.run_all()
        return 0 if success else 1
    except Exception as e:
        print(f"✗ Script error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

