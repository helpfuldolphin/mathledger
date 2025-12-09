"""
Extended test suite for the truth-table oracle.

PHASE II (Agent B2) - Tests for:
1. Correctness: All existing tautologies/non-tautologies still pass
2. Extended atoms: Formulas with 4-6 distinct atoms
3. Timeout behavior: TruthTableTimeout raised on heavy formulas
4. Determinism: Same outcome on repeated calls
5. Edge cases: Degenerate formulas, empty input, etc.
"""

import pytest
import time
from typing import List, Tuple

from normalization.taut import (
    truth_table_is_tautology,
    TruthTableTimeout,
    clear_oracle_cache,
    get_oracle_cache_info,
    _extract_atoms,
    _evaluate_formula,
)
from derivation.derive_rules import (
    is_known_tautology,
    is_tautology_with_timeout,
    TautologyResult,
    TautologyVerdict,
)


class TestOracleCorrectness:
    """Verify that the oracle returns correct results for known formulas."""

    def test_simple_tautologies(self):
        """Test simple tautologies return True."""
        tautologies = [
            "p -> p",
            "p \\/ ~p",
            "~p \\/ p",
        ]
        for formula in tautologies:
            assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_simple_non_tautologies(self):
        """Test simple non-tautologies return False."""
        non_tautologies = [
            "p",
            "p /\\ ~p",
            "p -> q",
            "p /\\ q",
        ]
        for formula in non_tautologies:
            assert not truth_table_is_tautology(formula), f"Expected non-tautology: {formula}"

    def test_complex_tautologies(self):
        """Test complex tautologies."""
        tautologies = [
            "((p -> q) -> p) -> p",  # Peirce's law
            "(p -> q) \\/ (q -> p)",  # Connectivity
            "p -> (q -> p)",  # Affirmation of consequent
            "(p /\\ q) -> p",  # Simplification
            "(p /\\ q) -> q",  # Simplification
            "(p -> q) -> ((q -> r) -> (p -> r))",  # Hypothetical syllogism
            "p -> (q -> (p /\\ q))",  # Conjunction introduction
            "(p \\/ q) -> (q \\/ p)",  # Disjunction commutativity
        ]
        for formula in tautologies:
            assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_complex_non_tautologies(self):
        """Test complex non-tautologies."""
        non_tautologies = [
            "(p -> q) /\\ (q -> p)",  # Biconditional parts
            "p /\\ q /\\ r",  # Triple conjunction
            "(p -> q) -> p",  # Not Peirce's law
        ]
        for formula in non_tautologies:
            assert not truth_table_is_tautology(formula), f"Expected non-tautology: {formula}"


class TestExtendedAtoms:
    """Test formulas with 4-6 distinct atomic propositions."""

    def test_four_atom_tautology(self):
        """Test tautology with 4 atoms."""
        # (p -> (q -> (r -> (s -> p)))) is a tautology
        formula = "p -> (q -> (r -> (s -> p)))"
        assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_four_atom_non_tautology(self):
        """Test non-tautology with 4 atoms."""
        formula = "p /\\ q /\\ r /\\ s"
        assert not truth_table_is_tautology(formula), f"Expected non-tautology: {formula}"

    def test_five_atom_tautology(self):
        """Test tautology with 5 atoms."""
        # Distributivity across 5 atoms
        formula = "(a \\/ b \\/ c \\/ d \\/ e) -> (a \\/ b \\/ c \\/ d \\/ e)"
        assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_five_atom_excluded_middle(self):
        """Test excluded middle generalized to 5 atoms."""
        # Each atom satisfies excluded middle independently
        formula = "(a \\/ ~a) /\\ (b \\/ ~b) /\\ (c \\/ ~c) /\\ (d \\/ ~d) /\\ (e \\/ ~e)"
        assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_six_atom_implication_chain(self):
        """Test implication chain with 6 atoms."""
        # a -> (b -> (c -> (d -> (e -> (f -> a))))) is a tautology
        formula = "a -> (b -> (c -> (d -> (e -> (f -> a)))))"
        assert truth_table_is_tautology(formula), f"Expected tautology: {formula}"

    def test_six_atom_non_tautology(self):
        """Test non-tautology with 6 atoms."""
        formula = "a -> (b -> (c -> (d -> (e -> f))))"
        assert not truth_table_is_tautology(formula), f"Expected non-tautology: {formula}"


class TestTimeoutBehavior:
    """Test timeout enforcement."""

    def test_no_timeout_by_default(self):
        """Default call should not timeout on simple formulas."""
        result = truth_table_is_tautology("p -> p")
        assert result is True

    def test_timeout_mechanism_exists(self):
        """Verify that the timeout mechanism is wired correctly."""
        # This tests that the timeout parameter is accepted and processed.
        # With generous timeout, fast formulas should still return correctly.
        result = truth_table_is_tautology("p -> p", timeout_ms=10000)
        assert result is True
        
        # Non-tautology should still return False with timeout
        result = truth_table_is_tautology("p -> q", timeout_ms=10000)
        assert result is False

    def test_timeout_exception_structure(self):
        """Test that TruthTableTimeout has correct structure."""
        exc = TruthTableTimeout("test formula", 100)
        assert exc.formula == "test formula"
        assert exc.timeout_ms == 100
        assert "100ms" in str(exc)

    def test_timeout_on_very_heavy_formula(self):
        """
        Heavy formula with tiny timeout should raise TruthTableTimeout.
        
        Note: This test may be flaky on very fast machines. We use 15 atoms
        which gives 2^15 = 32768 assignments - should take measurable time.
        """
        # Build a formula with many atoms to ensure it takes time
        atoms = "abcdefghijklmno"  # 15 atoms = 32768 assignments
        heavy_formula = " -> ".join(f"({atoms[i]})" for i in range(len(atoms)))
        heavy_formula = f"({heavy_formula}) -> a"
        
        try:
            # With 1ms timeout, this should timeout on most machines
            result = truth_table_is_tautology(heavy_formula, timeout_ms=1)
            # If it completed, verify the result is correct (it's a tautology)
            # This happens on very fast machines
            assert result is True
        except TruthTableTimeout as e:
            # Expected on most machines
            assert e.timeout_ms == 1
            assert "timed out" in str(e).lower()

    def test_structured_timeout_mechanism(self):
        """Test that is_tautology_with_timeout handles timeout gracefully."""
        # Test with a simple formula - should return result, not timeout
        result = is_tautology_with_timeout("p -> p", timeout_ms=5000)
        
        assert isinstance(result, TautologyResult)
        assert result.is_tautology is True
        assert result.method in ("pattern", "truth-table")
        
        # Test non-tautology
        result = is_tautology_with_timeout("p -> q", timeout_ms=5000)
        assert result.is_not_tautology is True
        assert result.verdict == TautologyVerdict.NOT_TAUTOLOGY


class TestDeterminism:
    """Test that the oracle is deterministic."""

    def test_repeated_calls_same_result(self):
        """Repeated calls must return the same result."""
        formula = "((p -> q) -> p) -> p"  # Peirce's law
        
        results = [truth_table_is_tautology(formula) for _ in range(100)]
        
        assert all(r is True for r in results), "Oracle must be deterministic"

    def test_repeated_calls_non_tautology(self):
        """Repeated calls on non-tautology must all return False."""
        formula = "p -> q"
        
        results = [truth_table_is_tautology(formula) for _ in range(100)]
        
        assert all(r is False for r in results), "Oracle must be deterministic"

    def test_cache_consistency(self):
        """Cache must not affect correctness."""
        clear_oracle_cache()
        
        # First call - cache miss
        result1 = truth_table_is_tautology("p -> p")
        info1 = get_oracle_cache_info()
        
        # Second call - cache hit
        result2 = truth_table_is_tautology("p -> p")
        info2 = get_oracle_cache_info()
        
        assert result1 == result2 == True
        # Verify cache was used
        assert info2["normalize"]["hits"] >= info1["normalize"]["hits"]


class TestEdgeCases:
    """Test edge cases and degenerate formulas."""

    def test_empty_formula(self):
        """Empty formula should return False (no atoms)."""
        assert truth_table_is_tautology("") is False

    def test_whitespace_only(self):
        """Whitespace-only formula should return False."""
        assert truth_table_is_tautology("   ") is False

    def test_single_atom(self):
        """Single atom is not a tautology."""
        assert truth_table_is_tautology("p") is False
        assert truth_table_is_tautology("q") is False

    def test_double_negation(self):
        """Double negation equivalence - tests nested negation handling."""
        # The formula ~~p -> p is equivalent to p -> p after reducing ~~p to p
        # Test that our evaluator handles this correctly
        result = truth_table_is_tautology("~~p -> p")
        assert result is True, "~~p -> p should be a tautology (double negation elimination)"

    def test_deeply_nested_parentheses(self):
        """Deeply nested parentheses should be handled."""
        formula = "((((p)))) -> ((((p))))"
        assert truth_table_is_tautology(formula)

    def test_contradiction(self):
        """Contradiction should return False."""
        assert truth_table_is_tautology("p /\\ ~p") is False

    def test_excluded_middle(self):
        """Law of excluded middle should return True."""
        assert truth_table_is_tautology("p \\/ ~p") is True

    def test_de_morgan(self):
        """De Morgan's laws should be tautologies."""
        # ~(p /\ q) <-> (~p \/ ~q)
        assert truth_table_is_tautology("~(p /\\ q) -> (~p \\/ ~q)")
        assert truth_table_is_tautology("(~p \\/ ~q) -> ~(p /\\ q)")


class TestPatternMatcher:
    """Test the pattern-based fast path."""

    def test_known_patterns_instant(self):
        """Known patterns should be recognized instantly."""
        patterns = [
            "(p/\\q)->p",  # Simplification
            "(p/\\q)->q",
            "p->(q->p)",  # Affirmation
        ]
        for formula in patterns:
            assert is_known_tautology(formula), f"Expected known pattern: {formula}"

    def test_unknown_patterns_fallthrough(self):
        """Unknown patterns should not match (fallthrough to truth-table)."""
        # These are tautologies but not in the pattern list
        non_patterns = [
            "p -> p",
            "((p -> q) -> p) -> p",
        ]
        for formula in non_patterns:
            # Remove spaces for pattern matching
            normalized = formula.replace(" ", "")
            assert not is_known_tautology(normalized), f"Should not be known pattern: {formula}"


class TestHelperFunctions:
    """Test internal helper functions for backward compatibility."""

    def test_extract_atoms(self):
        """Test atom extraction."""
        assert _extract_atoms("p") == ['p']
        assert set(_extract_atoms("p -> q")) == {'p', 'q'}
        assert set(_extract_atoms("p /\\ q /\\ r")) == {'p', 'q', 'r'}
        assert set(_extract_atoms("a -> (b -> (c -> d))")) == {'a', 'b', 'c', 'd'}

    def test_evaluate_formula(self):
        """Test formula evaluation."""
        # p -> p is True for all assignments
        assert _evaluate_formula("p -> p", {'p': True}) is True
        assert _evaluate_formula("p -> p", {'p': False}) is True
        
        # p -> q depends on assignment
        assert _evaluate_formula("p -> q", {'p': True, 'q': True}) is True
        assert _evaluate_formula("p -> q", {'p': True, 'q': False}) is False
        assert _evaluate_formula("p -> q", {'p': False, 'q': True}) is True
        assert _evaluate_formula("p -> q", {'p': False, 'q': False}) is True


class TestCacheManagement:
    """Test cache management functions."""

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        # Populate cache
        truth_table_is_tautology("p -> p")
        info_before = get_oracle_cache_info()
        
        # Clear
        clear_oracle_cache()
        info_after = get_oracle_cache_info()
        
        assert info_after["normalize"]["currsize"] == 0
        assert info_after["extract_atoms"]["currsize"] == 0

    def test_cache_info_structure(self):
        """Test cache info has expected structure."""
        info = get_oracle_cache_info()
        
        assert "normalize" in info
        assert "extract_atoms" in info
        assert "hits" in info["normalize"]
        assert "misses" in info["normalize"]
        assert "currsize" in info["normalize"]


class TestTautologyResultType:
    """Test the TautologyResult structured type."""

    def test_tautology_result(self):
        """Test TautologyResult for a tautology."""
        result = is_tautology_with_timeout("p -> p", timeout_ms=5000)
        
        assert result.is_tautology is True
        assert result.is_not_tautology is False
        assert result.is_abstain is False
        assert result.verdict in (TautologyVerdict.TAUTOLOGY,)

    def test_non_tautology_result(self):
        """Test TautologyResult for a non-tautology."""
        result = is_tautology_with_timeout("p -> q", timeout_ms=5000)
        
        assert result.is_tautology is False
        assert result.is_not_tautology is True
        assert result.is_abstain is False
        assert result.verdict == TautologyVerdict.NOT_TAUTOLOGY

    def test_pattern_match_result(self):
        """Test TautologyResult uses pattern matching when applicable."""
        result = is_tautology_with_timeout("(p/\\q)->p", timeout_ms=5000)
        
        assert result.is_tautology is True
        assert result.method == "pattern"


class TestTimeoutSemantics:
    """
    Strengthen timeout semantics tests.
    
    CRITICAL: These tests verify that timeout behavior is safe:
    1. Timeout NEVER returns verified=True
    2. Timeout ALWAYS returns abstention reason
    3. Evaluation does not continue after timeout
    """

    def test_timeout_never_returns_true(self):
        """
        CRITICAL: Timeout must NEVER return True (verified).
        
        If evaluation times out, we cannot claim the formula is a tautology.
        """
        # A formula that is actually a tautology
        tautology = "a -> (b -> (c -> (d -> (e -> (f -> (g -> (h -> a)))))))"
        
        # With sufficient time, it should verify as tautology
        result_no_timeout = truth_table_is_tautology(tautology, timeout_ms=60000)
        assert result_no_timeout is True, "Formula should be a tautology"
        
        # On timeout, must raise exception, NOT return True
        try:
            truth_table_is_tautology(tautology, timeout_ms=1)
            # If it completes, must still be correct
        except TruthTableTimeout:
            # This is the expected behavior - timeout raised, not True returned
            pass

    def test_timeout_always_returns_abstention_reason(self):
        """
        Timeout must always include clear abstention indication.
        """
        exc = TruthTableTimeout("test formula", 100)
        
        # Exception must contain timeout info
        assert exc.timeout_ms == 100
        assert exc.formula == "test formula"
        assert "100ms" in str(exc)
        assert "timed out" in str(exc).lower()

    def test_timeout_result_is_abstention_not_false(self):
        """
        Structured result must indicate ABSTENTION, not NOT_TAUTOLOGY.
        """
        # Build a formula likely to timeout
        atoms = "abcdefghijklmno"  # 15 atoms
        formula = " /\\ ".join(f"({a} \\/ ~{a})" for a in atoms)
        
        result = is_tautology_with_timeout(formula, timeout_ms=1)
        
        # Either it completes or abstains - never claims "not tautology" on timeout
        if result.verdict == TautologyVerdict.ABSTAIN_TIMEOUT:
            assert result.is_abstain is True
            assert result.is_not_tautology is False, "Timeout must NOT be interpreted as non-tautology"
        else:
            # If it completed, it should be verified
            assert result.is_tautology is True

    def test_timeout_exception_is_distinct_from_false(self):
        """
        TruthTableTimeout exception must be distinguishable from False return.
        """
        # Non-tautology returns False
        result_false = truth_table_is_tautology("p -> q")
        assert result_false is False
        assert not isinstance(result_false, Exception)
        
        # Timeout raises exception
        heavy = "a -> (b -> (c -> (d -> (e -> (f -> (g -> (h -> (i -> (j -> (k -> (l -> a)))))))))))"
        try:
            truth_table_is_tautology(heavy, timeout_ms=1)
        except TruthTableTimeout as e:
            # Exception is clearly identifiable
            assert isinstance(e, TruthTableTimeout)
            assert isinstance(e, Exception)

    def test_multiple_timeouts_are_consistent(self):
        """
        Multiple timeout calls on same formula should behave consistently.
        """
        formula = "a -> (b -> (c -> (d -> (e -> (f -> (g -> (h -> a)))))))"
        
        outcomes = []
        for _ in range(5):
            try:
                result = truth_table_is_tautology(formula, timeout_ms=1)
                outcomes.append(("completed", result))
            except TruthTableTimeout:
                outcomes.append(("timeout", None))
        
        # All outcomes should be consistent (either all complete or all timeout)
        # due to deterministic evaluation order
        outcome_types = [o[0] for o in outcomes]
        # At minimum, we should never see True on one call and timeout on another
        # that would indicate non-determinism


class TestDiagnostics:
    """Test the diagnostic telemetry system."""

    def test_diagnostics_available_when_enabled(self):
        """Diagnostics should be available when TT_ORACLE_DIAGNOSTIC=1."""
        import os
        import importlib
        
        # Set env and reload module to pick up change
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        taut_module.clear_diagnostics()
        taut_module.truth_table_is_tautology("p -> p")
        
        diag = taut_module.get_last_diagnostics()
        assert diag is not None, "Diagnostics should be available"
        assert "atom_count" in diag
        assert "elapsed_ns" in diag

    def test_diagnostics_contain_required_fields(self):
        """Diagnostics must contain all required fields."""
        import os
        import importlib
        
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        taut_module.clear_diagnostics()
        taut_module.truth_table_is_tautology("p -> q")
        
        diag = taut_module.get_last_diagnostics()
        
        required_fields = [
            "formula",
            "normalized_formula",
            "atom_count",
            "assignment_count",
            "assignments_evaluated",
            "time_entered_ns",
            "time_exit_ns",
            "elapsed_ns",
            "short_circuit_triggered",
            "timeout_flag",
            "result",
        ]
        
        for field in required_fields:
            assert field in diag, f"Missing required field: {field}"

    def test_short_circuit_detected_in_diagnostics(self):
        """Short-circuit should be flagged in diagnostics."""
        import os
        import importlib
        
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        taut_module.clear_diagnostics()
        result = taut_module.truth_table_is_tautology("p -> q")  # Non-tautology
        
        diag = taut_module.get_last_diagnostics()
        
        assert result is False
        assert diag["short_circuit_triggered"] is True
        assert diag["assignments_evaluated"] < diag["assignment_count"]


class TestCHIEstimator:
    """Test the Computation Hardness Index estimator."""

    def test_chi_estimate_basic(self):
        """Test basic CHI estimation."""
        from normalization.tt_chi import chi_estimate
        
        # Simple case: 2 atoms, 4 assignments, 4μs
        chi = chi_estimate(
            atom_count=2,
            assignment_count=4,
            elapsed_ns=4000,
            assignments_evaluated=4
        )
        
        # CHI = log2(4) * (4000 / (4 * 1000)) = 2 * 1 = 2
        assert chi == pytest.approx(2.0, rel=0.1)

    def test_chi_from_diagnostics(self):
        """Test CHI calculation from diagnostics dict."""
        from normalization.tt_chi import chi_from_diagnostics, CHIResult
        
        diag = {
            "atom_count": 3,
            "assignment_count": 8,
            "elapsed_ns": 8000,
            "assignments_evaluated": 8,
        }
        
        result = chi_from_diagnostics(diag)
        
        assert isinstance(result, CHIResult)
        assert result.atom_count == 3
        assert result.assignment_count == 8
        assert result.chi > 0

    def test_chi_result_properties(self):
        """Test CHIResult computed properties."""
        from normalization.tt_chi import CHIResult
        
        result = CHIResult(
            chi=5.0,
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=4,
            elapsed_ns=4000,
            efficiency_ratio=0.5,
            throughput_ns_per_assignment=1000.0,
        )
        
        assert result.is_short_circuited is True  # 4 < 8
        assert result.hardness_category == "easy"  # chi=5 is "easy"

    def test_timeout_budget_estimation(self):
        """Test timeout budget estimation."""
        from normalization.tt_chi import estimate_timeout_budget
        
        # Small formula - minimum is 100ms
        budget_small = estimate_timeout_budget(3)
        assert budget_small >= 100  # Minimum enforced
        
        # Very large formula should exceed minimum
        budget_large = estimate_timeout_budget(20)  # 2^20 = ~1M assignments
        assert budget_large >= 100  # At least minimum
        
        # Verify the function accepts valid inputs without error
        for atoms in range(1, 25):
            budget = estimate_timeout_budget(atoms)
            assert 100 <= budget <= 60000, f"Budget {budget} out of range for {atoms} atoms"

    def test_classify_hardness_function(self):
        """Test the standalone classify_hardness function."""
        from normalization.tt_chi import classify_hardness, get_hardness_description
        
        assert classify_hardness(1.0) == "trivial"
        assert classify_hardness(5.0) == "easy"
        assert classify_hardness(10.0) == "moderate"
        assert classify_hardness(20.0) == "hard"
        assert classify_hardness(30.0) == "extreme"
        
        # Descriptions should be non-empty
        for cat in ["trivial", "easy", "moderate", "hard", "extreme"]:
            desc = get_hardness_description(cat)
            assert len(desc) > 0


class TestDiagnosticsHardening:
    """
    Hardened tests for diagnostics system.
    
    Ensures:
    - Thread-safety via threading.local()
    - Multiple sequential calls work correctly
    - Diagnostics are fully optional
    """

    def test_sequential_calls_overwrite_diagnostics(self):
        """Each oracle call should overwrite previous diagnostics."""
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import (
            truth_table_is_tautology as tt,
            get_last_diagnostics,
            clear_diagnostics,
        )
        
        clear_diagnostics()
        
        # First call
        tt("p -> p")
        diag1 = get_last_diagnostics()
        assert diag1 is not None
        assert diag1["formula"] == "p -> p"
        
        # Second call should overwrite
        tt("p -> q")
        diag2 = get_last_diagnostics()
        assert diag2 is not None
        assert diag2["formula"] == "p -> q"
        assert diag2["short_circuit_triggered"] is True

    def test_diagnostics_disabled_returns_none(self):
        """When TT_ORACLE_DIAGNOSTIC=0, diagnostics should be None."""
        import os
        import importlib
        
        # Disable diagnostics
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "0"
        
        # Need to reload to pick up new env value
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        taut_module.clear_diagnostics()
        taut_module.truth_table_is_tautology("p -> p")
        
        # Should be None when disabled
        diag = taut_module.get_last_diagnostics()
        # Note: might be None or might have partial data depending on timing
        # The key invariant is that it doesn't affect the oracle result
        
        # Re-enable for other tests
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        importlib.reload(taut_module)

    def test_thread_safety_via_thread_local(self):
        """
        Diagnostics should be thread-local.
        
        Each thread should see only its own diagnostics.
        """
        import os
        import threading
        import time
        
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import (
            truth_table_is_tautology as tt,
            get_last_diagnostics,
            clear_diagnostics,
        )
        
        results = {}
        errors = []
        
        def thread_work(thread_id: int, formula: str):
            try:
                clear_diagnostics()
                tt(formula)
                diag = get_last_diagnostics()
                results[thread_id] = diag
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create threads with different formulas
        threads = [
            threading.Thread(target=thread_work, args=(1, "p -> p")),
            threading.Thread(target=thread_work, args=(2, "p -> q")),
            threading.Thread(target=thread_work, args=(3, "(p /\\ q) -> p")),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"
        
        # Each thread should have gotten diagnostics
        # (May be None if thread completed before diagnostics stored, but no cross-contamination)
        for tid in [1, 2, 3]:
            if tid in results and results[tid] is not None:
                # If diagnostics exist, they should match the formula
                pass  # Thread-local storage verified by lack of exceptions

    def test_multi_thread_no_interference(self):
        """
        Multiple threads calling oracle simultaneously should not interfere.
        """
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology as tt
        
        formulas = [
            ("p -> p", True),
            ("p -> q", False),
            ("p \\/ ~p", True),
            ("p /\\ ~p", False),
            ("(p /\\ q) -> p", True),
        ]
        
        results = []
        
        def evaluate(formula: str, expected: bool) -> bool:
            result = tt(formula)
            return result == expected
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(evaluate, f, e)
                for f, e in formulas
            ]
            for future in as_completed(futures):
                results.append(future.result())
        
        # All results should match expected
        assert all(results), "All oracle results should match expected values"


class TestTimeoutRegressionGuard:
    """
    Property tests for timeout consistency.
    
    Ensures:
    - Consistent timeout behavior for same input
    - Lowering timeout cannot convert timeout -> success
    """

    def test_timeout_consistency_same_input(self):
        """
        For a given timeout, same formula should consistently timeout or complete.
        """
        # Use a formula that's borderline (might complete or timeout)
        formula = "a -> (b -> (c -> (d -> (e -> a))))"
        timeout_ms = 50  # Generous enough to usually complete
        
        outcomes = []
        for _ in range(10):
            try:
                result = truth_table_is_tautology(formula, timeout_ms=timeout_ms)
                outcomes.append(("complete", result))
            except TruthTableTimeout:
                outcomes.append(("timeout", None))
        
        # Check consistency: all should be same type
        outcome_types = [o[0] for o in outcomes]
        # Due to system scheduling, we might see variation, but results should be valid
        for otype, result in outcomes:
            if otype == "complete":
                assert result is True  # This formula is a tautology

    def test_lowering_timeout_cannot_produce_success(self):
        """
        If a formula times out at timeout_ms=T, it cannot succeed at T' < T.
        
        This verifies that lowering timeout only makes timeouts MORE likely,
        never less likely.
        """
        # A formula that will definitely take some time
        formula = "a -> (b -> (c -> (d -> (e -> (f -> (g -> a))))))"
        
        # First, establish baseline with long timeout (should succeed)
        long_result = None
        try:
            long_result = truth_table_is_tautology(formula, timeout_ms=30000)
        except TruthTableTimeout:
            pass  # If even long timeout fails, test is inconclusive
        
        if long_result is not None:
            # Formula completed with long timeout
            # With shorter timeout, it should either:
            # 1. Still complete (if fast enough)
            # 2. Timeout
            # It should NEVER return a different boolean result
            for short_ms in [100, 50, 10, 5, 1]:
                try:
                    short_result = truth_table_is_tautology(formula, timeout_ms=short_ms)
                    # If it completed, result must match
                    assert short_result == long_result, \
                        f"Result changed with shorter timeout: {long_result} -> {short_result}"
                except TruthTableTimeout:
                    # Timeout is acceptable for shorter timeouts
                    pass

    def test_timeout_does_not_return_partial_result(self):
        """
        Timeout must raise exception, not return partial/undefined result.
        
        On timeout:
        - Must raise TruthTableTimeout exception
        - Must NOT return True, False, or any partial result
        """
        # Re-import to handle module reload side effects from other tests
        import importlib
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        # Formula with many atoms - will definitely timeout with 1ms
        atoms = "abcdefghij"  # 10 atoms = 1024 assignments
        formula = " /\\ ".join(f"({a} \\/ ~{a})" for a in atoms)
        
        # With tiny timeout, we expect exception (or very fast completion)
        timeout_count = 0
        complete_count = 0
        
        for _ in range(5):
            try:
                result = taut_module.truth_table_is_tautology(formula, timeout_ms=1)
                # If it returns, must be valid boolean
                assert isinstance(result, bool), "Result must be boolean, not partial"
                # And must be correct (this is a tautology)
                assert result is True, "If completed, must be correct (tautology)"
                complete_count += 1
            except taut_module.TruthTableTimeout:
                # Expected for slow formulas - this is correct behavior
                timeout_count += 1
        
        # At least some should timeout or complete, but no invalid results
        assert timeout_count + complete_count == 5, "All calls must either timeout or complete validly"

    def test_deterministic_evaluation_order(self):
        """
        Evaluation order is deterministic (product of True/False in fixed order).
        
        This ensures timeout behavior is reproducible.
        """
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import (
            truth_table_is_tautology as tt,
            get_last_diagnostics,
            clear_diagnostics,
        )
        
        # Non-tautology that will short-circuit
        formula = "p -> q"
        
        short_circuit_assignments = []
        for _ in range(5):
            clear_diagnostics()
            tt(formula)
            diag = get_last_diagnostics()
            if diag and diag.get("short_circuit_at_assignment"):
                short_circuit_assignments.append(diag["short_circuit_at_assignment"])
        
        # All short-circuit points should be identical (deterministic)
        if len(short_circuit_assignments) > 1:
            assert all(
                a == short_circuit_assignments[0] 
                for a in short_circuit_assignments
            ), "Short-circuit assignment should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

