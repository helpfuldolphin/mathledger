"""
Unit Tests for RFL Failure Classifier
======================================

Tests the failure classification module introduced by Agent B4 (verifier-ops-4).

PHASE II — VERIFICATION BUREAU
Tests classification of:
- Timeout (subprocess.TimeoutExpired)
- Crash (CalledProcessError, MemoryError)
- Parse error (SyntaxError, ValueError)
- Budget flags (context with {"budget_exhausted": True})
- Unknown (RuntimeError)

INVARIANT: FailureState.SUCCESS is never emitted from exception classification.
"""

import subprocess
import pytest
from unittest.mock import MagicMock

from rfl.verification.failure_classifier import (
    FailureState,
    classify_exception,
    classify_from_result,
    classify_from_status,
    normalize_legacy_key,
    LEGACY_KEY_MAP,
)


class TestFailureStateEnum:
    """Tests for the FailureState enum itself."""
    
    def test_all_states_have_unique_values(self):
        """Verify all enum members have distinct string values."""
        values = [state.value for state in FailureState]
        assert len(values) == len(set(values)), "Duplicate enum values detected"
    
    def test_expected_states_exist(self):
        """Verify all required states are defined."""
        required = {
            "success",
            "timeout_abstain",
            "budget_exhausted",
            "crash_abstain",
            "invalid_formula",
            "skipped_by_budget",
            "unknown_error",
        }
        actual = {state.value for state in FailureState}
        assert required == actual, f"Missing states: {required - actual}"


class TestClassifyException:
    """Tests for classify_exception() function."""
    
    def test_timeout_expired_returns_timeout_abstain(self):
        """subprocess.TimeoutExpired → TIMEOUT_ABSTAIN"""
        exc = subprocess.TimeoutExpired(cmd="test", timeout=60)
        result = classify_exception(exc)
        assert result == FailureState.TIMEOUT_ABSTAIN
    
    def test_memory_error_returns_crash_abstain(self):
        """MemoryError → CRASH_ABSTAIN"""
        exc = MemoryError("out of memory")
        result = classify_exception(exc)
        assert result == FailureState.CRASH_ABSTAIN
    
    def test_called_process_error_returns_crash_abstain(self):
        """subprocess.CalledProcessError → CRASH_ABSTAIN"""
        exc = subprocess.CalledProcessError(returncode=1, cmd="test")
        result = classify_exception(exc)
        assert result == FailureState.CRASH_ABSTAIN
    
    def test_nonzero_returncode_context_returns_crash_abstain(self):
        """Context with returncode != 0 → CRASH_ABSTAIN"""
        exc = RuntimeError("something failed")
        result = classify_exception(exc, context={"returncode": 1})
        assert result == FailureState.CRASH_ABSTAIN
    
    def test_syntax_error_returns_invalid_formula(self):
        """SyntaxError → INVALID_FORMULA"""
        exc = SyntaxError("invalid syntax")
        result = classify_exception(exc)
        assert result == FailureState.INVALID_FORMULA
    
    def test_value_error_with_parse_keyword_returns_invalid_formula(self):
        """ValueError with parse-related keywords → INVALID_FORMULA"""
        test_cases = [
            ValueError("invalid formula expression"),
            ValueError("syntax error in input"),
            ValueError("failed to parse"),
            ValueError("normalize failed"),
            ValueError("malformed input"),
        ]
        for exc in test_cases:
            result = classify_exception(exc)
            assert result == FailureState.INVALID_FORMULA, f"Failed for: {exc}"
    
    def test_value_error_without_parse_keyword_returns_unknown(self):
        """ValueError without parse keywords → UNKNOWN_ERROR"""
        exc = ValueError("some other error")
        result = classify_exception(exc)
        assert result == FailureState.UNKNOWN_ERROR
    
    def test_invalid_formula_context_flag(self):
        """Context with invalid_formula=True → INVALID_FORMULA"""
        exc = RuntimeError("something")
        result = classify_exception(exc, context={"invalid_formula": True})
        assert result == FailureState.INVALID_FORMULA
    
    def test_budget_exhausted_context_returns_budget_exhausted(self):
        """Context with budget_exhausted=True → BUDGET_EXHAUSTED"""
        exc = RuntimeError("budget exceeded")
        result = classify_exception(exc, context={"budget_exhausted": True})
        assert result == FailureState.BUDGET_EXHAUSTED
    
    def test_budget_exceeded_context_returns_budget_exhausted(self):
        """Context with budget_exceeded=True → BUDGET_EXHAUSTED"""
        exc = RuntimeError("stopped")
        result = classify_exception(exc, context={"budget_exceeded": True})
        assert result == FailureState.BUDGET_EXHAUSTED
    
    def test_skipped_by_budget_context_returns_skipped(self):
        """Context with skipped_by_budget=True → SKIPPED_BY_BUDGET"""
        exc = RuntimeError("not attempted")
        result = classify_exception(exc, context={"skipped_by_budget": True})
        assert result == FailureState.SKIPPED_BY_BUDGET
    
    def test_budget_skip_takes_precedence_over_exhausted(self):
        """skipped_by_budget takes precedence over budget_exhausted"""
        exc = RuntimeError("stopped")
        result = classify_exception(exc, context={
            "skipped_by_budget": True,
            "budget_exhausted": True,
        })
        assert result == FailureState.SKIPPED_BY_BUDGET
    
    def test_timeout_context_flag(self):
        """Context with timeout=True → TIMEOUT_ABSTAIN"""
        exc = RuntimeError("timed out")
        result = classify_exception(exc, context={"timeout": True})
        assert result == FailureState.TIMEOUT_ABSTAIN
    
    def test_unknown_exception_returns_unknown_error(self):
        """Unrecognized exceptions → UNKNOWN_ERROR"""
        exc = RuntimeError("weird error")
        result = classify_exception(exc)
        assert result == FailureState.UNKNOWN_ERROR
    
    def test_never_returns_success(self):
        """
        INVARIANT: classify_exception never returns SUCCESS.
        Success is determined by higher-level code, not exception classification.
        """
        # Try various exceptions that should all map to failure states
        exceptions = [
            RuntimeError("test"),
            ValueError("test"),
            Exception("test"),
            KeyError("test"),
            AttributeError("test"),
        ]
        for exc in exceptions:
            result = classify_exception(exc)
            assert result != FailureState.SUCCESS, f"Returned SUCCESS for {type(exc)}"


class TestClassifyFromStatus:
    """Tests for classify_from_status() function."""
    
    def test_success_status_returns_success(self):
        """status='success' → SUCCESS"""
        result = classify_from_status("success")
        assert result == FailureState.SUCCESS
    
    def test_aborted_status_returns_timeout_abstain(self):
        """status='aborted' → TIMEOUT_ABSTAIN"""
        result = classify_from_status("aborted")
        assert result == FailureState.TIMEOUT_ABSTAIN
    
    def test_failed_with_timeout_message_returns_timeout(self):
        """status='failed' with 'timeout' in message → TIMEOUT_ABSTAIN"""
        result = classify_from_status("failed", "Operation timed out after 3600s")
        assert result == FailureState.TIMEOUT_ABSTAIN
    
    def test_failed_with_crash_message_returns_crash(self):
        """status='failed' with crash keywords → CRASH_ABSTAIN"""
        test_cases = [
            "process crashed unexpectedly",
            "killed by signal 9",
            "memory allocation failed",
            "segfault detected",
            "returncode was 1",
        ]
        for msg in test_cases:
            result = classify_from_status("failed", msg)
            assert result == FailureState.CRASH_ABSTAIN, f"Failed for: {msg}"
    
    def test_failed_with_cli_failure_returns_crash(self):
        """status='failed' with CLI failure message → CRASH_ABSTAIN"""
        result = classify_from_status("failed", "Derive CLI failed with code 1")
        assert result == FailureState.CRASH_ABSTAIN
    
    def test_failed_with_parse_message_returns_invalid(self):
        """status='failed' with parse keywords → INVALID_FORMULA"""
        test_cases = [
            "syntax error in expression",
            "failed to parse input",
            "invalid formula structure",
            "normalize returned None",
        ]
        for msg in test_cases:
            result = classify_from_status("failed", msg)
            assert result == FailureState.INVALID_FORMULA, f"Failed for: {msg}"
    
    def test_failed_with_budget_skip_returns_skipped(self):
        """status='failed' with 'budget skip' → SKIPPED_BY_BUDGET"""
        result = classify_from_status("failed", "budget skip: max candidates reached")
        assert result == FailureState.SKIPPED_BY_BUDGET
    
    def test_failed_with_budget_returns_exhausted(self):
        """status='failed' with 'budget' (without skip) → BUDGET_EXHAUSTED"""
        result = classify_from_status("failed", "budget exceeded")
        assert result == FailureState.BUDGET_EXHAUSTED
    
    def test_failed_without_message_returns_unknown(self):
        """status='failed' without error message → UNKNOWN_ERROR"""
        result = classify_from_status("failed")
        assert result == FailureState.UNKNOWN_ERROR
    
    def test_failed_with_generic_message_returns_unknown(self):
        """status='failed' with generic message → UNKNOWN_ERROR"""
        result = classify_from_status("failed", "something went wrong")
        assert result == FailureState.UNKNOWN_ERROR
    
    def test_unknown_status_returns_unknown(self):
        """Unrecognized status → UNKNOWN_ERROR"""
        result = classify_from_status("weird_status")
        assert result == FailureState.UNKNOWN_ERROR


class TestClassifyFromResult:
    """Tests for classify_from_result() function."""
    
    def test_success_result_returns_success(self):
        """ExperimentResult with status='success' → SUCCESS"""
        result = MagicMock()
        result.status = "success"
        result.error_message = None
        
        state = classify_from_result(result)
        assert state == FailureState.SUCCESS
    
    def test_failed_result_uses_status_classification(self):
        """ExperimentResult with status='failed' delegates to classify_from_status"""
        result = MagicMock()
        result.status = "failed"
        result.error_message = "Derive CLI failed with code 1"
        
        state = classify_from_result(result)
        assert state == FailureState.CRASH_ABSTAIN
    
    def test_aborted_result_returns_timeout(self):
        """ExperimentResult with status='aborted' → TIMEOUT_ABSTAIN"""
        result = MagicMock()
        result.status = "aborted"
        result.error_message = "timed out"
        
        state = classify_from_result(result)
        assert state == FailureState.TIMEOUT_ABSTAIN


class TestNormalizeLegacyKey:
    """Tests for normalize_legacy_key() function."""
    
    def test_engine_failure_maps_to_crash_abstain(self):
        """Legacy 'engine_failure' → 'crash_abstain'"""
        result = normalize_legacy_key("engine_failure")
        assert result == FailureState.CRASH_ABSTAIN.value
    
    def test_timeout_maps_to_timeout_abstain(self):
        """Legacy 'timeout' → 'timeout_abstain'"""
        result = normalize_legacy_key("timeout")
        assert result == FailureState.TIMEOUT_ABSTAIN.value
    
    def test_unexpected_error_maps_to_unknown_error(self):
        """Legacy 'unexpected_error' → 'unknown_error'"""
        result = normalize_legacy_key("unexpected_error")
        assert result == FailureState.UNKNOWN_ERROR.value
    
    def test_lean_failure_maps_to_crash_abstain(self):
        """Legacy 'lean_failure' → 'crash_abstain'"""
        result = normalize_legacy_key("lean_failure")
        assert result == FailureState.CRASH_ABSTAIN.value
    
    def test_budget_skip_maps_to_skipped_by_budget(self):
        """Legacy 'budget_skip' → 'skipped_by_budget'"""
        result = normalize_legacy_key("budget_skip")
        assert result == FailureState.SKIPPED_BY_BUDGET.value
    
    def test_canonical_keys_pass_through(self):
        """Already-canonical keys pass through unchanged"""
        for state in FailureState:
            result = normalize_legacy_key(state.value)
            assert result == state.value
    
    def test_unknown_key_passes_through(self):
        """Unknown keys pass through with warning (no data loss)"""
        result = normalize_legacy_key("some_unknown_key")
        assert result == "some_unknown_key"
    
    def test_all_legacy_keys_are_mapped(self):
        """Verify all documented legacy keys have mappings"""
        expected_legacy_keys = {
            "engine_failure",
            "timeout",
            "unexpected_error",
            "pending_validation",
            "empty_run",
            "no_successful_proofs",
            "zero_throughput",
            "lean_failure",
            "derivation_abstain",
            "budget_skip",
            "cycle_budget_exhausted",
        }
        for key in expected_legacy_keys:
            assert key in LEGACY_KEY_MAP, f"Missing mapping for legacy key: {key}"


class TestClassificationDeterminism:
    """Tests ensuring deterministic, repeatable classification."""
    
    def test_same_exception_same_result(self):
        """Same exception should always produce same classification."""
        exc = subprocess.TimeoutExpired(cmd="test", timeout=60)
        
        results = [classify_exception(exc) for _ in range(100)]
        assert all(r == FailureState.TIMEOUT_ABSTAIN for r in results)
    
    def test_same_status_same_result(self):
        """Same status/message should always produce same classification."""
        results = [
            classify_from_status("failed", "Derive CLI failed with code 1")
            for _ in range(100)
        ]
        assert all(r == FailureState.CRASH_ABSTAIN for r in results)
    
    def test_classification_is_stateless(self):
        """Classification has no side effects or state."""
        # Classify various exceptions
        exc1 = subprocess.TimeoutExpired(cmd="test", timeout=60)
        exc2 = MemoryError("OOM")
        exc3 = SyntaxError("bad")
        
        r1_a = classify_exception(exc1)
        r2_a = classify_exception(exc2)
        r3_a = classify_exception(exc3)
        
        # Reclassify in different order
        r3_b = classify_exception(exc3)
        r1_b = classify_exception(exc1)
        r2_b = classify_exception(exc2)
        
        assert r1_a == r1_b == FailureState.TIMEOUT_ABSTAIN
        assert r2_a == r2_b == FailureState.CRASH_ABSTAIN
        assert r3_a == r3_b == FailureState.INVALID_FORMULA


class TestIntegrationWithExperimentResult:
    """Integration tests with ExperimentResult-like objects."""
    
    def test_classify_experiment_result_success(self):
        """Verify classification of successful experiment result."""
        result = MagicMock()
        result.status = "success"
        result.error_message = None
        
        state = classify_from_result(result)
        assert state == FailureState.SUCCESS
    
    def test_classify_experiment_result_timeout(self):
        """Verify classification of timed-out experiment result."""
        result = MagicMock()
        result.status = "aborted"
        result.error_message = "Derivation timed out after 1 hour"
        
        state = classify_from_result(result)
        assert state == FailureState.TIMEOUT_ABSTAIN
    
    def test_classify_experiment_result_crash(self):
        """Verify classification of crashed experiment result."""
        result = MagicMock()
        result.status = "failed"
        result.error_message = "Derive CLI failed with code 1: stderr output"
        
        state = classify_from_result(result)
        assert state == FailureState.CRASH_ABSTAIN

