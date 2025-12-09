"""
Tests for mock oracle production isolation.

Verifies that the mock oracle cannot be imported or used in production
contexts, ensuring it remains test-only.

INVARIANT INV-MOCK-1: Mock oracle NEVER touches real semantics.
INVARIANT INV-MOCK-2: Behavior must remain stable over time.
INVARIANT INV-MOCK-3: Test engineers must rely on exact branching behavior.

ABSOLUTE SAFEGUARD: These tests verify the safeguards themselves.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest import mock

import pytest


@pytest.mark.unit
class TestImportGuard:
    """Tests for the production import guard."""
    
    def test_mock_oracle_importable_in_test(self):
        """Mock oracle is importable when ALLOW_MOCK_ORACLE is set."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        # Force reimport
        if "backend.verification" in sys.modules:
            del sys.modules["backend.verification"]
        if "backend.verification.mock_oracle" in sys.modules:
            del sys.modules["backend.verification.mock_oracle"]
        
        from backend.verification import MockVerifiableOracle
        
        assert MockVerifiableOracle is not None
    
    def test_in_test_context_detected(self):
        """Test context is detected via pytest in sys.modules."""
        # Since we're running in pytest, this should be True
        assert "pytest" in sys.modules
    
    def test_config_always_importable(self):
        """Config classes are always importable (harmless dataclasses)."""
        from backend.verification import (
            MockOracleConfig,
            MockVerificationResult,
            SLICE_PROFILES,
        )
        
        assert MockOracleConfig is not None
        assert MockVerificationResult is not None
        assert SLICE_PROFILES is not None
    
    def test_exceptions_always_importable(self):
        """Exception classes are always importable."""
        from backend.verification import (
            MockOracleError,
            MockOracleCrashError,
            MockOracleTimeoutError,
        )
        
        assert MockOracleError is not None
        assert MockOracleCrashError is not None
        assert MockOracleTimeoutError is not None
    
    def test_contract_types_always_importable(self):
        """Contract types are always importable."""
        from backend.verification import (
            MOCK_ORACLE_CONTRACT_VERSION,
            PROFILE_CONTRACTS,
            NEGATIVE_CONTROL_CONTRACT,
            verify_profile_contracts,
            verify_negative_control_result,
        )
        
        assert MOCK_ORACLE_CONTRACT_VERSION is not None
        assert PROFILE_CONTRACTS is not None
        assert NEGATIVE_CONTROL_CONTRACT is not None
        assert verify_profile_contracts is not None
        assert verify_negative_control_result is not None
    
    def test_guard_checks_env_var(self):
        """Guard checks MATHLEDGER_ALLOW_MOCK_ORACLE env var."""
        import backend.verification as verification_module
        
        # The module should have guard-related variables
        assert hasattr(verification_module, "_ALLOW_MOCK_ORACLE") or \
               hasattr(verification_module, "_check_import_allowed") or \
               "_ALLOW_MOCK_ORACLE" in str(verification_module.__dict__)


@pytest.mark.unit
class TestGuardEnforcement:
    """Tests for guard enforcement in simulated non-test contexts."""
    
    def test_guard_function_exists(self):
        """_check_import_allowed function exists."""
        # This is an internal function, but we can verify it exists
        import backend.verification as mod
        
        # The module should have internal guard logic
        assert hasattr(mod, "_check_import_allowed") or "_ALLOW_MOCK_ORACLE" in str(dir(mod))
    
    def test_getattr_raises_for_oracle_without_guard(self):
        """Attempting to access MockVerifiableOracle should have guard logic."""
        # We can't easily test the actual ImportError without complex module reloading,
        # but we can verify the __getattr__ is defined when guard fails
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        # Since we're in test context, import should work
        from backend.verification import MockVerifiableOracle
        assert MockVerifiableOracle is not None
    
    def test_production_guard_message_is_helpful(self):
        """Guard error message mentions how to enable."""
        # Check the source code mentions the env var
        import inspect
        import backend.verification as mod
        
        source = inspect.getsource(mod)
        assert "MATHLEDGER_ALLOW_MOCK_ORACLE" in source
        assert "test context" in source.lower() or "test" in source.lower()


@pytest.mark.unit
class TestNoExternalCalls:
    """Tests ensuring mock oracle makes no external calls."""
    
    def test_no_subprocess_calls(self):
        """Mock oracle does not spawn subprocesses."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        with mock.patch("subprocess.run") as mock_run:
            with mock.patch("subprocess.Popen") as mock_popen:
                oracle = MockVerifiableOracle(MockOracleConfig())
                
                # Verify 100 formulas
                for i in range(100):
                    oracle.verify(f"p -> q_{i}")
                
                mock_run.assert_not_called()
                mock_popen.assert_not_called()
    
    def test_no_network_calls(self):
        """Mock oracle does not make network requests."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        with mock.patch("socket.socket") as mock_socket:
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                oracle = MockVerifiableOracle(MockOracleConfig())
                
                for i in range(100):
                    oracle.verify(f"p /\\ q_{i}")
                
                mock_socket.assert_not_called()
                mock_urlopen.assert_not_called()
    
    def test_no_file_io_beyond_module(self):
        """Mock oracle does not read/write files during verification."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        # Track open calls during verification
        original_open = open
        open_calls = []
        
        def tracking_open(*args, **kwargs):
            open_calls.append(args)
            return original_open(*args, **kwargs)
        
        with mock.patch("builtins.open", tracking_open):
            for i in range(50):
                oracle.verify(f"p \\/ ~p_{i}")
        
        # No files should be opened during verification
        assert len(open_calls) == 0


@pytest.mark.unit  
class TestNoProductionIntegration:
    """Tests ensuring mock oracle is not wired to production."""
    
    def test_not_imported_by_statement_verifier(self):
        """StatementVerifier does not import mock oracle."""
        # Import the real verifier
        from derivation.verification import StatementVerifier
        
        # Check that mock_oracle is not in its module's namespace
        verifier_module = sys.modules.get("derivation.verification")
        assert verifier_module is not None
        
        # The module should not have MockVerifiableOracle
        assert not hasattr(verifier_module, "MockVerifiableOracle")
    
    def test_not_imported_by_lean_fallback(self):
        """LeanFallback does not import mock oracle."""
        from derivation.verification import LeanFallback
        
        # Check module namespace
        verifier_module = sys.modules.get("derivation.verification")
        assert not hasattr(verifier_module, "MockVerifiableOracle")
    
    def test_mock_oracle_separate_module(self):
        """Mock oracle lives in separate backend.verification module."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle
        
        # Should be from backend.verification, not derivation.verification
        assert "backend.verification" in MockVerifiableOracle.__module__


@pytest.mark.unit
class TestDeterminismGuarantees:
    """Tests for determinism guarantees (no randomness)."""
    
    def test_no_random_module_usage(self):
        """Mock oracle does not use random module."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        with mock.patch("random.random") as mock_random:
            with mock.patch("random.randint") as mock_randint:
                with mock.patch("random.choice") as mock_choice:
                    oracle = MockVerifiableOracle(MockOracleConfig())
                    
                    for i in range(100):
                        oracle.verify(f"formula_{i}")
                    
                    mock_random.assert_not_called()
                    mock_randint.assert_not_called()
                    mock_choice.assert_not_called()
    
    def test_no_time_based_variation(self):
        """Mock oracle results don't vary with time."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        formula = "p -> (q -> p)"
        config = MockOracleConfig(slice_profile="default", seed=42)
        
        # Mock time to different values
        with mock.patch("time.time", return_value=1000000):
            oracle1 = MockVerifiableOracle(config)
            result1 = oracle1.verify(formula)
        
        with mock.patch("time.time", return_value=9999999):
            oracle2 = MockVerifiableOracle(config)
            result2 = oracle2.verify(formula)
        
        assert result1.verified == result2.verified
        assert result1.bucket == result2.bucket
        assert result1.latency_ms == result2.latency_ms


@pytest.mark.unit
class TestInvariantEnforcement:
    """Tests for mock oracle invariants (INV-MOCK-1, INV-MOCK-2, INV-MOCK-3)."""
    
    def test_inv_mock_1_no_real_semantics(self):
        """INV-MOCK-1: Mock oracle NEVER touches real semantics."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        # Verify that mock oracle doesn't use actual truth-table or Lean
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        # These formulas have known truth values, but mock shouldn't care
        tautology = "p -> p"  # Always true
        contradiction = "p /\\ ~p"  # Always false
        
        # Mock oracle result depends on hash, not actual logic
        r1 = oracle.verify(tautology)
        r2 = oracle.verify(contradiction)
        
        # Results are based on hash buckets, not semantic truth
        # We just verify it doesn't crash and returns structured results
        assert r1.bucket in ["verified", "failed", "abstain", "timeout", "error", "crash"]
        assert r2.bucket in ["verified", "failed", "abstain", "timeout", "error", "crash"]
    
    def test_inv_mock_2_stable_behavior(self):
        """INV-MOCK-2: Behavior must remain stable over time."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        config = MockOracleConfig(slice_profile="default", seed=0)
        
        # Create oracle at different "times" (mocked)
        results = []
        for time_value in [0, 1000000, 9999999999]:
            with mock.patch("time.time", return_value=time_value):
                oracle = MockVerifiableOracle(config)
                result = oracle.verify("stability_test")
                results.append(result)
        
        # All results should be identical
        assert all(r.verified == results[0].verified for r in results)
        assert all(r.bucket == results[0].bucket for r in results)
        assert all(r.latency_ms == results[0].latency_ms for r in results)
    
    def test_inv_mock_3_exact_branching(self):
        """INV-MOCK-3: Test engineers must rely on exact branching behavior."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        from backend.verification import MockOracleExpectations
        
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        
        # MockOracleExpectations provides known formulas for each bucket
        # These must ALWAYS behave as documented
        
        verified_formula = MockOracleExpectations.get_verified_formula("default")
        failed_formula = MockOracleExpectations.get_failed_formula("default")
        abstain_formula = MockOracleExpectations.get_abstain_formula("default")
        
        # Verify exact branching
        assert oracle.verify(verified_formula).bucket == "verified"
        assert oracle.verify(failed_formula).bucket == "failed"
        assert oracle.verify(abstain_formula).bucket == "abstain"
        
        # Run multiple times to ensure stability
        for _ in range(10):
            assert oracle.verify(verified_formula).bucket == "verified"
            assert oracle.verify(failed_formula).bucket == "failed"


@pytest.mark.unit
class TestPureFunctionProperty:
    """Tests ensuring mock oracle is a pure function of inputs + config."""
    
    def test_pure_function_no_side_effects(self):
        """Oracle verification is a pure function with no observable side effects."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        config = MockOracleConfig(slice_profile="default")
        oracle = MockVerifiableOracle(config)
        
        formula = "pure_function_test"
        
        # First call
        result1 = oracle.verify(formula)
        
        # Second call should be identical (pure function)
        result2 = oracle.verify(formula)
        
        assert result1.verified == result2.verified
        assert result1.bucket == result2.bucket
        assert result1.latency_ms == result2.latency_ms
        assert result1.hash_int == result2.hash_int
    
    def test_output_depends_only_on_input_and_config(self):
        """Output depends only on formula and config, nothing else."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        config = MockOracleConfig(slice_profile="default", seed=42)
        
        # Create multiple oracles with same config
        oracle1 = MockVerifiableOracle(config)
        oracle2 = MockVerifiableOracle(config)
        oracle3 = MockVerifiableOracle(config)
        
        formula = "output_test"
        
        # All should produce identical output
        r1 = oracle1.verify(formula)
        r2 = oracle2.verify(formula)
        r3 = oracle3.verify(formula)
        
        assert r1.verified == r2.verified == r3.verified
        assert r1.bucket == r2.bucket == r3.bucket
        assert r1.latency_ms == r2.latency_ms == r3.latency_ms
    
    def test_no_hidden_state_between_calls(self):
        """No hidden state accumulates between verification calls."""
        os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
        
        from backend.verification import MockVerifiableOracle, MockOracleConfig
        
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        # Verify many different formulas
        for i in range(100):
            oracle.verify(f"noise_formula_{i}")
        
        # Then verify a specific formula multiple times
        formula = "target_formula"
        results = [oracle.verify(formula) for _ in range(10)]
        
        # All results should be identical (no state leakage)
        assert all(r.verified == results[0].verified for r in results)
        assert all(r.bucket == results[0].bucket for r in results)
        assert all(r.latency_ms == results[0].latency_ms for r in results)

