"""
Tests for mock oracle contract enforcement.

These tests verify that the mock oracle adheres to its contractual guarantees:
1. Profile distributions match PROFILE_CONTRACTS exactly
2. Negative control contract holds for all inputs
3. Determinism contract holds across runs

These tests are designed for CI non-regression checking.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import hashlib
import os

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MOCK_ORACLE_CONTRACT_VERSION,
    PROFILE_CONTRACTS,
    NEGATIVE_CONTROL_CONTRACT,
    SLICE_PROFILES,
    MockOracleConfig,
    MockVerifiableOracle,
    ProfileCoverageMap,
    verify_profile_contracts,
    verify_negative_control_result,
    compute_profile_coverage,
)


# ============================================================================
# CONTRACT VERSION TESTS
# ============================================================================

@pytest.mark.unit
class TestContractVersion:
    """Tests for contract version tracking."""
    
    def test_contract_version_exists(self):
        """Contract version is defined."""
        assert MOCK_ORACLE_CONTRACT_VERSION is not None
        assert isinstance(MOCK_ORACLE_CONTRACT_VERSION, str)
    
    def test_contract_version_format(self):
        """Contract version follows semver format."""
        parts = MOCK_ORACLE_CONTRACT_VERSION.split(".")
        assert len(parts) == 3, "Version should be MAJOR.MINOR.PATCH"
        
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"
    
    def test_contract_version_is_1_0_0(self):
        """Current contract version is 1.0.0."""
        assert MOCK_ORACLE_CONTRACT_VERSION == "1.0.0"


# ============================================================================
# PROFILE CONTRACT TESTS
# ============================================================================

@pytest.mark.unit
class TestProfileContracts:
    """Tests for profile distribution contracts."""
    
    def test_profile_contracts_defined(self):
        """All profiles have contract definitions."""
        for profile in SLICE_PROFILES:
            assert profile in PROFILE_CONTRACTS, f"Missing contract for {profile}"
    
    def test_no_extra_profiles(self):
        """No profiles in SLICE_PROFILES without contracts."""
        for profile in SLICE_PROFILES:
            assert profile in PROFILE_CONTRACTS
    
    def test_no_extra_contracts(self):
        """No contracts without matching profiles."""
        for profile in PROFILE_CONTRACTS:
            assert profile in SLICE_PROFILES
    
    def test_all_contracts_sum_to_100(self):
        """Each contract sums to exactly 100%."""
        for profile, contract in PROFILE_CONTRACTS.items():
            total = sum(contract.values())
            assert total == 100.0, f"Profile '{profile}' sums to {total}, expected 100.0"
    
    def test_verify_profile_contracts_passes(self):
        """verify_profile_contracts() returns success."""
        is_valid, errors = verify_profile_contracts()
        assert is_valid, f"Contract violations: {errors}"
        assert len(errors) == 0


@pytest.mark.unit
class TestProfileContractsEpsilon:
    """Epsilon-based contract verification tests."""
    
    EPSILON = 0.001  # Tolerance for floating-point comparison
    
    def test_default_profile_contract(self):
        """Default profile matches contract within epsilon."""
        coverage = compute_profile_coverage("default")
        contract = PROFILE_CONTRACTS["default"]
        
        for bucket, expected in contract.items():
            actual = coverage[bucket]
            assert abs(actual - expected) < self.EPSILON, \
                f"default.{bucket}: expected {expected}, got {actual}"
    
    def test_goal_hit_profile_contract(self):
        """goal_hit profile matches contract within epsilon."""
        coverage = compute_profile_coverage("goal_hit")
        contract = PROFILE_CONTRACTS["goal_hit"]
        
        for bucket, expected in contract.items():
            actual = coverage[bucket]
            assert abs(actual - expected) < self.EPSILON, \
                f"goal_hit.{bucket}: expected {expected}, got {actual}"
    
    def test_sparse_profile_contract(self):
        """sparse profile matches contract within epsilon."""
        coverage = compute_profile_coverage("sparse")
        contract = PROFILE_CONTRACTS["sparse"]
        
        for bucket, expected in contract.items():
            actual = coverage[bucket]
            assert abs(actual - expected) < self.EPSILON, \
                f"sparse.{bucket}: expected {expected}, got {actual}"
    
    def test_tree_profile_contract(self):
        """tree profile matches contract within epsilon."""
        coverage = compute_profile_coverage("tree")
        contract = PROFILE_CONTRACTS["tree"]
        
        for bucket, expected in contract.items():
            actual = coverage[bucket]
            assert abs(actual - expected) < self.EPSILON, \
                f"tree.{bucket}: expected {expected}, got {actual}"
    
    def test_dependency_profile_contract(self):
        """dependency profile matches contract within epsilon."""
        coverage = compute_profile_coverage("dependency")
        contract = PROFILE_CONTRACTS["dependency"]
        
        for bucket, expected in contract.items():
            actual = coverage[bucket]
            assert abs(actual - expected) < self.EPSILON, \
                f"dependency.{bucket}: expected {expected}, got {actual}"
    
    def test_all_profiles_parametric(self):
        """All profiles match contracts (parametric test)."""
        for profile_name in PROFILE_CONTRACTS:
            coverage = compute_profile_coverage(profile_name)
            contract = PROFILE_CONTRACTS[profile_name]
            
            for bucket, expected in contract.items():
                actual = coverage[bucket]
                assert abs(actual - expected) < self.EPSILON, \
                    f"{profile_name}.{bucket}: expected {expected}, got {actual}"


# ============================================================================
# NEGATIVE CONTROL CONTRACT TESTS (100+ samples)
# ============================================================================

@pytest.mark.unit
class TestNegativeControlContract:
    """Tests for negative control contract enforcement."""
    
    def test_negative_control_contract_defined(self):
        """Negative control contract is defined."""
        assert NEGATIVE_CONTROL_CONTRACT is not None
        assert NEGATIVE_CONTROL_CONTRACT.verified is False
        assert NEGATIVE_CONTROL_CONTRACT.abstained is True
        assert NEGATIVE_CONTROL_CONTRACT.reason == "negative_control"
    
    def test_verify_negative_control_result_function(self):
        """verify_negative_control_result detects violations."""
        from backend.verification.mock_config import MockVerificationResult
        
        # Valid NC result
        valid = MockVerificationResult(
            verified=False,
            abstained=True,
            timed_out=False,
            crashed=False,
            reason="negative_control",
            latency_ms=10,
            bucket="negative_control",
        )
        is_valid, violations = verify_negative_control_result(valid)
        assert is_valid
        assert len(violations) == 0
        
        # Invalid NC result (verified=True)
        invalid = MockVerificationResult(
            verified=True,  # VIOLATION
            abstained=False,
            timed_out=False,
            crashed=False,
            reason="mock-verified",
            latency_ms=10,
            bucket="verified",
        )
        is_valid, violations = verify_negative_control_result(invalid)
        assert not is_valid
        assert len(violations) > 0


@pytest.mark.unit
class TestNegativeControlStress:
    """Stress tests for negative control mode with 100+ inputs."""
    
    NUM_SAMPLES = 200  # More than 100 as specified
    
    def test_nc_verified_always_false(self):
        """verified is False for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.verified is False, f"Sample {i}: verified should be False"
    
    def test_nc_abstained_always_true(self):
        """abstained is True for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.abstained is True, f"Sample {i}: abstained should be True"
    
    def test_nc_timed_out_always_false(self):
        """timed_out is False for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.timed_out is False, f"Sample {i}: timed_out should be False"
    
    def test_nc_crashed_always_false(self):
        """crashed is False for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.crashed is False, f"Sample {i}: crashed should be False"
    
    def test_nc_reason_always_negative_control(self):
        """reason is 'negative_control' for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.reason == "negative_control", f"Sample {i}: wrong reason"
    
    def test_nc_bucket_always_negative_control(self):
        """bucket is 'negative_control' for all 200 samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_stress_test_{i}"
            result = oracle.verify(formula)
            assert result.bucket == "negative_control", f"Sample {i}: wrong bucket"
    
    def test_nc_never_raises(self):
        """No exceptions raised for 200 samples, even with crashes enabled."""
        config = MockOracleConfig(negative_control=True, enable_crashes=True)
        oracle = MockVerifiableOracle(config)
        
        exceptions_raised = 0
        for i in range(self.NUM_SAMPLES):
            formula = f"nc_crash_test_{i}"
            try:
                oracle.verify(formula)
            except Exception:
                exceptions_raised += 1
        
        assert exceptions_raised == 0, f"NC mode raised {exceptions_raised} exceptions"
    
    def test_nc_stats_suppressed(self):
        """Stats remain zero after 200 verifications."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        for i in range(self.NUM_SAMPLES):
            oracle.verify(f"nc_stats_test_{i}")
        
        assert oracle.stats["total"] == 0, f"Stats not suppressed: {oracle.stats}"
        assert oracle.stats["verified"] == 0
        assert oracle.stats["abstain"] == 0
    
    def test_nc_diverse_formulas(self):
        """NC contract holds for diverse formula patterns."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        formulas = [
            # Simple
            "p -> p",
            "q -> q",
            # Complex
            "(p -> q) -> (~q -> ~p)",
            "((p -> q) /\\ (q -> r)) -> (p -> r)",
            # Edge cases
            "",
            " ",
            "~~~~~p",
            "p" * 1000,
            # Unicode (should still work)
            "α → β",
            # Special characters
            "p->q->r->s->t",
        ]
        
        for formula in formulas:
            result = oracle.verify(formula)
            is_valid, violations = verify_negative_control_result(result)
            assert is_valid, f"Formula '{formula[:20]}...' violated NC: {violations}"
    
    def test_nc_full_contract_verification_200_samples(self):
        """Full contract verification for 200 diverse samples."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        violations_found = []
        
        for i in range(self.NUM_SAMPLES):
            # Generate diverse formulas
            formula = f"full_contract_test_{i}_{'x' * (i % 20)}"
            result = oracle.verify(formula)
            
            is_valid, violations = verify_negative_control_result(result)
            if not is_valid:
                violations_found.append((i, formula, violations))
        
        assert len(violations_found) == 0, \
            f"Found {len(violations_found)} violations: {violations_found[:5]}"


# ============================================================================
# DETERMINISM CONTRACT TESTS
# ============================================================================

@pytest.mark.unit
class TestDeterminismContract:
    """Tests for determinism contract enforcement."""
    
    def test_same_input_same_hash(self):
        """Same input produces same hash."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        formula = "(p -> q) -> (~q -> ~p)"
        
        h1 = oracle._hash_formula(formula)
        h2 = oracle._hash_formula(formula)
        h3 = oracle._hash_formula(formula)
        
        assert h1 == h2 == h3
    
    def test_hash_is_sha256(self):
        """Hash uses SHA-256 algorithm."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        formula = "p -> p"
        
        # Compute expected SHA-256
        expected = int(hashlib.sha256(formula.encode("utf-8")).hexdigest(), 16)
        actual = oracle._hash_formula(formula)
        
        assert actual == expected
    
    def test_bucket_from_hash_mod_100(self):
        """Bucket selection uses hash % 100."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        
        formula = "p -> p"
        h = oracle._hash_formula(formula)
        
        # Default profile: 0-59 = verified
        mod = h % 100
        expected_bucket = "verified" if mod < 60 else (
            "failed" if mod < 75 else (
            "abstain" if mod < 85 else (
            "timeout" if mod < 93 else (
            "error" if mod < 97 else "crash"
        ))))
        
        actual_bucket = oracle._bucket_for_hash(h)
        assert actual_bucket == expected_bucket
    
    def test_determinism_across_instances(self):
        """Same formula produces identical results across instances."""
        config = MockOracleConfig(slice_profile="default", seed=42)
        
        oracle1 = MockVerifiableOracle(config)
        oracle2 = MockVerifiableOracle(config)
        
        for i in range(100):
            formula = f"determinism_test_{i}"
            
            r1 = oracle1.verify(formula)
            r2 = oracle2.verify(formula)
            
            assert r1.verified == r2.verified
            assert r1.bucket == r2.bucket
            assert r1.latency_ms == r2.latency_ms
            assert r1.reason == r2.reason
            assert r1.hash_int == r2.hash_int
    
    def test_no_randomness_in_100_runs(self):
        """100 runs of same formula produce identical results."""
        config = MockOracleConfig(slice_profile="default")
        oracle = MockVerifiableOracle(config)
        
        formula = "p /\\ q -> p"
        
        first_result = oracle.verify(formula)
        
        for _ in range(100):
            result = oracle.verify(formula)
            assert result.verified == first_result.verified
            assert result.bucket == first_result.bucket
            assert result.latency_ms == first_result.latency_ms


# ============================================================================
# CLI CONTRACT TESTS
# ============================================================================

@pytest.mark.unit
class TestCLIContractMode:
    """Tests for CLI --ci mode."""
    
    def test_ci_mode_passes(self):
        """--ci mode exits with 0 when contracts pass."""
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--ci", "--json"])
        
        assert result == 0
    
    def test_ci_mode_json_output(self):
        """--ci --json produces valid JSON."""
        import json
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            main(["--ci", "--json"])
        
        output = json.loads(stdout.getvalue())
        
        assert "contract_version" in output
        assert "checks" in output
        assert "passed" in output
        assert output["passed"] is True
    
    def test_ci_mode_checks_profiles(self):
        """--ci mode verifies profile contracts."""
        import json
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            main(["--ci", "--json"])
        
        output = json.loads(stdout.getvalue())
        
        assert "profile_contracts" in output["checks"]
        assert output["checks"]["profile_contracts"]["passed"] is True
    
    def test_ci_mode_checks_negative_control(self):
        """--ci mode verifies negative control contract."""
        import json
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            main(["--ci", "--json"])
        
        output = json.loads(stdout.getvalue())
        
        assert "negative_control" in output["checks"]
        assert output["checks"]["negative_control"]["passed"] is True
        assert output["checks"]["negative_control"]["samples_tested"] >= 100
    
    def test_ci_mode_checks_determinism(self):
        """--ci mode verifies determinism contract."""
        import json
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            main(["--ci", "--json"])
        
        output = json.loads(stdout.getvalue())
        
        assert "determinism" in output["checks"]
        assert output["checks"]["determinism"]["passed"] is True
    
    def test_version_flag(self):
        """--version shows contract version."""
        import json
        from backend.verification.mock_oracle_cli import main
        from io import StringIO
        from unittest import mock
        
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--version", "--json"])
        
        assert result == 0
        output = json.loads(stdout.getvalue())
        assert output["contract_version"] == MOCK_ORACLE_CONTRACT_VERSION

