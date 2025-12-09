"""
Tests for MockOracleExpectations library.

Verifies that the expectations library provides formulas that map to
their designated buckets, enabling reliable test assertions.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    MockOracleExpectations,
    SLICE_PROFILES,
)


@pytest.mark.unit
class TestVerifiedExpectations:
    """Tests for get_verified_formula."""
    
    def test_verified_formula_verifies(self):
        """get_verified_formula returns formula that verifies."""
        for profile in SLICE_PROFILES.keys():
            config = MockOracleConfig(slice_profile=profile)
            oracle = MockVerifiableOracle(config)
            
            formula = MockOracleExpectations.get_verified_formula(profile)
            result = oracle.verify(formula)
            
            assert result.verified is True, f"Profile {profile}: expected verified=True"
            assert result.bucket == "verified"
    
    def test_verified_formula_multiple_indices(self):
        """get_verified_formula supports multiple indices."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        
        formulas = [
            MockOracleExpectations.get_verified_formula("default", index=i)
            for i in range(5)
        ]
        
        # All should verify
        for formula in formulas:
            result = oracle.verify(formula)
            assert result.verified is True
        
        # All should be unique formulas
        assert len(set(formulas)) == len(formulas)


@pytest.mark.unit
class TestFailedExpectations:
    """Tests for get_failed_formula."""
    
    def test_failed_formula_fails(self):
        """get_failed_formula returns formula that fails."""
        for profile in SLICE_PROFILES.keys():
            config = MockOracleConfig(slice_profile=profile)
            oracle = MockVerifiableOracle(config)
            
            formula = MockOracleExpectations.get_failed_formula(profile)
            result = oracle.verify(formula)
            
            assert result.verified is False
            assert result.bucket == "failed"


@pytest.mark.unit
class TestAbstainExpectations:
    """Tests for get_abstain_formula."""
    
    def test_abstain_formula_abstains(self):
        """get_abstain_formula returns formula that abstains."""
        for profile in SLICE_PROFILES.keys():
            config = MockOracleConfig(slice_profile=profile)
            oracle = MockVerifiableOracle(config)
            
            formula = MockOracleExpectations.get_abstain_formula(profile)
            result = oracle.verify(formula)
            
            assert result.abstained is True
            assert result.bucket == "abstain"


@pytest.mark.unit
class TestTimeoutExpectations:
    """Tests for get_timeout_formula."""
    
    def test_timeout_formula_times_out(self):
        """get_timeout_formula returns formula that times out."""
        for profile in SLICE_PROFILES.keys():
            config = MockOracleConfig(slice_profile=profile)
            oracle = MockVerifiableOracle(config)
            
            formula = MockOracleExpectations.get_timeout_formula(profile)
            result = oracle.verify(formula)
            
            assert result.timed_out is True
            assert result.bucket == "timeout"


@pytest.mark.unit
class TestCrashExpectations:
    """Tests for get_crash_formula."""
    
    def test_crash_formula_crashes(self):
        """get_crash_formula returns formula that crashes."""
        for profile in SLICE_PROFILES.keys():
            config = MockOracleConfig(slice_profile=profile, enable_crashes=False)
            oracle = MockVerifiableOracle(config)
            
            formula = MockOracleExpectations.get_crash_formula(profile)
            result = oracle.verify(formula)
            
            assert result.crashed is True
            assert result.bucket == "crash"


@pytest.mark.unit
class TestSliceSpecificExpectations:
    """Tests for slice-specific helpers."""
    
    def test_goal_hit_set(self):
        """get_goal_hit_set returns formulas that verify under goal_hit."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="goal_hit"))
        
        formulas = MockOracleExpectations.get_goal_hit_set(count=5)
        
        assert len(formulas) == 5
        for formula in formulas:
            result = oracle.verify(formula)
            assert result.verified is True
    
    def test_chain_formulas(self):
        """get_chain_formulas returns formulas that verify under tree."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="tree"))
        
        formulas = MockOracleExpectations.get_chain_formulas(depth=3)
        
        assert len(formulas) == 3
        for formula in formulas:
            result = oracle.verify(formula)
            assert result.verified is True
    
    def test_dependency_goals(self):
        """get_dependency_goals returns formulas that verify under dependency."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="dependency"))
        
        formulas = MockOracleExpectations.get_dependency_goals(count=3)
        
        assert len(formulas) == 3
        for formula in formulas:
            result = oracle.verify(formula)
            assert result.verified is True


@pytest.mark.unit
class TestHelperMethods:
    """Tests for utility helper methods."""
    
    def test_get_formula_hash_mod(self):
        """get_formula_hash_mod returns hash mod 100."""
        mod = MockOracleExpectations.get_formula_hash_mod("p -> p")
        
        assert isinstance(mod, int)
        assert 0 <= mod < 100
    
    def test_get_all_profiles(self):
        """get_all_profiles returns all profile names."""
        profiles = MockOracleExpectations.get_all_profiles()
        
        assert "default" in profiles
        assert "goal_hit" in profiles
        assert "sparse" in profiles
        assert "tree" in profiles
        assert "dependency" in profiles
    
    def test_get_profile_boundaries(self):
        """get_profile_boundaries returns correct boundaries."""
        boundaries = MockOracleExpectations.get_profile_boundaries("default")
        
        assert boundaries["verified"] == 60
        assert boundaries["crash"] == 100
    
    def test_get_profile_boundaries_invalid_raises(self):
        """get_profile_boundaries raises for invalid profile."""
        with pytest.raises(ValueError):
            MockOracleExpectations.get_profile_boundaries("nonexistent")


@pytest.mark.unit
class TestCacheConsistency:
    """Tests for formula cache consistency."""
    
    def test_same_index_same_formula(self):
        """Same profile/bucket/index always returns same formula."""
        formula1 = MockOracleExpectations.get_verified_formula("default", index=0)
        formula2 = MockOracleExpectations.get_verified_formula("default", index=0)
        formula3 = MockOracleExpectations.get_verified_formula("default", index=0)
        
        assert formula1 == formula2 == formula3
    
    def test_different_indices_different_formulas(self):
        """Different indices return different formulas."""
        formulas = [
            MockOracleExpectations.get_verified_formula("default", index=i)
            for i in range(10)
        ]
        
        assert len(set(formulas)) == 10  # All unique

