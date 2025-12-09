"""
Tests for mock oracle slice profiles.

Verifies that each profile (default, goal_hit, sparse, tree, dependency)
produces the expected distribution of verification outcomes.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    SLICE_PROFILES,
)


def get_bucket_distribution(oracle: MockVerifiableOracle, formulas: list[str]) -> dict[str, float]:
    """Get empirical bucket distribution over formulas."""
    counter = Counter()
    
    for formula in formulas:
        result = oracle.verify(formula)
        counter[result.bucket] += 1
    
    total = len(formulas)
    return {bucket: count / total * 100 for bucket, count in counter.items()}


@pytest.mark.unit
class TestDefaultProfile:
    """Tests for default profile bucket distribution."""
    
    def test_default_profile_exists(self):
        """Default profile is defined in SLICE_PROFILES."""
        assert "default" in SLICE_PROFILES
    
    def test_default_profile_boundaries(self):
        """Default profile has correct bucket boundaries."""
        profile = SLICE_PROFILES["default"]
        
        assert profile["verified"] == 60   # 60%
        assert profile["failed"] == 75     # 15%
        assert profile["abstain"] == 85    # 10%
        assert profile["timeout"] == 93    # 8%
        assert profile["error"] == 97      # 4%
        assert profile["crash"] == 100     # 3%
    
    def test_default_distribution_approximate(self, many_formulas: list[str]):
        """Default profile produces approximately expected distribution."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        # Allow ±5% tolerance due to hash distribution
        assert 55 <= dist.get("verified", 0) <= 65, f"verified: {dist.get('verified', 0)}"
        assert 10 <= dist.get("failed", 0) <= 20, f"failed: {dist.get('failed', 0)}"


@pytest.mark.unit
class TestGoalHitProfile:
    """Tests for goal_hit profile — rare successes."""
    
    def test_goal_hit_profile_exists(self):
        """goal_hit profile is defined."""
        assert "goal_hit" in SLICE_PROFILES
    
    def test_goal_hit_low_verification_rate(self, many_formulas: list[str]):
        """goal_hit profile has low verification rate (~15%)."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="goal_hit"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        # Should have low verification rate
        assert dist.get("verified", 0) < 25, f"verified too high: {dist.get('verified', 0)}"
    
    def test_goal_hit_high_abstention(self, many_formulas: list[str]):
        """goal_hit profile has high abstention rate."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="goal_hit"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        # Should have high abstention
        assert dist.get("abstain", 0) > 25, f"abstain too low: {dist.get('abstain', 0)}"
    
    def test_goal_hit_boundaries(self):
        """goal_hit profile has correct boundaries."""
        profile = SLICE_PROFILES["goal_hit"]
        
        assert profile["verified"] == 15   # 15% — rare hits
        assert profile["abstain"] == 85    # 35% abstention


@pytest.mark.unit
class TestSparseProfile:
    """Tests for sparse profile — wide proof space."""
    
    def test_sparse_profile_exists(self):
        """sparse profile is defined."""
        assert "sparse" in SLICE_PROFILES
    
    def test_sparse_moderate_verification(self, many_formulas: list[str]):
        """sparse profile has moderate verification (~25%)."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="sparse"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        assert 15 <= dist.get("verified", 0) <= 35
    
    def test_sparse_wide_abstention_zone(self, many_formulas: list[str]):
        """sparse profile has wide abstention zone."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="sparse"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        assert dist.get("abstain", 0) > 20


@pytest.mark.unit
class TestTreeProfile:
    """Tests for tree profile — chain-depth patterns."""
    
    def test_tree_profile_exists(self):
        """tree profile is defined."""
        assert "tree" in SLICE_PROFILES
    
    def test_tree_medium_verification(self, many_formulas: list[str]):
        """tree profile has medium verification (~45%) for chain building."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="tree"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        assert 35 <= dist.get("verified", 0) <= 55
    
    def test_tree_boundaries(self):
        """tree profile boundaries support chain building."""
        profile = SLICE_PROFILES["tree"]
        
        assert profile["verified"] == 45  # Medium for chains


@pytest.mark.unit
class TestDependencyProfile:
    """Tests for dependency profile — multi-goal coordination."""
    
    def test_dependency_profile_exists(self):
        """dependency profile is defined."""
        assert "dependency" in SLICE_PROFILES
    
    def test_dependency_moderate_verification(self, many_formulas: list[str]):
        """dependency profile has moderate verification (~35%)."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="dependency"))
        dist = get_bucket_distribution(oracle, many_formulas)
        
        assert 25 <= dist.get("verified", 0) <= 45
    
    def test_dependency_boundaries(self):
        """dependency profile has correct boundaries."""
        profile = SLICE_PROFILES["dependency"]
        
        assert profile["verified"] == 35
        assert profile["abstain"] == 85


@pytest.mark.unit
class TestProfileSwitching:
    """Tests for dynamic profile switching."""
    
    def test_set_profile_changes_behavior(self):
        """set_profile changes oracle behavior."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        
        formula = "test_formula_42"
        result_default = oracle.verify(formula)
        
        oracle.set_profile("goal_hit")
        result_goal_hit = oracle.verify(formula)
        
        # Both results are valid, profile may or may not change bucket
        assert result_default.bucket in ["verified", "failed", "abstain", "timeout", "error", "crash"]
        assert result_goal_hit.bucket in ["verified", "failed", "abstain", "timeout", "error", "crash"]
    
    def test_invalid_profile_raises(self):
        """Setting invalid profile raises ValueError."""
        oracle = MockVerifiableOracle()
        
        with pytest.raises(ValueError, match="Invalid profile"):
            oracle.set_profile("nonexistent_profile")
    
    def test_profile_config_updated(self):
        """set_profile updates internal config."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        assert oracle.config.slice_profile == "default"
        
        oracle.set_profile("sparse")
        assert oracle.config.slice_profile == "sparse"


@pytest.mark.unit
class TestAllProfilesCoverage:
    """Tests ensuring all profiles are covered."""
    
    def test_all_profiles_instantiable(self, all_profiles: list[str]):
        """All profiles can be used to create oracles."""
        for profile in all_profiles:
            config = MockOracleConfig(slice_profile=profile)
            oracle = MockVerifiableOracle(config)
            
            # Should be able to verify a formula
            result = oracle.verify("p -> p")
            assert result.bucket in ["verified", "failed", "abstain", "timeout", "error", "crash"]
    
    def test_profiles_have_all_buckets(self, all_profiles: list[str]):
        """All profiles define all bucket boundaries."""
        required_buckets = {"verified", "failed", "abstain", "timeout", "error", "crash"}
        
        for profile in all_profiles:
            boundaries = SLICE_PROFILES[profile]
            assert set(boundaries.keys()) == required_buckets
    
    def test_bucket_distribution_display(self, default_oracle: MockVerifiableOracle):
        """get_bucket_distribution returns readable format."""
        dist = default_oracle.get_bucket_distribution()
        
        assert "verified" in dist
        assert "crash" in dist
        # Should contain percentage info
        assert "%" in dist["verified"]

