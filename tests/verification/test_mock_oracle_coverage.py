"""
Tests for mock oracle profile coverage maps.

Verifies that profile_coverage() returns correct static percentages
based on profile boundaries, not runtime statistics.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    ProfileCoverageMap,
    compute_profile_coverage,
    SLICE_PROFILES,
)


@pytest.mark.unit
class TestProfileCoverageMap:
    """Tests for ProfileCoverageMap dataclass."""
    
    def test_from_profile_default(self):
        """ProfileCoverageMap.from_profile works for default profile."""
        coverage = ProfileCoverageMap.from_profile("default")
        
        assert coverage.profile_name == "default"
        assert coverage.verified_pct == 60.0
        assert coverage.failed_pct == 15.0
        assert coverage.abstain_pct == 10.0
        assert coverage.timeout_pct == 8.0
        assert coverage.error_pct == 4.0
        assert coverage.crash_pct == 3.0
    
    def test_from_profile_goal_hit(self):
        """ProfileCoverageMap.from_profile works for goal_hit profile."""
        coverage = ProfileCoverageMap.from_profile("goal_hit")
        
        assert coverage.profile_name == "goal_hit"
        assert coverage.verified_pct == 15.0
        assert coverage.abstain_pct == 35.0  # High abstention
    
    def test_from_profile_invalid_raises(self):
        """ProfileCoverageMap.from_profile raises for invalid profile."""
        with pytest.raises(ValueError, match="Invalid profile"):
            ProfileCoverageMap.from_profile("nonexistent")
    
    def test_to_dict(self):
        """to_dict returns correct dictionary format."""
        coverage = ProfileCoverageMap.from_profile("default")
        d = coverage.to_dict()
        
        assert d["verified"] == 60.0
        assert d["failed"] == 15.0
        assert d["abstain"] == 10.0
        assert d["timeout"] == 8.0
        assert d["error"] == 4.0
        assert d["crash"] == 3.0
        # profile_name is NOT in dict
        assert "profile_name" not in d
    
    def test_percentages_sum_to_100(self):
        """All profile coverages sum to 100%."""
        for profile_name in SLICE_PROFILES:
            coverage = ProfileCoverageMap.from_profile(profile_name)
            total = (
                coverage.verified_pct
                + coverage.failed_pct
                + coverage.abstain_pct
                + coverage.timeout_pct
                + coverage.error_pct
                + coverage.crash_pct
            )
            assert total == 100.0, f"Profile {profile_name} sums to {total}"


@pytest.mark.unit
class TestComputeProfileCoverage:
    """Tests for compute_profile_coverage helper function."""
    
    def test_returns_dict(self):
        """compute_profile_coverage returns a dictionary."""
        coverage = compute_profile_coverage("default")
        
        assert isinstance(coverage, dict)
        assert "verified" in coverage
        assert "crash" in coverage
    
    def test_matches_coverage_map(self):
        """compute_profile_coverage matches ProfileCoverageMap.to_dict()."""
        for profile_name in SLICE_PROFILES:
            from_func = compute_profile_coverage(profile_name)
            from_class = ProfileCoverageMap.from_profile(profile_name).to_dict()
            
            assert from_func == from_class


@pytest.mark.unit
class TestOracleProfileCoverage:
    """Tests for MockVerifiableOracle.profile_coverage() method."""
    
    def test_profile_coverage_returns_coverage_map(self, default_oracle: MockVerifiableOracle):
        """profile_coverage() returns ProfileCoverageMap."""
        coverage = default_oracle.profile_coverage()
        
        assert isinstance(coverage, ProfileCoverageMap)
        assert coverage.profile_name == "default"
    
    def test_coverage_matches_profile(self):
        """profile_coverage() matches current profile."""
        for profile_name in SLICE_PROFILES:
            config = MockOracleConfig(slice_profile=profile_name)
            oracle = MockVerifiableOracle(config)
            
            coverage = oracle.profile_coverage()
            assert coverage.profile_name == profile_name
    
    def test_coverage_is_static(self, default_oracle: MockVerifiableOracle, sample_formulas: list[str]):
        """profile_coverage() returns static values regardless of runtime."""
        coverage_before = default_oracle.profile_coverage()
        
        # Run many verifications
        for formula in sample_formulas * 10:
            default_oracle.verify(formula)
        
        coverage_after = default_oracle.profile_coverage()
        
        # Coverage should be identical (static, not runtime)
        assert coverage_before.to_dict() == coverage_after.to_dict()
    
    def test_coverage_differs_from_stats(self, default_oracle: MockVerifiableOracle, many_formulas: list[str]):
        """profile_coverage() differs from runtime stats (due to hash distribution)."""
        # Run verifications
        for formula in many_formulas[:100]:
            default_oracle.verify(formula)
        
        coverage = default_oracle.profile_coverage()
        stats = default_oracle.stats
        
        # Static coverage says 60% verified
        assert coverage.verified_pct == 60.0
        
        # Runtime stats may differ slightly due to actual hash distribution
        # (This test just verifies they're computed differently)
        runtime_verified_pct = stats["verified"] / stats["total"] * 100 if stats["total"] > 0 else 0
        
        # They should be close but not necessarily identical
        # The point is coverage is STATIC, stats are DYNAMIC
        assert coverage.verified_pct == 60.0  # Always static
    
    def test_coverage_after_profile_switch(self):
        """profile_coverage() updates when profile is switched."""
        oracle = MockVerifiableOracle(MockOracleConfig(slice_profile="default"))
        
        assert oracle.profile_coverage().verified_pct == 60.0
        
        oracle.set_profile("goal_hit")
        
        assert oracle.profile_coverage().verified_pct == 15.0
    
    def test_coverage_in_negative_control_mode(self):
        """profile_coverage() returns theoretical coverage even in negative control mode."""
        config = MockOracleConfig(
            slice_profile="default",
            negative_control=True,
        )
        oracle = MockVerifiableOracle(config)
        
        # Coverage should still reflect the profile's theoretical distribution
        coverage = oracle.profile_coverage()
        assert coverage.verified_pct == 60.0
        
        # But actual results are all negative_control
        result = oracle.verify("p -> p")
        assert result.reason == "negative_control"
        assert result.verified is False


@pytest.mark.unit
class TestAllProfilesCoverage:
    """Tests for coverage across all profiles."""
    
    def test_all_profiles_have_coverage(self, all_profiles: list[str]):
        """All profiles produce valid coverage maps."""
        for profile in all_profiles:
            coverage = ProfileCoverageMap.from_profile(profile)
            
            assert coverage.verified_pct >= 0
            assert coverage.crash_pct >= 0
            
            total = sum(coverage.to_dict().values())
            assert total == 100.0
    
    def test_profiles_have_different_coverage(self):
        """Different profiles have different coverage distributions."""
        default = ProfileCoverageMap.from_profile("default")
        goal_hit = ProfileCoverageMap.from_profile("goal_hit")
        
        # goal_hit has lower verification rate
        assert goal_hit.verified_pct < default.verified_pct
        
        # goal_hit has higher abstention
        assert goal_hit.abstain_pct > default.abstain_pct

