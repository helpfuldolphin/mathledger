"""
Tests for Phase IV — Hardness-Aware Workload Shaping & Slice Policy Feeds.

TASK 1: Per-Slice Hardness Distribution
TASK 2: Workload Shaping Recommendation
TASK 3: Curriculum & Global Health Hooks

Agent B2 - Truth-Table Oracle & CHI Engineer
"""

import json
import pytest
from typing import Dict, Any, List


# =============================================================================
# TASK 1: PER-SLICE HARDNESS DISTRIBUTION TESTS
# =============================================================================

class TestSliceHardnessProfile:
    """
    Tests for build_slice_hardness_profile().
    
    Verify:
    - Profile schema is stable
    - Category and risk band counts are correct
    - Percentile calculations are accurate
    - Empty input handling
    """

    def test_profile_schema_version(self):
        """Profile must include schema version for compatibility."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            CHIResult,
            SLICE_PROFILE_SCHEMA_VERSION,
        )
        
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0),
        ]
        
        profile = build_slice_hardness_profile(results, "test_slice")
        
        assert "schema_version" in profile
        assert profile["schema_version"] == SLICE_PROFILE_SCHEMA_VERSION
        assert profile["schema_version"] == "1.0.0"

    def test_profile_contains_all_required_fields(self):
        """Profile must contain all specified fields."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        results = [
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        profile = build_slice_hardness_profile(results, "unit_1")
        
        required_fields = {
            "schema_version",
            "slice_name",
            "counts_by_category",
            "risk_band_counts",
            "median_chi",
            "p90_chi",
            "max_chi",
            "total_count",
        }
        
        assert set(profile.keys()) == required_fields

    def test_empty_results_returns_zero_counts(self):
        """Empty results should return zero counts."""
        from normalization.tt_chi import build_slice_hardness_profile
        
        profile = build_slice_hardness_profile([], "empty_slice")
        
        assert profile["total_count"] == 0
        assert profile["median_chi"] == 0.0
        assert profile["p90_chi"] == 0.0
        assert profile["max_chi"] == 0.0
        assert sum(profile["counts_by_category"].values()) == 0
        assert sum(profile["risk_band_counts"].values()) == 0

    def test_category_counts(self):
        """Category counts should match input results."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        results = [
            CHIResult(chi=1.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=1000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=500.0),  # trivial
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),  # easy
            CHIResult(chi=12.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=12000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1500.0),  # moderate
            CHIResult(chi=20.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=20000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),  # hard
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5),  # extreme
        ]
        
        profile = build_slice_hardness_profile(results, "mixed_slice")
        
        counts = profile["counts_by_category"]
        assert counts["trivial"] == 1
        assert counts["easy"] == 1
        assert counts["moderate"] == 1
        assert counts["hard"] == 1
        assert counts["extreme"] == 1
        assert profile["total_count"] == 5

    def test_risk_band_counts(self):
        """Risk band counts should match category mappings."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0),  # trivial → LOW
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),  # easy → LOW
            CHIResult(chi=12.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=12000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1500.0),  # moderate → MEDIUM
            CHIResult(chi=20.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=20000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),  # hard → HIGH
        ]
        
        profile = build_slice_hardness_profile(results, "risk_bands")
        
        risk_counts = profile["risk_band_counts"]
        assert risk_counts["LOW"] == 2
        assert risk_counts["MEDIUM"] == 1
        assert risk_counts["HIGH"] == 1
        assert risk_counts["EXTREME"] == 0

    def test_percentile_calculations(self):
        """Percentile calculations should be accurate."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        # Create 10 results with known CHI values
        chi_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        results = [
            CHIResult(chi=chi, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=int(chi * 1000),
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for chi in chi_values
        ]
        
        profile = build_slice_hardness_profile(results, "percentiles")
        
        # Median should be average of 5th and 6th (5.5)
        assert profile["median_chi"] == pytest.approx(5.5, rel=0.01)
        
        # p90 should be 9th value (9.0) for 10 items
        assert profile["p90_chi"] == pytest.approx(9.0, rel=0.01)
        
        # Max should be 10.0
        assert profile["max_chi"] == pytest.approx(10.0, rel=0.01)

    def test_percentile_odd_count(self):
        """Median calculation for odd count should use middle value."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        # 5 results
        results = [
            CHIResult(chi=float(i), atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=i*1000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for i in range(1, 6)  # 1.0, 2.0, 3.0, 4.0, 5.0
        ]
        
        profile = build_slice_hardness_profile(results, "odd_count")
        
        # Median should be 3.0 (middle value)
        assert profile["median_chi"] == pytest.approx(3.0, rel=0.01)

    def test_profile_is_json_safe(self):
        """Profile must be JSON-serializable."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        results = [
            CHIResult(chi=10.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=10000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        profile = build_slice_hardness_profile(results, "json_test")
        
        # Should not raise
        json_str = json.dumps(profile)
        parsed = json.loads(json_str)
        
        assert parsed["slice_name"] == profile["slice_name"]
        assert parsed["total_count"] == profile["total_count"]

    def test_profile_is_deterministic(self):
        """Same inputs should always produce identical profile."""
        from normalization.tt_chi import build_slice_hardness_profile, CHIResult
        
        results = [
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        profiles = [
            build_slice_hardness_profile(results, "deterministic")
            for _ in range(5)
        ]
        
        # All should be identical
        for profile in profiles:
            assert profile == profiles[0]


# =============================================================================
# TASK 2: WORKLOAD SHAPING RECOMMENDATION TESTS
# =============================================================================

class TestWorkloadShapingPolicy:
    """
    Tests for derive_tt_workload_shaping_policy().
    
    Verify:
    - Correct policy hints for different slice profiles
    - Async handling recommendations
    - Advisory nature (no behavioral side effects)
    """

    def test_policy_contains_required_fields(self):
        """Policy recommendation must contain all specified fields."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        profile = build_slice_hardness_profile(results, "test")
        policy = derive_tt_workload_shaping_policy(profile)
        
        required_fields = {
            "needs_async_handling",
            "suggested_timeout_ms",
            "policy_hint",
            "reasons",
        }
        
        assert set(policy.keys()) == required_fields

    def test_empty_profile_returns_keep_current(self):
        """Empty profile should recommend KEEP_CURRENT."""
        from normalization.tt_chi import derive_tt_workload_shaping_policy
        
        profile = {
            "total_count": 0,
            "risk_band_counts": {},
            "p90_chi": 0.0,
        }
        
        policy = derive_tt_workload_shaping_policy(profile)
        
        assert policy["policy_hint"] == "KEEP_CURRENT"
        assert policy["needs_async_handling"] is False

    def test_low_risk_profile_returns_keep_current(self):
        """Profile with mostly LOW risk should recommend KEEP_CURRENT."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        # All trivial/easy (LOW risk)
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(10)
        ]
        
        profile = build_slice_hardness_profile(results, "low_risk")
        policy = derive_tt_workload_shaping_policy(profile)
        
        assert policy["policy_hint"] == "KEEP_CURRENT"
        assert policy["needs_async_handling"] is False

    def test_moderate_risk_profile_returns_consider_async(self):
        """Profile with moderate risk should recommend CONSIDER_ASYNC."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        # Mix with some HIGH risk
        results = [
            CHIResult(chi=20.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=20000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0)  # hard → HIGH
            for _ in range(3)  # 3 out of 10 = 30% HIGH
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial → LOW
            for _ in range(7)
        ]
        
        profile = build_slice_hardness_profile(results, "moderate_risk")
        policy = derive_tt_workload_shaping_policy(profile)
        
        assert policy["policy_hint"] == "CONSIDER_ASYNC"
        assert policy["needs_async_handling"] is True

    def test_high_risk_profile_returns_reduce_tt_usage(self):
        """Profile with high EXTREME fraction should recommend REDUCE_TT_USAGE."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        # 2 EXTREME out of 10 = 20% EXTREME
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
            for _ in range(2)
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(8)
        ]
        
        profile = build_slice_hardness_profile(results, "high_risk")
        policy = derive_tt_workload_shaping_policy(profile)
        
        assert policy["policy_hint"] == "REDUCE_TT_USAGE"
        assert policy["needs_async_handling"] is True

    def test_high_p90_triggers_consider_async(self):
        """High p90_chi should trigger CONSIDER_ASYNC even with low risk counts."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        # All LOW risk but high p90
        results = [
            CHIResult(chi=25.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=25000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1562.5)  # hard → HIGH, but p90 will be 25.0
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(9)
        ]
        
        profile = build_slice_hardness_profile(results, "high_p90")
        policy = derive_tt_workload_shaping_policy(profile)
        
        # Should recommend async due to high p90
        assert policy["policy_hint"] in ["CONSIDER_ASYNC", "REDUCE_TT_USAGE"]
        assert policy["needs_async_handling"] is True

    def test_reasons_are_neutral(self):
        """Reasons should be neutral descriptions without prescriptive verbs."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)
            for _ in range(3)  # 3 out of 10 = 30% EXTREME
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(7)
        ]
        
        profile = build_slice_hardness_profile(results, "reasons_test")
        policy = derive_tt_workload_shaping_policy(profile)
        
        # Check that reasons don't contain prescriptive verbs
        all_reasons = " ".join(policy["reasons"]).lower()
        assert "must" not in all_reasons
        assert "should" not in all_reasons
        assert "fix" not in all_reasons
        assert "change" not in all_reasons
        assert len(policy["reasons"]) > 0

    def test_suggested_timeout_based_on_p90(self):
        """Suggested timeout should be based on p90_chi."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            suggest_timeout_ms,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=15.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=15000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1875.0)
        ]
        
        profile = build_slice_hardness_profile(results, "timeout_test")
        policy = derive_tt_workload_shaping_policy(profile)
        
        expected_timeout = suggest_timeout_ms(15.0)
        assert policy["suggested_timeout_ms"] == expected_timeout

    def test_policy_is_advisory_only(self):
        """Policy recommendations should not affect oracle behavior."""
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            CHIResult,
        )
        
        # Baseline oracle result
        result1 = truth_table_is_tautology("p -> p")
        
        # Build profile and derive policy (should have no side effects)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)
        ]
        profile = build_slice_hardness_profile(results, "advisory_test")
        policy = derive_tt_workload_shaping_policy(profile)
        
        # Oracle should behave identically
        result2 = truth_table_is_tautology("p -> p")
        
        assert result1 == result2 == True


# =============================================================================
# TASK 3: CURRICULUM & GLOBAL HEALTH HOOKS TESTS
# =============================================================================

class TestCurriculumSummary:
    """
    Tests for summarize_slice_hardness_for_curriculum().
    
    Verify:
    - Correct hardness status determination
    - Extreme fraction calculation
    - Recommendation hints
    """

    def test_summary_contains_required_fields(self):
        """Summary must contain all specified fields."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        profile = build_slice_hardness_profile(results, "test")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        required_fields = {
            "hardness_status",
            "extreme_fraction",
            "recommendation_hint",
        }
        
        assert set(summary.keys()) == required_fields

    def test_empty_profile_returns_ok(self):
        """Empty profile should return OK status."""
        from normalization.tt_chi import summarize_slice_hardness_for_curriculum
        
        profile = {
            "total_count": 0,
            "risk_band_counts": {},
        }
        
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        assert summary["hardness_status"] == "OK"
        assert summary["extreme_fraction"] == 0.0

    def test_low_extreme_fraction_returns_ok(self):
        """Low extreme fraction should return OK status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 1 EXTREME out of 20 = 5% (below 5% threshold)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(19)
        ]
        
        profile = build_slice_hardness_profile(results, "low_extreme")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        assert summary["hardness_status"] == "OK"
        assert summary["extreme_fraction"] == pytest.approx(0.05, rel=0.01)

    def test_moderate_extreme_fraction_returns_attention(self):
        """Moderate extreme fraction should return ATTENTION status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 1 EXTREME out of 10 = 10% (between 5% and 20%)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(9)
        ]
        
        profile = build_slice_hardness_profile(results, "moderate_extreme")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        assert summary["hardness_status"] == "ATTENTION"
        assert summary["extreme_fraction"] == pytest.approx(0.1, rel=0.01)

    def test_high_extreme_fraction_returns_hot(self):
        """High extreme fraction should return HOT status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 3 EXTREME out of 10 = 30% (above 20% threshold)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
            for _ in range(3)
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(7)
        ]
        
        profile = build_slice_hardness_profile(results, "high_extreme")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        assert summary["hardness_status"] == "HOT"
        assert summary["extreme_fraction"] == pytest.approx(0.3, rel=0.01)

    def test_low_safe_fraction_triggers_attention(self):
        """Low safe fraction should trigger ATTENTION."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 3 LOW out of 10 = 30% safe (below 70% threshold)
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial → LOW
            for _ in range(3)
        ] + [
            CHIResult(chi=20.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=20000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0)  # hard → HIGH
            for _ in range(7)
        ]
        
        profile = build_slice_hardness_profile(results, "low_safe")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        assert summary["hardness_status"] == "ATTENTION"

    def test_recommendation_hint_matches_status(self):
        """Recommendation hint should match hardness status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            summarize_slice_hardness_for_curriculum,
            CURRICULUM_HINTS,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(10)
        ]
        
        profile = build_slice_hardness_profile(results, "hint_test")
        summary = summarize_slice_hardness_for_curriculum(profile)
        
        expected_hint = CURRICULUM_HINTS.get(summary["hardness_status"])
        assert summary["recommendation_hint"] == expected_hint


class TestGlobalHealthSummary:
    """
    Tests for summarize_tt_risk_for_global_health().
    
    Verify:
    - Correct aggregation across slices
    - Overall status determination
    - Slice identification for policy attention
    """

    def test_summary_contains_required_fields(self):
        """Summary must contain all specified fields."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        summaries = [
            {
                "hardness_status": "OK",
                "extreme_fraction": 0.0,
                "recommendation_hint": "OK",
                "slice_name": "slice_1",
            }
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        required_fields = {
            "hot_slice_count",
            "overall_status",
            "slices_needing_policy_attention",
            "total_slices",
        }
        
        assert set(global_health.keys()) == required_fields

    def test_empty_summaries_returns_ok(self):
        """Empty summaries should return OK status."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        global_health = summarize_tt_risk_for_global_health([])
        
        assert global_health["hot_slice_count"] == 0
        assert global_health["overall_status"] == "OK"
        assert global_health["total_slices"] == 0
        assert len(global_health["slices_needing_policy_attention"]) == 0

    def test_all_ok_slices_returns_ok(self):
        """All OK slices should produce OK overall status."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        summaries = [
            {"hardness_status": "OK", "extreme_fraction": 0.0, "slice_name": f"slice_{i}"}
            for i in range(5)
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 0
        assert global_health["overall_status"] == "OK"
        assert global_health["total_slices"] == 5

    def test_single_hot_slice_triggers_warn(self):
        """Single HOT slice should trigger WARN status."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3, "slice_name": "hot_slice"},
            {"hardness_status": "OK", "extreme_fraction": 0.0, "slice_name": "ok_slice"},
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 1
        assert global_health["overall_status"] == "WARN"
        assert "hot_slice" in global_health["slices_needing_policy_attention"]

    def test_multiple_hot_slices_triggers_hot(self):
        """Multiple HOT slices should trigger HOT overall status."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3, "slice_name": "hot_1"},
            {"hardness_status": "HOT", "extreme_fraction": 0.25, "slice_name": "hot_2"},
            {"hardness_status": "OK", "extreme_fraction": 0.0, "slice_name": "ok_1"},
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 2
        assert global_health["overall_status"] == "HOT"
        assert len(global_health["slices_needing_policy_attention"]) == 2

    def test_high_hot_fraction_triggers_hot(self):
        """High fraction of HOT slices should trigger HOT status."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        # 2 HOT out of 10 = 20% (threshold)
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3, "slice_name": f"hot_{i}"}
            for i in range(2)
        ] + [
            {"hardness_status": "OK", "extreme_fraction": 0.0, "slice_name": f"ok_{i}"}
            for i in range(8)
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 2
        assert global_health["overall_status"] == "HOT"

    def test_high_attention_fraction_triggers_warn(self):
        """High fraction of ATTENTION slices should trigger WARN."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        # 4 ATTENTION out of 10 = 40% (above 30% threshold)
        summaries = [
            {"hardness_status": "ATTENTION", "extreme_fraction": 0.1, "slice_name": f"att_{i}"}
            for i in range(4)
        ] + [
            {"hardness_status": "OK", "extreme_fraction": 0.0, "slice_name": f"ok_{i}"}
            for i in range(6)
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 0
        assert global_health["overall_status"] == "WARN"

    def test_slices_without_names_not_in_attention_list(self):
        """Slices without slice_name should not appear in attention list."""
        from normalization.tt_chi import summarize_tt_risk_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3},  # No slice_name
            {"hardness_status": "HOT", "extreme_fraction": 0.25, "slice_name": "named_slice"},
        ]
        
        global_health = summarize_tt_risk_for_global_health(summaries)
        
        assert global_health["hot_slice_count"] == 2
        assert "named_slice" in global_health["slices_needing_policy_attention"]
        assert len(global_health["slices_needing_policy_attention"]) == 1  # Only named one


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhaseIVIntegration:
    """Integration tests across all Phase IV functions."""

    def test_full_workflow_from_results_to_global_health(self):
        """Test complete workflow from CHIResults to global health."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            summarize_slice_hardness_for_curriculum,
            summarize_tt_risk_for_global_health,
            CHIResult,
        )
        
        # Create two slices
        slice1_results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(10)
        ]
        
        slice2_results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)
            for _ in range(5)
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(5)
        ]
        
        # Build profiles
        profile1 = build_slice_hardness_profile(slice1_results, "unit_1")
        profile2 = build_slice_hardness_profile(slice2_results, "unit_2")
        
        # Derive workload policies
        policy1 = derive_tt_workload_shaping_policy(profile1)
        policy2 = derive_tt_workload_shaping_policy(profile2)
        
        # Summarize for curriculum
        summary1 = summarize_slice_hardness_for_curriculum(profile1)
        summary2 = summarize_slice_hardness_for_curriculum(profile2)
        summary2["slice_name"] = "unit_2"  # Add slice name for global health
        
        # Get global health
        global_health = summarize_tt_risk_for_global_health([summary1, summary2])
        
        # Verify chain works correctly
        assert profile1["total_count"] == 10
        assert profile2["total_count"] == 10
        assert policy1["policy_hint"] == "KEEP_CURRENT"
        assert policy2["policy_hint"] in ["CONSIDER_ASYNC", "REDUCE_TT_USAGE"]
        assert summary1["hardness_status"] == "OK"
        assert summary2["hardness_status"] in ["ATTENTION", "HOT"]
        assert global_health["total_slices"] == 2

    def test_all_functions_are_pure(self):
        """All Phase IV functions should be pure (no side effects)."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            derive_tt_workload_shaping_policy,
            summarize_slice_hardness_for_curriculum,
            summarize_tt_risk_for_global_health,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=10.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=10000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ]
        
        # Multiple calls should produce identical results
        for _ in range(3):
            profile1 = build_slice_hardness_profile(results, "test")
            profile2 = build_slice_hardness_profile(results, "test")
            assert profile1 == profile2
            
            policy1 = derive_tt_workload_shaping_policy(profile1)
            policy2 = derive_tt_workload_shaping_policy(profile1)
            assert policy1 == policy2
            
            summary1 = summarize_slice_hardness_for_curriculum(profile1)
            summary2 = summarize_slice_hardness_for_curriculum(profile1)
            assert summary1 == summary2
            
            global1 = summarize_tt_risk_for_global_health([summary1])
            global2 = summarize_tt_risk_for_global_health([summary1])
            assert global1 == global2


# =============================================================================
# PHASE IV EXTENSION: CURRICULUM GATE & TT CAPACITY TILE TESTS
# =============================================================================

class TestCurriculumGate:
    """
    Tests for evaluate_slice_hardness_for_curriculum().
    
    Verify:
    - OK/ATTENTION/BLOCK scenarios
    - Config threshold handling
    - Neutral reasons and suggested actions
    """

    def test_gate_contains_required_fields(self):
        """Gate evaluation must contain all specified fields."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
        ] * 10  # 10 results
        
        profile = build_slice_hardness_profile(results, "test")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        required_fields = {
            "admissible",
            "status",
            "reasons",
            "suggested_actions",
        }
        
        assert set(gate.keys()) == required_fields

    def test_ok_status_for_low_risk_slice(self):
        """Low risk slice should return OK status and be admissible."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # All trivial/easy (LOW risk)
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(10)
        ]
        
        profile = build_slice_hardness_profile(results, "low_risk")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["admissible"] is True
        assert gate["status"] == "OK"
        assert len(gate["reasons"]) > 0
        assert len(gate["suggested_actions"]) >= 0

    def test_attention_status_for_moderate_risk(self):
        """Moderate risk should return ATTENTION status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 1 EXTREME out of 10 = 10% (above 5% threshold)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(9)
        ]
        
        profile = build_slice_hardness_profile(results, "moderate_risk")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["admissible"] is True
        assert gate["status"] == "ATTENTION"
        assert "extreme" in " ".join(gate["reasons"]).lower()

    def test_block_status_for_high_extreme_fraction(self):
        """High extreme fraction should return BLOCK status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 2 EXTREME out of 10 = 20% (above 15% threshold)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
            for _ in range(2)
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(8)
        ]
        
        profile = build_slice_hardness_profile(results, "high_extreme")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["admissible"] is False
        assert gate["status"] == "BLOCK"
        assert "extreme" in " ".join(gate["reasons"]).lower()

    def test_block_status_for_high_max_chi(self):
        """Very high max_chi should return BLOCK status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # Max CHI = 55.0 (above 50.0 threshold)
        results = [
            CHIResult(chi=55.0, atom_count=6, assignment_count=64,
                     assignments_evaluated=64, elapsed_ns=55000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=859.375)  # extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(9)
        ]
        
        profile = build_slice_hardness_profile(results, "high_max_chi")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["admissible"] is False
        assert gate["status"] == "BLOCK"
        assert "maximum" in " ".join(gate["reasons"]).lower() or "max" in " ".join(gate["reasons"]).lower()

    def test_block_status_for_low_count(self):
        """Slice with too few results should return BLOCK status."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # Only 3 results (below minimum of 5)
        results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(3)
        ]
        
        profile = build_slice_hardness_profile(results, "low_count")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["admissible"] is False
        assert gate["status"] == "BLOCK"
        assert "minimum" in " ".join(gate["reasons"]).lower() or "count" in " ".join(gate["reasons"]).lower()

    def test_config_overrides_work(self):
        """Custom config should override default thresholds."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # 1 EXTREME out of 10 = 10% (default would be ATTENTION, but we'll set threshold to 0.08)
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)  # extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(9)
        ]
        
        profile = build_slice_hardness_profile(results, "config_test")
        
        # With default config, should be ATTENTION
        gate_default = evaluate_slice_hardness_for_curriculum(profile)
        assert gate_default["status"] == "ATTENTION"
        
        # With custom config (lower threshold), should be BLOCK
        custom_config = {"max_extreme_fraction": 0.08}
        gate_custom = evaluate_slice_hardness_for_curriculum(profile, config=custom_config)
        assert gate_custom["status"] == "BLOCK"

    def test_reasons_and_actions_are_neutral(self):
        """Reasons and suggested actions should be neutral (no prescriptive verbs)."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        results = [
            CHIResult(chi=30.0, atom_count=5, assignment_count=32,
                     assignments_evaluated=32, elapsed_ns=30000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=937.5)
            for _ in range(2)  # 2 out of 10 = 20% extreme
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)
            for _ in range(8)
        ]
        
        profile = build_slice_hardness_profile(results, "neutral_test")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        # Check that reasons and actions don't contain prescriptive verbs
        all_text = " ".join(gate["reasons"] + gate["suggested_actions"]).lower()
        assert "must" not in all_text
        assert "should" not in all_text
        assert "fix" not in all_text
        assert "change" not in all_text
        assert "do" not in all_text or "do not" in all_text  # "do not" is OK

    def test_attention_for_high_p90(self):
        """High p90_chi should trigger ATTENTION."""
        from normalization.tt_chi import (
            build_slice_hardness_profile,
            evaluate_slice_hardness_for_curriculum,
            CHIResult,
        )
        
        # High p90 but no extreme cases (all hard, not extreme)
        results = [
            CHIResult(chi=25.0, atom_count=4, assignment_count=16,
                     assignments_evaluated=16, elapsed_ns=25000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1562.5)  # hard (not extreme)
            for _ in range(2)  # 2 hard cases
        ] + [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0)  # trivial
            for _ in range(8)
        ]
        
        profile = build_slice_hardness_profile(results, "high_p90")
        gate = evaluate_slice_hardness_for_curriculum(profile)
        
        assert gate["status"] == "ATTENTION"
        # Should mention p90 or max_chi in reasons
        reasons_text = " ".join(gate["reasons"]).lower()
        assert ("90th percentile" in reasons_text or "p90" in reasons_text or 
                "maximum" in reasons_text or "max" in reasons_text)


class TestTTCapacityTile:
    """
    Tests for summarize_tt_capacity_for_global_health().
    
    Verify:
    - Multi-slice aggregation
    - Deterministic notes
    - Status determination (OK/WARN/HOT)
    """

    def test_capacity_contains_required_fields(self):
        """Capacity summary must contain all specified fields."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        required_fields = {
            "total_slices",
            "slices_at_high_risk",
            "global_tt_status",
            "notes",
        }
        
        assert set(capacity.keys()) == required_fields

    def test_empty_summaries_returns_ok(self):
        """Empty summaries should return OK status."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        capacity = summarize_tt_capacity_for_global_health([])
        
        assert capacity["total_slices"] == 0
        assert capacity["slices_at_high_risk"] == 0
        assert capacity["global_tt_status"] == "OK"
        assert len(capacity["notes"]) > 0

    def test_all_ok_slices_returns_ok(self):
        """All OK slices should produce OK global status."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "OK", "extreme_fraction": 0.0},
            {"hardness_status": "OK", "extreme_fraction": 0.0},
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity["total_slices"] == 3
        assert capacity["slices_at_high_risk"] == 0
        assert capacity["global_tt_status"] == "OK"

    def test_hot_slice_triggers_hot(self):
        """Any HOT slice should trigger HOT global status."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3},
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity["slices_at_high_risk"] == 1
        assert capacity["global_tt_status"] == "HOT"

    def test_block_slice_triggers_hot(self):
        """Any BLOCK slice should trigger HOT global status."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"status": "BLOCK", "admissible": False},  # From evaluate_slice_hardness_for_curriculum
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity["slices_at_high_risk"] == 1
        assert capacity["global_tt_status"] == "HOT"

    def test_high_attention_fraction_triggers_warn(self):
        """High fraction of ATTENTION slices should trigger WARN."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        # 4 ATTENTION out of 10 = 40% (above 30% threshold)
        summaries = [
            {"hardness_status": "ATTENTION", "extreme_fraction": 0.1}
            for _ in range(4)
        ] + [
            {"hardness_status": "OK", "extreme_fraction": 0.0}
            for _ in range(6)
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity["total_slices"] == 10
        assert capacity["slices_at_high_risk"] == 0
        assert capacity["global_tt_status"] == "WARN"

    def test_notes_are_deterministic(self):
        """Notes should be deterministic for same inputs."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "OK", "extreme_fraction": 0.0},
            {"hardness_status": "ATTENTION", "extreme_fraction": 0.1},
        ]
        
        capacity1 = summarize_tt_capacity_for_global_health(summaries)
        capacity2 = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity1["notes"] == capacity2["notes"]

    def test_notes_are_neutral(self):
        """Notes should be neutral descriptions."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        all_notes = " ".join(capacity["notes"]).lower()
        assert "must" not in all_notes
        assert "should" not in all_notes
        assert "fix" not in all_notes

    def test_mixed_status_counts(self):
        """Should correctly count slices at high risk."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "HOT", "extreme_fraction": 0.3},
            {"status": "BLOCK", "admissible": False},
            {"hardness_status": "ATTENTION", "extreme_fraction": 0.1},
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        assert capacity["total_slices"] == 4
        assert capacity["slices_at_high_risk"] == 2  # 1 HOT + 1 BLOCK
        assert capacity["global_tt_status"] == "HOT"

    def test_capacity_is_json_safe(self):
        """Capacity summary must be JSON-serializable."""
        from normalization.tt_chi import summarize_tt_capacity_for_global_health
        
        summaries = [
            {"hardness_status": "OK", "extreme_fraction": 0.0},
        ]
        
        capacity = summarize_tt_capacity_for_global_health(summaries)
        
        # Should not raise
        json_str = json.dumps(capacity)
        parsed = json.loads(json_str)
        
        assert parsed["global_tt_status"] == capacity["global_tt_status"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

