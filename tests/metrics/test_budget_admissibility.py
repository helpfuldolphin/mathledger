"""
Tests for Budget Admissibility Evaluator

==============================================================================
STATUS: PHASE II — NOT RUN IN PHASE I
==============================================================================

Table-based tests for admissibility regions matching BUDGET_ADMISSIBILITY_SPEC.md.

v2 additions:
- Snapshot contract tests
- Multi-run sentinel grid tests
- Governance/MAAS summary tests
"""
import json
import tempfile
from pathlib import Path

import pytest

from backend.metrics.budget_admissibility import (
    AdmissibilityClassification,
    BudgetAdmissibilityResult,
    BudgetStats,
    classify_budget_admissibility,
    format_admissibility_report,
    TAU_MAX,
    TAU_SUSPICIOUS,
    TAU_REJECT,
    DELTA_SYM,
    DELTA_REJECT,
    KAPPA_MIN,
    COMPLETE_SKIP_MAX,
    # v2 imports
    BUDGET_SNAPSHOT_SCHEMA_VERSION,
    BUDGET_SENTINEL_GRID_SCHEMA_VERSION,
    build_budget_admissibility_snapshot,
    save_budget_admissibility_snapshot,
    load_budget_admissibility_snapshot,
    build_budget_sentinel_grid,
    summarize_budget_for_governance,
    # v3 imports (Phase III)
    BUDGET_DRIFT_LEDGER_SCHEMA_VERSION,
    BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION,
    DRIFT_RATE_THRESHOLD,
    DRIFT_ASYMMETRY_THRESHOLD,
    REPEATED_INVALID_THRESHOLD,
    UpliftBudgetGate,
    BudgetStability,
    build_budget_drift_ledger,
    evaluate_budget_for_uplift,
    summarize_budget_for_global_health,
    # v4 imports (Phase IV)
    BUDGET_RISK_MAP_SCHEMA_VERSION,
    BUDGET_METRIC_UPLIFT_SCHEMA_VERSION,
    BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION,
    RiskBand,
    JointUpliftStatus,
    StatusLight,
    build_budget_risk_map,
    summarize_budget_and_metrics_for_uplift,
    build_budget_director_panel,
    # v5 imports (Phase V)
    BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION,
    BUDGET_POLICY_IMPACT_SCHEMA_VERSION,
    BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION,
    RiskTrend,
    build_budget_risk_trajectory,
    summarize_budget_impact_on_policy,
    build_budget_director_panel_v2,
    # v5 adapters (Global Console + Governance)
    BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION,
    BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION,
    GovernanceSignal,
    summarize_budget_trajectory_for_global_console,
    to_governance_signal_for_budget,
)


# =============================================================================
# Helper Functions
# =============================================================================

def make_stats(
    total_candidates: int = 1000,
    total_skipped: int = 0,
    exhausted_cycles: int = 0,
    complete_skip_cycles: int = 0,
    total_cycles: int = 10,
    min_cycle_coverage: float = 1.0,
    skip_trend_significant: bool = False,
) -> BudgetStats:
    """Create BudgetStats with sensible defaults."""
    return BudgetStats(
        total_candidates=total_candidates,
        total_skipped=total_skipped,
        exhausted_cycles=exhausted_cycles,
        complete_skip_cycles=complete_skip_cycles,
        total_cycles=total_cycles,
        min_cycle_coverage=min_cycle_coverage,
        skip_trend_significant=skip_trend_significant,
    )


# =============================================================================
# Threshold Constants Tests
# =============================================================================

class TestThresholdConstants:
    """Verify threshold constants match spec."""

    def test_tau_max(self):
        """τ_max = 0.15 (15%)"""
        assert TAU_MAX == 0.15

    def test_tau_suspicious(self):
        """τ_suspicious = 0.30 (30%)"""
        assert TAU_SUSPICIOUS == 0.30

    def test_tau_reject(self):
        """τ_reject = 0.50 (50%)"""
        assert TAU_REJECT == 0.50

    def test_delta_sym(self):
        """δ_sym = 0.05 (5%)"""
        assert DELTA_SYM == 0.05

    def test_delta_reject(self):
        """δ_reject = 0.20 (20%)"""
        assert DELTA_REJECT == 0.20

    def test_kappa_min(self):
        """κ_min = 0.50 (50%)"""
        assert KAPPA_MIN == 0.50

    def test_complete_skip_max(self):
        """complete_skip_max = 0.10 (10%)"""
        assert COMPLETE_SKIP_MAX == 0.10


# =============================================================================
# BudgetStats Property Tests
# =============================================================================

class TestBudgetStatsProperties:
    """Test BudgetStats computed properties."""

    def test_exhaustion_rate_zero_candidates(self):
        """Zero candidates returns 0.0 exhaustion rate."""
        stats = make_stats(total_candidates=0, total_skipped=0)
        assert stats.exhaustion_rate == 0.0

    def test_exhaustion_rate_no_skips(self):
        """No skips returns 0.0 exhaustion rate."""
        stats = make_stats(total_candidates=1000, total_skipped=0)
        assert stats.exhaustion_rate == 0.0

    def test_exhaustion_rate_some_skips(self):
        """Some skips returns correct rate."""
        stats = make_stats(total_candidates=1000, total_skipped=100)
        assert stats.exhaustion_rate == 0.10

    def test_exhaustion_rate_all_skips(self):
        """All skips returns 1.0."""
        stats = make_stats(total_candidates=1000, total_skipped=1000)
        assert stats.exhaustion_rate == 1.0

    def test_complete_skip_fraction_zero_cycles(self):
        """Zero cycles returns 0.0 fraction."""
        stats = make_stats(total_cycles=0, complete_skip_cycles=0)
        assert stats.complete_skip_fraction == 0.0

    def test_complete_skip_fraction_some(self):
        """Some complete-skip cycles returns correct fraction."""
        stats = make_stats(total_cycles=10, complete_skip_cycles=2)
        assert stats.complete_skip_fraction == 0.20


# =============================================================================
# Admissibility Condition Tests (A1-A4)
# =============================================================================

class TestAdmissibilityConditions:
    """Test A1-A4 admissibility conditions."""

    def test_a1_bounded_rate_satisfied(self):
        """A1: B_rate <= τ_max (15%) → satisfied."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=120)  # 12%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a1_bounded_rate is True

    def test_a1_bounded_rate_violated(self):
        """A1: B_rate > τ_max (15%) → violated."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        rfl = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a1_bounded_rate is False

    def test_a2_symmetry_satisfied(self):
        """A2: asymmetry <= δ_sym (5%) → satisfied."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=120)  # 12%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a2_symmetry is True
        assert result.asymmetry == pytest.approx(0.02)

    def test_a2_symmetry_violated(self):
        """A2: asymmetry > δ_sym (5%) → violated."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)  # 5%
        rfl = make_stats(total_candidates=1000, total_skipped=120)  # 12%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a2_symmetry is False
        assert result.asymmetry == pytest.approx(0.07)

    def test_a3_independence_satisfied(self):
        """A3: no skip trend → satisfied."""
        baseline = make_stats(skip_trend_significant=False)
        rfl = make_stats(skip_trend_significant=False)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a3_independence is True

    def test_a3_independence_violated(self):
        """A3: skip trend detected → violated."""
        baseline = make_stats(skip_trend_significant=True)
        rfl = make_stats(skip_trend_significant=False)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a3_independence is False

    def test_a4_cycle_coverage_satisfied(self):
        """A4: min_coverage >= κ_min (50%) → satisfied."""
        baseline = make_stats(min_cycle_coverage=0.80)
        rfl = make_stats(min_cycle_coverage=0.75)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a4_cycle_coverage is True

    def test_a4_cycle_coverage_violated(self):
        """A4: min_coverage < κ_min (50%) → violated."""
        baseline = make_stats(min_cycle_coverage=0.45)
        rfl = make_stats(min_cycle_coverage=0.80)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a4_cycle_coverage is False


# =============================================================================
# Rejection Condition Tests (R1-R4)
# =============================================================================

class TestRejectionConditions:
    """Test R1-R4 rejection conditions."""

    def test_r1_excessive_rate_not_triggered(self):
        """R1: B_rate <= 50% → not triggered."""
        baseline = make_stats(total_candidates=1000, total_skipped=400)  # 40%
        rfl = make_stats(total_candidates=1000, total_skipped=450)  # 45%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r1_excessive_rate is False

    def test_r1_excessive_rate_triggered(self):
        """R1: B_rate > 50% → triggered → INVALID."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=550)  # 55%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r1_excessive_rate is True
        assert result.classification == AdmissibilityClassification.INVALID

    def test_r2_severe_asymmetry_not_triggered(self):
        """R2: asymmetry <= 20% → not triggered."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=280)  # 28%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r2_severe_asymmetry is False

    def test_r2_severe_asymmetry_triggered(self):
        """R2: asymmetry > 20% → triggered → INVALID."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)  # 5%
        rfl = make_stats(total_candidates=1000, total_skipped=300)  # 30%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r2_severe_asymmetry is True
        assert result.classification == AdmissibilityClassification.INVALID

    def test_r3_systematic_bias_not_triggered(self):
        """R3: no trend → not triggered."""
        baseline = make_stats(skip_trend_significant=False)
        rfl = make_stats(skip_trend_significant=False)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r3_systematic_bias is False

    def test_r3_systematic_bias_triggered(self):
        """R3: significant trend → triggered → INVALID."""
        baseline = make_stats(skip_trend_significant=False)
        rfl = make_stats(skip_trend_significant=True)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r3_systematic_bias is True
        assert result.classification == AdmissibilityClassification.INVALID

    def test_r4_complete_failures_not_triggered(self):
        """R4: complete-skip cycles <= 10% → not triggered."""
        baseline = make_stats(total_cycles=20, complete_skip_cycles=2)  # 10%
        rfl = make_stats(total_cycles=20, complete_skip_cycles=1)  # 5%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r4_complete_failures is False

    def test_r4_complete_failures_triggered(self):
        """R4: complete-skip cycles > 10% → triggered → INVALID."""
        baseline = make_stats(total_cycles=10, complete_skip_cycles=2)  # 20%
        rfl = make_stats(total_cycles=10, complete_skip_cycles=0)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r4_complete_failures is True
        assert result.classification == AdmissibilityClassification.INVALID


# =============================================================================
# Classification Region Tests (Table-Driven)
# =============================================================================

class TestClassificationRegions:
    """
    Table-based tests for classification regions.
    Matches BUDGET_ADMISSIBILITY_SPEC.md §3.3 Partition Table.
    """

    # SAFE region tests
    @pytest.mark.parametrize("b_rate_pct,expected", [
        (0, AdmissibilityClassification.SAFE),
        (5, AdmissibilityClassification.SAFE),
        (10, AdmissibilityClassification.SAFE),
        (14, AdmissibilityClassification.SAFE),
    ])
    def test_safe_region_low_exhaustion(self, b_rate_pct, expected):
        """B_rate <= 15% with symmetric conditions → SAFE."""
        baseline = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        rfl = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.classification == expected

    # SUSPICIOUS region tests
    @pytest.mark.parametrize("b_rate_pct,effect,expected", [
        (20, 0.05, AdmissibilityClassification.SUSPICIOUS),  # 15-30%, small effect
        (25, 0.10, AdmissibilityClassification.SUSPICIOUS),  # 15-30%, medium effect
        (35, 0.20, AdmissibilityClassification.SUSPICIOUS),  # 30-50%, large effect
    ])
    def test_suspicious_region(self, b_rate_pct, effect, expected):
        """Various SUSPICIOUS region conditions."""
        baseline = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        rfl = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        result = classify_budget_admissibility(baseline, rfl, effect_magnitude=effect)
        assert result.classification == expected

    # INVALID region tests
    @pytest.mark.parametrize("b_rate_pct,effect,expected", [
        (55, 0.05, AdmissibilityClassification.INVALID),  # >50% → INVALID
        (55, 0.20, AdmissibilityClassification.INVALID),  # >50% even with large effect
        (35, 0.05, AdmissibilityClassification.INVALID),  # 30-50%, small effect
        (40, 0.10, AdmissibilityClassification.INVALID),  # 30-50%, medium effect
    ])
    def test_invalid_region(self, b_rate_pct, effect, expected):
        """Various INVALID region conditions."""
        baseline = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        rfl = make_stats(total_candidates=1000, total_skipped=b_rate_pct * 10)
        result = classify_budget_admissibility(baseline, rfl, effect_magnitude=effect)
        assert result.classification == expected


class TestAsymmetryModifier:
    """
    Test asymmetry modifier from BUDGET_ADMISSIBILITY_SPEC.md §3.4.
    """

    def test_asymmetry_less_than_2pct_no_change(self):
        """Asymmetry <= 2% → no change to classification."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=115)  # 11.5% → 1.5% asym
        result = classify_budget_admissibility(baseline, rfl)
        assert result.classification == AdmissibilityClassification.SAFE

    def test_asymmetry_2_to_5pct_no_change(self):
        """Asymmetry 2-5% → no change (within tolerance)."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)  # 10%
        rfl = make_stats(total_candidates=1000, total_skipped=140)  # 14% → 4% asym
        result = classify_budget_admissibility(baseline, rfl)
        assert result.classification == AdmissibilityClassification.SAFE

    def test_asymmetry_10_to_20pct_downgrade(self):
        """Asymmetry 10-20% → downgrade SAFE → SUSPICIOUS."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)  # 5%
        rfl = make_stats(total_candidates=1000, total_skipped=180)  # 18% → 13% asym
        result = classify_budget_admissibility(baseline, rfl)
        # 13% asymmetry with 18% max rate → SUSPICIOUS
        assert result.classification == AdmissibilityClassification.SUSPICIOUS

    def test_asymmetry_greater_than_20pct_invalid(self):
        """Asymmetry > 20% → INVALID."""
        baseline = make_stats(total_candidates=1000, total_skipped=30)  # 3%
        rfl = make_stats(total_candidates=1000, total_skipped=280)  # 28% → 25% asym
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r2_severe_asymmetry is True
        assert result.classification == AdmissibilityClassification.INVALID


class TestEffectSizeMitigation:
    """Test effect size mitigation of budget concerns."""

    def test_large_effect_mitigates_moderate_exhaustion(self):
        """15-30% exhaustion with large effect (>=15%) → SAFE."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        rfl = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        result = classify_budget_admissibility(baseline, rfl, effect_magnitude=0.20)
        assert result.classification == AdmissibilityClassification.SAFE

    def test_small_effect_with_moderate_exhaustion(self):
        """15-30% exhaustion with small effect (<15%) → SUSPICIOUS."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        rfl = make_stats(1000, total_skipped=200)  # 20%
        result = classify_budget_admissibility(baseline, rfl, effect_magnitude=0.05)
        assert result.classification == AdmissibilityClassification.SUSPICIOUS

    def test_no_effect_assumes_conservative(self):
        """No effect_magnitude → assumes small effect (conservative)."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        rfl = make_stats(total_candidates=1000, total_skipped=200)  # 20%
        result = classify_budget_admissibility(baseline, rfl, effect_magnitude=None)
        assert result.classification == AdmissibilityClassification.SUSPICIOUS


# =============================================================================
# Result Properties Tests
# =============================================================================

class TestResultProperties:
    """Test BudgetAdmissibilityResult computed properties."""

    def test_is_valid_for_safe(self):
        """SAFE → is_valid = True."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.is_valid is True

    def test_is_valid_for_suspicious(self):
        """SUSPICIOUS → is_valid = True."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)
        rfl = make_stats(total_candidates=1000, total_skipped=200)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.is_valid is True

    def test_is_valid_for_invalid(self):
        """INVALID → is_valid = False."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.is_valid is False

    def test_any_rejection_true_when_r1(self):
        """any_rejection = True when R1 triggered."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.any_rejection is True

    def test_any_rejection_false_when_safe(self):
        """any_rejection = False when SAFE."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.any_rejection is False

    def test_all_admissible_true_when_all_conditions_met(self):
        """all_admissible = True when A1-A4 all satisfied."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.all_admissible is True


# =============================================================================
# Report Formatting Tests
# =============================================================================

class TestReportFormatting:
    """Test human-readable report generation."""

    def test_safe_report_contains_classification(self):
        """SAFE report contains classification header."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        report = format_admissibility_report(result)
        assert "BUDGET ADMISSIBILITY: SAFE" in report

    def test_suspicious_report_contains_classification(self):
        """SUSPICIOUS report contains classification header."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)
        rfl = make_stats(total_candidates=1000, total_skipped=200)
        result = classify_budget_admissibility(baseline, rfl)
        report = format_admissibility_report(result)
        assert "BUDGET ADMISSIBILITY: SUSPICIOUS" in report

    def test_invalid_report_contains_classification(self):
        """INVALID report contains classification header."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)
        report = format_admissibility_report(result)
        assert "BUDGET ADMISSIBILITY: INVALID" in report

    def test_report_contains_rates(self):
        """Report contains exhaustion rates."""
        baseline = make_stats(total_candidates=1000, total_skipped=48)  # 4.8%
        rfl = make_stats(total_candidates=1000, total_skipped=52)  # 5.2%
        result = classify_budget_admissibility(baseline, rfl)
        report = format_admissibility_report(result)
        assert "4.8%" in report
        assert "5.2%" in report


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_candidates_both(self):
        """Zero candidates in both → SAFE (no data, no problem)."""
        baseline = make_stats(total_candidates=0, total_skipped=0)
        rfl = make_stats(total_candidates=0, total_skipped=0)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.classification == AdmissibilityClassification.SAFE

    def test_exact_threshold_tau_max(self):
        """Exactly at τ_max (15%) → SAFE."""
        baseline = make_stats(total_candidates=1000, total_skipped=150)  # 15%
        rfl = make_stats(total_candidates=1000, total_skipped=150)  # 15%
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a1_bounded_rate is True
        assert result.classification == AdmissibilityClassification.SAFE

    def test_just_above_tau_max(self):
        """Just above τ_max (15.1%) → a1 violated."""
        baseline = make_stats(total_candidates=1000, total_skipped=151)  # 15.1%
        rfl = make_stats(total_candidates=1000, total_skipped=151)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.a1_bounded_rate is False

    def test_exact_threshold_tau_reject(self):
        """Exactly at τ_reject (50%) → not R1 (boundary)."""
        baseline = make_stats(total_candidates=1000, total_skipped=500)  # 50%
        rfl = make_stats(total_candidates=1000, total_skipped=500)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r1_excessive_rate is False  # <= 50% is not excessive

    def test_just_above_tau_reject(self):
        """Just above τ_reject (50.1%) → R1 triggered."""
        baseline = make_stats(total_candidates=1000, total_skipped=501)  # 50.1%
        rfl = make_stats(total_candidates=1000, total_skipped=501)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r1_excessive_rate is True
        assert result.classification == AdmissibilityClassification.INVALID

    def test_multiple_rejection_conditions(self):
        """Multiple R conditions triggered → still INVALID."""
        baseline = make_stats(
            total_candidates=1000,
            total_skipped=600,
            total_cycles=10,
            complete_skip_cycles=3,
            skip_trend_significant=True,
        )
        rfl = make_stats(total_candidates=1000, total_skipped=100)
        result = classify_budget_admissibility(baseline, rfl)
        assert result.r1_excessive_rate is True
        assert result.r2_severe_asymmetry is True
        assert result.r3_systematic_bias is True
        assert result.r4_complete_failures is True
        assert result.classification == AdmissibilityClassification.INVALID

    def test_reasons_list_populated(self):
        """Reasons list is populated with explanations."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        assert len(result.reasons) > 0
        assert any("satisfied" in r.lower() for r in result.reasons)


# =============================================================================
# BUDGET SENTINEL GRID v2 — Snapshot Contract Tests
# =============================================================================

class TestBudgetSnapshotContract:
    """Tests for build_budget_admissibility_snapshot() contract."""

    def test_snapshot_contains_schema_version(self):
        """Snapshot includes schema_version."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result)
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == BUDGET_SNAPSHOT_SCHEMA_VERSION

    def test_snapshot_contains_required_keys(self):
        """Snapshot contains all required keys."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)
        rfl = make_stats(total_candidates=1000, total_skipped=120)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result, slice_name="test_slice")

        required_keys = {
            "schema_version",
            "slice_name",
            "classification",
            "exhaustion_rate_baseline",
            "exhaustion_rate_rfl",
            "asymmetry_score",
            "max_exhaustion_rate",
            "min_cycle_coverage",
            "A_flags",
            "R_flags",
        }
        assert required_keys.issubset(set(snapshot.keys()))

    def test_snapshot_a_flags_structure(self):
        """A_flags contains A1-A4 booleans."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result)

        a_flags = snapshot["A_flags"]
        assert "a1_bounded_rate" in a_flags
        assert "a2_symmetry" in a_flags
        assert "a3_independence" in a_flags
        assert "a4_cycle_coverage" in a_flags
        assert all(isinstance(v, bool) for v in a_flags.values())

    def test_snapshot_r_flags_structure(self):
        """R_flags contains R1-R4 booleans."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=100)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result)

        r_flags = snapshot["R_flags"]
        assert "r1_excessive_rate" in r_flags
        assert "r2_severe_asymmetry" in r_flags
        assert "r3_systematic_bias" in r_flags
        assert "r4_complete_failures" in r_flags
        assert all(isinstance(v, bool) for v in r_flags.values())

    def test_snapshot_slice_name_optional(self):
        """slice_name is None when not provided."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result)
        assert snapshot["slice_name"] is None

    def test_snapshot_slice_name_when_provided(self):
        """slice_name is set when provided."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result, slice_name="prop_depth4")
        assert snapshot["slice_name"] == "prop_depth4"

    def test_snapshot_determinism(self):
        """Same input → same snapshot (deterministic)."""
        baseline = make_stats(total_candidates=1000, total_skipped=150)
        rfl = make_stats(total_candidates=1000, total_skipped=180)
        result = classify_budget_admissibility(baseline, rfl)

        snapshot1 = build_budget_admissibility_snapshot(result, slice_name="test")
        snapshot2 = build_budget_admissibility_snapshot(result, slice_name="test")

        # Convert to JSON for deterministic comparison
        json1 = json.dumps(snapshot1, sort_keys=True)
        json2 = json.dumps(snapshot2, sort_keys=True)
        assert json1 == json2

    def test_snapshot_json_serializable(self):
        """Snapshot is fully JSON-serializable."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result, slice_name="test")

        # Should not raise
        json_str = json.dumps(snapshot)
        loaded = json.loads(json_str)
        assert loaded == snapshot


class TestBudgetSnapshotIO:
    """Tests for save/load snapshot helpers."""

    def test_save_and_load_roundtrip(self):
        """save/load roundtrip preserves data."""
        baseline = make_stats(total_candidates=1000, total_skipped=100)
        rfl = make_stats(total_candidates=1000, total_skipped=120)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result, slice_name="test_slice")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_budget_admissibility_snapshot(path, snapshot)
            loaded = load_budget_admissibility_snapshot(path)
            assert loaded == snapshot
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_missing_file_raises(self):
        """load raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_budget_admissibility_snapshot("/nonexistent/path/snapshot.json")

    def test_save_creates_valid_json(self):
        """save creates valid JSON file."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        snapshot = build_budget_admissibility_snapshot(result)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_budget_admissibility_snapshot(path, snapshot)
            with open(path, 'r') as f:
                content = f.read()
            # Should be valid JSON
            parsed = json.loads(content)
            assert "schema_version" in parsed
        finally:
            Path(path).unlink(missing_ok=True)


# =============================================================================
# BUDGET SENTINEL GRID v2 — Multi-Run Grid Tests
# =============================================================================

class TestBudgetSentinelGrid:
    """Tests for build_budget_sentinel_grid() aggregation."""

    def test_grid_contains_schema_version(self):
        """Grid includes schema_version."""
        snapshots = []
        grid = build_budget_sentinel_grid(snapshots)
        assert "schema_version" in grid
        assert grid["schema_version"] == BUDGET_SENTINEL_GRID_SCHEMA_VERSION

    def test_grid_empty_snapshots(self):
        """Empty snapshots → zero counts."""
        grid = build_budget_sentinel_grid([])
        assert grid["total_runs"] == 0
        assert grid["safe_count"] == 0
        assert grid["suspicious_count"] == 0
        assert grid["invalid_count"] == 0
        assert grid["per_slice"] == {}

    def test_grid_counts_safe_correctly(self):
        """Grid counts SAFE classifications correctly."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "s1"},
            {"classification": "SAFE", "slice_name": "s2"},
            {"classification": "SAFE", "slice_name": "s3"},
        ]
        grid = build_budget_sentinel_grid(snapshots)
        assert grid["total_runs"] == 3
        assert grid["safe_count"] == 3
        assert grid["suspicious_count"] == 0
        assert grid["invalid_count"] == 0

    def test_grid_counts_mixed_correctly(self):
        """Grid counts mixed classifications correctly."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "s1"},
            {"classification": "SUSPICIOUS", "slice_name": "s2"},
            {"classification": "INVALID", "slice_name": "s3"},
            {"classification": "SAFE", "slice_name": "s4"},
        ]
        grid = build_budget_sentinel_grid(snapshots)
        assert grid["total_runs"] == 4
        assert grid["safe_count"] == 2
        assert grid["suspicious_count"] == 1
        assert grid["invalid_count"] == 1

    def test_grid_per_slice_last_classification(self):
        """per_slice shows last classification per slice."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "slice_a"},
            {"classification": "SUSPICIOUS", "slice_name": "slice_a"},  # Later
            {"classification": "SAFE", "slice_name": "slice_b"},
        ]
        grid = build_budget_sentinel_grid(snapshots)
        assert grid["per_slice"]["slice_a"] == "SUSPICIOUS"  # Last one wins
        assert grid["per_slice"]["slice_b"] == "SAFE"

    def test_grid_per_slice_sorted_alphabetically(self):
        """per_slice keys are sorted alphabetically."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "zebra"},
            {"classification": "SAFE", "slice_name": "alpha"},
            {"classification": "SAFE", "slice_name": "beta"},
        ]
        grid = build_budget_sentinel_grid(snapshots)
        keys = list(grid["per_slice"].keys())
        assert keys == sorted(keys)
        assert keys == ["alpha", "beta", "zebra"]

    def test_grid_ignores_none_slice_names(self):
        """Snapshots with None slice_name don't appear in per_slice."""
        snapshots = [
            {"classification": "SAFE", "slice_name": None},
            {"classification": "SAFE", "slice_name": "named_slice"},
        ]
        grid = build_budget_sentinel_grid(snapshots)
        assert "named_slice" in grid["per_slice"]
        assert None not in grid["per_slice"]
        assert len(grid["per_slice"]) == 1

    def test_grid_determinism(self):
        """Same snapshots → same grid (deterministic)."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "b"},
            {"classification": "SUSPICIOUS", "slice_name": "a"},
            {"classification": "INVALID", "slice_name": "c"},
        ]
        grid1 = build_budget_sentinel_grid(snapshots)
        grid2 = build_budget_sentinel_grid(snapshots)

        json1 = json.dumps(grid1, sort_keys=True)
        json2 = json.dumps(grid2, sort_keys=True)
        assert json1 == json2

    def test_grid_json_serializable(self):
        """Grid is fully JSON-serializable."""
        snapshots = [
            {"classification": "SAFE", "slice_name": "test"},
        ]
        grid = build_budget_sentinel_grid(snapshots)

        # Should not raise
        json_str = json.dumps(grid)
        loaded = json.loads(json_str)
        assert loaded == grid


# =============================================================================
# BUDGET SENTINEL GRID v2 — Governance/MAAS Summary Tests
# =============================================================================

class TestGovernanceSummary:
    """Tests for summarize_budget_for_governance() helper."""

    def test_safe_is_admissible(self):
        """SAFE → is_budget_admissible = True."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert summary["is_budget_admissible"] is True
        assert summary["classification"] == "SAFE"

    def test_suspicious_is_admissible(self):
        """SUSPICIOUS → is_budget_admissible = True (not rejected)."""
        baseline = make_stats(total_candidates=1000, total_skipped=200)
        rfl = make_stats(total_candidates=1000, total_skipped=200)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert summary["is_budget_admissible"] is True
        assert summary["classification"] == "SUSPICIOUS"

    def test_invalid_is_not_admissible(self):
        """INVALID → is_budget_admissible = False."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert summary["is_budget_admissible"] is False
        assert summary["classification"] == "INVALID"

    def test_rejection_reasons_populated_for_r1(self):
        """R1 trigger populates rejection_reasons."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert len(summary["rejection_reasons"]) > 0
        assert any("R1:" in r for r in summary["rejection_reasons"])

    def test_rejection_reasons_populated_for_r2(self):
        """R2 trigger populates rejection_reasons."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=300)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert any("R2:" in r for r in summary["rejection_reasons"])

    def test_rejection_reasons_populated_for_r3(self):
        """R3 trigger populates rejection_reasons."""
        baseline = make_stats(skip_trend_significant=True)
        rfl = make_stats(skip_trend_significant=False)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert any("R3:" in r for r in summary["rejection_reasons"])

    def test_rejection_reasons_populated_for_r4(self):
        """R4 trigger populates rejection_reasons."""
        baseline = make_stats(total_cycles=10, complete_skip_cycles=3)
        rfl = make_stats(total_cycles=10, complete_skip_cycles=0)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert any("R4:" in r for r in summary["rejection_reasons"])

    def test_rejection_reasons_empty_when_safe(self):
        """SAFE → rejection_reasons is empty."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        assert summary["rejection_reasons"] == []

    def test_rejection_reasons_only_r_conditions(self):
        """rejection_reasons only includes R-condition messages."""
        baseline = make_stats(
            total_candidates=1000,
            total_skipped=600,
            total_cycles=10,
            complete_skip_cycles=3,
        )
        rfl = make_stats(total_candidates=1000, total_skipped=100)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        # All reasons should start with R1:, R2:, R3:, or R4:
        for reason in summary["rejection_reasons"]:
            assert reason.startswith(("R1:", "R2:", "R3:", "R4:"))

    def test_summary_determinism(self):
        """Same result → same summary (deterministic)."""
        baseline = make_stats(total_candidates=1000, total_skipped=600)
        rfl = make_stats(total_candidates=1000, total_skipped=600)
        result = classify_budget_admissibility(baseline, rfl)

        summary1 = summarize_budget_for_governance(result)
        summary2 = summarize_budget_for_governance(result)

        json1 = json.dumps(summary1, sort_keys=True)
        json2 = json.dumps(summary2, sort_keys=True)
        assert json1 == json2

    def test_summary_json_serializable(self):
        """Summary is fully JSON-serializable."""
        baseline = make_stats(total_candidates=1000, total_skipped=50)
        rfl = make_stats(total_candidates=1000, total_skipped=50)
        result = classify_budget_admissibility(baseline, rfl)
        summary = summarize_budget_for_governance(result)

        # Should not raise
        json_str = json.dumps(summary)
        loaded = json.loads(json_str)
        assert loaded == summary


# =============================================================================
# BUDGET SENTINEL GRID v3 — Phase III Drift Ledger Tests
# =============================================================================

class TestBudgetDriftLedger:
    """Tests for build_budget_drift_ledger() drift detection."""

    def test_ledger_schema_version(self):
        """Ledger includes schema_version."""
        ledger = build_budget_drift_ledger([])
        assert ledger["schema_version"] == BUDGET_DRIFT_LEDGER_SCHEMA_VERSION

    def test_ledger_empty_snapshots(self):
        """Empty snapshots → no drift, zero counts."""
        ledger = build_budget_drift_ledger([])
        assert ledger["run_count"] == 0
        assert ledger["consecutive_invalids"] == 0
        assert ledger["has_concerning_drift"] is False
        assert ledger["per_slice_health"] == {}

    def test_ledger_single_snapshot_no_drift(self):
        """Single snapshot cannot have drift."""
        snapshots = [
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.02, "classification": "SAFE", "slice_name": "s1"}
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["run_count"] == 1
        assert ledger["exhaustion_rate_drift"]["has_drift"] is False
        assert ledger["asymmetry_drift"]["has_drift"] is False

    def test_ledger_detects_increasing_exhaustion_drift(self):
        """Detects drift when exhaustion rate increases beyond threshold."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.08, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.12, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["exhaustion_rate_drift"]["has_drift"] is True
        assert ledger["exhaustion_rate_drift"]["direction"] == "increasing"
        assert ledger["exhaustion_rate_drift"]["delta"] == pytest.approx(0.07)

    def test_ledger_detects_decreasing_exhaustion_drift(self):
        """Detects drift when exhaustion rate decreases beyond threshold."""
        snapshots = [
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["exhaustion_rate_drift"]["has_drift"] is True
        assert ledger["exhaustion_rate_drift"]["direction"] == "decreasing"

    def test_ledger_no_drift_below_threshold(self):
        """No drift when change is below threshold."""
        snapshots = [
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.11, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.12, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        # 2% change is below 5% threshold
        assert ledger["exhaustion_rate_drift"]["has_drift"] is False
        assert ledger["exhaustion_rate_drift"]["direction"] == "stable"

    def test_ledger_detects_asymmetry_drift(self):
        """Detects drift in asymmetry scores."""
        snapshots = [
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.03, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.06, "classification": "SUSPICIOUS", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["asymmetry_drift"]["has_drift"] is True
        assert ledger["asymmetry_drift"]["direction"] == "increasing"

    def test_ledger_counts_consecutive_invalids(self):
        """Counts maximum consecutive INVALID classifications."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["consecutive_invalids"] == 2  # Max consecutive is 2

    def test_ledger_concerning_drift_from_increasing_exhaustion(self):
        """has_concerning_drift is True when exhaustion increases."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["has_concerning_drift"] is True

    def test_ledger_concerning_drift_from_repeated_invalids(self):
        """has_concerning_drift is True with repeated INVALIDs."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        assert ledger["consecutive_invalids"] >= REPEATED_INVALID_THRESHOLD
        assert ledger["has_concerning_drift"] is True

    def test_ledger_per_slice_health(self):
        """Tracks per-slice health correctly."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "slice_a"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "slice_b"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "slice_a"},
        ]
        ledger = build_budget_drift_ledger(snapshots)

        assert "slice_a" in ledger["per_slice_health"]
        assert "slice_b" in ledger["per_slice_health"]

        assert ledger["per_slice_health"]["slice_a"]["run_count"] == 2
        assert ledger["per_slice_health"]["slice_a"]["last_classification"] == "SAFE"

        assert ledger["per_slice_health"]["slice_b"]["run_count"] == 1
        assert ledger["per_slice_health"]["slice_b"]["last_classification"] == "INVALID"

    def test_ledger_determinism(self):
        """Same snapshots → same ledger."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.10, "asymmetry_score": 0.02, "classification": "SAFE", "slice_name": "s2"},
        ]
        ledger1 = build_budget_drift_ledger(snapshots)
        ledger2 = build_budget_drift_ledger(snapshots)

        json1 = json.dumps(ledger1, sort_keys=True)
        json2 = json.dumps(ledger2, sort_keys=True)
        assert json1 == json2

    def test_ledger_json_serializable(self):
        """Ledger is fully JSON-serializable."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)

        json_str = json.dumps(ledger)
        loaded = json.loads(json_str)
        assert loaded == ledger


# =============================================================================
# BUDGET SENTINEL GRID v3 — Phase III Uplift Gate Tests
# =============================================================================

class TestUpliftBudgetGate:
    """Tests for evaluate_budget_for_uplift() gate decisions."""

    def test_gate_ok_for_healthy_snapshot(self):
        """OK gate for healthy SAFE snapshot."""
        snapshot = {
            "classification": "SAFE",
            "max_exhaustion_rate": 0.05,
            "asymmetry_score": 0.02,
            "exhaustion_rate_baseline": 0.05,
            "exhaustion_rate_rfl": 0.05,
            "A_flags": {"a1_bounded_rate": True, "a2_symmetry": True, "a3_independence": True, "a4_cycle_coverage": True},
            "R_flags": {"r1_excessive_rate": False, "r2_severe_asymmetry": False, "r3_systematic_bias": False, "r4_complete_failures": False},
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "OK"
        assert result["can_proceed"] is True
        assert result["requires_sensitivity_analysis"] is False

    def test_gate_warn_for_suspicious_snapshot(self):
        """WARN gate for SUSPICIOUS snapshot."""
        snapshot = {
            "classification": "SUSPICIOUS",
            "max_exhaustion_rate": 0.20,
            "asymmetry_score": 0.03,
            "exhaustion_rate_baseline": 0.20,
            "exhaustion_rate_rfl": 0.20,
            "A_flags": {"a1_bounded_rate": False, "a2_symmetry": True, "a3_independence": True, "a4_cycle_coverage": True},
            "R_flags": {"r1_excessive_rate": False, "r2_severe_asymmetry": False, "r3_systematic_bias": False, "r4_complete_failures": False},
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "WARN"
        assert result["can_proceed"] is True
        assert result["requires_sensitivity_analysis"] is True

    def test_gate_warn_for_borderline_safe(self):
        """WARN gate for SAFE with borderline metrics."""
        snapshot = {
            "classification": "SAFE",
            "max_exhaustion_rate": 0.12,  # >10%, triggers warning
            "asymmetry_score": 0.04,      # >3%, triggers warning
            "exhaustion_rate_baseline": 0.12,
            "exhaustion_rate_rfl": 0.12,
            "A_flags": {"a1_bounded_rate": True, "a2_symmetry": True, "a3_independence": True, "a4_cycle_coverage": True},
            "R_flags": {"r1_excessive_rate": False, "r2_severe_asymmetry": False, "r3_systematic_bias": False, "r4_complete_failures": False},
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "WARN"
        assert result["can_proceed"] is True
        assert "approaching threshold" in str(result["reasons"]) or "elevated" in str(result["reasons"])

    def test_gate_block_for_invalid_snapshot(self):
        """BLOCK gate for INVALID snapshot."""
        snapshot = {
            "classification": "INVALID",
            "max_exhaustion_rate": 0.60,
            "asymmetry_score": 0.05,
            "exhaustion_rate_baseline": 0.60,
            "exhaustion_rate_rfl": 0.60,
            "A_flags": {"a1_bounded_rate": False, "a2_symmetry": True, "a3_independence": True, "a4_cycle_coverage": True},
            "R_flags": {"r1_excessive_rate": True, "r2_severe_asymmetry": False, "r3_systematic_bias": False, "r4_complete_failures": False},
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "BLOCK"
        assert result["can_proceed"] is False
        assert result["requires_sensitivity_analysis"] is False

    def test_gate_block_includes_r_flag_reasons(self):
        """BLOCK gate includes R-flag specific reasons."""
        snapshot = {
            "classification": "INVALID",
            "max_exhaustion_rate": 0.60,
            "asymmetry_score": 0.25,
            "exhaustion_rate_baseline": 0.60,
            "exhaustion_rate_rfl": 0.60,
            "A_flags": {},
            "R_flags": {"r1_excessive_rate": True, "r2_severe_asymmetry": True, "r3_systematic_bias": False, "r4_complete_failures": False},
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "BLOCK"
        assert any("R1:" in r for r in result["reasons"])
        assert any("R2:" in r for r in result["reasons"])

    def test_gate_block_for_unknown_classification(self):
        """BLOCK gate for unknown classification."""
        snapshot = {
            "classification": "UNKNOWN",
            "max_exhaustion_rate": 0.10,
            "asymmetry_score": 0.02,
        }
        result = evaluate_budget_for_uplift(snapshot)

        assert result["gate"] == "BLOCK"
        assert result["can_proceed"] is False

    def test_gate_determinism(self):
        """Same snapshot → same gate decision."""
        snapshot = {
            "classification": "SAFE",
            "max_exhaustion_rate": 0.05,
            "asymmetry_score": 0.02,
            "A_flags": {},
            "R_flags": {},
        }
        result1 = evaluate_budget_for_uplift(snapshot)
        result2 = evaluate_budget_for_uplift(snapshot)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_gate_json_serializable(self):
        """Gate result is JSON-serializable."""
        snapshot = {
            "classification": "SAFE",
            "max_exhaustion_rate": 0.05,
            "asymmetry_score": 0.02,
            "A_flags": {},
            "R_flags": {},
        }
        result = evaluate_budget_for_uplift(snapshot)

        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded == result


# =============================================================================
# BUDGET SENTINEL GRID v3 — Phase III Global Health Tests
# =============================================================================

class TestGlobalHealthSignal:
    """Tests for summarize_budget_for_global_health() global signal."""

    def test_health_schema_version(self):
        """Health signal includes schema_version."""
        ledger = build_budget_drift_ledger([])
        health = summarize_budget_for_global_health(ledger)
        assert health["schema_version"] == BUDGET_GLOBAL_HEALTH_SCHEMA_VERSION

    def test_health_empty_ledger_is_stable(self):
        """Empty ledger → STABLE, ready for uplift."""
        ledger = build_budget_drift_ledger([])
        health = summarize_budget_for_global_health(ledger)

        assert health["budget_stability"] == "STABLE"
        assert health["unsafe_slices"] == []
        assert health["uplift_budget_ready"] is True

    def test_health_stable_with_healthy_snapshots(self):
        """STABLE when all snapshots are healthy."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.06, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.02, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["budget_stability"] == "STABLE"
        assert health["uplift_budget_ready"] is True

    def test_health_degrading_with_drift(self):
        """DEGRADING when drift is detected but not majority unsafe."""
        # With multiple slices, one showing drift but overall not majority unsafe
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.12, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},  # Drift in s1
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        # Has drift but less than majority unsafe → DEGRADING
        assert health["budget_stability"] == "DEGRADING"

    def test_health_degrading_with_unsafe_slice(self):
        """DEGRADING when a slice has INVALID classification."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["budget_stability"] == "DEGRADING"
        assert "s2" in health["unsafe_slices"]

    def test_health_unstable_with_many_consecutive_invalids(self):
        """UNSTABLE with 3+ consecutive INVALIDs."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["budget_stability"] == "UNSTABLE"
        assert health["uplift_budget_ready"] is False

    def test_health_unstable_with_majority_unsafe_slices(self):
        """UNSTABLE when majority of slices are unsafe."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        # 2 out of 3 slices are unsafe (>50%)
        assert health["budget_stability"] == "UNSTABLE"

    def test_health_unsafe_slices_sorted_alphabetically(self):
        """unsafe_slices are sorted alphabetically."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "zebra"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "alpha"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["unsafe_slices"] == ["alpha", "zebra"]

    def test_health_uplift_ready_when_stable_no_unsafe(self):
        """uplift_budget_ready is True when STABLE and no unsafe slices."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["uplift_budget_ready"] is True

    def test_health_uplift_not_ready_with_unsafe_slices(self):
        """uplift_budget_ready is False when unsafe slices exist."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["uplift_budget_ready"] is False

    def test_health_summary_metrics(self):
        """Summary metrics are populated correctly."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        assert health["summary_metrics"]["run_count"] == 2
        assert health["summary_metrics"]["unsafe_slice_count"] == 1

    def test_health_determinism(self):
        """Same ledger → same health signal."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health1 = summarize_budget_for_global_health(ledger)
        health2 = summarize_budget_for_global_health(ledger)

        assert json.dumps(health1, sort_keys=True) == json.dumps(health2, sort_keys=True)

    def test_health_json_serializable(self):
        """Health signal is JSON-serializable."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        health = summarize_budget_for_global_health(ledger)

        json_str = json.dumps(health)
        loaded = json.loads(json_str)
        assert loaded == health


# =============================================================================
# Phase III Threshold Constants Tests
# =============================================================================

class TestPhaseIIIThresholds:
    """Verify Phase III threshold constants."""

    def test_drift_rate_threshold(self):
        """DRIFT_RATE_THRESHOLD = 0.05 (5%)"""
        assert DRIFT_RATE_THRESHOLD == 0.05

    def test_drift_asymmetry_threshold(self):
        """DRIFT_ASYMMETRY_THRESHOLD = 0.03 (3%)"""
        assert DRIFT_ASYMMETRY_THRESHOLD == 0.03

    def test_repeated_invalid_threshold(self):
        """REPEATED_INVALID_THRESHOLD = 2"""
        assert REPEATED_INVALID_THRESHOLD == 2


# =============================================================================
# BUDGET SENTINEL GRID v4 — Phase IV Budget Risk Map Tests
# =============================================================================

class TestBudgetRiskMap:
    """Tests for build_budget_risk_map() cross-slice risk analysis."""

    def test_risk_map_schema_version(self):
        """Risk map includes schema_version."""
        ledger = build_budget_drift_ledger([])
        risk_map = build_budget_risk_map(ledger)
        assert risk_map["schema_version"] == BUDGET_RISK_MAP_SCHEMA_VERSION

    def test_risk_map_empty_ledger_is_low_risk(self):
        """Empty ledger → LOW risk band."""
        ledger = build_budget_drift_ledger([])
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["risk_band"] == "LOW"
        assert risk_map["slices_with_concerning_drift"] == []
        assert risk_map["slices_with_repeated_invalid"] == []

    def test_risk_map_healthy_slices_low_risk(self):
        """Healthy slices → LOW risk band."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["risk_band"] == "LOW"
        assert risk_map["summary"]["problem_slice_count"] == 0

    def test_risk_map_identifies_drift_slices(self):
        """Identifies slices with concerning drift."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift in s1
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert "s1" in risk_map["slices_with_concerning_drift"]
        assert "s2" not in risk_map["slices_with_concerning_drift"]

    def test_risk_map_identifies_repeated_invalid_slices(self):
        """Identifies slices with repeated INVALID classifications."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert "s1" in risk_map["slices_with_repeated_invalid"]
        assert "s2" not in risk_map["slices_with_repeated_invalid"]

    def test_risk_map_medium_with_drift(self):
        """MEDIUM risk with concerning drift but not majority affected."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["risk_band"] == "MEDIUM"

    def test_risk_map_high_with_many_invalids(self):
        """HIGH risk with 3+ consecutive INVALIDs."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["risk_band"] == "HIGH"

    def test_risk_map_high_with_majority_problem_slices(self):
        """HIGH risk when majority slices have issues."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},  # 2 invalids
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},  # 2 invalids
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        # 2 out of 3 slices have problems → HIGH
        assert risk_map["risk_band"] == "HIGH"

    def test_risk_map_slices_sorted_alphabetically(self):
        """Lists are sorted alphabetically."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "zebra"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "zebra"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "zebra"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "alpha"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "alpha"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["slices_with_repeated_invalid"] == ["alpha", "zebra"]

    def test_risk_map_summary_metrics(self):
        """Summary includes correct slice counts."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        assert risk_map["summary"]["total_slices"] == 2
        assert risk_map["summary"]["drift_slice_count"] == 1
        assert risk_map["summary"]["invalid_slice_count"] == 0
        assert risk_map["summary"]["problem_slice_count"] == 1

    def test_risk_map_determinism(self):
        """Same ledger → same risk map."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk1 = build_budget_risk_map(ledger)
        risk2 = build_budget_risk_map(ledger)

        assert json.dumps(risk1, sort_keys=True) == json.dumps(risk2, sort_keys=True)

    def test_risk_map_json_serializable(self):
        """Risk map is JSON-serializable."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)

        json_str = json.dumps(risk_map)
        loaded = json.loads(json_str)
        assert loaded == risk_map


# =============================================================================
# BUDGET SENTINEL GRID v4 — Phase IV Joint Uplift Adapter Tests
# =============================================================================

class TestJointUpliftAdapter:
    """Tests for summarize_budget_and_metrics_for_uplift() joint adapter."""

    def test_joint_uplift_schema_version(self):
        """Joint summary includes schema_version."""
        ledger = build_budget_drift_ledger([])
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)
        assert result["schema_version"] == BUDGET_METRIC_UPLIFT_SCHEMA_VERSION

    def test_joint_uplift_ok_when_all_healthy(self):
        """OK status when budget and metrics are healthy."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        assert result["status"] == "OK"
        assert result["uplift_ready"] is True
        assert result["blocking_slices"] == []

    def test_joint_uplift_warn_with_medium_risk(self):
        """WARN status with MEDIUM budget risk."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        assert result["status"] == "WARN"
        assert result["uplift_ready"] is True
        assert "Budget risk band is MEDIUM" in result["notes"]

    def test_joint_uplift_block_with_high_risk(self):
        """BLOCK status with HIGH budget risk."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        assert result["status"] == "BLOCK"
        assert result["uplift_ready"] is False
        assert "Budget risk band is HIGH" in result["notes"]

    def test_joint_uplift_block_when_metrics_not_ready(self):
        """BLOCK status when metrics are not ready."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": False, "blocking_metrics": ["delta_p", "significance"]}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        assert result["status"] == "BLOCK"
        assert result["uplift_ready"] is False
        assert "Metrics are not ready for uplift" in result["notes"]
        assert any("Blocking metrics" in n for n in result["notes"])

    def test_joint_uplift_blocking_slices_combined(self):
        """Blocking slices combines drift and invalid slices."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s2"},  # Repeated invalid
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        # Both s1 (drift) and s2 (invalid) should be in blocking slices
        assert "s1" in result["blocking_slices"]
        assert "s2" in result["blocking_slices"]
        # Sorted alphabetically
        assert result["blocking_slices"] == ["s1", "s2"]

    def test_joint_uplift_notes_are_neutral(self):
        """Notes use neutral language (no 'good'/'bad')."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        for note in result["notes"]:
            assert "good" not in note.lower()
            assert "bad" not in note.lower()
            assert "great" not in note.lower()
            assert "terrible" not in note.lower()

    def test_joint_uplift_determinism(self):
        """Same inputs → same output."""
        ledger = build_budget_drift_ledger([])
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result1 = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)
        result2 = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_joint_uplift_json_serializable(self):
        """Result is JSON-serializable."""
        ledger = build_budget_drift_ledger([])
        risk_map = build_budget_risk_map(ledger)
        metric_summary = {"ready": True, "blocking_metrics": []}

        result = summarize_budget_and_metrics_for_uplift(risk_map, metric_summary)

        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded == result


# =============================================================================
# BUDGET SENTINEL GRID v4 — Phase IV Director Budget Panel Tests
# =============================================================================

class TestDirectorBudgetPanel:
    """Tests for build_budget_director_panel() director-level dashboard."""

    def test_panel_schema_version(self):
        """Director panel includes schema_version."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)
        assert panel["schema_version"] == BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION

    def test_panel_green_when_healthy(self):
        """GREEN status light when stable and low risk."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["status_light"] == "GREEN"
        assert panel["budget_stability"] == "STABLE"
        assert panel["uplift_budget_ready"] is True

    def test_panel_yellow_with_degrading_stability(self):
        """YELLOW status light when stability is DEGRADING."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["status_light"] == "YELLOW"

    def test_panel_yellow_with_medium_risk(self):
        """YELLOW status light when risk is MEDIUM."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["status_light"] == "YELLOW"
        assert risk_map["risk_band"] == "MEDIUM"

    def test_panel_red_with_unstable_budget(self):
        """RED status light when stability is UNSTABLE."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["status_light"] == "RED"
        assert panel["budget_stability"] == "UNSTABLE"

    def test_panel_red_with_high_risk(self):
        """RED status light when risk band is HIGH."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["status_light"] == "RED"
        assert risk_map["risk_band"] == "HIGH"

    def test_panel_headline_is_neutral(self):
        """Headline uses neutral, non-interpretive language."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        headline = panel["headline"].lower()
        assert "good" not in headline
        assert "bad" not in headline
        assert "excellent" not in headline
        assert "terrible" not in headline
        assert "great" not in headline

    def test_panel_headline_green(self):
        """GREEN headline mentions expected bounds."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert "expected bounds" in panel["headline"].lower() or "within" in panel["headline"].lower()

    def test_panel_headline_yellow_with_problems(self):
        """YELLOW headline mentions slices requiring attention."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s2"},
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s3"},
            {"max_exhaustion_rate": 0.15, "asymmetry_score": 0.01, "classification": "SUSPICIOUS", "slice_name": "s1"},  # Drift
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        headline = panel["headline"].lower()
        assert "attention" in headline or "monitoring" in headline or "require" in headline

    def test_panel_headline_red_mentions_gating(self):
        """RED headline mentions gating or review."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        headline = panel["headline"].lower()
        assert "gating" in headline or "review" in headline or "exceeded" in headline or "outside" in headline

    def test_panel_details_populated(self):
        """Details section has expected fields."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert "run_count" in panel["details"]
        assert "unsafe_slice_count" in panel["details"]
        assert "risk_band" in panel["details"]
        assert "has_drift" in panel["details"]
        assert "consecutive_invalids" in panel["details"]

    def test_panel_uplift_ready_from_global_health(self):
        """uplift_budget_ready comes from global health."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        assert panel["uplift_budget_ready"] == global_health["uplift_budget_ready"]

    def test_panel_determinism(self):
        """Same inputs → same panel."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel1 = build_budget_director_panel(global_health, risk_map)
        panel2 = build_budget_director_panel(global_health, risk_map)

        assert json.dumps(panel1, sort_keys=True) == json.dumps(panel2, sort_keys=True)

    def test_panel_json_serializable(self):
        """Panel is JSON-serializable."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel(global_health, risk_map)

        json_str = json.dumps(panel)
        loaded = json.loads(json_str)
        assert loaded == panel


# =============================================================================
# Phase IV Schema Version Tests
# =============================================================================

class TestPhaseIVSchemaVersions:
    """Verify Phase IV schema version constants."""

    def test_budget_risk_map_schema_version(self):
        """BUDGET_RISK_MAP_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_RISK_MAP_SCHEMA_VERSION == "1.0.0"

    def test_budget_metric_uplift_schema_version(self):
        """BUDGET_METRIC_UPLIFT_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_METRIC_UPLIFT_SCHEMA_VERSION == "1.0.0"

    def test_budget_director_panel_schema_version(self):
        """BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_DIRECTOR_PANEL_SCHEMA_VERSION == "1.0.0"


# =============================================================================
# Phase IV Enum Tests
# =============================================================================

class TestPhaseIVEnums:
    """Verify Phase IV enum values."""

    def test_risk_band_values(self):
        """RiskBand has LOW, MEDIUM, HIGH values."""
        assert RiskBand.LOW.value == "LOW"
        assert RiskBand.MEDIUM.value == "MEDIUM"
        assert RiskBand.HIGH.value == "HIGH"

    def test_joint_uplift_status_values(self):
        """JointUpliftStatus has OK, WARN, BLOCK values."""
        assert JointUpliftStatus.OK.value == "OK"
        assert JointUpliftStatus.WARN.value == "WARN"
        assert JointUpliftStatus.BLOCK.value == "BLOCK"

    def test_status_light_values(self):
        """StatusLight has GREEN, YELLOW, RED values."""
        assert StatusLight.GREEN.value == "GREEN"
        assert StatusLight.YELLOW.value == "YELLOW"
        assert StatusLight.RED.value == "RED"


# =============================================================================
# BUDGET SENTINEL GRID v5 — Phase V Budget Risk Trajectory Tests
# =============================================================================

class TestBudgetRiskTrajectory:
    """Tests for build_budget_risk_trajectory() temporal tracking."""

    def test_trajectory_schema_version(self):
        """Trajectory includes schema_version."""
        trajectory = build_budget_risk_trajectory([])
        assert trajectory["schema_version"] == BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION

    def test_trajectory_empty_history(self):
        """Empty history → UNKNOWN trend, zero counts."""
        trajectory = build_budget_risk_trajectory([])
        assert trajectory["run_count"] == 0
        assert trajectory["risk_band_series"] == []
        assert trajectory["trend"] == "UNKNOWN"
        assert trajectory["invalid_run_streaks"]["max_streak"] == 0

    def test_trajectory_single_run_unknown_trend(self):
        """Single run → UNKNOWN trend (insufficient data)."""
        history = [{"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["run_count"] == 1
        assert trajectory["trend"] == "UNKNOWN"

    def test_trajectory_improving_trend(self):
        """Detects IMPROVING trend when risk decreases over time."""
        history = [
            {"classification": "INVALID"},  # HIGH
            {"classification": "INVALID"},  # HIGH
            {"classification": "SUSPICIOUS"},  # MEDIUM
            {"classification": "SAFE"},  # LOW
            {"classification": "SAFE"},  # LOW
            {"classification": "SAFE"},  # LOW
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["trend"] == "IMPROVING"
        assert trajectory["summary"]["improving_runs"] > 0

    def test_trajectory_degrading_trend(self):
        """Detects DEGRADING trend when risk increases over time."""
        history = [
            {"classification": "SAFE"},  # LOW
            {"classification": "SAFE"},  # LOW
            {"classification": "SUSPICIOUS"},  # MEDIUM
            {"classification": "SUSPICIOUS"},  # MEDIUM
            {"classification": "INVALID"},  # HIGH
            {"classification": "INVALID"},  # HIGH
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["trend"] == "DEGRADING"
        assert trajectory["summary"]["degrading_runs"] > 0

    def test_trajectory_stable_trend(self):
        """Detects STABLE trend when risk is consistent."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["trend"] == "STABLE"
        assert trajectory["summary"]["stable_runs"] >= 2

    def test_trajectory_risk_band_series(self):
        """Builds correct risk_band_series from classifications."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SUSPICIOUS"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["risk_band_series"] == ["LOW", "MEDIUM", "HIGH"]

    def test_trajectory_uses_explicit_risk_band(self):
        """Uses explicit risk_band if provided."""
        history = [
            {"classification": "SAFE", "risk_band": "MEDIUM"},  # Override
            {"classification": "INVALID", "risk_band": "LOW"},  # Override
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["risk_band_series"] == ["MEDIUM", "LOW"]

    def test_trajectory_invalid_streak_single(self):
        """Counts single INVALID streak correctly."""
        history = [
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        streaks = trajectory["invalid_run_streaks"]
        assert streaks["max_streak"] == 3
        assert streaks["streak_count"] == 1
        assert streaks["total_invalids"] == 3

    def test_trajectory_invalid_streak_multiple(self):
        """Counts multiple INVALID streaks correctly."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        streaks = trajectory["invalid_run_streaks"]
        assert streaks["max_streak"] == 3
        assert streaks["streak_count"] == 2
        assert streaks["total_invalids"] == 5

    def test_trajectory_chronic_invalid_streak(self):
        """Detects chronic (3+) INVALID streaks."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        assert trajectory["invalid_run_streaks"]["max_streak"] == 4

    def test_trajectory_determinism(self):
        """Same history → same trajectory."""
        history = [
            {"classification": "SAFE"},
            {"classification": "INVALID"},
        ]
        traj1 = build_budget_risk_trajectory(history)
        traj2 = build_budget_risk_trajectory(history)
        assert json.dumps(traj1, sort_keys=True) == json.dumps(traj2, sort_keys=True)

    def test_trajectory_json_serializable(self):
        """Trajectory is JSON-serializable."""
        history = [{"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        json_str = json.dumps(trajectory)
        loaded = json.loads(json_str)
        assert loaded == trajectory


# =============================================================================
# BUDGET SENTINEL GRID v5 — Phase V Policy Impact Tests
# =============================================================================

class TestBudgetPolicyImpact:
    """Tests for summarize_budget_impact_on_policy() correlation detection."""

    def test_policy_impact_schema_version(self):
        """Policy impact includes schema_version."""
        trajectory = build_budget_risk_trajectory([])
        policy_radar = {"slices_with_drift": [], "high_drift_slices": []}
        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)
        assert impact["schema_version"] == BUDGET_POLICY_IMPACT_SCHEMA_VERSION

    def test_policy_impact_no_correlation(self):
        """No correlation when budget and policy are both healthy."""
        history = [{"classification": "SAFE"}, {"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {"slices_with_drift": [], "high_drift_slices": []}

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert impact["correlation_strength"] == "NONE"
        assert impact["suspected_budget_limited_learning"] is False
        assert impact["correlated_slices"] == []

    def test_policy_impact_correlated_slices(self):
        """Detects correlated slices with both budget risk and policy drift."""
        history = [{"classification": "INVALID"}, {"classification": "INVALID"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {
            "slices_with_drift": ["s1", "s2"],
            "high_drift_slices": ["s1", "s2"],
            "budget_high_risk_slices": ["s1", "s3"],  # s1 is in both
        }

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert "s1" in impact["correlated_slices"]
        assert impact["suspected_budget_limited_learning"] is True
        assert impact["correlation_strength"] == "MODERATE"

    def test_policy_impact_systemic_correlation(self):
        """Detects systemic correlation without explicit slice overlap."""
        # History with >30% high risk runs
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {
            "slices_with_drift": ["s1"],
            "high_drift_slices": ["s1"],  # Has high drift
        }

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        # Should detect systemic correlation (>30% HIGH + has high drift)
        assert impact["correlation_strength"] in ("MODERATE", "WEAK")

    def test_policy_impact_chronic_invalid_triggers_learning_flag(self):
        """Chronic INVALID streak + high drift → suspected limited learning."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {
            "slices_with_drift": ["s1"],
            "high_drift_slices": ["s1"],
        }

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert impact["suspected_budget_limited_learning"] is True
        assert any("Chronic" in n for n in impact["notes"])

    def test_policy_impact_strong_correlation(self):
        """Strong correlation with many correlated slices."""
        history = [{"classification": "INVALID"}, {"classification": "INVALID"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {
            "slices_with_drift": ["s1", "s2", "s3", "s4"],
            "high_drift_slices": ["s1", "s2", "s3"],
            "budget_high_risk_slices": ["s1", "s2", "s3"],  # All 3 overlap
        }

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert impact["correlation_strength"] == "STRONG"
        assert len(impact["correlated_slices"]) == 3

    def test_policy_impact_weak_correlation(self):
        """Weak correlation with repeated INVALIDs and some drift."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {
            "slices_with_drift": ["s1"],  # Has drift
            "high_drift_slices": [],  # No HIGH drift
        }

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert impact["correlation_strength"] == "WEAK"

    def test_policy_impact_notes_include_trend(self):
        """Notes include trend information when DEGRADING."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SUSPICIOUS"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {"slices_with_drift": [], "high_drift_slices": []}

        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert any("degrading" in n.lower() for n in impact["notes"])

    def test_policy_impact_determinism(self):
        """Same inputs → same output."""
        history = [{"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_radar = {"slices_with_drift": [], "high_drift_slices": []}

        impact1 = summarize_budget_impact_on_policy(trajectory, policy_radar)
        impact2 = summarize_budget_impact_on_policy(trajectory, policy_radar)

        assert json.dumps(impact1, sort_keys=True) == json.dumps(impact2, sort_keys=True)

    def test_policy_impact_json_serializable(self):
        """Impact result is JSON-serializable."""
        trajectory = build_budget_risk_trajectory([])
        policy_radar = {"slices_with_drift": [], "high_drift_slices": []}
        impact = summarize_budget_impact_on_policy(trajectory, policy_radar)

        json_str = json.dumps(impact)
        loaded = json.loads(json_str)
        assert loaded == impact


# =============================================================================
# BUDGET SENTINEL GRID v5 — Phase V Extended Director Panel Tests
# =============================================================================

class TestDirectorPanelV2:
    """Tests for build_budget_director_panel_v2() extended panel."""

    def test_panel_v2_schema_version(self):
        """Panel v2 includes schema_version."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel_v2(global_health, risk_map)
        assert panel["schema_version"] == BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION

    def test_panel_v2_includes_trend(self):
        """Panel v2 includes trend field."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel_v2(global_health, risk_map)
        assert "trend" in panel
        assert panel["trend"] == "UNKNOWN"  # No trajectory provided

    def test_panel_v2_trend_from_trajectory(self):
        """Panel v2 extracts trend from trajectory."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)

        panel = build_budget_director_panel_v2(global_health, risk_map, budget_trajectory=trajectory)
        assert panel["trend"] == "IMPROVING"

    def test_panel_v2_includes_invalid_streaks_max(self):
        """Panel v2 includes invalid_streaks_max field."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel_v2(global_health, risk_map)
        assert "invalid_streaks_max" in panel
        assert panel["invalid_streaks_max"] == 0

    def test_panel_v2_invalid_streaks_from_trajectory(self):
        """Panel v2 extracts invalid_streaks_max from trajectory."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)

        panel = build_budget_director_panel_v2(global_health, risk_map, budget_trajectory=trajectory)
        assert panel["invalid_streaks_max"] == 3

    def test_panel_v2_no_correlation_flag_without_policy(self):
        """Panel v2 omits correlation flag when policy impact not provided."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel_v2(global_health, risk_map)
        assert "budget_policy_correlation_flag" not in panel

    def test_panel_v2_includes_correlation_flag(self):
        """Panel v2 includes correlation flag from policy impact."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        policy_impact = {
            "suspected_budget_limited_learning": True,
            "correlated_slices": ["s1"],
        }

        panel = build_budget_director_panel_v2(
            global_health, risk_map, budget_policy_impact=policy_impact
        )
        assert panel["budget_policy_correlation_flag"] is True

    def test_panel_v2_correlation_flag_false(self):
        """Panel v2 sets correlation flag to False when no correlation."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        policy_impact = {
            "suspected_budget_limited_learning": False,
            "correlated_slices": [],
        }

        panel = build_budget_director_panel_v2(
            global_health, risk_map, budget_policy_impact=policy_impact
        )
        assert panel["budget_policy_correlation_flag"] is False

    def test_panel_v2_yellow_on_degrading_trend_with_invalids(self):
        """Panel v2 upgrades to YELLOW when degrading with repeated INVALIDs."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        # Degrading trend with 2+ invalids
        history = [
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)

        panel = build_budget_director_panel_v2(global_health, risk_map, budget_trajectory=trajectory)
        assert panel["status_light"] == "YELLOW"

    def test_panel_v2_headline_mentions_improving(self):
        """Panel v2 headline mentions improving trend when applicable."""
        snapshots = [
            {"max_exhaustion_rate": 0.05, "asymmetry_score": 0.01, "classification": "SAFE", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)

        panel = build_budget_director_panel_v2(global_health, risk_map, budget_trajectory=trajectory)
        assert "improving" in panel["headline"].lower()

    def test_panel_v2_headline_chronic_invalids(self):
        """Panel v2 headline mentions chronic INVALIDs in RED state."""
        snapshots = [
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
            {"max_exhaustion_rate": 0.60, "asymmetry_score": 0.01, "classification": "INVALID", "slice_name": "s1"},
        ]
        ledger = build_budget_drift_ledger(snapshots)
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)

        panel = build_budget_director_panel_v2(global_health, risk_map, budget_trajectory=trajectory)
        assert panel["status_light"] == "RED"
        assert "chronic" in panel["headline"].lower() or "consecutive" in panel["headline"].lower()

    def test_panel_v2_determinism(self):
        """Same inputs → same panel."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel1 = build_budget_director_panel_v2(global_health, risk_map)
        panel2 = build_budget_director_panel_v2(global_health, risk_map)

        assert json.dumps(panel1, sort_keys=True) == json.dumps(panel2, sort_keys=True)

    def test_panel_v2_json_serializable(self):
        """Panel v2 is JSON-serializable."""
        ledger = build_budget_drift_ledger([])
        global_health = summarize_budget_for_global_health(ledger)
        risk_map = build_budget_risk_map(ledger)

        panel = build_budget_director_panel_v2(global_health, risk_map)

        json_str = json.dumps(panel)
        loaded = json.loads(json_str)
        assert loaded == panel


# =============================================================================
# Phase V Schema Version Tests
# =============================================================================

class TestPhaseVSchemaVersions:
    """Verify Phase V schema version constants."""

    def test_budget_risk_trajectory_schema_version(self):
        """BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_RISK_TRAJECTORY_SCHEMA_VERSION == "1.0.0"

    def test_budget_policy_impact_schema_version(self):
        """BUDGET_POLICY_IMPACT_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_POLICY_IMPACT_SCHEMA_VERSION == "1.0.0"

    def test_budget_director_panel_v2_schema_version(self):
        """BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_DIRECTOR_PANEL_V2_SCHEMA_VERSION == "1.0.0"


# =============================================================================
# Phase V Enum Tests
# =============================================================================

class TestPhaseVEnums:
    """Verify Phase V enum values."""

    def test_risk_trend_values(self):
        """RiskTrend has IMPROVING, STABLE, DEGRADING, UNKNOWN values."""
        assert RiskTrend.IMPROVING.value == "IMPROVING"
        assert RiskTrend.STABLE.value == "STABLE"
        assert RiskTrend.DEGRADING.value == "DEGRADING"
        assert RiskTrend.UNKNOWN.value == "UNKNOWN"


# =============================================================================
# BUDGET SENTINEL GRID v5 — Global Console Adapter Tests
# =============================================================================

class TestGlobalConsoleAdapter:
    """Tests for summarize_budget_trajectory_for_global_console() adapter."""

    def test_global_console_schema_version(self):
        """Global console summary includes schema_version."""
        trajectory = build_budget_risk_trajectory([])
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["schema_version"] == BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION

    def test_global_console_budget_ok_green(self):
        """budget_ok is True when status_light is GREEN."""
        trajectory = build_budget_risk_trajectory([{"classification": "SAFE"}])
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["budget_ok"] is True

    def test_global_console_budget_ok_yellow_with_uplift_ready(self):
        """budget_ok is True when YELLOW but uplift_budget_ready."""
        trajectory = build_budget_risk_trajectory([{"classification": "SUSPICIOUS"}])
        panel = {"status_light": "YELLOW", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["budget_ok"] is True

    def test_global_console_budget_not_ok_yellow_blocked(self):
        """budget_ok is False when YELLOW and uplift not ready."""
        trajectory = build_budget_risk_trajectory([{"classification": "SUSPICIOUS"}])
        panel = {"status_light": "YELLOW", "uplift_budget_ready": False}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["budget_ok"] is False

    def test_global_console_budget_not_ok_red(self):
        """budget_ok is False when status_light is RED."""
        trajectory = build_budget_risk_trajectory([{"classification": "INVALID"}])
        panel = {"status_light": "RED", "uplift_budget_ready": False}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["budget_ok"] is False

    def test_global_console_includes_status_light(self):
        """Result includes status_light from panel."""
        trajectory = build_budget_risk_trajectory([])
        panel = {"status_light": "YELLOW", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["status_light"] == "YELLOW"

    def test_global_console_includes_trend(self):
        """Result includes trend from trajectory."""
        history = [
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["trend"] == "IMPROVING"

    def test_global_console_includes_invalid_streaks_max(self):
        """Result includes invalid_run_streaks_max from trajectory."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["invalid_run_streaks_max"] == 3

    def test_global_console_summary_has_chronic_flag(self):
        """Summary includes has_chronic_invalids flag."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        panel = {"status_light": "RED", "uplift_budget_ready": False}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["summary"]["has_chronic_invalids"] is True

    def test_global_console_summary_no_chronic_flag(self):
        """Summary has_chronic_invalids is False for low streak."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        panel = {"status_light": "YELLOW", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert result["summary"]["has_chronic_invalids"] is False

    def test_global_console_determinism(self):
        """Same inputs → same output."""
        trajectory = build_budget_risk_trajectory([{"classification": "SAFE"}])
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result1 = summarize_budget_trajectory_for_global_console(trajectory, panel)
        result2 = summarize_budget_trajectory_for_global_console(trajectory, panel)
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_global_console_json_serializable(self):
        """Result is JSON-serializable."""
        trajectory = build_budget_risk_trajectory([])
        panel = {"status_light": "GREEN", "uplift_budget_ready": True}
        result = summarize_budget_trajectory_for_global_console(trajectory, panel)
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded == result


# =============================================================================
# BUDGET SENTINEL GRID v5 — Governance Signal Adapter Tests
# =============================================================================

class TestGovernanceSignalAdapter:
    """Tests for to_governance_signal_for_budget() adapter."""

    def test_governance_signal_schema_version(self):
        """Governance signal includes schema_version."""
        trajectory = build_budget_risk_trajectory([])
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["schema_version"] == BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION

    def test_governance_ok_on_improving_trend(self):
        """OK signal when trend is IMPROVING with low invalid rate."""
        # Need more SAFE runs to keep invalid rate below 30%
        history = [
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "OK"
        assert result["can_proceed"] is True
        assert result["requires_review"] is False

    def test_governance_ok_on_stable_no_issues(self):
        """OK signal when STABLE with no concerning patterns."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "OK"

    def test_governance_warn_on_stable_with_invalids(self):
        """WARN signal when STABLE but has repeated INVALIDs."""
        history = [
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "WARN"
        assert result["can_proceed"] is True
        assert result["requires_review"] is True

    def test_governance_warn_on_degrading_no_chronic(self):
        """WARN signal when DEGRADING without chronic INVALIDs."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "SUSPICIOUS"},
            {"classification": "SUSPICIOUS"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "WARN"
        assert any("DEGRADING" in r for r in result["reasons"])

    def test_governance_warn_on_suspected_limited_learning(self):
        """WARN signal when suspected budget-limited learning."""
        history = [{"classification": "SAFE"}, {"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": True, "correlation_strength": "WEAK"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "WARN"
        assert any("limiting" in r.lower() for r in result["reasons"])

    def test_governance_warn_on_moderate_correlation(self):
        """WARN signal on moderate budget-policy correlation."""
        history = [{"classification": "SAFE"}, {"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "MODERATE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "WARN"
        assert any("correlation" in r.lower() for r in result["reasons"])

    def test_governance_warn_on_high_invalid_rate(self):
        """WARN signal when INVALID rate exceeds 30%."""
        # Create a STABLE trend with high invalid rate (no consecutive invalids, balanced)
        # Use a balanced pattern where first and second half have equal invalids
        history = [
            {"classification": "INVALID"},
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "SAFE"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        # 2/4 = 50% > 30%, max_streak=1 < 2, trend should be STABLE
        assert trajectory["trend"] == "STABLE"
        assert result["signal"] == "WARN"
        assert any("rate" in r.lower() for r in result["reasons"])

    def test_governance_block_on_degrading_with_chronic_invalids(self):
        """BLOCK signal when DEGRADING + chronic INVALID streaks."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SAFE"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "BLOCK"
        assert result["can_proceed"] is False
        assert result["requires_review"] is True

    def test_governance_block_on_degrading_with_limited_learning(self):
        """BLOCK signal when DEGRADING + suspected limited learning."""
        history = [
            {"classification": "SAFE"},
            {"classification": "SUSPICIOUS"},
            {"classification": "SUSPICIOUS"},
            {"classification": "SUSPICIOUS"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": True, "correlation_strength": "MODERATE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "BLOCK"

    def test_governance_block_on_severe_invalid_streak(self):
        """BLOCK signal when very long INVALID streak (4+)."""
        history = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
        ]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "BLOCK"
        assert any("severe" in r.lower() or "4" in r for r in result["reasons"])

    def test_governance_block_on_strong_correlation(self):
        """BLOCK signal on strong budget-policy correlation."""
        history = [{"classification": "SAFE"}, {"classification": "SAFE"}]
        trajectory = build_budget_risk_trajectory(history)
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "STRONG"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "BLOCK"
        assert any("strong" in r.lower() for r in result["reasons"])

    def test_governance_ok_on_unknown_no_issues(self):
        """OK signal when UNKNOWN trend with no issues."""
        trajectory = build_budget_risk_trajectory([{"classification": "SAFE"}])
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        assert result["signal"] == "OK"

    def test_governance_determinism(self):
        """Same inputs → same output."""
        trajectory = build_budget_risk_trajectory([{"classification": "SAFE"}])
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result1 = to_governance_signal_for_budget(trajectory, policy_impact)
        result2 = to_governance_signal_for_budget(trajectory, policy_impact)
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_governance_json_serializable(self):
        """Result is JSON-serializable."""
        trajectory = build_budget_risk_trajectory([])
        policy_impact = {"suspected_budget_limited_learning": False, "correlation_strength": "NONE"}
        result = to_governance_signal_for_budget(trajectory, policy_impact)
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded == result


# =============================================================================
# Phase V Adapter Schema Version Tests
# =============================================================================

class TestPhaseVAdapterSchemaVersions:
    """Verify Phase V adapter schema version constants."""

    def test_global_console_schema_version(self):
        """BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_GLOBAL_CONSOLE_SCHEMA_VERSION == "1.0.0"

    def test_governance_signal_schema_version(self):
        """BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION = 1.0.0"""
        assert BUDGET_GOVERNANCE_SIGNAL_SCHEMA_VERSION == "1.0.0"


# =============================================================================
# Phase V Adapter Enum Tests
# =============================================================================

class TestPhaseVAdapterEnums:
    """Verify Phase V adapter enum values."""

    def test_governance_signal_values(self):
        """GovernanceSignal has OK, WARN, BLOCK values."""
        assert GovernanceSignal.OK.value == "OK"
        assert GovernanceSignal.WARN.value == "WARN"
        assert GovernanceSignal.BLOCK.value == "BLOCK"
