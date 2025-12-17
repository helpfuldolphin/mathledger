"""
Tests for Phase X Budget Binding: P3/P4 Integration

Tests cover:
1. Deterministic mapping from drift_class → noise coefficient
2. Deterministic mapping from stability_class → severity multiplier
3. JSON serialization of BudgetRiskSignal and related structures
4. P3 BudgetAwareDeltaPMetrics integration
5. P4 BudgetAwareDivergenceSummary integration

See: docs/system_law/Budget_PhaseX_Doctrine.md
"""

import json
import pytest

from backend.topology.first_light.budget_binding import (
    BudgetDriftClass,
    BudgetStabilityClass,
    BudgetRiskSignal,
    NOISE_MULTIPLIER_TABLE,
    SEVERITY_MULTIPLIER_TABLE,
    compute_noise_multiplier,
    compute_noise_multiplier_continuous,
    compute_severity_multiplier,
    compute_severity_multiplier_from_health,
    compute_rsi_correction_factor,
    drift_class_from_value,
    stability_class_from_health,
    build_budget_risk_signal,
    extend_stability_report_with_budget,
    adjust_divergence_severity,
    compute_tda_context,
    # Task 1-3 functions
    build_budget_risk_summary_for_p3,
    build_budget_context_for_p4,
    attach_budget_risk_to_evidence,
)


# =============================================================================
# Test: Drift Class Classification
# =============================================================================

class TestDriftClassification:
    """Tests for drift_class_from_value() mapping."""

    def test_stable_threshold(self):
        """STABLE: |drift| <= 0.05"""
        assert drift_class_from_value(0.0) == BudgetDriftClass.STABLE
        assert drift_class_from_value(0.05) == BudgetDriftClass.STABLE
        assert drift_class_from_value(-0.05) == BudgetDriftClass.STABLE
        assert drift_class_from_value(0.01) == BudgetDriftClass.STABLE

    def test_drifting_threshold(self):
        """DRIFTING: 0.05 < |drift| <= 0.15"""
        assert drift_class_from_value(0.051) == BudgetDriftClass.DRIFTING
        assert drift_class_from_value(0.15) == BudgetDriftClass.DRIFTING
        assert drift_class_from_value(-0.10) == BudgetDriftClass.DRIFTING
        assert drift_class_from_value(0.08) == BudgetDriftClass.DRIFTING

    def test_diverging_threshold(self):
        """DIVERGING: 0.15 < |drift| <= 0.25"""
        assert drift_class_from_value(0.151) == BudgetDriftClass.DIVERGING
        assert drift_class_from_value(0.25) == BudgetDriftClass.DIVERGING
        assert drift_class_from_value(-0.20) == BudgetDriftClass.DIVERGING
        assert drift_class_from_value(0.18) == BudgetDriftClass.DIVERGING

    def test_critical_threshold(self):
        """CRITICAL: |drift| > 0.25"""
        assert drift_class_from_value(0.251) == BudgetDriftClass.CRITICAL
        assert drift_class_from_value(0.5) == BudgetDriftClass.CRITICAL
        assert drift_class_from_value(-0.30) == BudgetDriftClass.CRITICAL
        assert drift_class_from_value(1.0) == BudgetDriftClass.CRITICAL

    def test_boundary_values(self):
        """Test exact boundary values."""
        # Boundaries are inclusive of lower class
        assert drift_class_from_value(0.05) == BudgetDriftClass.STABLE
        assert drift_class_from_value(0.15) == BudgetDriftClass.DRIFTING
        assert drift_class_from_value(0.25) == BudgetDriftClass.DIVERGING


# =============================================================================
# Test: Noise Multiplier Mapping
# =============================================================================

class TestNoiseMultiplierMapping:
    """Tests for drift_class → noise_multiplier deterministic mapping."""

    def test_table_values(self):
        """Verify table values match doctrine."""
        assert NOISE_MULTIPLIER_TABLE[BudgetDriftClass.STABLE] == 1.0
        assert NOISE_MULTIPLIER_TABLE[BudgetDriftClass.DRIFTING] == 1.3
        assert NOISE_MULTIPLIER_TABLE[BudgetDriftClass.DIVERGING] == 1.6
        assert NOISE_MULTIPLIER_TABLE[BudgetDriftClass.CRITICAL] == 2.0

    def test_compute_noise_multiplier_stable(self):
        """STABLE drift → multiplier 1.0"""
        assert compute_noise_multiplier(0.0) == 1.0
        assert compute_noise_multiplier(0.05) == 1.0
        assert compute_noise_multiplier(-0.03) == 1.0

    def test_compute_noise_multiplier_drifting(self):
        """DRIFTING drift → multiplier 1.3"""
        assert compute_noise_multiplier(0.10) == 1.3
        assert compute_noise_multiplier(-0.12) == 1.3

    def test_compute_noise_multiplier_diverging(self):
        """DIVERGING drift → multiplier 1.6"""
        assert compute_noise_multiplier(0.20) == 1.6
        assert compute_noise_multiplier(-0.22) == 1.6

    def test_compute_noise_multiplier_critical(self):
        """CRITICAL drift → multiplier 2.0"""
        assert compute_noise_multiplier(0.30) == 2.0
        assert compute_noise_multiplier(-0.50) == 2.0

    def test_continuous_multiplier_bounds(self):
        """Continuous multiplier is bounded [1.0, 3.0]."""
        assert compute_noise_multiplier_continuous(0.0) == 1.0
        assert compute_noise_multiplier_continuous(0.99) == pytest.approx(10.0, rel=1.0)  # Clamped
        assert compute_noise_multiplier_continuous(0.99) == 3.0  # Clamped to max
        assert compute_noise_multiplier_continuous(1.0) == 3.0

    def test_determinism(self):
        """Same input always produces same output."""
        for _ in range(100):
            assert compute_noise_multiplier(0.10) == 1.3
            assert compute_noise_multiplier(0.20) == 1.6


# =============================================================================
# Test: Stability Class Classification
# =============================================================================

class TestStabilityClassification:
    """Tests for stability_class_from_health() mapping."""

    def test_stable_classification(self):
        """STABLE: health >= 80 AND stability >= 0.95"""
        assert stability_class_from_health(100.0, 1.0) == BudgetStabilityClass.STABLE
        assert stability_class_from_health(80.0, 0.95) == BudgetStabilityClass.STABLE
        assert stability_class_from_health(90.0, 0.98) == BudgetStabilityClass.STABLE

    def test_drifting_classification(self):
        """DRIFTING: health in [70, 80) OR stability in [0.7, 0.95)"""
        assert stability_class_from_health(75.0, 1.0) == BudgetStabilityClass.DRIFTING
        assert stability_class_from_health(100.0, 0.90) == BudgetStabilityClass.DRIFTING
        assert stability_class_from_health(79.9, 0.95) == BudgetStabilityClass.DRIFTING
        assert stability_class_from_health(80.0, 0.94) == BudgetStabilityClass.DRIFTING

    def test_volatile_classification(self):
        """VOLATILE: health < 70 OR stability < 0.7"""
        assert stability_class_from_health(60.0, 1.0) == BudgetStabilityClass.VOLATILE
        assert stability_class_from_health(100.0, 0.5) == BudgetStabilityClass.VOLATILE
        assert stability_class_from_health(69.9, 0.95) == BudgetStabilityClass.VOLATILE
        assert stability_class_from_health(80.0, 0.69) == BudgetStabilityClass.VOLATILE


# =============================================================================
# Test: Severity Multiplier Mapping
# =============================================================================

class TestSeverityMultiplierMapping:
    """Tests for stability_class → severity_multiplier deterministic mapping."""

    def test_table_values(self):
        """Verify table values match doctrine."""
        assert SEVERITY_MULTIPLIER_TABLE[BudgetStabilityClass.STABLE] == 1.0
        assert SEVERITY_MULTIPLIER_TABLE[BudgetStabilityClass.DRIFTING] == 0.7
        assert SEVERITY_MULTIPLIER_TABLE[BudgetStabilityClass.VOLATILE] == 0.4

    def test_compute_severity_multiplier(self):
        """Test compute_severity_multiplier() returns correct values."""
        assert compute_severity_multiplier(BudgetStabilityClass.STABLE) == 1.0
        assert compute_severity_multiplier(BudgetStabilityClass.DRIFTING) == 0.7
        assert compute_severity_multiplier(BudgetStabilityClass.VOLATILE) == 0.4

    def test_compute_from_health(self):
        """Test compute_severity_multiplier_from_health() end-to-end."""
        assert compute_severity_multiplier_from_health(100.0, 1.0) == 1.0
        assert compute_severity_multiplier_from_health(75.0, 0.90) == 0.7
        assert compute_severity_multiplier_from_health(60.0, 0.5) == 0.4

    def test_determinism(self):
        """Same input always produces same output."""
        for _ in range(100):
            assert compute_severity_multiplier(BudgetStabilityClass.DRIFTING) == 0.7


# =============================================================================
# Test: RSI Correction Factor
# =============================================================================

class TestRSICorrectionFactor:
    """Tests for RSI correction factor mapping."""

    def test_correction_values(self):
        """Test correction factor values."""
        assert compute_rsi_correction_factor(BudgetDriftClass.STABLE) == 1.0
        assert compute_rsi_correction_factor(BudgetDriftClass.DRIFTING) == 0.95
        assert compute_rsi_correction_factor(BudgetDriftClass.DIVERGING) == 0.85
        assert compute_rsi_correction_factor(BudgetDriftClass.CRITICAL) == 0.75


# =============================================================================
# Test: BudgetRiskSignal Construction
# =============================================================================

class TestBudgetRiskSignal:
    """Tests for BudgetRiskSignal construction and serialization."""

    def test_default_signal(self):
        """Default signal has stable values."""
        signal = build_budget_risk_signal()
        assert signal.drift_class == BudgetDriftClass.STABLE
        assert signal.stability_class == BudgetStabilityClass.STABLE
        assert signal.noise_multiplier == 1.0
        assert signal.severity_multiplier == 1.0
        assert signal.budget_confounded is False
        assert signal.admissibility_hint == "OK"

    def test_drifting_signal(self):
        """Signal with drifting values."""
        signal = build_budget_risk_signal(
            drift_value=0.10,
            health_score=75.0,
            stability_index=0.90,
        )
        assert signal.drift_class == BudgetDriftClass.DRIFTING
        assert signal.stability_class == BudgetStabilityClass.DRIFTING
        assert signal.noise_multiplier == 1.3
        assert signal.severity_multiplier == 0.7
        assert signal.budget_confounded is True
        assert signal.admissibility_hint == "WARN"

    def test_volatile_signal(self):
        """Signal with volatile values."""
        signal = build_budget_risk_signal(
            drift_value=0.30,
            health_score=60.0,
            stability_index=0.5,
            inv_bud_failures=["INV-BUD-1"],
        )
        assert signal.drift_class == BudgetDriftClass.CRITICAL
        assert signal.stability_class == BudgetStabilityClass.VOLATILE
        assert signal.noise_multiplier == 2.0
        assert signal.severity_multiplier == 0.4
        assert signal.budget_confounded is True
        assert signal.admissibility_hint == "BLOCK"
        assert "INV-BUD-1" in signal.inv_bud_failures

    def test_signal_to_dict(self):
        """Test JSON serialization."""
        signal = build_budget_risk_signal(drift_value=0.10, health_score=75.0)
        d = signal.to_dict()

        assert d["drift_class"] == "DRIFTING"
        assert d["stability_class"] == "DRIFTING"
        assert d["noise_multiplier"] == 1.3
        assert d["severity_multiplier"] == 0.7
        assert d["budget_confounded"] is True
        assert d["admissibility_hint"] == "WARN"

        # Verify JSON serializable
        json_str = json.dumps(d)
        assert "DRIFTING" in json_str


# =============================================================================
# Test: JSON Serialization
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON serialization of budget structures."""

    def test_budget_risk_signal_json(self):
        """BudgetRiskSignal serializes to valid JSON."""
        signal = build_budget_risk_signal(
            drift_value=0.15,
            health_score=70.0,
            stability_index=0.8,
            inv_bud_failures=["INV-BUD-2", "INV-BUD-3"],
        )
        d = signal.to_dict()

        # Serialize
        json_str = json.dumps(d, indent=2)

        # Deserialize
        loaded = json.loads(json_str)

        # Verify roundtrip
        assert loaded["drift_class"] == "DRIFTING"
        assert loaded["noise_multiplier"] == 1.3
        assert loaded["inv_bud_failures"] == ["INV-BUD-2", "INV-BUD-3"]

    def test_stability_report_extension(self):
        """extend_stability_report_with_budget() produces valid JSON."""
        base_report = {
            "delta_p_success": 0.01,
            "success_rate_final": 0.85,
        }
        signal = build_budget_risk_signal(drift_value=0.20)

        extended = extend_stability_report_with_budget(base_report, signal)

        # Verify structure
        assert "budget_risk" in extended
        assert extended["budget_risk"]["drift_class"] == "DIVERGING"
        assert extended["budget_risk"]["noise_multiplier"] == 1.6

        # Verify JSON serializable
        json_str = json.dumps(extended)
        assert "budget_risk" in json_str

    def test_adjusted_severity_json(self):
        """adjust_divergence_severity() produces valid JSON."""
        signal = build_budget_risk_signal(health_score=60.0, stability_index=0.5)
        result = adjust_divergence_severity("CRITICAL", signal)

        # Verify structure
        assert result["raw_severity"] == "CRITICAL"
        assert result["adjusted_severity"] in ["NONE", "INFO", "WARN", "CRITICAL"]
        assert result["severity_multiplier"] == 0.4

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert "adjusted_severity" in json_str


# =============================================================================
# Test: TDA Context
# =============================================================================

class TestTDAContext:
    """Tests for TDA context computation."""

    def test_nominal_context(self):
        """Stable budget → NOMINAL context."""
        signal = build_budget_risk_signal(health_score=100.0, stability_index=1.0)
        assert compute_tda_context(signal) == "NOMINAL"

    def test_drift_context(self):
        """Drifting budget → BUDGET_DRIFT context."""
        signal = build_budget_risk_signal(health_score=75.0, stability_index=0.90)
        assert compute_tda_context(signal) == "BUDGET_DRIFT"

    def test_unstable_context(self):
        """Volatile budget → BUDGET_UNSTABLE context."""
        signal = build_budget_risk_signal(health_score=50.0, stability_index=0.5)
        assert compute_tda_context(signal) == "BUDGET_UNSTABLE"


# =============================================================================
# Test: Adjusted Severity
# =============================================================================

class TestAdjustedSeverity:
    """Tests for severity adjustment based on budget stability."""

    def test_stable_no_adjustment(self):
        """STABLE budget → no severity adjustment."""
        signal = build_budget_risk_signal(health_score=100.0, stability_index=1.0)
        result = adjust_divergence_severity("CRITICAL", signal)
        assert result["raw_severity"] == "CRITICAL"
        assert result["adjusted_severity"] == "CRITICAL"
        assert result["adjustment_applied"] is False

    def test_drifting_adjustment(self):
        """DRIFTING budget → some severity reduction."""
        signal = build_budget_risk_signal(health_score=75.0, stability_index=0.90)
        result = adjust_divergence_severity("CRITICAL", signal)
        assert result["raw_severity"] == "CRITICAL"
        # CRITICAL (3) * 0.7 = 2.1 → WARN (2)
        assert result["adjusted_severity"] == "WARN"
        assert result["adjustment_applied"] is True

    def test_volatile_adjustment(self):
        """VOLATILE budget → strong severity reduction."""
        signal = build_budget_risk_signal(health_score=50.0, stability_index=0.5)
        result = adjust_divergence_severity("CRITICAL", signal)
        assert result["raw_severity"] == "CRITICAL"
        # CRITICAL (3) * 0.4 = 1.2 → INFO (1)
        assert result["adjusted_severity"] == "INFO"
        assert result["adjustment_applied"] is True

    def test_none_severity_unchanged(self):
        """NONE severity stays NONE regardless of budget."""
        signal = build_budget_risk_signal(health_score=50.0, stability_index=0.5)
        result = adjust_divergence_severity("NONE", signal)
        assert result["adjusted_severity"] == "NONE"
        assert result["adjustment_applied"] is False


# =============================================================================
# Test: P3 Integration (BudgetAwareDeltaPMetrics)
# =============================================================================

class TestP3Integration:
    """Tests for P3 BudgetAwareDeltaPMetrics integration."""

    def test_import_and_create(self):
        """BudgetAwareDeltaPMetrics can be imported and created."""
        from backend.topology.first_light.delta_p_computer import (
            DeltaPMetrics,
            BudgetAwareDeltaPMetrics,
        )

        metrics = DeltaPMetrics(
            delta_p_success=0.01,
            success_rate_final=0.85,
            mean_rsi=0.70,
            total_windows=10,
        )

        budget_aware = BudgetAwareDeltaPMetrics.from_metrics_and_budget(
            metrics=metrics,
            drift_value=0.10,
            health_score=75.0,
            stability_index=0.90,
        )

        assert budget_aware.noise_multiplier == 1.3
        assert budget_aware.budget_signal.drift_class == BudgetDriftClass.DRIFTING

    def test_stability_report_has_budget_risk(self):
        """build_stability_report() includes budget_risk section."""
        from backend.topology.first_light.delta_p_computer import (
            DeltaPMetrics,
            BudgetAwareDeltaPMetrics,
        )

        metrics = DeltaPMetrics(mean_rsi=0.70)
        budget_aware = BudgetAwareDeltaPMetrics.from_metrics_and_budget(
            metrics=metrics,
            drift_value=0.20,
        )

        report = budget_aware.build_stability_report()

        assert "budget_risk" in report
        assert report["budget_risk"]["drift_class"] == "DIVERGING"
        assert report["budget_risk"]["noise_multiplier"] == 1.6
        assert report["budget_risk"]["admissibility_hint"] == "OK"

    def test_stability_report_json_serializable(self):
        """Stability report is JSON serializable."""
        from backend.topology.first_light.delta_p_computer import (
            DeltaPMetrics,
            BudgetAwareDeltaPMetrics,
        )

        metrics = DeltaPMetrics()
        budget_aware = BudgetAwareDeltaPMetrics.from_metrics_and_budget(
            metrics=metrics,
            drift_value=0.15,
            health_score=70.0,
        )

        report = budget_aware.build_stability_report()
        json_str = json.dumps(report, indent=2)

        assert "budget_risk" in json_str
        assert "noise_multiplier" in json_str


# =============================================================================
# Test: P4 Integration (BudgetAwareDivergenceSummary)
# =============================================================================

class TestP4Integration:
    """Tests for P4 BudgetAwareDivergenceSummary integration."""

    def test_import_and_create(self):
        """BudgetAwareDivergenceSummary can be imported and created."""
        from backend.topology.first_light.divergence_analyzer import (
            DivergenceSummary,
            BudgetAwareDivergenceSummary,
        )

        summary = DivergenceSummary(
            total_comparisons=100,
            total_divergences=20,
            minor_divergences=10,
            moderate_divergences=7,
            severe_divergences=3,
        )

        budget_aware = BudgetAwareDivergenceSummary.from_summary_and_budget(
            summary=summary,
            health_score=60.0,
            stability_index=0.5,
        )

        assert budget_aware.budget_signal.stability_class == BudgetStabilityClass.VOLATILE
        assert budget_aware.tda_context == "BUDGET_UNSTABLE"

    def test_volatile_severity_adjustment(self):
        """VOLATILE budget adjusts severity counts."""
        from backend.topology.first_light.divergence_analyzer import (
            DivergenceSummary,
            BudgetAwareDivergenceSummary,
        )

        summary = DivergenceSummary(
            total_divergences=20,
            minor_divergences=10,
            moderate_divergences=7,
            severe_divergences=3,
        )

        budget_aware = BudgetAwareDivergenceSummary.from_summary_and_budget(
            summary=summary,
            health_score=50.0,
            stability_index=0.5,
        )

        # VOLATILE: severe → moderate, moderate → minor
        assert budget_aware.adjusted_severe_divergences == 0
        assert budget_aware.adjusted_moderate_divergences == 3  # former severe
        assert budget_aware.adjusted_minor_divergences == 17  # former minor + moderate

    def test_budget_aware_summary_json_serializable(self):
        """BudgetAwareDivergenceSummary is JSON serializable."""
        from backend.topology.first_light.divergence_analyzer import (
            DivergenceSummary,
            BudgetAwareDivergenceSummary,
        )

        summary = DivergenceSummary(total_comparisons=50, total_divergences=10)
        budget_aware = BudgetAwareDivergenceSummary.from_summary_and_budget(
            summary=summary,
            health_score=75.0,
        )

        d = budget_aware.to_dict()
        json_str = json.dumps(d, indent=2)

        assert "budget_context" in json_str
        assert "adjusted_severity" in json_str
        assert "tda_context" in json_str


# =============================================================================
# Test: Determinism
# =============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_noise_multiplier_determinism(self):
        """Same drift always produces same multiplier."""
        values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        for v in values:
            results = [compute_noise_multiplier(v) for _ in range(10)]
            assert len(set(results)) == 1, f"Non-deterministic for drift={v}"

    def test_severity_multiplier_determinism(self):
        """Same health/stability always produces same multiplier."""
        test_cases = [
            (100.0, 1.0),
            (75.0, 0.90),
            (60.0, 0.5),
            (70.0, 0.7),
        ]
        for health, stability in test_cases:
            results = [
                compute_severity_multiplier_from_health(health, stability)
                for _ in range(10)
            ]
            assert len(set(results)) == 1

    def test_budget_signal_determinism(self):
        """Same inputs always produce same BudgetRiskSignal."""
        for _ in range(10):
            signal1 = build_budget_risk_signal(
                drift_value=0.15,
                health_score=72.0,
                stability_index=0.85,
            )
            signal2 = build_budget_risk_signal(
                drift_value=0.15,
                health_score=72.0,
                stability_index=0.85,
            )

            assert signal1.drift_class == signal2.drift_class
            assert signal1.stability_class == signal2.stability_class
            assert signal1.noise_multiplier == signal2.noise_multiplier
            assert signal1.severity_multiplier == signal2.severity_multiplier


# =============================================================================
# Test: Task 1 - build_budget_risk_summary_for_p3()
# =============================================================================

class TestBuildBudgetRiskSummaryForP3:
    """Tests for build_budget_risk_summary_for_p3() function."""

    def test_default_values(self):
        """Default call produces stable, OK summary."""
        summary = build_budget_risk_summary_for_p3()

        assert summary["drift_class"] == "STABLE"
        assert summary["stability_class"] == "STABLE"
        assert summary["noise_multiplier"] == 1.0
        assert summary["severity_multiplier"] == 1.0
        assert summary["admissibility_hint"] == "OK"
        assert summary["budget_confounded"] is False

    def test_drifting_values(self):
        """Drifting conditions produce WARN summary."""
        summary = build_budget_risk_summary_for_p3(
            drift_value=0.10,
            health_score=75.0,
            stability_index=0.90,
        )

        assert summary["drift_class"] == "DRIFTING"
        assert summary["stability_class"] == "DRIFTING"
        assert summary["noise_multiplier"] == 1.3
        assert summary["severity_multiplier"] == 0.7
        assert summary["admissibility_hint"] == "WARN"
        assert summary["budget_confounded"] is True

    def test_volatile_values(self):
        """Volatile conditions produce BLOCK summary."""
        summary = build_budget_risk_summary_for_p3(
            drift_value=0.30,
            health_score=60.0,
            stability_index=0.5,
            inv_bud_failures=["INV-BUD-1"],
        )

        assert summary["drift_class"] == "CRITICAL"
        assert summary["stability_class"] == "VOLATILE"
        assert summary["noise_multiplier"] == 2.0
        assert summary["severity_multiplier"] == 0.4
        assert summary["admissibility_hint"] == "BLOCK"
        assert summary["budget_confounded"] is True
        assert "INV-BUD-1" in summary["inv_bud_failures"]

    def test_all_required_fields_present(self):
        """Summary contains all required fields."""
        summary = build_budget_risk_summary_for_p3()

        required_fields = [
            "drift_class",
            "stability_class",
            "noise_multiplier",
            "severity_multiplier",
            "admissibility_hint",
            "rsi_correction_factor",
            "budget_confounded",
            "health_score",
            "stability_index",
            "inv_bud_failures",
        ]

        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    def test_json_serializable(self):
        """Summary is JSON serializable."""
        summary = build_budget_risk_summary_for_p3(
            drift_value=0.15,
            health_score=72.0,
            inv_bud_failures=["INV-BUD-2"],
        )

        json_str = json.dumps(summary, indent=2)
        loaded = json.loads(json_str)

        assert loaded["drift_class"] == "DRIFTING"
        assert loaded["inv_bud_failures"] == ["INV-BUD-2"]

    def test_attach_to_stability_report(self):
        """Summary can be attached to stability report."""
        stability_report = {
            "delta_p_success": 0.01,
            "success_rate_final": 0.85,
        }

        summary = build_budget_risk_summary_for_p3(drift_value=0.10)
        stability_report["budget_risk"] = summary

        assert "budget_risk" in stability_report
        assert stability_report["budget_risk"]["drift_class"] == "DRIFTING"


# =============================================================================
# Test: Task 2 - build_budget_context_for_p4()
# =============================================================================

class TestBuildBudgetContextForP4:
    """Tests for build_budget_context_for_p4() function."""

    def test_stable_context(self):
        """Stable budget produces nominal context."""
        divergence_summary = {
            "accuracy": {"divergence_rate": 0.15},
        }
        signal = build_budget_risk_signal(health_score=100.0, stability_index=1.0)

        context = build_budget_context_for_p4(divergence_summary, signal)

        assert context["budget_confounded"] is False
        assert context["effective_severity_shift"] == 0
        assert context["tda_context"] == "NOMINAL"
        assert context["stability_class"] == "STABLE"
        assert context["calibration_note"] == "No severity calibration needed."

    def test_drifting_context(self):
        """Drifting budget produces drift context."""
        divergence_summary = {
            "accuracy": {"divergence_rate": 0.20},
        }
        signal = build_budget_risk_signal(health_score=75.0, stability_index=0.90)

        context = build_budget_context_for_p4(divergence_summary, signal)

        assert context["budget_confounded"] is True
        assert context["effective_severity_shift"] == -1
        assert context["tda_context"] == "BUDGET_DRIFT"
        assert context["stability_class"] == "DRIFTING"
        assert "DRIFTING" in context["calibration_note"]

    def test_volatile_context(self):
        """Volatile budget produces unstable context."""
        divergence_summary = {
            "accuracy": {"divergence_rate": 0.30},
        }
        signal = build_budget_risk_signal(health_score=50.0, stability_index=0.5)

        context = build_budget_context_for_p4(divergence_summary, signal)

        assert context["budget_confounded"] is True
        assert context["effective_severity_shift"] == -2
        assert context["tda_context"] == "BUDGET_UNSTABLE"
        assert context["stability_class"] == "VOLATILE"
        assert "VOLATILE" in context["calibration_note"]

    def test_extracts_divergence_rate(self):
        """Context extracts divergence rate from summary."""
        divergence_summary = {
            "accuracy": {"divergence_rate": 0.25},
        }
        signal = build_budget_risk_signal()

        context = build_budget_context_for_p4(divergence_summary, signal)

        assert context["raw_divergence_rate"] == 0.25

    def test_handles_missing_accuracy(self):
        """Context handles missing accuracy section gracefully."""
        divergence_summary = {}
        signal = build_budget_risk_signal()

        context = build_budget_context_for_p4(divergence_summary, signal)

        assert context["raw_divergence_rate"] == 0.0

    def test_all_required_fields_present(self):
        """Context contains all required fields."""
        context = build_budget_context_for_p4({}, build_budget_risk_signal())

        required_fields = [
            "budget_confounded",
            "effective_severity_shift",
            "tda_context",
            "stability_class",
            "severity_multiplier",
            "raw_divergence_rate",
            "adjusted_interpretation",
            "calibration_note",
        ]

        for field in required_fields:
            assert field in context, f"Missing field: {field}"

    def test_json_serializable(self):
        """Context is JSON serializable."""
        divergence_summary = {"accuracy": {"divergence_rate": 0.15}}
        signal = build_budget_risk_signal(drift_value=0.15)

        context = build_budget_context_for_p4(divergence_summary, signal)
        json_str = json.dumps(context, indent=2)

        assert "tda_context" in json_str
        assert "effective_severity_shift" in json_str


# =============================================================================
# Test: Task 3 - attach_budget_risk_to_evidence()
# =============================================================================

class TestAttachBudgetRiskToEvidence:
    """Tests for attach_budget_risk_to_evidence() function."""

    def test_non_mutating(self):
        """Original evidence is not mutated."""
        original = {
            "proof_hash": "abc123",
            "governance": {"aligned": True},
        }
        signal = build_budget_risk_signal(drift_value=0.15)

        new_evidence = attach_budget_risk_to_evidence(original, signal)

        # Original should NOT have budget_risk
        assert "budget_risk" not in original.get("governance", {})

        # New should have budget_risk
        assert "budget_risk" in new_evidence["governance"]

    def test_creates_governance_if_missing(self):
        """Creates governance section if not present."""
        original = {"proof_hash": "abc123"}
        signal = build_budget_risk_signal()

        new_evidence = attach_budget_risk_to_evidence(original, signal)

        assert "governance" in new_evidence
        assert "budget_risk" in new_evidence["governance"]

    def test_preserves_existing_governance(self):
        """Preserves existing governance fields."""
        original = {
            "proof_hash": "abc123",
            "governance": {
                "aligned": True,
                "policy_id": "POL-001",
            },
        }
        signal = build_budget_risk_signal()

        new_evidence = attach_budget_risk_to_evidence(original, signal)

        assert new_evidence["governance"]["aligned"] is True
        assert new_evidence["governance"]["policy_id"] == "POL-001"
        assert "budget_risk" in new_evidence["governance"]

    def test_budget_risk_structure(self):
        """Budget risk has correct structure."""
        original = {"proof_hash": "abc123"}
        signal = build_budget_risk_signal(
            drift_value=0.10,
            health_score=75.0,
            stability_index=0.90,
        )

        new_evidence = attach_budget_risk_to_evidence(original, signal)
        budget_risk = new_evidence["governance"]["budget_risk"]

        assert budget_risk["drift_class"] == "DRIFTING"
        assert budget_risk["stability_class"] == "DRIFTING"
        assert budget_risk["noise_multiplier"] == 1.3
        assert budget_risk["severity_multiplier"] == 0.7
        assert budget_risk["admissibility_hint"] == "WARN"
        assert budget_risk["budget_confounded"] is True
        assert budget_risk["tda_context"] == "BUDGET_DRIFT"

    def test_tda_context_included(self):
        """TDA context is included in budget_risk."""
        original = {}
        signal = build_budget_risk_signal(health_score=50.0, stability_index=0.5)

        new_evidence = attach_budget_risk_to_evidence(original, signal)
        budget_risk = new_evidence["governance"]["budget_risk"]

        assert budget_risk["tda_context"] == "BUDGET_UNSTABLE"

    def test_all_required_fields_present(self):
        """Budget risk contains all required fields."""
        new_evidence = attach_budget_risk_to_evidence({}, build_budget_risk_signal())
        budget_risk = new_evidence["governance"]["budget_risk"]

        required_fields = [
            "drift_class",
            "stability_class",
            "noise_multiplier",
            "severity_multiplier",
            "admissibility_hint",
            "budget_confounded",
            "tda_context",
            "health_score",
            "stability_index",
        ]

        for field in required_fields:
            assert field in budget_risk, f"Missing field: {field}"

    def test_json_serializable(self):
        """New evidence is JSON serializable."""
        original = {
            "proof_hash": "abc123",
            "governance": {"aligned": True},
        }
        signal = build_budget_risk_signal(drift_value=0.20)

        new_evidence = attach_budget_risk_to_evidence(original, signal)
        json_str = json.dumps(new_evidence, indent=2)

        assert "budget_risk" in json_str
        assert "tda_context" in json_str

    def test_deep_copy_nested_structures(self):
        """Deep copies nested structures properly."""
        original = {
            "governance": {
                "policies": ["POL-1", "POL-2"],
                "nested": {"deep": "value"},
            },
        }
        signal = build_budget_risk_signal()

        new_evidence = attach_budget_risk_to_evidence(original, signal)

        # Modify original nested structure
        original["governance"]["policies"].append("POL-3")
        original["governance"]["nested"]["deep"] = "modified"

        # New evidence should not be affected
        assert len(new_evidence["governance"]["policies"]) == 2
        assert new_evidence["governance"]["nested"]["deep"] == "value"
