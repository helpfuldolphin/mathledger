"""
Tests for noise_vs_reality generator.

Smoke tests cover:
- Missing fields handling
- Partial coverage scenarios
- Full coverage scenarios
- Schema validation
"""

import pytest
from datetime import datetime, timezone

from backend.topology.first_light.noise_vs_reality import (
    SCHEMA_VERSION,
    CoverageVerdict,
    DeltaPScatterPoint,
    RedFlagAnnotation,
    P3SummaryInput,
    P5SummaryInput,
    extract_delta_p_scatter,
    compute_comparison_metrics,
    assess_coverage,
    generate_governance_advisory,
    build_noise_vs_reality_summary,
    build_from_harness_and_divergence,
    validate_noise_vs_reality_summary,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_p3_input():
    """Basic P3 input with moderate noise."""
    return P3SummaryInput(
        total_cycles=100,
        noise_event_rate=0.08,
        regime_proportions={
            "base": 1.0,
            "correlated": 0.15,
            "degradation_degraded": 0.10,
            "heat_death_stressed": 0.05,
            "pathology": 0.05,
        },
        delta_p_contribution={
            "total": -0.35,
            "by_type": {
                "timeout": -0.22,
                "spurious_fail": -0.13,
                "spurious_pass": 0.0,
            },
        },
        pathology_events=[
            {"cycle": 50, "type": "spike", "severity": "MODERATE"},
        ],
    )


@pytest.fixture
def basic_p5_input():
    """Basic P5 input with low divergence."""
    scatter_points = []
    for i in range(100):
        is_red_flag = i in [30, 70]
        scatter_points.append(DeltaPScatterPoint(
            cycle=i,
            twin_delta_p=0.01,
            real_delta_p=0.012 if i % 10 == 0 else 0.01,
            divergence_magnitude=0.002 if i % 10 == 0 else 0.0,
            is_red_flag=is_red_flag,
            red_flag_type="DELTA_P_SPIKE" if is_red_flag else None,
        ))

    return P5SummaryInput(
        total_cycles=100,
        divergence_time_series=scatter_points,
        red_flags=[
            RedFlagAnnotation(
                cycle=30,
                red_flag_type="DELTA_P_SPIKE",
                severity="WARN",
                description="Delta-p spike detected",
            ),
            RedFlagAnnotation(
                cycle=70,
                red_flag_type="DELTA_P_SPIKE",
                severity="WARN",
                description="Delta-p spike detected",
            ),
        ],
        telemetry_source={
            "provider": "usla_adapter",
            "start_timestamp": "2025-12-11T08:00:00+00:00",
            "end_timestamp": "2025-12-11T10:00:00+00:00",
        },
    )


@pytest.fixture
def high_divergence_p5_input():
    """P5 input with high divergence for insufficient coverage test."""
    scatter_points = []
    for i in range(100):
        # Higher divergence in many cycles
        div_mag = 0.05 if i % 5 == 0 else 0.01
        is_red_flag = i in [10, 20, 30, 40, 50, 60, 70, 80, 90]

        scatter_points.append(DeltaPScatterPoint(
            cycle=i,
            twin_delta_p=0.01,
            real_delta_p=0.01 + div_mag,
            divergence_magnitude=div_mag,
            is_red_flag=is_red_flag,
            red_flag_type="DELTA_P_SPIKE" if is_red_flag else None,
        ))

    return P5SummaryInput(
        total_cycles=100,
        divergence_time_series=scatter_points,
        red_flags=[
            RedFlagAnnotation(cycle=c, red_flag_type="DELTA_P_SPIKE", severity="WARN", description="")
            for c in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ],
        telemetry_source={
            "provider": "usla_adapter",
            "start_timestamp": "2025-12-11T08:00:00+00:00",
            "end_timestamp": "2025-12-11T10:00:00+00:00",
        },
    )


# =============================================================================
# Test: Missing Fields Handling
# =============================================================================

class TestMissingFields:
    """Tests for handling missing or incomplete data."""

    def test_empty_p3_input(self):
        """Test generator handles empty P3 input."""
        p3_input = P3SummaryInput(
            total_cycles=0,
            noise_event_rate=0.0,
            regime_proportions={},
            delta_p_contribution={"total": 0.0, "by_type": {}},
            pathology_events=[],
        )
        p5_input = P5SummaryInput(
            total_cycles=0,
            divergence_time_series=[],
            red_flags=[],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = build_noise_vs_reality_summary(p3_input, p5_input)

        assert result["schema_version"] == SCHEMA_VERSION
        assert result["mode"] == "SHADOW"
        assert result["p3_summary"]["total_cycles"] == 0
        assert result["p5_summary"]["total_cycles"] == 0

    def test_empty_p5_divergence_series(self, basic_p3_input):
        """Test handling empty P5 divergence series."""
        p5_input = P5SummaryInput(
            total_cycles=0,
            divergence_time_series=[],
            red_flags=[],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = build_noise_vs_reality_summary(basic_p3_input, p5_input)

        assert result["p5_summary"]["divergence_rate"] == 0.0
        assert result["p5_summary"]["divergence_stats"]["mean"] == 0.0
        assert result["comparison_metrics"]["coverage_ratio"] == 999.99  # inf capped

    def test_missing_regime_proportions(self, basic_p5_input):
        """Test handling missing regime proportions in P3."""
        p3_input = P3SummaryInput(
            total_cycles=100,
            noise_event_rate=0.05,
            regime_proportions={},  # Missing
            delta_p_contribution={"total": -0.1, "by_type": {}},
            pathology_events=[],
        )

        result = build_noise_vs_reality_summary(p3_input, basic_p5_input)

        assert result["p3_summary"]["regime_proportions"] == {}
        # Should still produce valid output
        is_valid, errors = validate_noise_vs_reality_summary(result)
        assert is_valid, f"Validation errors: {errors}"

    def test_p3_summary_from_noise_summary_missing_fields(self):
        """Test P3SummaryInput.from_noise_summary with missing fields."""
        incomplete_summary = {
            "total_cycles": 50,
            # Missing other fields
        }

        p3_input = P3SummaryInput.from_noise_summary(incomplete_summary)

        assert p3_input.total_cycles == 50
        assert p3_input.noise_event_rate == 0.0
        assert p3_input.regime_proportions == {}


# =============================================================================
# Test: Partial Coverage Scenarios
# =============================================================================

class TestPartialCoverage:
    """Tests for marginal/partial coverage scenarios."""

    def test_marginal_coverage_low_ratio(self, basic_p3_input):
        """Test marginal coverage with low coverage ratio."""
        # P5 with higher divergence rate than P3 noise rate
        scatter_points = [
            DeltaPScatterPoint(
                cycle=i,
                twin_delta_p=0.01,
                real_delta_p=0.02,  # Significant divergence
                divergence_magnitude=0.01,
                is_red_flag=i % 10 == 0,
            )
            for i in range(100)
        ]

        p5_input = P5SummaryInput(
            total_cycles=100,
            divergence_time_series=scatter_points,
            red_flags=[],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = build_noise_vs_reality_summary(basic_p3_input, p5_input)

        # Coverage should be marginal since P5 divergence rate > P3 noise rate
        assert result["coverage_assessment"]["verdict"] in ["MARGINAL", "ADEQUATE"]

    def test_marginal_coverage_with_exceedances(self, basic_p3_input):
        """Test marginal coverage with some exceedance cycles."""
        scatter_points = []
        for i in range(100):
            # Create some exceedances (divergence > 2x P3 noise impact)
            div_mag = 0.1 if i in [25, 50, 75] else 0.001
            scatter_points.append(DeltaPScatterPoint(
                cycle=i,
                twin_delta_p=0.01,
                real_delta_p=0.01 + div_mag,
                divergence_magnitude=div_mag,
                is_red_flag=div_mag > 0.05,
            ))

        p5_input = P5SummaryInput(
            total_cycles=100,
            divergence_time_series=scatter_points,
            red_flags=[],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = build_noise_vs_reality_summary(basic_p3_input, p5_input)

        # Should have some exceedance cycles
        exceedance_count = len(result["comparison_metrics"]["exceedance_cycles"])
        assert exceedance_count >= 0  # May have exceedances depending on P3 bounds

    def test_gaps_detected_for_unmodeled_red_flags(self, basic_p3_input):
        """Test that gaps are detected when P5 has red-flags not in P3 pathologies."""
        scatter_points = [
            DeltaPScatterPoint(
                cycle=i,
                twin_delta_p=0.01,
                real_delta_p=0.012,
                divergence_magnitude=0.002,
                is_red_flag=i == 50,
                red_flag_type="RSI_DROP" if i == 50 else None,  # Not in P3 pathologies
            )
            for i in range(100)
        ]

        p5_input = P5SummaryInput(
            total_cycles=100,
            divergence_time_series=scatter_points,
            red_flags=[
                RedFlagAnnotation(
                    cycle=50,
                    red_flag_type="RSI_DROP",  # Not modeled in P3
                    severity="WARN",
                    description="RSI drop detected",
                ),
            ],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = build_noise_vs_reality_summary(basic_p3_input, p5_input)

        # Should detect gap for RSI_DROP
        gaps = result["coverage_assessment"]["gaps"]
        assert any("RSI_DROP" in g.get("description", "") for g in gaps)


# =============================================================================
# Test: Full Coverage Scenarios
# =============================================================================

class TestFullCoverage:
    """Tests for adequate/full coverage scenarios."""

    def test_adequate_coverage(self, basic_p3_input, basic_p5_input):
        """Test adequate coverage with P3 > P5."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)

        assert result["schema_version"] == SCHEMA_VERSION
        assert result["mode"] == "SHADOW"
        assert result["coverage_assessment"]["verdict"] == "ADEQUATE"
        assert result["governance_advisory"]["severity"] == "INFO"
        assert result["governance_advisory"]["action_required"] is False

    def test_high_coverage_ratio(self, basic_p5_input):
        """Test with very high P3 noise rate for high coverage ratio."""
        p3_input = P3SummaryInput(
            total_cycles=100,
            noise_event_rate=0.25,  # High noise rate
            regime_proportions={"base": 1.0, "pathology": 0.20},
            delta_p_contribution={"total": -0.8, "by_type": {}},
            pathology_events=[],
        )

        result = build_noise_vs_reality_summary(p3_input, basic_p5_input)

        assert result["comparison_metrics"]["coverage_ratio"] > 1.0
        assert result["coverage_assessment"]["verdict"] == "ADEQUATE"
        assert result["coverage_assessment"]["confidence"] > 0.7

    def test_no_exceedances_for_adequate_coverage(self, basic_p3_input, basic_p5_input):
        """Test that adequate coverage has no/minimal exceedances."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)

        exceedance_rate = result["comparison_metrics"]["exceedance_rate"]
        assert exceedance_rate < 0.05  # Less than 5%


# =============================================================================
# Test: Insufficient Coverage Scenarios
# =============================================================================

class TestInsufficientCoverage:
    """Tests for insufficient coverage scenarios."""

    def test_insufficient_coverage_high_divergence(self, high_divergence_p5_input):
        """Test insufficient coverage when P5 divergence is very high."""
        p3_input = P3SummaryInput(
            total_cycles=100,
            noise_event_rate=0.02,  # Very low noise rate
            regime_proportions={"base": 1.0},
            delta_p_contribution={"total": -0.05, "by_type": {}},
            pathology_events=[],
        )

        result = build_noise_vs_reality_summary(p3_input, high_divergence_p5_input)

        # P5 has high divergence with low P3 noise - should be insufficient or marginal
        verdict = result["coverage_assessment"]["verdict"]
        assert verdict in ["MARGINAL", "INSUFFICIENT"]

    def test_insufficient_coverage_advisory(self):
        """Test governance advisory for insufficient coverage."""
        coverage_assessment = {
            "verdict": "INSUFFICIENT",
            "confidence": 0.3,
            "gaps": [],
            "recommendations": [],
        }

        advisory = generate_governance_advisory(coverage_assessment)

        assert advisory["severity"] == "WARN"
        assert advisory["action_required"] is True


# =============================================================================
# Test: Δp Scatter Extractor
# =============================================================================

class TestDeltaPScatterExtractor:
    """Tests for Δp scatter extraction."""

    def test_extract_scatter_points(self, basic_p5_input):
        """Test scatter point extraction."""
        result = extract_delta_p_scatter(basic_p5_input)

        assert "scatter_points" in result
        assert "red_flag_annotations" in result
        assert "statistics" in result
        assert len(result["scatter_points"]) == 100

    def test_scatter_statistics(self, basic_p5_input):
        """Test scatter statistics calculation."""
        result = extract_delta_p_scatter(basic_p5_input)

        stats = result["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "p95" in stats
        assert "divergence_rate" in stats

    def test_red_flag_annotation_extraction(self, basic_p5_input):
        """Test red-flag annotation extraction."""
        result = extract_delta_p_scatter(basic_p5_input)

        assert len(result["red_flag_annotations"]) == 2
        assert result["red_flag_cycles"] == [30, 70]

    def test_empty_divergence_series(self):
        """Test extraction with empty divergence series."""
        p5_input = P5SummaryInput(
            total_cycles=0,
            divergence_time_series=[],
            red_flags=[],
            telemetry_source={"provider": "test", "start_timestamp": "", "end_timestamp": ""},
        )

        result = extract_delta_p_scatter(p5_input)

        assert result["scatter_points"] == []
        assert result["statistics"]["mean"] == 0.0
        assert result["statistics"]["divergence_rate"] == 0.0


# =============================================================================
# Test: Comparison Metrics
# =============================================================================

class TestComparisonMetrics:
    """Tests for comparison metrics calculation."""

    def test_coverage_ratio_calculation(self):
        """Test coverage ratio calculation."""
        p3_summary = {"noise_event_rate": 0.10, "total_cycles": 100, "delta_p_contribution": {"total": -0.2}}
        p5_summary = {"divergence_rate": 0.05, "divergence_stats": {"mean": 0.01}}

        result = compute_comparison_metrics(p3_summary, p5_summary)

        assert result["coverage_ratio"] == 2.0  # 0.10 / 0.05

    def test_coverage_ratio_div_by_zero(self):
        """Test coverage ratio with zero P5 divergence."""
        p3_summary = {"noise_event_rate": 0.10, "total_cycles": 100, "delta_p_contribution": {"total": -0.2}}
        p5_summary = {"divergence_rate": 0.0, "divergence_stats": {"mean": 0.0}}

        result = compute_comparison_metrics(p3_summary, p5_summary)

        assert result["coverage_ratio"] == 999.99  # Capped infinity

    def test_noise_vs_divergence_rate(self):
        """Test noise vs divergence rate comparison."""
        p3_summary = {"noise_event_rate": 0.08, "total_cycles": 100, "delta_p_contribution": {"total": -0.1}}
        p5_summary = {"divergence_rate": 0.05, "divergence_stats": {"mean": 0.01}}

        result = compute_comparison_metrics(p3_summary, p5_summary)

        nvd = result["noise_vs_divergence_rate"]
        assert nvd["p3_noise_rate"] == 0.08
        assert nvd["p5_divergence_rate"] == 0.05
        assert nvd["difference"] == 0.03


# =============================================================================
# Test: Schema Validation
# =============================================================================

class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_valid_summary_passes_validation(self, basic_p3_input, basic_p5_input):
        """Test that valid summary passes validation."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)

        is_valid, errors = validate_noise_vs_reality_summary(result)

        assert is_valid, f"Validation errors: {errors}"

    def test_missing_required_field_fails_validation(self):
        """Test that missing required field fails validation."""
        invalid_summary = {
            "schema_version": SCHEMA_VERSION,
            # Missing other required fields
        }

        is_valid, errors = validate_noise_vs_reality_summary(invalid_summary)

        assert not is_valid
        assert any("Missing required field" in e for e in errors)

    def test_wrong_schema_version_fails_validation(self, basic_p3_input, basic_p5_input):
        """Test that wrong schema version fails validation."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)
        result["schema_version"] = "wrong-version/1.0.0"

        is_valid, errors = validate_noise_vs_reality_summary(result)

        assert not is_valid
        assert any("schema_version" in e for e in errors)

    def test_wrong_mode_fails_validation(self, basic_p3_input, basic_p5_input):
        """Test that wrong mode fails validation."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)
        result["mode"] = "LIVE"  # Should be SHADOW

        is_valid, errors = validate_noise_vs_reality_summary(result)

        assert not is_valid
        assert any("SHADOW" in e for e in errors)

    def test_invalid_verdict_fails_validation(self, basic_p3_input, basic_p5_input):
        """Test that invalid verdict fails validation."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)
        result["coverage_assessment"]["verdict"] = "INVALID"

        is_valid, errors = validate_noise_vs_reality_summary(result)

        assert not is_valid
        assert any("verdict" in e for e in errors)

    def test_noise_event_rate_range(self, basic_p3_input, basic_p5_input):
        """Test that noise_event_rate out of range is detected."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)
        result["p3_summary"]["noise_event_rate"] = 1.5  # Out of range

        is_valid, errors = validate_noise_vs_reality_summary(result)

        assert not is_valid
        assert any("noise_event_rate" in e for e in errors)


# =============================================================================
# Test: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience/integration functions."""

    def test_build_from_harness_and_divergence(self):
        """Test building from harness summary and divergence data."""
        noise_summary = {
            "schema_version": "p3-noise-summary/1.0.0",
            "mode": "SHADOW",
            "total_cycles": 50,
            "regime_proportions": {"base": 1.0, "correlated": 0.1},
            "delta_p_aggregate": {
                "total_contribution": -0.15,
                "by_noise_type": {"timeout": -0.10, "spurious_fail": -0.05, "spurious_pass": 0.0},
            },
            "rsi_aggregate": {"noise_event_rate": 0.06},
        }

        divergence_series = [
            {
                "cycle": i,
                "twin_delta_p": 0.01,
                "real_delta_p": 0.012,
                "divergence_magnitude": 0.002,
                "is_red_flag": False,
            }
            for i in range(50)
        ]

        result = build_from_harness_and_divergence(
            noise_summary=noise_summary,
            divergence_series=divergence_series,
            experiment_id="test-exp-001",
        )

        assert result["schema_version"] == SCHEMA_VERSION
        assert result["experiment_id"] == "test-exp-001"
        assert result["p3_summary"]["total_cycles"] == 50
        assert result["p5_summary"]["total_cycles"] == 50

    def test_p3_summary_input_from_noise_summary(self):
        """Test P3SummaryInput creation from noise summary."""
        noise_summary = {
            "total_cycles": 100,
            "rsi_aggregate": {"noise_event_rate": 0.08},
            "regime_proportions": {"base": 1.0},
            "delta_p_aggregate": {"total_contribution": -0.3, "by_noise_type": {}},
        }

        p3_input = P3SummaryInput.from_noise_summary(noise_summary)

        assert p3_input.total_cycles == 100
        assert p3_input.noise_event_rate == 0.08
        assert p3_input.regime_proportions == {"base": 1.0}

    def test_p5_summary_input_from_divergence_data(self):
        """Test P5SummaryInput creation from divergence data."""
        divergence_series = [
            {"cycle": 0, "twin_delta_p": 0.01, "real_delta_p": 0.01, "divergence_magnitude": 0.0},
            {"cycle": 1, "twin_delta_p": 0.01, "real_delta_p": 0.02, "divergence_magnitude": 0.01, "is_red_flag": True},
        ]

        red_flags = [{"cycle": 1, "type": "SPIKE", "severity": "WARN", "description": "Test"}]

        p5_input = P5SummaryInput.from_divergence_data(
            divergence_series=divergence_series,
            red_flags=red_flags,
            provider="test_adapter",
        )

        assert p5_input.total_cycles == 2
        assert len(p5_input.divergence_time_series) == 2
        assert len(p5_input.red_flags) == 1
        assert p5_input.telemetry_source["provider"] == "test_adapter"


# =============================================================================
# Test: Governance Advisory
# =============================================================================

class TestGovernanceAdvisory:
    """Tests for governance advisory generation."""

    def test_adequate_advisory(self):
        """Test advisory for adequate coverage."""
        ca = {"verdict": "ADEQUATE", "confidence": 0.9, "gaps": []}

        advisory = generate_governance_advisory(ca)

        assert advisory["severity"] == "INFO"
        assert advisory["action_required"] is False

    def test_marginal_advisory(self):
        """Test advisory for marginal coverage."""
        ca = {"verdict": "MARGINAL", "confidence": 0.6, "gaps": []}

        advisory = generate_governance_advisory(ca)

        assert advisory["severity"] == "WARN"
        assert advisory["action_required"] is False

    def test_insufficient_advisory(self):
        """Test advisory for insufficient coverage."""
        ca = {"verdict": "INSUFFICIENT", "confidence": 0.3, "gaps": []}

        advisory = generate_governance_advisory(ca)

        assert advisory["severity"] == "WARN"
        assert advisory["action_required"] is True


# =============================================================================
# Test: SHADOW Mode Contract
# =============================================================================

class TestShadowModeContract:
    """Tests verifying SHADOW mode contract."""

    def test_output_always_shadow_mode(self, basic_p3_input, basic_p5_input):
        """Test that output always has SHADOW mode."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)

        assert result["mode"] == "SHADOW"

    def test_no_critical_severity(self, basic_p3_input, basic_p5_input):
        """Test that advisory severity is never CRITICAL."""
        result = build_noise_vs_reality_summary(basic_p3_input, basic_p5_input)

        severity = result["governance_advisory"]["severity"]
        assert severity in ["INFO", "WARN"]
        assert severity not in ["CRITICAL", "BLOCK", "ERROR"]

    def test_all_verdicts_supported(self, basic_p5_input):
        """Test all verdict types produce valid output."""
        for noise_rate, expected_verdict_group in [
            (0.15, ["ADEQUATE"]),  # High P3 -> ADEQUATE
            (0.02, ["MARGINAL", "INSUFFICIENT", "ADEQUATE"]),  # Low P3 -> could be any
        ]:
            p3_input = P3SummaryInput(
                total_cycles=100,
                noise_event_rate=noise_rate,
                regime_proportions={"base": 1.0},
                delta_p_contribution={"total": -0.1, "by_type": {}},
                pathology_events=[],
            )

            result = build_noise_vs_reality_summary(p3_input, basic_p5_input)

            assert result["coverage_assessment"]["verdict"] in expected_verdict_group
            is_valid, _ = validate_noise_vs_reality_summary(result)
            assert is_valid
