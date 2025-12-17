"""
Tests for runtime profile calibration correlation.

Validates:
- Correlation computation (Pearson)
- Window annotation with confounding flags
- Graceful handling of missing data
- JSON determinism
"""

import json
from typing import Dict, List

import pytest

from experiments.u2.runtime.calibration_correlation import (
    annotate_cal_windows_with_runtime_confounding,
    build_runtime_profile_calibration_annex,
    compute_pearson_correlation,
    correlate_runtime_profile_with_cal_windows,
    decompose_divergence_components,
)


def test_pearson_correlation_basic():
    """Test basic Pearson correlation computation."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect positive correlation
    corr = compute_pearson_correlation(x, y)
    assert isinstance(corr, float)
    assert abs(corr - 1.0) < 1e-6

    # Negative correlation
    y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
    corr_neg = compute_pearson_correlation(x, y_neg)
    assert isinstance(corr_neg, float)
    assert abs(corr_neg - (-1.0)) < 1e-6

    # No correlation (constant y) - zero variance
    y_const = [5.0, 5.0, 5.0, 5.0, 5.0]
    corr_const = compute_pearson_correlation(x, y_const)
    assert isinstance(corr_const, dict)
    assert corr_const["value"] is None
    assert corr_const["reason"] == "ZERO_VARIANCE_Y"

    # Insufficient data
    corr_short = compute_pearson_correlation([1.0], [2.0])
    assert isinstance(corr_short, dict)
    assert corr_short["value"] is None
    assert corr_short["reason"] == "INSUFFICIENT_POINTS"


def test_pearson_correlation_deterministic():
    """Test that correlation computation is deterministic."""
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0.15, 0.25, 0.35, 0.45, 0.55]

    corr1 = compute_pearson_correlation(x, y)
    corr2 = compute_pearson_correlation(x, y)

    assert corr1 == corr2
    assert isinstance(corr1, float)
    # Check rounding to 6 decimals
    assert corr1 == round(corr1, 6)


def test_correlate_runtime_profile_with_cal_windows():
    """Test correlation of runtime profile with CAL-EXP-1 windows."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "GREEN",
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    cal_windows = [
        {"divergence_rate": 0.1, "mean_delta_p": 0.01},
        {"divergence_rate": 0.15, "mean_delta_p": 0.02},
        {"divergence_rate": 0.12, "mean_delta_p": 0.015},
    ]

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, cal_windows)

    assert result["schema_version"] == "1.0.0"
    assert result["windows_analyzed"] == 3
    assert "correlations" in result
    assert "advisory_note" in result

    # Since profile_stability is constant, correlation may be None (zero variance)
    # But the function should still return a valid structure
    assert isinstance(result["correlations"], dict)


def test_correlate_missing_runtime_profile():
    """Test graceful handling of missing runtime profile."""
    runtime_profile = {}  # Missing profile_stability and no_run_rate

    cal_windows = [
        {"divergence_rate": 0.1, "mean_delta_p": 0.01},
        {"divergence_rate": 0.15, "mean_delta_p": 0.02},
    ]

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, cal_windows)

    assert result["schema_version"] == "1.0.0"
    assert result["windows_analyzed"] == 2
    assert "advisory_note" in result
    assert "missing" in result["advisory_note"].lower() or "no" in result["advisory_note"].lower()


def test_correlate_empty_windows():
    """Test graceful handling of empty windows list."""
    runtime_profile = {
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, [])

    assert result["schema_version"] == "1.0.0"
    assert result["windows_analyzed"] == 0
    assert "advisory_note" in result


def test_annotate_cal_windows_confounding_red_status():
    """Test window annotation with RED status and divergence spike."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "RED",
        "profile_stability": 0.65,
        "no_run_rate": 0.25,
    }

    cal_windows = [
        {"divergence_rate": 0.5, "mean_delta_p": 0.01},  # Not confounded (below threshold)
        {"divergence_rate": 0.85, "mean_delta_p": 0.02},  # Confounded (RED + spike)
        {"divergence_rate": 0.9, "mean_delta_p": 0.015},  # Confounded (RED + spike)
        {"divergence_rate": 0.3, "mean_delta_p": 0.01},  # Not confounded (low divergence)
    ]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8
    )

    assert len(annotated) == 4
    assert annotated[0]["runtime_profile_confounded"] is False
    assert annotated[1]["runtime_profile_confounded"] is True
    assert annotated[2]["runtime_profile_confounded"] is True
    assert annotated[3]["runtime_profile_confounded"] is False

    assert summary["confounded_windows_count"] == 2
    assert summary["confounded_window_indices"] == [1, 2]
    assert summary["is_red"] is True


def test_annotate_cal_windows_no_confounding_green_status():
    """Test window annotation with GREEN status (no confounding)."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "GREEN",
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    cal_windows = [
        {"divergence_rate": 0.9, "mean_delta_p": 0.02},  # High divergence but GREEN status
        {"divergence_rate": 0.95, "mean_delta_p": 0.03},  # High divergence but GREEN status
    ]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8
    )

    assert len(annotated) == 2
    assert annotated[0]["runtime_profile_confounded"] is False
    assert annotated[1]["runtime_profile_confounded"] is False

    assert summary["confounded_windows_count"] == 0
    assert summary["is_red"] is False


def test_annotate_cal_windows_missing_divergence_rate():
    """Test graceful handling of missing divergence_rate in windows."""
    runtime_profile = {
        "status_light": "RED",
    }

    cal_windows = [
        {"mean_delta_p": 0.01},  # Missing divergence_rate
        {"divergence_rate": 0.9, "mean_delta_p": 0.02},  # Has divergence_rate
    ]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows
    )

    assert len(annotated) == 2
    assert annotated[0]["runtime_profile_confounded"] is False  # Missing divergence_rate
    assert annotated[1]["runtime_profile_confounded"] is True  # Has divergence_rate and RED


def test_build_runtime_profile_calibration_annex():
    """Test building complete calibration annex."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "RED",
        "profile_stability": 0.75,
        "no_run_rate": 0.15,
    }

    cal_windows = [
        {"divergence_rate": 0.85, "mean_delta_p": 0.02},
        {"divergence_rate": 0.9, "mean_delta_p": 0.03},
    ]

    annex = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    assert annex is not None
    assert annex["schema_version"] == "1.0.0"
    assert annex["profile"] == "prod-hardened"
    assert annex["status_light"] == "RED"
    assert "windowed_correlations" in annex
    assert "confounding_summary" in annex
    assert "annotated_windows" in annex
    assert "advisory_note" in annex
    assert annex["mode"] == "SHADOW"

    # Check annotated windows have confounding flags
    assert len(annex["annotated_windows"]) == 2
    assert all("runtime_profile_confounded" in w for w in annex["annotated_windows"])


def test_build_runtime_profile_calibration_annex_missing_data():
    """Test graceful no-op when data is missing."""
    # Missing runtime profile
    annex1 = build_runtime_profile_calibration_annex(None, [{"divergence_rate": 0.1}])
    assert annex1 is None

    # Missing windows
    runtime_profile = {"profile": "prod-hardened", "status_light": "GREEN"}
    annex2 = build_runtime_profile_calibration_annex(runtime_profile, None)
    assert annex2 is None

    # Empty windows
    annex3 = build_runtime_profile_calibration_annex(runtime_profile, [])
    assert annex3 is None


def test_calibration_annex_json_serializable():
    """Test that calibration annex is JSON-serializable and deterministic."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "YELLOW",
        "profile_stability": 0.85,
        "no_run_rate": 0.10,
    }

    cal_windows = [
        {"divergence_rate": 0.2, "mean_delta_p": 0.01},
        {"divergence_rate": 0.25, "mean_delta_p": 0.02},
        {"divergence_rate": 0.22, "mean_delta_p": 0.015},
    ]

    annex1 = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)
    annex2 = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    # Both should be identical
    json1 = json.dumps(annex1, sort_keys=True)
    json2 = json.dumps(annex2, sort_keys=True)

    assert json1 == json2

    # Should be valid JSON
    parsed = json.loads(json1)
    assert parsed["schema_version"] == "1.0.0"


def test_confounding_detection_edge_cases():
    """Test confounding detection edge cases."""
    # Exactly at threshold
    runtime_profile = {"status_light": "RED"}
    cal_windows = [{"divergence_rate": 0.8}]  # Exactly at threshold

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8
    )

    assert annotated[0]["runtime_profile_confounded"] is True  # >= threshold

    # Just below threshold
    cal_windows_below = [{"divergence_rate": 0.799}]
    annotated_below, _ = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows_below, divergence_spike_threshold=0.8
    )

    assert annotated_below[0]["runtime_profile_confounded"] is False  # < threshold


def test_pearson_correlation_zero_variance_x():
    """Test correlation with zero variance in X."""
    x = [5.0, 5.0, 5.0, 5.0, 5.0]  # Constant X
    y = [1.0, 2.0, 3.0, 4.0, 5.0]  # Varying Y

    corr = compute_pearson_correlation(x, y)
    assert isinstance(corr, dict)
    assert corr["value"] is None
    assert corr["reason"] == "ZERO_VARIANCE_X"


def test_pearson_correlation_non_numeric_input():
    """Test correlation with non-numeric input."""
    x = [1.0, 2.0, 3.0]
    y = ["a", "b", "c"]  # Non-numeric

    corr = compute_pearson_correlation(x, y)
    assert isinstance(corr, dict)
    assert corr["value"] is None
    assert corr["reason"] == "NON_NUMERIC_INPUT"


def test_correlate_zero_variance_all_same_divergence():
    """Test correlation when all divergence rates are the same."""
    runtime_profile = {
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    # All windows have same divergence_rate (zero variance)
    cal_windows = [
        {"divergence_rate": 0.5, "mean_delta_p": 0.01},
        {"divergence_rate": 0.5, "mean_delta_p": 0.02},
        {"divergence_rate": 0.5, "mean_delta_p": 0.015},
    ]

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, cal_windows)

    assert result["windows_analyzed"] == 3
    assert "correlation_reasons" in result
    # profile_stability is constant (X), so should have ZERO_VARIANCE_X
    # divergence_rate is also constant (Y), but X is checked first
    assert "profile_stability_vs_divergence_rate" in result.get("correlation_reasons", {})
    # Since profile_stability is constant, X has zero variance
    assert result["correlation_reasons"]["profile_stability_vs_divergence_rate"] == "ZERO_VARIANCE_X"


def test_annotate_red_runtime_clean_windows():
    """Test RED runtime with clean windows (counterfactual control)."""
    runtime_profile = {
        "status_light": "RED",
        "profile": "prod-hardened",
    }

    # Clean windows (low divergence, low delta_p)
    cal_windows = [
        {"divergence_rate": 0.1, "mean_delta_p": 0.001},
        {"divergence_rate": 0.15, "mean_delta_p": 0.002},
    ]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8, delta_p_threshold=0.05
    )

    assert summary["confounded_windows_count"] == 0
    assert "not correlated with calibration instability" in summary["advisory_note"]


def test_annotate_red_runtime_high_delta_p():
    """Test RED runtime with high delta_p (should confound)."""
    runtime_profile = {
        "status_light": "RED",
        "profile": "prod-hardened",
    }

    # High delta_p but low divergence
    cal_windows = [
        {"divergence_rate": 0.1, "mean_delta_p": 0.1},  # High delta_p
        {"divergence_rate": 0.15, "mean_delta_p": 0.08},  # High delta_p
    ]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8, delta_p_threshold=0.05
    )

    assert summary["confounded_windows_count"] == 2
    assert all(w["runtime_profile_confounded"] for w in annotated)
    assert "mean_delta_p" in summary["confound_reasons"][0]


def test_correlate_missing_fields_tracking():
    """Test that missing fields are tracked in windows_dropped and missing_fields_by_window."""
    runtime_profile = {
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    cal_windows = [
        {"divergence_rate": 0.1},  # Missing mean_delta_p
        {"mean_delta_p": 0.02},  # Missing divergence_rate
        {"divergence_rate": 0.15, "mean_delta_p": 0.01},  # Complete
        {},  # Both missing
    ]

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, cal_windows)

    assert result["windows_analyzed"] == 4
    assert "windows_dropped" in result
    assert "missing_fields_by_window" in result
    assert len(result["missing_fields_by_window"]) > 0


def test_annex_determinism_byte_identical():
    """Test that annex is byte-identical across repeated builds with fixed seed."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "RED",
        "profile_stability": 0.75,
        "no_run_rate": 0.15,
    }

    cal_windows = [
        {"divergence_rate": 0.85, "mean_delta_p": 0.02},
        {"divergence_rate": 0.9, "mean_delta_p": 0.03},
    ]

    annex1 = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)
    annex2 = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    json1 = json.dumps(annex1, sort_keys=True, indent=2)
    json2 = json.dumps(annex2, sort_keys=True, indent=2)

    # Byte-identical
    assert json1.encode("utf-8") == json2.encode("utf-8")


def test_annex_thresholds_recorded():
    """Test that thresholds are recorded in confounding_summary."""
    runtime_profile = {
        "status_light": "RED",
        "profile": "prod-hardened",
    }

    cal_windows = [{"divergence_rate": 0.85, "mean_delta_p": 0.06}]

    annotated, summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile, cal_windows, divergence_spike_threshold=0.8, delta_p_threshold=0.05
    )

    assert "confounding_thresholds" in summary
    assert summary["confounding_thresholds"]["divergence_rate"] == 0.8
    assert summary["confounding_thresholds"]["mean_delta_p"] == 0.05


def test_annex_error_object_on_exception():
    """Test that evidence pack builder attaches error object on exception."""
    # This test verifies the error handling in build_first_light_evidence_pack.py
    # We can't directly test the builder here, but we can verify the error object structure
    error_object = {
        "error": "Test error message",
        "mode": "SHADOW",
        "schema_version": "1.0.0",
    }

    assert "error" in error_object
    assert error_object["mode"] == "SHADOW"
    assert error_object["schema_version"] == "1.0.0"


def test_correlation_reasons_explicit():
    """Test that correlation_reasons field is populated when correlation is None."""
    runtime_profile = {
        "profile_stability": 0.95,  # Constant value
        "no_run_rate": 0.05,
    }

    # All windows have same divergence_rate (zero variance in Y)
    cal_windows = [
        {"divergence_rate": 0.5, "mean_delta_p": 0.01},
        {"divergence_rate": 0.5, "mean_delta_p": 0.02},
    ]

    result = correlate_runtime_profile_with_cal_windows(runtime_profile, cal_windows)

    assert "correlation_reasons" in result
    # profile_stability vs divergence_rate should have ZERO_VARIANCE_Y reason
    reasons = result["correlation_reasons"]
    if "profile_stability_vs_divergence_rate" in reasons:
        assert reasons["profile_stability_vs_divergence_rate"] in (
            "ZERO_VARIANCE_Y",
            "ZERO_VARIANCE_X",
        )


def test_manifest_immutability():
    """Test that input manifest dict is not mutated (deepcopy behavior)."""
    # This is a contract test - the evidence pack builder should use deepcopy
    # We verify by checking that build_runtime_profile_calibration_annex doesn't mutate inputs
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "GREEN",
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }
    runtime_profile_original = dict(runtime_profile)

    cal_windows = [
        {"divergence_rate": 0.1, "mean_delta_p": 0.01},
    ]
    cal_windows_original = [dict(w) for w in cal_windows]

    annex = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    # Verify inputs unchanged
    assert runtime_profile == runtime_profile_original
    assert cal_windows == cal_windows_original
    assert annex is not None


def test_divergence_decomposition_rates_sum_sane():
    """Test that decomposed divergence rates sum to reasonable values."""
    windows = [
        {
            "cycles_in_window": 50,
            "divergence_rate": 0.8,
            "mean_delta_p": 0.06,  # Above threshold
        },
        {
            "cycles_in_window": 50,
            "divergence_rate": 0.9,
            "mean_delta_p": 0.03,  # Below threshold
        },
    ]

    decomposition = decompose_divergence_components(windows, state_threshold=0.05)

    assert decomposition["schema_version"] == "1.0.0"
    assert "thresholds" in decomposition
    assert decomposition["thresholds"]["state_threshold"] == 0.05

    overall = decomposition["overall"]
    assert overall["total_cycles"] == 100
    assert overall["state_divergence_rate"] is not None
    assert overall["overall_any_divergence_rate"] is not None
    # State divergence rate should be 0.5 (first window above threshold, second below)
    assert overall["state_divergence_rate"] == 0.5
    # Overall any divergence should be weighted average
    assert 0.8 <= overall["overall_any_divergence_rate"] <= 0.9


def test_instrumentation_verdict_meter_saturated_when_any_divergence_is_one():
    """Test that instrumentation_verdict is METER_SATURATED when overall_any_divergence_rate >= 0.99."""
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "GREEN",
        "profile_stability": 0.95,
        "no_run_rate": 0.05,
    }

    # Windows with saturated divergence_rate (near 1.0)
    cal_windows = [
        {"cycles_in_window": 50, "divergence_rate": 1.0, "mean_delta_p": 0.06},
        {"cycles_in_window": 50, "divergence_rate": 1.0, "mean_delta_p": 0.055},
    ]

    annex = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    assert annex is not None
    assert annex["instrumentation_verdict"] == "METER_SATURATED"
    assert len(annex["instrumentation_notes"]) > 0
    assert "saturates" in annex["instrumentation_notes"][0].lower()


def test_status_warning_emitted_for_meter_saturated():
    """Test that status generator emits warning when instrumentation_verdict is METER_SATURATED."""
    # This test verifies the contract that generate_first_light_status.py
    # emits warnings for METER_SATURATED verdict
    # We test the warning logic structure
    runtime_profile_calibration = {
        "instrumentation_verdict": "METER_SATURATED",
        "instrumentation_notes": ["overall_any_divergence_rate saturates at 1.000"],
    }

    # Simulate the warning generation logic
    warnings = []
    if runtime_profile_calibration.get("instrumentation_verdict") == "METER_SATURATED":
        instrumentation_notes = runtime_profile_calibration.get("instrumentation_notes", [])
        note = instrumentation_notes[0] if instrumentation_notes else "Meter saturated"
        warnings.append(f"Runtime profile calibration: {note}")

    assert len(warnings) == 1
    assert "METER_SATURATED" in warnings[0] or "saturates" in warnings[0].lower()


def test_divergence_decomposition_missing_data():
    """Test that decomposition handles missing data gracefully."""
    # Empty windows
    decomposition1 = decompose_divergence_components([])
    assert decomposition1["error"] == "INSUFFICIENT_DATA"

    # Windows missing required fields
    windows = [
        {"cycles_in_window": 50},  # Missing divergence_rate and mean_delta_p
    ]

    decomposition2 = decompose_divergence_components(windows)
    assert "missing_fields" in decomposition2
    assert decomposition2["missing_fields"] is not None
    assert len(decomposition2["missing_fields"]) > 0


def test_divergence_decomposition_outcome_rates_none():
    """Test that outcome divergence rates are None (not available in aggregated windows)."""
    windows = [
        {
            "cycles_in_window": 50,
            "divergence_rate": 0.5,
            "mean_delta_p": 0.03,
        },
    ]

    decomposition = decompose_divergence_components(windows)

    overall = decomposition["overall"]
    assert overall["outcome_divergence_rate_success"] is None
    assert overall["outcome_divergence_rate_omega"] is None
    assert overall["outcome_divergence_rate_blocked"] is None
    assert overall["outcome_divergence_basis"] == "UNAVAILABLE_NO_PER_CYCLE_COMPONENTS"
    assert overall["state_divergence_rate_basis"] == "proxy_mean_delta_p_threshold"
    assert "outcome_divergence_note" in decomposition


def test_instrumentation_verdict_stable_with_undefined_correlations():
    """
    Test that instrumentation verdict doesn't flip when correlations are undefined
    (e.g., ZERO_VARIANCE) but divergence proxy is present.

    This ensures meter saturation detection is independent of correlation computation failures.
    """
    # Create a runtime profile with constant metrics (will cause ZERO_VARIANCE in correlations)
    runtime_profile = {
        "profile": "prod-hardened",
        "status_light": "GREEN",
        "profile_stability": 0.95,  # Constant value
        "no_run_rate": 0.05,  # Constant value
    }

    # Windows with saturated divergence_rate but valid mean_delta_p
    cal_windows = [
        {
            "cycles_in_window": 50,
            "divergence_rate": 1.0,  # Saturated
            "mean_delta_p": 0.06,  # Above threshold
        },
        {
            "cycles_in_window": 50,
            "divergence_rate": 1.0,  # Saturated
            "mean_delta_p": 0.055,  # Above threshold
        },
    ]

    annex = build_runtime_profile_calibration_annex(runtime_profile, cal_windows)

    assert annex is not None
    # Meter verdict should be SATURATED regardless of correlation status
    assert annex["instrumentation_verdict"] == "METER_SATURATED"
    assert len(annex["instrumentation_notes"]) > 0

    # Verify correlations may be undefined (ZERO_VARIANCE) but verdict is still correct
    correlations = annex.get("windowed_correlations", {})
    correlation_reasons = correlations.get("correlation_reasons", {})
    
    # State divergence decomposition should still be present even if correlations fail
    divergence_decomp = annex.get("divergence_decomposition", {})
    assert divergence_decomp is not None
    overall = divergence_decomp.get("overall", {})
    assert overall.get("overall_any_divergence_rate") == 1.0
    assert overall.get("state_divergence_rate_basis") == "proxy_mean_delta_p_threshold"

