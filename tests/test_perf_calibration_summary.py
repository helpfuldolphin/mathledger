"""Tests for performance calibration summary.

Tests cover:
- Uplift shape classification (monotonic, plateau, oscillatory)
- Calibration summary generation from CAL-EXP-1/CAL-EXP-2 data
- Evidence pack attachment
- JSON serialization safety
"""

import json
import pytest
from experiments.verify_perf_equivalence import (
    classify_uplift_shape,
    map_uplift_shape_to_readiness_hint,
    build_perf_calibration_summary,
    attach_perf_calibration_summary,
    extract_calibration_readiness_status,
    _extract_hint_string,
)


class TestUpliftShapeClassification:
    """Test uplift shape classification from delta_p trajectory."""

    def test_classify_monotonic_increasing(self):
        """Test classification of monotonic increasing trajectory."""
        trajectory = [0.1, 0.2, 0.3, 0.4, 0.5]
        shape = classify_uplift_shape(trajectory)
        assert shape == "monotonic"

    def test_classify_monotonic_decreasing(self):
        """Test classification of monotonic decreasing trajectory."""
        trajectory = [0.5, 0.4, 0.3, 0.2, 0.1]
        shape = classify_uplift_shape(trajectory)
        assert shape == "monotonic"

    def test_classify_plateau(self):
        """Test classification of plateau (stable) trajectory."""
        trajectory = [0.3, 0.31, 0.29, 0.30, 0.30]
        shape = classify_uplift_shape(trajectory)
        assert shape == "plateau"

    def test_classify_oscillatory(self):
        """Test classification of oscillatory trajectory."""
        trajectory = [0.2, 0.4, 0.1, 0.5, 0.2, 0.4, 0.1]
        shape = classify_uplift_shape(trajectory)
        assert shape == "oscillatory"

    def test_classify_insufficient_data(self):
        """Test classification with insufficient data defaults to plateau."""
        shape = classify_uplift_shape([])
        assert shape == "plateau"

        shape = classify_uplift_shape([0.3])
        assert shape == "plateau"

    def test_classify_two_points(self):
        """Test classification with two points."""
        shape = classify_uplift_shape([0.1, 0.2])
        assert shape == "monotonic"

        shape = classify_uplift_shape([0.2, 0.1])
        assert shape == "monotonic"

        shape = classify_uplift_shape([0.2, 0.2])
        assert shape == "plateau"


class TestReadinessHintMapping:
    """Test uplift shape to readiness hint mapping."""

    def test_map_monotonic_decreasing_to_ready(self):
        """Test monotonic decreasing maps to READY_FOR_EXTENDED_RUN."""
        trajectory = [0.5, 0.4, 0.3, 0.2, 0.1]
        shape = classify_uplift_shape(trajectory)
        assert shape == "monotonic"
        hint_dict = map_uplift_shape_to_readiness_hint(shape, trajectory)
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "READY_FOR_EXTENDED_RUN"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_map_monotonic_increasing_to_ready(self):
        """Test monotonic increasing also maps to READY_FOR_EXTENDED_RUN."""
        trajectory = [0.1, 0.2, 0.3, 0.4, 0.5]
        shape = classify_uplift_shape(trajectory)
        assert shape == "monotonic"
        hint_dict = map_uplift_shape_to_readiness_hint(shape, trajectory)
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "READY_FOR_EXTENDED_RUN"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_map_plateau_to_parameter_tuning(self):
        """Test plateau maps to NEEDS_PARAMETER_TUNING."""
        trajectory = [0.3, 0.31, 0.29, 0.30, 0.30]
        shape = classify_uplift_shape(trajectory)
        assert shape == "plateau"
        hint_dict = map_uplift_shape_to_readiness_hint(shape, trajectory)
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint"] == "NEEDS_PARAMETER_TUNING"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_map_oscillatory_to_unstable(self):
        """Test oscillatory maps to UNSTABLE_CALIBRATION."""
        trajectory = [0.2, 0.4, 0.1, 0.5, 0.2, 0.4, 0.1]
        shape = classify_uplift_shape(trajectory)
        assert shape == "oscillatory"
        hint_dict = map_uplift_shape_to_readiness_hint(shape, trajectory)
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint"] == "UNSTABLE_CALIBRATION"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_map_unknown_shape_defaults_to_tuning(self):
        """Test unknown shape defaults to NEEDS_PARAMETER_TUNING."""
        hint_dict = map_uplift_shape_to_readiness_hint("unknown_shape")
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "NEEDS_PARAMETER_TUNING"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_map_without_trajectory(self):
        """Test mapping works without trajectory (uses shape only)."""
        hint_dict = map_uplift_shape_to_readiness_hint("monotonic")
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "READY_FOR_EXTENDED_RUN"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

        hint_dict = map_uplift_shape_to_readiness_hint("plateau")
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint"] == "NEEDS_PARAMETER_TUNING"

        hint_dict = map_uplift_shape_to_readiness_hint("oscillatory")
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint"] == "UNSTABLE_CALIBRATION"


class TestCalibrationSummary:
    """Test calibration summary generation."""

    def test_build_calibration_summary_cal_exp1_only(self):
        """Test building summary from CAL-EXP-1 data only."""
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.1, "start_cycle": 0, "end_cycle": 50},
                {"mean_delta_p": 0.2, "start_cycle": 50, "end_cycle": 100},
                {"mean_delta_p": 0.3, "start_cycle": 100, "end_cycle": 150},
            ]
        }

        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)

        assert summary["schema_version"] == "1.0.0"
        assert "cal_exp1" in summary["experiments"]
        assert "cal_exp1" in summary["uplift_shapes"]
        assert "cal_exp1" in summary["delta_p_trajectories"]
        assert "cal_exp1" in summary["calibration_readiness_hints"]
        assert summary["uplift_shapes"]["cal_exp1"] == "monotonic"
        assert summary["delta_p_trajectories"]["cal_exp1"] == [0.1, 0.2, 0.3]
        hint_dict = summary["calibration_readiness_hints"]["cal_exp1"]
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "READY_FOR_EXTENDED_RUN"
        assert hint_dict["basis"] == "delta_p_trajectory_shape"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_build_calibration_summary_both_experiments(self):
        """Test building summary from both CAL-EXP-1 and CAL-EXP-2."""
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.1},
                {"mean_delta_p": 0.2},
                {"mean_delta_p": 0.3},
            ]
        }
        cal_exp2_data = {
            "windows": [
                {"mean_delta_p": 0.3},
                {"mean_delta_p": 0.31},
                {"mean_delta_p": 0.29},
            ]
        }

        summary = build_perf_calibration_summary(
            cal_exp1_data=cal_exp1_data, cal_exp2_data=cal_exp2_data
        )

        assert "cal_exp1" in summary["experiments"]
        assert "cal_exp2" in summary["experiments"]
        assert summary["uplift_shapes"]["cal_exp1"] == "monotonic"
        assert summary["uplift_shapes"]["cal_exp2"] == "plateau"
        hint1 = summary["calibration_readiness_hints"]["cal_exp1"]
        hint2 = summary["calibration_readiness_hints"]["cal_exp2"]
        assert isinstance(hint1, dict)
        assert isinstance(hint2, dict)
        assert hint1["hint_schema_version"] == "1.0.0"
        assert hint2["hint_schema_version"] == "1.0.0"
        assert hint1["hint"] == "READY_FOR_EXTENDED_RUN"
        assert hint2["hint"] == "NEEDS_PARAMETER_TUNING"
        assert hint1["basis"] == "delta_p_trajectory_shape"
        assert hint2["basis"] == "delta_p_trajectory_shape"
        assert hint1["scope_note"] == "ADVISORY_ONLY_NO_GATE"
        assert hint2["scope_note"] == "ADVISORY_ONLY_NO_GATE"

    def test_build_calibration_summary_empty_data(self):
        """Test building summary with empty data."""
        summary = build_perf_calibration_summary()
        assert summary["schema_version"] == "1.0.0"
        assert summary["experiments"] == {}
        assert summary["uplift_shapes"] == {}
        assert summary["delta_p_trajectories"] == {}
        assert summary["calibration_readiness_hints"] == {}

    def test_build_calibration_summary_missing_mean_delta_p(self):
        """Test building summary when windows lack mean_delta_p."""
        cal_exp1_data = {
            "windows": [
                {"start_cycle": 0, "end_cycle": 50},
                {"start_cycle": 50, "end_cycle": 100},
            ]
        }

        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)
        # Should handle missing mean_delta_p gracefully
        assert "cal_exp1" not in summary["uplift_shapes"]


class TestEvidenceIntegration:
    """Test evidence pack integration."""

    def test_attach_calibration_summary(self):
        """Test attaching calibration summary to evidence pack."""
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.1},
                {"mean_delta_p": 0.2},
                {"mean_delta_p": 0.3},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_perf_calibration_summary(evidence, summary)

        assert "governance" in result
        assert "perf_calibration_summary" in result["governance"]
        assert result["governance"]["perf_calibration_summary"] == summary

    def test_attach_calibration_summary_read_only(self):
        """Test that attach_perf_calibration_summary is read-only."""
        summary = {
            "schema_version": "1.0.0",
            "experiments": {},
            "uplift_shapes": {},
            "delta_p_trajectories": {},
        }

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_perf_calibration_summary(evidence, summary)

        # Original evidence should be unchanged
        assert "governance" not in evidence

        # Result should have summary attached
        assert "governance" in result
        assert "perf_calibration_summary" in result["governance"]

    def test_attach_calibration_summary_json_serializable(self):
        """Test that evidence pack with calibration summary is JSON serializable."""
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.1},
                {"mean_delta_p": 0.2},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_perf_calibration_summary(evidence, summary)

        # Should serialize to JSON
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to round-trip
        parsed = json.loads(json_str)
        assert parsed == result
        assert "governance" in parsed
        assert "perf_calibration_summary" in parsed["governance"]
        assert "calibration_readiness_hints" in parsed["governance"]["perf_calibration_summary"]

    def test_readiness_hints_non_blocking(self):
        """Test that readiness hints are advisory only (non-blocking)."""
        # All hints should be present but not affect execution
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.2},
                {"mean_delta_p": 0.4},
                {"mean_delta_p": 0.1},
                {"mean_delta_p": 0.5},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)

        # Oscillatory should map to UNSTABLE_CALIBRATION
        hint_dict = summary["calibration_readiness_hints"]["cal_exp1"]
        assert isinstance(hint_dict, dict)
        assert hint_dict["hint_schema_version"] == "1.0.0"
        assert hint_dict["hint"] == "UNSTABLE_CALIBRATION"
        assert hint_dict["scope_note"] == "ADVISORY_ONLY_NO_GATE"

        # The hint is present but does not block anything
        # (This is a structural test - actual blocking would be in integration code)
        assert "calibration_readiness_hints" in summary

    def test_extract_calibration_readiness_status(self):
        """Test extraction of one-line status hint."""
        # Test with UNSTABLE (highest priority)
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.2},
                {"mean_delta_p": 0.4},
                {"mean_delta_p": 0.1},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)
        status = extract_calibration_readiness_status(summary)
        assert status == "UNSTABLE_CALIBRATION"

        # Test with NEEDS_TUNING
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.3},
                {"mean_delta_p": 0.31},
                {"mean_delta_p": 0.29},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)
        status = extract_calibration_readiness_status(summary)
        assert status == "NEEDS_PARAMETER_TUNING"

        # Test with READY
        cal_exp1_data = {
            "windows": [
                {"mean_delta_p": 0.5},
                {"mean_delta_p": 0.4},
                {"mean_delta_p": 0.3},
            ]
        }
        summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1_data)
        status = extract_calibration_readiness_status(summary)
        assert status == "READY_FOR_EXTENDED_RUN"

        # Test with multiple experiments (priority: UNSTABLE > TUNING > READY)
        cal_exp1_data = {
            "windows": [{"mean_delta_p": 0.1}, {"mean_delta_p": 0.2}]
        }
        cal_exp2_data = {
            "windows": [
                {"mean_delta_p": 0.2},
                {"mean_delta_p": 0.4},
                {"mean_delta_p": 0.1},
            ]
        }
        summary = build_perf_calibration_summary(
            cal_exp1_data=cal_exp1_data, cal_exp2_data=cal_exp2_data
        )
        status = extract_calibration_readiness_status(summary)
        assert status == "UNSTABLE_CALIBRATION"  # UNSTABLE takes priority

        # Test with empty summary
        empty_summary = {
            "schema_version": "1.0.0",
            "experiments": {},
            "uplift_shapes": {},
            "delta_p_trajectories": {},
            "calibration_readiness_hints": {},
        }
        status = extract_calibration_readiness_status(empty_summary)
        assert status == "UNKNOWN"

    def test_extract_calibration_readiness_status_backward_compat(self):
        """Test status extraction with old string format (backward compatibility)."""
        # Test with old string format
        old_format_summary = {
            "schema_version": "1.0.0",
            "calibration_readiness_hints": {
                "cal_exp1": "UNSTABLE_CALIBRATION",  # Old string format
                "cal_exp2": "READY_FOR_EXTENDED_RUN",  # Old string format
            },
        }
        status = extract_calibration_readiness_status(old_format_summary)
        assert status == "UNSTABLE_CALIBRATION"  # Should handle old format

        # Test with mixed format (old + new)
        mixed_format_summary = {
            "schema_version": "1.0.0",
            "calibration_readiness_hints": {
                "cal_exp1": "NEEDS_PARAMETER_TUNING",  # Old string format
                "cal_exp2": {  # New dict format
                    "hint": "UNSTABLE_CALIBRATION",
                    "basis": "delta_p_trajectory_shape",
                    "scope_note": "ADVISORY_ONLY_NO_GATE",
                },
            },
        }
        status = extract_calibration_readiness_status(mixed_format_summary)
        assert status == "UNSTABLE_CALIBRATION"  # Should handle both formats

    def test_mixed_format_ingestion_same_run(self):
        """Test mixed-format ingestion (old string + new dict) in same run."""
        # Simulate a calibration summary with mixed formats from different sources
        mixed_format_summary = {
            "schema_version": "1.0.0",
            "calibration_readiness_hints": {
                "cal_exp1": "READY_FOR_EXTENDED_RUN",  # Old string format (legacy)
                "cal_exp2": {  # New dict format (v1.0.0)
                    "hint_schema_version": "1.0.0",
                    "hint": "UNSTABLE_CALIBRATION",
                    "basis": "delta_p_trajectory_shape",
                    "scope_note": "ADVISORY_ONLY_NO_GATE",
                },
                "cal_exp3": {  # New dict format (v1.0.0)
                    "hint_schema_version": "1.0.0",
                    "hint": "NEEDS_PARAMETER_TUNING",
                    "basis": "delta_p_trajectory_shape",
                    "scope_note": "ADVISORY_ONLY_NO_GATE",
                },
            },
        }

        # Status extraction should handle mixed formats correctly
        status = extract_calibration_readiness_status(mixed_format_summary)
        assert status == "UNSTABLE_CALIBRATION"  # Highest priority hint

        # Verify old format is handled
        hint1 = _extract_hint_string(mixed_format_summary["calibration_readiness_hints"]["cal_exp1"])
        assert hint1 == "READY_FOR_EXTENDED_RUN"

        # Verify new format is handled
        hint2 = _extract_hint_string(mixed_format_summary["calibration_readiness_hints"]["cal_exp2"])
        assert hint2 == "UNSTABLE_CALIBRATION"

        hint3 = _extract_hint_string(mixed_format_summary["calibration_readiness_hints"]["cal_exp3"])
        assert hint3 == "NEEDS_PARAMETER_TUNING"



