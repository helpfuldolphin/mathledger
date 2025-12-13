"""
Tests for harmonic curriculum status signal extraction and GGFL adapter.

STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

Tests that:
- Status signal extraction works correctly
- Delta fields are included when present
- Warning generation is capped at one warning
- GGFL adapter produces correct format
- All outputs are JSON-safe, deterministic, and non-mutating
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from backend.health.harmonic_alignment_p3p4_integration import (
    check_harmonic_curriculum_warning,
    extract_harmonic_curriculum_signal_for_status,
    harmonic_grid_for_alignment_view,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_panel() -> dict[str, Any]:
    """Sample harmonic curriculum panel without delta."""
    return {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "harmonic_band_counts": {
            "COHERENT": 1,
            "PARTIAL": 1,
            "MISMATCHED": 1,
        },
        "top_driver_concepts": ["slice_b", "slice_a", "slice_d"],
        "top_driver_cal_ids": {
            "slice_b": ["CAL-EXP-2", "CAL-EXP-3"],
            "slice_a": ["CAL-EXP-1"],
            "slice_d": ["CAL-EXP-3"],
        },
    }


@pytest.fixture
def sample_panel_with_delta() -> dict[str, Any]:
    """Sample harmonic curriculum panel with delta."""
    return {
        "schema_version": "1.0.0",
        "num_experiments": 3,
        "harmonic_band_counts": {
            "COHERENT": 2,
            "PARTIAL": 1,
            "MISMATCHED": 0,
        },
        "top_driver_concepts": ["slice_a", "slice_b"],
        "top_driver_cal_ids": {
            "slice_a": ["CAL-EXP-1", "CAL-EXP-2"],
            "slice_b": ["CAL-EXP-1"],
        },
        "delta": {
            "schema_version": "1.0.0",
            "top_driver_overlap": ["slice_a", "slice_b"],
            "top_driver_only_mock": ["slice_c"],
            "top_driver_only_real": ["slice_d"],
            "frequency_shifts": [
                {"concept": "slice_a", "mock_frequency": 2, "real_frequency": 3, "delta": 1},
                {"concept": "slice_b", "mock_frequency": 1, "real_frequency": 1, "delta": 0},
            ],
        },
    }


# =============================================================================
# TEST GROUP 1: STATUS SIGNAL EXTRACTION
# =============================================================================


class TestExtractHarmonicCurriculumSignalForStatus:
    """Tests for extract_harmonic_curriculum_signal_for_status."""

    def test_01_signal_has_required_keys(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal has all required keys."""
        signal = extract_harmonic_curriculum_signal_for_status(sample_panel)

        required_keys = {
            "schema_version",
            "mode",
            "extraction_source",
            "band_counts",
            "top_driver_concepts_top5",
        }
        assert required_keys.issubset(set(signal.keys()))

    def test_02_signal_extracts_correct_values(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal extracts correct values from panel."""
        signal = extract_harmonic_curriculum_signal_for_status(
            sample_panel, extraction_source="MANIFEST"
        )

        assert signal["schema_version"] == "1.0.0"
        assert signal["mode"] == "SHADOW"
        assert signal["extraction_source"] == "MANIFEST"
        assert signal["band_counts"]["MISMATCHED"] == 1
        assert signal["top_driver_concepts_top5"] == ["slice_b", "slice_a", "slice_d"]

    def test_03_signal_includes_delta_when_present(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that signal includes delta when present in panel."""
        signal = extract_harmonic_curriculum_signal_for_status(sample_panel_with_delta)

        assert "delta" in signal
        assert "top_driver_overlap_top5" in signal["delta"]
        assert "frequency_shift_top3" in signal["delta"]
        assert len(signal["delta"]["top_driver_overlap_top5"]) <= 5
        assert len(signal["delta"]["frequency_shift_top3"]) <= 3

    def test_04_signal_omits_delta_when_absent(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal omits delta when not present in panel."""
        signal = extract_harmonic_curriculum_signal_for_status(sample_panel)

        assert "delta" not in signal

    def test_05_signal_limits_top_drivers_to_5(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal limits top_driver_concepts_top5 to 5."""
        # Create panel with more than 5 drivers
        large_panel = dict(sample_panel)
        large_panel["top_driver_concepts"] = [f"slice_{i}" for i in range(10)]

        signal = extract_harmonic_curriculum_signal_for_status(large_panel)

        assert len(signal["top_driver_concepts_top5"]) <= 5

    def test_06_signal_limits_frequency_shifts_to_3(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that signal limits frequency_shift_top3 to 3."""
        # Create panel with more than 3 shifts
        large_panel = dict(sample_panel_with_delta)
        large_panel["delta"]["frequency_shifts"] = [
            {"concept": f"slice_{i}", "mock_frequency": 1, "real_frequency": 1, "delta": 0}
            for i in range(10)
        ]

        signal = extract_harmonic_curriculum_signal_for_status(large_panel)

        assert len(signal["delta"]["frequency_shift_top3"]) <= 3

    def test_07_signal_serializes_to_json(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that signal can be serialized to JSON."""
        signal = extract_harmonic_curriculum_signal_for_status(sample_panel_with_delta)

        json_str = json.dumps(signal)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == signal

    def test_08_signal_is_deterministic(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that signal is deterministic."""
        signals = [
            extract_harmonic_curriculum_signal_for_status(
                sample_panel_with_delta, extraction_source="MANIFEST"
            )
            for _ in range(5)
        ]

        for i in range(1, len(signals)):
            assert signals[0] == signals[i]

    def test_09_signal_includes_extraction_source(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal includes extraction_source field."""
        signal_manifest = extract_harmonic_curriculum_signal_for_status(
            sample_panel, extraction_source="MANIFEST"
        )
        signal_evidence = extract_harmonic_curriculum_signal_for_status(
            sample_panel, extraction_source="EVIDENCE_JSON"
        )
        signal_missing = extract_harmonic_curriculum_signal_for_status(
            sample_panel, extraction_source="MISSING"
        )

        assert signal_manifest["extraction_source"] == "MANIFEST"
        assert signal_evidence["extraction_source"] == "EVIDENCE_JSON"
        assert signal_missing["extraction_source"] == "MISSING"


# =============================================================================
# TEST GROUP 2: WARNING GENERATION
# =============================================================================


class TestCheckHarmonicCurriculumWarning:
    """Tests for check_harmonic_curriculum_warning."""

    def test_09_warning_for_mismatched_count(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that warning is generated when MISMATCHED count > 0."""
        warning = check_harmonic_curriculum_warning(sample_panel)

        assert warning is not None
        assert "MISMATCHED" in warning
        assert "3" in warning  # num_experiments

    def test_10_no_warning_when_no_mismatched(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that no warning is generated when MISMATCHED count == 0."""
        warning = check_harmonic_curriculum_warning(sample_panel_with_delta)

        assert warning is None

    def test_11_warning_for_frequency_shift(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that warning is generated for frequency shift >= threshold."""
        # Modify delta to have large shift
        panel = dict(sample_panel_with_delta)
        panel["harmonic_band_counts"]["MISMATCHED"] = 0  # No mismatched
        panel["delta"]["frequency_shifts"] = [
            {"concept": "slice_a", "mock_frequency": 1, "real_frequency": 4, "delta": 3},
        ]

        warning = check_harmonic_curriculum_warning(panel, frequency_shift_threshold=2)

        assert warning is not None
        assert "frequency shift" in warning.lower()
        assert "slice_a" in warning

    def test_12_warning_prioritizes_mismatched_over_delta(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that MISMATCHED warning takes priority over delta warning."""
        panel = dict(sample_panel_with_delta)
        panel["harmonic_band_counts"]["MISMATCHED"] = 1  # Has mismatched
        panel["delta"]["frequency_shifts"] = [
            {"concept": "slice_a", "mock_frequency": 1, "real_frequency": 4, "delta": 3},
        ]

        warning = check_harmonic_curriculum_warning(panel, frequency_shift_threshold=2)

        assert warning is not None
        assert "MISMATCHED" in warning
        assert "frequency shift" not in warning.lower()

    def test_13_warning_caps_at_one(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that only one warning is returned (not a list)."""
        warning = check_harmonic_curriculum_warning(sample_panel)

        assert warning is None or isinstance(warning, str)
        if warning:
            # Should be a single string, not a list
            assert isinstance(warning, str)


# =============================================================================
# TEST GROUP 3: GGFL ADAPTER
# =============================================================================


class TestHarmonicGridForAlignmentView:
    """Tests for harmonic_grid_for_alignment_view."""

    def test_14_ggfl_has_required_keys(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output has all required keys."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        required_keys = {
            "signal_type",
            "status",
            "conflict",
            "drivers",
            "summary",
        }
        assert required_keys.issubset(set(ggfl.keys()))

    def test_15_ggfl_signal_type_is_sig_har(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that signal_type is SIG-HAR."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert ggfl["signal_type"] == "SIG-HAR"

    def test_16_ggfl_conflict_is_false(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that conflict is always False."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert ggfl["conflict"] is False

    def test_17_ggfl_status_warn_with_mismatched(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that status is 'warn' when MISMATCHED count > 0."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert ggfl["status"] == "warn"

    def test_18_ggfl_status_ok_without_mismatched(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that status is 'ok' when MISMATCHED count == 0."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel_with_delta)

        assert ggfl["status"] == "ok"

    def test_19_ggfl_drivers_are_reason_codes(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that drivers are reason codes, not concept strings."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        # Should contain reason codes (frozen set)
        assert len(ggfl["drivers"]) <= 3
        valid_driver_codes = {
            "DRIVER_MISMATCHED_PRESENT",
            "DRIVER_DELTA_SHIFT_PRESENT",
            "DRIVER_TOP_CONCEPTS_PRESENT",
        }
        for driver in ggfl["drivers"]:
            assert driver in valid_driver_codes, f"Invalid driver code: {driver}"

    def test_29_ggfl_driver_codes_frozen(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that driver codes are exactly the frozen set (enforce contract)."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        # Enforce exact driver codes (no drift)
        valid_driver_codes = {
            "DRIVER_MISMATCHED_PRESENT",
            "DRIVER_DELTA_SHIFT_PRESENT",
            "DRIVER_TOP_CONCEPTS_PRESENT",
        }
        for driver in ggfl["drivers"]:
            assert driver in valid_driver_codes, f"Driver code '{driver}' not in frozen set"

    def test_30_ggfl_has_top_reason_code(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output includes top_reason_code."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert "top_reason_code" in ggfl
        top_reason = ggfl["top_reason_code"]
        # Should be first driver or "NONE"
        if ggfl["drivers"]:
            assert top_reason == ggfl["drivers"][0]
        else:
            assert top_reason == "NONE"

    def test_20_ggfl_drivers_deterministic_ordering(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that drivers follow deterministic ordering: mismatched → delta shift → top concepts."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        # Should have MISMATCHED first (if present)
        if "DRIVER_MISMATCHED_PRESENT" in ggfl["drivers"]:
            assert ggfl["drivers"][0] == "DRIVER_MISMATCHED_PRESENT"

    def test_21_ggfl_drivers_include_delta_shift_when_triggered(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that DRIVER_DELTA_SHIFT_PRESENT appears when threshold triggered."""
        # Create panel with delta shift >= threshold
        panel = dict(sample_panel_with_delta)
        panel["harmonic_band_counts"]["MISMATCHED"] = 0  # No mismatched
        panel["delta"]["frequency_shifts"] = [
            {"concept": "slice_b", "mock_frequency": 1, "real_frequency": 4, "delta": 3},
        ]

        ggfl = harmonic_grid_for_alignment_view(panel, frequency_shift_threshold=2)

        assert "DRIVER_DELTA_SHIFT_PRESENT" in ggfl["drivers"]

    def test_22_ggfl_drivers_exclude_delta_shift_below_threshold(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that DRIVER_DELTA_SHIFT_PRESENT does not appear when below threshold."""
        # Create panel with delta shift < threshold
        panel = dict(sample_panel_with_delta)
        panel["harmonic_band_counts"]["MISMATCHED"] = 0  # No mismatched
        panel["delta"]["frequency_shifts"] = [
            {"concept": "slice_b", "mock_frequency": 1, "real_frequency": 2, "delta": 1},
        ]

        ggfl = harmonic_grid_for_alignment_view(panel, frequency_shift_threshold=2)

        assert "DRIVER_DELTA_SHIFT_PRESENT" not in ggfl["drivers"]

    def test_23_ggfl_has_shadow_mode_invariants(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output includes shadow_mode_invariants with global schema."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert "shadow_mode_invariants" in ggfl
        invariants = ggfl["shadow_mode_invariants"]
        assert invariants.get("advisory_only") is True
        assert invariants.get("no_enforcement") is True
        assert invariants.get("conflict_invariant") is True

    def test_24_ggfl_has_weight_hint(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output includes weight_hint: LOW."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert ggfl["weight_hint"] == "LOW"

    def test_25_ggfl_summary_is_neutral(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that summary is a neutral sentence."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        assert isinstance(ggfl["summary"], str)
        assert len(ggfl["summary"]) > 0
        # Should not contain judgmental terms
        assert "good" not in ggfl["summary"].lower()
        assert "bad" not in ggfl["summary"].lower()
        assert "healthy" not in ggfl["summary"].lower()
        assert "unhealthy" not in ggfl["summary"].lower()

    def test_26_ggfl_works_with_signal(
        self, sample_panel_with_delta: dict[str, Any]
    ) -> None:
        """Test that GGFL adapter works with status signal."""
        signal = extract_harmonic_curriculum_signal_for_status(
            sample_panel_with_delta, extraction_source="MANIFEST"
        )
        ggfl = harmonic_grid_for_alignment_view(signal)

        assert ggfl["signal_type"] == "SIG-HAR"
        assert ggfl["status"] in ["ok", "warn"]

    def test_27_ggfl_is_deterministic(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output is deterministic."""
        ggfls = [harmonic_grid_for_alignment_view(sample_panel) for _ in range(5)]

        for i in range(1, len(ggfls)):
            assert ggfls[0] == ggfls[i]

    def test_28_ggfl_serializes_to_json(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that GGFL output can be serialized to JSON."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        json_str = json.dumps(ggfl)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == ggfl

    def test_31_ggfl_invariants_match_global_schema(
        self, sample_panel: dict[str, Any]
    ) -> None:
        """Test that shadow_mode_invariants match global 3-boolean pattern."""
        ggfl = harmonic_grid_for_alignment_view(sample_panel)

        invariants = ggfl["shadow_mode_invariants"]
        # Must match global schema exactly
        assert invariants == {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        }

