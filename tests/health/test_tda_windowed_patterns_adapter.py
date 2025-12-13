"""
Tests for TDA Windowed Patterns Adapter

Tests for:
- extract_tda_windowed_patterns_signal_for_status()
- extract_tda_windowed_patterns_warnings()
- check_single_shot_windowed_disagreement()
- extract_pattern_disagreement_for_status()

All tests verify SHADOW MODE markers and deterministic behavior.
"""

import pytest
from typing import Any, Dict, Optional

from backend.health.tda_windowed_patterns_adapter import (
    extract_tda_windowed_patterns_signal_for_status,
    extract_tda_windowed_patterns_warnings,
    check_single_shot_windowed_disagreement,
    extract_pattern_disagreement_for_status,
    tda_windowed_patterns_for_alignment_view,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_windowed_patterns_signal(
    dominant_pattern: str = "NONE",
    max_streak_pattern: str = "NONE",
    max_streak_length: int = 0,
    high_confidence_count: int = 0,
    total_windows: int = 10,
    windows_with_patterns: int = 0,
    top_events_count: int = 0,
) -> Dict[str, Any]:
    """Create a windowed patterns signal for testing."""
    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "status": {
            "dominant_pattern": dominant_pattern,
            "max_streak": {
                "pattern": max_streak_pattern,
                "length": max_streak_length,
            },
            "high_confidence_count": high_confidence_count,
            "coverage": {
                "total_windows": total_windows,
                "windows_with_patterns": windows_with_patterns,
            },
        },
        "top_events": [{"window_index": i, "pattern": dominant_pattern, "confidence": 0.8}
                       for i in range(top_events_count)],
        "top_events_count": top_events_count,
    }


def make_manifest_with_windowed_patterns(
    windowed_patterns: Optional[Dict[str, Any]] = None,
    single_shot_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a manifest with windowed patterns in signals section."""
    manifest: Dict[str, Any] = {"signals": {}, "governance": {"tda": {}}}
    if windowed_patterns:
        manifest["signals"]["tda_windowed_patterns"] = windowed_patterns
    if single_shot_pattern:
        manifest["governance"]["tda"]["patterns"] = {"pattern": single_shot_pattern}
    return manifest


def make_evidence_with_windowed_patterns(
    windowed_patterns: Optional[Dict[str, Any]] = None,
    single_shot_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Create evidence.json with windowed patterns in signals section."""
    evidence: Dict[str, Any] = {"signals": {}, "governance": {"tda": {}}}
    if windowed_patterns:
        evidence["signals"]["tda_windowed_patterns"] = windowed_patterns
    if single_shot_pattern:
        evidence["governance"]["tda"]["patterns"] = {"pattern": single_shot_pattern}
    return evidence


# =============================================================================
# Test: extract_tda_windowed_patterns_signal_for_status()
# =============================================================================

class TestExtractTdaWindowedPatternsSignalForStatus:
    """Tests for signal extraction."""

    def test_returns_none_when_no_data(self):
        """Returns None when no manifest or evidence."""
        result = extract_tda_windowed_patterns_signal_for_status(None, None)
        assert result is None

    def test_returns_none_when_empty_data(self):
        """Returns None when empty manifest and evidence."""
        result = extract_tda_windowed_patterns_signal_for_status({}, {})
        assert result is None

    def test_extracts_from_manifest_signals(self):
        """Extracts signal from manifest.signals.tda_windowed_patterns."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="STRUCTURAL_BREAK",
            high_confidence_count=3,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_signal_for_status(manifest, None)

        assert result is not None
        assert result["dominant_pattern"] == "STRUCTURAL_BREAK"
        assert result["high_confidence_count"] == 3
        assert result["extraction_source"] == "MANIFEST"
        assert result["mode"] == "SHADOW"

    def test_extracts_from_evidence_fallback(self):
        """Falls back to evidence.json when manifest missing."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="DRIFT",
            max_streak_length=5,
        )
        evidence = make_evidence_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_signal_for_status(None, evidence)

        assert result is not None
        assert result["dominant_pattern"] == "DRIFT"
        assert result["max_streak"]["length"] == 5
        assert result["extraction_source"] == "EVIDENCE_JSON"

    def test_prefers_manifest_over_evidence(self):
        """Prefers manifest source over evidence fallback."""
        manifest_windowed = make_windowed_patterns_signal(dominant_pattern="STRUCTURAL_BREAK")
        evidence_windowed = make_windowed_patterns_signal(dominant_pattern="DRIFT")

        manifest = make_manifest_with_windowed_patterns(manifest_windowed)
        evidence = make_evidence_with_windowed_patterns(evidence_windowed)

        result = extract_tda_windowed_patterns_signal_for_status(manifest, evidence)

        assert result["dominant_pattern"] == "STRUCTURAL_BREAK"
        assert result["extraction_source"] == "MANIFEST"

    def test_extracts_all_required_fields(self):
        """Extracts all required fields for status."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="NOISE_AMPLIFICATION",
            max_streak_pattern="NOISE_AMPLIFICATION",
            max_streak_length=3,
            high_confidence_count=2,
            total_windows=20,
            windows_with_patterns=5,
            top_events_count=3,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_signal_for_status(manifest, None)

        assert result["schema_version"] == "1.0.0"
        assert result["mode"] == "SHADOW"
        assert result["dominant_pattern"] == "NOISE_AMPLIFICATION"
        assert result["max_streak"]["pattern"] == "NOISE_AMPLIFICATION"
        assert result["max_streak"]["length"] == 3
        assert result["high_confidence_count"] == 2
        assert result["top_events_count"] == 3
        assert result["coverage"]["total_windows"] == 20
        assert result["coverage"]["windows_with_patterns"] == 5

    def test_handles_flat_structure(self):
        """Handles flat structure without nested status object."""
        windowed = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "dominant_pattern": "PHASE_LAG",
            "max_streak_pattern": "PHASE_LAG",
            "max_streak_length": 2,
            "high_confidence_count": 1,
            "total_windows": 10,
        }
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_signal_for_status(manifest, None)

        assert result["dominant_pattern"] == "PHASE_LAG"


# =============================================================================
# Test: extract_tda_windowed_patterns_warnings()
# =============================================================================

class TestExtractTdaWindowedPatternsWarnings:
    """Tests for warning extraction."""

    def test_returns_empty_when_no_signal(self):
        """Returns empty list when no signal found."""
        result = extract_tda_windowed_patterns_warnings(None, None)
        assert result == []

    def test_returns_empty_when_dominant_is_none(self):
        """Returns empty when dominant pattern is NONE and no high_conf."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="NONE",
            high_confidence_count=0,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_warnings(manifest, None)

        assert result == []

    def test_warns_when_dominant_not_none(self):
        """Generates warning when dominant pattern is not NONE."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="STRUCTURAL_BREAK",
            max_streak_pattern="STRUCTURAL_BREAK",
            max_streak_length=3,
            high_confidence_count=0,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_warnings(manifest, None)

        assert len(result) == 1
        assert "STRUCTURAL_BREAK" in result[0]
        assert "dominant=" in result[0]

    def test_warns_when_high_confidence_positive(self):
        """Generates warning when high_confidence_count > 0."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="NONE",
            high_confidence_count=2,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_warnings(manifest, None)

        assert len(result) == 1
        assert "high_conf=2" in result[0]

    def test_warning_includes_streak_info(self):
        """Warning includes streak info when streak > 1."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="DRIFT",
            max_streak_pattern="DRIFT",
            max_streak_length=5,
            high_confidence_count=1,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_warnings(manifest, None)

        assert len(result) == 1
        assert "streak=DRIFT(5)" in result[0]

    def test_warning_capped_to_one_line(self):
        """Warning is capped to one line maximum."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="STRUCTURAL_BREAK",
            max_streak_pattern="STRUCTURAL_BREAK",
            max_streak_length=10,
            high_confidence_count=5,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_warnings(manifest, None)

        assert len(result) <= 1


# =============================================================================
# Test: check_single_shot_windowed_disagreement()
# =============================================================================

class TestCheckSingleShotWindowedDisagreement:
    """Tests for disagreement detection."""

    def test_returns_none_when_both_inputs_none(self):
        """Returns None when both inputs are None."""
        result = check_single_shot_windowed_disagreement(None, None)
        assert result is None

    def test_returns_none_when_single_shot_none(self):
        """Returns None when single_shot is None."""
        result = check_single_shot_windowed_disagreement(None, "DRIFT")
        assert result is None

    def test_returns_none_when_windowed_none(self):
        """Returns None when windowed is None."""
        result = check_single_shot_windowed_disagreement("NONE", None)
        assert result is None

    def test_returns_none_when_both_none_pattern(self):
        """Returns None when both are NONE pattern (no disagreement)."""
        result = check_single_shot_windowed_disagreement("NONE", "NONE")
        assert result is None

    def test_returns_none_when_both_non_none(self):
        """Returns None when both have patterns (no disagreement)."""
        result = check_single_shot_windowed_disagreement("DRIFT", "STRUCTURAL_BREAK")
        assert result is None

    def test_detects_windowed_detected_pattern(self):
        """Detects when windowed finds pattern but single-shot says NONE."""
        result = check_single_shot_windowed_disagreement("NONE", "STRUCTURAL_BREAK")

        assert result is not None
        assert result["disagreement_detected"] is True
        assert result["reason_code"] == "DRIVER_WINDOWED_DETECTED_PATTERN"
        assert result["single_shot_pattern"] == "NONE"
        assert result["windowed_dominant_pattern"] == "STRUCTURAL_BREAK"
        assert result["mode"] == "SHADOW"

    def test_detects_single_shot_detected_pattern(self):
        """Detects when single-shot finds pattern but windowed says NONE."""
        result = check_single_shot_windowed_disagreement("DRIFT", "NONE")

        assert result is not None
        assert result["disagreement_detected"] is True
        assert result["reason_code"] == "DRIVER_SINGLE_SHOT_DETECTED_PATTERN"
        assert result["single_shot_pattern"] == "DRIFT"
        assert result["windowed_dominant_pattern"] == "NONE"

    def test_case_insensitive(self):
        """Pattern comparison is case insensitive."""
        result = check_single_shot_windowed_disagreement("none", "drift")

        assert result is not None
        assert result["reason_code"] == "DRIVER_WINDOWED_DETECTED_PATTERN"

    def test_reason_code_format(self):
        """Reason code uses DRIVER_ prefix format."""
        result = check_single_shot_windowed_disagreement("NONE", "DRIFT")

        assert result["reason_code"].startswith("DRIVER_")
        assert "advisory_note" not in result  # No prose


# =============================================================================
# Test: extract_pattern_disagreement_for_status()
# =============================================================================

class TestExtractPatternDisagreementForStatus:
    """Tests for disagreement extraction from evidence."""

    def test_returns_none_when_no_patterns(self):
        """Returns None when no patterns in evidence."""
        result = extract_pattern_disagreement_for_status(None, None)
        assert result is None

    def test_extracts_disagreement_from_manifest(self):
        """Extracts disagreement when patterns in manifest disagree."""
        windowed = make_windowed_patterns_signal(dominant_pattern="STRUCTURAL_BREAK")
        manifest = make_manifest_with_windowed_patterns(windowed, single_shot_pattern="NONE")

        result = extract_pattern_disagreement_for_status(manifest, None)

        assert result is not None
        assert result["disagreement_detected"] is True
        assert result["reason_code"] == "DRIVER_WINDOWED_DETECTED_PATTERN"

    def test_extracts_disagreement_from_evidence(self):
        """Extracts disagreement when patterns in evidence disagree."""
        windowed = make_windowed_patterns_signal(dominant_pattern="DRIFT")
        evidence = make_evidence_with_windowed_patterns(windowed, single_shot_pattern="NONE")

        result = extract_pattern_disagreement_for_status(None, evidence)

        assert result is not None
        assert result["disagreement_detected"] is True

    def test_returns_none_when_no_disagreement(self):
        """Returns None when patterns agree."""
        windowed = make_windowed_patterns_signal(dominant_pattern="NONE")
        manifest = make_manifest_with_windowed_patterns(windowed, single_shot_pattern="NONE")

        result = extract_pattern_disagreement_for_status(manifest, None)

        assert result is None


# =============================================================================
# Test: SHADOW MODE Invariants
# =============================================================================

class TestShadowModeInvariants:
    """Tests ensuring SHADOW MODE is always respected."""

    def test_signal_always_shadow(self):
        """Signal always has mode='SHADOW'."""
        windowed = make_windowed_patterns_signal(dominant_pattern="DRIFT")
        manifest = make_manifest_with_windowed_patterns(windowed)

        result = extract_tda_windowed_patterns_signal_for_status(manifest, None)

        assert result["mode"] == "SHADOW"

    def test_disagreement_always_shadow(self):
        """Disagreement always has mode='SHADOW'."""
        result = check_single_shot_windowed_disagreement("NONE", "DRIFT")

        assert result["mode"] == "SHADOW"


# =============================================================================
# Test: Deterministic Behavior
# =============================================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior."""

    def test_signal_extraction_deterministic(self):
        """Signal extraction produces identical results."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="STRUCTURAL_BREAK",
            max_streak_length=5,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result1 = extract_tda_windowed_patterns_signal_for_status(manifest, None)
        result2 = extract_tda_windowed_patterns_signal_for_status(manifest, None)

        assert result1 == result2

    def test_warning_extraction_deterministic(self):
        """Warning extraction produces identical results."""
        windowed = make_windowed_patterns_signal(
            dominant_pattern="DRIFT",
            high_confidence_count=3,
        )
        manifest = make_manifest_with_windowed_patterns(windowed)

        result1 = extract_tda_windowed_patterns_warnings(manifest, None)
        result2 = extract_tda_windowed_patterns_warnings(manifest, None)

        assert result1 == result2

    def test_disagreement_check_deterministic(self):
        """Disagreement check produces identical results."""
        result1 = check_single_shot_windowed_disagreement("NONE", "DRIFT")
        result2 = check_single_shot_windowed_disagreement("NONE", "DRIFT")

        assert result1 == result2


# =============================================================================
# Test: GGFL Adapter tda_windowed_patterns_for_alignment_view()
# =============================================================================

class TestTdaWindowedPatternsForAlignmentView:
    """Tests for GGFL adapter function."""

    def test_returns_ok_when_no_signal(self):
        """Returns ok status when no signal provided."""
        result = tda_windowed_patterns_for_alignment_view(None)

        assert result["signal_type"] == "SIG-TDAW"
        assert result["status"] == "ok"
        assert result["conflict"] is False
        assert result["weight_hint"] == "LOW"
        assert result["extraction_source"] == "MISSING"
        assert result["drivers"] == []
        assert "No TDA windowed patterns signal available" in result["summary"]

    def test_returns_ok_when_dominant_none(self):
        """Returns ok status when dominant pattern is NONE."""
        signal = {
            "dominant_pattern": "NONE",
            "max_streak": {"pattern": "NONE", "length": 0},
            "high_confidence_count": 0,
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["signal_type"] == "SIG-TDAW"
        assert result["status"] == "ok"
        assert result["conflict"] is False
        assert "no dominant patterns" in result["summary"]

    def test_returns_warn_when_dominant_not_none(self):
        """Returns warn status when dominant pattern is not NONE."""
        signal = {
            "dominant_pattern": "STRUCTURAL_BREAK",
            "max_streak": {"pattern": "STRUCTURAL_BREAK", "length": 3},
            "high_confidence_count": 2,
            "extraction_source": "MANIFEST",
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["signal_type"] == "SIG-TDAW"
        assert result["status"] == "warn"
        assert result["conflict"] is False
        assert result["extraction_source"] == "MANIFEST"
        assert "DRIVER_DOMINANT_PATTERN:STRUCTURAL_BREAK" in result["drivers"]
        assert "STRUCTURAL_BREAK" in result["summary"]

    def test_returns_warn_when_disagreement_present(self):
        """Returns warn status when disagreement is present."""
        signal = {
            "dominant_pattern": "NONE",
            "max_streak": {"pattern": "NONE", "length": 0},
            "high_confidence_count": 0,
            "extraction_source": "EVIDENCE_JSON",
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
        }

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        assert result["status"] == "warn"
        assert result["extraction_source"] == "EVIDENCE_JSON"
        assert "DRIVER_WINDOWED_DETECTED_PATTERN" in result["drivers"]
        assert "disagreement" in result["summary"]

    def test_drivers_capped_to_three(self):
        """Drivers list is capped to 3 items."""
        signal = {
            "dominant_pattern": "DRIFT",
            "max_streak": {"pattern": "DRIFT", "length": 5},
            "high_confidence_count": 3,
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_SINGLE_SHOT_DETECTED_PATTERN",
        }

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        assert len(result["drivers"]) <= 3

    def test_fixed_shape_always_present(self):
        """All required GGFL fields are always present."""
        # Test with None
        result1 = tda_windowed_patterns_for_alignment_view(None)
        assert "signal_type" in result1
        assert "status" in result1
        assert "conflict" in result1
        assert "weight_hint" in result1
        assert "extraction_source" in result1
        assert "drivers" in result1
        assert "summary" in result1

        # Test with signal
        signal = {"dominant_pattern": "DRIFT", "max_streak": {"pattern": "NONE", "length": 0}}
        result2 = tda_windowed_patterns_for_alignment_view(signal)
        assert "signal_type" in result2
        assert "status" in result2
        assert "conflict" in result2
        assert "weight_hint" in result2
        assert "extraction_source" in result2
        assert "drivers" in result2
        assert "summary" in result2

    def test_signal_type_is_sig_tdaw(self):
        """Signal type is always SIG-TDAW."""
        result1 = tda_windowed_patterns_for_alignment_view(None)
        result2 = tda_windowed_patterns_for_alignment_view({"dominant_pattern": "DRIFT"})

        assert result1["signal_type"] == "SIG-TDAW"
        assert result2["signal_type"] == "SIG-TDAW"

    def test_conflict_always_false(self):
        """Conflict is always False."""
        result1 = tda_windowed_patterns_for_alignment_view(None)
        result2 = tda_windowed_patterns_for_alignment_view({"dominant_pattern": "DRIFT"})
        result3 = tda_windowed_patterns_for_alignment_view(
            {"dominant_pattern": "NONE"},
            {"disagreement_detected": True, "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN"}
        )

        assert result1["conflict"] is False
        assert result2["conflict"] is False
        assert result3["conflict"] is False

    def test_summary_is_one_sentence(self):
        """Summary is a single sentence (ends with period)."""
        signal = {
            "dominant_pattern": "STRUCTURAL_BREAK",
            "max_streak": {"pattern": "STRUCTURAL_BREAK", "length": 3},
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
        }

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        assert result["summary"].endswith(".")
        # Should be one sentence (single period at end)
        assert result["summary"].count(".") == 1

    def test_streak_driver_only_when_length_greater_than_one(self):
        """Streak driver only included when length > 1."""
        signal_no_streak = {
            "dominant_pattern": "DRIFT",
            "max_streak": {"pattern": "DRIFT", "length": 1},
        }
        signal_with_streak = {
            "dominant_pattern": "DRIFT",
            "max_streak": {"pattern": "DRIFT", "length": 3},
        }

        result1 = tda_windowed_patterns_for_alignment_view(signal_no_streak)
        result2 = tda_windowed_patterns_for_alignment_view(signal_with_streak)

        streak_drivers1 = [d for d in result1["drivers"] if "DRIVER_STREAK" in d]
        streak_drivers2 = [d for d in result2["drivers"] if "DRIVER_STREAK" in d]

        assert len(streak_drivers1) == 0
        assert len(streak_drivers2) == 1
        assert "DRIVER_STREAK:DRIFT(3)" in result2["drivers"]

    def test_deterministic_output(self):
        """GGFL adapter produces identical output for identical input."""
        signal = {
            "dominant_pattern": "STRUCTURAL_BREAK",
            "max_streak": {"pattern": "STRUCTURAL_BREAK", "length": 5},
            "high_confidence_count": 2,
            "extraction_source": "MANIFEST",
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
        }

        result1 = tda_windowed_patterns_for_alignment_view(signal, disagreement)
        result2 = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        assert result1 == result2

    def test_handles_missing_max_streak(self):
        """Handles signal without max_streak field."""
        signal = {
            "dominant_pattern": "DRIFT",
            "high_confidence_count": 1,
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["status"] == "warn"
        assert "DRIVER_DOMINANT_PATTERN:DRIFT" in result["drivers"]

    def test_handles_empty_disagreement(self):
        """Handles empty disagreement dict."""
        signal = {"dominant_pattern": "NONE"}
        disagreement = {}

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        assert result["status"] == "ok"  # No disagreement_detected key

    def test_case_insensitive_none_check(self):
        """NONE pattern check is case insensitive."""
        signal_lower = {"dominant_pattern": "none"}
        signal_upper = {"dominant_pattern": "NONE"}
        signal_mixed = {"dominant_pattern": "None"}

        result1 = tda_windowed_patterns_for_alignment_view(signal_lower)
        result2 = tda_windowed_patterns_for_alignment_view(signal_upper)
        result3 = tda_windowed_patterns_for_alignment_view(signal_mixed)

        assert result1["status"] == "ok"
        assert result2["status"] == "ok"
        assert result3["status"] == "ok"

    def test_extraction_source_manifest_normalization(self):
        """MANIFEST_GOVERNANCE normalizes to MANIFEST."""
        signal = {
            "dominant_pattern": "NONE",
            "extraction_source": "MANIFEST_GOVERNANCE",
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["extraction_source"] == "MANIFEST"

    def test_extraction_source_evidence_normalization(self):
        """EVIDENCE_GOVERNANCE normalizes to EVIDENCE_JSON."""
        signal = {
            "dominant_pattern": "NONE",
            "extraction_source": "EVIDENCE_GOVERNANCE",
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["extraction_source"] == "EVIDENCE_JSON"

    def test_extraction_source_unknown_becomes_missing(self):
        """Unknown extraction_source normalizes to MISSING."""
        signal = {
            "dominant_pattern": "NONE",
            "extraction_source": "SOME_UNKNOWN_SOURCE",
        }

        result = tda_windowed_patterns_for_alignment_view(signal)

        assert result["extraction_source"] == "MISSING"

    def test_drivers_use_driver_prefix_format(self):
        """All drivers use DRIVER_ prefix format."""
        signal = {
            "dominant_pattern": "DRIFT",
            "max_streak": {"pattern": "DRIFT", "length": 5},
            "extraction_source": "MANIFEST",
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
        }

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        for driver in result["drivers"]:
            assert driver.startswith("DRIVER_"), f"Driver '{driver}' missing DRIVER_ prefix"


# =============================================================================
# Red-Flag Storm Guard Tests (CAL-EXP-2 Prep)
# =============================================================================


class TestRedFlagStormGuard:
    """
    Tests verifying SHADOW MODE non-interference under red-flag storms.

    When TDA windowed patterns emits many red-flags (high confidence patterns,
    long streaks, disagreements), SHADOW MODE must:
    1. NOT interfere with processing
    2. Cap warnings to prevent log flooding
    3. Preserve all signal data for observability
    """

    def test_storm_with_many_high_confidence_patterns_stays_shadow(self):
        """Storm of high-confidence patterns does not trigger enforcement."""
        manifest = {
            "signals": {
                "tda_windowed_patterns": {
                    "schema_version": "1.0.0",
                    "mode": "SHADOW",
                    "status": {
                        "dominant_pattern": "DRIFT",
                        "max_streak": {"pattern": "DRIFT", "length": 10},
                        "high_confidence_count": 50,  # Storm: many red flags
                        "coverage": {"total_windows": 100, "windows_with_patterns": 80},
                    },
                    "top_events": [
                        {"window_index": i, "pattern": "DRIFT", "confidence": 0.95}
                        for i in range(20)  # Many high-confidence events
                    ],
                },
            },
            "governance": {
                "tda": {
                    "patterns": {"pattern": "NONE"},  # Disagreement
                },
            },
        }

        # Extract signal
        signal = extract_tda_windowed_patterns_signal_for_status(
            manifest=manifest, evidence_data=None,
        )

        # Verify SHADOW MODE preserved
        assert signal is not None
        assert signal.get("mode") == "SHADOW"
        assert signal.get("high_confidence_count") == 50

        # Verify no enforcement fields (should not exist)
        assert "enforcement_action" not in signal
        assert "blocked" not in signal

    def test_storm_warnings_capped_at_one(self):
        """Storm of red flags produces max 1 warning (cap enforced)."""
        manifest = {
            "signals": {
                "tda_windowed_patterns": {
                    "schema_version": "1.0.0",
                    "mode": "SHADOW",
                    "status": {
                        "dominant_pattern": "DRIFT",
                        "max_streak": {"pattern": "OSCILLATION", "length": 15},
                        "high_confidence_count": 100,
                        "coverage": {"total_windows": 200, "windows_with_patterns": 150},
                    },
                    "top_events": [
                        {"window_index": i, "pattern": "DRIFT", "confidence": 0.99}
                        for i in range(50)
                    ],
                },
            },
        }

        warnings = extract_tda_windowed_patterns_warnings(
            manifest=manifest, evidence_data=None,
        )

        # Verify warning cap
        assert len(warnings) <= 1, f"Expected max 1 warning, got {len(warnings)}"

        # Verify warning content includes key metrics
        if warnings:
            warning = warnings[0]
            assert "dominant=" in warning or "DRIFT" in warning

    def test_storm_disagreement_warning_also_capped(self):
        """Storm with disagreement still produces max 1 disagreement warning."""
        manifest = {
            "signals": {
                "tda_windowed_patterns": {
                    "schema_version": "1.0.0",
                    "mode": "SHADOW",
                    "status": {
                        "dominant_pattern": "DRIFT",
                        "max_streak": {"pattern": "DRIFT", "length": 20},
                        "high_confidence_count": 200,
                        "coverage": {"total_windows": 500, "windows_with_patterns": 400},
                    },
                },
            },
            "governance": {
                "tda": {
                    "patterns": {"pattern": "NONE"},
                },
            },
        }

        disagreement = extract_pattern_disagreement_for_status(
            manifest=manifest, evidence_data=None,
        )

        assert disagreement is not None
        assert disagreement.get("disagreement_detected") is True
        assert disagreement.get("mode") == "SHADOW"

        # Only one disagreement record (not one per window/event)
        assert isinstance(disagreement, dict)  # Single record, not list

    def test_storm_ggfl_adapter_preserves_all_drivers(self):
        """GGFL adapter under storm preserves all driver codes."""
        signal = {
            "dominant_pattern": "OSCILLATION",
            "max_streak": {"pattern": "OSCILLATION", "length": 25},
            "high_confidence_count": 300,
            "extraction_source": "MANIFEST",
        }
        disagreement = {
            "disagreement_detected": True,
            "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
        }

        result = tda_windowed_patterns_for_alignment_view(signal, disagreement)

        # Verify all drivers captured
        drivers = result.get("drivers", [])
        assert any("DOMINANT_PATTERN:OSCILLATION" in d for d in drivers)
        assert any("STREAK:OSCILLATION" in d for d in drivers)
        assert any("WINDOWED_DETECTED_PATTERN" in d for d in drivers)

        # Verify SHADOW mode semantics (no blocking)
        assert result.get("conflict") is False  # Advisory only
        assert result.get("weight_hint") == "LOW"

    def test_storm_signal_deterministic(self):
        """Storm conditions produce deterministic output."""
        manifest = {
            "signals": {
                "tda_windowed_patterns": {
                    "schema_version": "1.0.0",
                    "mode": "SHADOW",
                    "status": {
                        "dominant_pattern": "DRIFT",
                        "max_streak": {"pattern": "DRIFT", "length": 50},
                        "high_confidence_count": 500,
                        "coverage": {"total_windows": 1000, "windows_with_patterns": 900},
                    },
                },
            },
        }

        # Run extraction multiple times
        results = [
            extract_tda_windowed_patterns_signal_for_status(manifest=manifest, evidence_data=None)
            for _ in range(5)
        ]

        # All results must be identical
        for result in results[1:]:
            assert result == results[0], "Storm extraction must be deterministic"

    def test_storm_no_state_mutation(self):
        """Storm processing does not mutate input manifest."""
        import copy

        manifest = {
            "signals": {
                "tda_windowed_patterns": {
                    "schema_version": "1.0.0",
                    "mode": "SHADOW",
                    "status": {
                        "dominant_pattern": "DRIFT",
                        "max_streak": {"pattern": "DRIFT", "length": 100},
                        "high_confidence_count": 1000,
                        "coverage": {"total_windows": 2000, "windows_with_patterns": 1800},
                    },
                },
            },
            "governance": {
                "tda": {
                    "patterns": {"pattern": "NONE"},
                },
            },
        }

        manifest_copy = copy.deepcopy(manifest)

        # Run all extractions
        _ = extract_tda_windowed_patterns_signal_for_status(manifest=manifest, evidence_data=None)
        _ = extract_tda_windowed_patterns_warnings(manifest=manifest, evidence_data=None)
        _ = extract_pattern_disagreement_for_status(manifest=manifest, evidence_data=None)

        # Manifest must not be mutated
        assert manifest == manifest_copy, "Storm processing mutated input manifest"
