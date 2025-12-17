"""Tests for structural drill signal extraction in generate_first_light_status.

Tests verify:
1. Status signal extraction from manifest
2. GGFL adapter stub mapping with reason codes
3. Missing artifact handling (explicit optional - returns None)
4. Deterministic output
5. Reason code extraction (replaces drivers)
6. Manifest-first precedence with evidence.json fallback
7. Warning hygiene (single warning cap with drill_id)
8. Extraction source tracking (MANIFEST|EVIDENCE_JSON|MISSING)
9. Golden warning string lock (banned words, single-line, stable format)

SHADOW MODE: All tests verify observational behavior only.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

# Import reusable warning neutrality helpers
from tests.helpers.warning_neutrality import (
    pytest_assert_warning_neutral,
    BANNED_ALARM_WORDS,
)

from backend.health.structural_drill_adapter import (
    extract_structural_drill_signal,
    structural_drill_for_alignment_view,
    extract_drill_drivers,
    extract_drill_reason_codes,
    extract_structural_drill_from_sources,
    generate_structural_drill_warning,
    derive_worst_severity_from_pattern_counts,
    STRUCTURAL_DRILL_SCHEMA_VERSION,
    EXTRACTION_SOURCE_MANIFEST,
    EXTRACTION_SOURCE_EVIDENCE_JSON,
    EXTRACTION_SOURCE_MISSING,
    REASON_WORST_SEVERITY_CRITICAL,
    REASON_MAX_STREAK_GE_2,
    REASON_DRILL_SUCCESS_FALSE,
)


class TestExtractStructuralDrillSignal:
    """Test status signal extraction from drill reference."""

    def test_extracts_drill_success(self):
        """Extracts drill_success from reference."""
        drill_ref = {
            "drill_id": "drill_abc123",
            "scenario_id": "DRILL-SB-001",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"NONE": 10, "DRIFT": 3, "STRUCTURAL_BREAK": 2},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["drill_success"] is True

    def test_extracts_max_streak(self):
        """Extracts max_streak from reference."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 12,
            "pattern_counts": {"STRUCTURAL_BREAK": 1},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["max_streak"] == 12

    def test_derives_worst_severity_critical(self):
        """Derives CRITICAL severity when STRUCTURAL_BREAK present."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"NONE": 10, "DRIFT": 3, "STRUCTURAL_BREAK": 2},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["worst_severity"] == "CRITICAL"

    def test_derives_worst_severity_warn(self):
        """Derives WARN severity when only DRIFT present."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"NONE": 20, "DRIFT": 5},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["worst_severity"] == "WARN"

    def test_derives_worst_severity_info(self):
        """Derives INFO severity when only NONE patterns."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"NONE": 25},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["worst_severity"] == "INFO"

    def test_includes_schema_version(self):
        """Signal includes schema_version."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["schema_version"] == STRUCTURAL_DRILL_SCHEMA_VERSION

    def test_includes_mode_shadow(self):
        """Signal includes mode=SHADOW."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["mode"] == "SHADOW"

    def test_handles_empty_reference(self):
        """Returns empty dict for empty reference."""
        signal = extract_structural_drill_signal({})

        assert signal == {}

    def test_handles_none_reference(self):
        """Returns empty dict for None reference."""
        signal = extract_structural_drill_signal(None)

        assert signal == {}


class TestStructuralDrillForAlignmentView:
    """Test GGFL adapter stub mapping."""

    def test_maps_critical_to_unhealthy(self):
        """Maps CRITICAL severity to unhealthy alignment."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 8,
            "pattern_counts": {"STRUCTURAL_BREAK": 5},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert result["alignment"] == "unhealthy"
        assert "STRUCTURAL_BREAK" in result["advisory"]
        assert result["mode"] == "SHADOW"
        assert "reason_codes" in result

    def test_maps_warn_to_degraded(self):
        """Maps WARN severity to degraded alignment."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"DRIFT": 3},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert result["alignment"] == "degraded"
        assert "DRIFT" in result["advisory"]
        assert result["mode"] == "SHADOW"
        assert "reason_codes" in result

    def test_maps_info_to_healthy(self):
        """Maps INFO severity to healthy alignment."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"NONE": 20},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert result["alignment"] == "healthy"
        assert result["mode"] == "SHADOW"
        assert "reason_codes" in result

    def test_missing_artifact_returns_none(self):
        """Missing artifact returns None (explicit optional, no false health)."""
        result = structural_drill_for_alignment_view(None)

        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict returns None (explicit optional, no false health)."""
        result = structural_drill_for_alignment_view({})

        assert result is None

    def test_present_artifact_always_has_shadow_mode(self):
        """Present artifacts include mode=SHADOW."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert result is not None
        assert result["mode"] == "SHADOW"


class TestDeriveWorstSeverity:
    """Test worst severity derivation helper."""

    def test_structural_break_is_critical(self):
        """STRUCTURAL_BREAK pattern yields CRITICAL."""
        pattern_counts = {"NONE": 10, "DRIFT": 5, "STRUCTURAL_BREAK": 1}

        severity = derive_worst_severity_from_pattern_counts(pattern_counts)

        assert severity == "CRITICAL"

    def test_drift_without_break_is_warn(self):
        """DRIFT without STRUCTURAL_BREAK yields WARN."""
        pattern_counts = {"NONE": 10, "DRIFT": 5}

        severity = derive_worst_severity_from_pattern_counts(pattern_counts)

        assert severity == "WARN"

    def test_none_only_is_info(self):
        """Only NONE patterns yields INFO."""
        pattern_counts = {"NONE": 20}

        severity = derive_worst_severity_from_pattern_counts(pattern_counts)

        assert severity == "INFO"

    def test_empty_counts_is_info(self):
        """Empty pattern counts yields INFO."""
        severity = derive_worst_severity_from_pattern_counts({})

        assert severity == "INFO"


class TestDeterministicOutput:
    """Test that signal extraction is deterministic."""

    def test_same_input_same_output(self):
        """Same input always produces same output."""
        drill_ref = {
            "drill_id": "drill_test123",
            "scenario_id": "DRILL-SB-001",
            "drill_success": True,
            "max_streak": 7,
            "pattern_counts": {"NONE": 10, "DRIFT": 3, "STRUCTURAL_BREAK": 2},
        }

        # Extract multiple times
        signals = [extract_structural_drill_signal(drill_ref) for _ in range(5)]

        # All should be identical
        for signal in signals[1:]:
            assert signal == signals[0]

    def test_alignment_view_deterministic(self):
        """GGFL alignment view is deterministic."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        # Map multiple times
        results = [structural_drill_for_alignment_view(drill_ref) for _ in range(5)]

        # All should be identical
        for result in results[1:]:
            assert result == results[0]


class TestMissingArtifactHandling:
    """Test handling of missing structural drill artifact (explicit optional)."""

    def test_none_returns_none_for_alignment(self):
        """None reference returns None for alignment (explicit optional)."""
        alignment = structural_drill_for_alignment_view(None)

        assert alignment is None

    def test_none_returns_empty_dict_for_signal(self):
        """None reference returns empty dict for signal extraction."""
        signal = extract_structural_drill_signal(None)

        assert signal == {}

    def test_empty_dict_returns_none_for_alignment(self):
        """Empty dict returns None for alignment (explicit optional)."""
        alignment = structural_drill_for_alignment_view({})

        assert alignment is None

    def test_empty_dict_returns_empty_for_signal(self):
        """Empty dict returns empty dict for signal extraction."""
        signal = extract_structural_drill_signal({})

        assert signal == {}

    def test_partial_reference_has_alignment(self):
        """Partial reference (with at least one key) returns alignment."""
        partial_ref = {"drill_success": True}  # Missing max_streak, pattern_counts

        alignment = structural_drill_for_alignment_view(partial_ref)

        # Partial ref is truthy, so returns alignment (not None)
        assert alignment is not None
        assert alignment["alignment"] == "healthy"


class TestExtractDrillDrivers:
    """Test deterministic driver extraction (now returns reason codes)."""

    def test_drivers_include_critical_severity_code(self):
        """CRITICAL severity produces reason code."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        drivers = extract_drill_drivers(drill_ref)

        assert len(drivers) >= 1
        assert REASON_WORST_SEVERITY_CRITICAL in drivers

    def test_drivers_include_max_streak_when_ge_2(self):
        """max_streak included when >= 2."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 7,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        drivers = extract_drill_drivers(drill_ref)

        assert REASON_MAX_STREAK_GE_2 in drivers

    def test_drivers_exclude_max_streak_when_lt_2(self):
        """max_streak excluded when < 2."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 1,
            "pattern_counts": {"NONE": 10},
        }

        drivers = extract_drill_drivers(drill_ref)

        assert REASON_MAX_STREAK_GE_2 not in drivers

    def test_drivers_max_three(self):
        """Drivers list capped at 3."""
        drill_ref = {
            "drill_success": False,
            "max_streak": 10,
            "pattern_counts": {"STRUCTURAL_BREAK": 5},
        }

        drivers = extract_drill_drivers(drill_ref)

        assert len(drivers) <= 3

    def test_drivers_deterministic_ordering(self):
        """Drivers have deterministic ordering: severity, streak, success."""
        drill_ref = {
            "drill_success": False,
            "max_streak": 8,
            "pattern_counts": {"STRUCTURAL_BREAK": 3},
        }

        drivers = extract_drill_drivers(drill_ref)

        # Verify ordering: CRITICAL first, then STREAK, then SUCCESS_FALSE
        assert drivers[0] == REASON_WORST_SEVERITY_CRITICAL
        assert drivers[1] == REASON_MAX_STREAK_GE_2
        assert drivers[2] == REASON_DRILL_SUCCESS_FALSE

    def test_drivers_empty_for_missing_artifact(self):
        """Returns empty list for missing artifact."""
        drivers = extract_drill_drivers(None)

        assert drivers == []

    def test_alignment_view_includes_reason_codes(self):
        """Alignment view result includes reason_codes list."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert "reason_codes" in result
        assert isinstance(result["reason_codes"], list)
        assert len(result["reason_codes"]) <= 3


class TestReasonCodeExtraction:
    """Test reason code extraction (replaces drivers)."""

    def test_critical_severity_reason_code(self):
        """CRITICAL severity produces DRIVER_WORST_SEVERITY_CRITICAL."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"STRUCTURAL_BREAK": 1},
        }

        codes = extract_drill_reason_codes(drill_ref)

        assert REASON_WORST_SEVERITY_CRITICAL in codes

    def test_max_streak_ge_2_reason_code(self):
        """max_streak >= 2 produces DRIVER_MAX_STREAK_GE_2."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 3,
            "pattern_counts": {"NONE": 10},
        }

        codes = extract_drill_reason_codes(drill_ref)

        assert REASON_MAX_STREAK_GE_2 in codes

    def test_drill_success_false_reason_code(self):
        """drill_success=False produces DRIVER_DRILL_SUCCESS_FALSE."""
        drill_ref = {
            "drill_success": False,
            "max_streak": 0,
            "pattern_counts": {"NONE": 10},
        }

        codes = extract_drill_reason_codes(drill_ref)

        assert REASON_DRILL_SUCCESS_FALSE in codes

    def test_no_reason_codes_for_clean_drill(self):
        """Clean drill (success, low streak, no CRITICAL) has no reason codes."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 1,
            "pattern_counts": {"NONE": 20},
        }

        codes = extract_drill_reason_codes(drill_ref)

        assert len(codes) == 0

    def test_reason_codes_max_three(self):
        """Reason codes capped at 3."""
        drill_ref = {
            "drill_success": False,  # DRIVER_DRILL_SUCCESS_FALSE
            "max_streak": 5,  # DRIVER_MAX_STREAK_GE_2
            "pattern_counts": {"STRUCTURAL_BREAK": 3},  # DRIVER_WORST_SEVERITY_CRITICAL
        }

        codes = extract_drill_reason_codes(drill_ref)

        assert len(codes) <= 3
        assert REASON_WORST_SEVERITY_CRITICAL in codes
        assert REASON_MAX_STREAK_GE_2 in codes
        assert REASON_DRILL_SUCCESS_FALSE in codes

    def test_reason_codes_empty_for_missing_drill(self):
        """Returns empty list for missing drill."""
        codes = extract_drill_reason_codes(None)

        assert codes == []

    def test_backward_compat_extract_drill_drivers(self):
        """extract_drill_drivers returns reason codes (backward compat)."""
        drill_ref = {
            "drill_success": False,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 1},
        }

        # Old function name should still work
        drivers = extract_drill_drivers(drill_ref)
        codes = extract_drill_reason_codes(drill_ref)

        assert drivers == codes


class TestManifestFirstPrecedence:
    """Test manifest-first extraction with evidence.json fallback."""

    def test_manifest_takes_precedence(self):
        """Manifest drill reference takes precedence over evidence.json."""
        manifest_drill = {
            "drill_id": "drill_manifest_001",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }
        evidence_drill = {
            "drill_id": "drill_evidence_001",
            "drill_success": False,
            "max_streak": 10,
            "pattern_counts": {"DRIFT": 5},
        }

        pack_manifest = {"governance": {"structure": {"drill": manifest_drill}}}
        evidence_data = {"governance": {"structure": {"drill": evidence_drill}}}

        result, source = extract_structural_drill_from_sources(pack_manifest, evidence_data)

        assert result["drill_id"] == "drill_manifest_001"
        assert result["max_streak"] == 5
        assert source == EXTRACTION_SOURCE_MANIFEST

    def test_falls_back_to_evidence_when_manifest_missing(self):
        """Falls back to evidence.json when manifest has no drill."""
        evidence_drill = {
            "drill_id": "drill_evidence_002",
            "drill_success": True,
            "max_streak": 3,
            "pattern_counts": {"DRIFT": 2},
        }

        pack_manifest = {"governance": {"structure": {}}}  # No drill
        evidence_data = {"governance": {"structure": {"drill": evidence_drill}}}

        result, source = extract_structural_drill_from_sources(pack_manifest, evidence_data)

        assert result["drill_id"] == "drill_evidence_002"
        assert result["max_streak"] == 3
        assert source == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_returns_none_when_both_missing(self):
        """Returns None when neither manifest nor evidence has drill."""
        pack_manifest = {"governance": {"structure": {}}}
        evidence_data = {"governance": {"structure": {}}}

        result, source = extract_structural_drill_from_sources(pack_manifest, evidence_data)

        assert result is None
        assert source == EXTRACTION_SOURCE_MISSING

    def test_returns_none_when_both_none(self):
        """Returns None when both sources are None."""
        result, source = extract_structural_drill_from_sources(None, None)

        assert result is None
        assert source == EXTRACTION_SOURCE_MISSING

    def test_manifest_only_works(self):
        """Works when only manifest is provided."""
        manifest_drill = {
            "drill_id": "drill_only_manifest",
            "drill_success": True,
            "max_streak": 2,
            "pattern_counts": {"NONE": 10},
        }
        pack_manifest = {"governance": {"structure": {"drill": manifest_drill}}}

        result, source = extract_structural_drill_from_sources(pack_manifest, None)

        assert result["drill_id"] == "drill_only_manifest"
        assert source == EXTRACTION_SOURCE_MANIFEST

    def test_evidence_only_works(self):
        """Works when only evidence.json is provided."""
        evidence_drill = {
            "drill_id": "drill_only_evidence",
            "drill_success": False,
            "max_streak": 7,
            "pattern_counts": {"STRUCTURAL_BREAK": 3},
        }
        evidence_data = {"governance": {"structure": {"drill": evidence_drill}}}

        result, source = extract_structural_drill_from_sources(None, evidence_data)

        assert result["drill_id"] == "drill_only_evidence"
        assert source == EXTRACTION_SOURCE_EVIDENCE_JSON


class TestWarningHygiene:
    """Test warning hygiene: single warning cap with drill_id."""

    def test_warning_generated_for_critical(self):
        """Warning generated when worst_severity is CRITICAL."""
        drill_ref = {
            "drill_id": "drill_critical_001",
            "drill_success": True,
            "max_streak": 1,
            "pattern_counts": {"STRUCTURAL_BREAK": 3},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is not None
        assert "STRUCTURAL_BREAK" in warning

    def test_warning_generated_for_high_streak(self):
        """Warning generated when max_streak >= 2."""
        drill_ref = {
            "drill_id": "drill_streak_001",
            "drill_success": True,
            "max_streak": 2,
            "pattern_counts": {"DRIFT": 5},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is not None
        assert "streak=2" in warning

    def test_no_warning_for_low_severity_low_streak(self):
        """No warning when severity is INFO/WARN and streak < 2."""
        drill_ref = {
            "drill_id": "drill_clean_001",
            "drill_success": True,
            "max_streak": 1,
            "pattern_counts": {"NONE": 20},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is None

    def test_no_warning_for_warn_severity_low_streak(self):
        """No warning when severity is WARN and streak < 2."""
        drill_ref = {
            "drill_id": "drill_warn_low",
            "drill_success": True,
            "max_streak": 1,
            "pattern_counts": {"DRIFT": 3},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is None

    def test_warning_includes_drill_id(self):
        """Warning includes drill_id when present."""
        drill_ref = {
            "drill_id": "drill_with_id_123",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert "drill_with_id_123" in warning

    def test_warning_works_without_drill_id(self):
        """Warning works when drill_id is missing."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 3,
            "pattern_counts": {"STRUCTURAL_BREAK": 1},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is not None
        assert "STRUCTURAL_BREAK" in warning

    def test_warning_includes_severity(self):
        """Warning includes severity context."""
        drill_ref = {
            "drill_id": "drill_severity_test",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert "severity=CRITICAL" in warning
        assert "informational" in warning

    def test_warning_is_single_line(self):
        """Warning is a single line (no newlines)."""
        drill_ref = {
            "drill_id": "drill_single_line",
            "drill_success": True,
            "max_streak": 10,
            "pattern_counts": {"STRUCTURAL_BREAK": 5},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert "\n" not in warning

    def test_no_warning_for_none_reference(self):
        """No warning for None reference."""
        warning = generate_structural_drill_warning(None)

        assert warning is None

    def test_no_warning_for_empty_reference(self):
        """No warning for empty reference."""
        warning = generate_structural_drill_warning({})

        assert warning is None


class TestManifestFirstDeterminism:
    """Test determinism of manifest-first extraction."""

    def test_extraction_is_deterministic(self):
        """Same inputs always produce same output."""
        manifest_drill = {
            "drill_id": "drill_determinism_001",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }
        evidence_drill = {
            "drill_id": "drill_evidence_fallback",
            "drill_success": False,
            "max_streak": 3,
            "pattern_counts": {"DRIFT": 5},
        }

        pack_manifest = {"governance": {"structure": {"drill": manifest_drill}}}
        evidence_data = {"governance": {"structure": {"drill": evidence_drill}}}

        # Extract multiple times
        results = [
            extract_structural_drill_from_sources(pack_manifest, evidence_data)
            for _ in range(5)
        ]

        # All should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_warning_generation_is_deterministic(self):
        """Warning generation is deterministic."""
        drill_ref = {
            "drill_id": "drill_warn_det",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        # Generate multiple times
        warnings = [generate_structural_drill_warning(drill_ref) for _ in range(5)]

        # All should be identical
        for warning in warnings[1:]:
            assert warning == warnings[0]


class TestExtractionSourceTracking:
    """Test extraction_source is included in signal."""

    def test_signal_includes_extraction_source_manifest(self):
        """Signal includes extraction_source when from MANIFEST."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 3,
            "pattern_counts": {"STRUCTURAL_BREAK": 1},
        }

        signal = extract_structural_drill_signal(
            drill_ref,
            extraction_source=EXTRACTION_SOURCE_MANIFEST,
        )

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MANIFEST

    def test_signal_includes_extraction_source_evidence_json(self):
        """Signal includes extraction_source when from EVIDENCE_JSON."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 2,
            "pattern_counts": {"DRIFT": 3},
        }

        signal = extract_structural_drill_signal(
            drill_ref,
            extraction_source=EXTRACTION_SOURCE_EVIDENCE_JSON,
        )

        assert signal["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_signal_default_extraction_source_missing(self):
        """Signal defaults to MISSING extraction_source."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"NONE": 10},
        }

        signal = extract_structural_drill_signal(drill_ref)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MISSING


class TestGGFLExplicitOptional:
    """Test GGFL returns None on missing drill (explicit optional)."""

    def test_ggfl_returns_none_for_none_reference(self):
        """GGFL returns None when drill_reference is None."""
        result = structural_drill_for_alignment_view(None)

        assert result is None

    def test_ggfl_returns_none_for_empty_reference(self):
        """GGFL returns None when drill_reference is empty dict."""
        result = structural_drill_for_alignment_view({})

        assert result is None

    def test_ggfl_returns_dict_for_valid_reference(self):
        """GGFL returns dict when drill_reference is valid."""
        drill_ref = {
            "drill_success": True,
            "max_streak": 0,
            "pattern_counts": {"NONE": 10},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        assert result is not None
        assert isinstance(result, dict)
        assert "alignment" in result
        assert "reason_codes" in result
        assert result["mode"] == "SHADOW"

    def test_ggfl_reason_codes_are_reason_codes_not_strings(self):
        """GGFL reason_codes are actual reason code constants."""
        drill_ref = {
            "drill_success": False,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 2},
        }

        result = structural_drill_for_alignment_view(drill_ref)

        # All reason codes should be from the constant set
        valid_codes = {
            REASON_WORST_SEVERITY_CRITICAL,
            REASON_MAX_STREAK_GE_2,
            REASON_DRILL_SUCCESS_FALSE,
        }
        for code in result["reason_codes"]:
            assert code in valid_codes


class TestGoldenWarningStringLock:
    """Regression test: golden warning string lock.

    Ensures structural drill warning strings:
    1. Are single-line (no newlines)
    2. Contain no banned alarm words
    3. Remain stable for CRITICAL and max_streak>=2 cases

    Uses reusable helpers from tests.helpers.warning_neutrality.

    SHADOW MODE: Observational verification only.
    """

    # ==========================================================================
    # GOLDEN FORMAT CONSTANTS
    # ==========================================================================
    # Update these constants to evolve warning format without breaking tests.
    # Tests verify format contains these substrings, allowing minor evolution.
    # ==========================================================================

    GOLDEN_PREFIX = "Structural drill:"
    GOLDEN_CRITICAL_PHRASE = "STRUCTURAL_BREAK pattern recorded"
    GOLDEN_SEVERITY_SUFFIX_TEMPLATE = "(severity={severity}, informational)"

    def test_critical_warning_neutral(self):
        """CRITICAL warning is neutral (single-line, no banned words)."""
        drill_ref = {
            "drill_id": "drill_golden_critical",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 3},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is not None
        pytest_assert_warning_neutral(warning, context="CRITICAL warning")

    def test_max_streak_warning_neutral(self):
        """max_streak>=2 warning is neutral (single-line, no banned words)."""
        drill_ref = {
            "drill_id": "drill_golden_streak",
            "drill_success": True,
            "max_streak": 4,
            "pattern_counts": {"DRIFT": 5},
        }

        warning = generate_structural_drill_warning(drill_ref)

        assert warning is not None
        pytest_assert_warning_neutral(warning, context="max_streak>=2 warning")

    def test_critical_warning_stable_format(self):
        """CRITICAL warning has stable expected format."""
        drill_ref = {
            "drill_id": "drill_stable_001",
            "drill_success": True,
            "max_streak": 5,
            "pattern_counts": {"STRUCTURAL_BREAK": 3},
        }

        warning = generate_structural_drill_warning(drill_ref)

        # Verify against golden format constants
        assert warning is not None
        assert warning.startswith(self.GOLDEN_PREFIX)
        assert self.GOLDEN_CRITICAL_PHRASE in warning
        assert "[drill_stable_001]" in warning
        assert self.GOLDEN_SEVERITY_SUFFIX_TEMPLATE.format(severity="CRITICAL") in warning

    def test_max_streak_warning_stable_format(self):
        """max_streak>=2 warning has stable expected format."""
        drill_ref = {
            "drill_id": "drill_stable_002",
            "drill_success": True,
            "max_streak": 4,
            "pattern_counts": {"DRIFT": 5},
        }

        warning = generate_structural_drill_warning(drill_ref)

        # Verify against golden format constants
        assert warning is not None
        assert warning.startswith(self.GOLDEN_PREFIX)
        assert "streak=4" in warning
        assert "[drill_stable_002]" in warning
        assert self.GOLDEN_SEVERITY_SUFFIX_TEMPLATE.format(severity="WARN") in warning

    def test_combined_critical_and_streak_stable_format(self):
        """Combined CRITICAL + high streak uses CRITICAL format."""
        drill_ref = {
            "drill_id": "drill_combined_001",
            "drill_success": True,
            "max_streak": 10,
            "pattern_counts": {"STRUCTURAL_BREAK": 5},
        }

        warning = generate_structural_drill_warning(drill_ref)

        # When both conditions met, CRITICAL takes precedence in message
        assert warning is not None
        assert self.GOLDEN_CRITICAL_PHRASE in warning
        assert self.GOLDEN_SEVERITY_SUFFIX_TEMPLATE.format(severity="CRITICAL") in warning
