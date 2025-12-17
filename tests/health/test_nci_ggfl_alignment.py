"""
Tests for NCI P5 GGFL alignment view adapter and curated docs audit.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating influence is tested
- Determinism is verified

Tests cover:
1. GGFL alignment view adapter fixed shape
2. Deterministic output
3. Neutral summary generation
4. Curated docs audit report
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from backend.health.nci_governance_adapter import (
    nci_p5_for_alignment_view,
    build_neutral_nci_summary,
    build_nci_status_warning,
    build_curated_docs_audit_report,
    NCI_CURATED_DOC_PATTERNS,
    MAX_DOC_SIZE_BYTES,
)
from tests.doc_samples import (
    NCI_P5_BREACH_SAMPLE,
    NCI_P5_BREACH_INPUT,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def healthy_nci_signal() -> Dict[str, Any]:
    """Sample healthy NCI P5 signal (compact format)."""
    return {
        "schema_version": "1.0.0",
        "mode": "DOC_ONLY",
        "global_nci": 0.92,
        "confidence": 0.85,
        "slo_status": "OK",
        "recommendation": "NONE",
        "tcl_aligned": True,
        "sic_aligned": True,
        "tcl_violation_count": 0,
        "sic_violation_count": 0,
        "warning_count": 0,
        "shadow_mode": True,
    }


@pytest.fixture
def warn_nci_signal() -> Dict[str, Any]:
    """Sample NCI P5 signal with warnings (compact format)."""
    return {
        "schema_version": "1.0.0",
        "mode": "TELEMETRY_CHECKED",
        "global_nci": 0.72,
        "confidence": 0.65,
        "slo_status": "WARN",
        "recommendation": "WARNING",
        "tcl_aligned": False,
        "sic_aligned": True,
        "tcl_violation_count": 3,
        "sic_violation_count": 0,
        "warning_count": 2,
        "shadow_mode": True,
    }


@pytest.fixture
def breach_nci_signal() -> Dict[str, Any]:
    """Sample NCI P5 signal with BREACH status (compact format)."""
    return {
        "schema_version": "1.0.0",
        "mode": "FULLY_BOUND",
        "global_nci": 0.45,
        "confidence": 0.40,
        "slo_status": "BREACH",
        "recommendation": "REVIEW",
        "tcl_aligned": False,
        "sic_aligned": False,
        "tcl_violation_count": 5,
        "sic_violation_count": 3,
        "warning_count": 4,
        "shadow_mode": True,
    }


@pytest.fixture
def full_nci_result() -> Dict[str, Any]:
    """Sample full NCI P5 result (nested format from evaluate_nci_p5)."""
    return {
        "schema_version": "1.0.0",
        "mode": "DOC_ONLY",
        "global_nci": 0.78,
        "confidence": 0.70,
        "slo_evaluation": {
            "status": "WARN",
            "threshold_warn": 0.80,
            "threshold_breach": 0.60,
        },
        "governance_signal": {
            "signal_type": "SIG-NAR",
            "recommendation": "WARNING",
            "shadow_mode": True,
        },
        "tcl_result": {
            "aligned": False,
            "checks_run": ["TCL-001", "TCL-002"],
            "violations": [
                {"doc": "docs/api.md", "field": "H", "violation_type": "TCL-002"},
                {"doc": "docs/spec.md", "field": "rho", "violation_type": "TCL-002"},
            ],
        },
        "sic_result": {
            "aligned": True,
            "checks_run": ["SIC-001", "SIC-002"],
            "violations": [],
        },
        "warnings": [
            {"warning_type": "LOW_DOC_COUNT", "message": "Only 5 docs scanned"},
        ],
        "shadow_mode": True,
    }


# =============================================================================
# Test: GGFL Alignment View Fixed Shape
# =============================================================================


class TestNciP5ForAlignmentView:
    """Tests for nci_p5_for_alignment_view() function."""

    def test_returns_fixed_shape_keys(self, healthy_nci_signal: Dict[str, Any]):
        """Alignment view always has required GGFL unified keys."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)

        # GGFL unified fields (required)
        assert "status" in result
        assert "alignment" in result
        assert "conflict" in result
        assert "drivers" in result  # Reason codes only

        # NCI-specific extensions
        assert "mode" in result
        assert "global_nci" in result
        assert "confidence" in result
        assert "tcl_aligned" in result
        assert "sic_aligned" in result

    def test_status_is_lowercase(self, healthy_nci_signal: Dict[str, Any]):
        """Status is normalized to lowercase (ok/warn/block)."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)
        assert result["status"] == "ok"

    def test_status_maps_breach_to_block(self, breach_nci_signal: Dict[str, Any]):
        """BREACH status maps to 'block' in GGFL."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        assert result["status"] == "block"

    def test_status_maps_warn_to_warn(self, warn_nci_signal: Dict[str, Any]):
        """WARN status maps to 'warn' in GGFL."""
        result = nci_p5_for_alignment_view(warn_nci_signal)
        assert result["status"] == "warn"

    def test_alignment_aligned_when_both_laws_pass(self, healthy_nci_signal: Dict[str, Any]):
        """Alignment is 'aligned' when TCL and SIC both pass."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)
        assert result["alignment"] == "aligned"

    def test_alignment_divergent_when_both_laws_fail(self, breach_nci_signal: Dict[str, Any]):
        """Alignment is 'divergent' when both TCL and SIC fail."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        assert result["alignment"] == "divergent"

    def test_alignment_tension_when_one_law_fails(self, warn_nci_signal: Dict[str, Any]):
        """Alignment is 'tension' when only one of TCL/SIC fails."""
        result = nci_p5_for_alignment_view(warn_nci_signal)
        assert result["alignment"] == "tension"

    def test_conflict_true_when_both_fail(self, breach_nci_signal: Dict[str, Any]):
        """Conflict flag is True when both TCL and SIC fail."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        assert result["conflict"] is True

    def test_conflict_false_when_aligned(self, healthy_nci_signal: Dict[str, Any]):
        """Conflict flag is False when aligned."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)
        assert result["conflict"] is False

    def test_drivers_is_list(self, healthy_nci_signal: Dict[str, Any]):
        """Drivers is always a list."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)
        assert isinstance(result["drivers"], list)

    def test_drivers_max_three(self, breach_nci_signal: Dict[str, Any]):
        """Drivers is limited to 3 entries."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        assert len(result["drivers"]) <= 3

    def test_handles_nested_result_format(self, full_nci_result: Dict[str, Any]):
        """Handles nested format from evaluate_nci_p5()."""
        result = nci_p5_for_alignment_view(full_nci_result)

        assert result["status"] == "warn"
        assert result["tcl_aligned"] is False
        assert result["sic_aligned"] is True
        assert result["alignment"] == "tension"


# =============================================================================
# Test: Deterministic Output
# =============================================================================


class TestAlignmentViewDeterminism:
    """Tests for deterministic output behavior."""

    def test_same_input_produces_same_output(self, healthy_nci_signal: Dict[str, Any]):
        """Same input always produces identical output."""
        result1 = nci_p5_for_alignment_view(healthy_nci_signal)
        result2 = nci_p5_for_alignment_view(healthy_nci_signal)

        assert result1 == result2

    def test_output_is_json_serializable(self, breach_nci_signal: Dict[str, Any]):
        """Output can be serialized to JSON."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        # Should not raise
        json_str = json.dumps(result, sort_keys=True)
        assert isinstance(json_str, str)

    def test_multiple_calls_identical(self, warn_nci_signal: Dict[str, Any]):
        """Multiple calls produce byte-identical JSON."""
        results = [nci_p5_for_alignment_view(warn_nci_signal) for _ in range(5)]
        json_strs = [json.dumps(r, sort_keys=True) for r in results]
        assert all(s == json_strs[0] for s in json_strs)


# =============================================================================
# Test: Neutral Summary
# =============================================================================


class TestBuildNeutralNciSummary:
    """Tests for build_neutral_nci_summary() function."""

    def test_returns_string(self, healthy_nci_signal: Dict[str, Any]):
        """Summary is always a string."""
        summary = build_neutral_nci_summary(healthy_nci_signal)
        assert isinstance(summary, str)

    def test_includes_mode(self, healthy_nci_signal: Dict[str, Any]):
        """Summary includes the NCI mode."""
        summary = build_neutral_nci_summary(healthy_nci_signal)
        assert "DOC_ONLY" in summary

    def test_includes_percentage(self, healthy_nci_signal: Dict[str, Any]):
        """Summary includes NCI percentage."""
        summary = build_neutral_nci_summary(healthy_nci_signal)
        assert "92%" in summary

    def test_no_violations_text(self, healthy_nci_signal: Dict[str, Any]):
        """Summary says 'no violations' when clean."""
        summary = build_neutral_nci_summary(healthy_nci_signal)
        assert "no violations" in summary

    def test_violations_count_shown(self, breach_nci_signal: Dict[str, Any]):
        """Summary shows violation count when present."""
        summary = build_neutral_nci_summary(breach_nci_signal)
        assert "8 violation" in summary  # 5 TCL + 3 SIC = 8

    def test_handles_nested_format(self, full_nci_result: Dict[str, Any]):
        """Summary handles nested format from evaluate_nci_p5()."""
        summary = build_neutral_nci_summary(full_nci_result)
        assert "DOC_ONLY" in summary
        assert "78%" in summary
        assert "2 violation" in summary  # 2 TCL violations

    def test_deterministic_output(self, warn_nci_signal: Dict[str, Any]):
        """Summary is deterministic."""
        s1 = build_neutral_nci_summary(warn_nci_signal)
        s2 = build_neutral_nci_summary(warn_nci_signal)
        assert s1 == s2


# =============================================================================
# Test: Curated Docs Audit Report
# =============================================================================


class TestBuildCuratedDocsAuditReport:
    """Tests for build_curated_docs_audit_report() function."""

    def test_returns_required_keys(self, tmp_path: Path):
        """Audit report has required structure."""
        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert "schema_version" in report
        assert "patterns_checked" in report
        assert "matched_files" in report
        assert "skipped_files" in report
        assert "summary" in report

    def test_summary_has_required_fields(self, tmp_path: Path):
        """Summary has required fields."""
        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])
        summary = report["summary"]

        assert "total_patterns" in summary
        assert "total_matched" in summary
        assert "total_skipped" in summary
        assert "matched_truncated" in summary
        assert "skipped_truncated" in summary

    def test_detects_matching_files(self, tmp_path: Path):
        """Audit detects files matching patterns."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1", encoding="utf-8")
        (tmp_path / "doc2.md").write_text("# Doc 2", encoding="utf-8")
        (tmp_path / "other.txt").write_text("Other", encoding="utf-8")

        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert report["summary"]["total_matched"] == 2
        paths = [f["path"] for f in report["matched_files"]]
        assert "doc1.md" in paths
        assert "doc2.md" in paths

    def test_bounded_to_max_entries(self, tmp_path: Path):
        """Matched files list is bounded to max_entries."""
        # Create 30 test files
        for i in range(30):
            (tmp_path / f"doc{i:02d}.md").write_text(f"# Doc {i}", encoding="utf-8")

        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"], max_entries=20)

        assert len(report["matched_files"]) == 20
        assert report["summary"]["total_matched"] == 30
        assert report["summary"]["matched_truncated"] is True

    def test_skips_too_large_files(self, tmp_path: Path):
        """Files exceeding size limit are skipped."""
        # Create a small file
        (tmp_path / "small.md").write_text("# Small", encoding="utf-8")

        # Create a large file (simulate by checking the constant)
        large_content = "x" * (MAX_DOC_SIZE_BYTES + 1)
        (tmp_path / "large.md").write_text(large_content, encoding="utf-8")

        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert report["summary"]["total_matched"] == 1
        assert report["summary"]["total_skipped"] == 1
        assert report["skipped_files"][0]["reason"] == "too_large"

    def test_deterministic_output(self, tmp_path: Path):
        """Audit report is deterministic."""
        (tmp_path / "a.md").write_text("# A", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B", encoding="utf-8")

        report1 = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])
        report2 = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        # Compare JSON serialization for determinism
        json1 = json.dumps(report1, sort_keys=True)
        json2 = json.dumps(report2, sort_keys=True)
        assert json1 == json2

    def test_handles_empty_directory(self, tmp_path: Path):
        """Handles empty directory gracefully."""
        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert report["summary"]["total_matched"] == 0
        assert report["summary"]["total_skipped"] == 0
        assert report["matched_files"] == []
        assert report["skipped_files"] == []

    def test_uses_default_patterns_if_none(self, tmp_path: Path):
        """Uses NCI_CURATED_DOC_PATTERNS if patterns is None."""
        report = build_curated_docs_audit_report(tmp_path, patterns=None)

        # Should use default patterns
        assert report["patterns_checked"] == NCI_CURATED_DOC_PATTERNS

    def test_matched_entry_has_required_fields(self, tmp_path: Path):
        """Matched file entries have required fields."""
        (tmp_path / "doc.md").write_text("# Doc", encoding="utf-8")

        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert len(report["matched_files"]) == 1
        entry = report["matched_files"][0]
        assert "path" in entry
        assert "size_bytes" in entry
        assert "pattern" in entry

    def test_skipped_entry_has_reason(self, tmp_path: Path):
        """Skipped file entries have reason field."""
        # Create oversized file
        (tmp_path / "big.md").write_text("x" * (MAX_DOC_SIZE_BYTES + 1), encoding="utf-8")

        report = build_curated_docs_audit_report(tmp_path, patterns=["*.md"])

        assert len(report["skipped_files"]) == 1
        entry = report["skipped_files"][0]
        assert "path" in entry
        assert "reason" in entry
        assert entry["reason"] == "too_large"


# =============================================================================
# Test: Reason Code Drivers (No Natural Language)
# =============================================================================


# Valid reason codes for GGFL drivers
VALID_DRIVER_CODES = [
    "DRIVER_SLO_BREACH",
    "DRIVER_RECOMMENDATION_NON_NONE",
    "DRIVER_CONFIDENCE_LOW",
]

# Forbidden alarm terms that should NOT appear in drivers or summary
FORBIDDEN_ALARM_TERMS = [
    "violation",
    "warning",
    "error",
    "failure",
    "critical",
    "danger",
    "breach",
    "alert",
    "emergency",
]


class TestReasonCodeDrivers:
    """Tests for reason code-only GGFL drivers."""

    def test_drivers_are_codes_only(self, breach_nci_signal: Dict[str, Any]):
        """All drivers must be valid reason codes only."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        for driver in result["drivers"]:
            assert driver in VALID_DRIVER_CODES, f"Invalid driver code: {driver}"

    def test_healthy_signal_has_no_drivers(self, healthy_nci_signal: Dict[str, Any]):
        """Healthy signal should have empty drivers list."""
        result = nci_p5_for_alignment_view(healthy_nci_signal)
        assert result["drivers"] == []

    def test_breach_signal_has_slo_breach_driver(self, breach_nci_signal: Dict[str, Any]):
        """BREACH status produces DRIVER_SLO_BREACH."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        assert "DRIVER_SLO_BREACH" in result["drivers"]

    def test_non_none_recommendation_produces_driver(self):
        """Non-NONE recommendation produces DRIVER_RECOMMENDATION_NON_NONE."""
        signal = {
            "slo_status": "OK",
            "recommendation": "WARNING",
            "tcl_aligned": True,
            "sic_aligned": True,
        }
        result = nci_p5_for_alignment_view(signal)
        assert "DRIVER_RECOMMENDATION_NON_NONE" in result["drivers"]

    def test_low_confidence_produces_driver(self):
        """Low confidence produces DRIVER_CONFIDENCE_LOW."""
        signal = {
            "slo_status": "OK",
            "recommendation": "NONE",
            "confidence": 0.3,
            "tcl_aligned": True,
            "sic_aligned": True,
        }
        result = nci_p5_for_alignment_view(signal)
        assert "DRIVER_CONFIDENCE_LOW" in result["drivers"]

    def test_drivers_are_uppercase_codes(self, breach_nci_signal: Dict[str, Any]):
        """Drivers must be uppercase code format (DRIVER_*)."""
        result = nci_p5_for_alignment_view(breach_nci_signal)
        for driver in result["drivers"]:
            assert driver.startswith("DRIVER_"), f"Driver must start with DRIVER_: {driver}"
            assert driver == driver.upper(), f"Driver must be uppercase: {driver}"

    def test_drivers_max_three_codes(self):
        """Drivers are capped at 3 codes."""
        signal = {
            "slo_status": "BREACH",
            "recommendation": "REVIEW",
            "confidence": 0.3,
            "tcl_aligned": False,
            "sic_aligned": False,
        }
        result = nci_p5_for_alignment_view(signal)
        assert len(result["drivers"]) <= 3


class TestNeutralSummaryLanguage:
    """Tests for neutral language in build_neutral_nci_summary."""

    def test_summary_no_forbidden_terms_healthy(self, healthy_nci_signal: Dict[str, Any]):
        """Healthy summary has no forbidden alarm terms."""
        summary = build_neutral_nci_summary(healthy_nci_signal)
        summary_lower = summary.lower()
        for term in FORBIDDEN_ALARM_TERMS:
            # 'violation' is allowed in context of count (e.g. "no violations")
            if term == "violation" and "no violation" in summary_lower:
                continue
            # Don't flag if it's part of a count context
            if term == "violation" and "violation(s)" in summary_lower:
                continue
            # Check forbidden terms are not standalone alarm words
            if term != "violation":
                assert term not in summary_lower, f"Found forbidden term '{term}' in summary: {summary}"

    def test_summary_no_forbidden_terms_breach(self, breach_nci_signal: Dict[str, Any]):
        """Breach summary has no forbidden standalone alarm terms."""
        summary = build_neutral_nci_summary(breach_nci_signal)
        summary_lower = summary.lower()
        # 'violation' is allowed in count context
        # Other terms should not appear
        for term in ["warning", "error", "failure", "critical", "danger", "alert", "emergency"]:
            assert term not in summary_lower, f"Found forbidden term '{term}' in summary: {summary}"


# =============================================================================
# Test: Status Warning Hygiene
# =============================================================================


class TestBuildNciStatusWarning:
    """Tests for build_nci_status_warning() function."""

    def test_returns_none_for_ok_status(self, healthy_nci_signal: Dict[str, Any]):
        """Returns None when slo_status is OK and recommendation is NONE."""
        warning = build_nci_status_warning(healthy_nci_signal)
        assert warning is None

    def test_returns_warning_for_breach(self, breach_nci_signal: Dict[str, Any]):
        """Returns warning string when slo_status is BREACH."""
        warning = build_nci_status_warning(breach_nci_signal)
        assert warning is not None
        assert "BREACH" in warning

    def test_returns_warning_for_non_none_recommendation(self):
        """Returns warning when recommendation != NONE even if slo_status is OK."""
        signal = {
            "slo_status": "OK",
            "recommendation": "WARNING",
            "global_nci": 0.85,
            "confidence": 0.75,
        }
        warning = build_nci_status_warning(signal)
        assert warning is not None
        assert "WARNING" in warning

    def test_returns_warning_for_review_recommendation(self):
        """Returns warning when recommendation is REVIEW."""
        signal = {
            "slo_status": "WARN",
            "recommendation": "REVIEW",
            "global_nci": 0.70,
            "confidence": 0.60,
        }
        warning = build_nci_status_warning(signal)
        assert warning is not None
        assert "REVIEW" in warning

    def test_warning_includes_global_nci(self, breach_nci_signal: Dict[str, Any]):
        """Warning includes global_nci percentage."""
        warning = build_nci_status_warning(breach_nci_signal)
        assert warning is not None
        assert "45%" in warning  # 0.45 * 100

    def test_warning_includes_confidence(self, breach_nci_signal: Dict[str, Any]):
        """Warning includes confidence percentage."""
        warning = build_nci_status_warning(breach_nci_signal)
        assert warning is not None
        assert "40%" in warning  # 0.40 * 100

    def test_warning_is_single_string(self, breach_nci_signal: Dict[str, Any]):
        """Warning is a single string (not list)."""
        warning = build_nci_status_warning(breach_nci_signal)
        assert isinstance(warning, str)

    def test_warning_cap_single_warning(self, breach_nci_signal: Dict[str, Any]):
        """Only one warning is returned regardless of violations."""
        # Even with multiple violations, only a single warning string
        warning = build_nci_status_warning(breach_nci_signal)
        assert warning is not None
        # No way to get multiple warnings - function returns Optional[str]
        assert isinstance(warning, str)

    def test_deterministic_output(self, breach_nci_signal: Dict[str, Any]):
        """Warning output is deterministic."""
        w1 = build_nci_status_warning(breach_nci_signal)
        w2 = build_nci_status_warning(breach_nci_signal)
        assert w1 == w2

    def test_handles_nested_format(self, full_nci_result: Dict[str, Any]):
        """Handles nested format from evaluate_nci_p5()."""
        warning = build_nci_status_warning(full_nci_result)
        # full_nci_result has slo_status WARN and recommendation WARNING
        assert warning is not None
        assert "WARNING" in warning


# =============================================================================
# Test: Warning Format Lock (DOC/TEST SYNC)
# =============================================================================


import re


class TestWarningFormatLock:
    """Regression tests to ensure warning format matches documented sample.

    SHADOW MODE CONTRACT:
    - Non-gating: format mismatch is a test failure, not a runtime gate
    - Ensures doc sample in First_Light_External_Verification.md stays in sync

    Frozen format patterns:
    - BREACH: "NCI BREACH: {pct}% consistency (confidence {pct}%)"
    - REVIEW/WARNING: "NCI {REC}: {pct}% consistency (confidence {pct}%)"
    """

    # Frozen regex pattern matching documented sample format
    WARNING_FORMAT_PATTERN = re.compile(
        r"^NCI (BREACH|REVIEW|WARNING): \d+% consistency \(confidence \d+%\)$"
    )

    def test_breach_warning_matches_frozen_format(self):
        """BREACH warning matches frozen format: 'NCI BREACH: {pct}% consistency (confidence {pct}%)'."""
        signal = {
            "slo_status": "BREACH",
            "recommendation": "NONE",
            "global_nci": 0.72,
            "confidence": 0.65,
        }
        warning = build_nci_status_warning(signal)
        assert warning is not None
        assert self.WARNING_FORMAT_PATTERN.match(warning), (
            f"Warning format mismatch. Expected pattern: "
            f"'NCI BREACH: {{pct}}% consistency (confidence {{pct}}%)'. "
            f"Got: '{warning}'"
        )

    def test_review_warning_matches_frozen_format(self):
        """REVIEW warning matches frozen format: 'NCI REVIEW: {pct}% consistency (confidence {pct}%)'."""
        signal = {
            "slo_status": "OK",
            "recommendation": "REVIEW",
            "global_nci": 0.85,
            "confidence": 0.70,
        }
        warning = build_nci_status_warning(signal)
        assert warning is not None
        assert self.WARNING_FORMAT_PATTERN.match(warning), (
            f"Warning format mismatch. Expected pattern: "
            f"'NCI REVIEW: {{pct}}% consistency (confidence {{pct}}%)'. "
            f"Got: '{warning}'"
        )

    def test_warning_warning_matches_frozen_format(self):
        """WARNING recommendation matches frozen format: 'NCI WARNING: {pct}% consistency (confidence {pct}%)'."""
        signal = {
            "slo_status": "WARN",
            "recommendation": "WARNING",
            "global_nci": 0.78,
            "confidence": 0.60,
        }
        warning = build_nci_status_warning(signal)
        assert warning is not None
        assert self.WARNING_FORMAT_PATTERN.match(warning), (
            f"Warning format mismatch. Expected pattern: "
            f"'NCI WARNING: {{pct}}% consistency (confidence {{pct}}%)'. "
            f"Got: '{warning}'"
        )

    def test_documented_sample_exact_match(self):
        """Exact sample from Section 11.10 via shared constant."""
        warning = build_nci_status_warning(NCI_P5_BREACH_INPUT)
        assert warning == NCI_P5_BREACH_SAMPLE, (
            f"Documented sample mismatch. "
            f"Expected: '{NCI_P5_BREACH_SAMPLE}'. "
            f"Got: '{warning}'"
        )

    def test_triangle_lock_doc_regex_implementation(self):
        """Triangle lock: doc sample == regex match == implementation output.

        Single source of truth test ensuring:
        1. Doc sample (from tests/doc_samples.py, matches §11.10) matches frozen regex
        2. Implementation output matches the doc sample exactly
        3. All three are in sync: DOC ↔ REGEX ↔ IMPL

        To update: change NCI_P5_BREACH_SAMPLE in tests/doc_samples.py
        """
        # 1. Doc sample matches frozen regex
        assert self.WARNING_FORMAT_PATTERN.match(NCI_P5_BREACH_SAMPLE), (
            f"Doc sample does not match frozen regex. "
            f"Sample: '{NCI_P5_BREACH_SAMPLE}'"
        )

        # 2. Implementation output
        impl_output = build_nci_status_warning(NCI_P5_BREACH_INPUT)

        # 3. Implementation output matches doc sample exactly
        assert impl_output == NCI_P5_BREACH_SAMPLE, (
            f"Implementation output differs from doc sample. "
            f"Doc: '{NCI_P5_BREACH_SAMPLE}'. Impl: '{impl_output}'"
        )

        # 4. Implementation output matches frozen regex (redundant but explicit)
        assert self.WARNING_FORMAT_PATTERN.match(impl_output), (
            f"Implementation output does not match frozen regex. "
            f"Output: '{impl_output}'"
        )
