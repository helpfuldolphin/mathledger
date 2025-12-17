"""
Tests for What-If Auto-Detection and Status Signal Extraction.

Tests:
- Auto-detection of what_if_report.json in search paths
- Status signal extraction (hypothetical_block_rate, blocking_gate_distribution, first_block_cycle)
- Malformed report handling (warnings, not crashes)
- Deterministic hashing
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.governance.evidence_pack import (
    WhatIfStatusSignal,
    detect_what_if_report,
    extract_what_if_status,
    attach_what_if_to_evidence,
    _compute_report_hash,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_what_if_report() -> Dict[str, Any]:
    """Valid What-If report matching schema."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test-run-001",
        "analysis_timestamp": "2025-01-01T00:00:00Z",
        "mode": "HYPOTHETICAL",
        "summary": {
            "total_cycles": 100,
            "hypothetical_allows": 85,
            "hypothetical_blocks": 15,
            "hypothetical_block_rate": 0.15,
            "blocking_gate_distribution": {
                "G2_INVARIANT": 5,
                "G3_SAFE_REGION": 7,
                "G4_SOFT": 3,
            },
            "max_consecutive_blocks": 4,
            "first_hypothetical_block_cycle": 12,
        },
        "gate_analysis": {
            "g2_invariant": {"hypothetical_fail_count": 5},
            "g3_safe_region": {"hypothetical_fail_count": 7},
            "g4_soft": {"hypothetical_fail_count": 3},
        },
        "notable_events": [],
        "calibration_recommendations": [],
        "auditor_notes": "Test run.",
    }


@pytest.fixture
def malformed_report_missing_summary() -> Dict[str, Any]:
    """Report missing summary section."""
    return {
        "schema_version": "1.0.0",
        "mode": "HYPOTHETICAL",
        # No summary section
    }


@pytest.fixture
def malformed_report_wrong_mode() -> Dict[str, Any]:
    """Report with wrong mode (should be HYPOTHETICAL, not SHADOW)."""
    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",  # Wrong!
        "summary": {
            "total_cycles": 10,
            "hypothetical_blocks": 2,
            "hypothetical_block_rate": 0.2,
        },
    }


@pytest.fixture
def malformed_report_invalid_types() -> Dict[str, Any]:
    """Report with invalid field types."""
    return {
        "schema_version": "1.0.0",
        "mode": "HYPOTHETICAL",
        "summary": {
            "total_cycles": "not_a_number",  # Should be int
            "hypothetical_blocks": None,  # Should be int
            "hypothetical_block_rate": "invalid",  # Should be float
            "blocking_gate_distribution": "not_a_dict",  # Should be dict
            "first_hypothetical_block_cycle": "not_int",  # Should be int
        },
    }


@pytest.fixture
def temp_report_file(valid_what_if_report) -> Path:
    """Create temporary what_if_report.json file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as f:
        json.dump(valid_what_if_report, f)
        return Path(f.name)


@pytest.fixture
def temp_dir_with_report(valid_what_if_report) -> Path:
    """Create temp directory with what_if_report.json."""
    temp_dir = Path(tempfile.mkdtemp())
    report_path = temp_dir / "what_if_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(valid_what_if_report, f)
    return temp_dir


# =============================================================================
# AUTO-DETECTION TESTS
# =============================================================================

class TestAutoDetection:
    """Tests for what_if_report.json auto-detection."""

    def test_detect_report_in_explicit_path(self, temp_dir_with_report):
        """Should find report in explicitly provided search path."""
        found = detect_what_if_report(search_paths=[temp_dir_with_report])
        assert found is not None
        assert found.exists()
        assert found.name == "what_if_report.json"

    def test_detect_report_returns_none_when_not_found(self):
        """Should return None when report not found."""
        nonexistent = Path(tempfile.mkdtemp())
        found = detect_what_if_report(search_paths=[nonexistent])
        assert found is None

    def test_detect_report_custom_filename(self, temp_dir_with_report):
        """Should support custom filename."""
        # Create file with different name
        custom_path = temp_dir_with_report / "custom_report.json"
        with open(custom_path, "w") as f:
            json.dump({}, f)

        found = detect_what_if_report(
            search_paths=[temp_dir_with_report],
            filename="custom_report.json",
        )
        assert found is not None
        assert found.name == "custom_report.json"

    def test_detect_report_first_match_wins(self, valid_what_if_report):
        """Should return first match in search order."""
        dir1 = Path(tempfile.mkdtemp())
        dir2 = Path(tempfile.mkdtemp())

        # Create report in both dirs
        path1 = dir1 / "what_if_report.json"
        path2 = dir2 / "what_if_report.json"

        with open(path1, "w") as f:
            json.dump({"marker": "first"}, f)
        with open(path2, "w") as f:
            json.dump({"marker": "second"}, f)

        found = detect_what_if_report(search_paths=[dir1, dir2])
        assert found == path1


# =============================================================================
# STATUS EXTRACTION TESTS
# =============================================================================

class TestStatusExtraction:
    """Tests for status signal extraction from What-If reports."""

    def test_extract_status_from_valid_report(self, valid_what_if_report):
        """Should extract all status fields from valid report."""
        status, warnings = extract_what_if_status(report_dict=valid_what_if_report)

        assert status is not None
        assert len(warnings) == 0

        assert status.hypothetical_block_rate == 0.15
        assert status.blocking_gate_distribution == {
            "G2_INVARIANT": 5,
            "G3_SAFE_REGION": 7,
            "G4_SOFT": 3,
        }
        assert status.first_block_cycle == 12
        assert status.total_cycles == 100
        assert status.hypothetical_blocks == 15
        assert status.mode == "HYPOTHETICAL"

    def test_extract_status_from_file_path(self, temp_report_file):
        """Should extract status from file path."""
        status, warnings = extract_what_if_status(report_path=temp_report_file)

        assert status is not None
        assert status.hypothetical_block_rate == 0.15

    def test_extract_status_computes_sha256(self, valid_what_if_report):
        """Should compute SHA-256 hash of report."""
        status, _ = extract_what_if_status(report_dict=valid_what_if_report)

        assert status is not None
        assert len(status.report_sha256) == 64
        assert all(c in "0123456789abcdef" for c in status.report_sha256)

    def test_extract_status_no_input_returns_none_with_warning(self):
        """Should return None with warning when no input provided."""
        status, warnings = extract_what_if_status()

        assert status is None
        assert len(warnings) == 1
        assert "No report path or dictionary provided" in warnings[0]


# =============================================================================
# MALFORMED REPORT TESTS
# =============================================================================

class TestMalformedReportHandling:
    """Tests for malformed report handling (warnings, not crashes)."""

    def test_missing_summary_yields_defaults(self, malformed_report_missing_summary):
        """Should use defaults for missing summary fields."""
        status, warnings = extract_what_if_status(
            report_dict=malformed_report_missing_summary
        )

        assert status is not None
        # Defaults applied
        assert status.hypothetical_block_rate == 0.0
        assert status.blocking_gate_distribution == {}
        assert status.first_block_cycle is None
        assert status.total_cycles == 0
        assert status.hypothetical_blocks == 0

    def test_wrong_mode_yields_warning(self, malformed_report_wrong_mode):
        """Should warn when mode is not HYPOTHETICAL."""
        status, warnings = extract_what_if_status(
            report_dict=malformed_report_wrong_mode
        )

        assert status is not None
        assert status.mode == "SHADOW"  # Preserves actual value

        # Warning about mode constraint
        assert any("HYPOTHETICAL" in w and "Mode constraint violated" in w for w in warnings)

    def test_invalid_types_yield_warnings_and_defaults(
        self, malformed_report_invalid_types
    ):
        """Should warn and use defaults for invalid field types."""
        status, warnings = extract_what_if_status(
            report_dict=malformed_report_invalid_types
        )

        assert status is not None

        # Should have multiple warnings
        assert len(warnings) >= 4

        # Defaults applied
        assert status.hypothetical_block_rate == 0.0
        assert status.total_cycles == 0
        assert status.hypothetical_blocks == 0
        assert status.blocking_gate_distribution == {}
        assert status.first_block_cycle is None

    def test_invalid_json_file_yields_warning(self):
        """Should warn on invalid JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json}")
            path = Path(f.name)

        status, warnings = extract_what_if_status(report_path=path)

        assert status is None
        assert len(warnings) == 1
        assert "Invalid JSON" in warnings[0]

    def test_nonexistent_file_yields_warning(self):
        """Should warn on nonexistent file."""
        status, warnings = extract_what_if_status(
            report_path=Path("/nonexistent/path/report.json")
        )

        assert status is None
        assert len(warnings) == 1
        assert "not found" in warnings[0]


# =============================================================================
# DETERMINISTIC HASHING TESTS
# =============================================================================

class TestDeterministicHashing:
    """Tests for deterministic SHA-256 hashing."""

    def test_same_report_produces_same_hash(self, valid_what_if_report):
        """Same report should always produce same hash."""
        hashes = []
        for _ in range(3):
            status, _ = extract_what_if_status(report_dict=valid_what_if_report)
            hashes.append(status.report_sha256)

        assert hashes[0] == hashes[1] == hashes[2]

    def test_different_reports_produce_different_hashes(
        self, valid_what_if_report, malformed_report_wrong_mode
    ):
        """Different reports should produce different hashes."""
        status1, _ = extract_what_if_status(report_dict=valid_what_if_report)
        status2, _ = extract_what_if_status(report_dict=malformed_report_wrong_mode)

        assert status1.report_sha256 != status2.report_sha256

    def test_hash_matches_manual_computation(self, valid_what_if_report):
        """Hash should match manual SHA-256 computation."""
        status, _ = extract_what_if_status(report_dict=valid_what_if_report)

        # Manual computation
        report_json = json.dumps(valid_what_if_report, sort_keys=True)
        expected = hashlib.sha256(report_json.encode("utf-8")).hexdigest()

        assert status.report_sha256 == expected

    def test_hash_uses_sorted_keys(self, valid_what_if_report):
        """Hash computation should use sorted keys for determinism."""
        # Create report with different key order
        reordered = dict(reversed(list(valid_what_if_report.items())))

        status1, _ = extract_what_if_status(report_dict=valid_what_if_report)
        status2, _ = extract_what_if_status(report_dict=reordered)

        # Should produce same hash due to sort_keys=True
        assert status1.report_sha256 == status2.report_sha256


# =============================================================================
# EVIDENCE ATTACHMENT TESTS
# =============================================================================

class TestEvidenceAttachment:
    """Tests for attaching What-If report to evidence pack."""

    def test_attach_to_empty_evidence(self, valid_what_if_report):
        """Should create proper structure in empty evidence."""
        evidence = {}
        evidence, warnings = attach_what_if_to_evidence(
            evidence,
            report_dict=valid_what_if_report,
            auto_detect=False,
        )

        assert len(warnings) == 0

        # Check governance.what_if_analysis
        assert "governance" in evidence
        assert "what_if_analysis" in evidence["governance"]
        assert "report" in evidence["governance"]["what_if_analysis"]
        assert "status" in evidence["governance"]["what_if_analysis"]
        assert "report_sha256" in evidence["governance"]["what_if_analysis"]
        assert "attached_at" in evidence["governance"]["what_if_analysis"]

        # Check signals.what_if
        assert "signals" in evidence
        assert "what_if" in evidence["signals"]

    def test_attach_preserves_existing_evidence(self, valid_what_if_report):
        """Should preserve existing evidence fields."""
        evidence = {
            "proof_hash": "abc123",
            "governance": {
                "final_check": {"verdict": "ALLOW"}
            },
            "signals": {
                "usla": {"rho": 0.85}
            }
        }

        evidence, _ = attach_what_if_to_evidence(
            evidence,
            report_dict=valid_what_if_report,
            auto_detect=False,
        )

        # Original fields preserved
        assert evidence["proof_hash"] == "abc123"
        assert evidence["governance"]["final_check"]["verdict"] == "ALLOW"
        assert evidence["signals"]["usla"]["rho"] == 0.85

        # New fields added
        assert "what_if_analysis" in evidence["governance"]
        assert "what_if" in evidence["signals"]

    def test_signals_what_if_contains_required_fields(self, valid_what_if_report):
        """signals.what_if should contain required status fields."""
        evidence = {}
        evidence, _ = attach_what_if_to_evidence(
            evidence,
            report_dict=valid_what_if_report,
            auto_detect=False,
        )

        what_if = evidence["signals"]["what_if"]

        assert "hypothetical_block_rate" in what_if
        assert "blocking_gate_distribution" in what_if
        assert "first_block_cycle" in what_if
        assert "total_cycles" in what_if
        assert "hypothetical_blocks" in what_if
        assert "mode" in what_if
        assert "report_sha256" in what_if

        # Verify values
        assert what_if["hypothetical_block_rate"] == 0.15
        assert what_if["blocking_gate_distribution"]["G2_INVARIANT"] == 5
        assert what_if["first_block_cycle"] == 12

    def test_attach_with_auto_detect(self, temp_dir_with_report):
        """Should auto-detect report when path not provided."""
        evidence = {}
        evidence, warnings = attach_what_if_to_evidence(
            evidence,
            auto_detect=True,
            search_paths=[temp_dir_with_report],
        )

        assert len(warnings) == 0
        assert "what_if" in evidence.get("signals", {})

    def test_attach_without_report_returns_warning(self):
        """Should return warning when no report found."""
        evidence = {}
        nonexistent = Path(tempfile.mkdtemp())

        evidence, warnings = attach_what_if_to_evidence(
            evidence,
            auto_detect=True,
            search_paths=[nonexistent],
        )

        assert len(warnings) == 1
        assert "found" in warnings[0].lower()  # "No ... found in search paths"
