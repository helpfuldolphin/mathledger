"""
Tests for scripts/generate_what_if_report.py

Tests fixture telemetry parsing, report schema shape, evidence pack attachment,
and determinism (same input → same output hash).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_what_if_report import (
    attach_to_evidence_pack,
    compute_report_hash,
    extract_field,
    generate_what_if_report,
    load_telemetry_jsonl,
    normalize_telemetry,
    parse_telemetry_line,
)
from backend.governance.what_if_engine import WhatIfConfig, WhatIfReport


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_telemetry() -> List[Dict[str, Any]]:
    """Simple telemetry fixture with known values."""
    return [
        {
            "cycle": 1,
            "timestamp": "2025-01-01T00:00:00Z",
            "invariant_violations": [],
            "in_omega": True,
            "omega_exit_streak": 0,
            "rho": 0.85,
            "rho_collapse_streak": 0,
        },
        {
            "cycle": 2,
            "timestamp": "2025-01-01T00:01:00Z",
            "invariant_violations": [],
            "in_omega": True,
            "omega_exit_streak": 0,
            "rho": 0.82,
            "rho_collapse_streak": 0,
        },
        {
            "cycle": 3,
            "timestamp": "2025-01-01T00:02:00Z",
            "invariant_violations": [],
            "in_omega": True,
            "omega_exit_streak": 0,
            "rho": 0.78,
            "rho_collapse_streak": 0,
        },
    ]


@pytest.fixture
def g2_fail_telemetry() -> List[Dict[str, Any]]:
    """Telemetry with G2 invariant failure."""
    return [
        {
            "cycle": 1,
            "timestamp": "2025-01-01T00:00:00Z",
            "invariant_violations": ["monotone_violated"],
            "in_omega": True,
            "omega_exit_streak": 0,
            "rho": 0.9,
            "rho_collapse_streak": 0,
        },
    ]


@pytest.fixture
def g3_fail_telemetry() -> List[Dict[str, Any]]:
    """Telemetry with G3 safe region failure."""
    return [
        {
            "cycle": 1,
            "timestamp": "2025-01-01T00:00:00Z",
            "invariant_violations": [],
            "in_omega": False,
            "omega_exit_streak": 150,  # Above 100 threshold
            "rho": 0.9,
            "rho_collapse_streak": 0,
        },
    ]


@pytest.fixture
def g4_fail_telemetry() -> List[Dict[str, Any]]:
    """Telemetry with G4 RSI failure."""
    return [
        {
            "cycle": 1,
            "timestamp": "2025-01-01T00:00:00Z",
            "invariant_violations": [],
            "in_omega": True,
            "omega_exit_streak": 0,
            "rho": 0.3,  # Below 0.4 threshold
            "rho_collapse_streak": 15,  # Above 10 streak threshold
        },
    ]


@pytest.fixture
def alternative_field_names_telemetry() -> List[Dict[str, Any]]:
    """Telemetry using alternative field names."""
    return [
        {
            "step": 1,
            "ts": "2025-01-01T00:00:00Z",
            "violations": [],
            "is_safe": True,
            "outside_omega_cycles": 0,
            "rsi": 0.88,
            "rsi_streak": 0,
        },
        {
            "iteration": 2,
            "time": "2025-01-01T00:01:00Z",
            "failed_invariants": ["CDI-010"],
            "omega_safe": False,
            "safe_region_exit_streak": 5,
            "stability": 0.75,
            "stability_collapse_streak": 2,
        },
    ]


@pytest.fixture
def temp_jsonl_file(simple_telemetry) -> Path:
    """Create temporary JSONL file with simple telemetry."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as f:
        for record in simple_telemetry:
            f.write(json.dumps(record) + "\n")
        return Path(f.name)


# =============================================================================
# FIELD EXTRACTION TESTS
# =============================================================================

class TestFieldExtraction:
    """Tests for telemetry field extraction."""

    def test_extract_primary_field_name(self):
        """Should extract field using primary name."""
        data = {"cycle": 42, "rho": 0.8}
        assert extract_field(data, "cycle") == 42
        assert extract_field(data, "rho") == 0.8

    def test_extract_alternative_field_name(self):
        """Should extract field using alternative names."""
        data = {"step": 42, "rsi": 0.8}
        assert extract_field(data, "cycle") == 42
        assert extract_field(data, "rho") == 0.8

    def test_extract_missing_field_returns_default(self):
        """Should return default for missing fields."""
        data = {"foo": "bar"}
        assert extract_field(data, "cycle", 0) == 0
        assert extract_field(data, "rho", 1.0) == 1.0

    def test_extract_invariant_violations(self):
        """Should extract invariant_violations from various names."""
        data1 = {"invariant_violations": ["a", "b"]}
        data2 = {"violations": ["x"]}
        data3 = {"failed_invariants": ["y", "z"]}

        assert extract_field(data1, "invariant_violations") == ["a", "b"]
        assert extract_field(data2, "invariant_violations") == ["x"]
        assert extract_field(data3, "invariant_violations") == ["y", "z"]


# =============================================================================
# TELEMETRY PARSING TESTS
# =============================================================================

class TestTelemetryParsing:
    """Tests for JSONL parsing and normalization."""

    def test_parse_valid_json_line(self):
        """Should parse valid JSON line."""
        line = '{"cycle": 1, "rho": 0.9}'
        result = parse_telemetry_line(line, 1)
        assert result == {"cycle": 1, "rho": 0.9}

    def test_parse_empty_line_returns_none(self):
        """Should return None for empty lines."""
        assert parse_telemetry_line("", 1) is None
        assert parse_telemetry_line("   ", 1) is None

    def test_parse_comment_line_returns_none(self):
        """Should return None for comment lines."""
        assert parse_telemetry_line("# This is a comment", 1) is None

    def test_parse_invalid_json_returns_none(self):
        """Should return None for invalid JSON."""
        result = parse_telemetry_line("{not valid json}", 1)
        assert result is None

    def test_normalize_telemetry_with_all_fields(self, simple_telemetry):
        """Should normalize telemetry preserving all fields."""
        raw = simple_telemetry[0]
        normalized = normalize_telemetry(raw, 1)

        assert normalized["cycle"] == 1
        assert normalized["timestamp"] == "2025-01-01T00:00:00Z"
        assert normalized["invariant_violations"] == []
        assert normalized["in_omega"] is True
        assert normalized["omega_exit_streak"] == 0
        assert normalized["rho"] == 0.85
        assert normalized["rho_collapse_streak"] == 0

    def test_normalize_telemetry_with_alternative_fields(
        self, alternative_field_names_telemetry
    ):
        """Should normalize telemetry using alternative field names."""
        raw = alternative_field_names_telemetry[0]
        normalized = normalize_telemetry(raw, 1)

        assert normalized["cycle"] == 1
        assert normalized["timestamp"] == "2025-01-01T00:00:00Z"
        assert normalized["in_omega"] is True
        assert normalized["rho"] == 0.88

    def test_normalize_string_boolean(self):
        """Should convert string booleans."""
        raw = {"in_omega": "true"}
        normalized = normalize_telemetry(raw, 1)
        assert normalized["in_omega"] is True

        raw2 = {"in_omega": "false"}
        normalized2 = normalize_telemetry(raw2, 1)
        assert normalized2["in_omega"] is False

    def test_normalize_string_violations(self):
        """Should split comma-separated violation strings."""
        raw = {"invariant_violations": "error1, error2, error3"}
        normalized = normalize_telemetry(raw, 1)
        assert normalized["invariant_violations"] == ["error1", "error2", "error3"]

    def test_load_telemetry_jsonl(self, temp_jsonl_file):
        """Should load and normalize JSONL file."""
        telemetry = load_telemetry_jsonl(temp_jsonl_file)

        assert len(telemetry) == 3
        assert telemetry[0]["cycle"] == 1
        assert telemetry[1]["cycle"] == 2
        assert telemetry[2]["cycle"] == 3


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

class TestReportGeneration:
    """Tests for What-If report generation."""

    def test_generate_report_from_simple_telemetry(self, temp_jsonl_file):
        """Should generate report from simple telemetry."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
            run_id="test-run-001",
        )

        assert report.mode == "HYPOTHETICAL"
        assert report.run_id == "test-run-001"
        assert report.total_cycles == 3
        assert report.hypothetical_blocks == 0  # No failures in simple telemetry
        assert report.hypothetical_allows == 3

        # Verify output file was written
        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            written_report = json.load(f)
        assert written_report["mode"] == "HYPOTHETICAL"
        assert written_report["run_id"] == "test-run-001"

    def test_generate_report_with_g2_failure(self, g2_fail_telemetry):
        """Should detect G2 invariant failures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in g2_fail_telemetry:
                f.write(json.dumps(record) + "\n")
            jsonl_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )

        assert report.hypothetical_blocks == 1
        assert "G2_INVARIANT" in report.blocking_gate_distribution

    def test_generate_report_with_g3_failure(self, g3_fail_telemetry):
        """Should detect G3 safe region failures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in g3_fail_telemetry:
                f.write(json.dumps(record) + "\n")
            jsonl_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )

        assert report.hypothetical_blocks == 1
        assert "G3_SAFE_REGION" in report.blocking_gate_distribution

    def test_generate_report_with_g4_failure(self, g4_fail_telemetry):
        """Should detect G4 RSI failures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in g4_fail_telemetry:
                f.write(json.dumps(record) + "\n")
            jsonl_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )

        assert report.hypothetical_blocks == 1
        assert "G4_SOFT" in report.blocking_gate_distribution


# =============================================================================
# REPORT SCHEMA SHAPE TESTS
# =============================================================================

class TestReportSchemaShape:
    """Tests for report JSON schema compliance."""

    def test_report_has_required_top_level_fields(self, temp_jsonl_file):
        """Should have all required top-level fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Top-level fields
        assert "schema_version" in report
        assert "run_id" in report
        assert "analysis_timestamp" in report
        assert "mode" in report
        assert "summary" in report
        assert "gate_analysis" in report
        assert "notable_events" in report
        assert "calibration_recommendations" in report
        assert "auditor_notes" in report

    def test_report_summary_structure(self, temp_jsonl_file):
        """Should have correct summary structure."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        summary = report["summary"]
        assert "total_cycles" in summary
        assert "hypothetical_allows" in summary
        assert "hypothetical_blocks" in summary
        assert "hypothetical_block_rate" in summary
        assert "blocking_gate_distribution" in summary
        assert "max_consecutive_blocks" in summary
        assert "first_hypothetical_block_cycle" in summary

    def test_report_gate_analysis_structure(self, temp_jsonl_file):
        """Should have correct gate analysis structure."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        gate_analysis = report["gate_analysis"]
        assert "g2_invariant" in gate_analysis
        assert "g3_safe_region" in gate_analysis
        assert "g4_soft" in gate_analysis

        # Each gate analysis should have these fields
        for gate in ["g2_invariant", "g3_safe_region", "g4_soft"]:
            analysis = gate_analysis[gate]
            assert "hypothetical_fail_count" in analysis
            assert "hypothetical_block_count" in analysis
            assert "fail_rate" in analysis
            assert "mean_margin_to_threshold" in analysis
            assert "threshold_breaches" in analysis

    def test_mode_is_always_hypothetical(self, temp_jsonl_file):
        """Mode should always be HYPOTHETICAL (SHADOW MODE)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        assert report.mode == "HYPOTHETICAL"

        with open(output_path, "r", encoding="utf-8") as f:
            written = json.load(f)
        assert written["mode"] == "HYPOTHETICAL"


# =============================================================================
# EVIDENCE PACK ATTACHMENT TESTS
# =============================================================================

class TestEvidencePackAttachment:
    """Tests for evidence pack attachment functionality."""

    def test_attach_to_empty_evidence(self, temp_jsonl_file):
        """Should create governance structure in empty evidence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        evidence = {}
        evidence = attach_to_evidence_pack(evidence, report)

        assert "governance" in evidence
        assert "what_if_analysis" in evidence["governance"]
        assert "report" in evidence["governance"]["what_if_analysis"]
        assert "report_sha256" in evidence["governance"]["what_if_analysis"]
        assert "attached_at" in evidence["governance"]["what_if_analysis"]

    def test_attach_to_existing_evidence(self, temp_jsonl_file):
        """Should preserve existing evidence fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        evidence = {
            "proof_hash": "abc123",
            "governance": {
                "final_check": {"verdict": "PASS"}
            }
        }
        evidence = attach_to_evidence_pack(evidence, report)

        # Original fields preserved
        assert evidence["proof_hash"] == "abc123"
        assert evidence["governance"]["final_check"]["verdict"] == "PASS"

        # New fields added
        assert "what_if_analysis" in evidence["governance"]

    def test_report_hash_is_sha256(self, temp_jsonl_file):
        """Should compute valid SHA-256 hash."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        report_hash = compute_report_hash(report)

        # SHA-256 produces 64 hex characters
        assert len(report_hash) == 64
        assert all(c in "0123456789abcdef" for c in report_hash)

    def test_report_hash_matches_manual_computation(self, temp_jsonl_file):
        """Hash should match manual SHA-256 computation."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=temp_jsonl_file,
            output_path=output_path,
        )

        # Compute hash manually
        report_json = json.dumps(report.to_dict(), sort_keys=True)
        expected_hash = hashlib.sha256(report_json.encode("utf-8")).hexdigest()

        # Compare to function output
        actual_hash = compute_report_hash(report)
        assert actual_hash == expected_hash


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic output (same input → same hash)."""

    def test_same_input_produces_same_hash(self, simple_telemetry):
        """Same telemetry input should produce same report hash."""
        hashes = []

        for _ in range(3):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as f:
                for record in simple_telemetry:
                    f.write(json.dumps(record) + "\n")
                jsonl_path = Path(f.name)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
                output_path = Path(out_f.name)

            # Use fixed run_id for determinism
            report = generate_what_if_report(
                telemetry_path=jsonl_path,
                output_path=output_path,
                run_id="determinism-test",
            )

            # Compute hash excluding timestamp (which varies)
            report_dict = report.to_dict()
            # Remove variable fields for comparison
            del report_dict["analysis_timestamp"]

            report_json = json.dumps(report_dict, sort_keys=True)
            h = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
            hashes.append(h)

        # All hashes should be identical
        assert hashes[0] == hashes[1] == hashes[2]

    def test_different_input_produces_different_hash(
        self, simple_telemetry, g2_fail_telemetry
    ):
        """Different telemetry should produce different hashes."""
        # Generate report from simple telemetry
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in simple_telemetry:
                f.write(json.dumps(record) + "\n")
            path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path1 = Path(out_f.name)

        report1 = generate_what_if_report(
            telemetry_path=path1,
            output_path=output_path1,
            run_id="test-1",
        )

        # Generate report from failure telemetry
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in g2_fail_telemetry:
                f.write(json.dumps(record) + "\n")
            path2 = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path2 = Path(out_f.name)

        report2 = generate_what_if_report(
            telemetry_path=path2,
            output_path=output_path2,
            run_id="test-2",
        )

        hash1 = compute_report_hash(report1)
        hash2 = compute_report_hash(report2)

        assert hash1 != hash2


# =============================================================================
# CONFIGURATION OVERRIDE TESTS
# =============================================================================

class TestConfigurationOverrides:
    """Tests for configuration threshold overrides."""

    def test_custom_rho_threshold_affects_g4(self):
        """Custom rho threshold should change G4 behavior."""
        telemetry = [
            {
                "cycle": 1,
                "timestamp": "2025-01-01T00:00:00Z",
                "rho": 0.35,  # Would fail with default 0.4, pass with 0.3
                "rho_collapse_streak": 15,
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in telemetry:
                f.write(json.dumps(record) + "\n")
            jsonl_path = Path(f.name)

        # Default config (rho_min=0.4) should fail
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report_default = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )
        assert report_default.hypothetical_blocks == 1

        # Relaxed config (rho_min=0.3) should pass
        config = WhatIfConfig(rho_min=0.3)
        report_relaxed = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
            config=config,
        )
        assert report_relaxed.hypothetical_blocks == 0

    def test_custom_omega_threshold_affects_g3(self):
        """Custom omega threshold should change G3 behavior."""
        telemetry = [
            {
                "cycle": 1,
                "timestamp": "2025-01-01T00:00:00Z",
                "in_omega": False,
                "omega_exit_streak": 80,  # Fails at 100, passes at 50
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for record in telemetry:
                f.write(json.dumps(record) + "\n")
            jsonl_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        # Default (100) should pass
        report_default = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )
        assert report_default.hypothetical_blocks == 0

        # Strict (50) should fail
        config = WhatIfConfig(omega_exit_threshold=50)
        report_strict = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
            config=config,
        )
        assert report_strict.hypothetical_blocks == 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_telemetry_file(self):
        """Should handle empty telemetry file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            # Write nothing
            jsonl_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            output_path = Path(out_f.name)

        report = generate_what_if_report(
            telemetry_path=jsonl_path,
            output_path=output_path,
        )

        assert report.total_cycles == 0
        assert report.hypothetical_blocks == 0
        assert report.hypothetical_allows == 0
        assert report.hypothetical_block_rate == 0.0

    def test_telemetry_with_comments_and_blanks(self):
        """Should skip comments and blank lines."""
        content = """# Header comment
{"cycle": 1, "rho": 0.9}

{"cycle": 2, "rho": 0.85}
# Mid-file comment

{"cycle": 3, "rho": 0.8}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            jsonl_path = Path(f.name)

        telemetry = load_telemetry_jsonl(jsonl_path)
        assert len(telemetry) == 3

    def test_telemetry_with_invalid_json_lines(self):
        """Should skip invalid JSON lines and continue."""
        content = """{"cycle": 1, "rho": 0.9}
{not valid json}
{"cycle": 3, "rho": 0.8}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            jsonl_path = Path(f.name)

        telemetry = load_telemetry_jsonl(jsonl_path)
        # Should have 2 valid records
        assert len(telemetry) == 2
        assert telemetry[0]["cycle"] == 1
        assert telemetry[1]["cycle"] == 3
