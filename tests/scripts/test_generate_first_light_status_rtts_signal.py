"""
Tests for RTTS validation signal in first_light_status.json

Phase X P5.2: Tests for RTTS status surface in First-Light status.

Tests:
1. Signal extracted when rtts_validation.json present
2. Signal not present when rtts_validation.json missing
3. overall_status surfaced correctly
4. top3_mock_codes sorted alphabetically
5. Warning generated for WARN status (normal form)
6. Warning generated for CRITICAL status (normal form)
7. top_driver_category computed correctly (frozen enum)
8. top_driver_codes_top3 sorted from top category
9. Manifest reference loading with integrity check
10. Deterministic ordering
11. extraction_source provenance tracking
12. Warning string determinism

SHADOW MODE CONTRACT:
- All tests verify OBSERVATIONAL outputs only
- No gating or enforcement is tested
- mode="SHADOW" and action="LOGGED_ONLY" are always verified

FROZEN ENUMS:
- ExtractionSource: MANIFEST_REFERENCE | DIRECT_DISCOVERY | MISSING
- DriverCategory: STATISTICAL | CORRELATION | CONTINUITY | UNKNOWN
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


# =============================================================================
# Tests for extract_rtts_status_signal
# =============================================================================

class TestExtractRTTSStatusSignal:
    """Tests for extract_rtts_status_signal function."""

    def test_signal_with_rtts_available(self):
        """Test signal extraction when RTTS data is available."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_signal,
            DriverCategory,
            ExtractionSource,
        )

        rtts_validation = {
            "schema_version": "1.0.0",
            "overall_status": "OK",
            "warning_count": 0,
            "mock_detection_flags": [],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        assert signal["available"] is True
        assert signal["overall_status"] == "OK"
        assert signal["violation_count"] == 0
        assert signal["top3_mock_codes"] == []
        assert signal["top_driver_category"] == DriverCategory.UNKNOWN  # Frozen enum
        assert signal["top_driver_codes_top3"] == []
        assert signal["mode"] == "SHADOW"
        assert signal["action"] == "LOGGED_ONLY"
        assert signal["extraction_source"] == ExtractionSource.DIRECT_DISCOVERY

    def test_signal_without_rtts_unavailable(self):
        """Test signal extraction when RTTS data is missing."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_signal,
            DriverCategory,
            ExtractionSource,
        )

        signal = extract_rtts_status_signal(None)

        assert signal["available"] is False
        assert signal["overall_status"] == "UNKNOWN"
        assert signal["violation_count"] == 0
        assert signal["top3_mock_codes"] == []
        assert signal["top_driver_category"] == DriverCategory.UNKNOWN  # Frozen enum
        assert signal["top_driver_codes_top3"] == []
        assert signal["extraction_source"] == ExtractionSource.MISSING

    def test_top3_mock_codes_sorted(self):
        """Test that top3_mock_codes are sorted alphabetically."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal

        rtts_validation = {
            "overall_status": "WARN",
            "warning_count": 5,
            "mock_detection_flags": ["MOCK-009", "MOCK-001", "MOCK-005", "MOCK-003", "MOCK-007"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        # Should be top 3 sorted: MOCK-001, MOCK-003, MOCK-005
        assert signal["top3_mock_codes"] == ["MOCK-001", "MOCK-003", "MOCK-005"]

    def test_top3_mock_codes_less_than_3(self):
        """Test that top3_mock_codes works with fewer than 3 flags."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal

        rtts_validation = {
            "overall_status": "ATTENTION",
            "warning_count": 2,
            "mock_detection_flags": ["MOCK-003", "MOCK-001"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        assert signal["top3_mock_codes"] == ["MOCK-001", "MOCK-003"]

    def test_extraction_source_explicit(self):
        """Test that extraction_source can be explicitly set."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_signal,
            ExtractionSource,
        )

        rtts_validation = {
            "overall_status": "OK",
            "warning_count": 0,
            "mock_detection_flags": [],
        }

        signal = extract_rtts_status_signal(
            rtts_validation,
            extraction_source=ExtractionSource.MANIFEST_REFERENCE
        )

        assert signal["extraction_source"] == ExtractionSource.MANIFEST_REFERENCE


# =============================================================================
# Tests for top_driver computation
# =============================================================================

class TestTopDriverComputation:
    """Tests for top_driver_category and top_driver_codes_top3 (frozen enums)."""

    def test_top_driver_statistical(self):
        """Test top_driver when STATISTICAL is dominant category."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal, DriverCategory

        # MOCK-001, MOCK-002, MOCK-005, MOCK-006, MOCK-007 are STATISTICAL
        rtts_validation = {
            "overall_status": "WARN",
            "warning_count": 4,
            "mock_detection_flags": ["MOCK-001", "MOCK-002", "MOCK-005", "MOCK-003"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        assert signal["top_driver_category"] == DriverCategory.STATISTICAL
        assert signal["top_driver_codes_top3"] == ["MOCK-001", "MOCK-002", "MOCK-005"]

    def test_top_driver_correlation(self):
        """Test top_driver when CORRELATION is dominant category."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal, DriverCategory

        # MOCK-003, MOCK-004 are CORRELATION
        rtts_validation = {
            "overall_status": "WARN",
            "warning_count": 3,
            "mock_detection_flags": ["MOCK-003", "MOCK-004", "MOCK-001"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        assert signal["top_driver_category"] == DriverCategory.CORRELATION
        assert signal["top_driver_codes_top3"] == ["MOCK-003", "MOCK-004"]

    def test_top_driver_continuity(self):
        """Test top_driver when CONTINUITY is dominant category."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal, DriverCategory

        # MOCK-009, MOCK-010 are CONTINUITY
        rtts_validation = {
            "overall_status": "CRITICAL",
            "warning_count": 3,
            "mock_detection_flags": ["MOCK-009", "MOCK-010", "MOCK-001"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        assert signal["top_driver_category"] == DriverCategory.CONTINUITY
        assert signal["top_driver_codes_top3"] == ["MOCK-009", "MOCK-010"]

    def test_top_driver_tie_alphabetic(self):
        """Test top_driver uses alphabetic order to break ties."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal, DriverCategory

        # 1 STATISTICAL, 1 CONTINUITY - tie broken alphabetically (CONTINUITY < STATISTICAL)
        rtts_validation = {
            "overall_status": "ATTENTION",
            "warning_count": 2,
            "mock_detection_flags": ["MOCK-001", "MOCK-009"],
        }

        signal = extract_rtts_status_signal(rtts_validation)

        # "CONTINUITY" comes before "STATISTICAL" alphabetically
        assert signal["top_driver_category"] == DriverCategory.CONTINUITY
        assert signal["top_driver_codes_top3"] == ["MOCK-009"]

    def test_top_driver_deterministic(self):
        """Test that top_driver computation is deterministic."""
        from backend.health.rtts_status_adapter import extract_rtts_status_signal

        rtts_validation = {
            "overall_status": "WARN",
            "warning_count": 5,
            "mock_detection_flags": ["MOCK-007", "MOCK-001", "MOCK-005", "MOCK-002", "MOCK-006"],
        }

        # Run multiple times
        results = [extract_rtts_status_signal(rtts_validation) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert result["top_driver_category"] == results[0]["top_driver_category"]
            assert result["top_driver_codes_top3"] == results[0]["top_driver_codes_top3"]

    def test_frozen_enum_values(self):
        """Test that DriverCategory enum values are frozen strings."""
        from backend.health.rtts_status_adapter import DriverCategory, VALID_DRIVER_CATEGORIES

        # Verify enum values are uppercase strings
        assert DriverCategory.STATISTICAL == "STATISTICAL"
        assert DriverCategory.CORRELATION == "CORRELATION"
        assert DriverCategory.CONTINUITY == "CONTINUITY"
        assert DriverCategory.UNKNOWN == "UNKNOWN"

        # Verify frozen set contains all values
        assert len(VALID_DRIVER_CATEGORIES) == 4
        assert DriverCategory.STATISTICAL in VALID_DRIVER_CATEGORIES
        assert DriverCategory.CORRELATION in VALID_DRIVER_CATEGORIES
        assert DriverCategory.CONTINUITY in VALID_DRIVER_CATEGORIES
        assert DriverCategory.UNKNOWN in VALID_DRIVER_CATEGORIES


# =============================================================================
# Tests for generate_rtts_warning
# =============================================================================

class TestGenerateRTTSWarning:
    """Tests for generate_rtts_warning function (WARNING NORMAL FORM)."""

    def test_warning_for_warn_status_normal_form(self):
        """Test that warning is generated for WARN status in normal form."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "WARN",
            "violation_count": 3,
            "top3_mock_codes": ["MOCK-001", "MOCK-003"],
            "top_driver_category": DriverCategory.STATISTICAL,
        }

        warning = generate_rtts_warning(signal)

        assert warning is not None
        # Verify normal form: "RTTS {status}: {count} violations | driver={category} | flags=[{codes}]"
        assert warning == "RTTS WARN: 3 violations | driver=STATISTICAL | flags=[MOCK-001, MOCK-003]"

    def test_warning_for_critical_status_normal_form(self):
        """Test that warning is generated for CRITICAL status in normal form."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "CRITICAL",
            "violation_count": 10,
            "top3_mock_codes": ["MOCK-001", "MOCK-009"],
            "top_driver_category": DriverCategory.CONTINUITY,
        }

        warning = generate_rtts_warning(signal)

        assert warning is not None
        assert warning == "RTTS CRITICAL: 10 violations | driver=CONTINUITY | flags=[MOCK-001, MOCK-009]"

    def test_warning_no_flags_shows_none(self):
        """Test that warning shows 'none' when no flags present."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "WARN",
            "violation_count": 1,
            "top3_mock_codes": [],
            "top_driver_category": DriverCategory.UNKNOWN,
        }

        warning = generate_rtts_warning(signal)

        assert warning is not None
        assert warning == "RTTS WARN: 1 violations | driver=UNKNOWN | flags=[none]"

    def test_no_warning_for_ok_status(self):
        """Test that no warning is generated for OK status."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "OK",
            "violation_count": 0,
            "top3_mock_codes": [],
            "top_driver_category": DriverCategory.UNKNOWN,
        }

        warning = generate_rtts_warning(signal)

        assert warning is None

    def test_no_warning_for_attention_status(self):
        """Test that no warning is generated for ATTENTION status."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "ATTENTION",
            "violation_count": 1,
            "top3_mock_codes": ["MOCK-001"],
            "top_driver_category": DriverCategory.STATISTICAL,
        }

        warning = generate_rtts_warning(signal)

        assert warning is None

    def test_no_warning_when_unavailable(self):
        """Test that no warning is generated when RTTS unavailable."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": False,
            "overall_status": "CRITICAL",  # Even with CRITICAL, no warning if unavailable
            "violation_count": 10,
            "top3_mock_codes": [],
            "top_driver_category": DriverCategory.UNKNOWN,
        }

        warning = generate_rtts_warning(signal)

        assert warning is None

    def test_warning_determinism(self):
        """Test that warning string is deterministic across multiple calls."""
        from backend.health.rtts_status_adapter import generate_rtts_warning, DriverCategory

        signal = {
            "available": True,
            "overall_status": "WARN",
            "violation_count": 5,
            "top3_mock_codes": ["MOCK-003", "MOCK-001", "MOCK-005"],  # Pre-sorted
            "top_driver_category": DriverCategory.STATISTICAL,
        }

        # Generate warning 10 times
        warnings = [generate_rtts_warning(signal) for _ in range(10)]

        # All warnings should be identical
        expected = "RTTS WARN: 5 violations | driver=STATISTICAL | flags=[MOCK-003, MOCK-001, MOCK-005]"
        for warning in warnings:
            assert warning == expected


# =============================================================================
# Tests for load_rtts_validation_for_status
# =============================================================================

class TestLoadRTTSValidationForStatus:
    """Tests for load_rtts_validation_for_status function."""

    def test_load_from_run_dir(self):
        """Test loading rtts_validation.json from run directory."""
        from backend.health.rtts_status_adapter import load_rtts_validation_for_status

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create rtts_validation.json
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "OK",
                "mock_detection_flags": [],
            }
            with open(run_dir / "rtts_validation.json", "w") as f:
                json.dump(rtts_data, f)

            result = load_rtts_validation_for_status(run_dir)

            assert result is not None
            assert result["overall_status"] == "OK"

    def test_load_from_p4_shadow_subdir(self):
        """Test loading rtts_validation.json from p4_shadow subdirectory."""
        from backend.health.rtts_status_adapter import load_rtts_validation_for_status

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Create rtts_validation.json in p4_shadow
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "WARN",
                "mock_detection_flags": ["MOCK-001"],
            }
            with open(p4_shadow / "rtts_validation.json", "w") as f:
                json.dump(rtts_data, f)

            result = load_rtts_validation_for_status(run_dir)

            assert result is not None
            assert result["overall_status"] == "WARN"

    def test_load_returns_none_when_missing(self):
        """Test that load returns None when file is missing."""
        from backend.health.rtts_status_adapter import load_rtts_validation_for_status

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            result = load_rtts_validation_for_status(run_dir)

            assert result is None


# =============================================================================
# Tests for manifest reference loading with integrity check
# =============================================================================

class TestManifestReferenceLoading:
    """Tests for load_rtts_validation_from_manifest_reference function."""

    def test_load_from_manifest_reference_valid(self):
        """Test loading rtts_validation.json via manifest reference with valid sha256."""
        from backend.health.rtts_status_adapter import (
            load_rtts_validation_from_manifest_reference,
            compute_rtts_file_sha256,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            governance_dir = evidence_pack_dir / "governance"
            governance_dir.mkdir()

            # Create rtts_validation.json
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "WARN",
                "warning_count": 3,
                "mock_detection_flags": ["MOCK-001", "MOCK-003"],
            }
            rtts_path = governance_dir / "rtts_validation.json"
            with open(rtts_path, "w") as f:
                json.dump(rtts_data, f)

            # Compute actual sha256
            actual_sha256 = compute_rtts_file_sha256(rtts_path)

            # Create manifest reference
            manifest_reference = {
                "path": "governance/rtts_validation.json",
                "sha256": actual_sha256,
            }

            result = load_rtts_validation_from_manifest_reference(
                manifest_reference, evidence_pack_dir
            )

            assert result is not None
            assert result["overall_status"] == "WARN"
            assert result["mock_detection_flags"] == ["MOCK-001", "MOCK-003"]

    def test_load_from_manifest_reference_integrity_fail(self):
        """Test that integrity failure returns None (SHADOW MODE: no error)."""
        from backend.health.rtts_status_adapter import (
            load_rtts_validation_from_manifest_reference,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            governance_dir = evidence_pack_dir / "governance"
            governance_dir.mkdir()

            # Create rtts_validation.json
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "CRITICAL",
                "warning_count": 5,
            }
            rtts_path = governance_dir / "rtts_validation.json"
            with open(rtts_path, "w") as f:
                json.dump(rtts_data, f)

            # Create manifest reference with WRONG sha256
            manifest_reference = {
                "path": "governance/rtts_validation.json",
                "sha256": "0" * 64,  # Invalid hash
            }

            result = load_rtts_validation_from_manifest_reference(
                manifest_reference, evidence_pack_dir
            )

            # SHADOW MODE: returns None on integrity failure, no error raised
            assert result is None

    def test_load_from_manifest_reference_file_missing(self):
        """Test that missing file returns None."""
        from backend.health.rtts_status_adapter import (
            load_rtts_validation_from_manifest_reference,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)

            manifest_reference = {
                "path": "governance/rtts_validation.json",
                "sha256": "abc123",
            }

            result = load_rtts_validation_from_manifest_reference(
                manifest_reference, evidence_pack_dir
            )

            assert result is None

    def test_load_from_manifest_reference_no_path(self):
        """Test that manifest reference without path returns None."""
        from backend.health.rtts_status_adapter import (
            load_rtts_validation_from_manifest_reference,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)

            # Manifest reference missing path key
            manifest_reference = {
                "sha256": "abc123",
            }

            result = load_rtts_validation_from_manifest_reference(
                manifest_reference, evidence_pack_dir
            )

            assert result is None

    def test_load_from_manifest_reference_no_sha256_skips_check(self):
        """Test that missing sha256 skips integrity check (accepts file)."""
        from backend.health.rtts_status_adapter import (
            load_rtts_validation_from_manifest_reference,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            governance_dir = evidence_pack_dir / "governance"
            governance_dir.mkdir()

            # Create rtts_validation.json
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "OK",
            }
            rtts_path = governance_dir / "rtts_validation.json"
            with open(rtts_path, "w") as f:
                json.dump(rtts_data, f)

            # Manifest reference without sha256
            manifest_reference = {
                "path": "governance/rtts_validation.json",
            }

            result = load_rtts_validation_from_manifest_reference(
                manifest_reference, evidence_pack_dir
            )

            # Should load successfully without integrity check
            assert result is not None
            assert result["overall_status"] == "OK"


# =============================================================================
# Tests for extract_rtts_status_for_first_light (full pipeline)
# =============================================================================

class TestExtractRTTSStatusForFirstLight:
    """Tests for full extraction pipeline with manifest reference support and extraction_source."""

    def test_prefers_manifest_reference_with_extraction_source(self):
        """Test that manifest reference is preferred and extraction_source is set correctly."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_for_first_light,
            compute_rtts_file_sha256,
            ExtractionSource,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            governance_dir = evidence_pack_dir / "governance"
            governance_dir.mkdir()

            # Create rtts_validation.json in governance (via manifest)
            rtts_manifest_data = {
                "schema_version": "1.0.0",
                "overall_status": "WARN",  # Manifest version says WARN
                "warning_count": 3,
                "mock_detection_flags": ["MOCK-001"],
            }
            rtts_manifest_path = governance_dir / "rtts_validation.json"
            with open(rtts_manifest_path, "w") as f:
                json.dump(rtts_manifest_data, f)

            # Also create rtts_validation.json in run_dir (direct discovery)
            run_dir = Path(tmpdir) / "run_dir"
            run_dir.mkdir()
            rtts_direct_data = {
                "schema_version": "1.0.0",
                "overall_status": "CRITICAL",  # Direct version says CRITICAL
                "warning_count": 10,
                "mock_detection_flags": ["MOCK-009"],
            }
            with open(run_dir / "rtts_validation.json", "w") as f:
                json.dump(rtts_direct_data, f)

            # Compute sha256 for manifest reference
            actual_sha256 = compute_rtts_file_sha256(rtts_manifest_path)
            manifest_reference = {
                "path": "governance/rtts_validation.json",
                "sha256": actual_sha256,
            }

            # Extract with both manifest reference and run_dir
            signal = extract_rtts_status_for_first_light(
                run_dir=run_dir,
                manifest_reference=manifest_reference,
                evidence_pack_dir=evidence_pack_dir,
            )

            # Should use manifest reference (WARN), not direct discovery (CRITICAL)
            assert signal["overall_status"] == "WARN"
            assert signal["top3_mock_codes"] == ["MOCK-001"]
            # Verify extraction_source provenance
            assert signal["extraction_source"] == ExtractionSource.MANIFEST_REFERENCE

    def test_falls_back_to_direct_discovery_with_extraction_source(self):
        """Test fallback to direct discovery with correct extraction_source."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_for_first_light,
            ExtractionSource,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create rtts_validation.json in run_dir only
            rtts_data = {
                "schema_version": "1.0.0",
                "overall_status": "CRITICAL",
                "warning_count": 5,
                "mock_detection_flags": ["MOCK-009", "MOCK-010"],
            }
            with open(run_dir / "rtts_validation.json", "w") as f:
                json.dump(rtts_data, f)

            # No manifest reference
            signal = extract_rtts_status_for_first_light(run_dir=run_dir)

            # Should use direct discovery
            assert signal["available"] is True
            assert signal["overall_status"] == "CRITICAL"
            # Verify extraction_source provenance
            assert signal["extraction_source"] == ExtractionSource.DIRECT_DISCOVERY

    def test_returns_missing_extraction_source_when_no_file(self):
        """Test signal shows MISSING extraction_source when no rtts_validation.json exists."""
        from backend.health.rtts_status_adapter import (
            extract_rtts_status_for_first_light,
            ExtractionSource,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # No rtts_validation.json anywhere
            signal = extract_rtts_status_for_first_light(run_dir=run_dir)

            assert signal["available"] is False
            assert signal["overall_status"] == "UNKNOWN"
            # Verify extraction_source provenance
            assert signal["extraction_source"] == ExtractionSource.MISSING

    def test_extraction_source_enum_values(self):
        """Test that ExtractionSource enum values are frozen strings."""
        from backend.health.rtts_status_adapter import ExtractionSource

        # Verify enum values are uppercase strings
        assert ExtractionSource.MANIFEST_REFERENCE == "MANIFEST_REFERENCE"
        assert ExtractionSource.DIRECT_DISCOVERY == "DIRECT_DISCOVERY"
        assert ExtractionSource.MISSING == "MISSING"


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
