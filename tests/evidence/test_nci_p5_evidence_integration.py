"""
Tests for NCI P5 (Narrative Consistency Index) integration into Evidence Pack.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- Detection does not gate pack generation
- Status signals are advisory only

Tests cover:
1. Detection in root and calibration/ directories
2. Status signal determinism and non-gating behavior
3. Signal file vs result file priority
4. Cross-reference of both files when present
5. Graceful handling of malformed JSON
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.topology.first_light.evidence_pack import (
    NciP5Reference,
    NCI_P5_RESULT_ARTIFACT,
    NCI_P5_SIGNAL_ARTIFACT,
    NCI_P5_SUBDIR,
    compute_file_hash,
    detect_nci_p5_artifacts,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_nci_signal() -> dict:
    """Sample NCI P5 signal file content."""
    return {
        "schema_version": "1.0.0",
        "signal_type": "SIG-NAR",
        "mode": "DOC_ONLY",
        "global_nci": 0.85,
        "confidence": 0.75,
        "slo_status": "OK",
        "recommendation": "NONE",
        "tcl_aligned": True,
        "sic_aligned": True,
        "tcl_violation_count": 0,
        "sic_violation_count": 0,
        "warning_count": 2,
        "shadow_mode": True,
        "timestamp": "2025-12-12T10:00:00Z",
    }


@pytest.fixture
def sample_nci_result() -> dict:
    """Sample NCI P5 result file content (nested structure)."""
    return {
        "schema_version": "1.0.0",
        "mode": "TELEMETRY_CHECKED",
        "global_nci": 0.78,
        "confidence": 0.65,
        "slo_evaluation": {
            "status": "WARN",
            "threshold_warn": 0.80,
            "threshold_breach": 0.70,
        },
        "governance_signal": {
            "signal_type": "SIG-NAR",
            "recommendation": "WARNING",
            "shadow_mode": True,
        },
        "tcl_result": {
            "aligned": False,
            "violations": [
                {"doc": "docs/api.md", "field": "H", "violation_type": "TCL-002"},
            ],
        },
        "sic_result": {
            "aligned": True,
            "violations": [],
        },
        "warnings": [
            {"warning_type": "LOW_CONFIDENCE", "message": "Low sample size"},
        ],
        "shadow_mode": True,
    }


@pytest.fixture
def run_dir_with_signal(tmp_path: Path, sample_nci_signal: dict) -> Path:
    """Create a run directory with NCI P5 signal file in root."""
    signal_path = tmp_path / NCI_P5_SIGNAL_ARTIFACT
    signal_path.write_text(json.dumps(sample_nci_signal), encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_dir_with_result(tmp_path: Path, sample_nci_result: dict) -> Path:
    """Create a run directory with NCI P5 result file in root."""
    result_path = tmp_path / NCI_P5_RESULT_ARTIFACT
    result_path.write_text(json.dumps(sample_nci_result), encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_dir_with_both(tmp_path: Path, sample_nci_signal: dict, sample_nci_result: dict) -> Path:
    """Create a run directory with both signal and result files."""
    signal_path = tmp_path / NCI_P5_SIGNAL_ARTIFACT
    signal_path.write_text(json.dumps(sample_nci_signal), encoding="utf-8")
    result_path = tmp_path / NCI_P5_RESULT_ARTIFACT
    result_path.write_text(json.dumps(sample_nci_result), encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_dir_calibration_subdir(tmp_path: Path, sample_nci_signal: dict) -> Path:
    """Create a run directory with NCI P5 signal in calibration/ subdirectory."""
    cal_dir = tmp_path / NCI_P5_SUBDIR
    cal_dir.mkdir()
    signal_path = cal_dir / NCI_P5_SIGNAL_ARTIFACT
    signal_path.write_text(json.dumps(sample_nci_signal), encoding="utf-8")
    return tmp_path


# =============================================================================
# Test 1: Detection returns None for empty directory
# =============================================================================


class TestNciP5Detection:
    """Tests for NCI P5 artifact detection."""

    def test_detection_returns_none_for_empty_dir(self, tmp_path: Path):
        """Detection returns None when no NCI artifacts present."""
        ref = detect_nci_p5_artifacts(tmp_path)
        assert ref is None

    # =============================================================================
    # Test 2: Detection finds signal file in root
    # =============================================================================

    def test_detection_finds_signal_in_root(
        self, run_dir_with_signal: Path, sample_nci_signal: dict
    ):
        """Detection finds nci_p5_signal.json in root directory."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        assert ref.path == NCI_P5_SIGNAL_ARTIFACT
        assert ref.mode == sample_nci_signal["mode"]
        assert ref.global_nci == sample_nci_signal["global_nci"]
        assert ref.confidence == sample_nci_signal["confidence"]
        assert ref.slo_status == sample_nci_signal["slo_status"]
        assert ref.recommendation == sample_nci_signal["recommendation"]
        assert ref.tcl_aligned is True
        assert ref.sic_aligned is True
        assert ref.shadow_mode is True

    # =============================================================================
    # Test 3: Detection finds signal in calibration/ subdirectory
    # =============================================================================

    def test_detection_finds_signal_in_calibration_subdir(
        self, run_dir_calibration_subdir: Path, sample_nci_signal: dict
    ):
        """Detection finds nci_p5_signal.json in calibration/ subdirectory."""
        ref = detect_nci_p5_artifacts(run_dir_calibration_subdir)

        assert ref is not None
        assert ref.path == f"{NCI_P5_SUBDIR}/{NCI_P5_SIGNAL_ARTIFACT}"
        assert ref.mode == sample_nci_signal["mode"]
        assert ref.slo_status == sample_nci_signal["slo_status"]

    # =============================================================================
    # Test 4: Detection finds result file when signal absent
    # =============================================================================

    def test_detection_uses_result_when_no_signal(
        self, run_dir_with_result: Path, sample_nci_result: dict
    ):
        """Detection falls back to nci_p5_result.json when signal absent."""
        ref = detect_nci_p5_artifacts(run_dir_with_result)

        assert ref is not None
        assert ref.path == NCI_P5_RESULT_ARTIFACT
        assert ref.mode == sample_nci_result["mode"]
        assert ref.slo_status == sample_nci_result["slo_evaluation"]["status"]
        # Result file has nested TCL/SIC structure
        assert ref.tcl_aligned is False
        assert ref.tcl_violation_count == 1
        assert ref.sic_aligned is True

    # =============================================================================
    # Test 5: Detection prefers signal file over result file
    # =============================================================================

    def test_detection_prefers_signal_over_result(
        self, run_dir_with_both: Path, sample_nci_signal: dict, sample_nci_result: dict
    ):
        """Detection prefers signal file when both are present."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        # Should use signal file values (DOC_ONLY mode, not TELEMETRY_CHECKED)
        assert ref.path == NCI_P5_SIGNAL_ARTIFACT
        assert ref.mode == sample_nci_signal["mode"]  # DOC_ONLY
        assert ref.mode != sample_nci_result["mode"]  # Not TELEMETRY_CHECKED

    # =============================================================================
    # Test 6: Cross-reference result file when both present
    # =============================================================================

    def test_cross_reference_result_when_both_present(
        self, run_dir_with_both: Path
    ):
        """Detection includes result file cross-reference when both exist."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        assert ref.result_path == NCI_P5_RESULT_ARTIFACT
        assert ref.result_sha256 is not None
        # Verify result sha256 is valid
        result_path = run_dir_with_both / NCI_P5_RESULT_ARTIFACT
        expected_hash = compute_file_hash(result_path)
        assert ref.result_sha256 == expected_hash

    # =============================================================================
    # Test 7: Detection handles malformed JSON gracefully
    # =============================================================================

    def test_detection_handles_malformed_json(self, tmp_path: Path):
        """Detection returns reference with sha256 even for malformed JSON."""
        signal_path = tmp_path / NCI_P5_SIGNAL_ARTIFACT
        signal_path.write_text("{ invalid json }", encoding="utf-8")

        ref = detect_nci_p5_artifacts(tmp_path)

        # Should still return a reference (with sha256)
        assert ref is not None
        assert ref.path == NCI_P5_SIGNAL_ARTIFACT
        assert ref.sha256 is not None
        # Fields should have defaults when JSON can't be parsed
        assert ref.shadow_mode is True

    # =============================================================================
    # Test 8: SHA256 hash is computed correctly
    # =============================================================================

    def test_sha256_hash_computed_correctly(
        self, run_dir_with_signal: Path
    ):
        """SHA256 hash matches actual file hash."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        signal_path = run_dir_with_signal / NCI_P5_SIGNAL_ARTIFACT
        expected_hash = compute_file_hash(signal_path)
        assert ref.sha256 == expected_hash


# =============================================================================
# Test 9: Status signal is deterministic (same input -> same output)
# =============================================================================


class TestNciP5StatusSignalDeterminism:
    """Tests for deterministic status signal behavior."""

    def test_status_signal_is_deterministic(
        self, run_dir_with_signal: Path
    ):
        """Multiple detections produce identical results."""
        ref1 = detect_nci_p5_artifacts(run_dir_with_signal)
        ref2 = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref1 is not None
        assert ref2 is not None
        assert ref1.sha256 == ref2.sha256
        assert ref1.mode == ref2.mode
        assert ref1.global_nci == ref2.global_nci
        assert ref1.slo_status == ref2.slo_status

    # =============================================================================
    # Test 10: Non-gating behavior (detection never raises)
    # =============================================================================

    def test_detection_never_raises_on_any_content(self, tmp_path: Path):
        """Detection is non-gating - handles any file content gracefully."""
        test_cases = [
            "",  # Empty file
            "not json at all",  # Plain text
            "{}",  # Empty JSON object
            '{"partial": true',  # Truncated JSON
            "null",  # JSON null
            "[]",  # JSON array
        ]

        for content in test_cases:
            signal_path = tmp_path / NCI_P5_SIGNAL_ARTIFACT
            signal_path.write_text(content, encoding="utf-8")

            # Should never raise - non-gating behavior
            try:
                ref = detect_nci_p5_artifacts(tmp_path)
                # Either None or a valid reference
                assert ref is None or isinstance(ref, NciP5Reference)
            except Exception as e:
                pytest.fail(f"Detection raised for content '{content[:20]}...': {e}")

            # Clean up for next iteration
            signal_path.unlink()


# =============================================================================
# Test 11: Detection path is set correctly
# =============================================================================


class TestNciP5DetectionPath:
    """Tests for detection_path field in NciP5Reference."""

    def test_detection_path_root(self, run_dir_with_signal: Path):
        """Detection path is 'root' when artifact found in root directory."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        assert ref.detection_path == "root"

    def test_detection_path_calibration(self, run_dir_calibration_subdir: Path):
        """Detection path is 'calibration' when artifact found in calibration/ subdirectory."""
        ref = detect_nci_p5_artifacts(run_dir_calibration_subdir)

        assert ref is not None
        assert ref.detection_path == "calibration"

    def test_detection_path_deterministic_root(self, run_dir_with_signal: Path):
        """Detection path is deterministic for root location."""
        ref1 = detect_nci_p5_artifacts(run_dir_with_signal)
        ref2 = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref1 is not None
        assert ref2 is not None
        assert ref1.detection_path == ref2.detection_path == "root"

    def test_detection_path_deterministic_calibration(self, run_dir_calibration_subdir: Path):
        """Detection path is deterministic for calibration location."""
        ref1 = detect_nci_p5_artifacts(run_dir_calibration_subdir)
        ref2 = detect_nci_p5_artifacts(run_dir_calibration_subdir)

        assert ref1 is not None
        assert ref2 is not None
        assert ref1.detection_path == ref2.detection_path == "calibration"


# =============================================================================
# Test 12: Reference completeness (signal_sha256, result_sha256)
# =============================================================================


class TestNciP5ReferenceCompleteness:
    """Tests for reference completeness with both artifacts."""

    def test_signal_sha256_always_present(self, run_dir_with_signal: Path):
        """signal sha256 is always present in reference."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        assert ref.sha256 is not None
        assert len(ref.sha256) == 64  # SHA256 hex string length

    def test_result_sha256_when_both_present(self, run_dir_with_both: Path):
        """result_sha256 is present when both signal and result exist."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        assert ref.result_sha256 is not None
        assert len(ref.result_sha256) == 64

    def test_result_sha256_none_when_only_signal(self, run_dir_with_signal: Path):
        """result_sha256 is None when only signal file exists."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        assert ref.result_sha256 is None

    def test_result_path_when_both_present(self, run_dir_with_both: Path):
        """result_path is set when both artifacts exist."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        assert ref.result_path == NCI_P5_RESULT_ARTIFACT

    def test_reference_has_all_expected_fields(self, run_dir_with_both: Path):
        """Reference has all expected fields for manifest inclusion."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        # Required fields
        assert ref.path is not None
        assert ref.sha256 is not None
        assert ref.detection_path in ("root", "calibration")
        assert ref.extraction_source in ("MANIFEST_SIGNAL", "MANIFEST_RESULT", "EVIDENCE_JSON", "MISSING")
        assert ref.schema_version is not None
        # Optional cross-reference
        assert ref.result_path is not None
        assert ref.result_sha256 is not None


# =============================================================================
# Test 13: Extraction source is set correctly
# =============================================================================


class TestNciP5ExtractionSource:
    """Tests for extraction_source field in NciP5Reference."""

    def test_extraction_source_manifest_signal(self, run_dir_with_signal: Path):
        """Extraction source is MANIFEST_SIGNAL when signal file is found."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        assert ref.extraction_source == "MANIFEST_SIGNAL"

    def test_extraction_source_manifest_result(self, run_dir_with_result: Path):
        """Extraction source is MANIFEST_RESULT when only result file is found."""
        ref = detect_nci_p5_artifacts(run_dir_with_result)

        assert ref is not None
        assert ref.extraction_source == "MANIFEST_RESULT"

    def test_extraction_source_prefers_signal(self, run_dir_with_both: Path):
        """Extraction source is MANIFEST_SIGNAL when both files present."""
        ref = detect_nci_p5_artifacts(run_dir_with_both)

        assert ref is not None
        assert ref.extraction_source == "MANIFEST_SIGNAL"

    def test_extraction_source_deterministic(self, run_dir_with_signal: Path):
        """Extraction source is deterministic across multiple detections."""
        ref1 = detect_nci_p5_artifacts(run_dir_with_signal)
        ref2 = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref1 is not None
        assert ref2 is not None
        assert ref1.extraction_source == ref2.extraction_source == "MANIFEST_SIGNAL"

    def test_extraction_source_valid_values_only(self, run_dir_with_signal: Path):
        """Extraction source is always a valid enum value."""
        ref = detect_nci_p5_artifacts(run_dir_with_signal)

        assert ref is not None
        valid_sources = ["MANIFEST_SIGNAL", "MANIFEST_RESULT", "EVIDENCE_JSON", "MISSING"]
        assert ref.extraction_source in valid_sources
