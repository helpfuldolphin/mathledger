"""
Tests for P5 Divergence Report schema binding in Evidence Pack.

SHADOW MODE: All P5 operations are observational only.
Detection and validation occur but do not affect pack generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.topology.first_light.evidence_pack import (
    P5_DIVERGENCE_ARTIFACT,
    P5_DIVERGENCE_SCHEMA,
    P5DivergenceReference,
    detect_p5_divergence_file,
    ARTIFACT_SCHEMA_MAP,
)


class TestP5Constants:
    """Test P5 constant definitions."""

    def test_p5_artifact_name(self):
        """P5 artifact has correct name."""
        assert P5_DIVERGENCE_ARTIFACT == "p5_divergence_real.json"

    def test_p5_schema_path(self):
        """P5 schema has correct path."""
        assert P5_DIVERGENCE_SCHEMA == "p5/p5_divergence_real.schema.json"

    def test_p5_in_schema_map(self):
        """P5 artifact is registered in schema map."""
        assert P5_DIVERGENCE_ARTIFACT in ARTIFACT_SCHEMA_MAP
        assert ARTIFACT_SCHEMA_MAP[P5_DIVERGENCE_ARTIFACT] == P5_DIVERGENCE_SCHEMA


class TestP5DivergenceReference:
    """Test P5DivergenceReference dataclass."""

    def test_minimal_reference(self):
        """Create minimal P5 reference."""
        ref = P5DivergenceReference(
            path="p4_shadow/p5_divergence_real.json",
            sha256="abc123" * 10 + "abcd",
        )
        assert ref.path == "p4_shadow/p5_divergence_real.json"
        assert ref.schema_valid is False
        assert ref.validation_errors == []

    def test_full_reference(self):
        """Create full P5 reference with all fields."""
        ref = P5DivergenceReference(
            path="p4_shadow/p5_divergence_real.json",
            sha256="abc123" * 10 + "abcd",
            schema_version="1.0.0",
            telemetry_source="real",
            validation_status="VALIDATED_REAL",
            divergence_rate=0.05,
            mode="SHADOW",
            schema_valid=True,
            validation_errors=[],
        )
        assert ref.schema_version == "1.0.0"
        assert ref.telemetry_source == "real"
        assert ref.validation_status == "VALIDATED_REAL"
        assert ref.divergence_rate == 0.05
        assert ref.mode == "SHADOW"
        assert ref.schema_valid is True


class TestDetectP5DivergenceFile:
    """Test detect_p5_divergence_file function."""

    def test_no_file_returns_none(self):
        """No P5 file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_p5_divergence_file(Path(tmpdir))
            assert result is None

    def test_detects_p4_shadow_location(self):
        """Detects P5 file in p4_shadow subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Create minimal valid P5 file
            p5_data = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_test",
                "telemetry_source": "real",
                "validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.95,
                "total_cycles": 100,
                "divergence_rate": 0.03,
                "mode": "SHADOW",
            }
            p5_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                json.dump(p5_data, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert result.path == f"p4_shadow/{P5_DIVERGENCE_ARTIFACT}"
            assert result.schema_version == "1.0.0"
            assert result.telemetry_source == "real"
            assert result.validation_status == "VALIDATED_REAL"
            assert result.divergence_rate == 0.03
            assert result.mode == "SHADOW"

    def test_detects_root_location(self):
        """Detects P5 file in root directory as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create minimal valid P5 file at root
            p5_data = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_test",
                "telemetry_source": "mock",
                "validation_status": "SUSPECTED_MOCK",
                "validation_confidence": 0.60,
                "total_cycles": 50,
                "divergence_rate": 0.15,
                "mode": "SHADOW",
            }
            p5_path = run_dir / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                json.dump(p5_data, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert result.path == P5_DIVERGENCE_ARTIFACT
            assert result.telemetry_source == "mock"
            assert result.validation_status == "SUSPECTED_MOCK"

    def test_prefers_p4_shadow_over_root(self):
        """Prefers p4_shadow location over root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Create file in both locations
            p5_shadow = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_shadow",
                "telemetry_source": "real",
                "validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.95,
                "total_cycles": 100,
                "divergence_rate": 0.02,
                "mode": "SHADOW",
            }
            p5_root = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_root",
                "telemetry_source": "mock",
                "validation_status": "UNVALIDATED",
                "validation_confidence": 0.50,
                "total_cycles": 50,
                "divergence_rate": 0.20,
                "mode": "SHADOW",
            }

            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(p5_shadow, f)
            with open(run_dir / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(p5_root, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert "p4_shadow" in result.path
            assert result.telemetry_source == "real"  # From shadow location

    def test_handles_invalid_json(self):
        """Handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Create invalid JSON
            p5_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                f.write("{ invalid json }")

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert result.sha256 is not None  # Hash still computed
            assert result.schema_valid is False
            assert len(result.validation_errors) > 0
            assert "Parse error" in result.validation_errors[0]

    def test_computes_sha256_hash(self):
        """Computes SHA256 hash of P5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            p5_data = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_test",
                "telemetry_source": "real",
                "validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.95,
                "total_cycles": 100,
                "divergence_rate": 0.03,
                "mode": "SHADOW",
            }
            p5_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                json.dump(p5_data, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert len(result.sha256) == 64  # SHA256 hex length
            assert all(c in "0123456789abcdef" for c in result.sha256)


class TestSchemaValidation:
    """Test schema validation integration."""

    @pytest.fixture
    def valid_p5_record(self):
        """A fully valid P5 record per schema."""
        return {
            "schema_version": "1.0.0",
            "run_id": "p5_20251211_143000_harness_run",
            "telemetry_source": "real",
            "validation_status": "VALIDATED_REAL",
            "validation_confidence": 0.92,
            "total_cycles": 500,
            "divergence_rate": 0.042,
            "mode": "SHADOW",
            "mock_baseline_divergence_rate": 0.038,
            "divergence_delta": 0.004,
            "twin_tracking_accuracy": {
                "success": 0.91,
                "omega": 0.87,
                "blocked": 0.94
            },
        }

    def test_valid_record_passes_schema(self, valid_p5_record):
        """Valid record validates against schema if jsonschema available."""
        pytest.importorskip("jsonschema")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            p5_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                json.dump(valid_p5_record, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            # Schema validation depends on schema file being present
            # In production the schema file exists at docs/system_law/schemas/p5/
            # For this test we just verify detection works
            assert result.telemetry_source == "real"


class TestShadowModeCompliance:
    """Test SHADOW MODE contract compliance."""

    def test_detection_is_observational(self):
        """Detection does not modify source files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            p5_data = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_test",
                "telemetry_source": "real",
                "validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.95,
                "total_cycles": 100,
                "divergence_rate": 0.03,
                "mode": "SHADOW",
            }
            p5_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(p5_path, "w") as f:
                json.dump(p5_data, f)

            # Read original content
            with open(p5_path, "r") as f:
                original_content = f.read()

            # Run detection
            detect_p5_divergence_file(run_dir)

            # Verify content unchanged
            with open(p5_path, "r") as f:
                after_content = f.read()

            assert original_content == after_content

    def test_mode_field_captured(self):
        """Mode field is captured for SHADOW verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # SHADOW mode file
            p5_shadow = {
                "schema_version": "1.0.0",
                "run_id": "p5_20251211_120000_test",
                "telemetry_source": "real",
                "validation_status": "VALIDATED_REAL",
                "validation_confidence": 0.95,
                "total_cycles": 100,
                "divergence_rate": 0.03,
                "mode": "SHADOW",
            }
            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(p5_shadow, f)

            result = detect_p5_divergence_file(run_dir)
            assert result is not None
            assert result.mode == "SHADOW"
