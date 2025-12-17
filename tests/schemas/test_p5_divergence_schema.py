"""
Tests for P5 Real Telemetry Divergence Schema

Tests validate:
- Minimal field requirements
- Recommended field structures
- Optional diagnostic fields
- Example record validation
- Edge cases and invalid data rejection

SHADOW MODE CONTRACT:
All tests verify schema structure for observational artifacts only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

# Schema validation - required for these tests
jsonschema = pytest.importorskip("jsonschema")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def schema() -> Dict[str, Any]:
    """Load the P5 divergence schema."""
    schema_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "system_law"
        / "schemas"
        / "p5"
        / "p5_divergence_real.schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def minimal_record() -> Dict[str, Any]:
    """Minimal valid P5 divergence record with only required fields."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p5_20251215_143022_test",
        "telemetry_source": "real",
        "validation_status": "VALIDATED_REAL",
        "validation_confidence": 0.92,
        "total_cycles": 1000,
        "divergence_rate": 0.23,
        "mode": "SHADOW",
    }


@pytest.fixture
def full_example_record() -> Dict[str, Any]:
    """Full example P5 divergence record with all fields."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p5_20251215_143022_real_arithmetic",
        "telemetry_source": "real",
        "validation_status": "VALIDATED_REAL",
        "validation_confidence": 0.92,
        "total_cycles": 1000,
        "divergence_rate": 0.23,
        "mode": "SHADOW",
        "mock_baseline_divergence_rate": 0.97,
        "divergence_delta": -0.74,
        "twin_tracking_accuracy": {
            "success": 0.91,
            "omega": 0.88,
            "blocked": 0.85,
        },
        "manifold_validation": {
            "boundedness_ok": True,
            "continuity_ok": True,
            "correlation_ok": True,
            "violations": [],
        },
        "tda_comparison": {
            "sns_delta": 0.02,
            "pcs_delta": -0.01,
            "drs_delta": 0.05,
            "hss_delta": 0.03,
        },
        "warm_start_calibration": {
            "calibration_cycles": 50,
            "initial_divergence": 0.65,
            "final_divergence": 0.18,
            "convergence_achieved": True,
        },
        "divergence_decomposition": {
            "bias": 0.03,
            "variance": 0.02,
            "timing": 0.08,
            "structural": 0.10,
        },
        "pattern_classification": "NOMINAL",
        "pattern_confidence": 0.87,
        "mock_detection_flags": [],
        "noise_envelope": {
            "sigma_H": 0.012,
            "sigma_rho": 0.008,
            "autocorr_lag1": 0.32,
            "kurtosis": 0.45,
        },
        "governance_signals": {
            "sig_top_status": "OK",
            "sig_rpl_status": "OK",
            "sig_tel_status": "OK",
        },
        "fusion_advisory": {
            "recommendation": "ALLOW",
            "conflict_detected": False,
        },
        "recalibration_recommendations": [],
        "timing": {
            "start_time": "2025-12-15T14:30:22Z",
            "end_time": "2025-12-15T14:35:47Z",
            "duration_seconds": 325.4,
        },
    }


# =============================================================================
# Test: Schema Loading
# =============================================================================

class TestSchemaLoading:
    """Tests for schema structure and loading."""

    def test_schema_loads_successfully(self, schema: Dict[str, Any]) -> None:
        """Schema file loads as valid JSON."""
        assert schema is not None
        assert "$schema" in schema
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_schema_has_required_metadata(self, schema: Dict[str, Any]) -> None:
        """Schema has required metadata fields."""
        # v1.1.0: Added true_divergence_vector_v1 with explicit metric fields
        assert schema["$id"] == "https://mathledger.org/schemas/phase_x_p5/p5_divergence_real.v1.1.0.json"
        assert schema["title"] == "P5 Real Telemetry Divergence Report"
        assert "description" in schema

    def test_schema_defines_required_fields(self, schema: Dict[str, Any]) -> None:
        """Schema defines all minimal required fields."""
        required = schema["required"]
        expected_required = [
            "schema_version",
            "run_id",
            "telemetry_source",
            "validation_status",
            "validation_confidence",
            "total_cycles",
            "divergence_rate",
            "mode",
        ]
        assert set(required) == set(expected_required)


# =============================================================================
# Test: Minimal Field Validation
# =============================================================================

class TestMinimalFields:
    """Tests for minimal required field validation."""

    def test_minimal_record_validates(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Minimal record with only required fields validates."""
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_missing_required_field_fails(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Missing required field causes validation failure."""
        for field in ["schema_version", "run_id", "mode", "divergence_rate"]:
            invalid_record = {k: v for k, v in minimal_record.items() if k != field}
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(instance=invalid_record, schema=schema)

    def test_schema_version_must_be_1_0_0(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Schema version must be exactly '1.0.0'."""
        minimal_record["schema_version"] = "2.0.0"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_mode_must_be_shadow(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Mode must be exactly 'SHADOW'."""
        minimal_record["mode"] = "PRODUCTION"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_run_id_pattern_validation(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Run ID must match p5_YYYYMMDD_HHMMSS pattern."""
        # Valid patterns
        for valid_id in [
            "p5_20251215_143022_test",
            "p5_20251215_000000_real",
            "p5_20251215_235959_mock_arithmetic",
        ]:
            minimal_record["run_id"] = valid_id
            jsonschema.validate(instance=minimal_record, schema=schema)

        # Invalid patterns
        for invalid_id in ["p4_20251215_143022", "run_123", "p5-20251215-143022"]:
            minimal_record["run_id"] = invalid_id
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(instance=minimal_record, schema=schema)

    def test_telemetry_source_enum(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Telemetry source must be 'real' or 'mock'."""
        for valid in ["real", "mock"]:
            minimal_record["telemetry_source"] = valid
            jsonschema.validate(instance=minimal_record, schema=schema)

        minimal_record["telemetry_source"] = "synthetic"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_validation_status_enum(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Validation status must be valid enum."""
        for valid in ["VALIDATED_REAL", "SUSPECTED_MOCK", "UNVALIDATED"]:
            minimal_record["validation_status"] = valid
            jsonschema.validate(instance=minimal_record, schema=schema)

        minimal_record["validation_status"] = "INVALID_STATUS"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_numeric_bounds(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Numeric fields respect bounds."""
        # validation_confidence out of range
        minimal_record["validation_confidence"] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

        minimal_record["validation_confidence"] = -0.1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

        # divergence_rate out of range
        minimal_record["validation_confidence"] = 0.9
        minimal_record["divergence_rate"] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

        # total_cycles must be positive
        minimal_record["divergence_rate"] = 0.5
        minimal_record["total_cycles"] = 0
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)


# =============================================================================
# Test: Recommended Fields
# =============================================================================

class TestRecommendedFields:
    """Tests for recommended field structures."""

    def test_twin_tracking_accuracy_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Twin tracking accuracy object validates correctly."""
        minimal_record["twin_tracking_accuracy"] = {
            "success": 0.91,
            "omega": 0.88,
            "blocked": 0.85,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_twin_tracking_accuracy_bounds(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Twin tracking accuracy values must be in [0,1]."""
        minimal_record["twin_tracking_accuracy"] = {
            "success": 1.5,  # Invalid
            "omega": 0.88,
            "blocked": 0.85,
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_manifold_validation_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Manifold validation object validates correctly."""
        minimal_record["manifold_validation"] = {
            "boundedness_ok": True,
            "continuity_ok": True,
            "correlation_ok": False,
            "violations": ["V4_COR_STRUCTURE"],
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_manifold_violation_pattern(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Manifold violations must match RTTS pattern."""
        # Valid patterns
        minimal_record["manifold_validation"] = {
            "boundedness_ok": True,
            "continuity_ok": True,
            "correlation_ok": True,
            "violations": ["V1_BOUND_H", "V3_JUMP_RHO", "MOCK-001", "MOCK-010"],
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

        # Invalid pattern
        minimal_record["manifold_validation"]["violations"] = ["INVALID_CODE"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_tda_comparison_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """TDA comparison object validates correctly."""
        minimal_record["tda_comparison"] = {
            "sns_delta": 0.02,
            "pcs_delta": -0.01,
            "drs_delta": 0.05,
            "hss_delta": 0.03,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_warm_start_calibration_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Warm start calibration object validates correctly."""
        minimal_record["warm_start_calibration"] = {
            "calibration_cycles": 50,
            "initial_divergence": 0.65,
            "final_divergence": 0.18,
            "convergence_achieved": True,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)


# =============================================================================
# Test: Optional Diagnostic Fields
# =============================================================================

class TestOptionalDiagnostics:
    """Tests for optional diagnostic field structures."""

    def test_divergence_decomposition_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Divergence decomposition validates per RTTS Section 3.2."""
        minimal_record["divergence_decomposition"] = {
            "bias": 0.03,
            "variance": 0.02,
            "timing": 0.08,
            "structural": 0.10,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_pattern_classification_enum(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Pattern classification must be valid RTTS pattern."""
        valid_patterns = [
            "DRIFT",
            "NOISE_AMPLIFICATION",
            "PHASE_LAG",
            "ATTRACTOR_MISS",
            "TRANSIENT_MISS",
            "STRUCTURAL_BREAK",
            "NOMINAL",
        ]
        for pattern in valid_patterns:
            minimal_record["pattern_classification"] = pattern
            jsonschema.validate(instance=minimal_record, schema=schema)

        minimal_record["pattern_classification"] = "INVALID_PATTERN"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_mock_detection_flags_pattern(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Mock detection flags must match MOCK-NNN pattern."""
        minimal_record["mock_detection_flags"] = ["MOCK-001", "MOCK-009", "MOCK-010"]
        jsonschema.validate(instance=minimal_record, schema=schema)

        minimal_record["mock_detection_flags"] = ["MOCK-1"]  # Invalid
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_noise_envelope_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Noise envelope validates per RTTS Section 1.3."""
        minimal_record["noise_envelope"] = {
            "sigma_H": 0.012,
            "sigma_rho": 0.008,
            "autocorr_lag1": 0.32,
            "kurtosis": 0.45,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_fusion_advisory_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Fusion advisory validates per GGFL spec."""
        for recommendation in ["ALLOW", "WARN", "BLOCK"]:
            minimal_record["fusion_advisory"] = {
                "recommendation": recommendation,
                "conflict_detected": False,
            }
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_recalibration_recommendations_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Recalibration recommendations validate correctly."""
        minimal_record["recalibration_recommendations"] = [
            {
                "parameter": "tau_base",
                "current_value": 0.20,
                "suggested_value": 0.25,
                "rationale": "Reduce noise sensitivity",
            },
            {
                "parameter": "smoothing_window",
                "rationale": "Increase to reduce phase lag",
            },
        ]
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_timing_structure(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Timing metadata validates correctly."""
        minimal_record["timing"] = {
            "start_time": "2025-12-15T14:30:22Z",
            "end_time": "2025-12-15T14:35:47Z",
            "duration_seconds": 325.4,
        }
        jsonschema.validate(instance=minimal_record, schema=schema)


# =============================================================================
# Test: Full Example Record
# =============================================================================

class TestFullExampleRecord:
    """Tests for complete example record validation."""

    def test_full_example_validates(
        self, schema: Dict[str, Any], full_example_record: Dict[str, Any]
    ) -> None:
        """Full example record from schema sketch validates cleanly."""
        jsonschema.validate(instance=full_example_record, schema=schema)

    def test_full_example_has_all_field_categories(
        self, full_example_record: Dict[str, Any]
    ) -> None:
        """Full example includes minimal, recommended, and optional fields."""
        # Minimal fields
        assert "schema_version" in full_example_record
        assert "run_id" in full_example_record
        assert "mode" in full_example_record

        # Recommended fields
        assert "twin_tracking_accuracy" in full_example_record
        assert "manifold_validation" in full_example_record
        assert "tda_comparison" in full_example_record
        assert "warm_start_calibration" in full_example_record

        # Optional diagnostics
        assert "divergence_decomposition" in full_example_record
        assert "pattern_classification" in full_example_record
        assert "noise_envelope" in full_example_record
        assert "fusion_advisory" in full_example_record

    def test_full_example_shadow_mode_compliance(
        self, full_example_record: Dict[str, Any]
    ) -> None:
        """Full example complies with SHADOW MODE contract."""
        assert full_example_record["mode"] == "SHADOW"
        assert full_example_record["fusion_advisory"]["recommendation"] in [
            "ALLOW",
            "WARN",
            "BLOCK",
        ]


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_arrays_valid(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Empty arrays are valid for array fields."""
        minimal_record["mock_detection_flags"] = []
        minimal_record["recalibration_recommendations"] = []
        minimal_record["manifold_validation"] = {
            "boundedness_ok": True,
            "continuity_ok": True,
            "correlation_ok": True,
            "violations": [],
        }
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_boundary_values(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Boundary values for numeric fields validate."""
        # Exact boundaries
        minimal_record["validation_confidence"] = 0.0
        minimal_record["divergence_rate"] = 1.0
        minimal_record["total_cycles"] = 1
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_additional_properties_rejected(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Unknown top-level properties are rejected."""
        minimal_record["unknown_field"] = "value"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)

    def test_governance_signals_allows_additional(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """Governance signals allows additional signal types."""
        minimal_record["governance_signals"] = {
            "sig_top_status": "OK",
            "sig_custom_status": "WARN",  # Additional allowed
        }
        jsonschema.validate(instance=minimal_record, schema=schema)


# =============================================================================
# Test: Smoke Test Checklist Items
# =============================================================================

class TestSmokeTestChecklist:
    """Tests corresponding to Smoke-Test Readiness Checklist."""

    def test_gen_03_valid_json(
        self, schema: Dict[str, Any], full_example_record: Dict[str, Any]
    ) -> None:
        """GEN-03: File is valid JSON (parse succeeds)."""
        # Re-serialize and parse to verify JSON validity
        json_str = json.dumps(full_example_record)
        parsed = json.loads(json_str)
        jsonschema.validate(instance=parsed, schema=schema)

    def test_min_01_schema_version(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """MIN-01: schema_version equals '1.0.0'."""
        assert minimal_record["schema_version"] == "1.0.0"
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_min_08_mode_shadow(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """MIN-08: mode equals 'SHADOW'."""
        assert minimal_record["mode"] == "SHADOW"
        jsonschema.validate(instance=minimal_record, schema=schema)

    def test_shd_01_shadow_mode_enforced(
        self, schema: Dict[str, Any], minimal_record: Dict[str, Any]
    ) -> None:
        """SHD-01: mode field is 'SHADOW' (enforced by schema)."""
        minimal_record["mode"] = "ACTIVE"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=minimal_record, schema=schema)
