"""
Tests for Audit Plane v0 schema validation.

SHADOW-OBSERVE: These tests verify schema structure; no authority; no gating.
"""

import json
from pathlib import Path

import pytest

# Import from backend
from backend.audit.audit_root import (
    SCHEMA_VERSION,
    load_schema,
    validate_event,
)


class TestSchemaStructure:
    """Test the schema file structure and required fields."""

    def test_schema_file_exists(self):
        """Schema file exists at documented path."""
        schema_path = Path("schemas/audit/audit_event.schema.json")
        assert schema_path.exists(), f"Schema not found: {schema_path}"

    def test_schema_is_valid_json(self):
        """Schema file is valid JSON."""
        schema = load_schema()
        assert isinstance(schema, dict)

    def test_schema_has_required_top_level_fields(self):
        """Schema has $schema, title, type, required, properties."""
        schema = load_schema()
        assert "$schema" in schema
        assert "title" in schema
        assert schema.get("type") == "object"
        assert "required" in schema
        assert "properties" in schema

    def test_schema_version_is_1_0_0(self):
        """Schema enforces version 1.0.0."""
        schema = load_schema()
        version_spec = schema["properties"]["schema_version"]
        assert version_spec.get("const") == "1.0.0"

    def test_required_fields_list(self):
        """Schema requires exactly 8 fields."""
        schema = load_schema()
        required = schema.get("required", [])
        expected = [
            "schema_version",
            "event_id",
            "event_type",
            "subject",
            "digest",
            "timestamp",
            "severity",
            "source",
        ]
        assert set(required) == set(expected)

    def test_event_type_enum_values(self):
        """event_type enum has expected values."""
        schema = load_schema()
        event_types = schema["properties"]["event_type"]["enum"]
        expected = [
            "FS_TOUCH",
            "CMD_RUN",
            "HASH_EMITTED",
            "TEST_RESULT",
            "FUZZ_FINDING",
            "POLICY_CHECK",
            "OTHER",
        ]
        assert set(event_types) == set(expected)

    def test_severity_enum_values(self):
        """severity enum has INFO and WARN."""
        schema = load_schema()
        severities = schema["properties"]["severity"]["enum"]
        assert set(severities) == {"INFO", "WARN"}

    def test_digest_alg_is_sha256(self):
        """digest.alg must be sha256."""
        schema = load_schema()
        alg = schema["properties"]["digest"]["properties"]["alg"]
        assert alg.get("const") == "sha256"


class TestEventValidation:
    """Test event validation against schema."""

    @pytest.fixture
    def valid_event(self):
        """A minimal valid audit event."""
        return {
            "schema_version": "1.0.0",
            "event_id": "a" * 64,
            "event_type": "TEST_RESULT",
            "subject": {"kind": "TEST", "ref": "tests/example.py"},
            "digest": {"alg": "sha256", "hex": "b" * 64},
            "timestamp": "2025-12-18T12:00:00Z",
            "severity": "INFO",
            "source": "audit_plane_v0",
        }

    def test_valid_event_passes(self, valid_event):
        """Valid event passes validation."""
        is_valid, errors = validate_event(valid_event)
        assert is_valid, f"Expected valid, got errors: {errors}"

    def test_missing_required_field_fails(self, valid_event):
        """Missing required field fails validation."""
        del valid_event["event_type"]
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("event_type" in e for e in errors)

    def test_wrong_schema_version_fails(self, valid_event):
        """Wrong schema_version fails validation."""
        valid_event["schema_version"] = "2.0.0"
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("schema_version" in e for e in errors)

    def test_invalid_event_type_fails(self, valid_event):
        """Invalid event_type fails validation."""
        valid_event["event_type"] = "INVALID_TYPE"
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("event_type" in e for e in errors)

    def test_invalid_severity_fails(self, valid_event):
        """Invalid severity fails validation."""
        valid_event["severity"] = "ERROR"
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("severity" in e for e in errors)

    def test_invalid_digest_alg_fails(self, valid_event):
        """digest.alg != sha256 fails validation."""
        valid_event["digest"]["alg"] = "md5"
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("digest.alg" in e for e in errors)

    def test_invalid_digest_hex_length_fails(self, valid_event):
        """digest.hex != 64 chars fails validation."""
        valid_event["digest"]["hex"] = "abc123"
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("digest.hex" in e for e in errors)

    def test_missing_subject_kind_fails(self, valid_event):
        """Missing subject.kind fails validation."""
        del valid_event["subject"]["kind"]
        is_valid, errors = validate_event(valid_event)
        assert not is_valid
        assert any("subject" in e for e in errors)

    def test_all_event_types_valid(self, valid_event):
        """All defined event_types pass validation."""
        event_types = [
            "FS_TOUCH",
            "CMD_RUN",
            "HASH_EMITTED",
            "TEST_RESULT",
            "FUZZ_FINDING",
            "POLICY_CHECK",
            "OTHER",
        ]
        for etype in event_types:
            valid_event["event_type"] = etype
            is_valid, errors = validate_event(valid_event)
            assert is_valid, f"{etype} should be valid: {errors}"

    def test_all_subject_kinds_valid(self, valid_event):
        """Common subject kinds pass validation."""
        # Note: subject.kind enum is in schema but not enforced by minimal validator
        kinds = ["FILE", "COMMAND", "HASH", "TEST", "POLICY", "AGENT", "OTHER"]
        for kind in kinds:
            valid_event["subject"]["kind"] = kind
            is_valid, errors = validate_event(valid_event)
            assert is_valid, f"subject.kind={kind} should be valid: {errors}"
