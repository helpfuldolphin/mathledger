"""
Tests for Audit Plane v0 event_id determinism.

SHADOW-OBSERVE: These tests verify deterministic hashing; no authority; no gating.

Key properties:
1. Same event content => same event_id (regardless of field order)
2. Different content => different event_id
3. timestamp and event_id excluded from canonicalization
"""

import pytest

from backend.audit.audit_root import (
    canonicalize_event,
    compute_event_id,
)


class TestEventIdDeterminism:
    """Test that event_id computation is deterministic."""

    @pytest.fixture
    def base_event(self):
        """Base event for testing."""
        return {
            "schema_version": "1.0.0",
            "event_type": "TEST_RESULT",
            "subject": {"kind": "TEST", "ref": "tests/example.py"},
            "digest": {"alg": "sha256", "hex": "a" * 64},
            "timestamp": "2025-12-18T12:00:00Z",
            "severity": "INFO",
            "source": "audit_plane_v0",
        }

    def test_same_event_same_id(self, base_event):
        """Identical events produce identical event_ids."""
        id1 = compute_event_id(base_event)
        id2 = compute_event_id(base_event)
        assert id1 == id2

    def test_field_order_irrelevant(self, base_event):
        """Field ordering does not affect event_id."""
        # Create event with different field order
        reordered = {
            "source": base_event["source"],
            "severity": base_event["severity"],
            "timestamp": base_event["timestamp"],
            "digest": base_event["digest"],
            "subject": base_event["subject"],
            "event_type": base_event["event_type"],
            "schema_version": base_event["schema_version"],
        }

        id_original = compute_event_id(base_event)
        id_reordered = compute_event_id(reordered)
        assert id_original == id_reordered

    def test_timestamp_excluded(self, base_event):
        """Different timestamps produce same event_id."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event1["timestamp"] = "2025-12-18T12:00:00Z"
        event2["timestamp"] = "2025-12-18T23:59:59Z"

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 == id2

    def test_event_id_excluded(self, base_event):
        """Existing event_id field is excluded from canonicalization."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event1["event_id"] = "x" * 64
        event2["event_id"] = "y" * 64

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 == id2

    def test_different_content_different_id(self, base_event):
        """Different event content produces different event_id."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event2["event_type"] = "CMD_RUN"

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 != id2

    def test_different_digest_different_id(self, base_event):
        """Different digest.hex produces different event_id."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event2["digest"] = {"alg": "sha256", "hex": "b" * 64}

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 != id2

    def test_different_subject_different_id(self, base_event):
        """Different subject produces different event_id."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event2["subject"] = {"kind": "FILE", "ref": "/tmp/other.txt"}

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 != id2

    def test_event_id_is_64_hex_chars(self, base_event):
        """event_id is exactly 64 lowercase hex characters."""
        event_id = compute_event_id(base_event)
        assert len(event_id) == 64
        assert all(c in "0123456789abcdef" for c in event_id)

    def test_meta_note_excluded(self, base_event):
        """meta.note is excluded from canonicalization."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event1["meta"] = {"note": "First note"}
        event2["meta"] = {"note": "Different note"}

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 == id2

    def test_meta_other_fields_included(self, base_event):
        """meta fields other than note are included in canonicalization."""
        event1 = base_event.copy()
        event2 = base_event.copy()

        event1["meta"] = {"exit_code": 0}
        event2["meta"] = {"exit_code": 1}

        id1 = compute_event_id(event1)
        id2 = compute_event_id(event2)
        assert id1 != id2


class TestCanonicalization:
    """Test canonicalization produces stable bytes."""

    def test_canonical_is_bytes(self):
        """canonicalize_event returns bytes."""
        event = {
            "schema_version": "1.0.0",
            "event_type": "OTHER",
            "subject": {"kind": "OTHER", "ref": "x"},
            "digest": {"alg": "sha256", "hex": "a" * 64},
            "severity": "INFO",
            "source": "test",
        }
        result = canonicalize_event(event)
        assert isinstance(result, bytes)

    def test_canonical_is_compact_json(self):
        """Canonical form has no whitespace."""
        event = {
            "schema_version": "1.0.0",
            "event_type": "OTHER",
            "subject": {"kind": "OTHER", "ref": "x"},
            "digest": {"alg": "sha256", "hex": "a" * 64},
            "severity": "INFO",
            "source": "test",
        }
        result = canonicalize_event(event)
        text = result.decode("utf-8")
        assert " " not in text
        assert "\n" not in text

    def test_canonical_keys_sorted(self):
        """Canonical form has keys in alphabetical order."""
        event = {
            "source": "test",
            "severity": "INFO",
            "schema_version": "1.0.0",
            "event_type": "OTHER",
            "subject": {"ref": "x", "kind": "OTHER"},
            "digest": {"hex": "a" * 64, "alg": "sha256"},
        }
        result = canonicalize_event(event)
        text = result.decode("utf-8")

        # Keys should appear in alphabetical order
        import json
        parsed = json.loads(text)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

        # Nested keys too
        subject_keys = list(parsed["subject"].keys())
        assert subject_keys == sorted(subject_keys)
