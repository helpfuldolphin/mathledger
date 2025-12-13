"""
Tests for User Verified Input Events

Tests the event recording and attestation integration for UVI.
"""

import pytest
from backend.uvi.events import (
    UVIEvent,
    UVIEventType,
    record_confirmation,
    record_correction,
    record_flag,
    get_events_for_target,
    get_all_events,
    clear_events,
)
from backend.uvi.attestation import (
    compute_uvi_digest,
    compute_batch_digest,
    build_uvi_attestation_leaf,
    build_uvi_summary_leaf,
    verify_uvi_leaf,
)


@pytest.fixture(autouse=True)
def clean_events():
    """Clear events before and after each test."""
    clear_events()
    yield
    clear_events()


class TestUVIEventRecording:
    """Tests for UVI event recording."""

    def test_record_confirmation(self):
        """Confirmation events are recorded correctly."""
        event = record_confirmation(
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            rationale="Verified manually",
        )

        assert event.event_type == UVIEventType.CONFIRMATION
        assert event.target_hash == "abc123"
        assert event.target_type == "statement"
        assert event.user_id == "user_001"
        assert event.rationale == "Verified manually"
        assert event.event_id.startswith("uvi_")

    def test_record_correction(self):
        """Correction events require rationale."""
        event = record_correction(
            target_hash="def456",
            target_type="proof",
            user_id="user_002",
            rationale="Proof step 3 is invalid",
        )

        assert event.event_type == UVIEventType.CORRECTION
        assert event.rationale == "Proof step 3 is invalid"

    def test_record_flag(self):
        """Flag events can be recorded without rationale."""
        event = record_flag(
            target_hash="ghi789",
            target_type="derivation",
            user_id="user_003",
        )

        assert event.event_type == UVIEventType.FLAG
        assert event.rationale is None

    def test_events_are_immutable(self):
        """UVI events cannot be modified after creation."""
        event = record_confirmation(
            target_hash="test123",
            target_type="statement",
            user_id="user_001",
        )

        with pytest.raises(AttributeError):
            event.target_hash = "modified"

    def test_get_events_for_target(self):
        """Events can be retrieved by target hash."""
        record_confirmation("target_a", "statement", "user_001")
        record_correction("target_a", "statement", "user_002", "Issue found")
        record_flag("target_b", "statement", "user_001")

        events_a = get_events_for_target("target_a")
        events_b = get_events_for_target("target_b")

        assert len(events_a) == 2
        assert len(events_b) == 1

    def test_get_all_events(self):
        """All events can be retrieved."""
        record_confirmation("t1", "statement", "u1")
        record_correction("t2", "proof", "u2", "Issue")
        record_flag("t3", "derivation", "u3")

        all_events = get_all_events()
        assert len(all_events) == 3


class TestUVIEventDigest:
    """Tests for UVI event digest computation."""

    def test_digest_is_deterministic(self):
        """Same event produces same digest."""
        event1 = UVIEvent(
            event_id="test_001",
            event_type=UVIEventType.CONFIRMATION,
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            timestamp="2025-12-13T00:00:00Z",
        )
        event2 = UVIEvent(
            event_id="test_001",
            event_type=UVIEventType.CONFIRMATION,
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            timestamp="2025-12-13T00:00:00Z",
        )

        assert compute_uvi_digest(event1) == compute_uvi_digest(event2)

    def test_digest_changes_with_content(self):
        """Different events produce different digests."""
        event1 = UVIEvent(
            event_id="test_001",
            event_type=UVIEventType.CONFIRMATION,
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            timestamp="2025-12-13T00:00:00Z",
        )
        event2 = UVIEvent(
            event_id="test_002",  # Different ID
            event_type=UVIEventType.CONFIRMATION,
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            timestamp="2025-12-13T00:00:00Z",
        )

        assert compute_uvi_digest(event1) != compute_uvi_digest(event2)

    def test_batch_digest_is_ordered(self):
        """Batch digest is deterministic regardless of input order."""
        event1 = UVIEvent(
            event_id="aaa",
            event_type=UVIEventType.CONFIRMATION,
            target_hash="t1",
            target_type="statement",
            user_id="u1",
            timestamp="2025-12-13T00:00:00Z",
        )
        event2 = UVIEvent(
            event_id="bbb",
            event_type=UVIEventType.CORRECTION,
            target_hash="t2",
            target_type="proof",
            user_id="u2",
            timestamp="2025-12-13T00:01:00Z",
            rationale="Issue",
        )

        # Order shouldn't matter
        digest1 = compute_batch_digest([event1, event2])
        digest2 = compute_batch_digest([event2, event1])

        assert digest1 == digest2

    def test_empty_batch_has_sentinel(self):
        """Empty batch produces a defined sentinel digest."""
        digest = compute_batch_digest([])
        assert len(digest) == 64  # SHA-256 hex


class TestUVIAttestation:
    """Tests for UVI attestation leaf generation."""

    def test_attestation_leaf_structure(self):
        """Attestation leaf has correct structure."""
        event = record_confirmation(
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
        )

        leaf = build_uvi_attestation_leaf(event)

        assert leaf["schema_version"] == "1.0.0"
        assert leaf["leaf_type"] == "uvi_event"
        assert leaf["event_id"] == event.event_id
        assert leaf["event_type"] == "CONFIRMATION"
        assert leaf["target_hash"] == "abc123"
        assert leaf["digest"] == compute_uvi_digest(event)
        assert len(leaf["user_id_hash"]) == 16  # Anonymized

    def test_summary_leaf_counts(self):
        """Summary leaf has correct event counts."""
        record_confirmation("t1", "statement", "u1")
        record_confirmation("t2", "statement", "u2")
        record_correction("t3", "proof", "u3", "Issue")
        record_flag("t4", "derivation", "u4")

        events = get_all_events()
        summary = build_uvi_summary_leaf(events)

        assert summary["event_count"] == 4
        assert summary["type_counts"]["CONFIRMATION"] == 2
        assert summary["type_counts"]["CORRECTION"] == 1
        assert summary["type_counts"]["FLAG"] == 1
        assert summary["unique_target_count"] == 4
        assert summary["mode"] == "SHADOW"

    def test_verify_uvi_leaf(self):
        """Attestation leaf can be verified against original event."""
        event = record_confirmation(
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
        )

        leaf = build_uvi_attestation_leaf(event)

        assert verify_uvi_leaf(leaf, event) is True

    def test_verify_detects_tampering(self):
        """Verification fails for tampered leaves."""
        event = record_confirmation(
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
        )

        leaf = build_uvi_attestation_leaf(event)
        leaf["digest"] = "tampered_digest"

        assert verify_uvi_leaf(leaf, event) is False


class TestUVIShadowMode:
    """Tests for UVI Shadow Mode compliance."""

    def test_no_enforcement(self):
        """UVI events do not enforce any behavior."""
        # Record a correction - this should NOT block anything
        event = record_correction(
            target_hash="abc123",
            target_type="statement",
            user_id="user_001",
            rationale="This looks wrong",
        )

        # The event is recorded but has no enforcement effect
        # This test documents the Shadow Mode contract
        assert event.event_type == UVIEventType.CORRECTION
        # No exception raised, no blocking behavior

    def test_summary_marks_shadow_mode(self):
        """Summary leaf explicitly marks Shadow Mode."""
        record_confirmation("t1", "statement", "u1")
        events = get_all_events()
        summary = build_uvi_summary_leaf(events)

        assert summary["mode"] == "SHADOW"
