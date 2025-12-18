"""
Tests for Audit Plane v0 audit root (A_t) determinism.

SHADOW-OBSERVE: These tests verify A_t computation; no authority; no gating.

Key properties:
1. Same set of events => same A_t (regardless of input order)
2. Different events => different A_t
3. Empty event list produces well-defined A_t
"""

import hashlib

import pytest

from backend.audit.audit_root import (
    compute_audit_root,
    compute_event_id,
    generate_audit_root_artifact,
)


class TestAuditRootDeterminism:
    """Test that A_t computation is deterministic."""

    @pytest.fixture
    def sample_event_ids(self):
        """Sample event_ids for testing."""
        return [
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ]

    def test_same_ids_same_root(self, sample_event_ids):
        """Identical event_ids produce identical A_t."""
        root1 = compute_audit_root(sample_event_ids)
        root2 = compute_audit_root(sample_event_ids.copy())
        assert root1 == root2

    def test_order_irrelevant(self, sample_event_ids):
        """Input order does not affect A_t (sorting is internal)."""
        forward = sample_event_ids.copy()
        reverse = list(reversed(sample_event_ids))
        shuffled = [sample_event_ids[1], sample_event_ids[2], sample_event_ids[0]]

        root_forward = compute_audit_root(forward)
        root_reverse = compute_audit_root(reverse)
        root_shuffled = compute_audit_root(shuffled)

        assert root_forward == root_reverse
        assert root_forward == root_shuffled

    def test_different_ids_different_root(self, sample_event_ids):
        """Different event_ids produce different A_t."""
        ids1 = sample_event_ids.copy()
        ids2 = ["d" * 64, "e" * 64, "f" * 64]

        root1 = compute_audit_root(ids1)
        root2 = compute_audit_root(ids2)
        assert root1 != root2

    def test_subset_different_root(self, sample_event_ids):
        """Subset of events produces different A_t."""
        full = sample_event_ids.copy()
        subset = sample_event_ids[:2]

        root_full = compute_audit_root(full)
        root_subset = compute_audit_root(subset)
        assert root_full != root_subset

    def test_empty_list_defined(self):
        """Empty event list produces well-defined A_t."""
        root = compute_audit_root([])
        # SHA-256 of empty string
        expected = hashlib.sha256(b"").hexdigest()
        assert root == expected

    def test_single_event(self):
        """Single event produces valid A_t."""
        root = compute_audit_root(["a" * 64])
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)

    def test_root_is_64_hex_chars(self, sample_event_ids):
        """A_t is exactly 64 lowercase hex characters."""
        root = compute_audit_root(sample_event_ids)
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)

    def test_merkle_tree_structure(self):
        """Verify Merkle tree structure for 4 leaves."""
        # For 4 leaves, structure is:
        #       root
        #      /    \
        #   h01      h23
        #   / \      / \
        #  a   b    c   d

        ids = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        sorted_ids = sorted(ids)  # Already sorted

        # Manual computation
        a = bytes.fromhex(sorted_ids[0])
        b = bytes.fromhex(sorted_ids[1])
        c = bytes.fromhex(sorted_ids[2])
        d = bytes.fromhex(sorted_ids[3])

        h01 = hashlib.sha256(a + b).digest()
        h23 = hashlib.sha256(c + d).digest()
        expected_root = hashlib.sha256(h01 + h23).hexdigest()

        computed_root = compute_audit_root(ids)
        assert computed_root == expected_root


class TestAuditRootArtifact:
    """Test audit_root.json artifact generation."""

    @pytest.fixture
    def sample_events(self):
        """Sample events for artifact generation."""
        return [
            {
                "schema_version": "1.0.0",
                "event_type": "TEST_RESULT",
                "subject": {"kind": "TEST", "ref": "test1.py"},
                "digest": {"alg": "sha256", "hex": "a" * 64},
                "timestamp": "2025-12-18T12:00:00Z",
                "severity": "INFO",
                "source": "audit_plane_v0",
            },
            {
                "schema_version": "1.0.0",
                "event_type": "TEST_RESULT",
                "subject": {"kind": "TEST", "ref": "test2.py"},
                "digest": {"alg": "sha256", "hex": "b" * 64},
                "timestamp": "2025-12-18T12:00:01Z",
                "severity": "INFO",
                "source": "audit_plane_v0",
            },
        ]

    def test_artifact_has_required_fields(self, sample_events):
        """Artifact contains schema_version, event_count, audit_root, inputs."""
        artifact = generate_audit_root_artifact(sample_events)
        assert "schema_version" in artifact
        assert "event_count" in artifact
        assert "audit_root" in artifact
        assert "inputs" in artifact

    def test_artifact_event_count(self, sample_events):
        """event_count matches number of events."""
        artifact = generate_audit_root_artifact(sample_events)
        assert artifact["event_count"] == len(sample_events)

    def test_artifact_inputs_sorted(self, sample_events):
        """inputs are sorted lexicographically."""
        artifact = generate_audit_root_artifact(sample_events)
        inputs = artifact["inputs"]
        assert inputs == sorted(inputs)

    def test_artifact_deterministic(self, sample_events):
        """Same events produce identical artifact."""
        artifact1 = generate_audit_root_artifact(sample_events)
        artifact2 = generate_audit_root_artifact(sample_events)
        assert artifact1 == artifact2

    def test_artifact_uses_computed_ids(self, sample_events):
        """Artifact uses computed event_ids when not present."""
        # Events don't have event_id field
        artifact = generate_audit_root_artifact(sample_events)

        # Manually compute expected ids
        expected_ids = sorted([compute_event_id(e) for e in sample_events])
        assert artifact["inputs"] == expected_ids

    def test_artifact_uses_existing_ids(self, sample_events):
        """Artifact uses existing event_id if present and valid."""
        events_with_ids = []
        for e in sample_events:
            e_copy = e.copy()
            e_copy["event_id"] = compute_event_id(e)
            events_with_ids.append(e_copy)

        artifact = generate_audit_root_artifact(events_with_ids)

        expected_ids = sorted([e["event_id"] for e in events_with_ids])
        assert artifact["inputs"] == expected_ids


class TestAuditRootConsistency:
    """Test consistency between event_id and A_t computation."""

    def test_events_to_root_deterministic(self):
        """Full pipeline from events to A_t is deterministic."""
        events = [
            {
                "schema_version": "1.0.0",
                "event_type": "CMD_RUN",
                "subject": {"kind": "COMMAND", "ref": "make test"},
                "digest": {"alg": "sha256", "hex": "1" * 64},
                "timestamp": "2025-12-18T12:00:00Z",
                "severity": "INFO",
                "source": "audit_plane_v0",
            },
            {
                "schema_version": "1.0.0",
                "event_type": "FS_TOUCH",
                "subject": {"kind": "FILE", "ref": "/tmp/output.txt"},
                "digest": {"alg": "sha256", "hex": "2" * 64},
                "timestamp": "2025-12-18T12:00:01Z",
                "severity": "WARN",
                "source": "audit_plane_v0",
            },
        ]

        # Run multiple times
        roots = [generate_audit_root_artifact(events)["audit_root"] for _ in range(5)]

        # All should be identical
        assert all(r == roots[0] for r in roots)
