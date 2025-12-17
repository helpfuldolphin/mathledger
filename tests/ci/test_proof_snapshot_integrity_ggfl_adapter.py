"""
Tests for proof snapshot integrity GGFL adapter.

SHADOW MODE: These tests verify that proof snapshot integrity objects are
converted into deterministic GGFL alignment-view signals.
"""

import unittest

from backend.health.proof_snapshot_integrity_adapter import (
    PROOF_SNAPSHOT_INTEGRITY_FAILURE_PRIORITY,
    PROOF_SNAPSHOT_INTEGRITY_SIGNAL_TYPE,
    proof_snapshot_integrity_for_alignment_view,
)


class TestProofSnapshotIntegrityGGFLAdapter(unittest.TestCase):
    def test_returns_none_when_integrity_missing(self) -> None:
        assert proof_snapshot_integrity_for_alignment_view(None) is None

    def test_drivers_are_codes_only(self) -> None:
        canonical_codes = set(PROOF_SNAPSHOT_INTEGRITY_FAILURE_PRIORITY.keys())
        integrity = {
            "ok": False,
            "failure_codes": ["NOT_A_CODE", "CANONICAL_HASH_MISMATCH", "SHA256_MISMATCH"],
        }

        view = proof_snapshot_integrity_for_alignment_view(integrity)

        assert view is not None
        assert view["signal_type"] == PROOF_SNAPSHOT_INTEGRITY_SIGNAL_TYPE
        assert view["conflict"] is False
        assert view["weight_hint"] == "LOW"
        assert isinstance(view["drivers"], list)
        assert all(driver in canonical_codes for driver in view["drivers"])
        assert len(view["drivers"]) <= 3

    def test_deterministic_ordering_top_failure_code_first(self) -> None:
        integrity_a = {
            "ok": False,
            "failure_codes": [
                "ENTRY_COUNT_MISMATCH",
                "CANONICAL_HASH_MISMATCH",
                "SHA256_MISMATCH",
                "MISSING_FILE",
            ],
        }
        integrity_b = {
            "ok": False,
            "failure_codes": [
                "MISSING_FILE",
                "SHA256_MISMATCH",
                "CANONICAL_HASH_MISMATCH",
                "ENTRY_COUNT_MISMATCH",
                "MISSING_FILE",
            ],
        }

        view_a = proof_snapshot_integrity_for_alignment_view(integrity_a)
        view_b = proof_snapshot_integrity_for_alignment_view(integrity_b)

        assert view_a == view_b
        assert view_a is not None
        assert view_a["drivers"] == [
            "MISSING_FILE",
            "SHA256_MISMATCH",
            "CANONICAL_HASH_MISMATCH",
        ]

