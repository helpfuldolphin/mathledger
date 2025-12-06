# tests/test_ht_invariant.py
"""
HT Invariant Tests — Dual-Attestation Hash Verification

STATUS: PHASE II — REFERENCE TESTS ONLY
========================================
These tests validate a PROPOSED Hₜ specification (HT_INVARIANT_SPEC_v1.md).
They are NOT part of the Phase I Evidence Pack and do NOT test production code.

Current Phase I ground truth:
- backend/attestation/dual_root.py — actual attestation logic
- fo_baseline/attestation.json, fo_rfl/attestation.json — sealed artifacts

RFL and Hₜ are independent systems:
- RFL logs (abstention rates, 1000-cycle Dyno Charts) do NOT affect Hₜ
- Hₜ derives solely from First Organism dual-root attestation (attestation.json)
- No Phase I RFL run triggers or validates these Hₜ invariant tests

These tests:
- Use a self-contained reference implementation (not production imports)
- Are gated behind @pytest.mark.ht_spec to exclude from default CI
- Must NOT be cited as evidence of "formally verified Hₜ invariants"

To run these tests explicitly:
    pytest -m ht_spec tests/test_ht_invariant.py

Tests for the invariants specified in HT_INVARIANT_SPEC_v1.md:
- INV-CANON-1: Deterministic canonicalization
- INV-DOMAIN-1: Domain-separated hashing
- INV-ORDER-1: Sorted leaf order
- INV-CASE-1: Stable casing (lowercase hex)
- INV-WS-1: No whitespace in canonical JSON
- INV-VERIFY-1: Verifiable recomputation
"""

import hashlib
import json
import pytest
from typing import Any

# Gate all tests in this module behind ht_spec marker
pytestmark = pytest.mark.ht_spec

# Domain separation prefixes (must match HT_INVARIANT_SPEC_v1.md)
DOMAIN_PROOF = b"MathLedger:Proof:v1:"
DOMAIN_UI = b"MathLedger:UIEvent:v1:"
DOMAIN_HT = b"MathLedger:DualAttest:v1:"
DOMAIN_MERKLE_NODE = b"MathLedger:MerkleNode:v1:"


# ============================================================================
# Phase II: Future Hₜ-Adjacent Monitoring Design Notes
# ============================================================================
#
# GUARDRAIL: Hₜ is the "hard provenance invariant." Any RFL-related monitoring
# is optional observation that CANNOT flip a test from pass to fail without
# an explicit new spec and review.
#
# If future uplift experiments warrant Hₜ-adjacent monitoring, here is how
# it could work WITHOUT violating the Hₜ/RFL independence:
#
# 1. LOGGING ONLY (no test semantics):
#    A future test helper could log RFL metadata present in attestation.json
#    (e.g., epoch_id, run_type) purely for correlation analysis. Example:
#
#        def log_attestation_context(attestation: dict) -> None:
#            """Log RFL context if present. Does NOT affect test pass/fail."""
#            if "epoch_metadata" in attestation:
#                logger.info(f"Epoch context: {attestation['epoch_metadata']}")
#            # This is informational only. Hₜ tests ignore this data.
#
# 2. ΔHₜ OBSERVATION (post-hoc, not invariant):
#    If epoch-over-epoch Hₜ values are available, a monitoring script (NOT a
#    test) could compute ΔHₜ magnitude and correlate with abstention rates.
#    This lives in `monitoring/`, not `tests/`. Example structure:
#
#        # monitoring/ht_delta_observer.py (NOT a pytest file)
#        def observe_delta_ht(ht_series: list[bytes]) -> list[int]:
#            """Count differing bits between successive Hₜ values."""
#            deltas = []
#            for i in range(1, len(ht_series)):
#                xor = bytes(a ^ b for a, b in zip(ht_series[i-1], ht_series[i]))
#                deltas.append(bin(int.from_bytes(xor, 'big')).count('1'))
#            return deltas
#
# 3. WHAT THIS DOES NOT MEAN:
#    - No test in this file reads RFL logs
#    - No test pass/fail depends on abstention thresholds
#    - No Hₜ computation includes RFL metrics
#    - ΔHₜ correlation is observation, not proof
#
# 4. IMPLEMENTATION GATE:
#    Before any Hₜ-adjacent monitoring is added:
#    - [ ] Hₜ must be production-wired (currently ref implementation only)
#    - [ ] Explicit spec review in HT_INVARIANT_SPEC_v1.md Section 12
#    - [ ] Monitoring code lives in `monitoring/`, not `tests/`
#    - [ ] No CI integration without separate approval
#
# ============================================================================


# ============================================================================
# Reference Implementation (to be replaced with production imports)
# ============================================================================

def canonical_json(obj: dict) -> bytes:
    """
    Canonical JSON serialization per INV-CANON-1 and INV-WS-1.
    - Sorted keys
    - No whitespace
    - UTF-8 encoding
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')


def to_hex(data: bytes) -> str:
    """Lowercase hex encoding per INV-CASE-1."""
    return data.hex()


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def merkle_root(leaves: list[bytes]) -> bytes:
    """
    Compute Merkle root with sorted leaves per INV-ORDER-1.
    Domain-separated internal nodes per INV-DOMAIN-1.
    """
    if not leaves:
        return hashlib.sha256(DOMAIN_MERKLE_NODE + b"empty").digest()

    # Sort leaves lexicographically per INV-ORDER-1
    sorted_leaves = sorted(leaves)

    if len(sorted_leaves) == 1:
        return sorted_leaves[0]

    # Pad to power of 2
    while not is_power_of_2(len(sorted_leaves)):
        sorted_leaves.append(sorted_leaves[-1])

    # Build tree bottom-up
    current_level = sorted_leaves
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left, right = current_level[i], current_level[i + 1]
            # Consistent ordering within node
            if left > right:
                left, right = right, left
            parent = hashlib.sha256(DOMAIN_MERKLE_NODE + left + right).digest()
            next_level.append(parent)
        current_level = next_level

    return current_level[0]


def compute_proof_leaf(proof: dict) -> bytes:
    """Compute leaf hash for a proof object."""
    canonical = canonical_json(proof)
    return hashlib.sha256(DOMAIN_PROOF + canonical).digest()


def compute_ui_event_leaf(event: dict) -> bytes:
    """Compute leaf hash for a UI event object."""
    canonical = canonical_json(event)
    return hashlib.sha256(DOMAIN_UI + canonical).digest()


def compute_rt(proofs: list[dict]) -> bytes:
    """Compute Rₜ (Proof Merkle Root)."""
    leaves = [compute_proof_leaf(p) for p in proofs]
    return merkle_root(leaves)


def compute_ut(ui_events: list[dict]) -> bytes:
    """Compute Uₜ (UI Event Merkle Root)."""
    leaves = [compute_ui_event_leaf(e) for e in ui_events]
    return merkle_root(leaves)


def compute_ht(rt: bytes, ut: bytes) -> bytes:
    """Compute Hₜ (Dual-Attestation Hash)."""
    return hashlib.sha256(DOMAIN_HT + rt + ut).digest()


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_proofs() -> list[dict]:
    """Sample proof objects for testing."""
    return [
        {
            "statement_hash": "a" * 64,
            "method": "modus_ponens",
            "parent_hashes": ["b" * 64, "c" * 64],
            "timestamp_ms": 1700000000000,
        },
        {
            "statement_hash": "d" * 64,
            "method": "axiom_instance",
            "parent_hashes": [],
            "timestamp_ms": 1700000001000,
        },
    ]


@pytest.fixture
def sample_ui_events() -> list[dict]:
    """Sample UI event objects for testing."""
    return [
        {
            "event_type": "view",
            "target_hash": "a" * 64,
            "timestamp_ms": 1700000002000,
        },
        {
            "event_type": "query",
            "target_hash": "d" * 64,
            "timestamp_ms": 1700000003000,
        },
    ]


# ============================================================================
# Core Invariant Tests
# ============================================================================

class TestHTRecomputable:
    """INV-VERIFY-1: Hₜ must be independently recomputable."""

    def test_ht_recomputable(self, sample_proofs: list[dict], sample_ui_events: list[dict]):
        """
        Given the same proofs and UI events, computing Hₜ multiple times
        must always produce the same result.
        """
        # Compute Hₜ first time
        rt1 = compute_rt(sample_proofs)
        ut1 = compute_ut(sample_ui_events)
        ht1 = compute_ht(rt1, ut1)

        # Compute Hₜ second time (fresh computation)
        rt2 = compute_rt(sample_proofs)
        ut2 = compute_ut(sample_ui_events)
        ht2 = compute_ht(rt2, ut2)

        # Must be identical
        assert ht1 == ht2, "Hₜ must be deterministically recomputable"
        assert rt1 == rt2, "Rₜ must be deterministically recomputable"
        assert ut1 == ut2, "Uₜ must be deterministically recomputable"

    def test_ht_recomputable_with_reordered_dict_keys(self):
        """
        Canonical JSON must produce same output regardless of dict key insertion order.
        """
        proof1 = {"a": 1, "b": 2, "c": 3}
        proof2 = {"c": 3, "a": 1, "b": 2}
        proof3 = {"b": 2, "c": 3, "a": 1}

        assert canonical_json(proof1) == canonical_json(proof2)
        assert canonical_json(proof2) == canonical_json(proof3)

    def test_ht_recomputable_empty_inputs(self):
        """
        Empty proof and UI event lists must produce well-defined roots.
        """
        rt_empty = compute_rt([])
        ut_empty = compute_ut([])
        ht_empty = compute_ht(rt_empty, ut_empty)

        # Must be deterministic
        assert rt_empty == compute_rt([])
        assert ut_empty == compute_ut([])
        assert ht_empty == compute_ht(rt_empty, ut_empty)

        # Must be the expected "empty" hash
        expected_empty_root = hashlib.sha256(DOMAIN_MERKLE_NODE + b"empty").digest()
        assert rt_empty == expected_empty_root
        assert ut_empty == expected_empty_root


class TestHTMutationDetection:
    """Test that any mutation to inputs changes the output hash."""

    def test_ht_fails_on_mutation(self, sample_proofs: list[dict], sample_ui_events: list[dict]):
        """
        Mutating any proof or UI event must produce a different Hₜ.
        """
        import copy

        # Compute original Hₜ
        original_ht = compute_ht(compute_rt(sample_proofs), compute_ut(sample_ui_events))

        # Mutate a proof field
        mutated_proofs = copy.deepcopy(sample_proofs)
        mutated_proofs[0]["timestamp_ms"] += 1
        mutated_ht = compute_ht(compute_rt(mutated_proofs), compute_ut(sample_ui_events))

        assert original_ht != mutated_ht, "Mutating proof must change Hₜ"

    def test_ht_fails_on_ui_event_mutation(self, sample_proofs: list[dict], sample_ui_events: list[dict]):
        """
        Mutating any UI event must produce a different Hₜ.
        """
        import copy

        original_ht = compute_ht(compute_rt(sample_proofs), compute_ut(sample_ui_events))

        mutated_events = copy.deepcopy(sample_ui_events)
        mutated_events[0]["event_type"] = "export"
        mutated_ht = compute_ht(compute_rt(sample_proofs), compute_ut(mutated_events))

        assert original_ht != mutated_ht, "Mutating UI event must change Hₜ"

    def test_ht_fails_on_proof_removal(self, sample_proofs: list[dict], sample_ui_events: list[dict]):
        """
        Removing a proof must produce a different Hₜ.
        """
        original_ht = compute_ht(compute_rt(sample_proofs), compute_ut(sample_ui_events))

        reduced_proofs = sample_proofs[1:]  # Remove first proof
        reduced_ht = compute_ht(compute_rt(reduced_proofs), compute_ut(sample_ui_events))

        assert original_ht != reduced_ht, "Removing a proof must change Hₜ"

    def test_ht_fails_on_proof_addition(self, sample_proofs: list[dict], sample_ui_events: list[dict]):
        """
        Adding a proof must produce a different Hₜ.
        """
        original_ht = compute_ht(compute_rt(sample_proofs), compute_ut(sample_ui_events))

        extended_proofs = sample_proofs + [{
            "statement_hash": "e" * 64,
            "method": "truth_table",
            "parent_hashes": [],
            "timestamp_ms": 1700000004000,
        }]
        extended_ht = compute_ht(compute_rt(extended_proofs), compute_ut(sample_ui_events))

        assert original_ht != extended_ht, "Adding a proof must change Hₜ"


class TestDifferentRtUtProduceDifferentHt:
    """Test that different Rₜ and Uₜ combinations produce different Hₜ."""

    def test_ht_different_Rt_Ut_produce_different_Ht(self):
        """
        Different combinations of Rₜ and Uₜ must produce unique Hₜ values.
        """
        proofs_a = [{"id": "proof_a", "timestamp_ms": 1000}]
        proofs_b = [{"id": "proof_b", "timestamp_ms": 2000}]
        events_a = [{"id": "event_a", "timestamp_ms": 3000}]
        events_b = [{"id": "event_b", "timestamp_ms": 4000}]

        rt_a = compute_rt(proofs_a)
        rt_b = compute_rt(proofs_b)
        ut_a = compute_ut(events_a)
        ut_b = compute_ut(events_b)

        # All four combinations must produce different Hₜ
        ht_aa = compute_ht(rt_a, ut_a)
        ht_ab = compute_ht(rt_a, ut_b)
        ht_ba = compute_ht(rt_b, ut_a)
        ht_bb = compute_ht(rt_b, ut_b)

        all_hts = {to_hex(ht_aa), to_hex(ht_ab), to_hex(ht_ba), to_hex(ht_bb)}
        assert len(all_hts) == 4, "All Rt/Ut combinations must produce unique Ht"

    def test_same_proofs_different_events_different_ht(self, sample_proofs: list[dict]):
        """
        Same proofs with different UI events must produce different Hₜ.
        """
        events_a = [{"type": "view", "ts": 1}]
        events_b = [{"type": "view", "ts": 2}]

        rt = compute_rt(sample_proofs)
        ut_a = compute_ut(events_a)
        ut_b = compute_ut(events_b)

        ht_a = compute_ht(rt, ut_a)
        ht_b = compute_ht(rt, ut_b)

        assert ht_a != ht_b, "Same Rₜ with different Uₜ must produce different Hₜ"

    def test_different_proofs_same_events_different_ht(self, sample_ui_events: list[dict]):
        """
        Different proofs with same UI events must produce different Hₜ.
        """
        proofs_a = [{"id": "a", "ts": 1}]
        proofs_b = [{"id": "b", "ts": 1}]

        rt_a = compute_rt(proofs_a)
        rt_b = compute_rt(proofs_b)
        ut = compute_ut(sample_ui_events)

        ht_a = compute_ht(rt_a, ut)
        ht_b = compute_ht(rt_b, ut)

        assert ht_a != ht_b, "Different Rₜ with same Uₜ must produce different Hₜ"


class TestDomainSeparation:
    """INV-DOMAIN-1: Domain separation must be enforced."""

    def test_domain_separation_enforced(self):
        """
        Proof and UI event with identical content must produce different leaf hashes
        due to domain separation.
        """
        identical_obj = {
            "hash": "a" * 64,
            "timestamp_ms": 1700000000000,
        }

        proof_leaf = compute_proof_leaf(identical_obj)
        ui_event_leaf = compute_ui_event_leaf(identical_obj)

        assert proof_leaf != ui_event_leaf, (
            "Domain separation must produce different hashes for identical content"
        )

    def test_domain_prefixes_are_distinct(self):
        """
        All domain prefixes must be distinct.
        """
        domains = [DOMAIN_PROOF, DOMAIN_UI, DOMAIN_HT, DOMAIN_MERKLE_NODE]
        assert len(set(domains)) == len(domains), "All domain prefixes must be unique"

    def test_ht_domain_separation_from_rt_ut(self):
        """
        Hₜ must not equal raw concatenation without domain prefix.
        """
        proofs = [{"id": "test"}]
        events = [{"id": "test"}]

        rt = compute_rt(proofs)
        ut = compute_ut(events)

        # With domain separation
        ht_with_domain = compute_ht(rt, ut)

        # Without domain separation (WRONG)
        ht_without_domain = hashlib.sha256(rt + ut).digest()

        assert ht_with_domain != ht_without_domain, (
            "Hₜ must include domain separation prefix"
        )

    def test_merkle_internal_nodes_domain_separated(self):
        """
        Internal Merkle nodes must use domain separation.
        """
        leaf1 = hashlib.sha256(DOMAIN_PROOF + b"leaf1").digest()
        leaf2 = hashlib.sha256(DOMAIN_PROOF + b"leaf2").digest()

        # With domain separation
        node_with_domain = hashlib.sha256(
            DOMAIN_MERKLE_NODE + min(leaf1, leaf2) + max(leaf1, leaf2)
        ).digest()

        # Without domain separation (WRONG)
        node_without_domain = hashlib.sha256(
            min(leaf1, leaf2) + max(leaf1, leaf2)
        ).digest()

        assert node_with_domain != node_without_domain, (
            "Merkle internal nodes must use domain separation"
        )


class TestCanonicalJSON:
    """INV-CANON-1 and INV-WS-1: Canonical JSON tests."""

    def test_no_whitespace_in_canonical_json(self):
        """
        Canonical JSON must not contain extraneous whitespace.
        """
        obj = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        canonical = canonical_json(obj).decode('utf-8')

        # No spaces after colons or commas
        assert ": " not in canonical
        assert ", " not in canonical
        assert "{ " not in canonical
        assert " }" not in canonical
        assert "[ " not in canonical
        assert " ]" not in canonical

    def test_keys_sorted_in_canonical_json(self):
        """
        Canonical JSON must have sorted keys.
        """
        obj = {"z": 1, "a": 2, "m": 3}
        canonical = canonical_json(obj).decode('utf-8')

        # Keys must appear in sorted order
        assert canonical == '{"a":2,"m":3,"z":1}'

    def test_nested_objects_sorted(self):
        """
        Nested objects must also have sorted keys.
        """
        obj = {"outer": {"z": 1, "a": 2}, "inner": {"b": 3, "a": 4}}
        canonical = canonical_json(obj).decode('utf-8')

        expected = '{"inner":{"a":4,"b":3},"outer":{"a":2,"z":1}}'
        assert canonical == expected


class TestHexCasing:
    """INV-CASE-1: Stable lowercase hex encoding."""

    def test_hex_lowercase_only(self):
        """
        All hex representations must be lowercase.
        """
        data = bytes([0xAB, 0xCD, 0xEF, 0x01, 0x23])
        hex_str = to_hex(data)

        assert hex_str == hex_str.lower(), "Hex must be lowercase"
        assert hex_str == "abcdef0123"

    def test_hash_hex_lowercase(self, sample_proofs: list[dict]):
        """
        Hash outputs must be lowercase hex.
        """
        rt = compute_rt(sample_proofs)
        hex_rt = to_hex(rt)

        assert hex_rt == hex_rt.lower(), "Hash hex must be lowercase"
        assert len(hex_rt) == 64, "SHA-256 hex must be 64 characters"


class TestSortedLeafOrder:
    """INV-ORDER-1: Merkle leaves must be sorted."""

    def test_merkle_order_independence(self):
        """
        Input order must not affect Merkle root (leaves are sorted internally).
        """
        proofs_ordered = [
            {"id": "a", "ts": 1},
            {"id": "b", "ts": 2},
            {"id": "c", "ts": 3},
        ]
        proofs_shuffled = [
            {"id": "c", "ts": 3},
            {"id": "a", "ts": 1},
            {"id": "b", "ts": 2},
        ]

        rt_ordered = compute_rt(proofs_ordered)
        rt_shuffled = compute_rt(proofs_shuffled)

        assert rt_ordered == rt_shuffled, (
            "Merkle root must be independent of input order (leaves are sorted)"
        )

    def test_duplicate_leaves_handled(self):
        """
        Duplicate leaves must be handled correctly.
        """
        proofs = [
            {"id": "same", "ts": 1},
            {"id": "same", "ts": 1},  # Exact duplicate
        ]

        # Should not raise and should produce valid root
        rt = compute_rt(proofs)
        assert len(rt) == 32, "Must produce valid 32-byte hash"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_single_proof_single_event(self):
        """
        Single proof and single event must produce valid Hₜ.
        """
        proofs = [{"id": "only_proof", "ts": 1}]
        events = [{"id": "only_event", "ts": 2}]

        rt = compute_rt(proofs)
        ut = compute_ut(events)
        ht = compute_ht(rt, ut)

        assert len(ht) == 32, "Hₜ must be 32 bytes"

    def test_large_number_of_leaves(self):
        """
        Large number of leaves must be handled correctly.
        """
        proofs = [{"id": f"proof_{i}", "ts": i} for i in range(1000)]
        events = [{"id": f"event_{i}", "ts": i} for i in range(500)]

        rt = compute_rt(proofs)
        ut = compute_ut(events)
        ht = compute_ht(rt, ut)

        assert len(ht) == 32, "Hₜ must be 32 bytes for large inputs"

    def test_unicode_in_content(self):
        """
        Unicode content must be handled correctly.
        """
        proofs = [{"statement": "∀x. P(x) → Q(x)", "ts": 1}]
        events = [{"query": "検索クエリ", "ts": 2}]

        rt = compute_rt(proofs)
        ut = compute_ut(events)
        ht = compute_ht(rt, ut)

        # Must be deterministic with unicode
        rt2 = compute_rt(proofs)
        ut2 = compute_ut(events)
        ht2 = compute_ht(rt2, ut2)

        assert ht == ht2, "Unicode content must hash deterministically"

    def test_special_json_values(self):
        """
        Special JSON values (null, boolean) must be handled correctly.
        """
        proofs = [{"value": None, "flag": True, "other": False, "ts": 1}]

        rt1 = compute_rt(proofs)
        rt2 = compute_rt(proofs)

        assert rt1 == rt2, "Special JSON values must hash deterministically"
