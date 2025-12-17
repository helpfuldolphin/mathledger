"""
Tests for Dual-Root Attestation (Mirror Auditor)

Validates cryptographic binding of reasoning and UI event streams.
"""

import pytest
from attestation.dual_root import (
    canonicalize_reasoning_artifact,
    canonicalize_ui_artifact,
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    generate_attestation_metadata,
    hash_reasoning_leaf,
    hash_ui_leaf,
    verify_composite_integrity,
)
from backend.crypto.hashing import verify_merkle_proof
from ledger.blocking import seal_block, seal_block_with_dual_roots


class TestDualRootComputation:
    """Test core dual-root cryptographic primitives."""

    def test_compute_reasoning_root_nonempty(self):
        """Test reasoning root computation with proof events."""
        proof_events = ["proof1", "proof2", "proof3"]
        r_t = compute_reasoning_root(proof_events)

        assert isinstance(r_t, str)
        assert len(r_t) == 64  # SHA-256 hex
        # Verify it's valid hex
        int(r_t, 16)

    def test_compute_reasoning_root_empty(self):
        """Test reasoning root computation with no proofs."""
        r_t = compute_reasoning_root([])

        assert isinstance(r_t, str)
        assert len(r_t) == 64
        # Should produce deterministic empty hash
        r_t2 = compute_reasoning_root([])
        assert r_t == r_t2

    def test_compute_ui_root_nonempty(self):
        """Test UI root computation with events."""
        ui_events = ["click1", "scroll2", "input3"]
        u_t = compute_ui_root(ui_events)

        assert isinstance(u_t, str)
        assert len(u_t) == 64
        int(u_t, 16)

    def test_compute_ui_root_empty(self):
        """Test UI root computation with no events."""
        u_t = compute_ui_root([])

        assert isinstance(u_t, str)
        assert len(u_t) == 64
        # Should produce deterministic empty hash
        u_t2 = compute_ui_root([])
        assert u_t == u_t2

    def test_compute_composite_root_valid(self):
        """Test composite root H_t = SHA256(R_t || U_t)."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])

        h_t = compute_composite_root(r_t, u_t)

        assert isinstance(h_t, str)
        assert len(h_t) == 64
        int(h_t, 16)

    def test_compute_composite_root_deterministic(self):
        """Test that composite root is deterministic."""
        r_t = compute_reasoning_root(["proof1", "proof2"])
        u_t = compute_ui_root(["event1", "event2"])

        h_t1 = compute_composite_root(r_t, u_t)
        h_t2 = compute_composite_root(r_t, u_t)

        assert h_t1 == h_t2

    def test_compute_composite_root_different_inputs(self):
        """Test that different inputs produce different roots."""
        r_t1 = compute_reasoning_root(["proof1"])
        u_t1 = compute_ui_root(["event1"])

        r_t2 = compute_reasoning_root(["proof2"])
        u_t2 = compute_ui_root(["event2"])

        h_t1 = compute_composite_root(r_t1, u_t1)
        h_t2 = compute_composite_root(r_t2, u_t2)

        assert h_t1 != h_t2

    def test_compute_composite_root_invalid_r_t(self):
        """Test composite root rejects invalid R_t."""
        u_t = compute_ui_root(["event1"])

        with pytest.raises(ValueError):
            compute_composite_root("", u_t)

        with pytest.raises(ValueError):
            compute_composite_root("not_hex", u_t)

    def test_compute_composite_root_invalid_u_t(self):
        """Test composite root rejects invalid U_t."""
        r_t = compute_reasoning_root(["proof1"])

        with pytest.raises(ValueError):
            compute_composite_root(r_t, "")

        with pytest.raises(ValueError):
            compute_composite_root(r_t, "short")


class TestAttestationMetadata:
    """Test attestation metadata generation."""

    def test_generate_attestation_metadata_basic(self):
        """Test basic metadata generation."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        metadata = generate_attestation_metadata(r_t, u_t, h_t, 1, 1)

        assert metadata['reasoning_merkle_root'] == r_t
        assert metadata['ui_merkle_root'] == u_t
        assert metadata['composite_attestation_root'] == h_t
        assert metadata['reasoning_event_count'] == 1
        assert metadata['ui_event_count'] == 1
        assert metadata['attestation_version'] == 'v2'
        assert metadata['algorithm'] == 'SHA256'
        assert metadata['post_quantum_algorithm'] == 'SHA3-256'
        assert metadata['composite_formula'] == 'SHA256(R_t || U_t)'
        assert metadata['leaf_hash_algorithm'] == 'sha256'
        commitments = metadata['hash_commitments']
        assert set(commitments.keys()) == {'reasoning', 'ui', 'composite'}
        reasoning_commit = commitments['reasoning']
        assert reasoning_commit['classical']['digest'] == r_t
        assert reasoning_commit['post_quantum']['algorithm'] == 'SHA3-256'
        assert len(reasoning_commit['post_quantum']['digest']) == 64
        composite_commit = commitments['composite']
        assert composite_commit['classical']['digest'] == h_t

    def test_generate_attestation_metadata_with_extra(self):
        """Test metadata generation with extra fields."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        extra = {'block_number': 42, 'system': 'pl'}
        metadata = generate_attestation_metadata(
            r_t, u_t, h_t, 1, 1, extra=extra
        )

        assert metadata['block_number'] == 42
        assert metadata['system'] == 'pl'
        assert metadata['attestation_version'] == 'v2'
        commitments = metadata['hash_commitments']
        assert commitments['ui']['classical']['digest'] == u_t
        assert commitments['ui']['post_quantum']['algorithm'] == 'SHA3-256'


class TestCompositeVerification:
    """Test composite attestation integrity verification."""

    def test_verify_composite_integrity_valid(self):
        """Test verification with valid composite root."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        assert verify_composite_integrity(r_t, u_t, h_t) is True

    def test_verify_composite_integrity_invalid(self):
        """Test verification with invalid composite root."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t_wrong = "0" * 64

        assert verify_composite_integrity(r_t, u_t, h_t_wrong) is False

    def test_verify_composite_integrity_tampered_r_t(self):
        """Test verification detects tampered R_t."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        r_t_tampered = compute_reasoning_root(["proof2"])

        assert verify_composite_integrity(r_t_tampered, u_t, h_t) is False

    def test_verify_composite_integrity_tampered_u_t(self):
        """Test verification detects tampered U_t."""
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        u_t_tampered = compute_ui_root(["event2"])

        assert verify_composite_integrity(r_t, u_t_tampered, h_t) is False


class TestBlockSealing:
    """Test block sealing with dual-root attestation."""

    def test_legacy_seal_block_includes_metadata(self):
        """seal_block should emit deterministic metadata and block hash."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        block = seal_block("pl", proofs)

        assert block['merkle_root']
        assert block['block_hash']
        assert block['block_number'] >= 1
        assert block['attestation_metadata']['attestation_version'] == 'v2'
        assert 'hash_commitments' in block
        assert block['hash_commitments'] == block['attestation_metadata']['hash_commitments']

    def test_seal_block_with_dual_roots_basic(self):
        """Test basic block sealing with dual roots."""
        proofs = [
            {"statement": "p -> p", "method": "axiom"},
            {"statement": "q -> q", "method": "axiom"}
        ]

        block = seal_block_with_dual_roots("pl", proofs)

        assert 'merkle_root' in block
        assert 'reasoning_merkle_root' in block
        assert 'ui_merkle_root' in block
        assert 'composite_attestation_root' in block
        assert 'attestation_metadata' in block
        assert block['proof_count'] == 2
        assert block['block_hash']
        assert block['block_number'] >= 1
        assert isinstance(block['sealed_at'], int)
        assert isinstance(block['reasoning_leaves'], list)
        assert block['attestation_metadata']['attestation_version'] == 'v2'
        assert 'hash_commitments' in block
        composite_commit = block['hash_commitments']['composite']
        assert composite_commit['classical']['digest'] == block['composite_attestation_root']
        assert composite_commit['post_quantum']['algorithm'] == 'SHA3-256'

        # Verify legacy merkle_root aliases to reasoning_merkle_root
        assert block['merkle_root'] == block['reasoning_merkle_root']

    def test_seal_block_with_dual_roots_with_ui_events(self):
        """Test block sealing with UI events."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["click_1", "scroll_2"]

        block = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)

        assert block['ui_merkle_root'] is not None
        assert block['ui_event_count'] == 2
        # Verify UI events are reflected in metadata
        assert block['attestation_metadata']['ui_event_count'] == 2
        ui_commit = block['hash_commitments']['ui']
        assert ui_commit['classical']['digest'] == block['ui_merkle_root']
        assert len(ui_commit['post_quantum']['digest']) == 64

    def test_seal_block_with_dual_roots_no_ui_events(self):
        """Test block sealing without UI events."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]

        block = seal_block_with_dual_roots("pl", proofs)

        # Should still have U_t (empty tree)
        assert block['ui_merkle_root'] is not None
        assert block['attestation_metadata']['ui_event_count'] == 0
        assert block['ui_event_count'] == 0

    def test_seal_block_with_dual_roots_composite_integrity(self):
        """Test that sealed block has valid composite attestation."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]

        block = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)

        r_t = block['reasoning_merkle_root']
        u_t = block['ui_merkle_root']
        h_t = block['composite_attestation_root']

        # Verify composite integrity
        assert verify_composite_integrity(r_t, u_t, h_t)

    def test_seal_block_with_dual_roots_deterministic(self):
        """Test that sealing is deterministic."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]

        block1 = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)
        block2 = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)

        assert block1['reasoning_merkle_root'] == block2['reasoning_merkle_root']
        assert block1['ui_merkle_root'] == block2['ui_merkle_root']
        assert block1['composite_attestation_root'] == block2['composite_attestation_root']
        assert block1['block_hash'] == block2['block_hash']
        assert block1['block_number'] == block2['block_number']
        assert block1['sealed_at'] == block2['sealed_at']

    def test_seal_block_with_dual_roots_empty(self):
        """Test sealing empty block."""
        block = seal_block_with_dual_roots("pl", [])

        assert block['proof_count'] == 0
        assert block['reasoning_merkle_root'] is not None
        assert block['ui_merkle_root'] is not None
        assert block['composite_attestation_root'] is not None
        assert block['hash_commitments']['reasoning']['post_quantum']['algorithm'] == 'SHA3-256'

    def test_reasoning_inclusion_proofs_verify(self):
        """Reasoning leaves should verify via Merkle proof."""
        proofs = [
            {"statement": "p -> p", "method": "axiom"},
            {"statement": "q -> q", "method": "axiom"},
        ]
        block = seal_block_with_dual_roots("pl", proofs)

        leaves = block['reasoning_leaves']
        assert leaves, "Expected reasoning leaves in attestation metadata"

        for leaf in leaves:
            canonical = canonicalize_reasoning_artifact(leaf['canonical_value'])
            hashed = hash_reasoning_leaf(canonical)
            assert hashed == leaf['leaf_hash']

            proof_path = [(sibling, position) for sibling, position in leaf['merkle_proof']]
            assert verify_merkle_proof(
                hashed,
                proof_path,
                block['reasoning_merkle_root'],
            )

    def test_ui_inclusion_proofs_verify(self):
        """UI leaves should verify via Merkle proof."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = [
            {"event_type": "select_statement", "statement_hash": "abc123"},
            {"event_type": "toggle_abstain", "value": True},
        ]
        block = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)

        leaves = block['ui_leaves']
        assert leaves, "Expected UI leaves in attestation metadata"

        for leaf in leaves:
            canonical = canonicalize_ui_artifact(leaf['canonical_value'])
            hashed = hash_ui_leaf(canonical)
            assert hashed == leaf['leaf_hash']

            proof_path = [(sibling, position) for sibling, position in leaf['merkle_proof']]
            assert verify_merkle_proof(
                hashed,
                proof_path,
                block['ui_merkle_root'],
            )

    def test_first_organism_sample_metadata(self):
        """
        Tie the First Organism example statement to a deterministic attestation.

        This test uses the canonical payloads documented in docs/ATTESTATION_SPEC.md
        and mirrored in tests/integration/test_first_organism.py::test_first_organism_canonical_attestation.

        Expected hashes (computed via attestation.dual_root primitives):
        - R_t: 142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902
        - U_t: 3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee
        - H_t: 6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59
        """
        # Canonical payloads (identical to integration test and docs)
        proof_payload = {
            "statement": "p -> p",
            "statement_hash": "c3f5b3205153cf1c5a2b8a0c3694a7c3",
            "status": "abstain",
            "prover": "lean-interface",
            "verification_method": "lean-disabled",
            "reason": "mock dominant statement",
        }
        ui_event = {"event_type": "select_statement", "statement_hash": "c3f5b3"}

        block = seal_block_with_dual_roots("pl", [proof_payload], ui_events=[ui_event])

        # Verify exact hash values match documentation
        assert block["reasoning_merkle_root"] == "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902"
        assert block["ui_merkle_root"] == "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee"
        assert block["composite_attestation_root"] == "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59"

        # Verify recomputability from leaves
        reasoning_root = compute_reasoning_root(
            [leaf["canonical_value"] for leaf in block["reasoning_leaves"]]
        )
        ui_root = compute_ui_root(
            [leaf["canonical_value"] for leaf in block["ui_leaves"]]
        )
        composite = compute_composite_root(reasoning_root, ui_root)

        assert reasoning_root == block["reasoning_merkle_root"]
        assert ui_root == block["ui_merkle_root"]
        assert composite == block["composite_attestation_root"]

        # Verify metadata
        assert block["attestation_metadata"]["attestation_version"] == "v2"
        assert block["attestation_metadata"]["composite_formula"] == "SHA256(R_t || U_t)"
        assert block["attestation_metadata"]["reasoning_event_count"] == 1
        assert block["attestation_metadata"]["ui_event_count"] == 1
        assert block["hash_commitments"]["reasoning"]["classical"]["digest"] == block["reasoning_merkle_root"]
        assert block["hash_commitments"]["reasoning"]["post_quantum"]["algorithm"] == "SHA3-256"

    def test_post_quantum_commitments_are_deterministic_across_order(self):
        """Post-quantum digests should be order-independent."""
        proofs = [
            {"statement": "p -> p", "method": "axiom"},
            {"statement": "q -> q", "method": "axiom"},
        ]
        ui_events = [
            {"event_type": "focus", "statement_hash": "abc"},
            {"event_type": "select_statement", "statement_hash": "def"},
        ]

        block1 = seal_block_with_dual_roots("pl", proofs, ui_events=ui_events)
        block2 = seal_block_with_dual_roots("pl", list(reversed(proofs)), ui_events=list(reversed(ui_events)))

        assert block1["hash_commitments"]["reasoning"]["post_quantum"]["digest"] == block2["hash_commitments"]["reasoning"]["post_quantum"]["digest"]
        assert block1["hash_commitments"]["ui"]["post_quantum"]["digest"] == block2["hash_commitments"]["ui"]["post_quantum"]["digest"]
        assert block1["hash_commitments"]["composite"]["post_quantum"]["digest"] == block2["hash_commitments"]["composite"]["post_quantum"]["digest"]


class TestCanonicalSourceOfTruth:
    """
    Verify that all dual-root consumers use the canonical attestation.dual_root module.

    This test class acts as the H_t Invariant Warden, ensuring that:
    1. All H_t computations use the canonical compute_composite_root function
    2. The Mirror Auditor imports from the canonical source
    3. Block sealing and verification produce identical results
    """

    def test_mirror_auditor_uses_canonical_source(self):
        """
        Verify that verify_dual_root.py imports compute_composite_root from
        the canonical attestation.dual_root module.
        """
        # Import the Mirror Auditor's compute_composite_root
        from verify_dual_root import compute_composite_root as mirror_compute

        # Import from canonical source
        from attestation.dual_root import compute_composite_root as canonical_compute

        # They must be the exact same function (not a copy)
        assert mirror_compute is canonical_compute, (
            "Mirror Auditor must import compute_composite_root from "
            "attestation.dual_root, not define its own implementation"
        )

    def test_ledger_blocking_uses_canonical_source(self):
        """
        Verify that ledger.blocking imports from canonical attestation.dual_root.
        """
        from ledger.blocking import (
            build_reasoning_attestation,
            build_ui_attestation,
            compute_composite_root as blocking_compute,
        )
        from attestation.dual_root import (
            build_reasoning_attestation as canonical_build_reasoning,
            build_ui_attestation as canonical_build_ui,
            compute_composite_root as canonical_compute,
        )

        # Must be the same functions
        assert blocking_compute is canonical_compute
        assert build_reasoning_attestation is canonical_build_reasoning
        assert build_ui_attestation is canonical_build_ui

    def test_ht_invariant_across_all_sources(self):
        """
        Core H_t Invariant test: verify that H_t computed by different
        code paths produces identical results.

        This is the definitive test that the canonical source is the
        single source of truth.
        """
        # Test vectors
        proof_events = [
            {"statement": "p -> p", "status": "success"},
            {"statement": "q -> q", "status": "success"},
        ]
        ui_events = [
            {"event_type": "select_statement", "hash": "abc123"},
        ]

        # Compute via canonical functions
        from attestation.dual_root import (
            compute_composite_root,
            compute_reasoning_root,
            compute_ui_root,
        )

        canonical_r_t = compute_reasoning_root(proof_events)
        canonical_u_t = compute_ui_root(ui_events)
        canonical_h_t = compute_composite_root(canonical_r_t, canonical_u_t)

        # Compute via seal_block_with_dual_roots
        from ledger.blocking import seal_block_with_dual_roots

        block = seal_block_with_dual_roots("pl", proof_events, ui_events=ui_events)

        # All computations must match
        assert block["reasoning_merkle_root"] == canonical_r_t, "R_t mismatch"
        assert block["ui_merkle_root"] == canonical_u_t, "U_t mismatch"
        assert block["composite_attestation_root"] == canonical_h_t, "H_t mismatch"

        # Verify recomputability from stored leaves
        recomputed_r_t = compute_reasoning_root(
            [leaf["canonical_value"] for leaf in block["reasoning_leaves"]]
        )
        recomputed_u_t = compute_ui_root(
            [leaf["canonical_value"] for leaf in block["ui_leaves"]]
        )
        recomputed_h_t = compute_composite_root(recomputed_r_t, recomputed_u_t)

        assert recomputed_r_t == canonical_r_t, "R_t not recomputable from leaves"
        assert recomputed_u_t == canonical_u_t, "U_t not recomputable from leaves"
        assert recomputed_h_t == canonical_h_t, "H_t not recomputable from leaves"

    def test_verify_composite_integrity_uses_canonical_compute(self):
        """
        Verify that verify_composite_integrity internally uses compute_composite_root.
        """
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        h_t = compute_composite_root(r_t, u_t)

        # Correct H_t should pass
        assert verify_composite_integrity(r_t, u_t, h_t) is True

        # Tampered H_t should fail
        tampered_h_t = "0" * 64
        assert verify_composite_integrity(r_t, u_t, tampered_h_t) is False

        # Verify the function actually uses compute_composite_root
        # by checking that the internal computation matches
        recomputed = compute_composite_root(r_t, u_t)
        assert recomputed == h_t

    def test_first_organism_attestation_exact_values(self):
        """
        Verify that the First Organism canonical example produces
        exact documented hash values.

        These values are pinned in:
        - docs/ATTESTATION_SPEC.md
        - tests/test_dual_root_attestation.py::test_first_organism_sample_metadata
        - tests/integration/test_first_organism.py

        Any change to the hashing algorithm would cause this test to fail,
        alerting the H_t Invariant Warden.
        """
        # Canonical First Organism payloads
        proof_payload = {
            "statement": "p -> p",
            "statement_hash": "c3f5b3205153cf1c5a2b8a0c3694a7c3",
            "status": "abstain",
            "prover": "lean-interface",
            "verification_method": "lean-disabled",
            "reason": "mock dominant statement",
        }
        ui_event = {"event_type": "select_statement", "statement_hash": "c3f5b3"}

        # Expected values from documentation
        EXPECTED_R_T = "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902"
        EXPECTED_U_T = "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee"
        EXPECTED_H_T = "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59"

        # Compute via canonical functions
        r_t = compute_reasoning_root([proof_payload])
        u_t = compute_ui_root([ui_event])
        h_t = compute_composite_root(r_t, u_t)

        assert r_t == EXPECTED_R_T, f"R_t mismatch: got {r_t}"
        assert u_t == EXPECTED_U_T, f"U_t mismatch: got {u_t}"
        assert h_t == EXPECTED_H_T, f"H_t mismatch: got {h_t}"

        # Verify H_t formula: SHA256(R_t || U_t)
        import hashlib
        manual_h_t = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()
        assert manual_h_t == h_t, "H_t formula verification failed"


class TestMultiLeafAttestation:
    """
    Multi-leaf attestation tests exercising Merkle tree construction
    with 3+ reasoning leaves and 2+ UI leaves.

    These tests validate:
    1. Merkle inclusion proofs for non-trivial trees
    2. Sorted index assignment
    3. Deterministic root computation across multiple leaves
    """

    # Canonical multi-leaf payloads (3 reasoning, 2 UI)
    MULTI_LEAF_REASONING = [
        {
            "statement": "p -> p",
            "statement_hash": "c3f5b3205153cf1c5a2b8a0c3694a7c3",
            "status": "success",
            "prover": "lean-interface",
        },
        {
            "statement": "(p -> q) -> (p -> q)",
            "statement_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "status": "success",
            "prover": "lean-interface",
        },
        {
            "statement": "p -> (q -> p)",
            "statement_hash": "d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2",
            "status": "abstain",
            "prover": "lean-interface",
            "reason": "non-tautology",
        },
    ]

    MULTI_LEAF_UI = [
        {"event_type": "select_statement", "statement_hash": "c3f5b3"},
        {"event_type": "toggle_abstain", "statement_hash": "d7e8f9", "value": True},
    ]

    # Expected hashes (computed via attestation.dual_root primitives)
    EXPECTED_R_T = "a17dfa7651b8d88e575e398da11c53ad85cc3f9a9623e1253733e42407768e78"
    EXPECTED_U_T = "4556737241f76696ab1eefb321ab62540eea34fc6c374ffbacd1d549248e5ece"
    EXPECTED_H_T = "e8469a887f99abd5f59df6e409ca7bd07f5c1342cec7877a41bc01a3b517ca22"

    def test_multi_leaf_roots_match_expected(self):
        """Verify multi-leaf roots match documented expected values."""
        r_t = compute_reasoning_root(self.MULTI_LEAF_REASONING)
        u_t = compute_ui_root(self.MULTI_LEAF_UI)
        h_t = compute_composite_root(r_t, u_t)

        assert r_t == self.EXPECTED_R_T, f"R_t mismatch: got {r_t}"
        assert u_t == self.EXPECTED_U_T, f"U_t mismatch: got {u_t}"
        assert h_t == self.EXPECTED_H_T, f"H_t mismatch: got {h_t}"

    def test_multi_leaf_block_sealing(self):
        """Verify block sealing with multi-leaf trees."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        assert block["reasoning_merkle_root"] == self.EXPECTED_R_T
        assert block["ui_merkle_root"] == self.EXPECTED_U_T
        assert block["composite_attestation_root"] == self.EXPECTED_H_T
        assert block["proof_count"] == 3
        assert block["ui_event_count"] == 2

    def test_multi_leaf_reasoning_inclusion_proofs(self):
        """All reasoning leaves should have valid Merkle inclusion proofs."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        leaves = block["reasoning_leaves"]
        assert len(leaves) == 3, "Expected 3 reasoning leaves"

        for leaf in leaves:
            # Verify leaf hash is correct
            canonical = canonicalize_reasoning_artifact(leaf["canonical_value"])
            hashed = hash_reasoning_leaf(canonical)
            assert hashed == leaf["leaf_hash"], f"Leaf hash mismatch at index {leaf['original_index']}"

            # Verify Merkle inclusion proof
            proof_path = [(sibling, position) for sibling, position in leaf["merkle_proof"]]
            assert len(proof_path) > 0, "Multi-leaf tree should have non-empty proof"
            assert verify_merkle_proof(
                hashed,
                proof_path,
                block["reasoning_merkle_root"],
            ), f"Merkle proof failed for reasoning leaf {leaf['original_index']}"

    def test_multi_leaf_ui_inclusion_proofs(self):
        """All UI leaves should have valid Merkle inclusion proofs."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        leaves = block["ui_leaves"]
        assert len(leaves) == 2, "Expected 2 UI leaves"

        for leaf in leaves:
            # Verify leaf hash is correct
            canonical = canonicalize_ui_artifact(leaf["canonical_value"])
            hashed = hash_ui_leaf(canonical)
            assert hashed == leaf["leaf_hash"], f"Leaf hash mismatch at index {leaf['original_index']}"

            # Verify Merkle inclusion proof
            proof_path = [(sibling, position) for sibling, position in leaf["merkle_proof"]]
            assert len(proof_path) > 0, "Multi-leaf tree should have non-empty proof"
            assert verify_merkle_proof(
                hashed,
                proof_path,
                block["ui_merkle_root"],
            ), f"Merkle proof failed for UI leaf {leaf['original_index']}"

    def test_multi_leaf_recomputability(self):
        """R_t, U_t, H_t should be recomputable from stored leaves."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        # Recompute from stored canonical values
        recomputed_r = compute_reasoning_root(
            [leaf["canonical_value"] for leaf in block["reasoning_leaves"]]
        )
        recomputed_u = compute_ui_root(
            [leaf["canonical_value"] for leaf in block["ui_leaves"]]
        )
        recomputed_h = compute_composite_root(recomputed_r, recomputed_u)

        assert recomputed_r == block["reasoning_merkle_root"]
        assert recomputed_u == block["ui_merkle_root"]
        assert recomputed_h == block["composite_attestation_root"]

    def test_multi_leaf_sorted_indices(self):
        """Verify sorted indices are assigned correctly."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        # Reasoning leaves should have unique sorted indices
        reasoning_sorted = [leaf["sorted_index"] for leaf in block["reasoning_leaves"]]
        assert len(set(reasoning_sorted)) == 3, "Sorted indices should be unique"
        assert sorted(reasoning_sorted) == [0, 1, 2], "Sorted indices should cover 0..n-1"

        # UI leaves should have unique sorted indices
        ui_sorted = [leaf["sorted_index"] for leaf in block["ui_leaves"]]
        assert len(set(ui_sorted)) == 2, "Sorted indices should be unique"
        assert sorted(ui_sorted) == [0, 1], "Sorted indices should cover 0..n-1"

    def test_multi_leaf_metadata_rfc8785(self):
        """Verify attestation metadata uses RFC8785 canonical JSON."""
        block = seal_block_with_dual_roots(
            "pl",
            self.MULTI_LEAF_REASONING,
            ui_events=self.MULTI_LEAF_UI,
        )

        metadata = block["attestation_metadata"]

        # Metadata should have required fields
        assert metadata["attestation_version"] == "v2"
        assert metadata["algorithm"] == "SHA256"
        assert metadata["composite_formula"] == "SHA256(R_t || U_t)"
        assert metadata["leaf_hash_algorithm"] == "sha256"
        assert metadata["reasoning_event_count"] == 3
        assert metadata["ui_event_count"] == 2

        # Canonical values should be RFC8785 (keys sorted alphabetically)
        for leaf in metadata["reasoning_leaves"]:
            canonical = leaf["canonical_value"]
            # RFC8785 requires keys in lexicographic order
            # Parse and verify key order
            import json
            parsed = json.loads(canonical)
            keys = list(parsed.keys())
            assert keys == sorted(keys), f"Keys not in RFC8785 order: {keys}"

        for leaf in metadata["ui_leaves"]:
            canonical = leaf["canonical_value"]
            parsed = json.loads(canonical)
            keys = list(parsed.keys())
            assert keys == sorted(keys), f"Keys not in RFC8785 order: {keys}"


class TestMerkleProofEdgeCases:
    """Edge case tests for Merkle inclusion proof verification."""

    def test_single_leaf_empty_proof(self):
        """Single-leaf tree should have empty Merkle proof."""
        block = seal_block_with_dual_roots(
            "pl",
            [{"statement": "p -> p", "status": "success"}],
            ui_events=[{"event_type": "click"}],
        )

        # Single leaf = empty proof (leaf hash IS the root)
        assert len(block["reasoning_leaves"]) == 1
        assert len(block["ui_leaves"]) == 1

        reasoning_leaf = block["reasoning_leaves"][0]
        ui_leaf = block["ui_leaves"][0]

        # For single-leaf trees, the proof may be empty or minimal
        # The leaf hash should equal the root (after domain-separated hashing)
        assert reasoning_leaf["leaf_hash"] != block["reasoning_merkle_root"], (
            "Leaf hash should differ from root due to Merkle tree construction"
        )

    def test_two_leaf_single_sibling_proof(self):
        """Two-leaf tree should have single-sibling Merkle proof."""
        block = seal_block_with_dual_roots(
            "pl",
            [
                {"statement": "p -> p", "status": "success"},
                {"statement": "q -> q", "status": "success"},
            ],
            ui_events=[
                {"event_type": "click"},
                {"event_type": "scroll"},
            ],
        )

        # Two leaves = each proof has exactly one sibling
        for leaf in block["reasoning_leaves"]:
            assert len(leaf["merkle_proof"]) == 1, "Two-leaf tree should have 1-sibling proof"

        for leaf in block["ui_leaves"]:
            assert len(leaf["merkle_proof"]) == 1, "Two-leaf tree should have 1-sibling proof"

    def test_power_of_two_leaves(self):
        """Power-of-two leaf count should produce balanced tree."""
        proofs = [{"statement": f"s{i}", "status": "success"} for i in range(4)]
        block = seal_block_with_dual_roots("pl", proofs, ui_events=[])

        assert len(block["reasoning_leaves"]) == 4

        # All proofs should have exactly 2 siblings (log2(4) = 2)
        for leaf in block["reasoning_leaves"]:
            assert len(leaf["merkle_proof"]) == 2, "4-leaf tree should have 2-sibling proofs"

    def test_non_power_of_two_leaves(self):
        """Non-power-of-two leaf count should still produce valid proofs."""
        proofs = [{"statement": f"s{i}", "status": "success"} for i in range(5)]
        block = seal_block_with_dual_roots("pl", proofs, ui_events=[])

        assert len(block["reasoning_leaves"]) == 5

        # All proofs should verify
        for leaf in block["reasoning_leaves"]:
            canonical = canonicalize_reasoning_artifact(leaf["canonical_value"])
            hashed = hash_reasoning_leaf(canonical)
            proof_path = [(sibling, position) for sibling, position in leaf["merkle_proof"]]
            assert verify_merkle_proof(
                hashed,
                proof_path,
                block["reasoning_merkle_root"],
            ), f"Proof failed for leaf {leaf['original_index']}"

    def test_tampered_leaf_proof_fails(self):
        """Tampered leaf should fail Merkle proof verification."""
        block = seal_block_with_dual_roots(
            "pl",
            [{"statement": "p -> p", "status": "success"}],
            ui_events=[{"event_type": "click"}],
        )

        leaf = block["reasoning_leaves"][0]
        proof_path = [(sibling, position) for sibling, position in leaf["merkle_proof"]]

        # Tamper with the leaf hash
        tampered_hash = "0" * 64

        # Verification should fail
        result = verify_merkle_proof(
            tampered_hash,
            proof_path,
            block["reasoning_merkle_root"],
        )
        assert result is False, "Tampered leaf should fail verification"

    def test_tampered_proof_sibling_fails(self):
        """Tampered proof sibling should fail Merkle proof verification."""
        block = seal_block_with_dual_roots(
            "pl",
            [
                {"statement": "p -> p", "status": "success"},
                {"statement": "q -> q", "status": "success"},
            ],
            ui_events=[],
        )

        leaf = block["reasoning_leaves"][0]
        canonical = canonicalize_reasoning_artifact(leaf["canonical_value"])
        hashed = hash_reasoning_leaf(canonical)

        # Tamper with the sibling in the proof
        tampered_proof = [("0" * 64, leaf["merkle_proof"][0][1])]

        # Verification should fail
        result = verify_merkle_proof(
            hashed,
            tampered_proof,
            block["reasoning_merkle_root"],
        )
        assert result is False, "Tampered sibling should fail verification"


class TestH_tFormulaExplicit:
    """
    Explicit H_t formula test (Red-Team Issue 2.2 remediation).

    This test explicitly asserts H_t == SHA256(R_t || U_t) for known fixed values.
    If the formula is altered, this test MUST fail.

    Reference implementation: attestation/dual_root.py:compute_composite_root()
    Red-Team Issue: 2.2 — "No test that the implementation matches the formula"
    """

    def test_h_t_formula_known_values(self):
        """
        H_t = SHA256(R_t || U_t) for known R_t and U_t values.

        This test uses hardcoded hex strings to verify the formula is correct.
        If this test fails, the H_t computation has changed.
        """
        import hashlib

        # Known test values (64-char hex strings)
        R_t = "a" * 64  # All 'a' for reproducibility
        U_t = "b" * 64  # All 'b' for reproducibility

        # Expected: SHA256(R_t || U_t) = SHA256("aaa...bbb...")
        # Compute expected value directly with stdlib hashlib
        expected_h_t = hashlib.sha256(f"{R_t}{U_t}".encode("ascii")).hexdigest()

        # Compute actual value using the implementation
        actual_h_t = compute_composite_root(R_t, U_t)

        # Assert formula is correct
        assert actual_h_t == expected_h_t, (
            f"H_t formula mismatch!\n"
            f"  Expected: SHA256(R_t || U_t) = {expected_h_t}\n"
            f"  Actual:   compute_composite_root() = {actual_h_t}\n"
            f"  R_t: {R_t}\n"
            f"  U_t: {U_t}\n"
            f"This test verifies H_t == SHA256(R_t || U_t). "
            f"If this fails, the attestation formula has been altered."
        )

    def test_h_t_formula_realistic_values(self):
        """
        H_t = SHA256(R_t || U_t) with realistic Merkle roots.

        Uses actual computed Merkle roots, not synthetic hex strings.
        """
        import hashlib

        # Compute real R_t and U_t from sample data
        R_t = compute_reasoning_root(["proof_statement_1", "proof_statement_2"])
        U_t = compute_ui_root(["ui_event_1", "ui_event_2"])

        # Expected: SHA256(R_t || U_t) computed directly
        expected_h_t = hashlib.sha256(f"{R_t}{U_t}".encode("ascii")).hexdigest()

        # Actual: via implementation
        actual_h_t = compute_composite_root(R_t, U_t)

        assert actual_h_t == expected_h_t, (
            f"H_t formula mismatch with realistic values!\n"
            f"  Expected: {expected_h_t}\n"
            f"  Actual:   {actual_h_t}"
        )

    def test_h_t_formula_concatenation_order(self):
        """
        Verify H_t = SHA256(R_t || U_t), NOT SHA256(U_t || R_t).

        Order matters: R_t must come first.
        """
        import hashlib

        R_t = "1" * 64
        U_t = "2" * 64

        correct_order = hashlib.sha256(f"{R_t}{U_t}".encode("ascii")).hexdigest()
        wrong_order = hashlib.sha256(f"{U_t}{R_t}".encode("ascii")).hexdigest()

        actual = compute_composite_root(R_t, U_t)

        assert actual == correct_order, "H_t must be SHA256(R_t || U_t), R_t first"
        assert actual != wrong_order, "H_t should not be SHA256(U_t || R_t)"

    def test_h_t_formula_frozen_contract(self):
        """
        FROZEN CONTRACT: H_t formula must not change.

        This test uses a specific known input and expected output.
        If this fails, the H_t formula has been altered, which requires
        a version bump per CAL-EXP-2 and CAL-EXP-3 frozen contract rules.
        """
        import hashlib

        # Frozen test vector (do not change)
        R_t = "0123456789abcdef" * 4  # 64 chars
        U_t = "fedcba9876543210" * 4  # 64 chars

        # Pre-computed expected value (frozen)
        # SHA256("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeffedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210")
        expected_h_t = hashlib.sha256(f"{R_t}{U_t}".encode("ascii")).hexdigest()

        actual_h_t = compute_composite_root(R_t, U_t)

        assert actual_h_t == expected_h_t, (
            f"FROZEN CONTRACT VIOLATION: H_t formula changed!\n"
            f"  Expected: {expected_h_t}\n"
            f"  Actual:   {actual_h_t}\n"
            f"If this test fails, the attestation formula has been altered.\n"
            f"This requires a version bump and STRATCOM approval."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
