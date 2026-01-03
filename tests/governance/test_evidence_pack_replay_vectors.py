"""
Evidence Pack Replay Vectors Test

Generates and validates deterministic test vectors for the JS verifier.
These vectors define the contract between Python and JS implementations.

Spec: docs/EVIDENCE_PACK_VERIFIER_SPEC.md

Run with:
    uv run pytest tests/governance/test_evidence_pack_replay_vectors.py -v

Generate vectors:
    uv run python tests/governance/test_evidence_pack_replay_vectors.py
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)
from substrate.crypto.core import rfc8785_canonicalize


# ---------------------------------------------------------------------------
# Test Vectors: Valid Evidence Packs
# ---------------------------------------------------------------------------

VALID_PACK_MINIMAL = {
    "name": "valid_minimal",
    "description": "Minimal valid pack with one event and one artifact",
    "uvil_events": [
        {
            "event_id": "evt_test_001",
            "event_type": "COMMIT",
            "committed_partition_id": "part_test_001",
            "user_fingerprint": "test_user_001",
            "epoch": 1,
        }
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "art_test_001",
            "claim_id": "claim_test_001",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "proof_payload": {
                "validator": "arithmetic_v1",
                "claim_text": "2 + 2 = 4",
            },
        }
    ],
}

VALID_PACK_EMPTY = {
    "name": "valid_empty",
    "description": "Valid pack with empty arrays (edge case)",
    "uvil_events": [],
    "reasoning_artifacts": [],
}

VALID_PACK_ABSTAINED = {
    "name": "valid_abstained",
    "description": "Valid pack with ABSTAINED outcome (FV claim)",
    "uvil_events": [
        {
            "event_id": "evt_fv_001",
            "event_type": "COMMIT",
            "committed_partition_id": "part_fv_001",
            "user_fingerprint": "test_user_fv",
            "epoch": 2,
        }
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "art_fv_001",
            "claim_id": "claim_fv_001",
            "trust_class": "FV",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {
                "claim_text": "For all x, P(x) implies Q(x)",
                "abstention_reason": "No FV verifier in v0",
            },
        }
    ],
}

VALID_PACK_MULTIPLE = {
    "name": "valid_multiple",
    "description": "Valid pack with multiple events and artifacts",
    "uvil_events": [
        {
            "event_id": "evt_multi_001",
            "event_type": "COMMIT",
            "committed_partition_id": "part_multi",
            "user_fingerprint": "user_a",
            "epoch": 1,
        },
        {
            "event_id": "evt_multi_002",
            "event_type": "EDIT",
            "committed_partition_id": "part_multi",
            "user_fingerprint": "user_a",
            "epoch": 2,
        },
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "art_multi_001",
            "claim_id": "claim_multi_001",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "proof_payload": {"claim_text": "1 + 1 = 2"},
        },
        {
            "artifact_id": "art_multi_002",
            "claim_id": "claim_multi_002",
            "trust_class": "PA",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {"claim_text": "The budget is approved"},
        },
    ],
}


# ---------------------------------------------------------------------------
# Test Vectors: Tampered Evidence Packs
# ---------------------------------------------------------------------------

def create_tampered_ut_pack() -> Dict[str, Any]:
    """Create a pack where uvil_events don't match recorded u_t."""
    base = VALID_PACK_MINIMAL.copy()
    base = {
        "name": "tampered_ut",
        "description": "uvil_events modified after u_t was recorded",
        "uvil_events": [
            {
                "event_id": "evt_TAMPERED",  # Changed from evt_test_001
                "event_type": "COMMIT",
                "committed_partition_id": "part_test_001",
                "user_fingerprint": "test_user_001",
                "epoch": 1,
            }
        ],
        "reasoning_artifacts": VALID_PACK_MINIMAL["reasoning_artifacts"],
        # u_t was computed from original uvil_events
        "override_ut": compute_ui_root(VALID_PACK_MINIMAL["uvil_events"]),
    }
    return base


def create_tampered_rt_pack() -> Dict[str, Any]:
    """Create a pack where reasoning_artifacts don't match recorded r_t."""
    base = {
        "name": "tampered_rt",
        "description": "reasoning_artifacts modified after r_t was recorded",
        "uvil_events": VALID_PACK_MINIMAL["uvil_events"],
        "reasoning_artifacts": [
            {
                "artifact_id": "art_TAMPERED",  # Changed from art_test_001
                "claim_id": "claim_test_001",
                "trust_class": "MV",
                "validation_outcome": "REFUTED",  # Changed from VERIFIED
                "proof_payload": {
                    "validator": "arithmetic_v1",
                    "claim_text": "2 + 2 = 5",  # Changed
                },
            }
        ],
        # r_t was computed from original reasoning_artifacts
        "override_rt": compute_reasoning_root(VALID_PACK_MINIMAL["reasoning_artifacts"]),
    }
    return base


def create_tampered_ht_pack() -> Dict[str, Any]:
    """Create a pack where h_t field was directly modified."""
    base = {
        "name": "tampered_ht",
        "description": "h_t field directly modified",
        "uvil_events": VALID_PACK_MINIMAL["uvil_events"],
        "reasoning_artifacts": VALID_PACK_MINIMAL["reasoning_artifacts"],
        # h_t is a random wrong value
        "override_ht": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    return base


# ---------------------------------------------------------------------------
# Vector Generation
# ---------------------------------------------------------------------------

def generate_evidence_pack(template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate complete evidence pack from template.

    Computes u_t, r_t, h_t unless overrides are provided (for tampered packs).
    """
    uvil_events = template["uvil_events"]
    reasoning_artifacts = template["reasoning_artifacts"]

    # Compute roots
    u_t = template.get("override_ut") or compute_ui_root(uvil_events)
    r_t = template.get("override_rt") or compute_reasoning_root(reasoning_artifacts)
    h_t = template.get("override_ht") or compute_composite_root(r_t, u_t)

    return {
        "schema_version": "v1",
        "name": template["name"],
        "description": template["description"],
        "uvil_events": uvil_events,
        "reasoning_artifacts": reasoning_artifacts,
        "u_t": u_t,
        "r_t": r_t,
        "h_t": h_t,
    }


def verify_evidence_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify an evidence pack by recomputing hashes.

    Returns verification result with match status for each root.
    """
    recomputed_ut = compute_ui_root(pack["uvil_events"])
    recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
    recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

    ut_match = recomputed_ut == pack["u_t"]
    rt_match = recomputed_rt == pack["r_t"]
    ht_match = recomputed_ht == pack["h_t"]

    return {
        "overall": ut_match and rt_match and ht_match,
        "u_t": {
            "recorded": pack["u_t"],
            "recomputed": recomputed_ut,
            "match": ut_match,
        },
        "r_t": {
            "recorded": pack["r_t"],
            "recomputed": recomputed_rt,
            "match": rt_match,
        },
        "h_t": {
            "recorded": pack["h_t"],
            "recomputed": recomputed_ht,
            "match": ht_match,
        },
    }


# ---------------------------------------------------------------------------
# Tests: Valid Packs Must PASS
# ---------------------------------------------------------------------------

class TestValidPacksPass:
    """Valid evidence packs must pass verification."""

    def test_valid_minimal_passes(self):
        """Minimal valid pack passes verification."""
        pack = generate_evidence_pack(VALID_PACK_MINIMAL)
        result = verify_evidence_pack(pack)

        assert result["overall"] is True, "Valid minimal pack should PASS"
        assert result["u_t"]["match"] is True
        assert result["r_t"]["match"] is True
        assert result["h_t"]["match"] is True

    def test_valid_empty_passes(self):
        """Empty pack (edge case) passes verification."""
        pack = generate_evidence_pack(VALID_PACK_EMPTY)
        result = verify_evidence_pack(pack)

        assert result["overall"] is True, "Valid empty pack should PASS"

    def test_valid_abstained_passes(self):
        """Pack with ABSTAINED outcome passes verification."""
        pack = generate_evidence_pack(VALID_PACK_ABSTAINED)
        result = verify_evidence_pack(pack)

        assert result["overall"] is True, "Valid ABSTAINED pack should PASS"

    def test_valid_multiple_passes(self):
        """Pack with multiple events/artifacts passes verification."""
        pack = generate_evidence_pack(VALID_PACK_MULTIPLE)
        result = verify_evidence_pack(pack)

        assert result["overall"] is True, "Valid multiple pack should PASS"


# ---------------------------------------------------------------------------
# Tests: Tampered Packs Must FAIL
# ---------------------------------------------------------------------------

class TestTamperedPacksFail:
    """Tampered evidence packs must fail verification."""

    def test_tampered_ut_fails(self):
        """Pack with tampered uvil_events fails with U_t mismatch."""
        template = create_tampered_ut_pack()
        pack = generate_evidence_pack(template)
        result = verify_evidence_pack(pack)

        assert result["overall"] is False, "Tampered U_t pack should FAIL"
        assert result["u_t"]["match"] is False, "U_t should mismatch"
        # R_t should still match (not tampered)
        # H_t will mismatch because it depends on U_t

    def test_tampered_rt_fails(self):
        """Pack with tampered reasoning_artifacts fails with R_t mismatch."""
        template = create_tampered_rt_pack()
        pack = generate_evidence_pack(template)
        result = verify_evidence_pack(pack)

        assert result["overall"] is False, "Tampered R_t pack should FAIL"
        assert result["r_t"]["match"] is False, "R_t should mismatch"

    def test_tampered_ht_fails(self):
        """Pack with directly modified h_t fails with H_t mismatch."""
        template = create_tampered_ht_pack()
        pack = generate_evidence_pack(template)
        result = verify_evidence_pack(pack)

        assert result["overall"] is False, "Tampered H_t pack should FAIL"
        assert result["h_t"]["match"] is False, "H_t should mismatch"
        # U_t and R_t should still match
        assert result["u_t"]["match"] is True
        assert result["r_t"]["match"] is True


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify computation is deterministic."""

    def test_same_input_same_output(self):
        """Same input always produces same hashes."""
        pack1 = generate_evidence_pack(VALID_PACK_MINIMAL)
        pack2 = generate_evidence_pack(VALID_PACK_MINIMAL)

        assert pack1["u_t"] == pack2["u_t"], "U_t must be deterministic"
        assert pack1["r_t"] == pack2["r_t"], "R_t must be deterministic"
        assert pack1["h_t"] == pack2["h_t"], "H_t must be deterministic"

    def test_canonicalization_is_deterministic(self):
        """RFC 8785 canonicalization is deterministic."""
        obj = {"b": 2, "a": 1, "c": [3, 2, 1]}
        canon1 = rfc8785_canonicalize(obj)
        canon2 = rfc8785_canonicalize(obj)

        assert canon1 == canon2, "Canonicalization must be deterministic"
        # Keys should be sorted
        assert canon1 == '{"a":1,"b":2,"c":[3,2,1]}'


# ---------------------------------------------------------------------------
# Tests: Composite Root Contract
# ---------------------------------------------------------------------------

class TestCompositeRootContract:
    """Verify H_t = SHA256(R_t || U_t) contract."""

    def test_composite_formula(self):
        """H_t is exactly SHA256(R_t || U_t)."""
        pack = generate_evidence_pack(VALID_PACK_MINIMAL)

        # Manual computation
        r_t = pack["r_t"]
        u_t = pack["u_t"]
        expected_ht = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()

        assert pack["h_t"] == expected_ht, "H_t must equal SHA256(R_t || U_t)"


# ---------------------------------------------------------------------------
# Vector Export for JS Verifier
# ---------------------------------------------------------------------------

def export_test_vectors() -> Dict[str, Any]:
    """
    Export all test vectors as JSON for JS verifier validation.

    Returns dict with:
    - valid_packs: Packs that should PASS
    - invalid_packs: Packs that should FAIL with specific mismatches
    - canonicalization_tests: RFC 8785 test cases
    """
    valid_packs = [
        generate_evidence_pack(VALID_PACK_MINIMAL),
        generate_evidence_pack(VALID_PACK_EMPTY),
        generate_evidence_pack(VALID_PACK_ABSTAINED),
        generate_evidence_pack(VALID_PACK_MULTIPLE),
    ]

    invalid_packs = [
        {
            "pack": generate_evidence_pack(create_tampered_ut_pack()),
            "expected_failures": ["u_t", "h_t"],
        },
        {
            "pack": generate_evidence_pack(create_tampered_rt_pack()),
            "expected_failures": ["r_t", "h_t"],
        },
        {
            "pack": generate_evidence_pack(create_tampered_ht_pack()),
            "expected_failures": ["h_t"],
        },
    ]

    canonicalization_tests = [
        {
            "input": {"b": 2, "a": 1},
            "expected": '{"a":1,"b":2}',
        },
        {
            "input": {"nested": {"z": 1, "a": 2}},
            "expected": '{"nested":{"a":2,"z":1}}',
        },
        {
            "input": [3, 1, 2],
            "expected": "[3,1,2]",  # Arrays preserve order
        },
        {
            "input": {"unicode": "\u0041"},  # A
            "expected": '{"unicode":"A"}',
        },
    ]

    return {
        "version": "v1",
        "generated_by": "test_evidence_pack_replay_vectors.py",
        "valid_packs": valid_packs,
        "invalid_packs": invalid_packs,
        "canonicalization_tests": canonicalization_tests,
    }


# ---------------------------------------------------------------------------
# CLI: Generate and print vectors
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vectors = export_test_vectors()

    print("=" * 60)
    print("EVIDENCE PACK TEST VECTORS")
    print("=" * 60)
    print()
    print(json.dumps(vectors, indent=2))
    print()
    print("=" * 60)
    print(f"Valid packs: {len(vectors['valid_packs'])}")
    print(f"Invalid packs: {len(vectors['invalid_packs'])}")
    print(f"Canonicalization tests: {len(vectors['canonicalization_tests'])}")
    print("=" * 60)
