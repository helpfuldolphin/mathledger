#!/usr/bin/env python3
"""
Generate Canonical Evidence Pack Verifier Test Vectors

This script generates the version-pinned test vector artifact at:
    releases/evidence_pack_verifier_vectors.v0.2.0.json

CRITICAL: Uses the SAME canonicalization + hashing functions as replay_verify.
No reimplementation. No handmade vectors.

Run with:
    uv run python scripts/generate_verifier_vectors.py

The output is a deterministic JSON file that:
1. Is repo-tracked as authoritative test vectors
2. Can be used by JS verifier implementation for parity testing
3. Includes human-readable names and expected results
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import the SAME functions used by replay_verify
from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)
from substrate.crypto.core import rfc8785_canonicalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "releases" / "evidence_pack_verifier_vectors.v0.2.0.json"

# ---------------------------------------------------------------------------
# Vector Templates
# ---------------------------------------------------------------------------

# Valid pack #1: Minimal with one event and one artifact
VALID_MINIMAL_TEMPLATE = {
    "uvil_events": [
        {
            "event_id": "evt_verifier_001",
            "event_type": "COMMIT",
            "committed_partition_id": "partition_verifier_001",
            "user_fingerprint": "verifier_test_user",
            "epoch": 1,
        }
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "art_verifier_001",
            "claim_id": "claim_verifier_001",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "proof_payload": {
                "validator": "arithmetic_v1",
                "claim_text": "3 + 5 = 8",
                "computed_value": 8,
            },
        }
    ],
}

# Valid pack #2: Multiple events and artifacts
VALID_MULTI_TEMPLATE = {
    "uvil_events": [
        {
            "event_id": "evt_multi_001",
            "event_type": "COMMIT",
            "committed_partition_id": "partition_multi",
            "user_fingerprint": "user_alpha",
            "epoch": 1,
        },
        {
            "event_id": "evt_multi_002",
            "event_type": "PROMOTE",
            "committed_partition_id": "partition_multi",
            "user_fingerprint": "user_alpha",
            "epoch": 2,
        },
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "art_multi_001",
            "claim_id": "claim_multi_001",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "proof_payload": {"claim_text": "10 - 3 = 7"},
        },
        {
            "artifact_id": "art_multi_002",
            "claim_id": "claim_multi_002",
            "trust_class": "PA",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {"claim_text": "Project deadline is Q2"},
        },
    ],
}


# ---------------------------------------------------------------------------
# Vector Generation Functions
# ---------------------------------------------------------------------------


def generate_valid_pack(template: Dict[str, Any], name: str, description: str) -> Dict[str, Any]:
    """
    Generate a VALID evidence pack using the SAME functions as replay_verify.

    Returns a pack with:
    - Correctly computed u_t, r_t, h_t
    - expected_result: "PASS"
    """
    uvil_events = template["uvil_events"]
    reasoning_artifacts = template["reasoning_artifacts"]

    # Use SAME functions as replay_verify
    u_t = compute_ui_root(uvil_events)
    r_t = compute_reasoning_root(reasoning_artifacts)
    h_t = compute_composite_root(r_t, u_t)

    return {
        "name": name,
        "description": description,
        "expected_result": "PASS",
        "expected_failure_reason": None,
        "pack": {
            "schema_version": "v1",
            "uvil_events": uvil_events,
            "reasoning_artifacts": reasoning_artifacts,
            "u_t": u_t,
            "r_t": r_t,
            "h_t": h_t,
        },
    }


def generate_tampered_ht_pack() -> Dict[str, Any]:
    """
    Generate INVALID pack: h_t field directly tampered.

    The h_t field is set to zeros, but u_t and r_t are correct.
    Verifier should detect H_t mismatch.
    """
    uvil_events = VALID_MINIMAL_TEMPLATE["uvil_events"]
    reasoning_artifacts = VALID_MINIMAL_TEMPLATE["reasoning_artifacts"]

    # Compute CORRECT u_t and r_t
    u_t = compute_ui_root(uvil_events)
    r_t = compute_reasoning_root(reasoning_artifacts)

    # TAMPER: Set h_t to wrong value
    h_t = "0" * 64  # All zeros

    return {
        "name": "invalid_tampered_ht",
        "description": "h_t field directly modified to zeros. u_t and r_t are correct.",
        "expected_result": "FAIL",
        "expected_failure_reason": "h_t_mismatch",
        "pack": {
            "schema_version": "v1",
            "uvil_events": uvil_events,
            "reasoning_artifacts": reasoning_artifacts,
            "u_t": u_t,
            "r_t": r_t,
            "h_t": h_t,
        },
    }


def generate_tampered_rt_pack() -> Dict[str, Any]:
    """
    Generate INVALID pack: reasoning_artifacts leaf tampered after r_t was computed.

    The reasoning_artifacts are modified, but r_t reflects the ORIGINAL artifacts.
    Verifier should detect R_t mismatch.
    """
    # Original artifacts (used to compute r_t)
    original_artifacts = VALID_MINIMAL_TEMPLATE["reasoning_artifacts"]

    # Compute r_t from ORIGINAL artifacts
    r_t = compute_reasoning_root(original_artifacts)

    # TAMPERED artifacts (what's in the pack)
    tampered_artifacts = [
        {
            "artifact_id": "art_TAMPERED",  # Changed
            "claim_id": "claim_verifier_001",
            "trust_class": "MV",
            "validation_outcome": "REFUTED",  # Changed from VERIFIED
            "proof_payload": {
                "validator": "arithmetic_v1",
                "claim_text": "3 + 5 = 9",  # Wrong!
                "computed_value": 8,
            },
        }
    ]

    uvil_events = VALID_MINIMAL_TEMPLATE["uvil_events"]
    u_t = compute_ui_root(uvil_events)
    h_t = compute_composite_root(r_t, u_t)

    return {
        "name": "invalid_tampered_reasoning_leaf",
        "description": "reasoning_artifacts modified after r_t was recorded. artifact_id and claim_text changed.",
        "expected_result": "FAIL",
        "expected_failure_reason": "r_t_mismatch",
        "pack": {
            "schema_version": "v1",
            "uvil_events": uvil_events,
            "reasoning_artifacts": tampered_artifacts,  # Tampered!
            "u_t": u_t,
            "r_t": r_t,  # From original
            "h_t": h_t,
        },
    }


def generate_missing_field_pack() -> Dict[str, Any]:
    """
    Generate INVALID pack: missing required field (validation_outcome).

    The reasoning_artifacts are missing the validation_outcome field.
    Verifier should detect schema violation.
    """
    uvil_events = VALID_MINIMAL_TEMPLATE["uvil_events"]

    # Artifacts missing validation_outcome
    incomplete_artifacts = [
        {
            "artifact_id": "art_incomplete_001",
            "claim_id": "claim_incomplete_001",
            "trust_class": "MV",
            # validation_outcome is MISSING
            "proof_payload": {
                "claim_text": "2 + 2 = 4",
            },
        }
    ]

    # We compute hashes anyway (verifier should fail before checking hashes)
    u_t = compute_ui_root(uvil_events)
    r_t = compute_reasoning_root(incomplete_artifacts)
    h_t = compute_composite_root(r_t, u_t)

    return {
        "name": "invalid_missing_validation_outcome",
        "description": "reasoning_artifacts[0] is missing required field 'validation_outcome'.",
        "expected_result": "FAIL",
        "expected_failure_reason": "missing_required_field",
        "pack": {
            "schema_version": "v1",
            "uvil_events": uvil_events,
            "reasoning_artifacts": incomplete_artifacts,
            "u_t": u_t,
            "r_t": r_t,
            "h_t": h_t,
        },
    }


def generate_canonicalization_tests() -> List[Dict[str, Any]]:
    """
    Generate RFC 8785 canonicalization edge tests.

    Uses the SAME rfc8785_canonicalize function from substrate/crypto/core.py.
    """
    tests = []

    # Test 1: Key ordering
    input1 = {"zebra": 1, "alpha": 2, "beta": 3}
    expected1 = rfc8785_canonicalize(input1)
    tests.append({
        "name": "canon_key_ordering",
        "description": "Keys must be sorted lexicographically: alpha < beta < zebra",
        "input": input1,
        "expected_canonical": expected1,
        "verification_note": "Output should be: {\"alpha\":2,\"beta\":3,\"zebra\":1}",
    })

    # Test 2: Nested object ordering + whitespace
    input2 = {"outer": {"z_inner": 1, "a_inner": 2}, "level1": True}
    expected2 = rfc8785_canonicalize(input2)
    tests.append({
        "name": "canon_nested_ordering",
        "description": "Nested objects also have keys sorted. No whitespace.",
        "input": input2,
        "expected_canonical": expected2,
        "verification_note": "Inner keys also sorted: a_inner < z_inner",
    })

    return tests


# ---------------------------------------------------------------------------
# Main Generation
# ---------------------------------------------------------------------------


def generate_all_vectors() -> Dict[str, Any]:
    """
    Generate complete test vector artifact.

    Returns a dict suitable for JSON serialization.
    """
    # Valid packs (2 required)
    valid_packs = [
        generate_valid_pack(
            VALID_MINIMAL_TEMPLATE,
            "valid_minimal",
            "Minimal valid pack: one uvil_event, one reasoning_artifact with VERIFIED outcome.",
        ),
        generate_valid_pack(
            VALID_MULTI_TEMPLATE,
            "valid_multiple",
            "Valid pack with multiple events (COMMIT, PROMOTE) and multiple artifacts (MV+PA).",
        ),
    ]

    # Invalid packs (3 required, each failing for different reason)
    invalid_packs = [
        generate_tampered_ht_pack(),      # a) tampered h_t
        generate_tampered_rt_pack(),       # b) tampered reasoning_artifacts leaf
        generate_missing_field_pack(),     # c) missing required field
    ]

    # Canonicalization edge tests (2 required)
    canonicalization_tests = generate_canonicalization_tests()

    # Compute artifact hash for integrity
    artifact_content = {
        "valid_packs": valid_packs,
        "invalid_packs": invalid_packs,
        "canonicalization_tests": canonicalization_tests,
    }
    content_hash = hashlib.sha256(
        json.dumps(artifact_content, sort_keys=True).encode()
    ).hexdigest()

    return {
        "$schema": "evidence_pack_verifier_vectors.schema.json",
        "metadata": {
            "version": "v0.2.0",
            "generated_at": "2026-01-02T00:00:00Z",
            "generated_by": "scripts/generate_verifier_vectors.py",
            "spec": "docs/EVIDENCE_PACK_VERIFIER_SPEC.md",
            "content_hash": content_hash,
            "description": "Canonical test vectors for Evidence Pack Verifier. Uses SAME functions as replay_verify.",
        },
        "valid_packs": valid_packs,
        "invalid_packs": invalid_packs,
        "canonicalization_tests": canonicalization_tests,
    }


def main():
    """Generate and write the test vector artifact."""
    vectors = generate_all_vectors()

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write with consistent formatting
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Generated: {OUTPUT_PATH}")
    print(f"Content hash: {vectors['metadata']['content_hash']}")
    print(f"Valid packs: {len(vectors['valid_packs'])}")
    print(f"Invalid packs: {len(vectors['invalid_packs'])}")
    print(f"Canonicalization tests: {len(vectors['canonicalization_tests'])}")


if __name__ == "__main__":
    main()
