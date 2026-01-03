#!/usr/bin/env python3
"""
Generate Evidence Pack Examples for Auditors

Creates ready-to-verify evidence pack examples at:
    releases/evidence_pack_examples.v0.2.1.json

Contains:
- 1 valid pack (expected: PASS)
- 1 tampered H_t pack (expected: FAIL - h_t mismatch)
- 1 tampered R_t pack (expected: FAIL - r_t mismatch)

IMPORTANT: All examples are generated using the SAME canonical functions
as replay verification. No hand-crafted hashes.

Usage:
    uv run python scripts/generate_evidence_pack_examples.py
    uv run python scripts/generate_evidence_pack_examples.py --verify
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "releases" / "evidence_pack_examples.v0.2.1.json"


# ---------------------------------------------------------------------------
# Example Data: Valid Pack (Boundary Demo Scenario)
# ---------------------------------------------------------------------------

VALID_PACK_BOUNDARY_DEMO = {
    "name": "valid_boundary_demo",
    "description": "Valid evidence pack from boundary demo (2+2=4 with different trust classes)",
    "uvil_events": [
        {
            "event_id": "boundary_demo_evt_001",
            "event_type": "COMMIT",
            "committed_partition_id": "boundary_demo_part_001",
            "user_fingerprint": "auditor_test_user",
            "epoch": 1,
            "timestamp_iso": "2026-01-03T12:00:00Z",
        },
        {
            "event_id": "boundary_demo_evt_002",
            "event_type": "VERIFICATION_COMPLETE",
            "committed_partition_id": "boundary_demo_part_001",
            "user_fingerprint": "auditor_test_user",
            "epoch": 1,
            "timestamp_iso": "2026-01-03T12:00:01Z",
        },
    ],
    "reasoning_artifacts": [
        {
            "artifact_id": "boundary_demo_art_001",
            "claim_id": "claim_mv_verified",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "claim_text": "2 + 2 = 4",
            "proof_payload": {
                "validator": "arithmetic_v1",
                "parsed_expression": {"left": 4, "right": 4, "operator": "+"},
                "verification_time_ms": 1,
            },
        },
        {
            "artifact_id": "boundary_demo_art_002",
            "claim_id": "claim_pa_abstained",
            "trust_class": "PA",
            "validation_outcome": "ABSTAINED",
            "claim_text": "2 + 2 = 4",
            "proof_payload": {
                "abstention_reason": "PA claims authority-bearing but not mechanically verified in v0",
            },
        },
        {
            "artifact_id": "boundary_demo_art_003",
            "claim_id": "claim_mv_refuted",
            "trust_class": "MV",
            "validation_outcome": "REFUTED",
            "claim_text": "3 * 3 = 8",
            "proof_payload": {
                "validator": "arithmetic_v1",
                "parsed_expression": {"left": 9, "right": 8, "operator": "*"},
                "refutation_reason": "9 != 8",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Pack Generation (using canonical functions)
# ---------------------------------------------------------------------------

def generate_pack(template: dict) -> dict:
    """
    Generate complete evidence pack from template.

    Computes u_t, r_t, h_t using the SAME functions as replay verification.
    """
    uvil_events = template["uvil_events"]
    reasoning_artifacts = template["reasoning_artifacts"]

    # Use SAME functions as backend/api/uvil.py replay_verify
    u_t = compute_ui_root(uvil_events)
    r_t = compute_reasoning_root(reasoning_artifacts)
    h_t = compute_composite_root(r_t, u_t)

    return {
        "schema_version": "v1",
        "pack_version": "v0.2.1",
        "name": template["name"],
        "description": template["description"],
        "uvil_events": uvil_events,
        "reasoning_artifacts": reasoning_artifacts,
        "u_t": u_t,
        "r_t": r_t,
        "h_t": h_t,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "scripts/generate_evidence_pack_examples.py",
    }


def create_tampered_ht_pack(valid_pack: dict) -> dict:
    """
    Create a tampered pack where h_t is directly modified.

    Expected verification result: FAIL with h_t mismatch.
    """
    tampered = valid_pack.copy()
    tampered["name"] = "tampered_ht_mismatch"
    tampered["description"] = (
        "Tampered evidence pack: h_t field directly modified to zeros. "
        "Expected result: FAIL with h_t mismatch. "
        "This demonstrates tamper detection."
    )
    # Replace h_t with all zeros
    tampered["h_t"] = "0" * 64
    tampered["expected_result"] = {
        "verdict": "FAIL",
        "reason": "h_t mismatch",
        "details": "Recorded h_t does not match computed SHA256(r_t || u_t)",
    }
    return tampered


def create_tampered_rt_pack() -> dict:
    """
    Create a tampered pack where reasoning_artifacts were modified after r_t was recorded.

    Expected verification result: FAIL with r_t mismatch.
    """
    # Original artifacts (used to compute original r_t)
    original_artifacts = [
        {
            "artifact_id": "original_art_001",
            "claim_id": "claim_original",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "claim_text": "2 + 2 = 4",
            "proof_payload": {"validator": "arithmetic_v1"},
        },
    ]

    # Tampered artifacts (what's in the pack now)
    tampered_artifacts = [
        {
            "artifact_id": "original_art_001",
            "claim_id": "claim_original",
            "trust_class": "MV",
            "validation_outcome": "REFUTED",  # CHANGED from VERIFIED
            "claim_text": "2 + 2 = 5",  # CHANGED from "2 + 2 = 4"
            "proof_payload": {"validator": "arithmetic_v1"},
        },
    ]

    uvil_events = [
        {
            "event_id": "tamper_test_evt_001",
            "event_type": "COMMIT",
            "committed_partition_id": "tamper_test_part_001",
            "user_fingerprint": "auditor_test",
            "epoch": 1,
        },
    ]

    # Compute roots from ORIGINAL data
    u_t = compute_ui_root(uvil_events)
    r_t_original = compute_reasoning_root(original_artifacts)
    h_t = compute_composite_root(r_t_original, u_t)

    return {
        "schema_version": "v1",
        "pack_version": "v0.2.1",
        "name": "tampered_rt_mismatch",
        "description": (
            "Tampered evidence pack: reasoning_artifacts modified after r_t was recorded. "
            "The pack claims REFUTED for '2+2=5' but r_t was computed from original data "
            "showing VERIFIED for '2+2=4'. Expected result: FAIL with r_t mismatch."
        ),
        "uvil_events": uvil_events,
        "reasoning_artifacts": tampered_artifacts,  # TAMPERED
        "u_t": u_t,
        "r_t": r_t_original,  # From ORIGINAL artifacts
        "h_t": h_t,
        "expected_result": {
            "verdict": "FAIL",
            "reason": "r_t mismatch",
            "details": "Recorded r_t does not match recomputed Merkle root of reasoning_artifacts",
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "scripts/generate_evidence_pack_examples.py",
    }


def verify_pack(pack: dict) -> dict:
    """
    Verify an evidence pack by recomputing hashes.

    This is the SAME logic as replay_verify in backend/api/uvil.py.
    """
    recomputed_ut = compute_ui_root(pack["uvil_events"])
    recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
    recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

    ut_match = recomputed_ut == pack["u_t"]
    rt_match = recomputed_rt == pack["r_t"]
    ht_match = recomputed_ht == pack["h_t"]

    overall = ut_match and rt_match and ht_match

    result = {
        "verdict": "PASS" if overall else "FAIL",
        "u_t": {"recorded": pack["u_t"], "recomputed": recomputed_ut, "match": ut_match},
        "r_t": {"recorded": pack["r_t"], "recomputed": recomputed_rt, "match": rt_match},
        "h_t": {"recorded": pack["h_t"], "recomputed": recomputed_ht, "match": ht_match},
    }

    if not overall:
        mismatches = []
        if not ut_match:
            mismatches.append("u_t")
        if not rt_match:
            mismatches.append("r_t")
        if not ht_match:
            mismatches.append("h_t")
        result["mismatches"] = mismatches
        result["reason"] = f"{', '.join(mismatches)} mismatch"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_examples() -> dict:
    """Generate all evidence pack examples."""

    # 1. Valid pack
    valid_pack = generate_pack(VALID_PACK_BOUNDARY_DEMO)
    valid_pack["expected_result"] = {
        "verdict": "PASS",
        "reason": "All hashes match",
        "details": "u_t, r_t, and h_t all recompute correctly",
    }

    # 2. Tampered h_t pack
    tampered_ht = create_tampered_ht_pack(valid_pack)

    # 3. Tampered r_t pack
    tampered_rt = create_tampered_rt_pack()

    return {
        "schema_version": "v1",
        "description": (
            "Evidence pack examples for auditor verification. "
            "Upload each pack to /v0.2.1/evidence-pack/verify/ to see PASS/FAIL."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "scripts/generate_evidence_pack_examples.py",
        "usage_instructions": {
            "step_1": "Download one of the example packs below",
            "step_2": "Open https://mathledger.ai/v0.2.1/evidence-pack/verify/",
            "step_3": "Upload the pack JSON",
            "step_4": "Click 'Verify' and observe PASS or FAIL",
            "note": "The tampered packs demonstrate tamper detection - they should FAIL",
        },
        "examples": {
            "valid_boundary_demo": {
                "expected_verdict": "PASS",
                "pack": valid_pack,
            },
            "tampered_ht_mismatch": {
                "expected_verdict": "FAIL",
                "expected_reason": "h_t mismatch",
                "pack": tampered_ht,
            },
            "tampered_rt_mismatch": {
                "expected_verdict": "FAIL",
                "expected_reason": "r_t mismatch",
                "pack": tampered_rt,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate evidence pack examples")
    parser.add_argument("--verify", action="store_true", help="Verify examples after generation")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    args = parser.parse_args()

    print("=" * 60)
    print("EVIDENCE PACK EXAMPLES GENERATOR")
    print("=" * 60)
    print()

    # Generate
    examples = generate_all_examples()

    if args.stdout:
        print(json.dumps(examples, indent=2))
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2)
        print(f"OK: Written to: {OUTPUT_PATH}")
        print(f"    Size: {OUTPUT_PATH.stat().st_size:,} bytes")

    # Verify
    if args.verify:
        print()
        print("-" * 60)
        print("VERIFICATION:")
        print("-" * 60)
        print()

        for name, example in examples["examples"].items():
            pack = example["pack"]
            expected = example["expected_verdict"]
            result = verify_pack(pack)
            actual = result["verdict"]

            if actual == expected:
                print(f"OK: {name}: {actual} (expected {expected})")
            else:
                print(f"FAIL: {name}: {actual} (expected {expected})")
                print(f"    Mismatches: {result.get('mismatches', [])}")

    print()
    print("=" * 60)
    print("USAGE:")
    print("  1. Copy a pack from 'examples' to a file")
    print("  2. Upload to https://mathledger.ai/v0.2.1/evidence-pack/verify/")
    print("  3. Click 'Verify' to see PASS or FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()
