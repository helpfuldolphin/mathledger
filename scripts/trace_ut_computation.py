#!/usr/bin/env python3
"""
Stage-wise trace for U_t computation.

This script traces every step of U_t computation to identify divergence
between Python and JS implementations.

Usage:
    uv run python scripts/trace_ut_computation.py
"""

import json
import hashlib
from pathlib import Path

# Import the actual functions used in Python
from substrate.crypto.core import rfc8785_canonicalize
from substrate.crypto.hashing import sha256_hex, sha256_bytes, DOMAIN_LEAF, DOMAIN_NODE
from attestation.dual_root import DOMAIN_UI_LEAF, hash_ui_leaf
from normalization.canon import normalize

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_PATH = REPO_ROOT / "site" / "v0.2.6" / "evidence-pack" / "examples.json"


def trace_ut_computation(uvil_events: list) -> dict:
    """Trace every step of U_t computation."""
    trace = {
        "input_events": uvil_events,
        "event_traces": [],
        "merkle_trace": None,
        "final_u_t": None,
    }

    leaf_hashes = []

    print("=" * 60)
    print("STAGE 1: Hash each UI event with DOMAIN_UI_LEAF")
    print("=" * 60)

    for i, event in enumerate(uvil_events):
        event_trace = {"index": i, "original": event}

        # Step 1: RFC 8785 canonicalization
        canonical = rfc8785_canonicalize(event)
        event_trace["canonical_json"] = canonical
        print(f"\n[Event {i}] RFC 8785 canonical:")
        print(f"  {canonical}")

        # Step 2: UTF-8 encode
        canonical_bytes = canonical.encode('utf-8')
        event_trace["canonical_bytes_hex"] = canonical_bytes.hex()
        print(f"  Bytes (UTF-8): {canonical_bytes.hex()}")

        # Step 3: Domain prefix
        domain_bytes = DOMAIN_UI_LEAF
        event_trace["domain_prefix_hex"] = domain_bytes.hex()
        print(f"  Domain prefix: {domain_bytes.hex()} ({domain_bytes!r})")

        # Step 4: Full payload to hash (domain + data)
        full_payload = domain_bytes + canonical_bytes
        event_trace["full_payload_hex"] = full_payload.hex()
        print(f"  Full payload: {full_payload.hex()[:40]}...")

        # Step 5: SHA256 hash
        leaf_hash = hashlib.sha256(full_payload).hexdigest()
        event_trace["leaf_hash"] = leaf_hash
        print(f"  Leaf hash: {leaf_hash}")

        # Verify matches our hash_ui_leaf function
        verify_hash = hash_ui_leaf(canonical)
        event_trace["hash_ui_leaf_result"] = verify_hash
        assert leaf_hash == verify_hash, f"Mismatch! {leaf_hash} != {verify_hash}"
        print(f"  (verified: matches hash_ui_leaf)")

        leaf_hashes.append(leaf_hash)
        trace["event_traces"].append(event_trace)

    print("\n" + "=" * 60)
    print("STAGE 2: Merkle tree construction")
    print("=" * 60)

    merkle_trace = trace_merkle_root(leaf_hashes)
    trace["merkle_trace"] = merkle_trace
    trace["final_u_t"] = merkle_trace["final_root"]

    return trace


def trace_merkle_root(leaf_hashes: list) -> dict:
    """Trace merkle_root computation step by step."""
    trace = {
        "input_leaf_hashes": leaf_hashes,
        "after_normalize": [],
        "after_encode": [],
        "sorted_leaves_hex": [],
        "levels": [],
        "final_root": None,
    }

    print(f"\nInput leaf hashes ({len(leaf_hashes)} items):")
    for i, lh in enumerate(leaf_hashes):
        print(f"  [{i}] {lh}")

    # Step 1: normalize() each leaf hash
    print("\n--- After normalize() ---")
    for i, lh in enumerate(leaf_hashes):
        normalized = normalize(lh)
        trace["after_normalize"].append(normalized)
        changed = "(CHANGED!)" if normalized != lh else "(unchanged)"
        print(f"  [{i}] {normalized} {changed}")

    # Step 2: UTF-8 encode
    print("\n--- After UTF-8 encode ---")
    encoded_leaves = []
    for i, norm_lh in enumerate(trace["after_normalize"]):
        encoded = norm_lh.encode('utf-8')
        encoded_leaves.append(encoded)
        trace["after_encode"].append(encoded.hex())
        print(f"  [{i}] {encoded.hex()}")

    # Step 3: Sort
    sorted_leaves = sorted(encoded_leaves)
    print("\n--- After sorting ---")
    for i, sl in enumerate(sorted_leaves):
        trace["sorted_leaves_hex"].append(sl.hex())
        print(f"  [{i}] {sl.hex()}")

    # Step 4: Hash each leaf with DOMAIN_LEAF
    print("\n--- Hash with DOMAIN_LEAF ---")
    nodes = []
    level_0 = []
    for i, leaf_bytes in enumerate(sorted_leaves):
        full_payload = DOMAIN_LEAF + leaf_bytes
        node_hash = hashlib.sha256(full_payload).digest()
        nodes.append(node_hash)
        level_0.append({
            "payload_hex": full_payload.hex(),
            "hash_hex": node_hash.hex()
        })
        print(f"  [{i}] payload: {full_payload.hex()[:40]}...")
        print(f"       hash: {node_hash.hex()}")
    trace["levels"].append(level_0)

    # Step 5: Build tree
    level_num = 1
    while len(nodes) > 1:
        print(f"\n--- Level {level_num} ({len(nodes)} -> {(len(nodes) + 1) // 2}) ---")

        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
            print(f"  (duplicated last node for odd count)")

        next_level = []
        level_trace = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i + 1]
            full_payload = DOMAIN_NODE + combined
            node_hash = hashlib.sha256(full_payload).digest()
            next_level.append(node_hash)
            level_trace.append({
                "left_hash": nodes[i].hex(),
                "right_hash": nodes[i + 1].hex(),
                "result_hash": node_hash.hex()
            })
            print(f"  [{i//2}] L: {nodes[i].hex()[:16]}... R: {nodes[i+1].hex()[:16]}...")
            print(f"       Result: {node_hash.hex()}")

        trace["levels"].append(level_trace)
        nodes = next_level
        level_num += 1

    final_root = nodes[0].hex()
    trace["final_root"] = final_root
    print(f"\nFinal U_t: {final_root}")

    return trace


def main():
    # Load test data
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    valid_pack = examples["examples"]["valid_boundary_demo"]["pack"]

    uvil_events = valid_pack["uvil_events"]
    expected_u_t = valid_pack["u_t"]

    print("=" * 60)
    print("U_t COMPUTATION TRACE")
    print("=" * 60)
    print(f"Expected U_t: {expected_u_t}")
    print()

    trace = trace_ut_computation(uvil_events)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Expected: {expected_u_t}")
    print(f"Computed: {trace['final_u_t']}")
    if trace['final_u_t'] == expected_u_t:
        print("STATUS: MATCH!")
    else:
        print("STATUS: MISMATCH!")

    # Save trace for comparison
    trace_path = REPO_ROOT / "tmp" / "ut_trace_python.json"
    trace_path.parent.mkdir(exist_ok=True)
    with open(trace_path, 'w') as f:
        json.dump(trace, f, indent=2)
    print(f"\nTrace saved to: {trace_path}")


if __name__ == "__main__":
    main()
