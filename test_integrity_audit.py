#!/usr/bin/env python3
"""
Integrity Sentinel Audit - Determinism and Canonicalization Tests
Tests the core guarantees of MathLedger: determinism, ASCII-purity, hash consistency.
"""

import sys
import hashlib
from typing import List, Tuple

# Import core modules
from normalization.canon import normalize, _to_ascii
from backend.crypto.hashing import merkle_root, sha256_hex, DOMAIN_LEAF, DOMAIN_NODE
from backend.ledger.blocking import seal_block
from backend.repro.determinism import (
    deterministic_timestamp,
    deterministic_uuid,
    deterministic_merkle_root,
    seeded_rng
)

# ============================================================================
# TEST 1: FORMULA CANONICALIZATION DETERMINISM
# ============================================================================

def test_canonicalization_determinism():
    """Verify that canonicalization produces identical output for equivalent formulas."""
    test_cases = [
        # Commutative operators (top-level)
        ("p /\\ q", "q /\\ p"),
        ("p \\/ q", "q \\/ p"),

        # Unicode to ASCII
        ("p → q", "p -> q"),
        ("p ∧ q", "p /\\ q"),
        ("p ∨ q", "p \\/ q"),
        ("¬p", "~p"),

        # Whitespace normalization
        ("p->q", "p -> q"),
        ("p  ->  q", "p -> q"),
        ("p->  q", "p ->q"),

        # Nested structures (top-level commutative)
        ("(p /\\ q) \\/ r", "r \\/ (p /\\ q)"),  # Note: AND operand order preserved
    ]

    failures = []
    for formula1, formula2 in test_cases:
        norm1 = normalize(formula1)
        norm2 = normalize(formula2)
        if norm1 != norm2:
            failures.append((formula1, formula2, norm1, norm2))

    if failures:
        print(f"[FAIL] Canonicalization determinism: {len(failures)} failures")
        for f1, f2, n1, n2 in failures[:3]:
            print(f"  {f1!r} -> {n1!r}")
            print(f"  {f2!r} -> {n2!r}")
        return False

    print(f"[PASS] Canonicalization determinism: {len(test_cases)} cases verified")
    return True


# ============================================================================
# TEST 2: ASCII-PURITY OF CANONICAL FORMS
# ============================================================================

def test_ascii_purity():
    """Verify that normalized formulas contain only ASCII characters."""
    test_formulas = [
        "p → q",
        "p ∧ q ∨ r",
        "¬(p → q)",
        "（p → q）",  # Full-width parentheses
        "p\u00A0→\u2002q",  # Non-breaking spaces
        "p ⇒ q",
        "p ⟹ q",
        "p ↔ q",
    ]

    failures = []
    for formula in test_formulas:
        norm = normalize(formula)
        try:
            norm.encode('ascii')
        except UnicodeEncodeError:
            failures.append((formula, norm))

    if failures:
        print(f"[FAIL] ASCII-purity: {len(failures)} non-ASCII outputs")
        for f, n in failures[:3]:
            print(f"  {f!r} -> {n!r}")
            print(f"  Contains: {[c for c in n if ord(c) > 127]}")
        return False

    print(f"[PASS] ASCII-purity verified: {len(test_formulas)} formulas")
    return True


# ============================================================================
# TEST 3: HASH CONSISTENCY (SHA-256)
# ============================================================================

def test_hash_consistency():
    """Verify that identical normalized formulas produce identical hashes."""
    test_formulas = [
        ("p -> p", "p->p"),
        ("p /\\ q", "q /\\ p"),
        ("p → q", "p -> q"),
    ]

    failures = []
    for f1, f2 in test_formulas:
        norm1 = normalize(f1)
        norm2 = normalize(f2)
        hash1 = hashlib.sha256(norm1.encode('utf-8')).hexdigest()
        hash2 = hashlib.sha256(norm2.encode('utf-8')).hexdigest()

        if hash1 != hash2:
            failures.append((f1, f2, hash1, hash2))

    if failures:
        print(f"[FAIL] Hash consistency: {len(failures)} mismatches")
        for f1, f2, h1, h2 in failures:
            print(f"  {f1!r} -> {h1}")
            print(f"  {f2!r} -> {h2}")
        return False

    print(f"[PASS] SHA-256 hash consistency: {len(test_formulas)} pairs verified")
    return True


# ============================================================================
# TEST 4: MERKLE ROOT DETERMINISM
# ============================================================================

def test_merkle_determinism():
    """Verify that Merkle root is order-independent (due to sorting)."""
    statements = ["p->p", "p->q->r", "(p/\\q)->p"]

    # Compute Merkle root in different orders
    root1 = merkle_root(statements)
    root2 = merkle_root(list(reversed(statements)))
    root3 = merkle_root([statements[1], statements[0], statements[2]])

    if root1 != root2 or root1 != root3:
        print(f"[FAIL] Merkle root determinism")
        print(f"  Order 1: {root1}")
        print(f"  Order 2: {root2}")
        print(f"  Order 3: {root3}")
        return False

    print(f"[PASS] Merkle root determinism: order-independent verified")
    return True


# ============================================================================
# TEST 5: TIMESTAMP DETERMINISM
# ============================================================================

def test_timestamp_determinism():
    """Verify that deterministic timestamps are consistent across calls."""
    seeds = [0, 42, 12345]

    failures = []
    for seed in seeds:
        ts1 = deterministic_timestamp(seed)
        ts2 = deterministic_timestamp(seed)
        if ts1 != ts2:
            failures.append((seed, ts1, ts2))

    if failures:
        print(f"[FAIL] Timestamp determinism: {len(failures)} mismatches")
        for seed, ts1, ts2 in failures:
            print(f"  Seed {seed}: {ts1} != {ts2}")
        return False

    print(f"[PASS] Timestamp determinism: {len(seeds)} seeds verified")
    return True


# ============================================================================
# TEST 6: UUID DETERMINISM
# ============================================================================

def test_uuid_determinism():
    """Verify that deterministic UUIDs are consistent for same content."""
    test_contents = ["p->p", "p/\\q->p", "(p->q)->r"]

    failures = []
    for content in test_contents:
        uuid1 = deterministic_uuid(content)
        uuid2 = deterministic_uuid(content)
        if uuid1 != uuid2:
            failures.append((content, uuid1, uuid2))

    if failures:
        print(f"[FAIL] UUID determinism: {len(failures)} mismatches")
        for c, u1, u2 in failures:
            print(f"  {c!r}: {u1} != {u2}")
        return False

    print(f"[PASS] UUID determinism: {len(test_contents)} contents verified")
    return True


# ============================================================================
# TEST 7: RNG DETERMINISM
# ============================================================================

def test_rng_determinism():
    """Verify that seeded RNG produces consistent sequences."""
    seed = 42
    rng1 = seeded_rng(seed)
    rng2 = seeded_rng(seed)

    vals1 = rng1.random(10)
    vals2 = rng2.random(10)

    # Handle both numpy arrays and lists
    try:
        import numpy as np
        if not np.array_equal(vals1, vals2):
            print(f"[FAIL] RNG determinism: sequences differ")
            print(f"  RNG1: {vals1}")
            print(f"  RNG2: {vals2}")
            return False
    except ImportError:
        if vals1 != vals2:
            print(f"[FAIL] RNG determinism: sequences differ")
            print(f"  RNG1: {vals1}")
            print(f"  RNG2: {vals2}")
            return False

    print(f"[PASS] RNG determinism: seeded sequences match")
    return True


# ============================================================================
# TEST 8: BLOCK SEALING DETERMINISM
# ============================================================================

def test_block_sealing_determinism():
    """Verify that block sealing produces consistent results."""
    proofs = [
        {"statement": "p->p", "hash": "abc123"},
        {"statement": "(p/\\q)->p", "hash": "def456"},
    ]

    # Seal the same proofs multiple times
    block1 = seal_block("pl", proofs)
    block2 = seal_block("pl", proofs)

    if block1["merkle_root"] != block2["merkle_root"]:
        print(f"[FAIL] Block sealing determinism: Merkle roots differ")
        print(f"  Block 1: {block1['merkle_root']}")
        print(f"  Block 2: {block2['merkle_root']}")
        return False

    if block1["sealed_at"] != block2["sealed_at"]:
        print(f"[FAIL] Block sealing determinism: timestamps differ")
        print(f"  Block 1: {block1['sealed_at']}")
        print(f"  Block 2: {block2['sealed_at']}")
        return False

    print(f"[PASS] Block sealing determinism: consistent sealing verified")
    return True


# ============================================================================
# TEST 9: DOMAIN SEPARATION IN HASHING
# ============================================================================

def test_domain_separation():
    """Verify that domain separation prevents collision attacks."""
    data = "test_data"

    hash_no_domain = sha256_hex(data)
    hash_leaf = sha256_hex(data, domain=DOMAIN_LEAF)
    hash_node = sha256_hex(data, domain=DOMAIN_NODE)

    if hash_no_domain == hash_leaf or hash_no_domain == hash_node or hash_leaf == hash_node:
        print(f"[FAIL] Domain separation: collision detected")
        print(f"  No domain: {hash_no_domain}")
        print(f"  LEAF:      {hash_leaf}")
        print(f"  NODE:      {hash_node}")
        return False

    print(f"[PASS] Domain separation: unique hashes per domain")
    return True


# ============================================================================
# TEST 10: RANDOMIZED DETERMINISM TEST
# ============================================================================

def test_randomized_determinism():
    """Execute multiple runs with random inputs to detect nondeterminism."""
    import random
    random.seed(42)

    failures = []
    for i in range(50):
        # Generate random formula
        ops = ["->", "/\\", "\\/"]
        atoms = ["p", "q", "r", "s"]
        formula = f"{random.choice(atoms)} {random.choice(ops)} {random.choice(atoms)}"

        # Normalize multiple times
        norm1 = normalize(formula)
        norm2 = normalize(formula)
        norm3 = normalize(formula)

        if not (norm1 == norm2 == norm3):
            failures.append((formula, norm1, norm2, norm3))

    if failures:
        print(f"[FAIL] Randomized determinism: {len(failures)} inconsistencies")
        for f, n1, n2, n3 in failures[:3]:
            print(f"  {f!r}: {n1!r}, {n2!r}, {n3!r}")
        return False

    print(f"[PASS] Randomized determinism: 50 random formulas stable")
    return True


# ============================================================================
# MAIN AUDIT RUNNER
# ============================================================================

def main():
    print("=" * 80)
    print("INTEGRITY SENTINEL AUDIT")
    print("Testing determinism, canonicalization, and cryptographic integrity")
    print("=" * 80)
    print()

    tests = [
        ("Canonicalization Determinism", test_canonicalization_determinism),
        ("ASCII-Purity", test_ascii_purity),
        ("Hash Consistency (SHA-256)", test_hash_consistency),
        ("Merkle Root Determinism", test_merkle_determinism),
        ("Timestamp Determinism", test_timestamp_determinism),
        ("UUID Determinism", test_uuid_determinism),
        ("RNG Determinism", test_rng_determinism),
        ("Block Sealing Determinism", test_block_sealing_determinism),
        ("Domain Separation", test_domain_separation),
        ("Randomized Determinism", test_randomized_determinism),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 80)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[ABSTAIN] {name}: Exception during test")
            print(f"  Error: {type(e).__name__}: {e}")
            results.append((name, None))

    # Summary
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    abstained = sum(1 for _, result in results if result is None)

    for name, result in results:
        status = "[PASS]" if result else "[ABSTAIN]" if result is None else "[FAIL]"
        print(f"{status} {name}")

    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Abstained: {abstained}")
    print()

    if failed == 0 and abstained == 0:
        print("[PASS] Integrity Verified systems={} determinism=VERIFIED".format(len(tests)))
        return 0
    elif failed > 0:
        print(f"[FAIL] Integrity Compromised failures={failed}")
        return 1
    else:
        print(f"[ABSTAIN] Drift Unquantified abstentions={abstained}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
