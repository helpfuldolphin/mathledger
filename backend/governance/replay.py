#!/usr/bin/env python3
"""
Lawkeeper Chain Replay Verification
Re-adjudicate provenance chains with full hash recomputation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.crypto.hashing import sha256_hex, DOMAIN_BLCK


def replay_governance_chain(chain_path: Path):
    """Replay governance chain and verify threading."""
    print("⚖️  REPLAYING GOVERNANCE CHAIN")
    print("=" * 60)

    with open(chain_path) as f:
        data = json.load(f)

    entries = data["entries"]

    if not entries:
        print("❌ Empty chain")
        return False

    print(f"📋 Chain entries: {len(entries)}")
    print()

    valid = True

    for i, entry in enumerate(entries):
        sig = entry["signature"]
        prev_sig = entry.get("prev_signature", "")
        score = entry["determinism_score"]
        status = entry["status"]

        print(f"Entry {i}:")
        print(f"  Signature:      {sig}")
        print(f"  Prev Signature: {prev_sig if prev_sig else '(genesis)'}")
        print(f"  Determinism:    {score}/100 {'✓' if score >= 95 else '✗'}")
        print(f"  Status:         {status}")

        # Verify signature format
        if len(sig) != 64 or not all(c in '0123456789abcdef' for c in sig):
            print(f"  ❌ INVALID signature format")
            valid = False
        else:
            print(f"  ✓ Valid signature format")

        # Verify determinism threshold
        if score < 95:
            print(f"  ❌ FAIL: Determinism below threshold (95)")
            valid = False
        else:
            print(f"  ✓ Determinism ≥95")

        # Verify threading (if not genesis)
        if i > 0:
            expected_prev = entries[i-1]["signature"]
            if prev_sig != expected_prev:
                print(f"  ❌ FAIL: Threading broken")
                print(f"     Expected: {expected_prev}")
                print(f"     Got:      {prev_sig}")
                valid = False
            else:
                print(f"  ✓ Threading intact")

        print()

    return valid


def replay_root_chain(roots_path: Path):
    """Replay root chain and verify prev_hash threading."""
    print("⚖️  REPLAYING ROOT CHAIN")
    print("=" * 60)

    with open(roots_path) as f:
        data = json.load(f)

    roots = data["roots"]

    if not roots:
        print("❌ Empty roots")
        return False

    print(f"📋 Block count: {len(roots)}")
    print()

    valid = True

    for i, root in enumerate(roots):
        block_num = root["block_number"]
        root_hash = root["root_hash"]
        prev_hash = root.get("prev_hash", "")
        stmt_count = root["statement_count"]
        sealed_at = root["sealed_at"]

        print(f"Block {block_num}:")
        print(f"  Root Hash:      {root_hash}")
        print(f"  Prev Hash:      {prev_hash if prev_hash else '(genesis)'}")
        print(f"  Statements:     {stmt_count}")
        print(f"  Sealed At:      {sealed_at}")

        # Verify root hash format
        if len(root_hash) != 64 or not all(c in '0123456789abcdef' for c in root_hash):
            print(f"  ❌ INVALID root hash format")
            valid = False
        else:
            print(f"  ✓ Valid root hash format")

        # Recompute expected prev_hash (if not genesis)
        if i > 0:
            prev_block = roots[i-1]
            prev_block_data = json.dumps({
                "block_number": prev_block["block_number"],
                "root_hash": prev_block["root_hash"],
                "sealed_at": prev_block["sealed_at"]
            }, sort_keys=True)

            expected_prev_hash = sha256_hex(prev_block_data, domain=DOMAIN_BLCK)

            print(f"  Recomputed prev_hash: {expected_prev_hash}")

            if prev_hash != expected_prev_hash:
                print(f"  ❌ FAIL: Threading broken")
                print(f"     Expected: {expected_prev_hash}")
                print(f"     Got:      {prev_hash}")
                valid = False
            else:
                print(f"  ✓ Threading verified (hash match)")

        print()

    return valid


def main():
    """Main re-adjudication routine."""
    print()
    print("⚖" * 30)
    print("⚖️  LAWKEEPER RE-ADJUDICATION")
    print("⚖" * 30)
    print()

    gov_chain = Path("artifacts/governance/governance_chain.json")
    roots = Path("artifacts/governance/declared_roots.json")

    # Replay governance chain
    gov_valid = replay_governance_chain(gov_chain)

    # Replay root chain
    roots_valid = replay_root_chain(roots)

    # Final verdict
    print("=" * 60)
    print("⚖️  FINAL VERDICT")
    print("=" * 60)

    if gov_valid and roots_valid:
        print("✅ [LAWFUL] Governance Chain Integrity OK")
        print("✅ All determinism scores ≥95")
        print("✅ All threading verified")
        print("✅ All hash formats valid")
        print()
        print("⚖️  Judicial order maintained.")
        return 0
    else:
        print("❌ [UNLAWFUL] Violations detected")
        print()
        print("⚖️  Judicial order breached. Exit 1 (fail-closed).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
