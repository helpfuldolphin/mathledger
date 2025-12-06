#!/usr/bin/env python3
"""
Generate valid test data for governance validation.
Creates properly-chained blocks with correct cryptographic hashes.
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.crypto.hashing import sha256_hex, DOMAIN_BLCK


def generate_valid_roots(output_path: Path):
    """Generate valid declared roots with proper chain threading."""

    roots = []

    # Generate proper 64-char Merkle roots
    root1_hash = sha256_hex("block1_statements", domain=b'\x00')
    root2_hash = sha256_hex("block2_statements", domain=b'\x00')
    root3_hash = sha256_hex("block3_statements", domain=b'\x00')

    # Block 1 (genesis)
    block1 = {
        "block_number": 1,
        "root_hash": root1_hash,
        "prev_hash": "",
        "statement_count": 10,
        "sealed_at": "2025-11-01T00:00:00.000000+00:00"
    }
    roots.append(block1)

    # Compute block 1 hash for block 2's prev_hash
    block1_data = json.dumps({
        "block_number": block1["block_number"],
        "root_hash": block1["root_hash"],
        "sealed_at": block1["sealed_at"]
    }, sort_keys=True)
    block1_hash = sha256_hex(block1_data, domain=DOMAIN_BLCK)

    # Block 2
    block2 = {
        "block_number": 2,
        "root_hash": root2_hash,
        "prev_hash": block1_hash,
        "statement_count": 15,
        "sealed_at": "2025-11-01T01:00:00.000000+00:00"
    }
    roots.append(block2)

    # Compute block 2 hash for block 3's prev_hash
    block2_data = json.dumps({
        "block_number": block2["block_number"],
        "root_hash": block2["root_hash"],
        "sealed_at": block2["sealed_at"]
    }, sort_keys=True)
    block2_hash = sha256_hex(block2_data, domain=DOMAIN_BLCK)

    # Block 3
    block3 = {
        "block_number": 3,
        "root_hash": root3_hash,
        "prev_hash": block2_hash,
        "statement_count": 20,
        "sealed_at": "2025-11-01T02:00:00.000000+00:00"
    }
    roots.append(block3)

    declared_roots = {
        "version": "1.0.0",
        "exported_at": "2025-11-04T18:40:00.000000Z",
        "block_count": len(roots),
        "roots": roots
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(declared_roots, f, indent=2, sort_keys=True)

    print(f"✅ Generated valid declared roots: {len(roots)} blocks")
    print(f"   Block 1 hash: {block1_hash}")
    print(f"   Block 2 hash: {block2_hash}")


if __name__ == "__main__":
    output_path = Path("artifacts/governance/declared_roots.json")
    generate_valid_roots(output_path)
