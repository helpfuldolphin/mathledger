#!/usr/bin/env python3
"""
Standalone Merkle root verification utility.

Verifies that Merkle roots in the database match recomputed values.
Can be run as a one-liner to validate blockchain integrity.

Usage:
  python tools/verify_merkle.py                    # Verify all blocks
  python tools/verify_merkle.py --block 123        # Verify specific block
  python tools/verify_merkle.py --latest           # Verify latest block only
"""

import argparse
import hashlib
import os
import sys
from typing import List, Optional

try:
    import psycopg
except ImportError:
    print("ERROR: psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(2)


def compute_merkle_root(statement_ids: List[str]) -> str:
    """
    Recompute Merkle root from statement IDs.
    Must match backend.ledger.blockchain.merkle_root logic.
    """
    if not statement_ids:
        return hashlib.sha256(b"").hexdigest()
    
    level = sorted([s.encode('utf-8') for s in statement_ids])
    nodes = [hashlib.sha256(x).digest() for x in level]
    
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nxt = []
        for i in range(0, len(nodes), 2):
            nxt.append(hashlib.sha256(nodes[i] + nodes[i+1]).digest())
        nodes = nxt
    
    return nodes[0].hex()


def verify_block_merkle(conn, block_number: int) -> bool:
    """Verify Merkle root for a specific block."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT block_number, merkle_root
            FROM blocks
            WHERE block_number = %s
        """, (block_number,))
        
        row = cur.fetchone()
        if not row:
            print(f"[ERROR] Block {block_number} not found")
            return False
        
        block_num, stored_merkle = row
        
        cur.execute("""
            SELECT s.hash
            FROM statements s
            JOIN proofs p ON p.statement_id = s.id
            WHERE p.block_id = (SELECT id FROM blocks WHERE block_number = %s)
            ORDER BY s.hash
        """, (block_number,))
        
        statement_hashes = [row[0] for row in cur.fetchall()]
        
        if not statement_hashes:
            print(f"[WARN] Block {block_number} has no statements")
            return True
        
        computed_merkle = compute_merkle_root(statement_hashes)
        
        if isinstance(stored_merkle, bytes):
            stored_merkle = stored_merkle.hex()
        
        if computed_merkle == stored_merkle:
            print(f"[PASS] Block {block_number}: Merkle root verified ({len(statement_hashes)} statements)")
            return True
        else:
            print(f"[FAIL] Block {block_number}: Merkle root mismatch")
            print(f"  Stored:   {stored_merkle}")
            print(f"  Computed: {computed_merkle}")
            print(f"  Statements: {len(statement_hashes)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Merkle roots in MathLedger blockchain"
    )
    parser.add_argument(
        "--block",
        type=int,
        help="Verify specific block number"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Verify latest block only"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of blocks to verify (default: 10)"
    )
    
    args = parser.parse_args()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL environment variable not set")
        return 2
    
    try:
        with psycopg.connect(db_url) as conn:
            if args.block:
                success = verify_block_merkle(conn, args.block)
                return 0 if success else 1
            
            elif args.latest:
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(block_number) FROM blocks")
                    latest = cur.fetchone()[0]
                    if latest is None:
                        print("[WARN] No blocks found in database")
                        return 0
                    success = verify_block_merkle(conn, latest)
                    return 0 if success else 1
            
            else:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT block_number 
                        FROM blocks 
                        ORDER BY block_number DESC 
                        LIMIT %s
                    """, (args.limit,))
                    block_numbers = [row[0] for row in cur.fetchall()]
                
                if not block_numbers:
                    print("[WARN] No blocks found in database")
                    return 0
                
                print(f"Verifying {len(block_numbers)} blocks...\n")
                
                all_passed = True
                for block_num in reversed(block_numbers):
                    if not verify_block_merkle(conn, block_num):
                        all_passed = False
                
                print()
                if all_passed:
                    print(f"[PASS] VERIFIED: All {len(block_numbers)} blocks have valid Merkle roots")
                    return 0
                else:
                    print(f"[FAIL] One or more blocks have invalid Merkle roots")
                    return 1
    
    except psycopg.OperationalError as e:
        print(f"[ERROR] Database connection failed: {e}")
        return 2
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
