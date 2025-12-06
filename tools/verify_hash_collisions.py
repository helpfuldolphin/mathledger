#!/usr/bin/env python3
"""
Hash collision detector for MathLedger statements.

Verifies that no two different normalized statements produce the same hash.
This is critical for ledger integrity.

Usage:
  python tools/verify_hash_collisions.py           # Check all statements
  python tools/verify_hash_collisions.py --sample 1000  # Check random sample
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from typing import Dict, List

try:
    import psycopg
except ImportError:
    print("ERROR: psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(2)


def verify_hash_collisions(conn, sample_size: int = None) -> bool:
    """
    Verify no hash collisions exist in the statements table.
    
    Returns True if no collisions found, False otherwise.
    """
    with conn.cursor() as cur:
        if sample_size:
            cur.execute("""
                SELECT hash, content_norm
                FROM statements
                ORDER BY RANDOM()
                LIMIT %s
            """, (sample_size,))
            print(f"Checking random sample of {sample_size} statements...")
        else:
            cur.execute("""
                SELECT hash, content_norm
                FROM statements
            """)
            print("Checking all statements...")
        
        hash_to_content: Dict[str, List[str]] = defaultdict(list)
        total_count = 0
        
        for row in cur.fetchall():
            stmt_hash, content = row
            
            if isinstance(stmt_hash, bytes):
                stmt_hash = stmt_hash.hex()
            
            hash_to_content[stmt_hash].append(content)
            total_count += 1
        
        print(f"Analyzed {total_count} statements")
        print(f"Unique hashes: {len(hash_to_content)}")
        
        collisions = []
        for stmt_hash, contents in hash_to_content.items():
            if len(contents) > 1:
                unique_contents = set(contents)
                if len(unique_contents) > 1:
                    collisions.append((stmt_hash, list(unique_contents)))
        
        if collisions:
            print(f"\n[FAIL] Found {len(collisions)} hash collisions:")
            for stmt_hash, contents in collisions:
                print(f"\nHash: {stmt_hash}")
                for i, content in enumerate(contents, 1):
                    print(f"  Content {i}: {content}")
            return False
        else:
            print(f"\n[PASS] VERIFIED: No hash collisions detected")
            return True


def verify_hash_recomputation(conn, sample_size: int = 100) -> bool:
    """
    Verify that stored hashes match recomputed hashes from content.
    
    Returns True if all hashes match, False otherwise.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hash, content_norm
            FROM statements
            ORDER BY RANDOM()
            LIMIT %s
        """, (sample_size,))
        
        print(f"\nVerifying hash recomputation for {sample_size} statements...")
        
        mismatches = []
        for row in cur.fetchall():
            stored_hash, content = row
            
            if isinstance(stored_hash, bytes):
                stored_hash = stored_hash.hex()
            
            computed_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if stored_hash != computed_hash:
                mismatches.append((content, stored_hash, computed_hash))
        
        if mismatches:
            print(f"\n[FAIL] Found {len(mismatches)} hash mismatches:")
            for content, stored, computed in mismatches[:5]:  # Show first 5
                print(f"\nContent: {content}")
                print(f"  Stored:   {stored}")
                print(f"  Computed: {computed}")
            if len(mismatches) > 5:
                print(f"\n... and {len(mismatches) - 5} more")
            return False
        else:
            print(f"[PASS] VERIFIED: All {sample_size} hashes match recomputed values")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify hash integrity in MathLedger statements"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Check random sample of N statements (default: all)"
    )
    parser.add_argument(
        "--recompute",
        type=int,
        default=100,
        help="Number of statements to verify hash recomputation (default: 100)"
    )
    
    args = parser.parse_args()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL environment variable not set")
        return 2
    
    try:
        with psycopg.connect(db_url) as conn:
            print("="*60)
            print("MathLedger Hash Integrity Verification")
            print("="*60)
            print()
            
            collision_check = verify_hash_collisions(conn, sample_size=args.sample)
            
            recompute_check = verify_hash_recomputation(conn, sample_size=args.recompute)
            
            print()
            print("="*60)
            if collision_check and recompute_check:
                print("[PASS] VERIFIED: ALL HASH INTEGRITY CHECKS PASSED")
                print("="*60)
                return 0
            else:
                print("[FAIL] Hash integrity violations detected")
                print("="*60)
                return 1
    
    except psycopg.OperationalError as e:
        print(f"[ERROR] Database connection failed: {e}")
        return 2
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
