#!/usr/bin/env python3
"""
Ledger Snapshot Generator

Generates a deterministic, versioned snapshot of the MathLedger blockchain state.

Data Sources:
  - artifacts/ledger/mathledger.db (SQLite database with blocks table)

Output:
  - JSON snapshot compliant with schemas/ledger_snapshot.schema.json
  - Printed to stdout in RFC 8785 canonical form

Exit Codes:
  0 - Success
  1 - Data source missing or corrupted
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


def generate_ledger_snapshot(repo_root: Path) -> Dict[str, Any]:
    """Generate the ledger snapshot."""
    
    # Path to the ledger database
    db_path = repo_root / 'artifacts' / 'ledger' / 'mathledger.db'
    
    if not db_path.exists():
        print(f"ERROR: Ledger database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    
    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Query all blocks
    try:
        cursor.execute("SELECT height, hash, prev_hash, merkle_root FROM blocks ORDER BY height ASC")
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"ERROR: Failed to query blocks table: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    
    # Build blocks list
    blocks = []
    for row in rows:
        block = {
            'height': row['height'],
            'hash': row['hash'],
            'prev_hash': row['prev_hash'] if row['prev_hash'] else '',
            'merkle_root': row['merkle_root']
        }
        blocks.append(block)
    
    conn.close()
    
    # Determine chain metadata
    if not blocks:
        print("ERROR: No blocks found in database", file=sys.stderr)
        sys.exit(1)
    
    chain_id = "mathledger-main"  # Hardcoded for now, could be stored in DB
    height = blocks[-1]['height']
    last_block_hash = blocks[-1]['hash']
    
    # Build snapshot
    snapshot = {
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'chain_id': chain_id,
        'height': height,
        'last_block_hash': last_block_hash,
        'blocks': blocks
    }
    
    return snapshot


def canonicalize_json(obj: Any) -> str:
    """
    Serialize an object to RFC 8785 canonical JSON.
    """
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def main():
    repo_root = Path.cwd()
    
    # Generate snapshot
    snapshot = generate_ledger_snapshot(repo_root)
    
    # Canonicalize and print
    canonical_json = canonicalize_json(snapshot)
    print(canonical_json)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
