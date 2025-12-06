#!/usr/bin/env python3
"""
validate_nightly.py - Validates nightly run consistency
"""

import os
import json
import sys
import psycopg
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Validate nightly run consistency")
    parser.add_argument('--offline', action='store_true',
                       help='Skip database checks, validate only file integrity')
    args = parser.parse_args()
    try:
        if args.offline:
            # Offline mode: skip database checks, validate only file integrity
            block_number = 999  # Use mock block number for offline validation
        else:
            from backend.security.runtime_env import get_required_env

            # Check DB connectivity
            db_url = get_required_env("DATABASE_URL")
            with psycopg.connect(db_url) as conn:
                with conn.cursor() as cur:
                    # Get latest block
                    cur.execute("""
                        SELECT block_number, merkle_root
                        FROM blocks
                        ORDER BY block_number DESC
                        LIMIT 1
                    """)
                    latest_block = cur.fetchone()

                    if not latest_block:
                        print('validate=fail reason="no blocks found in database"')
                        sys.exit(1)

                    block_number, merkle_root = latest_block

        # Check progress.md contains latest block
        progress_file = Path("docs/progress.md")
        if not progress_file.exists():
            print('validate=fail reason="docs/progress.md not found"')
            sys.exit(1)

        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_content = f.read()

        if f"Block {block_number}" not in progress_content:
            print(f'validate=fail reason="block {block_number} not found in progress.md"')
            sys.exit(1)

        # Check ratchet_last.txt exists and is valid JSON
        ratchet_file = Path("metrics/ratchet_last.txt")
        if not ratchet_file.exists():
            print('validate=fail reason="metrics/ratchet_last.txt not found"')
            sys.exit(1)

        try:
            with open(ratchet_file, 'r', encoding='utf-8-sig') as f:
                ratchet_data = json.load(f)

            # Validate required fields
            required_fields = ['decide', 'system', 'slice', 'atoms', 'depth', 'reason']
            for field in required_fields:
                if field not in ratchet_data:
                    print(f'validate=fail reason="ratchet_last.txt missing field: {field}"')
                    sys.exit(1)

            # Check for unexpected fields
            expected_fields = set(['decide', 'system', 'slice', 'atoms', 'depth', 'reason'])
            actual_fields = set(ratchet_data.keys())
            if actual_fields != expected_fields:
                print(f'validate=fail reason="ratchet_last.txt has unexpected fields: {", ".join(actual_fields - expected_fields)}"')
                sys.exit(1)

        except json.JSONDecodeError as e:
            print(f'validate=fail reason="ratchet_last.txt invalid JSON: {e}"')
            sys.exit(1)

        # All checks passed
        if args.offline:
            print('validate=ok-offline')
        else:
            print('validate=ok')
        sys.exit(0)

    except psycopg.Error as e:
        print(f'validate=fail reason="database error: {e}"')
        sys.exit(1)
    except Exception as e:
        print(f'validate=fail reason="{e}"')
        sys.exit(1)

if __name__ == "__main__":
    main()
