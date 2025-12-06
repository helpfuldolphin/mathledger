#!/usr/bin/env python3
"""
Hash State Drift Extractor
==========================

Extracts composite attestation roots (H_t) and verified event counts (N_v)
from the Ledger database for First Organism scaling analysis.

See: docs/RFL_DELTAH_ANALYSIS.md
"""

import argparse
import csv
import sys
import os
from typing import Optional

# Ensure project root is in path for module imports
sys.path.append(os.getcwd())

import psycopg
from backend.security.runtime_env import get_database_url


def extract_ht_data(output_file: Optional[str], system_id: str) -> None:
    """
    Extract H_t data to CSV.
    
    Args:
        output_file: Path to output CSV file (or None for stdout).
        system_id: The System ID to filter by.
    """
    dsn = get_database_url()
    
    query = """
        SELECT 
            block_number,
            composite_attestation_root,
            proof_count,
            sealed_at
        FROM blocks 
        WHERE system_id = %s
        ORDER BY block_number ASC
    """

    rows_extracted = 0
    min_nv = float('inf')
    max_nv = float('-inf')
    min_block = float('inf')
    max_block = float('-inf')
    
    # We compute N_v cumulatively in Python to be safe/explicit,
    # though a window function in SQL is also possible.
    cumulative_nv = 0

    # Handle output stream
    f_out = open(output_file, 'w', newline='', encoding='utf-8') if output_file else sys.stdout

    try:
        writer = csv.writer(f_out)
        writer.writerow(["sequence", "ht_hex", "nv", "sealed_at"])

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (system_id,))
                
                for block_number, ht_hex, proof_count, sealed_at in cur:
                    cumulative_nv += proof_count
                    
                    writer.writerow([
                        block_number,
                        ht_hex,
                        cumulative_nv,
                        sealed_at
                    ])
                    
                    rows_extracted += 1
                    min_block = min(min_block, block_number)
                    max_block = max(max_block, block_number)
                    
    finally:
        if output_file:
            f_out.close()

    if rows_extracted > 0:
        min_nv = 0  # Starts at 0 effectively
        max_nv = cumulative_nv
        
        # Log stats to stderr so it doesn't pollute stdout if used
        sys.stderr.write(f"Extracted {rows_extracted} rows.\n")
        sys.stderr.write(f"Block range: {min_block} -> {max_block}\n")
        sys.stderr.write(f"Nv range:    {min_nv} -> {max_nv}\n")
    else:
        sys.stderr.write(f"No data found for system_id={system_id}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract H_t data for scaling analysis.")
    parser.add_argument("--output", "-o", help="Output CSV file path (default: stdout)")
    parser.add_argument("--system-id", default="1", help="System ID to filter blocks (default: 1)")
    
    args = parser.parse_args()
    
    try:
        extract_ht_data(args.output, args.system_id)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
