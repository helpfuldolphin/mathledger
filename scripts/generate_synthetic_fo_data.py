#!/usr/bin/env python3
"""
Synthetic FO Data Generator
===========================

Generates a synthetic dataset for Hash State Drift analysis when the live database
is unavailable. Mimics a healthy SHA-256 avalanche behavior.

Output: reports/ht_dynamics.csv
"""

import csv
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_data(output_file, num_blocks=2000):
    print(f"Generating {num_blocks} synthetic blocks...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "ht_hex", "nv", "sealed_at"])
        
        current_nv = 0
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        
        for i in range(1, num_blocks + 1):
            # Simulate proofs added (random 0-50)
            proofs_in_block = random.randint(0, 50)
            current_nv += proofs_in_block
            
            # Simulate H_t (random 256-bit hex)
            # In a healthy system, H_t is effectively uniform random
            ht_int = random.getrandbits(256)
            ht_hex = f"{ht_int:064x}"
            
            # Simulate timestamp
            block_time = start_time + timedelta(minutes=i)
            
            writer.writerow([
                i,
                ht_hex,
                current_nv,
                block_time.isoformat()
            ])
            
    print(f"Done. Output saved to {output_file}")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    generate_synthetic_data("reports/ht_dynamics.csv")
