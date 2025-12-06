#!/usr/bin/env python3
"""
Synthetic FO Data Generator (Small Test)
========================================
"""
import csv
import random
import os
from datetime import datetime, timedelta

def generate_small(output_file, num_blocks=50):
    print(f"Generating {num_blocks} synthetic blocks for validation...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "ht_hex", "nv", "sealed_at"])
        current_nv = 0
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(1, num_blocks + 1):
            current_nv += random.randint(0, 50)
            ht_hex = f"{random.getrandbits(256):064x}"
            writer.writerow([i, ht_hex, current_nv, (start_time + timedelta(minutes=i)).isoformat()])
    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    generate_small("reports/test_ht_dynamics.csv")
