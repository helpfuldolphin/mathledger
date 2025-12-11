#!/usr/bin/env python3
"""
# DEMO-SCAFFOLD

PQ Activation Simulator

This script simulates the PQ activation process for training and testing purposes.
It creates synthetic blocks, simulates epoch transitions, and generates realistic
log output to help operators prepare for activation day.

WARNING: This is a DEMO-SCAFFOLD implementation. It does NOT interact with real
blockchain state. It is designed solely for operator training and readiness validation.

Usage:
    python3 scripts/pq_activation_simulator.py --start-block 9990 --activation-block 10000

Author: Manus-H
Date: 2025-12-10
"""

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

# Color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

@dataclass
class SimulatedBlock:
    """Simulated block structure."""
    block_number: int
    prev_hash: str
    merkle_root: str
    timestamp: float
    statements: List[str]
    # PQ fields (optional, present only after activation)
    pq_algorithm: Optional[int] = None
    pq_merkle_root: Optional[str] = None
    pq_prev_hash: Optional[str] = None
    dual_commitment: Optional[str] = None

def generate_random_hash() -> str:
    """Generate a random hash for simulation."""
    return "0x" + hashlib.sha256(str(random.random()).encode()).hexdigest()

def simulate_legacy_block(block_number: int, prev_hash: str) -> SimulatedBlock:
    """Simulate a legacy block (pre-activation)."""
    statements = [f"statement_{block_number}_{i}" for i in range(random.randint(5, 20))]
    merkle_root = generate_random_hash()
    
    return SimulatedBlock(
        block_number=block_number,
        prev_hash=prev_hash,
        merkle_root=merkle_root,
        timestamp=time.time(),
        statements=statements,
    )

def simulate_pq_block(block_number: int, prev_hash: str, prev_pq_hash: Optional[str]) -> SimulatedBlock:
    """Simulate a PQ block (post-activation with dual commitment)."""
    statements = [f"statement_{block_number}_{i}" for i in range(random.randint(5, 20))]
    merkle_root = generate_random_hash()
    pq_merkle_root = generate_random_hash()
    
    # Simulate dual commitment (hash of legacy + PQ roots)
    dual_input = f"{merkle_root}{pq_merkle_root}".encode()
    dual_commitment = "0x" + hashlib.sha256(dual_input).hexdigest()
    
    return SimulatedBlock(
        block_number=block_number,
        prev_hash=prev_hash,
        merkle_root=merkle_root,
        timestamp=time.time(),
        statements=statements,
        pq_algorithm=0x01,  # SHA3-256
        pq_merkle_root=pq_merkle_root,
        pq_prev_hash=prev_pq_hash or generate_random_hash(),
        dual_commitment=dual_commitment,
    )

def print_block_log(block: SimulatedBlock, is_activation: bool = False) -> None:
    """Print simulated block sealing log."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    if block.pq_algorithm is None:
        # Legacy block
        print(f"{timestamp} INFO [consensus] Sealed block {block.block_number}")
        print(f"  merkle_root: {block.merkle_root[:18]}...")
        print(f"  statements: {len(block.statements)}")
    else:
        # PQ block
        if is_activation:
            print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}EPOCH ACTIVATION EVENT{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")
            print(f"{timestamp} {Colors.GREEN}INFO [epoch] Activating new epoch:{Colors.END}")
            print(f"  algorithm: SHA3-256 (0x{block.pq_algorithm:02x})")
            print(f"  rule_version: v2-dual-required")
            print(f"  start_block: {block.block_number}")
            print()
        
        print(f"{timestamp} {Colors.GREEN}INFO [consensus] Sealed block {block.block_number} with dual commitment{Colors.END}")
        print(f"  legacy_hash: {block.merkle_root[:18]}...")
        print(f"  pq_hash: {block.pq_merkle_root[:18]}...")
        print(f"  dual_commitment: {block.dual_commitment[:18]}...")
        print(f"  statements: {len(block.statements)}")

def simulate_drift_event(block_number: int, event_type: str) -> None:
    """Simulate a drift radar alert (for training purposes)."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    if event_type == "algorithm_mismatch":
        print(f"\n{Colors.RED}{timestamp} CRITICAL [DRIFT_RADAR] Algorithm mismatch detected!{Colors.END}")
        print(f"{Colors.RED}  Block {block_number} uses algorithm 0x02 but epoch expects 0x01{Colors.END}")
        print(f"{Colors.RED}  This is a CRITICAL consensus violation!{Colors.END}\n")
    elif event_type == "missing_dual_commitment":
        print(f"\n{Colors.YELLOW}{timestamp} HIGH [DRIFT_RADAR] Missing dual commitment!{Colors.END}")
        print(f"{Colors.YELLOW}  Block {block_number} has PQ fields but missing dual_commitment{Colors.END}\n")

def run_simulation(start_block: int, activation_block: int, inject_drift: bool = False) -> None:
    """Run the full activation simulation."""
    print(f"{Colors.BOLD}PQ ACTIVATION SIMULATOR{Colors.END}")
    print(f"Start block: {start_block}")
    print(f"Activation block: {activation_block}")
    print(f"Inject drift events: {inject_drift}")
    print(f"{Colors.YELLOW}NOTE: This is a DEMO-SCAFFOLD simulation, not real blockchain data{Colors.END}\n")
    
    blocks: List[SimulatedBlock] = []
    prev_hash = generate_random_hash()
    prev_pq_hash = None
    
    # Simulate blocks leading up to activation
    print(f"{Colors.BOLD}Phase 1: Pre-Activation (Legacy Blocks){Colors.END}\n")
    for block_num in range(start_block, activation_block):
        block = simulate_legacy_block(block_num, prev_hash)
        blocks.append(block)
        print_block_log(block)
        prev_hash = block.merkle_root
        time.sleep(0.3)
    
    # Simulate activation
    print(f"\n{Colors.BOLD}Phase 2: Activation Event{Colors.END}\n")
    time.sleep(1)
    
    # First PQ block
    first_pq_block = simulate_pq_block(activation_block, prev_hash, prev_pq_hash)
    blocks.append(first_pq_block)
    print_block_log(first_pq_block, is_activation=True)
    prev_hash = first_pq_block.merkle_root
    prev_pq_hash = first_pq_block.pq_merkle_root
    time.sleep(0.5)
    
    # Simulate post-activation blocks
    print(f"\n{Colors.BOLD}Phase 3: Post-Activation (Dual-Commitment Blocks){Colors.END}\n")
    for i in range(1, 6):
        block_num = activation_block + i
        block = simulate_pq_block(block_num, prev_hash, prev_pq_hash)
        blocks.append(block)
        
        # Inject drift event for training
        if inject_drift and i == 3:
            simulate_drift_event(block_num, "algorithm_mismatch")
        
        print_block_log(block)
        prev_hash = block.merkle_root
        prev_pq_hash = block.pq_merkle_root
        time.sleep(0.3)
    
    # Summary
    print(f"\n{Colors.BOLD}Simulation Complete{Colors.END}")
    print(f"Total blocks simulated: {len(blocks)}")
    print(f"Legacy blocks: {sum(1 for b in blocks if b.pq_algorithm is None)}")
    print(f"PQ blocks: {sum(1 for b in blocks if b.pq_algorithm is not None)}")
    
    # Save simulation data
    output_file = f"pq_simulation_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(b) for b in blocks], f, indent=2)
    
    print(f"\nSimulation data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="PQ Activation Simulator (DEMO-SCAFFOLD)")
    parser.add_argument(
        "--start-block",
        type=int,
        default=9990,
        help="Block number to start simulation",
    )
    parser.add_argument(
        "--activation-block",
        type=int,
        default=10000,
        help="Block number where PQ epoch activates",
    )
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Inject a simulated drift event for training",
    )
    
    args = parser.parse_args()
    
    if args.activation_block <= args.start_block:
        print(f"{Colors.RED}Error: activation-block must be greater than start-block{Colors.END}")
        sys.exit(1)
    
    run_simulation(args.start_block, args.activation_block, args.inject_drift)

if __name__ == "__main__":
    main()
