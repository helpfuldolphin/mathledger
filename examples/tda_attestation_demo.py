#!/usr/bin/env python3
"""
TDA Pipeline Attestation Demo
==============================

Demonstrates how to:
1. Create attestation blocks with TDA pipeline hashes
2. Build an attestation chain
3. Verify the chain
4. Detect TDA configuration drift

This example shows the integration pattern for RFL and U2 runners.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attestation.experiment_integration import (
    create_rfl_attestation_block,
    save_attestation_block,
)
from attestation.chain_verifier import verify_experiment_attestation_chain


def demo_valid_chain():
    """Demonstrate a valid attestation chain."""
    print("=" * 60)
    print("DEMO 1: Valid Attestation Chain")
    print("=" * 60)
    
    # Simulate RFL config
    rfl_config = {
        "bounds": {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
        },
        "verifier": {
            "tier": "tier1",
            "timeout": 10.0,
            "budget": None,
        },
        "curriculum_slice": {
            "slice_name": "arithmetic_simple",
        },
        "abstention_strategy": "conservative",
        "experiment_id": "demo_exp_001",
    }
    
    # Create two consecutive blocks with same config
    blocks = []
    prev_hash = None
    
    for i in range(2):
        block = create_rfl_attestation_block(
            run_id=f"run_{i+1:03d}",
            experiment_id="demo_exp_001",
            reasoning_events=[f"proof_{i+1}_a", f"proof_{i+1}_b"],
            ui_events=[f"event_{i+1}_a", f"event_{i+1}_b"],
            rfl_config=rfl_config,
            gate_decisions={"G1": "PASS", "G2": "PASS"},
            prev_block_hash=prev_hash,
            block_number=i,
        )
        blocks.append(block)
        prev_hash = block.compute_block_hash()
        
        print(f"\nBlock {i}:")
        print(f"  run_id: {block.run_id}")
        print(f"  H_t: {block.composite_root[:16]}...")
        print(f"  TDA hash: {block.tda_pipeline_hash[:16]}...")
        print(f"  block_hash: {block.compute_block_hash()[:16]}...")
    
    # Verify chain
    print("\nVerifying chain...")
    result = verify_experiment_attestation_chain(blocks, strict_tda=True)
    
    if result.is_valid:
        print("✅ Chain verification PASSED")
        print(f"   Exit code: {result.error_code}")
    else:
        print(f"❌ Chain verification FAILED")
        print(f"   Error: {result.error_message}")
        print(f"   Exit code: {result.error_code}")
    
    return result.is_valid


def demo_tda_divergence():
    """Demonstrate TDA configuration divergence detection."""
    print("\n" + "=" * 60)
    print("DEMO 2: TDA Configuration Drift Detection")
    print("=" * 60)
    
    # Base config
    rfl_config_1 = {
        "bounds": {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
        },
        "verifier": {
            "tier": "tier1",
            "timeout": 10.0,
            "budget": None,
        },
        "curriculum_slice": {
            "slice_name": "arithmetic_simple",
        },
        "abstention_strategy": "conservative",
        "experiment_id": "demo_exp_002",
    }
    
    # Modified config (drift!)
    rfl_config_2 = rfl_config_1.copy()
    rfl_config_2["bounds"] = rfl_config_1["bounds"].copy()
    rfl_config_2["bounds"]["max_breadth"] = 200  # DRIFT!
    
    # Create two blocks with different configs
    blocks = []
    configs = [rfl_config_1, rfl_config_2]
    prev_hash = None
    
    for i, config in enumerate(configs):
        block = create_rfl_attestation_block(
            run_id=f"run_{i+1:03d}",
            experiment_id="demo_exp_002",
            reasoning_events=[f"proof_{i+1}_a"],
            ui_events=[f"event_{i+1}_a"],
            rfl_config=config,
            gate_decisions={"G1": "PASS", "G2": "ABANDONED_TDA"},
            prev_block_hash=prev_hash,
            block_number=i,
        )
        blocks.append(block)
        prev_hash = block.compute_block_hash()
        
        print(f"\nBlock {i}:")
        print(f"  run_id: {block.run_id}")
        print(f"  max_breadth: {block.tda_config['max_breadth']}")
        print(f"  TDA hash: {block.tda_pipeline_hash[:16]}...")
    
    # Verify chain (strict mode)
    print("\nVerifying chain (strict TDA mode)...")
    result = verify_experiment_attestation_chain(blocks, strict_tda=True)
    
    if result.is_valid:
        print("✅ Chain verification PASSED")
        print(f"   Exit code: {result.error_code}")
    else:
        print(f"❌ Chain verification FAILED")
        print(f"   Error: {result.error_message}")
        print(f"   Exit code: {result.error_code} (TDA_DIVERGENCE)")
        
        if result.divergences:
            print("\n   Divergence details:")
            for div in result.divergences:
                print(f"     {div}")
    
    return not result.is_valid  # Should fail


def demo_hard_gate_binding():
    """Demonstrate Hard Gate decision cryptographic binding."""
    print("\n" + "=" * 60)
    print("DEMO 3: Hard Gate Decision Binding")
    print("=" * 60)
    
    rfl_config = {
        "bounds": {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
        },
        "verifier": {
            "tier": "tier1",
            "timeout": 10.0,
        },
        "curriculum_slice": {
            "slice_name": "arithmetic_simple",
        },
        "abstention_strategy": "conservative",
        "experiment_id": "demo_exp_003",
    }
    
    # Create block with gate decisions
    block1 = create_rfl_attestation_block(
        run_id="run_001",
        experiment_id="demo_exp_003",
        reasoning_events=["proof_a"],
        ui_events=["event_a"],
        rfl_config=rfl_config,
        gate_decisions={"G1": "PASS", "G2": "ABANDONED_TDA"},
        block_number=0,
    )
    
    # Create identical block but different gate decision
    block2 = create_rfl_attestation_block(
        run_id="run_001",  # Same run_id
        experiment_id="demo_exp_003",
        reasoning_events=["proof_a"],  # Same events
        ui_events=["event_a"],
        rfl_config=rfl_config,  # Same config
        gate_decisions={"G1": "PASS", "G2": "PASS"},  # DIFFERENT!
        block_number=0,
    )
    
    hash1 = block1.compute_block_hash()
    hash2 = block2.compute_block_hash()
    
    print("\nBlock 1 gate decisions:", block1.gate_decisions)
    print(f"  Block hash: {hash1[:16]}...")
    
    print("\nBlock 2 gate decisions:", block2.gate_decisions)
    print(f"  Block hash: {hash2[:16]}...")
    
    print(f"\n🔒 Hashes differ: {hash1 != hash2}")
    print("   Gate decisions are cryptographically bound!")
    
    return hash1 != hash2


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("TDA PIPELINE ATTESTATION DEMONSTRATION")
    print("=" * 60)
    
    results = []
    
    # Demo 1: Valid chain
    results.append(("Valid Chain", demo_valid_chain()))
    
    # Demo 2: TDA divergence
    results.append(("TDA Divergence Detection", demo_tda_divergence()))
    
    # Demo 3: Hard gate binding
    results.append(("Hard Gate Binding", demo_hard_gate_binding()))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(passed for _, passed in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
