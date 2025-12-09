#!/usr/bin/env python3
"""
Attestation Chain Verification CLI
===================================

Verifies attestation chains for experiment runs, including:
- Dual-root attestation integrity (R_t, U_t, H_t)
- TDA pipeline configuration consistency
- Hard Gate decision binding
- Chain linkage integrity

Usage:
    python scripts/verify_attestation_chain.py <attestation_dir>
    python scripts/verify_attestation_chain.py --strict-tda <attestation_dir>
    python scripts/verify_attestation_chain.py --help

Exit Codes:
    0: Verification passed
    1: Attestation integrity failure
    2: Merkle root mismatch
    3: Chain linkage broken
    4: TDA-Ledger Divergence Detected

Example:
    # Verify attestation chain in artifacts directory
    python scripts/verify_attestation_chain.py artifacts/phase_ii/experiment_001/
    
    # Strict mode (fail on any TDA drift)
    python scripts/verify_attestation_chain.py --strict-tda artifacts/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attestation.chain_verifier import (
    AttestationVerificationError,
    ExperimentBlock,
    verify_experiment_attestation_chain,
)


def load_attestation_blocks(attestation_dir: Path) -> List[ExperimentBlock]:
    """
    Load experiment blocks from attestation directory.
    
    Expected structure:
        attestation_dir/
            run_001/attestation.json
            run_002/attestation.json
            ...
    
    Args:
        attestation_dir: Directory containing attestation files
        
    Returns:
        List of ExperimentBlock objects sorted by block_number
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If attestation files are malformed
    """
    if not attestation_dir.exists():
        raise FileNotFoundError(f"Attestation directory not found: {attestation_dir}")
    
    blocks = []
    
    # Find all attestation.json files
    attestation_files = sorted(attestation_dir.glob("*/attestation.json"))
    
    if not attestation_files:
        # Try direct attestation.json
        direct_file = attestation_dir / "attestation.json"
        if direct_file.exists():
            attestation_files = [direct_file]
        else:
            raise ValueError(f"No attestation files found in {attestation_dir}")
    
    for attestation_file in attestation_files:
        with open(attestation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract required fields
        try:
            block = ExperimentBlock(
                run_id=data.get("run_id", attestation_file.parent.name),
                experiment_id=data.get("experiment_id", "unknown"),
                reasoning_root=data["R_t"],
                ui_root=data["U_t"],
                composite_root=data["H_t"],
                tda_pipeline_hash=data["tda_pipeline_hash"],
                tda_config=data["tda_config"],
                gate_decisions=data.get("gate_decisions"),
                prev_block_hash=data.get("prev_block_hash"),
                block_number=data.get("block_number", 0),
            )
            blocks.append(block)
        except KeyError as e:
            raise ValueError(
                f"Missing required field in {attestation_file}: {e}"
            )
    
    # Sort by block number
    blocks.sort(key=lambda b: b.block_number)
    
    return blocks


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify attestation chain with TDA pipeline binding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "attestation_dir",
        type=Path,
        help="Directory containing attestation files"
    )
    
    parser.add_argument(
        "--strict-tda",
        action="store_true",
        help="Fail verification on TDA configuration drift (exit code 4)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load blocks
        if args.verbose:
            print(f"Loading attestation blocks from {args.attestation_dir}...")
        
        blocks = load_attestation_blocks(args.attestation_dir)
        
        if args.verbose:
            print(f"Loaded {len(blocks)} block(s)")
            for i, block in enumerate(blocks):
                print(f"  Block {i}: run_id={block.run_id}, block_number={block.block_number}")
        
        # Verify chain
        if args.verbose:
            print("\nVerifying attestation chain...")
        
        result = verify_experiment_attestation_chain(
            blocks,
            strict_tda=args.strict_tda
        )
        
        # Report results
        if result.is_valid:
            print("✅ Attestation chain verification PASSED")
            
            if result.divergences and not args.strict_tda:
                print("\n⚠️  Warnings:")
                for divergence in result.divergences:
                    print(f"\n{divergence}")
            
            sys.exit(int(AttestationVerificationError.SUCCESS))
        else:
            print("❌ Attestation chain verification FAILED", file=sys.stderr)
            print(f"   Error: {result.error_message}", file=sys.stderr)
            
            if result.divergences:
                print("\n📊 TDA Configuration Divergences:", file=sys.stderr)
                for divergence in result.divergences:
                    print(f"\n{divergence}", file=sys.stderr)
            
            sys.exit(int(result.error_code))
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
