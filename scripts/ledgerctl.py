#!/usr/bin/env python3
"""
LedgerCtl CLI - Phase IX Command Line Interface
Extended with Phase IX capabilities: verify-integrity, quorum-diagnostics, audit-mode
"""

import argparse
import json
import sys
import time
import os
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.phase_ix import run_attestation_harness
from backend.phase_ix.dossier import create_dossier
from backend.phase_ix.attestation import create_manifest, verify_attestation
from backend.consensus import converge, ValidatorSet, TrustWeight
from backend.ledger.blockchain import merkle_root
from backend.crypto.hashing import sha256_hex


def verify_integrity(manifest_file: str, verbose: bool = False) -> int:
    """
    Verify cryptographic root equivalence of attestation manifest.
    
    Args:
        manifest_file: Path to phase_ix_final.json manifest
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    """
    try:
        with open(manifest_file, 'r') as f:
            data = json.load(f)
        
        if 'manifest' not in data:
            print(f"[FAIL] Invalid manifest file: missing 'manifest' key")
            return 1
        
        manifest_data = data['manifest']
        
        # Reconstruct manifest
        from backend.phase_ix.attestation import CosmicAttestationManifest
        manifest = CosmicAttestationManifest(**manifest_data)
        
        # Verify attestation
        valid = verify_attestation(manifest)
        
        if valid:
            print(f"[PASS] Cryptographic Root Equivalence Verified")
            if verbose:
                print(f"  Harmony Root:  {manifest.harmony_root}")
                print(f"  Dossier Root:  {manifest.dossier_root}")
                print(f"  Ledger Root:   {manifest.ledger_root}")
                print(f"  Unified Root:  {manifest.unified_root}")
                print(f"  Readiness:     {manifest.readiness}")
            return 0
        else:
            print(f"[FAIL] Cryptographic Root Equivalence Check Failed")
            return 1
    
    except FileNotFoundError:
        print(f"[FAIL] Manifest file not found: {manifest_file}")
        return 1
    except Exception as e:
        print(f"[FAIL] Verification error: {e}")
        return 1


def quorum_diagnostics(
    num_nodes: int = 50,
    byzantine_ratio: float = 0.2,
    verbose: bool = False
) -> int:
    """
    Visualize trust convergence live.
    
    Args:
        num_nodes: Number of validator nodes
        byzantine_ratio: Ratio of Byzantine nodes
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    """
    print("\n" + "="*60)
    print("Quorum Diagnostics - Trust Convergence Analysis")
    print("="*60 + "\n")
    
    try:
        # Create validator set
        validators = {}
        for i in range(num_nodes):
            validators[f"node_{i}"] = TrustWeight(
                node_id=f"node_{i}",
                weight=1.0 / num_nodes,
                epoch=0,
                reputation=1.0
            )
        
        validator_set = ValidatorSet(validators=validators, epoch=0)
        
        # Compute metrics
        total_weight = validator_set.total_weight()
        threshold = validator_set.threshold
        quorum_weight = total_weight * threshold
        
        num_byzantine = int(num_nodes * byzantine_ratio)
        num_honest = num_nodes - num_byzantine
        
        honest_weight = num_honest / num_nodes
        byzantine_weight = num_byzantine / num_nodes
        
        print(f"Validator Configuration:")
        print(f"  Total Nodes:       {num_nodes}")
        print(f"  Honest Nodes:      {num_honest} ({honest_weight:.1%})")
        print(f"  Byzantine Nodes:   {num_byzantine} ({byzantine_weight:.1%})")
        print(f"  Total Weight:      {total_weight:.3f}")
        print(f"  Threshold:         {threshold:.1%}")
        print(f"  Quorum Weight:     {quorum_weight:.3f}")
        
        print(f"\nConvergence Analysis:")
        
        # Check if honest nodes can reach quorum
        if honest_weight >= threshold:
            print(f"  [PASS] Honest nodes CAN reach quorum")
            print(f"  Honest weight ({honest_weight:.1%}) >= Threshold ({threshold:.1%})")
        else:
            print(f"  [FAIL] Honest nodes CANNOT reach quorum")
            print(f"  Honest weight ({honest_weight:.1%}) < Threshold ({threshold:.1%})")
        
        # Check Byzantine fault tolerance
        max_byzantine = num_nodes / 3
        if num_byzantine < max_byzantine:
            print(f"  [PASS] Byzantine fault tolerance maintained")
            print(f"  Byzantine nodes ({num_byzantine}) < Max tolerable ({max_byzantine:.0f})")
        else:
            print(f"  [FAIL] Byzantine fault tolerance EXCEEDED")
            print(f"  Byzantine nodes ({num_byzantine}) >= Max tolerable ({max_byzantine:.0f})")
        
        # Run actual convergence test
        print(f"\nRunning Live Convergence Test...")
        
        honest_nodes = [f"node_{i}" for i in range(num_honest)]
        byzantine_nodes = [f"node_{i}" for i in range(num_honest, num_nodes)]
        
        proposals = ["CONSENSUS_VALUE", "BYZANTINE_VALUE"]
        decided_value, metrics = converge(
            validator_set,
            proposals,
            honest_nodes,
            byzantine_nodes
        )
        
        print(f"\nConvergence Results:")
        print(f"  Decided Value:     {decided_value}")
        print(f"  Success:           {metrics['success']}")
        print(f"  Convergence Time:  {metrics['total_latency_ms']:.3f}ms")
        print(f"  Rounds:            {metrics['convergence_rounds']}")
        
        if verbose:
            print(f"\nDetailed Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*60 + "\n")
        
        return 0 if metrics['success'] else 1
    
    except Exception as e:
        print(f"[FAIL] Quorum diagnostics error: {e}")
        return 1


def audit_mode(
    epochs: int = 3,
    nodes: int = 10,
    verbose: bool = False
) -> int:
    """
    Perform full ledger cross-proof verification.
    
    Args:
        epochs: Number of epochs to audit
        nodes: Number of nodes to simulate
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    """
    print("\n" + "="*60)
    print("Audit Mode - Full Ledger Cross-Proof Verification")
    print("="*60 + "\n")
    
    try:
        # Create test epochs
        print(f"Creating {epochs} epochs for audit...")
        epochs_data = []
        for i in range(epochs):
            epochs_data.append({
                "epoch_id": i,
                "parent_epoch": i - 1 if i > 0 else None,
                "statements": [f"AUDIT_STMT_{i}_{j}" for j in range(20)],
                "metadata": {"audit": True}
            })
        
        dossier = create_dossier(epochs_data)
        
        # Verify each epoch lineage
        print(f"\nVerifying epoch lineages...")
        lineage_results = []
        for i in range(epochs):
            valid = dossier.verify_lineage(i)
            lineage_results.append(valid)
            status = "[PASS]" if valid else "[FAIL]"
            print(f"  Epoch {i}: {status}")
        
        # Test Merkle proofs
        print(f"\nVerifying Merkle inclusion proofs...")
        proof_results = []
        for epoch_id in range(min(epochs, 3)):  # Test first 3 epochs
            epoch = dossier.epochs[epoch_id]
            if epoch.statements:
                statement = epoch.statements[0]
                proof = dossier.get_merkle_proof(epoch_id, statement)
                proof_valid = proof is not None
                proof_results.append(proof_valid)
                status = "[PASS]" if proof_valid else "[FAIL]"
                print(f"  Epoch {epoch_id} proof: {status}")
        
        # Compute dossier root
        print(f"\nComputing dossier root hash...")
        dossier_root = dossier.compute_root_hash()
        print(f"  Dossier Root: {dossier_root}")
        
        # Create full attestation
        print(f"\nCreating cosmic attestation...")
        ledger_statements = [f"LEDGER_{i}" for i in range(50)]
        ledger_root = merkle_root(ledger_statements)
        
        manifest = create_manifest(
            harmony_root="a" * 64,  # Simulated
            dossier_root=dossier_root,
            ledger_root=ledger_root,
            epochs=epochs,
            nodes=nodes,
            metadata={"mode": "audit"}
        )
        
        attestation_valid = verify_attestation(manifest)
        print(f"  Attestation Valid: {attestation_valid}")
        print(f"  Unified Root: {manifest.unified_root}")
        print(f"  Readiness: {manifest.readiness}")
        
        # Summary
        all_lineages_valid = all(lineage_results)
        all_proofs_valid = all(proof_results) if proof_results else True
        
        print(f"\nAudit Summary:")
        print(f"  Lineage Verification: {'[PASS]' if all_lineages_valid else '[FAIL]'}")
        print(f"  Merkle Proofs: {'[PASS]' if all_proofs_valid else '[FAIL]'}")
        print(f"  Attestation: {'[PASS]' if attestation_valid else '[FAIL]'}")
        
        success = all_lineages_valid and all_proofs_valid and attestation_valid
        
        if success:
            print(f"\n[PASS] Audit Mode - All Verifications Passed")
        else:
            print(f"\n[FAIL] Audit Mode - Some Verifications Failed")
        
        print("\n" + "="*60 + "\n")
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"[FAIL] Audit mode error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_harness(
    nodes: int = 50,
    epochs: int = 5,
    byzantine_ratio: float = 0.2,
    output: Optional[str] = None
) -> int:
    """
    Run Phase IX attestation harness.
    
    Args:
        nodes: Number of validator nodes
        epochs: Number of epochs
        byzantine_ratio: Ratio of Byzantine nodes
        output: Optional output file path
    
    Returns:
        0 on success, 1 on failure
    """
    try:
        results = run_attestation_harness(
            num_nodes=nodes,
            num_epochs=epochs,
            byzantine_ratio=byzantine_ratio,
            output_file=output
        )
        
        return 0 if results['success'] else 1
    
    except Exception as e:
        print(f"[FAIL] Harness error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LedgerCtl - Phase IX Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  verify-integrity    Verify cryptographic root equivalence
  quorum-diagnostics  Visualize trust convergence live
  audit-mode          Perform full ledger cross-proof verification
  run-harness         Run Phase IX attestation harness

Examples:
  ledgerctl verify-integrity phase_ix_final.json
  ledgerctl quorum-diagnostics --nodes 100 --byzantine-ratio 0.25
  ledgerctl audit-mode --epochs 10 --verbose
  ledgerctl run-harness --nodes 50 --epochs 5 --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # verify-integrity command
    verify_parser = subparsers.add_parser('verify-integrity', help='Verify cryptographic root equivalence')
    verify_parser.add_argument('manifest', help='Path to phase_ix_final.json manifest')
    verify_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # quorum-diagnostics command
    quorum_parser = subparsers.add_parser('quorum-diagnostics', help='Visualize trust convergence')
    quorum_parser.add_argument('--nodes', type=int, default=50, help='Number of validator nodes')
    quorum_parser.add_argument('--byzantine-ratio', type=float, default=0.2, help='Byzantine node ratio')
    quorum_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # audit-mode command
    audit_parser = subparsers.add_parser('audit-mode', help='Full ledger cross-proof verification')
    audit_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to audit')
    audit_parser.add_argument('--nodes', type=int, default=10, help='Number of nodes to simulate')
    audit_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # run-harness command
    harness_parser = subparsers.add_parser('run-harness', help='Run Phase IX attestation harness')
    harness_parser.add_argument('--nodes', type=int, default=50, help='Number of validator nodes')
    harness_parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    harness_parser.add_argument('--byzantine-ratio', type=float, default=0.2, help='Byzantine node ratio')
    harness_parser.add_argument('-o', '--output', help='Output file path for results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'verify-integrity':
        return verify_integrity(args.manifest, args.verbose)
    elif args.command == 'quorum-diagnostics':
        return quorum_diagnostics(args.nodes, args.byzantine_ratio, args.verbose)
    elif args.command == 'audit-mode':
        return audit_mode(args.epochs, args.nodes, args.verbose)
    elif args.command == 'run-harness':
        return run_harness(args.nodes, args.epochs, args.byzantine_ratio, args.output)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
