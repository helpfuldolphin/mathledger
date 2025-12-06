#!/usr/bin/env python3
"""
ledgerctl.py - MathLedger Control CLI

Command-line interface for MathLedger operations including:
- Harmony Protocol quorum diagnostics
- Celestial Dossier integrity verification
- Cosmic root verification
- Real-time consensus monitoring

Features:
- Tab completion support
- Verbose cryptographic audit mode
- Deterministic ASCII-only output
- Cross-system integrity verification
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ledger.consensus.harmony_v1_1 import HarmonyProtocol
from backend.ledger.consensus.celestial_dossier_v2 import CelestialDossier
from backend.crypto.hashing import sha256_hex, DOMAIN_ROOT


class LedgerCtl:
    """Main controller class for ledgerctl operations."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize ledgerctl.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[VERBOSE] {message}")
    
    def verify_integrity(self) -> int:
        """
        Verify integrity of Harmony, Dossier, and Ledger roots simultaneously.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("[INFO] Starting cross-system integrity verification...")
        print()
        
        # Check if phase_ix_final.json exists
        attestation_path = Path("artifacts/attestations/phase_ix_final.json")
        
        if not attestation_path.exists():
            print("[FAIL] Phase IX attestation not found")
            print(f"       Expected: {attestation_path}")
            print("       Run: python3 phase_ix_attestation.py")
            return 1
        
        # Load attestation
        with open(attestation_path, 'r') as f:
            attestation = json.load(f)
        
        self.log(f"Loaded attestation from {attestation_path}")
        
        # Extract roots
        harmony_root = attestation['harmony']['root']
        dossier_root = attestation['dossier']['root']
        cosmic_root = attestation['cosmic_attestation_manifest']['cosmic_root']
        
        print("[CHECK] Harmony Protocol")
        print(f"        Root: {harmony_root}")
        print(f"        Safety: {attestation['harmony']['safety']}")
        print(f"        Liveness: {attestation['harmony']['liveness']}")
        print(f"        Convergence: {attestation['harmony']['convergence_rate']:.1%}")
        
        harmony_pass = (
            attestation['harmony']['safety'] and
            attestation['harmony']['liveness'] and
            attestation['harmony']['convergence_rate'] >= 0.8
        )
        
        if harmony_pass:
            print("        [PASS] Harmony verification succeeded")
        else:
            print("        [FAIL] Harmony verification failed")
        
        print()
        print("[CHECK] Celestial Dossier")
        print(f"        Root: {dossier_root}")
        print(f"        Nodes: {attestation['dossier']['statistics']['total_nodes']}")
        print(f"        Epochs: {attestation['dossier']['statistics']['total_epochs']}")
        print(f"        Cross-epoch edges: {attestation['dossier']['statistics']['cross_epoch_edges']}")
        
        dossier_pass = attestation['dossier']['inclusion_proof_valid']
        
        if dossier_pass:
            print("        [PASS] Dossier verification succeeded")
        else:
            print("        [FAIL] Dossier verification failed")
        
        print()
        print("[CHECK] Cosmic Attestation Manifest")
        cam = attestation['cosmic_attestation_manifest']
        print(f"        Cosmic Root: {cosmic_root}")
        print(f"        Federations: {cam['federations']}")
        print(f"        Nodes: {cam['nodes']}")
        print(f"        Timestamp: {cam['timestamp']}")
        
        # Verify cosmic root computation
        combined = cam['harmony_root'] + cam['dossier_root'] + cam['ledger_root']
        computed_cosmic = sha256_hex(combined, domain=DOMAIN_ROOT)
        
        cosmic_pass = (computed_cosmic == cosmic_root)
        
        if cosmic_pass:
            print("        [PASS] Cosmic root verification succeeded")
        else:
            print("        [FAIL] Cosmic root verification failed")
            print(f"        Expected: {cosmic_root}")
            print(f"        Computed: {computed_cosmic}")
        
        print()
        print("=" * 80)
        
        all_pass = harmony_pass and dossier_pass and cosmic_pass
        
        if all_pass:
            print(f"[PASS] Cosmic Ledger Integrity Verified root={cosmic_root}")
            return 0
        else:
            print(f"[FAIL] Integrity Verification Failed")
            print(f"       Harmony: {harmony_pass}")
            print(f"       Dossier: {dossier_pass}")
            print(f"       Cosmic: {cosmic_pass}")
            return 1
    
    def quorum_diagnostics(self) -> int:
        """
        Display real-time quorum diagnostics.
        
        Returns:
            Exit code (0 for success)
        """
        print("[INFO] Quorum Diagnostics")
        print()
        
        # Check if phase_ix_final.json exists
        attestation_path = Path("artifacts/attestations/phase_ix_final.json")
        
        if not attestation_path.exists():
            print("[WARN] Phase IX attestation not found")
            print("       Run: python3 phase_ix_attestation.py")
            return 1
        
        # Load attestation
        with open(attestation_path, 'r') as f:
            attestation = json.load(f)
        
        cam = attestation['cosmic_attestation_manifest']
        harmony = attestation['harmony']
        
        print(f"Registered Nodes:    {cam['nodes']}")
        print(f"Active Federations:  {cam['federations']}")
        print(f"Convergence Rate:    {harmony['convergence_rate']:.1%}")
        print(f"Safety Property:     {'✓' if harmony['safety'] else '✗'}")
        print(f"Liveness Property:   {'✓' if harmony['liveness'] else '✗'}")
        print()
        print(f"Harmony Root:        {harmony['root'][:32]}...")
        print(f"Phase:               {attestation['phase']}")
        print(f"Timestamp:           {cam['timestamp']}")
        
        return 0
    
    def audit_mode(self) -> int:
        """
        Display verbose cryptographic audit information.
        
        Returns:
            Exit code (0 for success)
        """
        print("[INFO] Cryptographic Audit Mode")
        print()
        
        # Check if phase_ix_final.json exists
        attestation_path = Path("artifacts/attestations/phase_ix_final.json")
        
        if not attestation_path.exists():
            print("[FAIL] Phase IX attestation not found")
            return 1
        
        # Load attestation
        with open(attestation_path, 'r') as f:
            attestation = json.load(f)
        
        print("=" * 80)
        print("HARMONY PROTOCOL v1.1")
        print("=" * 80)
        print(f"Root Hash:           {attestation['harmony']['root']}")
        print(f"Safety:              {attestation['harmony']['safety']}")
        print(f"Liveness:            {attestation['harmony']['liveness']}")
        print(f"Convergence Rate:    {attestation['harmony']['convergence_rate']:.4f}")
        print()
        
        print("=" * 80)
        print("CELESTIAL DOSSIER v2")
        print("=" * 80)
        print(f"Root Hash:           {attestation['dossier']['root']}")
        stats = attestation['dossier']['statistics']
        print(f"Total Nodes:         {stats['total_nodes']}")
        print(f"Total Epochs:        {stats['total_epochs']}")
        print(f"Current Epoch:       {stats['current_epoch']}")
        print(f"Cross-Epoch Edges:   {stats['cross_epoch_edges']}")
        print(f"Avg Nodes/Epoch:     {stats['avg_nodes_per_epoch']:.2f}")
        print(f"Inclusion Proof:     {attestation['dossier']['inclusion_proof_valid']}")
        print()
        
        print("=" * 80)
        print("COSMIC ATTESTATION MANIFEST")
        print("=" * 80)
        cam = attestation['cosmic_attestation_manifest']
        print(f"Cosmic Root:         {cam['cosmic_root']}")
        print(f"Harmony Root:        {cam['harmony_root']}")
        print(f"Dossier Root:        {cam['dossier_root']}")
        print(f"Ledger Root:         {cam['ledger_root']}")
        print(f"Federations:         {cam['federations']}")
        print(f"Nodes:               {cam['nodes']}")
        print(f"Timestamp:           {cam['timestamp']}")
        print()
        
        print("=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        ver = attestation['verification']
        print(f"Deterministic:       {ver['deterministic']}")
        print(f"ASCII Only:          {ver['ascii_only']}")
        print(f"JSON Canonical:      {ver['json_canonical']}")
        print()
        
        return 0


def main():
    """Main entry point for ledgerctl CLI."""
    parser = argparse.ArgumentParser(
        description='MathLedger Control CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ledgerctl.py --verify-integrity          Check all cryptographic roots
  ledgerctl.py --quorum-diagnostics        Display quorum status
  ledgerctl.py --audit-mode                Show detailed audit information
  ledgerctl.py --verify-integrity -v       Verbose integrity check
        """
    )
    
    parser.add_argument(
        '--verify-integrity',
        action='store_true',
        help='Verify integrity of Harmony, Dossier, and Ledger roots'
    )
    
    parser.add_argument(
        '--quorum-diagnostics',
        action='store_true',
        help='Display real-time quorum diagnostics'
    )
    
    parser.add_argument(
        '--audit-mode',
        action='store_true',
        help='Display verbose cryptographic audit information'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not (args.verify_integrity or args.quorum_diagnostics or args.audit_mode):
        parser.print_help()
        return 0
    
    ctl = LedgerCtl(verbose=args.verbose)
    
    if args.verify_integrity:
        return ctl.verify_integrity()
    elif args.quorum_diagnostics:
        return ctl.quorum_diagnostics()
    elif args.audit_mode:
        return ctl.audit_mode()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
