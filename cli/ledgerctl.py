#!/usr/bin/env python3
"""
MathLedger Control CLI (ledgerctl)

Command-line interface for managing MathLedger federations and celestial consensus.

Commands:
    join-federation      Join a federation network
    list-federations     List known federations
    sync-celestial       Synchronize with celestial consensus
    verify-cosmic-root   Verify cosmic root integrity
    print-celestial-dossier  Print celestial dossier details

All commands output deterministic ASCII-only PASS lines.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.crypto.hashing import sha256_hex
from backend.ledger.v4.interfederation import (
    InterFederationGossip, Ed25519Signer, FederationIdentity,
    generate_pass_line as interfed_pass_line,
    compute_cosmic_root
)
from backend.ledger.v4.stellar import (
    StellarConsensus, QuorumLevel,
    generate_pass_line as stellar_pass_line,
    format_quorum_string
)
from tools.build_celestial_dossier import (
    CelestialDossier, FederatedDossier,
    load_federated_dossier,
    generate_pass_line as dossier_pass_line
)


class LedgerCtl:
    """MathLedger control interface."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.path.expanduser('~/.ledgerctl.json')
        self.config = self.load_config()
        
        # Initialize components
        self.federation_id = self.config.get('federation_id', 'local')
        self.signer = Ed25519Signer()
        self.gossip = InterFederationGossip(self.federation_id, self.signer)
        self.consensus = StellarConsensus(
            node_id='node-1',
            federation_id=self.federation_id,
            signer=self.signer
        )
        
        # Load known federations
        self.load_federations()
    
    def load_config(self) -> Dict:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_config(self) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_federations(self) -> None:
        """Load known federations from config."""
        federations = self.config.get('federations', [])
        for fed in federations:
            fed_id = fed.get('federation_id')
            pub_key_hex = fed.get('public_key')
            if fed_id and pub_key_hex:
                pub_key = bytes.fromhex(pub_key_hex)
                self.gossip.register_federation(
                    fed_id, pub_key, fed.get('metadata', {})
                )
    
    def join_federation(self, federation_id: str, 
                       endpoint: str,
                       public_key_hex: Optional[str] = None) -> bool:
        """
        Join a federation network.
        
        Args:
            federation_id: ID of federation to join
            endpoint: Network endpoint (e.g., "host:port")
            public_key_hex: Optional hex-encoded public key
            
        Returns:
            True if successful
        """
        # Generate or load public key
        if public_key_hex:
            pub_key = bytes.fromhex(public_key_hex)
        else:
            # Generate new key for federation
            temp_signer = Ed25519Signer()
            pub_key = temp_signer.public_key_bytes()
            public_key_hex = pub_key.hex()
        
        # Register federation
        metadata = {
            'endpoint': endpoint,
            'joined_at': time.time()
        }
        self.gossip.register_federation(federation_id, pub_key, metadata)
        
        # Update config
        if 'federations' not in self.config:
            self.config['federations'] = []
        
        fed_config = {
            'federation_id': federation_id,
            'endpoint': endpoint,
            'public_key': public_key_hex,
            'metadata': metadata
        }
        
        # Remove existing entry if present
        self.config['federations'] = [
            f for f in self.config['federations'] 
            if f.get('federation_id') != federation_id
        ]
        self.config['federations'].append(fed_config)
        
        self.save_config()
        
        # Output PASS line
        print(f"[PASS] Joined Federation federation={federation_id} endpoint={endpoint}")
        return True
    
    def list_federations(self, verbose: bool = False) -> List[Dict]:
        """
        List known federations.
        
        Args:
            verbose: Include detailed information
            
        Returns:
            List of federation dictionaries
        """
        federations = []
        
        for fed_id, identity in self.gossip.known_federations.items():
            trust = self.gossip.get_weighted_trust(fed_id)
            
            fed_info = {
                'federation_id': fed_id,
                'trust_score': round(trust, 3),
                'created_at': identity.created_at
            }
            
            if verbose:
                fed_info['public_key'] = identity.public_key.hex()[:16] + '...'
                fed_info['metadata'] = identity.metadata
            
            federations.append(fed_info)
        
        # Sort by trust score
        federations.sort(key=lambda f: f['trust_score'], reverse=True)
        
        # Print table
        print(f"[PASS] Listed Federations count={len(federations)}")
        print()
        print(f"{'Federation ID':<20} {'Trust':<8} {'Age (hours)':<12}")
        print("-" * 50)
        
        for fed in federations:
            age = (time.time() - fed['created_at']) / 3600
            print(f"{fed['federation_id']:<20} {fed['trust_score']:<8.3f} {age:<12.1f}")
        
        if verbose:
            print()
            print(json.dumps(federations, indent=2))
        
        return federations
    
    def sync_celestial(self, timeout_ms: float = 5000) -> bool:
        """
        Synchronize with celestial consensus.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if sync successful
        """
        start_time = time.time()
        
        # Get federation IDs
        fed_ids = list(self.gossip.known_federations.keys())
        
        if not fed_ids:
            print("[FAIL] No federations to sync")
            return False
        
        # Perform gossip round
        payload = {
            'type': 'sync_request',
            'federation_id': self.federation_id,
            'timestamp': time.time()
        }
        
        sent, successful = self.gossip.gossip_round(fed_ids, payload)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Output PASS line
        if successful > 0:
            print(f"[PASS] Celestial Sync Complete federations={successful} latency_ms={int(elapsed_ms)}")
            return True
        else:
            print(f"[FAIL] Celestial Sync Failed sent={sent} successful={successful}")
            return False
    
    def verify_cosmic_root(self, root: str, 
                          federation_roots: Optional[Dict[str, str]] = None) -> bool:
        """
        Verify cosmic root integrity.
        
        Args:
            root: Cosmic root hash to verify
            federation_roots: Optional dict of federation_id -> root_hash
            
        Returns:
            True if verification succeeds
        """
        if not federation_roots:
            # Load from known federations
            federation_roots = {}
            for fed_id in self.gossip.known_federations:
                # In production, fetch actual roots
                # For now, use placeholder
                federation_roots[fed_id] = sha256_hex(fed_id.encode('utf-8'))
        
        # Compute expected cosmic root
        roots_list = [(fed_id, root_hash) 
                     for fed_id, root_hash in federation_roots.items()]
        
        expected_root = compute_cosmic_root(roots_list)
        
        # Verify
        verified = (root == expected_root)
        
        if verified:
            print(f"[PASS] Cosmic Root Verified root={root[:16]}... federations={len(federation_roots)}")
        else:
            print(f"[FAIL] Cosmic Root Mismatch expected={expected_root[:16]}... got={root[:16]}...")
        
        return verified
    
    def print_celestial_dossier(self, dossier_file: str) -> None:
        """
        Print celestial dossier details.
        
        Args:
            dossier_file: Path to celestial dossier JSON file
        """
        try:
            with open(dossier_file, 'r') as f:
                dossier_data = json.load(f)
            
            # Extract key information
            version = dossier_data.get('version', 'unknown')
            federations = dossier_data.get('federations', [])
            cosmic_root = dossier_data.get('cosmic_root', '')
            timestamp = dossier_data.get('timestamp', 0)
            dossier_hash = dossier_data.get('hash', '')
            
            # Print summary
            print(f"[PASS] Celestial Dossier Loaded file={dossier_file}")
            print()
            print(f"Version:        {version}")
            print(f"Federations:    {len(federations)}")
            print(f"Cosmic Root:    {cosmic_root[:32]}...")
            print(f"Timestamp:      {timestamp}")
            print(f"Hash:           {dossier_hash[:32]}...")
            print()
            
            # Print federation details
            print("Federations:")
            for fed in federations:
                fed_id = fed.get('federation_id', 'unknown')
                root = fed.get('merkle_root', '')
                print(f"  - {fed_id}: {root[:32]}...")
            
            # Print signature chain
            sig_chain = dossier_data.get('signature_chain', [])
            if sig_chain:
                print()
                print(f"Signature Chain: {len(sig_chain)} signatures")
            
            # Print provenance graph
            prov_graph = dossier_data.get('provenance_graph', {})
            if prov_graph:
                nodes = prov_graph.get('nodes', {})
                edges = prov_graph.get('edges', [])
                print(f"Provenance Graph: {len(nodes)} nodes, {len(edges)} edges")
            
        except Exception as e:
            print(f"[FAIL] Error loading dossier: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MathLedger Control CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (default: ~/.ledgerctl.json)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # join-federation
    join_parser = subparsers.add_parser('join-federation', 
                                       help='Join a federation network')
    join_parser.add_argument('federation_id', help='Federation ID')
    join_parser.add_argument('endpoint', help='Network endpoint (host:port)')
    join_parser.add_argument('--public-key', help='Hex-encoded public key')
    
    # list-federations
    list_parser = subparsers.add_parser('list-federations',
                                       help='List known federations')
    list_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed information')
    
    # sync-celestial
    sync_parser = subparsers.add_parser('sync-celestial',
                                       help='Synchronize with celestial consensus')
    sync_parser.add_argument('--timeout', type=float, default=5000,
                            help='Timeout in milliseconds')
    
    # verify-cosmic-root
    verify_parser = subparsers.add_parser('verify-cosmic-root',
                                         help='Verify cosmic root integrity')
    verify_parser.add_argument('root', help='Cosmic root hash')
    verify_parser.add_argument('--federations', help='JSON file with federation roots')
    
    # print-celestial-dossier
    print_parser = subparsers.add_parser('print-celestial-dossier',
                                        help='Print celestial dossier details')
    print_parser.add_argument('dossier', help='Path to celestial dossier JSON')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    ctl = LedgerCtl(config_file=args.config)
    
    # Execute command
    try:
        if args.command == 'join-federation':
            ctl.join_federation(args.federation_id, args.endpoint, args.public_key)
        
        elif args.command == 'list-federations':
            ctl.list_federations(verbose=args.verbose)
        
        elif args.command == 'sync-celestial':
            if not ctl.sync_celestial(timeout_ms=args.timeout):
                sys.exit(1)
        
        elif args.command == 'verify-cosmic-root':
            fed_roots = None
            if args.federations:
                with open(args.federations, 'r') as f:
                    fed_roots = json.load(f)
            
            if not ctl.verify_cosmic_root(args.root, fed_roots):
                sys.exit(1)
        
        elif args.command == 'print-celestial-dossier':
            ctl.print_celestial_dossier(args.dossier)
        
    except Exception as e:
        print(f"[FAIL] Command failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
