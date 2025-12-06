#!/usr/bin/env python3
"""
Celestial Dossier Builder

Merges all Federated Dossiers into a unified Celestial Dossier embedding:
- Federation-of-Federations signature chain
- Provenance graph linking each Merkle lineage to parent root
- Cryptographic inclusion map (DOMAIN_CDOS)

Usage:
    python tools/build_celestial_dossier.py --federations fed1.json fed2.json ...
    python tools/build_celestial_dossier.py --config celestial_config.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.crypto.hashing import sha256_hex, merkle_root
from backend.ledger.v4.interfederation import (
    Ed25519Signer, canonical_json_encode, DOMAIN_CDOS, DOMAIN_FDOS,
    compute_cosmic_root
)
from backend.ledger.v4.stellar import StellarConsensus


class FederatedDossier:
    """Representation of a single federation's dossier."""
    
    def __init__(self, federation_id: str, data: Dict):
        self.federation_id = federation_id
        self.data = data
        self.root = data.get('merkle_root', '')
        self.timestamp = data.get('timestamp', time.time())
        self.signature = data.get('signature', '')
        self.metadata = data.get('metadata', {})
    
    def validate(self) -> bool:
        """Validate dossier has required fields."""
        required = ['federation_id', 'merkle_root', 'timestamp']
        return all(field in self.data for field in required)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'federation_id': self.federation_id,
            'merkle_root': self.root,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'metadata': self.metadata
        }


class ProvenanceGraph:
    """Graph linking Merkle lineages to parent roots."""
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[tuple] = []
    
    def add_node(self, node_id: str, data: Dict) -> None:
        """Add a node to the provenance graph."""
        self.nodes[node_id] = data
    
    def add_edge(self, from_id: str, to_id: str, label: str = '') -> None:
        """Add an edge between nodes."""
        self.edges.append((from_id, to_id, label))
    
    def link_federation(self, fed_id: str, parent_root: str, 
                       child_roots: List[str]) -> None:
        """Link a federation's roots to its parent."""
        # Add federation node
        self.add_node(fed_id, {
            'type': 'federation',
            'parent_root': parent_root,
            'child_count': len(child_roots)
        })
        
        # Add parent node if not exists
        if parent_root not in self.nodes:
            self.add_node(parent_root, {'type': 'root'})
        
        # Link federation to parent
        self.add_edge(fed_id, parent_root, 'parent')
        
        # Link children to federation
        for child_root in child_roots:
            if child_root not in self.nodes:
                self.add_node(child_root, {'type': 'child_root'})
            self.add_edge(child_root, fed_id, 'child')
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary."""
        return {
            'nodes': self.nodes,
            'edges': [{'from': f, 'to': t, 'label': l} 
                     for f, t, l in self.edges]
        }


class CelestialDossier:
    """Unified dossier representing multi-federation consensus."""
    
    def __init__(self):
        self.federations: List[FederatedDossier] = []
        self.signature_chain: List[Dict] = []
        self.provenance_graph = ProvenanceGraph()
        self.cosmic_root: Optional[str] = None
        self.inclusion_map: Dict[str, List[str]] = {}
        self.timestamp = time.time()
        self.metadata: Dict = {}
    
    def add_federation(self, dossier: FederatedDossier) -> None:
        """Add a federated dossier to the celestial dossier."""
        if not dossier.validate():
            raise ValueError(f"Invalid dossier for {dossier.federation_id}")
        
        self.federations.append(dossier)
        
        # Update inclusion map
        if dossier.federation_id not in self.inclusion_map:
            self.inclusion_map[dossier.federation_id] = []
        self.inclusion_map[dossier.federation_id].append(dossier.root)
    
    def build_signature_chain(self, signers: Dict[str, Ed25519Signer]) -> None:
        """
        Build Federation-of-Federations signature chain.
        
        Each federation signs the cosmic root and previous signatures.
        """
        self.signature_chain = []
        
        # Sort federations for determinism
        sorted_feds = sorted(self.federations, key=lambda f: f.federation_id)
        
        # Build chain
        previous_sig = None
        for dossier in sorted_feds:
            fed_id = dossier.federation_id
            
            if fed_id not in signers:
                continue
            
            signer = signers[fed_id]
            
            # Create signature payload
            chain_data = {
                'federation_id': fed_id,
                'merkle_root': dossier.root,
                'cosmic_root': self.cosmic_root or '',
                'previous_signature': previous_sig or '',
                'timestamp': time.time()
            }
            
            chain_bytes = canonical_json_encode(chain_data)
            signature = signer.sign(chain_bytes, domain=DOMAIN_CDOS)
            
            chain_entry = {
                **chain_data,
                'signature': signature
            }
            
            self.signature_chain.append(chain_entry)
            previous_sig = signature
    
    def build_provenance_graph(self) -> None:
        """Build provenance graph from all federations."""
        for dossier in self.federations:
            # Get child roots from inclusion map
            child_roots = self.inclusion_map.get(dossier.federation_id, [])
            
            # Link to cosmic root as parent
            parent_root = self.cosmic_root or dossier.root
            
            self.provenance_graph.link_federation(
                dossier.federation_id,
                parent_root,
                child_roots
            )
    
    def compute_cosmic_root(self) -> str:
        """Compute cosmic root from all federation roots."""
        federation_roots = [
            (dossier.federation_id, dossier.root)
            for dossier in self.federations
        ]
        
        self.cosmic_root = compute_cosmic_root(federation_roots)
        return self.cosmic_root
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the celestial dossier."""
        dossier_data = {
            'federations': [f.to_dict() for f in self.federations],
            'cosmic_root': self.cosmic_root,
            'signature_chain_length': len(self.signature_chain),
            'timestamp': self.timestamp
        }
        
        canonical = canonical_json_encode(dossier_data)
        return sha256_hex(canonical, domain=DOMAIN_CDOS)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'version': '4.0.0',
            'federations': [f.to_dict() for f in self.federations],
            'cosmic_root': self.cosmic_root,
            'signature_chain': self.signature_chain,
            'provenance_graph': self.provenance_graph.to_dict(),
            'inclusion_map': self.inclusion_map,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'hash': self.compute_hash()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    def save(self, filepath: str) -> None:
        """Save to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


def load_federated_dossier(filepath: str) -> FederatedDossier:
    """Load a federated dossier from file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    fed_id = data.get('federation_id', os.path.basename(filepath).split('.')[0])
    return FederatedDossier(fed_id, data)


def build_celestial_dossier(dossier_files: List[str],
                           output_file: Optional[str] = None) -> CelestialDossier:
    """
    Build celestial dossier from multiple federated dossiers.
    
    Args:
        dossier_files: List of paths to federated dossier JSON files
        output_file: Optional output file path
        
    Returns:
        CelestialDossier instance
    """
    celestial = CelestialDossier()
    
    # Load all federated dossiers
    for filepath in dossier_files:
        try:
            dossier = load_federated_dossier(filepath)
            celestial.add_federation(dossier)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}", file=sys.stderr)
    
    if not celestial.federations:
        raise ValueError("No valid federations loaded")
    
    # Compute cosmic root
    celestial.compute_cosmic_root()
    
    # Build provenance graph
    celestial.build_provenance_graph()
    
    # Build signature chain (with dummy signers for demonstration)
    # Note: In production, this would use actual federation private keys
    # provided by the federation operators
    signers = {}
    for dossier in celestial.federations:
        signers[dossier.federation_id] = Ed25519Signer()
    celestial.build_signature_chain(signers)
    
    # Save if output specified
    if output_file:
        celestial.save(output_file)
    
    return celestial


def generate_pass_line(federations: int, sha: str) -> str:
    """Generate standardized PASS line."""
    return f"[PASS] Celestial Dossier Built federations={federations} sha={sha[:16]}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build Celestial Dossier from Federated Dossiers'
    )
    parser.add_argument(
        '--federations', nargs='+',
        help='Paths to federated dossier JSON files'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path for celestial dossier'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    dossier_files = []
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        dossier_files = config.get('federations', [])
    elif args.federations:
        dossier_files = args.federations
    else:
        parser.error("Must specify --federations or --config")
    
    # Build celestial dossier
    try:
        celestial = build_celestial_dossier(dossier_files, args.output)
        
        # Output PASS line
        pass_line = generate_pass_line(
            len(celestial.federations),
            celestial.compute_hash()
        )
        print(pass_line)
        
        # Output summary
        if args.output:
            print(f"Celestial dossier saved to: {args.output}")
        else:
            print(celestial.to_json())
        
    except Exception as e:
        print(f"Error building celestial dossier: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
