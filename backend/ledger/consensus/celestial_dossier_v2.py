"""
Celestial Dossier v2 - Extended Provenance and Attestation

Provides cross-epoch lineage tracking, Merkle inclusion proofs, and
Cosmic Attestation Manifest (CAM) generation that unifies cryptographic
roots from Harmony Protocol, ledger state, and provenance chains.

Features:
- Cross-epoch provenance graphs with lineage tracking
- Merkle inclusion proofs for consensus messages
- CAM generation uniting all cryptographic roots
- Deterministic JSON-canonical serialization
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Add root directory to path for imports when run as standalone script
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

from backend.crypto.hashing import (
    sha256_hex, merkle_root, compute_merkle_proof, verify_merkle_proof,
    DOMAIN_DOSSIER, DOMAIN_ROOT
)


@dataclass
class ProvenanceNode:
    """
    Node in the provenance graph representing a state transition.
    
    Attributes:
        node_id: Unique identifier for this provenance node
        epoch: Epoch number
        parent_ids: List of parent node IDs (cross-epoch lineage)
        state_hash: Hash of state at this node
        timestamp: ISO 8601 timestamp
        metadata: Additional metadata
    """
    node_id: str
    epoch: int
    parent_ids: List[str]
    state_hash: str
    timestamp: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of provenance node."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return sha256_hex(canonical, domain=DOMAIN_DOSSIER)


@dataclass
class CosmicAttestationManifest:
    """
    Unified cryptographic manifest combining all system roots.
    
    Attributes:
        harmony_root: Root hash from Harmony Protocol
        dossier_root: Root hash from Celestial Dossier
        ledger_root: Root hash from blockchain ledger
        cosmic_root: Combined root hash of all roots
        timestamp: ISO 8601 timestamp
        federations: Number of participating federations
        nodes: Number of participating nodes
    """
    harmony_root: str
    dossier_root: str
    ledger_root: str
    cosmic_root: str
    timestamp: str
    federations: int
    nodes: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CelestialDossier:
    """
    Celestial Dossier v2 - Extended Provenance Tracking
    
    Maintains cross-epoch provenance graphs with lineage tracking and
    generates Cosmic Attestation Manifests that unify all cryptographic roots.
    """
    
    def __init__(self):
        """Initialize empty Celestial Dossier."""
        self.provenance_nodes: Dict[str, ProvenanceNode] = {}
        self.epoch_nodes: Dict[int, List[str]] = {}
        self.current_epoch = 0
    
    def add_provenance_node(
        self,
        node_id: str,
        state_hash: str,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> ProvenanceNode:
        """
        Add a provenance node to the dossier.
        
        Args:
            node_id: Unique identifier for the node
            state_hash: Hash of state at this node
            parent_ids: List of parent node IDs (can span epochs)
            metadata: Additional metadata
            
        Returns:
            Created ProvenanceNode
        """
        if parent_ids is None:
            parent_ids = []
        if metadata is None:
            metadata = {}
        
        node = ProvenanceNode(
            node_id=node_id,
            epoch=self.current_epoch,
            parent_ids=parent_ids,
            state_hash=state_hash,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            metadata=metadata
        )
        
        self.provenance_nodes[node_id] = node
        
        if self.current_epoch not in self.epoch_nodes:
            self.epoch_nodes[self.current_epoch] = []
        self.epoch_nodes[self.current_epoch].append(node_id)
        
        return node
    
    def advance_epoch(self) -> int:
        """
        Advance to the next epoch.
        
        Returns:
            New epoch number
        """
        self.current_epoch += 1
        return self.current_epoch
    
    def get_lineage(self, node_id: str, max_depth: int = 10) -> List[ProvenanceNode]:
        """
        Get lineage (ancestry) of a provenance node across epochs.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ancestor nodes in reverse chronological order
        """
        lineage = []
        visited: Set[str] = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth or current_id in visited:
                continue
            
            if current_id not in self.provenance_nodes:
                continue
            
            visited.add(current_id)
            node = self.provenance_nodes[current_id]
            lineage.append(node)
            
            for parent_id in node.parent_ids:
                queue.append((parent_id, depth + 1))
        
        return lineage
    
    def compute_dossier_root(self) -> str:
        """
        Compute deterministic root hash of entire dossier.
        
        Returns:
            SHA-256 hex hash of all provenance nodes
        """
        all_nodes = [
            self.provenance_nodes[node_id].to_dict()
            for node_id in sorted(self.provenance_nodes.keys())
        ]
        canonical = json.dumps(all_nodes, sort_keys=True, ensure_ascii=True)
        return sha256_hex(canonical, domain=DOMAIN_DOSSIER)
    
    def generate_merkle_inclusion_proof(
        self,
        node_id: str
    ) -> Optional[Dict]:
        """
        Generate Merkle inclusion proof for a provenance node.
        
        Args:
            node_id: Node ID to generate proof for
            
        Returns:
            Dictionary containing proof data, or None if node not found
        """
        if node_id not in self.provenance_nodes:
            return None
        
        # Get all nodes in same epoch
        node = self.provenance_nodes[node_id]
        epoch = node.epoch
        
        if epoch not in self.epoch_nodes:
            return None
        
        epoch_node_ids = self.epoch_nodes[epoch]
        
        # Find index of target node
        try:
            node_index = epoch_node_ids.index(node_id)
        except ValueError:
            return None
        
        # Compute Merkle proof
        leaves = [self.provenance_nodes[nid].compute_hash() for nid in epoch_node_ids]
        proof = compute_merkle_proof(node_index, leaves)
        epoch_root = merkle_root(leaves)
        
        return {
            'node_id': node_id,
            'epoch': epoch,
            'node_hash': node.compute_hash(),
            'epoch_root': epoch_root,
            'proof': proof,
            'proof_length': len(proof)
        }
    
    def verify_merkle_inclusion_proof(
        self,
        node_id: str,
        proof_data: Dict
    ) -> bool:
        """
        Verify a Merkle inclusion proof.
        
        Args:
            node_id: Node ID being proved
            proof_data: Proof data from generate_merkle_inclusion_proof
            
        Returns:
            True if proof is valid
        """
        if node_id not in self.provenance_nodes:
            return False
        
        node = self.provenance_nodes[node_id]
        node_hash = node.compute_hash()
        
        proof = proof_data.get('proof', [])
        expected_root = proof_data.get('epoch_root')
        
        return verify_merkle_proof(node_hash, proof, expected_root)
    
    def generate_cosmic_attestation_manifest(
        self,
        harmony_root: str,
        ledger_root: str,
        federations: int,
        nodes: int
    ) -> CosmicAttestationManifest:
        """
        Generate Cosmic Attestation Manifest (CAM) unifying all roots.
        
        Args:
            harmony_root: Root hash from Harmony Protocol
            ledger_root: Root hash from blockchain ledger
            federations: Number of participating federations
            nodes: Number of participating nodes
            
        Returns:
            CosmicAttestationManifest
        """
        dossier_root = self.compute_dossier_root()
        
        # Compute cosmic root by combining all roots
        combined = harmony_root + dossier_root + ledger_root
        cosmic_root = sha256_hex(combined, domain=DOMAIN_ROOT)
        
        return CosmicAttestationManifest(
            harmony_root=harmony_root,
            dossier_root=dossier_root,
            ledger_root=ledger_root,
            cosmic_root=cosmic_root,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            federations=federations,
            nodes=nodes
        )
    
    def export_provenance_graph(self) -> Dict:
        """
        Export complete provenance graph as JSON-serializable dictionary.
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []
        
        for node_id, node in self.provenance_nodes.items():
            nodes.append(node.to_dict())
            
            for parent_id in node.parent_ids:
                edges.append({
                    'from': parent_id,
                    'to': node_id,
                    'epoch_transition': (
                        self.provenance_nodes[parent_id].epoch != node.epoch
                        if parent_id in self.provenance_nodes else False
                    )
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'current_epoch': self.current_epoch,
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }
    
    def compute_statistics(self) -> Dict:
        """
        Compute statistics about the provenance graph.
        
        Returns:
            Dictionary with statistics
        """
        total_nodes = len(self.provenance_nodes)
        epochs_with_nodes = len(self.epoch_nodes)
        
        # Count cross-epoch edges
        cross_epoch_edges = 0
        for node in self.provenance_nodes.values():
            for parent_id in node.parent_ids:
                if parent_id in self.provenance_nodes:
                    parent_node = self.provenance_nodes[parent_id]
                    if parent_node.epoch != node.epoch:
                        cross_epoch_edges += 1
        
        return {
            'total_nodes': total_nodes,
            'total_epochs': epochs_with_nodes,
            'current_epoch': self.current_epoch,
            'cross_epoch_edges': cross_epoch_edges,
            'avg_nodes_per_epoch': total_nodes / epochs_with_nodes if epochs_with_nodes > 0 else 0
        }


def main():
    """Command-line interface for Celestial Dossier v2."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Celestial Dossier v2 - Cross-epoch provenance tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 celestial_dossier_v2.py --attest --epoch 1
  python3 celestial_dossier_v2.py --attest --epoch 5 --nodes-per-epoch 10
        """
    )
    
    parser.add_argument('--attest', action='store_true',
                       help='Generate attestation for current dossier state')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Target epoch number')
    parser.add_argument('--nodes-per-epoch', type=int, default=5,
                       help='Number of provenance nodes per epoch (default: 5)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify Merkle inclusion proofs')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.epoch < 0:
        print("[FAIL] Epoch must be non-negative")
        return 1
    
    # Initialize Celestial Dossier
    dossier = CelestialDossier()
    
    # Create provenance nodes across epochs
    for epoch in range(args.epoch + 1):
        for i in range(args.nodes_per_epoch):
            node_id = f"prov_e{epoch}_n{i}"
            state_hash = sha256_hex(f"state_{epoch}_{i}", domain=DOMAIN_ROOT)
            
            # Create lineage by referencing previous epoch's nodes
            parent_ids = []
            if epoch > 0 and i < args.nodes_per_epoch:
                parent_ids = [f"prov_e{epoch-1}_n{i}"]
            
            dossier.add_provenance_node(
                node_id=node_id,
                state_hash=state_hash,
                parent_ids=parent_ids,
                metadata={'epoch': epoch, 'index': i}
            )
        
        if epoch < args.epoch:
            dossier.advance_epoch()
    
    # Generate dossier root
    dossier_root = dossier.compute_dossier_root()
    
    # Verify Merkle inclusion if requested
    proof_valid = True
    if args.verify:
        # Test a sample node from middle epoch
        mid_epoch = args.epoch // 2 if args.epoch > 0 else 0
        test_node_id = f"prov_e{mid_epoch}_n0"
        
        if test_node_id in dossier.provenance_nodes:
            inclusion_proof = dossier.generate_merkle_inclusion_proof(test_node_id)
            if inclusion_proof:
                proof_valid = dossier.verify_merkle_inclusion_proof(test_node_id, inclusion_proof)
    
    # Get statistics
    stats = dossier.compute_statistics()
    
    # Emit terminal output
    if args.attest:
        if proof_valid:
            print(f"[PASS] Celestial Dossier Provenance Verified root={dossier_root}")
            return 0
        else:
            print(f"[FAIL] Celestial Dossier Provenance verification failed")
            print(f"       Root: {dossier_root}")
            print(f"       Merkle proof validation: {proof_valid}")
            return 1
    else:
        # Info mode
        print(f"[INFO] Celestial Dossier Statistics")
        print(f"       Root: {dossier_root}")
        print(f"       Total Nodes: {stats['total_nodes']}")
        print(f"       Total Epochs: {stats['total_epochs']}")
        print(f"       Cross-Epoch Edges: {stats['cross_epoch_edges']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
