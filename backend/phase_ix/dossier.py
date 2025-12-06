"""
Celestial Dossier v2 - Cross-Epoch Lineage Management
Constructs cross-epoch lineage graphs with Merkle inclusion proofs.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from backend.crypto.hashing import sha256_hex, merkle_root, compute_merkle_proof


DOMAIN_FED = b'FED_'
DOMAIN_NODE = b'NODE_'
DOMAIN_DOSSIER = b'DOSSIER_'


@dataclass
class EpochLineage:
    """Lineage record for a single epoch."""
    epoch_id: int
    parent_epoch: Optional[int]
    statements: List[str]
    merkle_root: str
    epoch_hash: str
    timestamp: float
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of this epoch."""
        content = f"{self.epoch_id}:{self.parent_epoch}:{self.merkle_root}:{self.timestamp}"
        return sha256_hex(content.encode('utf-8'), domain=DOMAIN_NODE)


@dataclass
class CelestialDossier:
    """
    Celestial Dossier v2 - Cross-epoch lineage tracker.
    
    Maintains cryptographic lineage across epochs with Merkle proofs.
    """
    epochs: Dict[int, EpochLineage]
    version: str = "2.0"
    created_at: float = field(default_factory=time.time)
    
    def add_epoch(self, epoch: EpochLineage):
        """Add an epoch to the dossier."""
        if epoch.epoch_id in self.epochs:
            raise ValueError(f"Epoch {epoch.epoch_id} already exists")
        
        # Verify parent link if specified
        if epoch.parent_epoch is not None:
            if epoch.parent_epoch not in self.epochs:
                raise ValueError(f"Parent epoch {epoch.parent_epoch} not found")
        
        self.epochs[epoch.epoch_id] = epoch
    
    def get_lineage_chain(self, epoch_id: int) -> List[EpochLineage]:
        """Get the lineage chain from genesis to specified epoch."""
        if epoch_id not in self.epochs:
            raise ValueError(f"Epoch {epoch_id} not found")
        
        chain = []
        current_id = epoch_id
        
        while current_id is not None:
            epoch = self.epochs[current_id]
            chain.append(epoch)
            current_id = epoch.parent_epoch
        
        return list(reversed(chain))
    
    def verify_lineage(self, epoch_id: int) -> bool:
        """Verify the integrity of lineage chain up to epoch."""
        try:
            chain = self.get_lineage_chain(epoch_id)
            
            # Verify each epoch's hash
            for epoch in chain:
                computed_hash = epoch.compute_hash()
                if computed_hash != epoch.epoch_hash:
                    return False
            
            # Verify parent links
            for i in range(1, len(chain)):
                if chain[i].parent_epoch != chain[i-1].epoch_id:
                    return False
            
            return True
        except (ValueError, KeyError):
            return False
    
    def get_merkle_proof(self, epoch_id: int, statement: str) -> Optional[List[Tuple[str, bool]]]:
        """Get Merkle inclusion proof for a statement in an epoch."""
        if epoch_id not in self.epochs:
            return None
        
        epoch = self.epochs[epoch_id]
        try:
            statement_index = epoch.statements.index(statement)
            return compute_merkle_proof(statement_index, epoch.statements)
        except ValueError:
            return None
    
    def compute_root_hash(self) -> str:
        """Compute the root hash of entire dossier."""
        if not self.epochs:
            return sha256_hex(b'', domain=DOMAIN_DOSSIER)
        
        # Sort epochs by ID for determinism
        sorted_epochs = sorted(self.epochs.values(), key=lambda e: e.epoch_id)
        epoch_hashes = [e.epoch_hash for e in sorted_epochs]
        
        # Compute Merkle root of all epoch hashes
        root = merkle_root(epoch_hashes)
        return sha256_hex(root.encode('utf-8'), domain=DOMAIN_DOSSIER)


def create_dossier(epochs_data: List[Dict[str, any]]) -> CelestialDossier:
    """
    Create a Celestial Dossier from epoch data.
    
    Args:
        epochs_data: List of epoch dictionaries with keys:
            - epoch_id: int
            - parent_epoch: Optional[int]
            - statements: List[str]
            - metadata: Optional[Dict]
    
    Returns:
        CelestialDossier instance
    """
    dossier = CelestialDossier(epochs={})
    
    # Sort by epoch_id to ensure proper ordering
    sorted_epochs = sorted(epochs_data, key=lambda e: e['epoch_id'])
    
    for epoch_data in sorted_epochs:
        statements = epoch_data['statements']
        mroot = merkle_root(statements) if statements else sha256_hex(b'')
        
        epoch = EpochLineage(
            epoch_id=epoch_data['epoch_id'],
            parent_epoch=epoch_data.get('parent_epoch'),
            statements=statements,
            merkle_root=mroot,
            epoch_hash='',  # Will be computed next
            timestamp=epoch_data.get('timestamp', time.time()),
            metadata=epoch_data.get('metadata', {})
        )
        
        # Compute and set epoch hash
        epoch.epoch_hash = epoch.compute_hash()
        
        dossier.add_epoch(epoch)
    
    return dossier
