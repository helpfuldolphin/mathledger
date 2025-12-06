"""
Cosmic Attestation Manifest (CAM) - Unified Provenance Binding
Binds Harmony → Dossier → Ledger with cryptographic attestations.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from backend.crypto.hashing import sha256_hex


DOMAIN_ROOT = b'ROOT_'


@dataclass
class CosmicAttestationManifest:
    """
    Cosmic Attestation Manifest (CAM) - Phase IX unified seal.
    
    Binds consensus, lineage, and ledger into a single cryptographic attestation.
    """
    version: str
    timestamp: float
    harmony_root: str
    dossier_root: str
    ledger_root: str
    unified_root: str
    epochs: int
    nodes: int
    readiness: str
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to canonical dictionary format."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "harmony_root": self.harmony_root,
            "dossier_root": self.dossier_root,
            "ledger_root": self.ledger_root,
            "unified_root": self.unified_root,
            "epochs": self.epochs,
            "nodes": self.nodes,
            "readiness": self.readiness,
            "metadata": self.metadata
        }
    
    def to_canonical_json(self) -> str:
        """Convert to canonical JSON (RFC 8785 discipline)."""
        data = self.to_dict()
        # Sort keys for determinism
        return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    
    def compute_signature(self) -> str:
        """Compute SHA-256 signature of canonical manifest."""
        canonical = self.to_canonical_json()
        return sha256_hex(canonical.encode('utf-8'), domain=DOMAIN_ROOT)


def create_manifest(
    harmony_root: str,
    dossier_root: str,
    ledger_root: str,
    epochs: int,
    nodes: int,
    metadata: Optional[Dict[str, any]] = None
) -> CosmicAttestationManifest:
    """
    Create a Cosmic Attestation Manifest.
    
    Args:
        harmony_root: Root hash from Harmony Protocol consensus
        dossier_root: Root hash from Celestial Dossier
        ledger_root: Root hash from blockchain ledger
        epochs: Number of epochs attested
        nodes: Number of validator nodes
        metadata: Optional additional metadata
    
    Returns:
        CosmicAttestationManifest instance
    """
    # Compute unified root from all three roots
    combined = harmony_root + dossier_root + ledger_root
    unified_root = sha256_hex(combined.encode('utf-8'), domain=DOMAIN_ROOT)
    
    # Compute readiness score (11.1/10 for complete attestation)
    readiness = "11.1/10" if all([harmony_root, dossier_root, ledger_root]) else "0/10"
    
    manifest = CosmicAttestationManifest(
        version="1.1",
        timestamp=time.time(),
        harmony_root=harmony_root,
        dossier_root=dossier_root,
        ledger_root=ledger_root,
        unified_root=unified_root,
        epochs=epochs,
        nodes=nodes,
        readiness=readiness,
        metadata=metadata or {}
    )
    
    return manifest


def verify_attestation(manifest: CosmicAttestationManifest) -> bool:
    """
    Verify the cryptographic integrity of an attestation manifest.
    
    Args:
        manifest: CosmicAttestationManifest to verify
    
    Returns:
        True if attestation is valid
    """
    # Verify unified root computation
    combined = manifest.harmony_root + manifest.dossier_root + manifest.ledger_root
    expected_unified = sha256_hex(combined.encode('utf-8'), domain=DOMAIN_ROOT)
    
    if manifest.unified_root != expected_unified:
        return False
    
    # Verify readiness score matches content
    expected_readiness = "11.1/10" if all([
        manifest.harmony_root,
        manifest.dossier_root,
        manifest.ledger_root
    ]) else "0/10"
    
    if manifest.readiness != expected_readiness:
        return False
    
    # Verify all roots are 64-char hex strings
    for root_name in ['harmony_root', 'dossier_root', 'ledger_root', 'unified_root']:
        root = getattr(manifest, root_name)
        if not isinstance(root, str) or len(root) != 64:
            return False
        try:
            int(root, 16)
        except ValueError:
            return False
    
    return True
