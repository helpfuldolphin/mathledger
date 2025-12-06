"""
Proof metadata models for embedding cryptographic proofs in statements.

Provides ProofMetadata dataclass with:
- Statement hash linkage
- Parent hash tracking
- Merkle root computation
- Ed25519 signature
- RFC 8785 serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.crypto.core import (
    rfc8785_canonicalize,
    merkle_root,
    ed25519_sign_b64,
    ed25519_verify_b64,
    ed25519_generate_keypair,
    sha256_hex,
)


# Global keypair for proof signing (in production, load from secure storage)
_PROOF_KEYPAIR: Optional[tuple[bytes, bytes]] = None


def _ensure_proof_keypair() -> tuple[bytes, bytes]:
    """Ensure Ed25519 keypair exists for proof signing."""
    global _PROOF_KEYPAIR
    if _PROOF_KEYPAIR is None:
        _PROOF_KEYPAIR = ed25519_generate_keypair()
    return _PROOF_KEYPAIR


@dataclass
class ProofMetadata:
    """
    Metadata for a cryptographically-verified proof.
    
    Attributes:
        statement_hash: SHA-256 hash of the proven statement
        parent_hashes: List of hashes of parent statements (dependencies)
        timestamp: ISO 8601 timestamp of proof generation
        merkle_root: Merkle root of the proof tree
        signature_b64: Base64-encoded Ed25519 signature
        derivation_rule: Name of derivation rule used
        verified: Whether the proof has been verified
    """
    
    statement_hash: str
    parent_hashes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    merkle_root: str = ""
    signature_b64: str = ""
    derivation_rule: str = "unknown"
    verified: bool = False
    
    def __post_init__(self):
        """Compute merkle root if not provided."""
        if not self.merkle_root and self.parent_hashes:
            self.merkle_root = merkle_root(self.parent_hashes)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return asdict(self)
    
    def to_canonical_json(self) -> str:
        """
        Serialize to RFC 8785 canonical JSON.
        
        Returns:
            Canonical JSON string
        """
        return rfc8785_canonicalize(self.to_dict())
    
    def sign(self, private_key: Optional[bytes] = None) -> str:
        """
        Sign the proof metadata with Ed25519.
        
        Args:
            private_key: Optional private key (uses global if not provided)
            
        Returns:
            Base64-encoded signature
        """
        if private_key is None:
            private_key, _ = _ensure_proof_keypair()
        
        # Create canonical JSON without signature and verified fields
        data = self.to_dict()
        data_for_signing = {k: v for k, v in data.items() if k not in ("signature_b64", "verified")}
        canonical = rfc8785_canonicalize(data_for_signing)
        self.signature_b64 = ed25519_sign_b64(canonical, private_key)
        return self.signature_b64
    
    def verify(self, public_key: Optional[bytes] = None) -> bool:
        """
        Verify the Ed25519 signature.
        
        Args:
            public_key: Optional public key (uses global if not provided)
            
        Returns:
            True if signature is valid
        """
        if not self.signature_b64:
            return False
        
        if public_key is None:
            _, public_key = _ensure_proof_keypair()
        
        # Create canonical JSON without signature and verified fields for verification
        data = self.to_dict()
        data_for_verification = {k: v for k, v in data.items() if k not in ("signature_b64", "verified")}
        canonical = rfc8785_canonicalize(data_for_verification)
        
        self.verified = ed25519_verify_b64(canonical, self.signature_b64, public_key)
        return self.verified
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProofMetadata:
        """
        Create ProofMetadata from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ProofMetadata instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> ProofMetadata:
        """
        Create ProofMetadata from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            ProofMetadata instance
        """
        import json
        return cls.from_dict(json.loads(json_str))
    
    def compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of canonical representation.
        
        Returns:
            64-character hex hash
        """
        canonical = self.to_canonical_json()
        return sha256_hex(canonical)


def create_proof_metadata(
    statement_hash: str,
    parent_hashes: List[str],
    derivation_rule: str = "unknown",
    sign_immediately: bool = True,
) -> ProofMetadata:
    """
    Create and optionally sign a ProofMetadata instance.
    
    Args:
        statement_hash: Hash of the proven statement
        parent_hashes: List of parent statement hashes
        derivation_rule: Name of the derivation rule
        sign_immediately: Whether to sign immediately
        
    Returns:
        ProofMetadata instance
    """
    metadata = ProofMetadata(
        statement_hash=statement_hash,
        parent_hashes=parent_hashes,
        derivation_rule=derivation_rule,
    )
    
    if sign_immediately:
        metadata.sign()
    
    return metadata


def verify_proof_chain(proofs: List[ProofMetadata]) -> tuple[bool, List[str]]:
    """
    Verify a chain of proofs.
    
    Args:
        proofs: List of ProofMetadata to verify
        
    Returns:
        Tuple of (all_valid, error_messages)
    """
    errors = []
    all_valid = True
    
    for i, proof in enumerate(proofs):
        if not proof.verify():
            errors.append(f"Proof {i} (statement={proof.statement_hash[:16]}...) failed signature verification")
            all_valid = False
        
        # Check if merkle root is consistent
        if proof.parent_hashes:
            expected_merkle = merkle_root(proof.parent_hashes)
            if proof.merkle_root != expected_merkle:
                errors.append(
                    f"Proof {i} merkle root mismatch: "
                    f"expected={expected_merkle[:16]}..., got={proof.merkle_root[:16]}..."
                )
                all_valid = False
    
    return all_valid, errors
