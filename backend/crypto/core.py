"""
Centralized cryptographic core operations for MathLedger.

This module provides canonical implementations of:
- Ed25519 signing and verification
- RFC 8785 JSON canonicalization
- Merkle tree operations with domain separation
- Statement and block hashing

All operations follow security best practices and provide deterministic outputs.
"""

import base64
import hashlib
import json
from typing import Any, Dict, List, Union

from normalization.canon import canonical_bytes, normalize

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Domain separation tags (prevent second preimage attacks)
DOMAIN_LEAF = b'\x00'
DOMAIN_NODE = b'\x01'
DOMAIN_STMT = b'\x02'
DOMAIN_BLCK = b'\x03'
DOMAIN_FED = b'\x04'
DOMAIN_NODE_ATTEST = b'\x05'
DOMAIN_DOSSIER = b'\x06'
DOMAIN_ROOT = b'\x07'


def rfc8785_canonicalize(obj: Any) -> str:
    """
    Canonicalize JSON according to RFC 8785 (JSON Canonicalization Scheme).
    
    Rules:
    - Keys sorted lexicographically
    - No insignificant whitespace
    - Unicode escapes normalized
    - Numbers in standard form
    
    Args:
        obj: Python object to canonicalize
        
    Returns:
        Canonical JSON string
    """
    def serialize_value(v: Any) -> str:
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, int):
            return str(v)
        elif isinstance(v, float):
            # RFC 8785 requires standard form for floats
            return json.dumps(v)
        elif isinstance(v, str):
            # Escape per RFC 8785
            return json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            items = [serialize_value(item) for item in v]
            return "[" + ",".join(items) + "]"
        elif isinstance(v, dict):
            # Sort keys lexicographically
            pairs = []
            for key in sorted(v.keys()):
                key_str = json.dumps(key, ensure_ascii=False)
                val_str = serialize_value(v[key])
                pairs.append(f"{key_str}:{val_str}")
            return "{" + ",".join(pairs) + "}"
        else:
            # Fallback for other types
            return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    
    return serialize_value(obj)


def sha256_hex(data: Union[str, bytes], domain: bytes = b'') -> str:
    """
    Compute SHA-256 hash and return as hex string.
    
    Args:
        data: Input data (string will be UTF-8 encoded)
        domain: Optional domain separation prefix
        
    Returns:
        64-character hex string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(domain + data).hexdigest()


def sha256_bytes(data: Union[str, bytes], domain: bytes = b'') -> bytes:
    """
    Compute SHA-256 hash and return as bytes.
    
    Args:
        data: Input data (string will be UTF-8 encoded)
        domain: Optional domain separation prefix
        
    Returns:
        32-byte digest
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(domain + data).digest()


def sha256_hex_concat(items: List[str]) -> str:
    """
    Compute SHA-256 of concatenated items.
    
    Args:
        items: List of strings to concatenate and hash
        
    Returns:
        64-character hex hash
    """
    concatenated = "".join(sorted(items))
    return sha256_hex(concatenated)


def hash_statement(statement: str) -> str:
    """
    Hash a statement with STMT domain separation.
    
    Args:
        statement: Normalized statement text
        
    Returns:
        64-character hex hash
    """
    canonical = canonical_bytes(statement)
    return sha256_hex(canonical, domain=DOMAIN_STMT)


def hash_block(block_data: str) -> str:
    """
    Hash block data with BLCK domain separation.
    
    Args:
        block_data: Block content to hash
        
    Returns:
        64-character hex hash
    """
    return sha256_hex(block_data, domain=DOMAIN_BLCK)


def merkle_root(ids: List[str]) -> str:
    """
    Compute deterministic Merkle root with domain separation.
    
    This implementation uses:
    - LEAF: prefix for leaf nodes (prevents leaf/internal confusion)
    - NODE: prefix for internal nodes
    - Sorted leaves for determinism
    - Duplicate last node for odd counts
    
    Args:
        ids: List of statement IDs or content to hash
        
    Returns:
        64-character hex Merkle root
        
    Security:
    - Domain separation prevents second preimage attacks (CVE-2012-2459)
    - Sorted leaves ensure deterministic output
    - Proper binary tree construction enables Merkle proofs
    """
    if not ids:
        return sha256_hex(b'', domain=DOMAIN_LEAF)
    
    # Normalize and sort leaves for determinism
    leaves = [normalize(x).encode('utf-8') for x in ids]
    leaves = sorted(leaves)
    
    nodes = [sha256_bytes(leaf, domain=DOMAIN_LEAF) for leaf in leaves]
    
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        
        next_level = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i + 1]
            next_level.append(sha256_bytes(combined, domain=DOMAIN_NODE))
        
        nodes = next_level
    
    return nodes[0].hex()


def ed25519_generate_keypair() -> tuple[bytes, bytes]:
    """
    Generate Ed25519 keypair.
    
    Returns:
        Tuple of (private_key_bytes, public_key_bytes)
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not available")
    
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    return private_bytes, public_bytes


def ed25519_sign_b64(data: Union[str, bytes], private_key: bytes) -> str:
    """
    Sign data with Ed25519 and return base64-encoded signature.
    
    Args:
        data: Data to sign (string will be UTF-8 encoded)
        private_key: 32-byte Ed25519 private key
        
    Returns:
        Base64-encoded signature
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not available")
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
    signature = private_key_obj.sign(data)
    
    return base64.b64encode(signature).decode('ascii')


def ed25519_verify_b64(data: Union[str, bytes], signature_b64: str, public_key: bytes) -> bool:
    """
    Verify Ed25519 signature.
    
    Args:
        data: Data that was signed
        signature_b64: Base64-encoded signature
        public_key: 32-byte Ed25519 public key
        
    Returns:
        True if signature is valid
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not available")
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    try:
        signature = base64.b64decode(signature_b64)
        public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        public_key_obj.verify(signature, data)
        return True
    except Exception:
        return False


def verify_merkle_proof(leaf: str, proof: List[tuple], root: str) -> bool:
    """
    Verify a Merkle proof for a leaf.
    
    Args:
        leaf: Leaf content to verify
        proof: List of (sibling_hash, is_left) tuples
        root: Expected Merkle root
        
    Returns:
        True if proof is valid
    """
    current = sha256_bytes(normalize(leaf).encode('utf-8'), domain=DOMAIN_LEAF)
    
    for sibling_hex, is_left in proof:
        sibling = bytes.fromhex(sibling_hex)
        if is_left:
            combined = sibling + current
        else:
            combined = current + sibling
        current = sha256_bytes(combined, domain=DOMAIN_NODE)
    
    return current.hex() == root


def compute_merkle_proof(leaf_index: int, leaves: List[str]) -> List[tuple]:
    """
    Compute Merkle proof for a leaf at given index.
    
    Args:
        leaf_index: Index of leaf to prove
        leaves: All leaves in the tree
        
    Returns:
        List of (sibling_hash, is_left) tuples
    """
    if leaf_index < 0 or leaf_index >= len(leaves):
        raise ValueError(f"Invalid leaf index: {leaf_index}")
    
    normalized = [(normalize(x).encode('utf-8'), i) for i, x in enumerate(leaves)]
    normalized.sort(key=lambda x: x[0])
    
    sorted_index = next(i for i, (_, orig_idx) in enumerate(normalized) if orig_idx == leaf_index)
    
    nodes = [sha256_bytes(leaf, domain=DOMAIN_LEAF) for leaf, _ in normalized]
    
    proof = []
    current_index = sorted_index
    
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        
        if current_index % 2 == 0:
            sibling_index = current_index + 1
            is_left = False
        else:
            sibling_index = current_index - 1
            is_left = True
        
        proof.append((nodes[sibling_index].hex(), is_left))
        
        next_level = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i + 1]
            next_level.append(sha256_bytes(combined, domain=DOMAIN_NODE))
        
        nodes = next_level
        current_index = current_index // 2
    
    return proof
