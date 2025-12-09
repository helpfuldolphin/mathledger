"""
Centralized cryptographic hashing operations with domain separation.

This module provides canonical implementations of all hash operations used in MathLedger,
with proper domain separation to prevent second preimage attacks (CVE-2012-2459 type).

Domain Separation Tags:
- LEAF: (0x00) - For leaf nodes in Merkle trees
- NODE: (0x01) - For internal nodes in Merkle trees
- STMT: (0x02) - For statement content hashing
- BLCK: (0x03) - For block header hashing
- FED_: (0x04) - For federation namespace
- NODE_: (0x05) - For node attestation namespace
- DOSSIER_: (0x06) - For celestial dossier namespace
- ROOT_: (0x07) - For root hash namespace
"""

import hashlib
from typing import List, Union

from normalization.canon import canonical_bytes, normalize


DOMAIN_LEAF = b'\x00'
DOMAIN_NODE = b'\x01'
DOMAIN_STMT = b'\x02'
DOMAIN_BLCK = b'\x03'
DOMAIN_FED = b'\x04'
DOMAIN_NODE_ATTEST = b'\x05'
DOMAIN_DOSSIER = b'\x06'
DOMAIN_ROOT = b'\x07'


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


def sha3_256_hex(data: Union[str, bytes], domain: bytes = b'') -> str:
    """
    Compute SHA3-256 hash (post-quantum preference) and return as hex string.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha3_256(domain + data).hexdigest()


def sha3_256_bytes(data: Union[str, bytes], domain: bytes = b'') -> bytes:
    """
    Compute SHA3-256 hash (post-quantum preference) and return raw bytes.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha3_256(domain + data).digest()


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
