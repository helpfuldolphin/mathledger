import json, time
from typing import List, Dict
from backend.crypto.hashing import merkle_root as crypto_merkle_root

def merkle_root(ids: List[str]) -> str:
    """
    Deterministic sha256-based Merkle root with domain separation.
    
    Uses centralized crypto module with LEAF:/NODE: domain tags
    to prevent second preimage attacks (CVE-2012-2459).
    
    - Normalize and sort leaves for determinism
    - LEAF: prefix for leaf nodes
    - NODE: prefix for internal nodes
    - Duplicate last node for odd counts
    
    Returns hex string.
    """
    return crypto_merkle_root(ids)

def seal_block(statement_ids: List[str], prev_hash: str, block_number: int, ts: float, version: str="v1") -> Dict:
    """
    Build a block dict with deterministic header + statement id list.
    """
    mroot = merkle_root(statement_ids)
    header = {
        "block_number": block_number,
        "prev_hash": prev_hash,
        "merkle_root": mroot,
        "timestamp": ts,
        "version": version,
    }
    return {"header": header, "statements": statement_ids}
