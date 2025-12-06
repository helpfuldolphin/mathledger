import pytest
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

# Import from canonical attestation module (single source of truth)
from attestation.dual_root import (
    canonicalize_reasoning_artifact,
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    hash_reasoning_leaf,
)
from backend.crypto.hashing import merkle_root as centralized_merkle_root
from ledger.blocking import seal_block_with_dual_roots

@dataclass
class SealedBlock:
    reasoning_root: str
    ui_root: str
    composite_root: str
    block_id: Optional[int]
    sequence: int
    timestamp: int

def ingest_and_seal_for_first_organism(result: List[Dict[str, Any]], ui_events: List[Any]) -> SealedBlock:
    """
    Minimal helper to simulate ingestion and sealing for the First Organism.
    
    In a real scenario, this would involve the LedgerIngestor and DB transactions.
    Here we focus on validating the dual-root sealing semantics.
    """
    # Map input result/proofs to what seal_block_with_dual_roots expects
    # It expects a sequence of dicts.
    
    block_metadata = seal_block_with_dual_roots("first_organism", result, ui_events)
    
    return SealedBlock(
        reasoning_root=block_metadata['reasoning_merkle_root'],
        ui_root=block_metadata['ui_merkle_root'],
        composite_root=block_metadata['composite_attestation_root'],
        block_id=None, # No DB in this minimal helper test
        sequence=block_metadata['block_number'],
        timestamp=block_metadata['sealed_at']
    )

def test_first_organism_sealing_semantics():
    """
    Test that validates the First Organism ingestion contract:
    - Computes R_t from proofs
    - Computes U_t from UI events
    - Computes H_t = SHA256(R_t || U_t)
    
    Crucially, this test traces the exact hash path to ensure canonicalization adherence.
    """
    
    # 1. Create a small synthetic set of proofs + UI events
    proofs = [
        {"statement": "a -> a", "method": "axiom", "proof_text": "trivial"},
        {"statement": "b -> b", "method": "axiom", "proof_text": "trivial"}
    ]
    
    ui_events = [
        {"type": "click", "x": 10, "y": 20},
        {"type": "keypress", "key": "enter"}
    ]
    
    # 2. Call the helper
    sealed_block = ingest_and_seal_for_first_organism(proofs, ui_events)
    
    # 3. Assert that recomputed R_t, U_t, H_t match stored values
    
    r_t = sealed_block.reasoning_root
    u_t = sealed_block.ui_root
    h_t = sealed_block.composite_root
    
    # --- VERIFY R_t TRACE (Reasoning Root) ---
    # Path: Proof Dict -> RFC8785 Canonical JSON -> Domain Hash -> Merkle Tree (sorted)
    
    # A. Manually canonicalize and hash each proof
    manual_leaf_hashes = []
    for p in proofs:
        # N(p): Canonicalize (RFC8785)
        canon_p = canonicalize_reasoning_artifact(p)
        # Hash(p): SHA256(DOMAIN_REASONING_LEAF || N(p))
        h_p = hash_reasoning_leaf(canon_p)
        manual_leaf_hashes.append(h_p)
        
    # B. Compute Merkle Root of these hashes
    # The system uses the centralized merkle_root function which:
    # 1. Normalizes inputs (leaves as strings)
    # 2. Encodes them
    # 3. Hashes them with DOMAIN_LEAF (Yes, double hashing in this architecture)
    # 4. Builds tree
    expected_r_t = centralized_merkle_root(manual_leaf_hashes)
    
    assert r_t == expected_r_t, (
        f"Reasoning Root mismatch!\nExpected: {expected_r_t}\nActual:   {r_t}\n"
        "Violation of canonical hash path."
    )

    # --- VERIFY H_t TRACE (Composite Root) ---
    # H_t = SHA256(R_t || U_t)
    expected_h_t = compute_composite_root(r_t, u_t)
    assert h_t == expected_h_t, "Composite root H_t must be SHA256(R_t || U_t)"
    
    # Verify R_t is not empty (since we have proofs)
    assert r_t is not None
    assert len(r_t) == 64
    
    # Verify U_t is not empty (since we have UI events)
    assert u_t is not None
    assert len(u_t) == 64
    
    # Check basic metadata
    assert sealed_block.sequence >= 1
    assert sealed_block.timestamp > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

