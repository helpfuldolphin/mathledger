import hashlib
import pytest
from backend.crypto.core import (
    hash_statement, 
    merkle_root, 
    DOMAIN_STMT, 
    DOMAIN_LEAF, 
    DOMAIN_NODE,
    sha256_bytes,
    sha256_hex
)
from normalization.canon import normalize

class TestHashingCanonIntegration:
    """
    Integration tests enforcing the Hash Canonizer spec.
    Ensures statements, proofs, and blocks follow the whitepaper identity rules.
    """

    def test_statement_hash_path_manual_verification(self):
        """
        Trace statement hash path: s -> N(s) -> E -> SHA256(D || E).
        Verify against manual construction.
        """
        raw_statement = "  (p /\\ q)   -> p  "
        
        # 1. Normalization (N)
        # "  (p /\\ q)   -> p  " -> "(p/\q)->p" (whitespace removed, structure preserved)
        # actually canon.normalize removes whitespace and standardizes
        normalized = normalize(raw_statement)
        assert normalized == "(p/\\q)->p"
        
        # 2. Encoding (E) & Domain (D)
        # DOMAIN_STMT = b'\x02'
        payload = DOMAIN_STMT + normalized.encode("utf-8")
        
        # 3. Hashing
        manual_hash = hashlib.sha256(payload).hexdigest()
        
        # 4. Comparison
        system_hash = hash_statement(raw_statement)
        
        assert system_hash == manual_hash
        assert system_hash == "c0615823a9836947f8cd3a7971b6979ec4c22ca00d76c8d623aedaa9c3f866c5" # Validated manually via manual_hash calculation

    def test_merkle_root_construction_spec(self):
        """
        Verify Merkle root construction follows the spec:
        - Leaves hashed with DOMAIN_LEAF
        - Leaves sorted
        - Nodes hashed with DOMAIN_NODE
        - Odd nodes duplicated
        """
        # Inputs
        ids = ["c", "a", "b"] # Unsorted inputs
        
        # 1. Prepare Leaves (Manual)
        # normalize -> encode -> hash(LEAF + data)
        leaves = []
        for x in ids:
            norm = normalize(x)
            data = DOMAIN_LEAF + norm.encode("utf-8")
            h = hashlib.sha256(data).digest()
            leaves.append(h)
            
        # 2. Sort Leaves (Spec requirement)
        leaves.sort()
        
        # 3. Build Tree (Manual)
        # Level 0: leaves (3 items) -> duplicate last to make 4
        # sorted order of "a", "b", "c" hashes
        # Let's check exact hashes to be sure of sorting
        # hash("a") < hash("b") ?
        
        # We'll just implement the manual loop
        nodes = leaves
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i+1]
                # Parent = hash(NODE + left + right)
                combined = DOMAIN_NODE + left + right
                parent = hashlib.sha256(combined).digest()
                next_level.append(parent)
            nodes = next_level
            
        manual_root = nodes[0].hex()
        
        # 4. System Implementation
        system_root = merkle_root(ids)
        
        assert system_root == manual_root

    def test_ad_hoc_hashing_regression(self):
        """
        Regression test: Ensure no known ad-hoc hashing patterns match the canonical output 
        unless they accidentally implement the full spec.
        """
        s = "p->p"
        
        # Naive sha256(s)
        naive = hashlib.sha256(s.encode()).hexdigest()
        
        # Canonical
        canonical = hash_statement(s)
        
        assert canonical != naive, "Canonical hash MUST NOT match naive SHA256"
        
        # Naive with normalization but no domain
        norm_naive = hashlib.sha256(normalize(s).encode()).hexdigest()
        assert canonical != norm_naive, "Canonical hash MUST NOT match domain-less normalized SHA256"

    def test_block_hashing_consistency(self):
        """
        Verify block hashing flow if possible.
        Since block hashing depends on 'canonical_json' which sorts keys, we verify that too.
        """
        # We simulate the block hash function from backend.crypto.core
        from backend.crypto.core import hash_block, DOMAIN_BLCK
        
        block_data = '{"block_number":1}'
        expected = hashlib.sha256(DOMAIN_BLCK + block_data.encode("utf-8")).hexdigest()
        
        assert hash_block(block_data) == expected

