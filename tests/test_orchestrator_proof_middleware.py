"""
Tests for Phase III proof middleware and metadata.

Tests proof_middleware.py and proof_metadata.py modules.
"""

import json
import pytest
from pathlib import Path
from backend.orchestrator.proof_middleware import (
    ProofOfExecutionMiddleware,
    get_execution_log_count,
    emit_proof_middleware_passline,
)
from backend.models.proof_metadata import (
    ProofMetadata,
    create_proof_metadata,
    verify_proof_chain,
)


class TestProofMetadata:
    """Test ProofMetadata dataclass."""
    
    def test_proof_metadata_creation(self):
        """Test basic ProofMetadata creation."""
        proof = ProofMetadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64, "c" * 64],
            derivation_rule="mp",
        )
        assert proof.statement_hash == "a" * 64
        assert len(proof.parent_hashes) == 2
        assert proof.derivation_rule == "mp"
    
    def test_proof_metadata_merkle_auto_compute(self):
        """Test automatic Merkle root computation."""
        proof = ProofMetadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64, "c" * 64],
        )
        assert proof.merkle_root  # Should be auto-computed
        assert len(proof.merkle_root) == 64
    
    def test_proof_metadata_to_dict(self):
        """Test ProofMetadata serialization to dict."""
        proof = ProofMetadata(
            statement_hash="test_hash",
            parent_hashes=["parent1"],
            derivation_rule="axiom",
        )
        d = proof.to_dict()
        assert d["statement_hash"] == "test_hash"
        assert d["parent_hashes"] == ["parent1"]
        assert d["derivation_rule"] == "axiom"
    
    def test_proof_metadata_to_canonical_json(self):
        """Test RFC 8785 canonical JSON serialization."""
        proof = ProofMetadata(
            statement_hash="hash1",
            parent_hashes=["hash2"],
        )
        canonical = proof.to_canonical_json()
        assert isinstance(canonical, str)
        # Should be valid JSON
        parsed = json.loads(canonical)
        assert "statement_hash" in parsed
    
    def test_proof_metadata_sign_and_verify(self):
        """Test signing and verification."""
        proof = ProofMetadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64],
            derivation_rule="mp",
        )
        
        # Sign
        signature = proof.sign()
        assert signature
        assert proof.signature_b64 == signature
        
        # Verify
        is_valid = proof.verify()
        assert is_valid is True
        assert proof.verified is True
    
    def test_proof_metadata_invalid_signature(self):
        """Test verification fails with tampered signature."""
        proof = ProofMetadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64],
        )
        proof.sign()
        
        # Tamper with signature
        proof.signature_b64 = "invalid_signature"
        
        is_valid = proof.verify()
        assert is_valid is False
    
    def test_proof_metadata_from_dict(self):
        """Test ProofMetadata creation from dict."""
        data = {
            "statement_hash": "test",
            "parent_hashes": ["p1", "p2"],
            "timestamp": "2025-01-01T00:00:00",
            "merkle_root": "root",
            "signature_b64": "sig",
            "derivation_rule": "mp",
            "verified": False,
        }
        proof = ProofMetadata.from_dict(data)
        assert proof.statement_hash == "test"
        assert len(proof.parent_hashes) == 2
    
    def test_proof_metadata_compute_content_hash(self):
        """Test content hash computation."""
        proof = ProofMetadata(
            statement_hash="hash1",
            parent_hashes=["hash2"],
        )
        content_hash = proof.compute_content_hash()
        assert len(content_hash) == 64


class TestCreateProofMetadata:
    """Test create_proof_metadata helper function."""
    
    def test_create_proof_metadata_basic(self):
        """Test basic proof creation."""
        proof = create_proof_metadata(
            statement_hash="stmt_hash",
            parent_hashes=["p1", "p2"],
            derivation_rule="mp",
            sign_immediately=False,
        )
        assert proof.statement_hash == "stmt_hash"
        assert len(proof.parent_hashes) == 2
        assert proof.signature_b64 == ""  # Not signed
    
    def test_create_proof_metadata_auto_sign(self):
        """Test automatic signing on creation."""
        proof = create_proof_metadata(
            statement_hash="stmt_hash",
            parent_hashes=["p1"],
            derivation_rule="axiom",
            sign_immediately=True,
        )
        assert proof.signature_b64  # Should be signed
        assert proof.verify() is True


class TestVerifyProofChain:
    """Test proof chain verification."""
    
    def test_verify_proof_chain_single_valid(self):
        """Test verification of single valid proof."""
        proof = create_proof_metadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64],
            derivation_rule="mp",
        )
        
        all_valid, errors = verify_proof_chain([proof])
        assert all_valid is True
        assert len(errors) == 0
    
    def test_verify_proof_chain_multiple_valid(self):
        """Test verification of multiple valid proofs."""
        proof1 = create_proof_metadata(
            statement_hash="a" * 64,
            parent_hashes=[],
            derivation_rule="axiom",
        )
        proof2 = create_proof_metadata(
            statement_hash="b" * 64,
            parent_hashes=["a" * 64],
            derivation_rule="mp",
        )
        
        all_valid, errors = verify_proof_chain([proof1, proof2])
        assert all_valid is True
        assert len(errors) == 0
    
    def test_verify_proof_chain_invalid_signature(self):
        """Test chain verification detects invalid signatures."""
        proof = create_proof_metadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64],
        )
        # Tamper
        proof.signature_b64 = "invalid"
        
        all_valid, errors = verify_proof_chain([proof])
        assert all_valid is False
        assert len(errors) > 0
    
    def test_verify_proof_chain_merkle_mismatch(self):
        """Test chain verification detects Merkle mismatches."""
        proof = create_proof_metadata(
            statement_hash="a" * 64,
            parent_hashes=["b" * 64],
        )
        # Tamper with merkle root
        proof.merkle_root = "wrong_root"
        
        all_valid, errors = verify_proof_chain([proof])
        assert all_valid is False
        assert any("merkle root mismatch" in e for e in errors)


class TestProofOfExecutionMiddleware:
    """Test ProofOfExecutionMiddleware class."""
    
    def test_middleware_creation(self, tmp_path):
        """Test middleware initialization."""
        log_path = tmp_path / "test_log.jsonl"
        
        middleware = ProofOfExecutionMiddleware(
            app=None,  # App not needed for init test
            log_path=str(log_path),
            enabled=True,
        )
        
        assert middleware.enabled is True
        assert middleware.request_count == 0
    
    def test_middleware_disabled(self, tmp_path):
        """Test middleware with disabled flag."""
        log_path = tmp_path / "test_log.jsonl"
        
        middleware = ProofOfExecutionMiddleware(
            app=None,
            log_path=str(log_path),
            enabled=False,
        )
        
        assert middleware.enabled is False


class TestExecutionLogHelpers:
    """Test execution log helper functions."""
    
    def test_get_execution_log_count_empty(self, tmp_path):
        """Test count on empty/non-existent log."""
        log_path = tmp_path / "nonexistent.jsonl"
        count = get_execution_log_count(str(log_path))
        assert count == 0
    
    def test_get_execution_log_count_with_entries(self, tmp_path):
        """Test count on log with entries."""
        log_path = tmp_path / "test.jsonl"
        with open(log_path, 'w') as f:
            f.write('{"test": 1}\n')
            f.write('{"test": 2}\n')
            f.write('\n')  # Empty line
            f.write('{"test": 3}\n')
        
        count = get_execution_log_count(str(log_path))
        assert count == 3  # Should skip empty line
    
    def test_emit_proof_middleware_passline(self, tmp_path, capsys):
        """Test pass-line emission."""
        log_path = tmp_path / "test.jsonl"
        with open(log_path, 'w') as f:
            f.write('{"test": 1}\n')
        
        emit_proof_middleware_passline(str(log_path))
        
        captured = capsys.readouterr()
        assert "[PASS] Proof-of-Execution Ledger Active" in captured.out
        assert "logs=1" in captured.out


# Integration tests
class TestPhase3ProofIntegration:
    """Integration tests for Phase III proof components."""
    
    def test_proof_metadata_with_crypto_core(self):
        """Test ProofMetadata integrates with crypto.core."""
        from backend.crypto.core import merkle_root, ed25519_verify_b64
        
        deps = ["dep1", "dep2"]
        expected_merkle = merkle_root(deps)
        
        proof = ProofMetadata(
            statement_hash="stmt",
            parent_hashes=deps,
        )
        
        assert proof.merkle_root == expected_merkle
    
    def test_end_to_end_proof_creation_verification(self):
        """Test complete proof creation and verification flow."""
        # Create proof
        proof = create_proof_metadata(
            statement_hash="test_statement_hash",
            parent_hashes=["parent1_hash", "parent2_hash"],
            derivation_rule="modus_ponens",
        )
        
        # Should be signed
        assert proof.signature_b64
        
        # Should verify
        assert proof.verify() is True
        
        # Should serialize
        json_str = proof.to_canonical_json()
        assert json_str
        
        # Should deserialize
        proof2 = ProofMetadata.from_json(json_str)
        # Note: signature won't match after round-trip without re-signing
        # but structure should be intact
        assert proof2.statement_hash == proof.statement_hash
        assert proof2.parent_hashes == proof.parent_hashes


# Parametrized tests
@pytest.mark.parametrize("num_parents", [0, 1, 2, 5, 10])
def test_proof_metadata_various_parent_counts(num_parents):
    """Test ProofMetadata with various numbers of parents."""
    parents = [f"parent{i}" for i in range(num_parents)]
    proof = ProofMetadata(
        statement_hash="test",
        parent_hashes=parents,
    )
    assert len(proof.parent_hashes) == num_parents
    if num_parents > 0:
        assert proof.merkle_root  # Should be computed
