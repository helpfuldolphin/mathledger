"""
Unit tests for replay verification system.

Tests the core invariant: replaying historical blocks from canonical
payloads MUST produce identical attestation roots.
"""

import pytest
from backend.ledger.replay import (
    recompute_attestation_roots,
    verify_block_integrity,
    replay_block,
    IntegrityResult,
)
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)


class TestRootRecomputation:
    """Test root recomputation from canonical payloads."""
    
    def test_recompute_roots_deterministic(self):
        """Test that recomputation is deterministic."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]
        
        r_t1, u_t1, h_t1 = recompute_attestation_roots([], proofs, ui_events)
        r_t2, u_t2, h_t2 = recompute_attestation_roots([], proofs, ui_events)
        
        assert r_t1 == r_t2
        assert u_t1 == u_t2
        assert h_t1 == h_t2
    
    def test_recompute_roots_matches_original(self):
        """Test that recomputation matches original sealing."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]
        
        # Original computation
        original_r_t = compute_reasoning_root(proofs)
        original_u_t = compute_ui_root(ui_events)
        original_h_t = compute_composite_root(original_r_t, original_u_t)
        
        # Recomputation
        recomputed_r_t, recomputed_u_t, recomputed_h_t = recompute_attestation_roots(
            [], proofs, ui_events
        )
        
        assert recomputed_r_t == original_r_t
        assert recomputed_u_t == original_u_t
        assert recomputed_h_t == original_h_t
    
    def test_recompute_empty_ui_events(self):
        """Test recomputation with empty UI events."""
        proofs = [{"statement": "p -> p"}]
        
        r_t, u_t, h_t = recompute_attestation_roots([], proofs, [])
        
        assert len(r_t) == 64
        assert len(u_t) == 64
        assert len(h_t) == 64


class TestIntegrityVerification:
    """Test integrity verification logic."""
    
    def test_verify_valid_block(self):
        """Test verification of valid block."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="a" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="c" * 64,
        )
        
        assert result.is_valid is True
        assert result.r_t_match is True
        assert result.u_t_match is True
        assert result.h_t_match is True
        assert result.error is None
    
    def test_verify_invalid_r_t(self):
        """Test verification detects R_t mismatch."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="x" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="c" * 64,
        )
        
        assert result.is_valid is False
        assert result.r_t_match is False
        assert result.u_t_match is True
        assert result.h_t_match is True
        assert "R_t" in result.error
    
    def test_verify_invalid_composite(self):
        """Test verification detects H_t mismatch."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="a" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="x" * 64,
        )
        
        assert result.is_valid is False
        assert result.h_t_match is False
        assert "H_t" in result.error
    
    def test_integrity_result_to_dict(self):
        """Test IntegrityResult serialization."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="a" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="c" * 64,
        )
        
        result_dict = result.to_dict()
        assert result_dict["is_valid"] is True
        assert result_dict["block_id"] == 1
        assert "stored_roots" in result_dict
        assert "recomputed_roots" in result_dict


class TestReplayBlock:
    """Test replay_block function."""
    
    def test_replay_block_missing_id(self):
        """Test replay_block raises on missing block ID."""
        with pytest.raises(ValueError, match="missing id"):
            replay_block({})
    
    def test_replay_block_missing_roots(self):
        """Test replay_block raises on missing attestation roots."""
        with pytest.raises(ValueError, match="missing attestation roots"):
            replay_block({"id": 1, "block_number": 1})
