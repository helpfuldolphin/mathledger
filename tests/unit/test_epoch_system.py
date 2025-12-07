"""
Unit tests for epoch root system.

Tests epoch-level aggregation of block attestation roots.
"""

import pytest
from backend.ledger.epoch import (
    compute_epoch_root,
    seal_epoch,
    verify_epoch_integrity,
    replay_epoch,
    DEFAULT_EPOCH_SIZE,
)


class TestEpochRootComputation:
    """Test epoch root computation."""
    
    def test_compute_epoch_root_deterministic(self):
        """Test that epoch root computation is deterministic."""
        composite_roots = ["a" * 64, "b" * 64, "c" * 64]
        
        e_t1 = compute_epoch_root(composite_roots)
        e_t2 = compute_epoch_root(composite_roots)
        
        assert e_t1 == e_t2
        assert len(e_t1) == 64
    
    def test_compute_epoch_root_empty_raises(self):
        """Test that empty composite roots raises error."""
        with pytest.raises(ValueError, match="empty"):
            compute_epoch_root([])
    
    def test_compute_epoch_root_different_inputs(self):
        """Test that different inputs produce different roots."""
        roots1 = ["a" * 64, "b" * 64]
        roots2 = ["c" * 64, "d" * 64]
        
        e_t1 = compute_epoch_root(roots1)
        e_t2 = compute_epoch_root(roots2)
        
        assert e_t1 != e_t2


class TestEpochSealing:
    """Test epoch sealing."""
    
    def test_seal_epoch_basic(self):
        """Test basic epoch sealing."""
        blocks = [
            {
                "id": 1,
                "block_number": 0,
                "composite_attestation_root": "a" * 64,
                "proof_count": 5,
                "ui_event_count": 2,
            },
            {
                "id": 2,
                "block_number": 1,
                "composite_attestation_root": "b" * 64,
                "proof_count": 3,
                "ui_event_count": 1,
            },
        ]
        
        epoch = seal_epoch(0, blocks, "system-uuid")
        
        assert epoch["epoch_number"] == 0
        assert epoch["block_count"] == 2
        assert epoch["start_block_number"] == 0
        assert epoch["end_block_number"] == 2
        assert len(epoch["epoch_root"]) == 64
        assert epoch["system_id"] == "system-uuid"
    
    def test_seal_epoch_empty_raises(self):
        """Test that sealing empty epoch raises error."""
        with pytest.raises(ValueError, match="empty"):
            seal_epoch(0, [], "system-uuid")
    
    def test_seal_epoch_aggregates_stats(self):
        """Test that epoch sealing aggregates statistics."""
        blocks = [
            {
                "id": 1,
                "block_number": 0,
                "composite_attestation_root": "a" * 64,
                "proof_count": 5,
                "ui_event_count": 2,
            },
            {
                "id": 2,
                "block_number": 1,
                "composite_attestation_root": "b" * 64,
                "proof_count": 3,
                "ui_event_count": 1,
            },
        ]
        
        epoch = seal_epoch(0, blocks, "system-uuid")
        
        assert epoch["total_proofs"] == 8
        assert epoch["total_ui_events"] == 3


class TestEpochVerification:
    """Test epoch verification."""
    
    def test_verify_epoch_integrity_valid(self):
        """Test verification of valid epoch."""
        blocks = [
            {"composite_attestation_root": "a" * 64},
            {"composite_attestation_root": "b" * 64},
        ]
        
        epoch_root = compute_epoch_root(["a" * 64, "b" * 64])
        epoch_data = {"epoch_root": epoch_root}
        
        is_valid = verify_epoch_integrity(epoch_data, blocks)
        assert is_valid is True
    
    def test_verify_epoch_integrity_invalid(self):
        """Test verification detects invalid epoch."""
        blocks = [
            {"composite_attestation_root": "a" * 64},
            {"composite_attestation_root": "b" * 64},
        ]
        
        epoch_data = {"epoch_root": "x" * 64}
        
        is_valid = verify_epoch_integrity(epoch_data, blocks)
        assert is_valid is False
    
    def test_replay_epoch(self):
        """Test epoch replay."""
        blocks = [
            {"composite_attestation_root": "a" * 64},
            {"composite_attestation_root": "b" * 64},
        ]
        
        epoch_root = compute_epoch_root(["a" * 64, "b" * 64])
        epoch_data = {
            "epoch_number": 0,
            "epoch_root": epoch_root,
        }
        
        result = replay_epoch(epoch_data, blocks)
        
        assert result["is_valid"] is True
        assert result["epoch_number"] == 0
        assert result["block_count"] == 2
