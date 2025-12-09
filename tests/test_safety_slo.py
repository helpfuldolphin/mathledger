"""
Tests for Safety SLO and TDA integration.

Covers PASS/WARN/BLOCK scenarios with TDA influence.
"""

import pytest
from datetime import datetime

from backend.safety import (
    SafetyEnvelope,
    SafetySLO,
    SLOStatus,
    evaluate_hard_gate_decision,
    check_gate_decision,
)
from backend.tda import TDAMode


class TestSafetySLO:
    """Test Safety SLO creation, serialization, and status handling."""
    
    def test_slo_creation(self):
        """Test creating a Safety SLO."""
        slo = SafetySLO(
            status=SLOStatus.PASS,
            message="Test passed",
            metadata={"test": "value"},
            timestamp="2024-01-01T00:00:00Z",
        )
        
        assert slo.status == SLOStatus.PASS
        assert slo.message == "Test passed"
        assert slo.metadata["test"] == "value"
        assert slo.timestamp == "2024-01-01T00:00:00Z"
    
    def test_slo_serialization(self):
        """Test SLO serialization and deserialization."""
        slo = SafetySLO(
            status=SLOStatus.WARN,
            message="Warning message",
            metadata={"key": "value"},
        )
        
        # Serialize
        data = slo.to_dict()
        assert data["status"] == "warn"
        assert data["message"] == "Warning message"
        assert data["metadata"]["key"] == "value"
        
        # Deserialize
        restored = SafetySLO.from_dict(data)
        assert restored.status == SLOStatus.WARN
        assert restored.message == "Warning message"
        assert restored.metadata["key"] == "value"


class TestSafetyEnvelope:
    """Test Safety Envelope functionality."""
    
    def test_envelope_creation(self):
        """Test creating a Safety Envelope."""
        slo = SafetySLO(status=SLOStatus.PASS, message="Test")
        envelope = SafetyEnvelope(
            slo=slo,
            tda_should_block=False,
            tda_mode="BLOCK",
            decision="proceed",
        )
        
        assert envelope.slo.status == SLOStatus.PASS
        assert not envelope.tda_should_block
        assert envelope.tda_mode == "BLOCK"
        assert envelope.decision == "proceed"
        assert not envelope.is_blocking()
    
    def test_envelope_blocking(self):
        """Test envelope blocking detection."""
        slo = SafetySLO(status=SLOStatus.BLOCK, message="Blocked")
        envelope = SafetyEnvelope(
            slo=slo,
            tda_should_block=True,
            tda_mode="BLOCK",
            decision="block",
        )
        
        assert envelope.is_blocking()
    
    def test_envelope_serialization(self):
        """Test envelope serialization."""
        slo = SafetySLO(status=SLOStatus.PASS, message="OK")
        envelope = SafetyEnvelope(
            slo=slo,
            tda_should_block=False,
            tda_mode="DRY_RUN",
            decision="proceed",
            audit_trail={"test": "data"},
        )
        
        data = envelope.to_dict()
        assert data["decision"] == "proceed"
        assert data["tda_mode"] == "DRY_RUN"
        assert data["audit_trail"]["test"] == "data"
        
        restored = SafetyEnvelope.from_dict(data)
        assert restored.decision == "proceed"
        assert restored.tda_mode == "DRY_RUN"


class TestCortexIntegration:
    """Test Cortex (evaluate_hard_gate_decision) integration."""
    
    def test_pass_scenario_block_mode(self):
        """Test PASS scenario in BLOCK mode."""
        context = {
            "abstention_rate": 0.1,
            "coverage_rate": 0.9,
            "verified_count": 10,
            "cycle_index": 5,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.BLOCK,
        )
        
        assert envelope.slo.status == SLOStatus.PASS
        assert not envelope.tda_should_block
        assert envelope.decision == "proceed"
        assert not envelope.is_blocking()
        assert "PASS" in envelope.slo.message
    
    def test_block_scenario_high_abstention(self):
        """Test BLOCK scenario with high abstention rate."""
        context = {
            "abstention_rate": 0.6,  # > 50% triggers block
            "coverage_rate": 0.8,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.BLOCK,
        )
        
        assert envelope.slo.status == SLOStatus.BLOCK
        assert envelope.tda_should_block
        assert envelope.decision == "block"
        assert envelope.is_blocking()
        assert "BLOCKED" in envelope.slo.message
        assert "abstention" in envelope.slo.message.lower()
    
    def test_block_scenario_poor_coverage(self):
        """Test BLOCK scenario with poor coverage."""
        context = {
            "abstention_rate": 0.2,
            "coverage_rate": 0.4,  # < 50% after 10 cycles triggers block
            "verified_count": 3,
            "cycle_index": 15,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.BLOCK,
        )
        
        assert envelope.slo.status == SLOStatus.BLOCK
        assert envelope.tda_should_block
        assert envelope.decision == "block"
        assert envelope.is_blocking()
    
    def test_block_scenario_no_verified_proofs(self):
        """Test BLOCK scenario with no verified proofs."""
        context = {
            "abstention_rate": 0.3,
            "coverage_rate": 0.7,
            "verified_count": 0,  # No proofs after 5 cycles triggers block
            "cycle_index": 8,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.BLOCK,
        )
        
        assert envelope.slo.status == SLOStatus.BLOCK
        assert envelope.tda_should_block
        assert envelope.decision == "block"
    
    def test_dry_run_mode_advisory(self):
        """Test DRY_RUN mode provides advisory without blocking."""
        # Context that would normally block
        context = {
            "abstention_rate": 0.7,
            "coverage_rate": 0.5,
            "verified_count": 0,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.DRY_RUN,
        )
        
        # In DRY_RUN, we get advisory status but don't block
        assert envelope.slo.status == SLOStatus.ADVISORY
        assert envelope.tda_should_block  # TDA says block
        assert envelope.decision == "proceed"  # But we proceed anyway
        assert not envelope.is_blocking()
        assert "DRY_RUN" in envelope.slo.message
        assert "Would block" in envelope.slo.message
    
    def test_shadow_mode_records_hypothetical(self):
        """Test SHADOW mode records hypothetical status."""
        # Context that would block
        context = {
            "abstention_rate": 0.6,
            "coverage_rate": 0.8,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(
            context=context,
            tda_mode=TDAMode.SHADOW,
        )
        
        # SHADOW mode records but doesn't block
        assert envelope.slo.status == SLOStatus.ADVISORY
        assert envelope.tda_should_block
        assert envelope.decision == "proceed"
        assert not envelope.is_blocking()
        assert "SHADOW" in envelope.slo.message
        assert "Hypothetical block" in envelope.slo.message
    
    def test_check_gate_decision_allows_proceed(self):
        """Test check_gate_decision allows proceeding when not blocked."""
        context = {
            "abstention_rate": 0.1,
            "coverage_rate": 0.9,
            "verified_count": 10,
            "cycle_index": 5,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Should not raise
        check_gate_decision(envelope)
    
    def test_check_gate_decision_blocks_execution(self):
        """Test check_gate_decision blocks when envelope says block."""
        context = {
            "abstention_rate": 0.7,
            "coverage_rate": 0.8,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Hard gate BLOCKED"):
            check_gate_decision(envelope)
    
    def test_deterministic_timestamp(self):
        """Test that envelope uses deterministic timestamp."""
        context = {
            "abstention_rate": 0.1,
            "coverage_rate": 0.9,
            "verified_count": 10,
            "cycle_index": 5,
        }
        
        # Call twice with same context
        envelope1 = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        envelope2 = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Timestamps should be identical (deterministic)
        assert envelope1.slo.timestamp == envelope2.slo.timestamp
    
    def test_audit_trail_completeness(self):
        """Test that audit trail contains all required information."""
        context = {
            "abstention_rate": 0.2,
            "coverage_rate": 0.85,
            "verified_count": 7,
            "cycle_index": 3,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Check audit trail has required fields
        assert "timestamp" in envelope.audit_trail
        assert "tda_decision" in envelope.audit_trail
        assert "context_keys" in envelope.audit_trail
        assert "decision_path" in envelope.audit_trail
        
        # Check TDA decision is properly recorded
        tda_data = envelope.audit_trail["tda_decision"]
        assert "should_block" in tda_data
        assert "mode" in tda_data
        assert "reason" in tda_data
        assert "confidence" in tda_data
        
        # Check decision path
        assert "BLOCK" in envelope.audit_trail["decision_path"]
    
    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip for CI persistence."""
        context = {
            "abstention_rate": 0.3,
            "coverage_rate": 0.75,
            "verified_count": 8,
            "cycle_index": 6,
        }
        
        # Create envelope
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Serialize to dict
        data = envelope.to_dict()
        
        # Deserialize
        restored = SafetyEnvelope.from_dict(data)
        
        # Check all critical fields preserved
        assert restored.decision == envelope.decision
        assert restored.tda_should_block == envelope.tda_should_block
        assert restored.tda_mode == envelope.tda_mode
        assert restored.slo.status == envelope.slo.status
        assert restored.slo.message == envelope.slo.message
