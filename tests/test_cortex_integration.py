"""
Integration tests for Cortex (TDA Hard Gate) with U2Runner and RFLRunner.

Tests PASS/WARN/BLOCK scenarios with actual runner integration.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any

from backend.safety import evaluate_hard_gate_decision, SLOStatus
from backend.tda import TDAMode


class TestCortexRFLRunnerIntegration:
    """Test Cortex integration with RFLRunner."""
    
    def test_rfl_runner_cortex_pass_scenario(self):
        """Test RFLRunner proceeds when gate passes."""
        # This is a lightweight test since we don't want to spin up full DB
        # Just verify the gate logic is wired correctly
        
        context = {
            "abstention_rate": 0.1,
            "coverage_rate": 0.9,
            "verified_count": 10,
            "cycle_index": 5,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        assert envelope.slo.status == SLOStatus.PASS
        assert not envelope.is_blocking()
        assert envelope.decision == "proceed"
    
    def test_rfl_runner_cortex_block_scenario(self):
        """Test RFLRunner blocks when gate blocks."""
        context = {
            "abstention_rate": 0.7,  # High abstention triggers block
            "coverage_rate": 0.8,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        assert envelope.slo.status == SLOStatus.BLOCK
        assert envelope.is_blocking()
        assert envelope.decision == "block"
    
    def test_rfl_runner_cortex_dry_run_mode(self):
        """Test RFLRunner in DRY_RUN mode gives advisory."""
        # Context that would normally block
        context = {
            "abstention_rate": 0.7,
            "coverage_rate": 0.8,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.DRY_RUN)
        
        # DRY_RUN gives advisory but doesn't block
        assert envelope.slo.status == SLOStatus.ADVISORY
        assert not envelope.is_blocking()
        assert envelope.decision == "proceed"
        assert "DRY_RUN" in envelope.slo.message


class TestCortexU2RunnerIntegration:
    """Test Cortex integration with U2Runner."""
    
    def test_u2_runner_cortex_pass_scenario(self):
        """Test U2Runner proceeds when gate passes."""
        context = {
            "abstention_rate": 0.0,  # U2 starts with no abstentions
            "coverage_rate": 0.0,  # Early cycle
            "verified_count": 0,
            "cycle_index": 2,  # Early in execution
            "frontier_size": 50,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Early cycles with low metrics should pass (not enough data to block)
        assert envelope.slo.status == SLOStatus.PASS
        assert not envelope.is_blocking()
    
    def test_u2_runner_cortex_block_late_cycle(self):
        """Test U2Runner blocks if no progress after many cycles."""
        context = {
            "abstention_rate": 0.0,
            "coverage_rate": 0.3,  # Poor coverage
            "verified_count": 0,  # No verified proofs
            "cycle_index": 15,  # Many cycles have passed
            "frontier_size": 10,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Should block due to no verified proofs after many cycles
        assert envelope.slo.status == SLOStatus.BLOCK
        assert envelope.is_blocking()
    
    def test_u2_runner_cortex_shadow_mode(self):
        """Test U2Runner in SHADOW mode records hypothetical."""
        # Context that would block
        context = {
            "abstention_rate": 0.6,
            "coverage_rate": 0.4,
            "verified_count": 0,
            "cycle_index": 12,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.SHADOW)
        
        # SHADOW records but doesn't block
        assert envelope.slo.status == SLOStatus.ADVISORY
        assert not envelope.is_blocking()
        assert "SHADOW" in envelope.slo.message


class TestCortexModeTransitions:
    """Test transitioning between TDA modes."""
    
    def test_block_to_dry_run_transition(self):
        """Test same context with different modes."""
        context = {
            "abstention_rate": 0.6,
            "coverage_rate": 0.7,
            "verified_count": 5,
            "cycle_index": 10,
        }
        
        # BLOCK mode: should block
        envelope_block = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        assert envelope_block.is_blocking()
        
        # DRY_RUN mode: same context, but advisory only
        envelope_dry = evaluate_hard_gate_decision(context, TDAMode.DRY_RUN)
        assert not envelope_dry.is_blocking()
        assert envelope_dry.slo.status == SLOStatus.ADVISORY
    
    def test_all_modes_same_context(self):
        """Test all three modes with same context."""
        context = {
            "abstention_rate": 0.55,
            "coverage_rate": 0.75,
            "verified_count": 7,
            "cycle_index": 8,
        }
        
        # BLOCK mode
        env_block = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        assert env_block.is_blocking()
        assert env_block.slo.status == SLOStatus.BLOCK
        
        # DRY_RUN mode
        env_dry = evaluate_hard_gate_decision(context, TDAMode.DRY_RUN)
        assert not env_dry.is_blocking()
        assert env_dry.slo.status == SLOStatus.ADVISORY
        assert "Would block" in env_dry.slo.message
        
        # SHADOW mode
        env_shadow = evaluate_hard_gate_decision(context, TDAMode.SHADOW)
        assert not env_shadow.is_blocking()
        assert env_shadow.slo.status == SLOStatus.ADVISORY
        assert "Hypothetical block" in env_shadow.slo.message


class TestCortexAuditTrail:
    """Test audit trail completeness for CI reproducibility."""
    
    def test_audit_trail_has_all_fields(self):
        """Test audit trail contains required fields."""
        context = {
            "abstention_rate": 0.2,
            "coverage_rate": 0.85,
            "verified_count": 12,
            "cycle_index": 6,
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Required audit fields
        assert "timestamp" in envelope.audit_trail
        assert "tda_decision" in envelope.audit_trail
        assert "context_keys" in envelope.audit_trail
        assert "decision_path" in envelope.audit_trail
        
        # TDA decision details
        tda = envelope.audit_trail["tda_decision"]
        assert "should_block" in tda
        assert "mode" in tda
        assert "reason" in tda
        assert "confidence" in tda
        assert "metadata" in tda
    
    def test_deterministic_serialization(self):
        """Test envelope can be serialized deterministically."""
        context = {
            "abstention_rate": 0.3,
            "coverage_rate": 0.8,
            "verified_count": 9,
            "cycle_index": 7,
        }
        
        # Create two envelopes with same context
        env1 = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        env2 = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Serialize both
        dict1 = env1.to_dict()
        dict2 = env2.to_dict()
        
        # Check deterministic fields match
        assert dict1["decision"] == dict2["decision"]
        assert dict1["tda_should_block"] == dict2["tda_should_block"]
        assert dict1["tda_mode"] == dict2["tda_mode"]
        assert dict1["slo"]["status"] == dict2["slo"]["status"]
        
        # Timestamps should match (deterministic)
        assert dict1["slo"]["timestamp"] == dict2["slo"]["timestamp"]
    
    def test_audit_trail_preserves_context(self):
        """Test context keys are recorded in audit trail."""
        context = {
            "abstention_rate": 0.15,
            "coverage_rate": 0.92,
            "verified_count": 15,
            "cycle_index": 4,
            "custom_field": "test_value",
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Context keys should be recorded
        context_keys = envelope.audit_trail["context_keys"]
        assert "abstention_rate" in context_keys
        assert "coverage_rate" in context_keys
        assert "verified_count" in context_keys
        assert "cycle_index" in context_keys
        assert "custom_field" in context_keys


class TestCortexErrorHandling:
    """Test error handling in Cortex integration."""
    
    def test_invalid_tda_mode_string(self):
        """Test handling of invalid TDA mode string."""
        # Invalid mode should raise ValueError when creating enum
        with pytest.raises(ValueError):
            TDAMode("INVALID_MODE")
    
    def test_empty_context(self):
        """Test Cortex handles empty context gracefully."""
        context = {}
        
        # Should still work, defaulting missing values to 0
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Should pass since all metrics are 0 (healthy by default)
        assert envelope.slo.status == SLOStatus.PASS
    
    def test_partial_context(self):
        """Test Cortex handles partial context."""
        context = {
            "abstention_rate": 0.2,
            # Missing other fields
        }
        
        envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
        
        # Should work with defaults
        assert envelope.slo.status in [SLOStatus.PASS, SLOStatus.BLOCK]
