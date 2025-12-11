"""
Tests for Cortex Telemetry Module
==================================

Verifies that Cortex gate decisions are correctly surfaced into:
- First Light summaries
- Uplift Safety Engine adapters
- Evidence Pack governance tiles

Tests ensure:
1. Determinism — same inputs produce same outputs
2. JSON compatibility — all data structures serialize correctly
3. No mutation — input evidence/envelopes are never mutated
4. Correct state mapping — gate states map to correct summary statuses
"""

import copy
import json
import pytest
from typing import Dict, Any

from rfl.cortex_telemetry import (
    TDAMode,
    HardGateStatus,
    CortexDecision,
    CortexEnvelope,
    CortexSummary,
    UpliftSafetyCortexAdapter,
    compute_cortex_summary,
    attach_cortex_governance_to_evidence,
)


class TestTDAMode:
    """Test TDA mode enum."""
    
    def test_tda_mode_values(self):
        """Verify TDA mode enum values."""
        assert TDAMode.BLOCK.value == "BLOCK"
        assert TDAMode.DRY_RUN.value == "DRY_RUN"
        assert TDAMode.SHADOW.value == "SHADOW"
    
    def test_tda_mode_string_conversion(self):
        """Verify TDA modes can be used as strings."""
        mode = TDAMode.BLOCK
        assert str(mode) == "TDAMode.BLOCK"  # str() includes enum name
        assert mode.value == "BLOCK"  # .value gives just the value


class TestHardGateStatus:
    """Test hard gate status enum."""
    
    def test_hard_gate_status_values(self):
        """Verify hard gate status enum values."""
        assert HardGateStatus.OK.value == "OK"
        assert HardGateStatus.WARN.value == "WARN"
        assert HardGateStatus.BLOCK.value == "BLOCK"


class TestCortexDecision:
    """Test individual Cortex decision records."""
    
    def test_decision_creation(self):
        """Test creating a Cortex decision."""
        decision = CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="test_item",
            blocked=False,
            advisory=True,
            rationale="Advisory violation detected",
            tda_mode=TDAMode.DRY_RUN,
            timestamp="2025-12-11T00:00:00Z",
        )
        
        assert decision.decision_id == "d001"
        assert decision.blocked is False
        assert decision.advisory is True
        assert decision.tda_mode == TDAMode.DRY_RUN
    
    def test_decision_to_dict(self):
        """Test JSON serialization of decision."""
        decision = CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="test_item",
            blocked=True,
            advisory=False,
            rationale="Hard block triggered",
            tda_mode=TDAMode.BLOCK,
        )
        
        data = decision.to_dict()
        
        assert data["decision_id"] == "d001"
        assert data["blocked"] is True
        assert data["tda_mode"] == "BLOCK"
        
        # Verify JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None


class TestCortexEnvelope:
    """Test Cortex decision envelope."""
    
    def test_empty_envelope(self):
        """Test creating an empty envelope."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        assert envelope.total_decisions() == 0
        assert envelope.blocked_decisions() == 0
        assert envelope.advisory_decisions() == 0
        assert envelope.compute_hard_gate_status() == HardGateStatus.OK
    
    def test_add_decisions(self):
        """Test adding decisions to envelope."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        decision1 = CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        )
        
        decision2 = CortexDecision(
            decision_id="d002",
            cycle_id=2,
            item="item2",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.DRY_RUN,
        )
        
        envelope.add_decision(decision1)
        envelope.add_decision(decision2)
        
        assert envelope.total_decisions() == 2
        assert envelope.blocked_decisions() == 1
        assert envelope.advisory_decisions() == 1
    
    def test_hard_gate_status_block_mode_with_blocks(self):
        """Test hard gate status: BLOCK mode with blocked decisions."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Hard block",
            tda_mode=TDAMode.BLOCK,
        ))
        
        assert envelope.compute_hard_gate_status() == HardGateStatus.BLOCK
    
    def test_hard_gate_status_dry_run_mode_with_violations(self):
        """Test hard gate status: DRY_RUN mode with violations."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        ))
        
        assert envelope.compute_hard_gate_status() == HardGateStatus.WARN
    
    def test_hard_gate_status_shadow_mode_with_violations(self):
        """Test hard gate status: SHADOW mode with violations."""
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Shadow advisory",
            tda_mode=TDAMode.SHADOW,
        ))
        
        assert envelope.compute_hard_gate_status() == HardGateStatus.WARN
    
    def test_hard_gate_status_no_violations(self):
        """Test hard gate status: no violations."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        # No decisions added
        assert envelope.compute_hard_gate_status() == HardGateStatus.OK
    
    def test_envelope_to_dict(self):
        """Test JSON serialization of envelope."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        data = envelope.to_dict()
        
        assert data["tda_mode"] == "BLOCK"
        assert len(data["decisions"]) == 1
        
        # Verify JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None


class TestCortexSummary:
    """Test First Light cortex summary."""
    
    def test_summary_from_envelope(self):
        """Test creating summary from envelope."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        envelope.add_decision(CortexDecision(
            decision_id="d002",
            cycle_id=2,
            item="item2",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.BLOCK,
        ))
        
        summary = CortexSummary.from_envelope(envelope)
        
        assert summary.total_decisions == 2
        assert summary.blocked_decisions == 1
        assert summary.advisory_decisions == 1
        assert summary.tda_mode == "BLOCK"
        assert summary.hard_gate_status == "BLOCK"
    
    def test_summary_to_dict(self):
        """Test JSON serialization of summary."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        summary = CortexSummary.from_envelope(envelope)
        
        data = summary.to_dict()
        
        assert "total_decisions" in data
        assert "blocked_decisions" in data
        assert "advisory_decisions" in data
        assert "tda_mode" in data
        assert "hard_gate_status" in data
        
        # Verify JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None
    
    def test_compute_cortex_summary(self):
        """Test compute_cortex_summary helper function."""
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        
        result = compute_cortex_summary(envelope)
        
        assert "cortex_summary" in result
        assert result["cortex_summary"]["tda_mode"] == "SHADOW"
        assert result["cortex_summary"]["hard_gate_status"] == "OK"


class TestUpliftSafetyCortexAdapter:
    """Test Uplift Safety Engine adapter."""
    
    def test_adapter_high_band(self):
        """Test HIGH band: blocked decisions in BLOCK mode."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "HIGH"
        assert adapter.hypothetical_block_rate == 1.0
        assert adapter.advisory_only is False
    
    def test_adapter_medium_band(self):
        """Test MEDIUM band: advisory decisions."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        ))
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "MEDIUM"
        assert adapter.hypothetical_block_rate == 1.0
        assert adapter.advisory_only is True
    
    def test_adapter_low_band(self):
        """Test LOW band: no violations."""
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "LOW"
        assert adapter.hypothetical_block_rate == 0.0
        assert adapter.advisory_only is True
    
    def test_adapter_hypothetical_block_rate(self):
        """Test hypothetical block rate calculation."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        # 2 out of 5 decisions violate
        for i in range(5):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item{i}",
                blocked=False,
                advisory=(i < 2),
                rationale="Advisory" if i < 2 else "OK",
                tda_mode=TDAMode.DRY_RUN,
            ))
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.hypothetical_block_rate == pytest.approx(0.4)
    
    def test_adapter_to_dict(self):
        """Test JSON serialization of adapter."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        data = adapter.to_dict()
        
        assert "cortex_gate_band" in data
        assert "hypothetical_block_rate" in data
        assert "advisory_only" in data
        
        # Verify JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None


class TestEvidencePackIntegration:
    """Test evidence pack cortex_gate tile attachment."""
    
    def test_attach_cortex_governance_basic(self):
        """Test basic attachment of cortex governance."""
        evidence = {
            "version": "1.0.0",
            "experiment": {"id": "test_exp"},
        }
        
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Hard block triggered",
            tda_mode=TDAMode.BLOCK,
        ))
        
        result = attach_cortex_governance_to_evidence(evidence, envelope)
        
        assert "governance" in result
        assert "cortex_gate" in result["governance"]
        
        cortex_gate = result["governance"]["cortex_gate"]
        assert cortex_gate["hard_gate_status"] == "BLOCK"
        assert cortex_gate["blocked_decisions"] == 1
        assert cortex_gate["tda_mode"] == "BLOCK"
        assert len(cortex_gate["rationales"]) == 1
        assert cortex_gate["rationales"][0] == "Hard block triggered"
    
    def test_attach_cortex_governance_no_mutation(self):
        """Test that original evidence is not mutated."""
        evidence = {
            "version": "1.0.0",
            "experiment": {"id": "test_exp"},
        }
        
        evidence_copy = copy.deepcopy(evidence)
        
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        ))
        
        result = attach_cortex_governance_to_evidence(evidence, envelope)
        
        # Original evidence should be unchanged
        assert evidence == evidence_copy
        
        # Result should have governance added
        assert "governance" in result
        assert "governance" not in evidence
    
    def test_attach_cortex_governance_existing_governance(self):
        """Test attaching cortex governance to evidence with existing governance."""
        evidence = {
            "version": "1.0.0",
            "experiment": {"id": "test_exp"},
            "governance": {
                "existing_key": "existing_value"
            }
        }
        
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        
        result = attach_cortex_governance_to_evidence(evidence, envelope)
        
        assert "governance" in result
        assert "existing_key" in result["governance"]
        assert "cortex_gate" in result["governance"]
        assert result["governance"]["existing_key"] == "existing_value"
    
    def test_attach_cortex_governance_rationale_limit(self):
        """Test that rationales are limited to top 3."""
        evidence = {"version": "1.0.0"}
        
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        for i in range(10):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item{i}",
                blocked=False,
                advisory=True,
                rationale=f"Rationale {i}",
                tda_mode=TDAMode.DRY_RUN,
            ))
        
        result = attach_cortex_governance_to_evidence(evidence, envelope)
        
        rationales = result["governance"]["cortex_gate"]["rationales"]
        assert len(rationales) == 3
        assert rationales == ["Rationale 0", "Rationale 1", "Rationale 2"]
    
    def test_attach_cortex_governance_json_serializable(self):
        """Test that evidence pack with cortex_gate is JSON serializable."""
        evidence = {"version": "1.0.0"}
        
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        result = attach_cortex_governance_to_evidence(evidence, envelope)
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["governance"]["cortex_gate"]["hard_gate_status"] == "BLOCK"


class TestDeterminism:
    """Test deterministic behavior of cortex telemetry."""
    
    def test_deterministic_summary_generation(self):
        """Test that same envelope produces same summary."""
        envelope1 = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        envelope1.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        envelope2 = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        envelope2.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Blocked",
            tda_mode=TDAMode.BLOCK,
        ))
        
        summary1 = CortexSummary.from_envelope(envelope1)
        summary2 = CortexSummary.from_envelope(envelope2)
        
        assert summary1.to_dict() == summary2.to_dict()
    
    def test_deterministic_adapter_generation(self):
        """Test that same envelope produces same adapter."""
        envelope1 = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        envelope1.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        ))
        
        envelope2 = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        envelope2.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Advisory",
            tda_mode=TDAMode.DRY_RUN,
        ))
        
        adapter1 = UpliftSafetyCortexAdapter.from_envelope(envelope1)
        adapter2 = UpliftSafetyCortexAdapter.from_envelope(envelope2)
        
        assert adapter1.to_dict() == adapter2.to_dict()
    
    def test_deterministic_evidence_attachment(self):
        """Test that same inputs produce same evidence output."""
        evidence = {"version": "1.0.0"}
        
        envelope1 = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        envelope1.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Shadow",
            tda_mode=TDAMode.SHADOW,
        ))
        
        envelope2 = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        envelope2.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=False,
            advisory=True,
            rationale="Shadow",
            tda_mode=TDAMode.SHADOW,
        ))
        
        result1 = attach_cortex_governance_to_evidence(evidence, envelope1)
        result2 = attach_cortex_governance_to_evidence(evidence, envelope2)
        
        # Convert to JSON and compare for determinism
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)
        
        assert json1 == json2
