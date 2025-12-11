"""
Integration Tests for Cortex Telemetry
======================================

End-to-end tests demonstrating Cortex telemetry flow:
1. First Light summary generation
2. Uplift Safety adapter integration
3. Evidence Pack governance tile

These tests show how Cortex decisions surface through the RFL infrastructure
without altering gating semantics (Cortex still owns the gate).
"""

import json
import pytest
from typing import Dict, Any

from rfl.cortex_telemetry import (
    CortexEnvelope,
    CortexDecision,
    TDAMode,
    compute_cortex_summary,
    UpliftSafetyCortexAdapter,
    attach_cortex_governance_to_evidence,
)


class TestFirstLightIntegration:
    """Test First Light summary generation with Cortex telemetry."""
    
    def test_first_light_summary_with_cortex(self):
        """Test that First Light summary includes cortex_summary."""
        # Simulate a run with some Cortex decisions
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        # Add some decisions
        for i in range(5):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item_{i}",
                blocked=False,
                advisory=(i < 2),  # 2 advisory violations
                rationale=f"Advisory for item {i}" if i < 2 else "OK",
                tda_mode=TDAMode.DRY_RUN,
            ))
        
        # Generate First Light summary
        first_light_result = compute_cortex_summary(envelope)
        
        assert "cortex_summary" in first_light_result
        summary = first_light_result["cortex_summary"]
        
        # Verify structure
        assert summary["total_decisions"] == 5
        assert summary["blocked_decisions"] == 0
        assert summary["advisory_decisions"] == 2
        assert summary["tda_mode"] == "DRY_RUN"
        assert summary["hard_gate_status"] == "WARN"  # Advisory violations in DRY_RUN mode
    
    def test_first_light_summary_no_violations(self):
        """Test First Light summary with no violations."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        # No decisions added
        first_light_result = compute_cortex_summary(envelope)
        summary = first_light_result["cortex_summary"]
        
        assert summary["total_decisions"] == 0
        assert summary["blocked_decisions"] == 0
        assert summary["advisory_decisions"] == 0
        assert summary["tda_mode"] == "BLOCK"
        assert summary["hard_gate_status"] == "OK"
    
    def test_first_light_summary_block_mode(self):
        """Test First Light summary with blocked decisions in BLOCK mode."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="bad_item",
            blocked=True,
            advisory=False,
            rationale="Hard violation detected",
            tda_mode=TDAMode.BLOCK,
        ))
        
        first_light_result = compute_cortex_summary(envelope)
        summary = first_light_result["cortex_summary"]
        
        assert summary["hard_gate_status"] == "BLOCK"
        assert summary["blocked_decisions"] == 1


class TestUpliftSafetyIntegration:
    """Test Uplift Safety Engine adapter integration."""
    
    def test_uplift_safety_adapter_low_band(self):
        """Test uplift safety adapter with LOW band (no violations)."""
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "LOW"
        assert adapter.hypothetical_block_rate == 0.0
        assert adapter.advisory_only is True
    
    def test_uplift_safety_adapter_medium_band(self):
        """Test uplift safety adapter with MEDIUM band (advisory violations)."""
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        for i in range(10):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item_{i}",
                blocked=False,
                advisory=(i < 3),  # 3 out of 10 are advisory
                rationale=f"Advisory {i}" if i < 3 else "OK",
                tda_mode=TDAMode.DRY_RUN,
            ))
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "MEDIUM"
        assert adapter.hypothetical_block_rate == pytest.approx(0.3)
        assert adapter.advisory_only is True
    
    def test_uplift_safety_adapter_high_band(self):
        """Test uplift safety adapter with HIGH band (hard blocks)."""
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        for i in range(10):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item_{i}",
                blocked=(i < 2),  # 2 out of 10 are blocked
                advisory=False,
                rationale=f"Blocked {i}" if i < 2 else "OK",
                tda_mode=TDAMode.BLOCK,
            ))
        
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        
        assert adapter.cortex_gate_band == "HIGH"
        assert adapter.hypothetical_block_rate == pytest.approx(0.2)
        assert adapter.advisory_only is False
    
    def test_uplift_safety_adapter_serialization(self):
        """Test that uplift safety adapter serializes correctly."""
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
        adapter_dict = adapter.to_dict()
        
        # Verify JSON serializable
        json_str = json.dumps(adapter_dict)
        assert json_str is not None
        
        # Verify structure
        deserialized = json.loads(json_str)
        assert deserialized["cortex_gate_band"] == "MEDIUM"
        assert "hypothetical_block_rate" in deserialized
        assert "advisory_only" in deserialized


class TestEvidencePackIntegration:
    """Test Evidence Pack governance tile integration."""
    
    def test_evidence_pack_cortex_governance(self):
        """Test attaching cortex governance to evidence pack."""
        # Create a minimal evidence pack
        evidence = {
            "version": "1.0.0",
            "experiment": {
                "id": "test_exp_001",
                "type": "rfl_experiment"
            },
            "artifacts": {
                "logs": [],
                "figures": []
            }
        }
        
        # Create Cortex envelope with decisions
        envelope = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        envelope.add_decision(CortexDecision(
            decision_id="d001",
            cycle_id=1,
            item="item1",
            blocked=True,
            advisory=False,
            rationale="Hard violation: item1 exceeds safety threshold",
            tda_mode=TDAMode.BLOCK,
        ))
        
        envelope.add_decision(CortexDecision(
            decision_id="d002",
            cycle_id=2,
            item="item2",
            blocked=False,
            advisory=True,
            rationale="Soft warning: item2 approaching threshold",
            tda_mode=TDAMode.BLOCK,
        ))
        
        # Attach cortex governance
        evidence_with_cortex = attach_cortex_governance_to_evidence(evidence, envelope)
        
        # Verify governance tile exists
        assert "governance" in evidence_with_cortex
        assert "cortex_gate" in evidence_with_cortex["governance"]
        
        cortex_gate = evidence_with_cortex["governance"]["cortex_gate"]
        
        # Verify cortex_gate structure
        assert cortex_gate["hard_gate_status"] == "BLOCK"
        assert cortex_gate["blocked_decisions"] == 1
        assert cortex_gate["tda_mode"] == "BLOCK"
        assert cortex_gate["total_decisions"] == 2
        assert cortex_gate["advisory_decisions"] == 1
        
        # Verify rationales
        assert len(cortex_gate["rationales"]) == 2
        assert "Hard violation" in cortex_gate["rationales"][0]
        assert "Soft warning" in cortex_gate["rationales"][1]
    
    def test_evidence_pack_cortex_governance_preserves_existing(self):
        """Test that cortex governance doesn't overwrite existing governance."""
        evidence = {
            "version": "1.0.0",
            "experiment": {"id": "test_exp"},
            "governance": {
                "approval": "approved",
                "reviewer": "alice"
            }
        }
        
        envelope = CortexEnvelope(tda_mode=TDAMode.SHADOW)
        evidence_with_cortex = attach_cortex_governance_to_evidence(evidence, envelope)
        
        # Verify existing governance preserved
        assert evidence_with_cortex["governance"]["approval"] == "approved"
        assert evidence_with_cortex["governance"]["reviewer"] == "alice"
        
        # Verify cortex_gate added
        assert "cortex_gate" in evidence_with_cortex["governance"]


class TestEndToEndFlow:
    """Test complete end-to-end flow of Cortex telemetry."""
    
    def test_complete_cortex_telemetry_flow(self):
        """
        Test complete flow:
        1. Create Cortex envelope with decisions
        2. Generate First Light summary
        3. Generate Uplift Safety adapter
        4. Attach to Evidence Pack
        """
        # Step 1: Create Cortex envelope
        envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)
        
        for i in range(20):
            envelope.add_decision(CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item_{i}",
                blocked=False,
                advisory=(i % 5 == 0),  # 4 out of 20 are advisory
                rationale=f"Advisory {i}" if i % 5 == 0 else "OK",
                tda_mode=TDAMode.DRY_RUN,
            ))
        
        # Step 2: Generate First Light summary
        first_light = compute_cortex_summary(envelope)
        assert first_light["cortex_summary"]["total_decisions"] == 20
        assert first_light["cortex_summary"]["advisory_decisions"] == 4
        assert first_light["cortex_summary"]["hard_gate_status"] == "WARN"
        
        # Step 3: Generate Uplift Safety adapter
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        assert adapter.cortex_gate_band == "MEDIUM"
        assert adapter.hypothetical_block_rate == pytest.approx(0.2)
        assert adapter.advisory_only is True
        
        # Step 4: Attach to Evidence Pack
        evidence = {
            "version": "1.0.0",
            "experiment": {"id": "integration_test"},
        }
        
        evidence_with_cortex = attach_cortex_governance_to_evidence(evidence, envelope)
        
        cortex_gate = evidence_with_cortex["governance"]["cortex_gate"]
        assert cortex_gate["total_decisions"] == 20
        assert cortex_gate["advisory_decisions"] == 4
        assert len(cortex_gate["rationales"]) == 3  # Limited to top 3
        
        # Verify complete result is JSON serializable
        complete_result = {
            "first_light": first_light,
            "uplift_safety": adapter.to_dict(),
            "evidence": evidence_with_cortex,
        }
        
        json_str = json.dumps(complete_result)
        assert json_str is not None
        
        # Verify deserialization
        deserialized = json.loads(json_str)
        assert deserialized["first_light"]["cortex_summary"]["total_decisions"] == 20
        assert deserialized["uplift_safety"]["cortex_gate_band"] == "MEDIUM"
        assert deserialized["evidence"]["governance"]["cortex_gate"]["hard_gate_status"] == "WARN"
    
    def test_cortex_telemetry_determinism(self):
        """Test that complete flow is deterministic."""
        # Create identical envelopes
        envelope1 = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        envelope2 = CortexEnvelope(tda_mode=TDAMode.BLOCK)
        
        for i in range(5):
            decision = CortexDecision(
                decision_id=f"d{i:03d}",
                cycle_id=i,
                item=f"item_{i}",
                blocked=(i == 0),
                advisory=False,
                rationale=f"Decision {i}",
                tda_mode=TDAMode.BLOCK,
            )
            envelope1.add_decision(decision)
            envelope2.add_decision(decision)
        
        # Generate outputs
        first_light1 = compute_cortex_summary(envelope1)
        first_light2 = compute_cortex_summary(envelope2)
        
        adapter1 = UpliftSafetyCortexAdapter.from_envelope(envelope1)
        adapter2 = UpliftSafetyCortexAdapter.from_envelope(envelope2)
        
        evidence = {"version": "1.0.0"}
        evidence1 = attach_cortex_governance_to_evidence(evidence, envelope1)
        evidence2 = attach_cortex_governance_to_evidence(evidence, envelope2)
        
        # Verify determinism
        assert json.dumps(first_light1, sort_keys=True) == json.dumps(first_light2, sort_keys=True)
        assert json.dumps(adapter1.to_dict(), sort_keys=True) == json.dumps(adapter2.to_dict(), sort_keys=True)
        assert json.dumps(evidence1, sort_keys=True) == json.dumps(evidence2, sort_keys=True)


class TestCortexTelemetryInvariantsMaintenance:
    """Test that Cortex telemetry maintains key invariants."""
    
    def test_no_gating_semantics_alteration(self):
        """
        Test that telemetry module does NOT alter gating semantics.
        
        This is a critical invariant: Cortex owns the gate, telemetry only surfaces.
        """
        # Create envelope with blocked decision
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
        
        # Generate all telemetry outputs
        first_light = compute_cortex_summary(envelope)
        adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
        evidence = attach_cortex_governance_to_evidence({"version": "1.0.0"}, envelope)
        
        # Verify envelope state unchanged
        assert envelope.total_decisions() == 1
        assert envelope.blocked_decisions() == 1
        assert envelope.tda_mode == TDAMode.BLOCK
        
        # Verify telemetry only reads, doesn't alter
        # (This test demonstrates the design principle: read-only access)
    
    def test_cortex_envelope_immutability_in_evidence(self):
        """Test that attaching to evidence doesn't mutate the envelope."""
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
        
        # Capture state
        initial_total = envelope.total_decisions()
        initial_advisory = envelope.advisory_decisions()
        
        # Attach to evidence multiple times
        for _ in range(3):
            attach_cortex_governance_to_evidence({"version": "1.0.0"}, envelope)
        
        # Verify envelope unchanged
        assert envelope.total_decisions() == initial_total
        assert envelope.advisory_decisions() == initial_advisory
