#!/usr/bin/env python3
"""
Test Suite for Replay Safety Governance Signal Adapter
=======================================================

Comprehensive tests verifying the BLOCK/WARN/OK logic for the unified
governance signal adapter.

Test Coverage:
- BLOCK when either side BLOCKs
- BLOCK when alignment is DIVERGENT
- WARN on TENSION cases
- OK when both OK and aligned
- Reason consolidation with prefixes
- Evidence pack harmonization
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.governance.replay_safety_signal import (
    SafetyEvaluation,
    RadarView,
    to_governance_signal_for_replay_safety,
    extend_evidence_pack_with_governance_status,
    assess_alignment,
    AlignmentStatus,
    GovernanceStatus,
)


class TestAlignmentAssessment:
    """Test alignment assessment between Safety and Radar."""
    
    def test_aligned_both_ok(self):
        """Test ALIGNED when both report OK."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain intact"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.ALIGNED
    
    def test_aligned_both_warn(self):
        """Test ALIGNED when both report WARN."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Determinism below threshold"]
        )
        radar = RadarView(
            status="WARN",
            threading_intact=True,
            signature_valid=False,
            reasons=["Signature format issue"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.ALIGNED
    
    def test_aligned_both_block(self):
        """Test ALIGNED when both report BLOCK."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="BLOCK",
            threading_intact=False,
            signature_valid=False,
            reasons=["Chain broken"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.ALIGNED
    
    def test_tension_safety_warn_radar_ok(self):
        """Test TENSION when Safety=WARN and Radar=OK."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Minor determinism issue"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.TENSION
    
    def test_tension_safety_ok_radar_warn(self):
        """Test TENSION when Safety=OK and Radar=WARN."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="WARN",
            threading_intact=True,
            signature_valid=False,
            reasons=["Signature format issue"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.TENSION
    
    def test_divergent_safety_block_radar_ok(self):
        """Test DIVERGENT when Safety=BLOCK and Radar=OK."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.DIVERGENT
    
    def test_divergent_safety_ok_radar_block(self):
        """Test DIVERGENT when Safety=OK and Radar=BLOCK."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="BLOCK",
            threading_intact=False,
            signature_valid=False,
            reasons=["Chain broken"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.DIVERGENT
    
    def test_divergent_safety_warn_radar_block(self):
        """Test DIVERGENT when Safety=WARN and Radar=BLOCK."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Minor issue"]
        )
        radar = RadarView(
            status="BLOCK",
            threading_intact=False,
            signature_valid=False,
            reasons=["Chain broken"]
        )
        
        alignment = assess_alignment(safety, radar)
        assert alignment == AlignmentStatus.DIVERGENT


class TestGovernanceSignalLogic:
    """Test final governance status determination."""
    
    def test_ok_both_ok_aligned(self):
        """Test OK when both OK and ALIGNED."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain intact"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "OK"
        assert signal["alignment"] == "ALIGNED"
        assert signal["safety_status"] == "OK"
        assert signal["radar_status"] == "OK"
    
    def test_warn_both_warn_aligned(self):
        """Test WARN when both WARN and ALIGNED."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Determinism below threshold"]
        )
        radar = RadarView(
            status="WARN",
            threading_intact=True,
            signature_valid=False,
            reasons=["Signature format issue"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "WARN"
        assert signal["alignment"] == "ALIGNED"
    
    def test_warn_on_tension(self):
        """Test WARN when alignment is TENSION."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Minor issue"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "WARN"
        assert signal["alignment"] == "TENSION"
        assert any("[CONFLICT]" in reason for reason in signal["reasons"])
    
    def test_block_safety_blocks(self):
        """Test BLOCK when Safety reports BLOCK."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "BLOCK"
        assert signal["alignment"] == "DIVERGENT"
    
    def test_block_radar_blocks(self):
        """Test BLOCK when Radar reports BLOCK."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="BLOCK",
            threading_intact=False,
            signature_valid=False,
            reasons=["Chain broken"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "BLOCK"
        assert signal["alignment"] == "DIVERGENT"
    
    def test_block_both_block(self):
        """Test BLOCK when both report BLOCK."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="BLOCK",
            threading_intact=False,
            signature_valid=False,
            reasons=["Chain broken"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "BLOCK"
        assert signal["alignment"] == "ALIGNED"
    
    def test_block_on_divergent(self):
        """Test BLOCK when alignment is DIVERGENT."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="WARN",
            threading_intact=True,
            signature_valid=False,
            reasons=["Minor signature issue"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "BLOCK"
        assert signal["alignment"] == "DIVERGENT"


class TestReasonConsolidation:
    """Test reason consolidation with prefixes."""
    
    def test_safety_prefix(self):
        """Test that safety reasons get [Safety] prefix."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All hashes verified", "Determinism score excellent"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain threading intact"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        safety_reasons = [r for r in signal["reasons"] if r.startswith("[Safety]")]
        assert len(safety_reasons) == 2
        assert "[Safety] All hashes verified" in signal["reasons"]
        assert "[Safety] Determinism score excellent" in signal["reasons"]
    
    def test_radar_prefix(self):
        """Test that radar reasons get [Radar] prefix."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain threading intact", "All signatures valid"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        radar_reasons = [r for r in signal["reasons"] if r.startswith("[Radar]")]
        assert len(radar_reasons) == 2
        assert "[Radar] Chain threading intact" in signal["reasons"]
        assert "[Radar] All signatures valid" in signal["reasons"]
    
    def test_conflict_prefix_on_tension(self):
        """Test that TENSION adds [CONFLICT] reason."""
        safety = SafetyEvaluation(
            status="WARN",
            determinism_score=92.0,
            hash_match=True,
            reasons=["Minor issue"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        conflict_reasons = [r for r in signal["reasons"] if r.startswith("[CONFLICT]")]
        assert len(conflict_reasons) == 1
        assert "TENSION" in conflict_reasons[0]
    
    def test_conflict_prefix_on_divergent(self):
        """Test that DIVERGENT adds [CONFLICT] reason."""
        safety = SafetyEvaluation(
            status="BLOCK",
            determinism_score=85.0,
            hash_match=False,
            reasons=["Hash mismatch"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["All checks passed"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        conflict_reasons = [r for r in signal["reasons"] if r.startswith("[CONFLICT]")]
        assert len(conflict_reasons) == 1
        assert "DIVERGENT" in conflict_reasons[0]


class TestMetadata:
    """Test metadata inclusion in governance signal."""
    
    def test_metadata_includes_all_fields(self):
        """Test that metadata includes all required fields."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain intact"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert "metadata" in signal
        assert signal["metadata"]["safety_determinism_score"] == 98.5
        assert signal["metadata"]["safety_hash_match"] is True
        assert signal["metadata"]["radar_threading_intact"] is True
        assert signal["metadata"]["radar_signature_valid"] is True


class TestEvidencePackHarmonization:
    """Test evidence pack extension with governance_status."""
    
    def test_governance_status_added(self):
        """Test that governance_status field is added to evidence pack."""
        evidence_pack = {
            "replay_hash": "abc123",
            "determinism_score": 98.5,
            "block_number": 42,
        }
        
        governance_signal = {
            "final_status": "OK",
            "alignment": "ALIGNED",
            "reasons": ["[Safety] All checks passed"],
            "safety_status": "OK",
            "radar_status": "OK",
        }
        
        extended = extend_evidence_pack_with_governance_status(
            evidence_pack,
            governance_signal
        )
        
        assert "governance_status" in extended
        assert extended["governance_status"]["final_status"] == "OK"
        assert extended["governance_status"]["alignment"] == "ALIGNED"
        assert extended["governance_status"]["safety_status"] == "OK"
        assert extended["governance_status"]["radar_status"] == "OK"
    
    def test_original_fields_preserved(self):
        """Test that original evidence pack fields are preserved."""
        evidence_pack = {
            "replay_hash": "abc123",
            "determinism_score": 98.5,
            "block_number": 42,
        }
        
        governance_signal = {
            "final_status": "OK",
            "alignment": "ALIGNED",
            "reasons": [],
            "safety_status": "OK",
            "radar_status": "OK",
        }
        
        extended = extend_evidence_pack_with_governance_status(
            evidence_pack,
            governance_signal
        )
        
        assert extended["replay_hash"] == "abc123"
        assert extended["determinism_score"] == 98.5
        assert extended["block_number"] == 42
    
    def test_original_pack_not_mutated(self):
        """Test that the original evidence pack is not mutated."""
        evidence_pack = {
            "replay_hash": "abc123",
        }
        
        governance_signal = {
            "final_status": "OK",
            "alignment": "ALIGNED",
            "reasons": [],
            "safety_status": "OK",
            "radar_status": "OK",
        }
        
        extended = extend_evidence_pack_with_governance_status(
            evidence_pack,
            governance_signal
        )
        
        assert "governance_status" not in evidence_pack
        assert "governance_status" in extended


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_case_insensitive_status(self):
        """Test that status comparison is case-insensitive."""
        safety = SafetyEvaluation(
            status="ok",  # lowercase
            determinism_score=98.5,
            hash_match=True,
            reasons=["All checks passed"]
        )
        radar = RadarView(
            status="OK",  # uppercase
            threading_intact=True,
            signature_valid=True,
            reasons=["Chain intact"]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "OK"
        assert signal["alignment"] == "ALIGNED"
    
    def test_empty_reasons_lists(self):
        """Test handling of empty reasons lists."""
        safety = SafetyEvaluation(
            status="OK",
            determinism_score=98.5,
            hash_match=True,
            reasons=[]
        )
        radar = RadarView(
            status="OK",
            threading_intact=True,
            signature_valid=True,
            reasons=[]
        )
        
        signal = to_governance_signal_for_replay_safety(safety, radar)
        
        assert signal["final_status"] == "OK"
        assert isinstance(signal["reasons"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
