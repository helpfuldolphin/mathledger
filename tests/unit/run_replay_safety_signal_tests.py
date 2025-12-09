#!/usr/bin/env python3
"""
Standalone Test Runner for Replay Safety Governance Signal
===========================================================

Runs all tests without requiring pytest.
"""

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
)


def test_ok_both_ok_aligned():
    """Test OK when both OK and ALIGNED."""
    print("Test: OK when both OK and ALIGNED...")
    
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
    
    assert signal["final_status"] == "OK", f"Expected OK, got {signal['final_status']}"
    assert signal["alignment"] == "ALIGNED", f"Expected ALIGNED, got {signal['alignment']}"
    assert signal["safety_status"] == "OK"
    assert signal["radar_status"] == "OK"
    
    print("  ✓ PASSED")


def test_warn_on_tension():
    """Test WARN when alignment is TENSION."""
    print("Test: WARN when alignment is TENSION...")
    
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
    
    assert signal["final_status"] == "WARN", f"Expected WARN, got {signal['final_status']}"
    assert signal["alignment"] == "TENSION", f"Expected TENSION, got {signal['alignment']}"
    assert any("[CONFLICT]" in reason for reason in signal["reasons"]), "Expected [CONFLICT] in reasons"
    
    print("  ✓ PASSED")


def test_block_safety_blocks():
    """Test BLOCK when Safety reports BLOCK."""
    print("Test: BLOCK when Safety reports BLOCK...")
    
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
    
    assert signal["final_status"] == "BLOCK", f"Expected BLOCK, got {signal['final_status']}"
    assert signal["alignment"] == "DIVERGENT", f"Expected DIVERGENT, got {signal['alignment']}"
    
    print("  ✓ PASSED")


def test_block_radar_blocks():
    """Test BLOCK when Radar reports BLOCK."""
    print("Test: BLOCK when Radar reports BLOCK...")
    
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
    
    assert signal["final_status"] == "BLOCK", f"Expected BLOCK, got {signal['final_status']}"
    assert signal["alignment"] == "DIVERGENT", f"Expected DIVERGENT, got {signal['alignment']}"
    
    print("  ✓ PASSED")


def test_block_both_block():
    """Test BLOCK when both report BLOCK."""
    print("Test: BLOCK when both report BLOCK...")
    
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
    
    assert signal["final_status"] == "BLOCK", f"Expected BLOCK, got {signal['final_status']}"
    assert signal["alignment"] == "ALIGNED", f"Expected ALIGNED, got {signal['alignment']}"
    
    print("  ✓ PASSED")


def test_block_on_divergent():
    """Test BLOCK when alignment is DIVERGENT."""
    print("Test: BLOCK when alignment is DIVERGENT...")
    
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
    
    assert signal["final_status"] == "BLOCK", f"Expected BLOCK, got {signal['final_status']}"
    assert signal["alignment"] == "DIVERGENT", f"Expected DIVERGENT, got {signal['alignment']}"
    
    print("  ✓ PASSED")


def test_reason_prefixes():
    """Test that reasons get proper prefixes."""
    print("Test: Reason prefixes...")
    
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
        reasons=["Chain threading intact", "All signatures valid"]
    )
    
    signal = to_governance_signal_for_replay_safety(safety, radar)
    
    safety_reasons = [r for r in signal["reasons"] if r.startswith("[Safety]")]
    radar_reasons = [r for r in signal["reasons"] if r.startswith("[Radar]")]
    
    assert len(safety_reasons) == 2, f"Expected 2 safety reasons, got {len(safety_reasons)}"
    assert len(radar_reasons) == 2, f"Expected 2 radar reasons, got {len(radar_reasons)}"
    assert "[Safety] All hashes verified" in signal["reasons"]
    assert "[Radar] Chain threading intact" in signal["reasons"]
    
    print("  ✓ PASSED")


def test_evidence_pack_harmonization():
    """Test evidence pack extension with governance_status."""
    print("Test: Evidence pack harmonization...")
    
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
    
    assert "governance_status" in extended, "governance_status field not added"
    assert extended["governance_status"]["final_status"] == "OK"
    assert extended["governance_status"]["alignment"] == "ALIGNED"
    assert extended["replay_hash"] == "abc123", "Original fields not preserved"
    assert "governance_status" not in evidence_pack, "Original pack was mutated"
    
    print("  ✓ PASSED")


def test_metadata_inclusion():
    """Test that metadata includes all required fields."""
    print("Test: Metadata inclusion...")
    
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
    
    assert "metadata" in signal, "metadata field missing"
    assert signal["metadata"]["safety_determinism_score"] == 98.5
    assert signal["metadata"]["safety_hash_match"] is True
    assert signal["metadata"]["radar_threading_intact"] is True
    assert signal["metadata"]["radar_signature_valid"] is True
    
    print("  ✓ PASSED")


def main():
    """Run all tests."""
    print()
    print("=" * 70)
    print("Replay Safety Governance Signal - Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_ok_both_ok_aligned,
        test_warn_on_tension,
        test_block_safety_blocks,
        test_block_radar_blocks,
        test_block_both_block,
        test_block_on_divergent,
        test_reason_prefixes,
        test_evidence_pack_harmonization,
        test_metadata_inclusion,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    print()
    
    if failed > 0:
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
