"""
Unit tests for Replay Safety Governance Signal adapter.

Phase VI: Tests the to_governance_signal_for_replay_safety function that
collapses Safety and Radar into a single, normalized governance signal.

Tests cover:
- BLOCK when either side BLOCKs or alignment is DIVERGENT
- WARN on TENSION cases
- OK if both OK and aligned
- Prefixed reasons from [Safety], [Radar], [CONFLICT]
- Evidence pack harmonization with governance_status field

Author: Claude-C (Task STRATCOM)
Date: 2025-12-09
Status: Phase VI Tests
"""

import pytest
from experiments.u2.replay_safety import (
    # Phase IV exports
    PromotionStatus,
    SafetyLevel,
    evaluate_replay_safety_for_promotion,
    summarize_replay_safety_for_evidence,
    build_replay_safety_director_panel,
    build_replay_safety_envelope,
    compute_replay_confidence,
    # Phase V exports
    GovernanceAlignment,
    build_replay_safety_governance_view,
    # Phase VI exports
    to_governance_signal_for_replay_safety,
)


# ============================================================================
# Fixtures: Mock data builders
# ============================================================================

def make_safety_eval(
    status: str = PromotionStatus.OK,
    reasons: list = None,
    safe_for_policy_update: bool = True,
    safe_for_promotion: bool = True,
) -> dict:
    """Create a mock safety evaluation result."""
    return {
        "status": status,
        "reasons": reasons or [],
        "safe_for_policy_update": safe_for_policy_update,
        "safe_for_promotion": safe_for_promotion,
    }


def make_radar_view(
    governance_status: str = PromotionStatus.OK,
    governance_alignment: str = GovernanceAlignment.ALIGNED,
    conflict: bool = False,
    reasons: list = None,
    safety_status: str = None,
) -> dict:
    """Create a mock governance radar view."""
    return {
        "governance_status": governance_status,
        "governance_alignment": governance_alignment,
        "conflict": conflict,
        "reasons": reasons or [],
        "safety_status": safety_status or governance_status,
    }


def make_envelope(
    safety_level: str = SafetyLevel.OK,
    policy_update_allowed: bool = True,
    is_fully_deterministic: bool = True,
    confidence_score: float = 1.0,
    replay_mode: str = "full",
    critical_mismatch_flags: dict = None,
    per_cycle_consistency: dict = None,
    error_details: dict = None,
) -> dict:
    """Create a mock safety envelope."""
    return {
        "safety_level": safety_level,
        "policy_update_allowed": policy_update_allowed,
        "is_fully_deterministic": is_fully_deterministic,
        "confidence_score": confidence_score,
        "replay_mode": replay_mode,
        "critical_mismatch_flags": critical_mismatch_flags or {},
        "per_cycle_consistency": per_cycle_consistency or {"coverage_pct": 100.0},
        "error_details": error_details,
    }


# ============================================================================
# Test Class: BLOCK status determination
# ============================================================================

class TestGovernanceSignalBlock:
    """Tests for BLOCK status determination in governance signal."""

    def test_block_when_safety_blocks(self):
        """BLOCK when safety_eval status is BLOCK."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.BLOCK,
            reasons=["Critical mismatch detected"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.DIVERGENT,
            conflict=True,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["conflict"] is True
        assert signal["governance_alignment"] == GovernanceAlignment.DIVERGENT

    def test_block_when_radar_blocks(self):
        """BLOCK when radar status is BLOCK."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["All safety checks passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.BLOCK,
            governance_alignment=GovernanceAlignment.DIVERGENT,
            conflict=True,
            reasons=["Drift detected in governance radar"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["conflict"] is True

    def test_block_when_both_block(self):
        """BLOCK when both safety and radar BLOCK."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.BLOCK,
            reasons=["H_t mismatch"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.BLOCK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            conflict=False,
            reasons=["Governance drift detected"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert signal["conflict"] is False  # Both agree to BLOCK

    def test_block_when_divergent_alignment(self):
        """BLOCK when alignment is DIVERGENT even if statuses are mixed."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["Safety check passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.BLOCK,
            governance_alignment=GovernanceAlignment.DIVERGENT,
            conflict=True,
            reasons=["Radar sees BLOCK"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["governance_alignment"] == GovernanceAlignment.DIVERGENT

    def test_block_conflict_flag_triggers_block(self):
        """BLOCK when conflict flag is True."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["Safety OK"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            conflict=True,  # Explicit conflict flag
            reasons=["Manual conflict set"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        # Conflict flag should trigger BLOCK
        assert signal["status"] == PromotionStatus.BLOCK


# ============================================================================
# Test Class: WARN status determination
# ============================================================================

class TestGovernanceSignalWarn:
    """Tests for WARN status determination in governance signal."""

    def test_warn_on_tension_alignment(self):
        """WARN when alignment is TENSION."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["Safety passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.WARN,
            governance_alignment=GovernanceAlignment.TENSION,
            conflict=False,
            reasons=["Minor drift detected"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.WARN
        assert signal["governance_alignment"] == GovernanceAlignment.TENSION

    def test_warn_when_safety_warns(self):
        """WARN when safety status is WARN."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.WARN,
            reasons=["Config hash mismatch"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.TENSION,
            conflict=False,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.WARN

    def test_warn_when_radar_warns(self):
        """WARN when radar status is WARN."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["All checks passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.WARN,
            governance_alignment=GovernanceAlignment.TENSION,
            conflict=False,
            reasons=["Approaching drift threshold"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.WARN

    def test_warn_when_both_warn_aligned(self):
        """WARN when both sides WARN and are aligned."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.WARN,
            reasons=["Coverage below 100%"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.WARN,
            governance_alignment=GovernanceAlignment.ALIGNED,
            conflict=False,
            reasons=["Tier skew detected"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.WARN
        assert signal["governance_alignment"] == GovernanceAlignment.ALIGNED


# ============================================================================
# Test Class: OK status determination
# ============================================================================

class TestGovernanceSignalOK:
    """Tests for OK status determination in governance signal."""

    def test_ok_when_both_ok_and_aligned(self):
        """OK when both safety and radar are OK and aligned."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["All safety checks passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            conflict=False,
            reasons=["No drift detected"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["status"] == PromotionStatus.OK
        assert signal["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert signal["conflict"] is False

    def test_ok_preserves_signal_type(self):
        """OK signal includes signal_type identifier."""
        safety_eval = make_safety_eval(status=PromotionStatus.OK)
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["signal_type"] == "replay_safety"

    def test_ok_includes_both_statuses(self):
        """OK signal includes both safety and governance statuses."""
        safety_eval = make_safety_eval(status=PromotionStatus.OK)
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        assert signal["safety_status"] == PromotionStatus.OK
        assert signal["governance_status"] == PromotionStatus.OK


# ============================================================================
# Test Class: Reason prefixing
# ============================================================================

class TestGovernanceSignalReasons:
    """Tests for reason prefixing in governance signal."""

    def test_safety_reasons_prefixed(self):
        """Safety reasons are prefixed with [Safety]."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["All checks passed", "Confidence high"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        safety_reasons = [r for r in signal["reasons"] if r.startswith("[Safety]")]
        assert len(safety_reasons) == 2
        assert "[Safety] All checks passed" in signal["reasons"]
        assert "[Safety] Confidence high" in signal["reasons"]

    def test_radar_reasons_prefixed(self):
        """Radar reasons are prefixed with [Radar]."""
        safety_eval = make_safety_eval(status=PromotionStatus.OK)
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            reasons=["No drift", "Stable metrics"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        radar_reasons = [r for r in signal["reasons"] if r.startswith("[Radar]")]
        assert len(radar_reasons) == 2
        assert "[Radar] No drift" in signal["reasons"]
        assert "[Radar] Stable metrics" in signal["reasons"]

    def test_conflict_reason_added(self):
        """CONFLICT reason added when conflict detected."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["Safety OK"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.BLOCK,
            governance_alignment=GovernanceAlignment.DIVERGENT,
            conflict=True,
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        conflict_reasons = [r for r in signal["reasons"] if r.startswith("[CONFLICT]")]
        assert len(conflict_reasons) >= 1
        assert any("diverges" in r.lower() or "manual review" in r.lower()
                   for r in conflict_reasons)

    def test_already_prefixed_reasons_not_double_prefixed(self):
        """Reasons already prefixed are not double-prefixed."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["[Safety] Already prefixed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            reasons=["[Radar] Also prefixed"],
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        # Should not have [Safety] [Safety] or [Radar] [Radar]
        for reason in signal["reasons"]:
            assert reason.count("[Safety]") <= 1
            assert reason.count("[Radar]") <= 1

    def test_no_duplicate_reasons(self):
        """Duplicate reasons from radar view are not added twice."""
        safety_eval = make_safety_eval(
            status=PromotionStatus.OK,
            reasons=["Check passed"],
        )
        radar_view = make_radar_view(
            governance_status=PromotionStatus.OK,
            governance_alignment=GovernanceAlignment.ALIGNED,
            reasons=["Check passed"],  # Same reason
        )

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_view)

        # Should have [Safety] Check passed and [Radar] Check passed
        # (they're different after prefixing)
        assert "[Safety] Check passed" in signal["reasons"]
        assert "[Radar] Check passed" in signal["reasons"]


# ============================================================================
# Test Class: Evidence Pack Harmonization
# ============================================================================

class TestEvidencePackHarmonization:
    """Tests for evidence pack governance_status field integration."""

    def test_governance_status_added_to_evidence(self):
        """governance_status field is added when governance_signal provided."""
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            policy_update_allowed=True,
            is_fully_deterministic=True,
        )
        confidence = 1.0

        governance_signal = {
            "status": PromotionStatus.OK,
            "governance_alignment": GovernanceAlignment.ALIGNED,
        }

        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=confidence,
            governance_signal=governance_signal,
        )

        assert "governance_status" in evidence
        assert evidence["governance_status"] == PromotionStatus.OK

    def test_governance_status_reflects_block(self):
        """governance_status reflects BLOCK from governance signal."""
        envelope = make_envelope(
            safety_level=SafetyLevel.FAIL,
            policy_update_allowed=False,
            is_fully_deterministic=False,
        )
        confidence = 0.3

        governance_signal = {
            "status": PromotionStatus.BLOCK,
            "governance_alignment": GovernanceAlignment.DIVERGENT,
        }

        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=confidence,
            governance_signal=governance_signal,
        )

        assert evidence["governance_status"] == PromotionStatus.BLOCK

    def test_governance_status_reflects_warn(self):
        """governance_status reflects WARN from governance signal."""
        envelope = make_envelope(
            safety_level=SafetyLevel.WARN,
            policy_update_allowed=True,
            is_fully_deterministic=True,
        )
        confidence = 0.85

        governance_signal = {
            "status": PromotionStatus.WARN,
            "governance_alignment": GovernanceAlignment.TENSION,
        }

        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=confidence,
            governance_signal=governance_signal,
        )

        assert evidence["governance_status"] == PromotionStatus.WARN

    def test_evidence_without_governance_signal(self):
        """Evidence pack works without governance_signal (backward compat)."""
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            policy_update_allowed=True,
            is_fully_deterministic=True,
        )
        confidence = 1.0

        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=confidence,
            governance_signal=None,
        )

        # governance_status should not be present
        assert "governance_status" not in evidence
        # Other fields should still be present
        assert "replay_safety_ok" in evidence
        assert "confidence_score" in evidence
        assert "status" in evidence

    def test_evidence_with_governance_view(self):
        """Evidence pack includes governance_alignment when governance_view provided."""
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            policy_update_allowed=True,
            is_fully_deterministic=True,
        )
        confidence = 1.0

        governance_view = {
            "governance_alignment": GovernanceAlignment.ALIGNED,
        }

        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=confidence,
            governance_view=governance_view,
        )

        assert "governance_alignment" in evidence
        assert evidence["governance_alignment"] == GovernanceAlignment.ALIGNED


# ============================================================================
# Test Class: Full Integration Flow
# ============================================================================

class TestFullIntegrationFlow:
    """Integration tests for full governance signal flow."""

    def test_full_flow_ok_case(self):
        """Full flow: envelope -> promotion eval -> radar -> signal -> evidence."""
        # Build envelope
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            policy_update_allowed=True,
            is_fully_deterministic=True,
            confidence_score=0.95,
        )

        # Evaluate for promotion
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        assert promotion_eval["status"] == PromotionStatus.OK

        # Build radar view (simulated)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(
            envelope=envelope,
            promotion_eval=promotion_eval,
            radar=radar,
        )
        assert governance_view["governance_alignment"] == GovernanceAlignment.ALIGNED

        # Build governance signal
        signal = to_governance_signal_for_replay_safety(
            safety_eval=promotion_eval,
            radar_view=governance_view,
        )
        assert signal["status"] == PromotionStatus.OK

        # Build evidence summary
        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=envelope["confidence_score"],
            governance_view=governance_view,
            governance_signal=signal,
        )

        assert evidence["replay_safety_ok"] is True
        assert evidence["governance_status"] == PromotionStatus.OK
        assert evidence["governance_alignment"] == GovernanceAlignment.ALIGNED

    def test_full_flow_block_case(self):
        """Full flow: block scenario with h_t mismatch."""
        # Build envelope with failure
        envelope = make_envelope(
            safety_level=SafetyLevel.FAIL,
            policy_update_allowed=False,
            is_fully_deterministic=False,
            confidence_score=0.1,
            critical_mismatch_flags={"ht_mismatch": True},
            error_details={"error_code": "RUN-44"},
        )

        # Evaluate for promotion
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        assert promotion_eval["status"] == PromotionStatus.BLOCK

        # Build radar view (radar also sees block)
        radar = {"status": "BLOCK", "reasons": ["Drift alert triggered"]}
        governance_view = build_replay_safety_governance_view(
            envelope=envelope,
            promotion_eval=promotion_eval,
            radar=radar,
        )
        assert governance_view["governance_alignment"] == GovernanceAlignment.ALIGNED

        # Build governance signal
        signal = to_governance_signal_for_replay_safety(
            safety_eval=promotion_eval,
            radar_view=governance_view,
        )
        assert signal["status"] == PromotionStatus.BLOCK

        # Build evidence summary
        evidence = summarize_replay_safety_for_evidence(
            envelope=envelope,
            confidence=envelope["confidence_score"],
            governance_view=governance_view,
            governance_signal=signal,
        )

        assert evidence["replay_safety_ok"] is False
        assert evidence["governance_status"] == PromotionStatus.BLOCK

    def test_full_flow_divergent_case(self):
        """Full flow: divergent scenario where safety OK but radar BLOCK."""
        # Build envelope (appears OK)
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            policy_update_allowed=True,
            is_fully_deterministic=True,
            confidence_score=0.92,
        )

        # Evaluate for promotion (OK from safety perspective)
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        assert promotion_eval["status"] == PromotionStatus.OK

        # Build radar view (radar sees BLOCK - divergent!)
        radar = {"status": "BLOCK", "reasons": ["External governance signal: BLOCK"]}
        governance_view = build_replay_safety_governance_view(
            envelope=envelope,
            promotion_eval=promotion_eval,
            radar=radar,
        )
        assert governance_view["governance_alignment"] == GovernanceAlignment.DIVERGENT
        assert governance_view["conflict"] is True

        # Build governance signal (should be BLOCK due to divergence)
        signal = to_governance_signal_for_replay_safety(
            safety_eval=promotion_eval,
            radar_view=governance_view,
        )
        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["conflict"] is True

        # Verify CONFLICT reason present
        conflict_reasons = [r for r in signal["reasons"] if "[CONFLICT]" in r]
        assert len(conflict_reasons) >= 1


# ============================================================================
# Test Class: Director Panel Integration
# ============================================================================

class TestDirectorPanelIntegration:
    """Tests for director panel with governance signal."""

    def test_director_panel_shows_conflict(self):
        """Director panel shows conflict when governance view has conflict."""
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            confidence_score=0.95,
        )

        promotion_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["All checks passed"],
        }

        governance_view = {
            "conflict": True,
            "safety_status": PromotionStatus.OK,
            "governance_status": PromotionStatus.BLOCK,
        }

        panel = build_replay_safety_director_panel(
            envelope=envelope,
            promotion_eval=promotion_eval,
            governance_view=governance_view,
        )

        assert panel["conflict_flag"] is True
        assert panel["conflict_note"] is not None
        assert "manual review" in panel["conflict_note"].lower()

    def test_director_panel_no_conflict(self):
        """Director panel shows no conflict when aligned."""
        envelope = make_envelope(
            safety_level=SafetyLevel.OK,
            confidence_score=0.95,
        )

        promotion_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["All checks passed"],
        }

        governance_view = {
            "conflict": False,
            "safety_status": PromotionStatus.OK,
            "governance_status": PromotionStatus.OK,
        }

        panel = build_replay_safety_director_panel(
            envelope=envelope,
            promotion_eval=promotion_eval,
            governance_view=governance_view,
        )

        assert panel["conflict_flag"] is False
        assert panel["conflict_note"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
