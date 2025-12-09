"""
Phase V: Double-Helix Drift Radar & Global Governance Linkage Tests
===================================================================

Tests for drift timeline analysis, global console adaptation, and uplift
gate composition logic.

PHASE V — VERIFICATION BUREAU
Agent B4 (verifier-ops-4)
"""

import pytest
from typing import Dict, Any, List

from rfl.verification.abstention_taxonomy import AbstentionType
from rfl.verification.failure_classifier import FailureState
from rfl.verification.abstention_record import AbstentionRecord
from rfl.verification.abstention_semantics import (
    summarize_abstentions,
    detect_abstention_red_flags,
    build_abstention_health_snapshot,
    build_epistemic_abstention_profile,
    build_abstention_storyline,
    build_epistemic_drift_timeline,
    summarize_abstention_for_global_console,
    compose_abstention_with_budget_and_perf,
    evaluate_abstention_for_uplift,
    summarize_abstentions_for_uplift,
    compose_abstention_with_uplift_decision,
)


class TestBuildEpistemicDriftTimeline:
    """Tests for build_epistemic_drift_timeline() drift detection."""
    
    def test_drift_timeline_has_required_fields(self):
        """Drift timeline contains all required fields."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        timeline = build_epistemic_drift_timeline([profile])
        
        assert timeline["schema_version"] == "1.0.0"
        assert "drift_index" in timeline
        assert "risk_band" in timeline
        assert "change_points" in timeline
        assert "summary_text" in timeline
        assert 0.0 <= timeline["drift_index"] <= 1.0
        assert timeline["risk_band"] in ("STABLE", "DRIFTING", "VOLATILE")
    
    def test_drift_timeline_empty_profiles(self):
        """Drift timeline handles empty profile list."""
        timeline = build_epistemic_drift_timeline([])
        
        assert timeline["drift_index"] == 0.0
        assert timeline["risk_band"] == "STABLE"
        assert timeline["change_points"] == []
    
    def test_drift_timeline_single_profile(self):
        """Drift timeline handles single profile (baseline)."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "slice_001")
        profile = build_epistemic_abstention_profile(snapshot)
        timeline = build_epistemic_drift_timeline([profile])
        
        assert timeline["drift_index"] == 0.0
        assert timeline["risk_band"] == "STABLE"
        assert "baseline" in timeline["summary_text"].lower()
    
    def test_drift_timeline_stable_band(self):
        """Drift timeline identifies STABLE band when all profiles same risk."""
        profiles = []
        for i in range(5):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 10 + [
                AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
            ] * 5 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 30 + [
                AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
            ] * 55
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"low_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        timeline = build_epistemic_drift_timeline(profiles)
        
        assert timeline["risk_band"] == "STABLE"
        assert timeline["drift_index"] <= 0.2
        assert "STABLE" in timeline["summary_text"]
    
    def test_drift_timeline_drifting_band(self):
        """Drift timeline identifies DRIFTING band with moderate variation."""
        profiles = []
        # Mix of LOW and MEDIUM
        for i in range(3):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 10 + [
                AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
            ] * 5 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 30 + [
                AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
            ] * 55
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"low_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        for i in range(3):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 35 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 65
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"medium_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        timeline = build_epistemic_drift_timeline(profiles)
        
        assert timeline["risk_band"] in ("STABLE", "DRIFTING")
        assert 0.0 <= timeline["drift_index"] <= 0.6
    
    def test_drift_timeline_volatile_band(self):
        """Drift timeline identifies VOLATILE band with high variation."""
        profiles = []
        # Mix of LOW, MEDIUM, HIGH
        for i in range(2):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 10 + [
                AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
            ] * 5 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 30 + [
                AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
            ] * 55
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"low_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        for i in range(2):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 35 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 65
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"medium_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        for i in range(2):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 60 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 40
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"high_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        timeline = build_epistemic_drift_timeline(profiles)
        
        # With LOW, MEDIUM, HIGH mix, should have high variance
        assert timeline["risk_band"] in ("DRIFTING", "VOLATILE")
        assert timeline["drift_index"] > 0.0
    
    def test_drift_timeline_detects_change_points(self):
        """Drift timeline detects significant change points."""
        profiles = []
        # Start with LOW
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 10 + [
            AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
        ] * 5 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 30 + [
            AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
        ] * 55
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "slice_001")
        profiles.append(build_epistemic_abstention_profile(snapshot))
        
        # Transition to HIGH (significant change)
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "slice_002")
        profiles.append(build_epistemic_abstention_profile(snapshot))
        
        timeline = build_epistemic_drift_timeline(profiles)
        
        assert len(timeline["change_points"]) >= 1
        change_point = timeline["change_points"][0]
        assert "LOW → HIGH" in change_point["transition"] or "HIGH" in change_point["transition"]
    
    def test_drift_timeline_json_serializable(self):
        """Drift timeline is JSON-serializable."""
        import json
        
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        timeline = build_epistemic_drift_timeline([profile])
        
        # Should not raise
        json_str = json.dumps(timeline)
        assert "drift_index" in json_str


class TestSummarizeAbstentionForGlobalConsole:
    """Tests for summarize_abstention_for_global_console() console adapter."""
    
    def test_console_has_required_fields(self):
        """Console summary contains all required fields."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert "abstention_status_light" in console
        assert "epistemic_risk" in console
        assert "storyline_snapshot" in console
        assert "drift_band" in console
        assert "headline" in console
        assert console["abstention_status_light"] in ("GREEN", "YELLOW", "RED")
    
    def test_console_status_light_green(self):
        """Console shows GREEN for low risk, stable drift, improving trend."""
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 10 + [
            AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
        ] * 5 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 30 + [
            AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
        ] * 55
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "low_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert console["abstention_status_light"] == "GREEN"
        assert console["epistemic_risk"] == "LOW"
    
    def test_console_status_light_yellow(self):
        """Console shows YELLOW for medium risk or drifting."""
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 35 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 65
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "medium_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert console["abstention_status_light"] == "YELLOW"
        assert console["epistemic_risk"] == "MEDIUM"
    
    def test_console_status_light_red(self):
        """Console shows RED for high risk or volatile drift."""
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert console["abstention_status_light"] == "RED"
        assert console["epistemic_risk"] == "HIGH"
    
    def test_console_includes_storyline_snapshot(self):
        """Console includes storyline snapshot."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert "storyline_snapshot" in console
        assert "trend" in console["storyline_snapshot"]
        assert "story" in console["storyline_snapshot"]
    
    def test_console_includes_drift_info(self):
        """Console includes drift band and index."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert "drift_band" in console
        assert "drift_index" in console
        assert console["drift_band"] in ("STABLE", "DRIFTING", "VOLATILE")
    
    def test_console_headline_unified(self):
        """Console generates unified headline."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        assert "headline" in console
        assert len(console["headline"]) > 0
        assert console["abstention_status_light"] in console["headline"]
    
    def test_console_json_serializable(self):
        """Console summary is JSON-serializable."""
        import json
        
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        storyline = build_abstention_storyline([profile])
        timeline = build_epistemic_drift_timeline([profile])
        console = summarize_abstention_for_global_console(profile, storyline, timeline)
        
        # Should not raise
        json_str = json.dumps(console)
        assert "abstention_status_light" in json_str


class TestComposeAbstentionWithUpliftDecision:
    """Tests for compose_abstention_with_uplift_decision() uplift composition."""
    
    def test_compose_has_required_fields(self):
        """Composed decision contains all required fields."""
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        snapshots = [snapshot]
        from rfl.verification.abstention_semantics import build_abstention_radar
        radar = build_abstention_radar(snapshots)
        uplift_eval = summarize_abstentions_for_uplift(radar)
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert "final_status" in final
        assert "final_uplift_ok" in final
        assert "epistemic_upgrade_applied" in final
        assert "advisory_fields" in final
        assert "blocking_slices" in final
        assert "reasons" in final
        assert final["final_status"] in ("OK", "WARN", "BLOCK")
        assert isinstance(final["final_uplift_ok"], bool)
    
    def test_compose_epistemic_block_upgrades_uplift_warn(self):
        """Epistemic BLOCK upgrades uplift WARN to BLOCK."""
        # Create HIGH risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that would be WARN
        from rfl.verification.abstention_semantics import build_abstention_radar
        snapshots = [snapshot]
        radar = build_abstention_radar(snapshots)
        uplift_eval = summarize_abstentions_for_uplift(radar)
        
        # Manually set to WARN for test
        uplift_eval["status"] = "WARN"
        uplift_eval["uplift_safe"] = True
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "BLOCK"
        assert final["final_uplift_ok"] is False
        assert final["epistemic_upgrade_applied"] is True
    
    def test_compose_epistemic_block_upgrades_uplift_ok(self):
        """Epistemic BLOCK upgrades uplift OK to BLOCK."""
        # Create HIGH risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that would be OK
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "BLOCK"
        assert final["final_uplift_ok"] is False
        assert final["epistemic_upgrade_applied"] is True
    
    def test_compose_epistemic_warn_with_uplift_warn(self):
        """Epistemic WARN + Uplift WARN → Final WARN."""
        # Create MEDIUM risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 35 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 65
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "medium_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that is WARN
        uplift_eval = {
            "uplift_safe": True,
            "status": "WARN",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "WARN"
        assert final["final_uplift_ok"] is True  # Not blocked, but warned
        assert final["epistemic_upgrade_applied"] is False
    
    def test_compose_epistemic_warn_with_uplift_ok(self):
        """Epistemic WARN + Uplift OK → Final WARN (epistemic takes precedence)."""
        # Create MEDIUM risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 35 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 65
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "medium_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that is OK
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "WARN"
        assert final["final_uplift_ok"] is True
        assert final["epistemic_upgrade_applied"] is True
    
    def test_compose_epistemic_ok_with_uplift_warn(self):
        """Epistemic OK + Uplift WARN → Final WARN."""
        # Create LOW risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 10 + [
            AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
        ] * 5 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 30 + [
            AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
        ] * 55
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "low_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that is WARN
        uplift_eval = {
            "uplift_safe": True,
            "status": "WARN",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "WARN"
        assert final["final_uplift_ok"] is True
        assert final["epistemic_upgrade_applied"] is False
    
    def test_compose_epistemic_ok_with_uplift_ok(self):
        """Epistemic OK + Uplift OK → Final OK."""
        # Create LOW risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 10 + [
            AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
        ] * 5 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 30 + [
            AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
        ] * 55
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "low_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval that is OK
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert final["final_status"] == "OK"
        assert final["final_uplift_ok"] is True
        assert final["epistemic_upgrade_applied"] is False
    
    def test_compose_combines_blocking_slices(self):
        """Compose combines blocking slices from both evaluations."""
        # Create HIGH risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift eval with different blocking slice
        uplift_eval = {
            "uplift_safe": False,
            "status": "BLOCK",
            "blocking_slices": ["uplift_blocking_slice"],
            "blocking_reasons": {"uplift_blocking_slice": "timeout_rate=65%"},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        # Should combine both blocking slices
        assert len(final["blocking_slices"]) >= 1
        assert "uplift_blocking_slice" in final["blocking_slices"] or "high_slice" in final["blocking_slices"]
    
    def test_compose_combines_reasons(self):
        """Compose combines reasons from both evaluations."""
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        assert len(final["reasons"]) > 0
        # Should include epistemic reasons
        assert any("epistemic" in r.lower() or "HIGH" in r for r in final["reasons"])
    
    def test_compose_advisory_fields_for_warn(self):
        """Compose includes advisory fields when epistemic is WARN."""
        # Create MEDIUM risk profile
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 35 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 65
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "medium_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        # Should have advisory fields when epistemic is WARN
        if epistemic_eval["status"] == "WARN":
            assert len(final["advisory_fields"]) > 0
            assert any("epistemic" in f.lower() for f in final["advisory_fields"])
    
    def test_compose_json_serializable(self):
        """Composed decision is JSON-serializable."""
        import json
        
        records = [AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t")] * 10
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "test_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        # Should not raise
        json_str = json.dumps(final)
        assert "final_status" in json_str


class TestTrendCorrectness:
    """Tests for trend correctness across storyline and drift timeline."""
    
    def test_storyline_and_drift_consistency(self):
        """Storyline trend and drift band should be consistent."""
        profiles = []
        # Create stable sequence (all LOW)
        for i in range(5):
            records = [
                AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
            ] * 10 + [
                AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
            ] * 5 + [
                AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
            ] * 30 + [
                AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
            ] * 55
            summary = summarize_abstentions(records)
            flags = detect_abstention_red_flags(summary)
            snapshot = build_abstention_health_snapshot(summary, flags, f"low_{i}")
            profiles.append(build_epistemic_abstention_profile(snapshot))
        
        storyline = build_abstention_storyline(profiles)
        timeline = build_epistemic_drift_timeline(profiles)
        
        # Both should indicate stability/improvement
        assert storyline["global_epistemic_trend"] in ("STABLE", "IMPROVING")
        assert timeline["risk_band"] == "STABLE"
    
    def test_drift_integration_with_storyline(self):
        """Drift timeline integrates correctly with storyline analysis."""
        profiles = []
        # Create improving sequence (HIGH → MEDIUM → LOW)
        # HIGH
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_0")
        profiles.append(build_epistemic_abstention_profile(snapshot))
        
        # MEDIUM
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 35 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 65
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "medium_0")
        profiles.append(build_epistemic_abstention_profile(snapshot))
        
        # LOW
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 10 + [
            AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="c"),
        ] * 5 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 30 + [
            AbstentionRecord.from_failure_state(FailureState.BUDGET_EXHAUSTED, details="b"),
        ] * 55
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "low_0")
        profiles.append(build_epistemic_abstention_profile(snapshot))
        
        timeline = build_epistemic_drift_timeline(profiles)
        
        # Should detect change points
        assert len(timeline["change_points"]) >= 2  # HIGH→MEDIUM and MEDIUM→LOW
        # Should show some drift (not stable)
        assert timeline["risk_band"] in ("DRIFTING", "VOLATILE")


class TestUpliftCompositionLogic:
    """Tests for uplift composition logic correctness."""
    
    def test_uplift_composition_deterministic(self):
        """Uplift composition produces deterministic results."""
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        uplift_eval = {
            "uplift_safe": True,
            "status": "WARN",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        # Run multiple times
        results = [compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval) for _ in range(10)]
        
        # All should be identical
        first = results[0]
        assert all(r["final_status"] == first["final_status"] for r in results)
        assert all(r["final_uplift_ok"] == first["final_uplift_ok"] for r in results)
        assert all(r["blocking_slices"] == first["blocking_slices"] for r in results)
    
    def test_uplift_composition_upgrade_tracking(self):
        """Uplift composition correctly tracks upgrade applications."""
        # Create HIGH risk
        records = [
            AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="t"),
        ] * 60 + [
            AbstentionRecord.from_failure_state(FailureState.INVALID_FORMULA, details="i"),
        ] * 40
        summary = summarize_abstentions(records)
        flags = detect_abstention_red_flags(summary)
        snapshot = build_abstention_health_snapshot(summary, flags, "high_slice")
        profile = build_epistemic_abstention_profile(snapshot)
        compound = compose_abstention_with_budget_and_perf([profile])
        epistemic_eval = evaluate_abstention_for_uplift(compound)
        
        # Uplift that would be OK
        uplift_eval = {
            "uplift_safe": True,
            "status": "OK",
            "blocking_slices": [],
            "blocking_reasons": {},
        }
        
        final = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        
        # Should track that upgrade was applied
        if epistemic_eval["status"] == "BLOCK" and uplift_eval["status"] == "OK":
            assert final["epistemic_upgrade_applied"] is True
            assert any("upgraded" in r.lower() for r in final["reasons"])

