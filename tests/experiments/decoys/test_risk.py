# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Family Risk Snapshot, Drift Analysis, and Governance Signals

This module tests:
1. Family risk snapshot generation
2. Multi-contract family drift analysis
3. Global health governance signals
4. Forbidden language enforcement

All tests verify deterministic behavior and neutral language constraints.
"""

import pytest
from typing import Any, Dict, List

from experiments.decoys.risk import (
    RISK_SCHEMA_VERSION,
    RISK_LEVELS,
    RISK_ORDER,
    FORBIDDEN_WORDS,
    compute_family_risk_level,
    build_family_risk_snapshot,
    compare_family_risk,
    summarize_confusability_for_global_health,
    check_forbidden_language,
    build_slice_confusability_view,
    summarize_decoy_confusability_for_uplift,
    build_confusability_director_panel,
    build_decoy_family_drift_governor,
    build_decoy_uplift_prescreen,
)
from experiments.decoys.contract import export_contract


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_contract_high_risk() -> Dict[str, Any]:
    """Contract with HIGH-risk families."""
    return {
        "schema_version": "1.1.0",
        "slice_name": "test_slice_high",
        "config_path": "test.yaml",
        "families": {
            "fam_high_1": {
                "members": ["f1", "f2", "f3"],
                "avg_confusability": 0.85,
                "difficulty_band": "hard",
            },
            "fam_high_2": {
                "members": ["f4", "f5"],
                "avg_confusability": 0.75,
                "difficulty_band": "hard",
            },
            "fam_medium": {
                "members": ["f6"],
                "avg_confusability": 0.55,
                "difficulty_band": "hard",
            },
        },
        "formulas": [],
        "summary": {
            "target_count": 3,
            "decoy_near_count": 2,
            "decoy_far_count": 1,
            "bridge_count": 0,
            "avg_confusability_near": 0.8,
            "avg_confusability_far": 0.4,
            "family_count": 3,
        },
    }


@pytest.fixture
def sample_contract_low_risk() -> Dict[str, Any]:
    """Contract with LOW-risk families."""
    return {
        "schema_version": "1.1.0",
        "slice_name": "test_slice_low",
        "config_path": "test.yaml",
        "families": {
            "fam_low_1": {
                "members": ["f1", "f2"],
                "avg_confusability": 0.3,
                "difficulty_band": "easy",
            },
            "fam_low_2": {
                "members": ["f3"],
                "avg_confusability": 0.45,
                "difficulty_band": "medium",
            },
        },
        "formulas": [],
        "summary": {
            "target_count": 2,
            "decoy_near_count": 1,
            "decoy_far_count": 0,
            "bridge_count": 0,
            "avg_confusability_near": 0.3,
            "avg_confusability_far": 0.0,
            "family_count": 2,
        },
    }


@pytest.fixture
def sample_contract_empty() -> Dict[str, Any]:
    """Contract with no families."""
    return {
        "schema_version": "1.1.0",
        "slice_name": "test_slice_empty",
        "config_path": "test.yaml",
        "families": {},
        "formulas": [],
        "summary": {
            "target_count": 0,
            "decoy_near_count": 0,
            "decoy_far_count": 0,
            "bridge_count": 0,
            "avg_confusability_near": 0.0,
            "avg_confusability_far": 0.0,
            "family_count": 0,
        },
    }


# =============================================================================
# TASK 1: FAMILY RISK SNAPSHOT TESTS
# =============================================================================

class TestComputeFamilyRiskLevel:
    """Tests for risk level computation."""
    
    def test_high_confusability_hard_band_is_high_risk(self):
        """High confusability + hard = HIGH risk."""
        assert compute_family_risk_level(0.85, "hard") == "HIGH"
        assert compute_family_risk_level(0.70, "hard") == "HIGH"
        assert compute_family_risk_level(1.0, "hard") == "HIGH"
    
    def test_high_confusability_non_hard_is_medium_risk(self):
        """High confusability + medium/easy = MEDIUM risk."""
        assert compute_family_risk_level(0.85, "medium") == "MEDIUM"
        assert compute_family_risk_level(0.70, "easy") == "MEDIUM"
    
    def test_moderate_confusability_hard_is_medium_risk(self):
        """Moderate confusability + hard = MEDIUM risk."""
        assert compute_family_risk_level(0.55, "hard") == "MEDIUM"
        assert compute_family_risk_level(0.40, "hard") == "MEDIUM"
    
    def test_moderate_confusability_non_hard_is_low_risk(self):
        """Moderate confusability + medium/easy = LOW risk."""
        assert compute_family_risk_level(0.55, "medium") == "LOW"
        assert compute_family_risk_level(0.40, "easy") == "LOW"
    
    def test_low_confusability_any_band_is_low_risk(self):
        """Low confusability = LOW risk regardless of difficulty."""
        assert compute_family_risk_level(0.3, "hard") == "LOW"
        assert compute_family_risk_level(0.2, "medium") == "LOW"
        assert compute_family_risk_level(0.1, "easy") == "LOW"
        assert compute_family_risk_level(0.0, "hard") == "LOW"


class TestBuildFamilyRiskSnapshot:
    """Tests for family risk snapshot generation."""
    
    def test_snapshot_has_required_fields(self, sample_contract_high_risk):
        """Snapshot should have all required fields."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        required = {
            "families", "high_risk_family_count", "medium_risk_family_count",
            "low_risk_family_count", "schema_version", "slice_name",
            "summary_notes", "total_family_count"
        }
        
        assert required.issubset(set(snapshot.keys()))
    
    def test_snapshot_schema_version(self, sample_contract_high_risk):
        """Snapshot should have correct schema version."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        assert snapshot["schema_version"] == RISK_SCHEMA_VERSION
    
    def test_snapshot_counts_high_risk(self, sample_contract_high_risk):
        """Should correctly count HIGH-risk families."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        assert snapshot["high_risk_family_count"] == 2  # fam_high_1, fam_high_2
        assert snapshot["medium_risk_family_count"] == 1  # fam_medium
        assert snapshot["low_risk_family_count"] == 0
    
    def test_snapshot_counts_low_risk(self, sample_contract_low_risk):
        """Should correctly count LOW-risk families."""
        snapshot = build_family_risk_snapshot(sample_contract_low_risk)
        
        assert snapshot["high_risk_family_count"] == 0
        assert snapshot["low_risk_family_count"] == 2
    
    def test_snapshot_empty_contract(self, sample_contract_empty):
        """Empty contract should produce zero counts."""
        snapshot = build_family_risk_snapshot(sample_contract_empty)
        
        assert snapshot["total_family_count"] == 0
        assert snapshot["high_risk_family_count"] == 0
        assert snapshot["medium_risk_family_count"] == 0
        assert snapshot["low_risk_family_count"] == 0
    
    def test_snapshot_family_entries_have_required_fields(self, sample_contract_high_risk):
        """Each family entry should have required fields."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        required_entry_fields = {
            "fingerprint", "members_count", "avg_confusability",
            "difficulty_band", "risk_level"
        }
        
        for fingerprint, entry in snapshot["families"].items():
            assert required_entry_fields.issubset(set(entry.keys())), (
                f"Family {fingerprint} missing fields"
            )
    
    def test_snapshot_deterministic(self, sample_contract_high_risk):
        """Snapshot generation should be deterministic."""
        snapshots = [
            build_family_risk_snapshot(sample_contract_high_risk)
            for _ in range(5)
        ]
        
        for i in range(1, len(snapshots)):
            assert snapshots[0] == snapshots[i]
    
    def test_snapshot_families_sorted_by_fingerprint(self, sample_contract_high_risk):
        """Family entries should be sorted by fingerprint."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        keys = list(snapshot["families"].keys())
        assert keys == sorted(keys)


class TestSnapshotSummaryNotes:
    """Tests for summary notes language constraints."""
    
    def test_summary_notes_no_forbidden_words(self, sample_contract_high_risk):
        """Summary notes should not contain forbidden words."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        forbidden = check_forbidden_language(snapshot["summary_notes"])
        assert len(forbidden) == 0, f"Found forbidden words: {forbidden}"
    
    def test_summary_notes_descriptive(self, sample_contract_high_risk):
        """Summary notes should be descriptive."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        notes = snapshot["summary_notes"]
        assert "Distribution:" in notes
        assert "HIGH" in notes or "MEDIUM" in notes or "LOW" in notes


# =============================================================================
# TASK 2: FAMILY DRIFT ANALYZER TESTS
# =============================================================================

class TestCompareFamilyRisk:
    """Tests for family drift analysis."""
    
    def test_compare_identical_snapshots(self, sample_contract_high_risk):
        """Identical snapshots should show no drift."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        drift = compare_family_risk(snapshot, snapshot)
        
        assert drift["families_new"] == []
        assert drift["families_removed"] == []
        assert drift["families_increased_risk"] == []
        assert drift["families_decreased_risk"] == []
        assert drift["net_risk_trend"] == "STABLE"
    
    def test_compare_detects_new_families(
        self, sample_contract_low_risk, sample_contract_high_risk
    ):
        """Should detect new families."""
        old_snap = build_family_risk_snapshot(sample_contract_low_risk)
        new_snap = build_family_risk_snapshot(sample_contract_high_risk)
        
        drift = compare_family_risk(old_snap, new_snap)
        
        # High risk contract has different families than low risk
        assert len(drift["families_new"]) > 0
    
    def test_compare_detects_removed_families(
        self, sample_contract_high_risk, sample_contract_low_risk
    ):
        """Should detect removed families."""
        old_snap = build_family_risk_snapshot(sample_contract_high_risk)
        new_snap = build_family_risk_snapshot(sample_contract_low_risk)
        
        drift = compare_family_risk(old_snap, new_snap)
        
        assert len(drift["families_removed"]) > 0
    
    def test_compare_detects_increased_risk(self):
        """Should detect families with increased risk."""
        old_contract = {
            "slice_name": "test",
            "families": {
                "fam_a": {
                    "members": ["f1"],
                    "avg_confusability": 0.3,
                    "difficulty_band": "easy",
                },
            },
        }
        new_contract = {
            "slice_name": "test",
            "families": {
                "fam_a": {
                    "members": ["f1"],
                    "avg_confusability": 0.85,
                    "difficulty_band": "hard",
                },
            },
        }
        
        old_snap = build_family_risk_snapshot(old_contract)
        new_snap = build_family_risk_snapshot(new_contract)
        
        drift = compare_family_risk(old_snap, new_snap)
        
        assert len(drift["families_increased_risk"]) == 1
        assert drift["families_increased_risk"][0]["fingerprint"] == "fam_a"
        assert drift["families_increased_risk"][0]["old_risk"] == "LOW"
        assert drift["families_increased_risk"][0]["new_risk"] == "HIGH"
    
    def test_compare_detects_decreased_risk(self):
        """Should detect families with decreased risk."""
        old_contract = {
            "slice_name": "test",
            "families": {
                "fam_b": {
                    "members": ["f2"],
                    "avg_confusability": 0.85,
                    "difficulty_band": "hard",
                },
            },
        }
        new_contract = {
            "slice_name": "test",
            "families": {
                "fam_b": {
                    "members": ["f2"],
                    "avg_confusability": 0.3,
                    "difficulty_band": "easy",
                },
            },
        }
        
        old_snap = build_family_risk_snapshot(old_contract)
        new_snap = build_family_risk_snapshot(new_contract)
        
        drift = compare_family_risk(old_snap, new_snap)
        
        assert len(drift["families_decreased_risk"]) == 1
        assert drift["families_decreased_risk"][0]["old_risk"] == "HIGH"
        assert drift["families_decreased_risk"][0]["new_risk"] == "LOW"
    
    def test_net_risk_trend_degrading(self, sample_contract_low_risk, sample_contract_high_risk):
        """Should detect DEGRADING trend when risk increases."""
        old_snap = build_family_risk_snapshot(sample_contract_low_risk)
        new_snap = build_family_risk_snapshot(sample_contract_high_risk)
        
        # Manually set same families to force comparison
        old_snap["families"]["fam_common"] = {
            "fingerprint": "fam_common",
            "members_count": 1,
            "avg_confusability": 0.3,
            "difficulty_band": "easy",
            "risk_level": "LOW",
        }
        new_snap["families"]["fam_common"] = {
            "fingerprint": "fam_common",
            "members_count": 1,
            "avg_confusability": 0.85,
            "difficulty_band": "hard",
            "risk_level": "HIGH",
        }
        new_snap["high_risk_family_count"] += 1
        
        drift = compare_family_risk(old_snap, new_snap)
        
        assert drift["net_risk_trend"] == "DEGRADING"
    
    def test_compare_deterministic(self, sample_contract_high_risk, sample_contract_low_risk):
        """Drift comparison should be deterministic."""
        old_snap = build_family_risk_snapshot(sample_contract_low_risk)
        new_snap = build_family_risk_snapshot(sample_contract_high_risk)
        
        drifts = [compare_family_risk(old_snap, new_snap) for _ in range(5)]
        
        for i in range(1, len(drifts)):
            assert drifts[0] == drifts[i]


# =============================================================================
# TASK 3: GOVERNANCE SIGNAL TESTS
# =============================================================================

class TestSummarizeConfusabilityForGlobalHealth:
    """Tests for governance signal generation."""
    
    def test_signal_has_required_fields(self, sample_contract_high_risk):
        """Signal should have all required fields."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        required = {"confusability_ok", "high_risk_family_count", "status", "summary"}
        assert required.issubset(set(signal.keys()))
    
    def test_status_ok_for_low_risk(self, sample_contract_low_risk):
        """LOW-risk only should yield OK status."""
        snapshot = build_family_risk_snapshot(sample_contract_low_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        assert signal["status"] == "OK"
        assert signal["confusability_ok"] is True
    
    def test_status_attention_for_high_risk(self, sample_contract_high_risk):
        """HIGH-risk families should yield ATTENTION or HOT status."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        assert signal["status"] in ("ATTENTION", "HOT")
        assert signal["confusability_ok"] is False
    
    def test_status_hot_with_degrading_trend(self, sample_contract_high_risk):
        """Multiple HIGH-risk + DEGRADING trend should yield HOT status."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        drift = {"net_risk_trend": "DEGRADING"}
        
        signal = summarize_confusability_for_global_health(snapshot, drift)
        
        assert signal["status"] == "HOT"
        assert signal["risk_trend"] == "DEGRADING"
    
    def test_signal_includes_trend_when_drift_provided(self, sample_contract_low_risk):
        """Signal should include risk_trend when drift is provided."""
        snapshot = build_family_risk_snapshot(sample_contract_low_risk)
        drift = {"net_risk_trend": "STABLE"}
        
        signal = summarize_confusability_for_global_health(snapshot, drift)
        
        assert "risk_trend" in signal
        assert signal["risk_trend"] == "STABLE"
    
    def test_signal_no_trend_without_drift(self, sample_contract_low_risk):
        """Signal should not include risk_trend when no drift provided."""
        snapshot = build_family_risk_snapshot(sample_contract_low_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        # risk_trend may or may not be present, but if present should be None
        if "risk_trend" in signal:
            assert signal["risk_trend"] is None
    
    def test_empty_contract_yields_ok(self, sample_contract_empty):
        """Empty contract should yield OK status."""
        snapshot = build_family_risk_snapshot(sample_contract_empty)
        signal = summarize_confusability_for_global_health(snapshot)
        
        assert signal["status"] == "OK"
        assert signal["high_risk_family_count"] == 0


class TestGovernanceSummaryLanguage:
    """Tests for governance summary language constraints."""
    
    def test_summary_no_forbidden_words(self, sample_contract_high_risk):
        """Summary should not contain forbidden words."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        forbidden = check_forbidden_language(signal["summary"])
        assert len(forbidden) == 0, f"Found forbidden words: {forbidden}"
    
    def test_summary_with_drift_no_forbidden_words(self, sample_contract_high_risk):
        """Summary with drift should not contain forbidden words."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        drift = {"net_risk_trend": "DEGRADING"}
        signal = summarize_confusability_for_global_health(snapshot, drift)
        
        forbidden = check_forbidden_language(signal["summary"])
        assert len(forbidden) == 0, f"Found forbidden words: {forbidden}"
    
    def test_summary_is_neutral_descriptive(self, sample_contract_high_risk):
        """Summary should be neutral and descriptive."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        signal = summarize_confusability_for_global_health(snapshot)
        
        summary = signal["summary"]
        
        # Should contain slice name
        assert "test_slice_high" in summary
        # Should contain status
        assert "Status:" in summary


# =============================================================================
# FORBIDDEN LANGUAGE TESTS
# =============================================================================

class TestForbiddenLanguage:
    """Tests for forbidden language detection."""
    
    def test_detect_prescriptive_verbs(self):
        """Should detect prescriptive verbs."""
        text = "You should fix this issue and change the formula."
        found = check_forbidden_language(text)
        
        assert "fix" in found
        assert "should" in found
        assert "change" in found
    
    def test_detect_value_judgments(self):
        """Should detect value judgment words."""
        text = "This is a bad design with poor confusability."
        found = check_forbidden_language(text)
        
        assert "bad" in found
        assert "poor" in found
    
    def test_detect_uplift_claims(self):
        """Should detect uplift-related claims."""
        text = "This shows improvement in the reward signal."
        found = check_forbidden_language(text)
        
        assert "improvement" in found
        assert "reward" in found
    
    def test_clean_text_passes(self):
        """Clean text should pass."""
        text = "Distribution: 2 HIGH-risk families, 1 LOW-risk family out of 3 total."
        found = check_forbidden_language(text)
        
        assert len(found) == 0
    
    def test_risk_module_outputs_clean(self, sample_contract_high_risk):
        """All risk module outputs should be clean."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        drift = compare_family_risk(snapshot, snapshot)
        signal = summarize_confusability_for_global_health(snapshot, drift)
        
        # Check snapshot notes
        found = check_forbidden_language(snapshot["summary_notes"])
        assert len(found) == 0, f"Snapshot notes: {found}"
        
        # Check signal summary
        found = check_forbidden_language(signal["summary"])
        assert len(found) == 0, f"Signal summary: {found}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with real contracts."""
    
    def test_real_contract_snapshot(self):
        """Test with real exported contract."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        
        try:
            contract = export_contract("slice_uplift_goal", config_path)
            contract_dict = contract.to_dict()
            
            snapshot = build_family_risk_snapshot(contract_dict)
            
            assert snapshot["total_family_count"] == len(contract_dict["families"])
            assert "summary_notes" in snapshot
        except FileNotFoundError:
            pytest.skip("Config file not found")
    
    def test_real_contract_drift(self):
        """Test drift analysis with real contract."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        
        try:
            contract = export_contract("slice_uplift_goal", config_path)
            contract_dict = contract.to_dict()
            
            snapshot = build_family_risk_snapshot(contract_dict)
            drift = compare_family_risk(snapshot, snapshot)
            
            assert drift["net_risk_trend"] == "STABLE"
        except FileNotFoundError:
            pytest.skip("Config file not found")
    
    def test_real_contract_governance_signal(self):
        """Test governance signal with real contract."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        
        try:
            contract = export_contract("slice_uplift_goal", config_path)
            contract_dict = contract.to_dict()
            
            snapshot = build_family_risk_snapshot(contract_dict)
            signal = summarize_confusability_for_global_health(snapshot)
            
            assert signal["status"] in ("OK", "ATTENTION", "HOT")
            
            # Check no forbidden language
            forbidden = check_forbidden_language(signal["summary"])
            assert len(forbidden) == 0
        except FileNotFoundError:
            pytest.skip("Config file not found")


# =============================================================================
# PHASE IV: CURRICULUM-COUPLED RISK CONTROL TESTS
# =============================================================================

class TestBuildSliceConfusabilityView:
    """Tests for slice-level confusability governance view."""
    
    def test_view_has_required_fields(self, sample_contract_high_risk):
        """View should have all required fields."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        view = build_slice_confusability_view(
            sample_contract_high_risk, snapshot
        )
        
        required = {
            "slice_name", "high_risk_family_fraction",
            "average_difficulty_band", "slice_confusability_status",
            "total_families", "high_risk_families"
        }
        assert required.issubset(set(view.keys()))
    
    def test_view_computes_high_risk_fraction(self, sample_contract_high_risk):
        """Should correctly compute high risk family fraction."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        view = build_slice_confusability_view(
            sample_contract_high_risk, snapshot
        )
        
        # Contract has 3 families, 2 are HIGH risk
        assert view["total_families"] == 3
        assert view["high_risk_families"] == 2
        assert abs(view["high_risk_family_fraction"] - 2/3) < 0.001
    
    def test_view_averages_difficulty_band(self, sample_contract_high_risk):
        """Should compute average difficulty band."""
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        view = build_slice_confusability_view(
            sample_contract_high_risk, snapshot
        )
        
        # All families are "hard" in sample
        assert view["average_difficulty_band"] == "hard"
    
    def test_view_determines_status_hot(self):
        """Should determine HOT status for high risk fraction."""
        contract = {
            "slice_name": "test_hot",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.85, "difficulty_band": "hard"},
                "fam2": {"members": ["f2"], "avg_confusability": 0.80, "difficulty_band": "hard"},
                "fam3": {"members": ["f3"], "avg_confusability": 0.75, "difficulty_band": "hard"},
            },
        }
        snapshot = build_family_risk_snapshot(contract)
        view = build_slice_confusability_view(contract, snapshot)
        
        assert view["slice_confusability_status"] == "HOT"
    
    def test_view_determines_status_attention(self):
        """Should determine ATTENTION status for moderate risk."""
        contract = {
            "slice_name": "test_attention",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.85, "difficulty_band": "hard"},
                "fam2": {"members": ["f2"], "avg_confusability": 0.3, "difficulty_band": "easy"},
                "fam3": {"members": ["f3"], "avg_confusability": 0.2, "difficulty_band": "easy"},
            },
        }
        snapshot = build_family_risk_snapshot(contract)
        view = build_slice_confusability_view(contract, snapshot)
        
        assert view["slice_confusability_status"] == "ATTENTION"
    
    def test_view_determines_status_ok(self, sample_contract_low_risk):
        """Should determine OK status for low risk."""
        snapshot = build_family_risk_snapshot(sample_contract_low_risk)
        view = build_slice_confusability_view(sample_contract_low_risk, snapshot)
        
        assert view["slice_confusability_status"] == "OK"
    
    def test_view_empty_contract(self, sample_contract_empty):
        """Empty contract should yield OK status."""
        snapshot = build_family_risk_snapshot(sample_contract_empty)
        view = build_slice_confusability_view(sample_contract_empty, snapshot)
        
        assert view["slice_confusability_status"] == "OK"
        assert view["total_families"] == 0
        assert view["high_risk_family_fraction"] == 0.0


class TestSummarizeDecoyConfusabilityForUplift:
    """Tests for uplift/MAAS decoy governance summary."""
    
    def test_summary_has_required_fields(self):
        """Summary should have all required fields."""
        slice_views = {
            "slice1": {
                "slice_name": "slice1",
                "slice_confusability_status": "OK",
            },
        }
        summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        required = {
            "decoy_ok_for_uplift", "slices_needing_review",
            "status", "hot_slices", "attention_slices"
        }
        assert required.issubset(set(summary.keys()))
    
    def test_summary_ok_when_all_slices_ok(self):
        """Should return OK when all slices are OK."""
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
            "slice2": {"slice_confusability_status": "OK"},
        }
        summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        assert summary["status"] == "OK"
        assert summary["decoy_ok_for_uplift"] is True
        assert len(summary["slices_needing_review"]) == 0
    
    def test_summary_block_when_any_hot(self):
        """Should return BLOCK when any slice is HOT."""
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
            "slice2": {"slice_confusability_status": "HOT"},
            "slice3": {"slice_confusability_status": "ATTENTION"},
        }
        summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        assert summary["status"] == "BLOCK"
        assert summary["decoy_ok_for_uplift"] is False
        assert "slice2" in summary["hot_slices"]
        assert "slice2" in summary["slices_needing_review"]
    
    def test_summary_attention_when_any_attention_no_hot(self):
        """Should return ATTENTION when any slice is ATTENTION but none HOT."""
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
            "slice2": {"slice_confusability_status": "ATTENTION"},
        }
        summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        assert summary["status"] == "ATTENTION"
        assert summary["decoy_ok_for_uplift"] is True  # Can proceed but review needed
        assert "slice2" in summary["attention_slices"]
        assert "slice2" in summary["slices_needing_review"]
    
    def test_summary_sorted_slices(self):
        """Slices should be sorted in output."""
        slice_views = {
            "slice_z": {"slice_confusability_status": "HOT"},
            "slice_a": {"slice_confusability_status": "ATTENTION"},
            "slice_m": {"slice_confusability_status": "HOT"},
        }
        summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        assert summary["hot_slices"] == sorted(summary["hot_slices"])
        assert summary["attention_slices"] == sorted(summary["attention_slices"])
        assert summary["slices_needing_review"] == sorted(summary["slices_needing_review"])


class TestBuildConfusabilityDirectorPanel:
    """Tests for director-level confusability panel."""
    
    def test_panel_has_required_fields(self):
        """Panel should have all required fields."""
        risk_snapshot = {
            "high_risk_family_count": 2,
            "families": {},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
        }
        panel = build_confusability_director_panel(risk_snapshot, None, slice_views)
        
        required = {
            "status_light", "high_risk_family_count",
            "net_risk_trend", "slices_hot", "slices_attention", "headline"
        }
        assert required.issubset(set(panel.keys()))
    
    def test_panel_status_light_green(self):
        """Should be GREEN when all indicators are good."""
        risk_snapshot = {
            "high_risk_family_count": 0,
            "families": {},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
        }
        panel = build_confusability_director_panel(risk_snapshot, None, slice_views)
        
        assert panel["status_light"] == "GREEN"
    
    def test_panel_status_light_yellow(self):
        """Should be YELLOW when there are concerns."""
        risk_snapshot = {
            "high_risk_family_count": 3,
            "families": {},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "ATTENTION"},
        }
        panel = build_confusability_director_panel(risk_snapshot, None, slice_views)
        
        assert panel["status_light"] == "YELLOW"
    
    def test_panel_status_light_red(self):
        """Should be RED when there are critical issues."""
        risk_snapshot = {
            "high_risk_family_count": 5,
            "families": {},
        }
        drift = {"net_risk_trend": "DEGRADING"}
        slice_views = {
            "slice1": {"slice_confusability_status": "HOT"},
        }
        panel = build_confusability_director_panel(risk_snapshot, drift, slice_views)
        
        assert panel["status_light"] == "RED"
    
    def test_panel_includes_drift_trend(self):
        """Should include drift trend when provided."""
        risk_snapshot = {
            "high_risk_family_count": 1,
            "families": {},
        }
        drift = {"net_risk_trend": "IMPROVING"}
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
        }
        panel = build_confusability_director_panel(risk_snapshot, drift, slice_views)
        
        assert panel["net_risk_trend"] == "IMPROVING"
    
    def test_panel_collects_hot_slices(self):
        """Should collect all HOT slices."""
        risk_snapshot = {
            "high_risk_family_count": 0,
            "families": {},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "HOT"},
            "slice2": {"slice_confusability_status": "OK"},
            "slice3": {"slice_confusability_status": "HOT"},
        }
        panel = build_confusability_director_panel(risk_snapshot, None, slice_views)
        
        assert len(panel["slices_hot"]) == 2
        assert "slice1" in panel["slices_hot"]
        assert "slice3" in panel["slices_hot"]
        assert panel["slices_hot"] == sorted(panel["slices_hot"])
    
    def test_panel_headline_no_forbidden_words(self):
        """Headline should not contain forbidden words."""
        risk_snapshot = {
            "high_risk_family_count": 2,
            "families": {},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "ATTENTION"},
        }
        panel = build_confusability_director_panel(risk_snapshot, None, slice_views)
        
        forbidden = check_forbidden_language(panel["headline"])
        assert len(forbidden) == 0, f"Found forbidden words: {forbidden}"
    
    def test_panel_headline_descriptive(self):
        """Headline should be descriptive and neutral."""
        risk_snapshot = {
            "high_risk_family_count": 1,
            "families": {},
        }
        drift = {"net_risk_trend": "STABLE"}
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
        }
        panel = build_confusability_director_panel(risk_snapshot, drift, slice_views)
        
        headline = panel["headline"]
        assert "status:" in headline.lower()
        assert len(headline) > 0


class TestPhaseIVIntegration:
    """Integration tests for Phase IV functions."""
    
    def test_full_workflow(self, sample_contract_high_risk):
        """Test complete workflow from contract to director panel."""
        # Build risk snapshot
        snapshot = build_family_risk_snapshot(sample_contract_high_risk)
        
        # Build slice view
        view = build_slice_confusability_view(sample_contract_high_risk, snapshot)
        
        # Build uplift summary
        slice_views = {"test_slice_high": view}
        uplift_summary = summarize_decoy_confusability_for_uplift(slice_views)
        
        # Build director panel
        drift = compare_family_risk(snapshot, snapshot)
        panel = build_confusability_director_panel(snapshot, drift, slice_views)
        
        # Verify all outputs are valid
        assert view["slice_confusability_status"] in ("OK", "ATTENTION", "HOT")
        assert uplift_summary["status"] in ("OK", "ATTENTION", "BLOCK")
        assert panel["status_light"] in ("GREEN", "YELLOW", "RED")
    
    def test_real_contract_slice_view(self):
        """Test slice view with real contract."""
        config_path = "config/curriculum_uplift_phase2.yaml"
        
        try:
            contract = export_contract("slice_uplift_goal", config_path)
            contract_dict = contract.to_dict()
            
            snapshot = build_family_risk_snapshot(contract_dict)
            view = build_slice_confusability_view(contract_dict, snapshot)
            
            assert view["slice_name"] == "slice_uplift_goal"
            assert view["slice_confusability_status"] in ("OK", "ATTENTION", "HOT")
        except FileNotFoundError:
            pytest.skip("Config file not found")


# =============================================================================
# PHASE IV FOLLOW-UP: DRIFT GOVERNANCE & UPLIFT PRE-SCREEN TESTS
# =============================================================================

class TestBuildDecoyFamilyDriftGovernor:
    """Tests for decoy family drift governor."""
    
    def test_governor_has_required_fields(self):
        """Governor should have all required fields."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "MEDIUM"},
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        required = {
            "drift_severity", "families_with_changed_risk",
            "new_high_risk_families", "neutral_notes"
        }
        assert required.issubset(set(governor.keys()))
    
    def test_governor_none_severity_no_changes(self):
        """Should return NONE when no risk changes."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
                "fam2": {"risk_level": "MEDIUM"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
                "fam2": {"risk_level": "MEDIUM"},
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        assert governor["drift_severity"] == "NONE"
        assert len(governor["families_with_changed_risk"]) == 0
    
    def test_governor_minor_severity_adjacent_bands(self):
        """Should return MINOR for adjacent band changes."""
        # LOW -> MEDIUM
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "MEDIUM"},
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        assert governor["drift_severity"] == "MINOR"
        
        # MEDIUM -> HIGH
        old_snap2 = {
            "families": {
                "fam2": {"risk_level": "MEDIUM"},
            },
        }
        new_snap2 = {
            "families": {
                "fam2": {"risk_level": "HIGH"},
            },
        }
        
        governor2 = build_decoy_family_drift_governor(old_snap2, new_snap2)
        assert governor2["drift_severity"] == "MINOR"
        
        # HIGH -> MEDIUM
        old_snap3 = {
            "families": {
                "fam3": {"risk_level": "HIGH"},
            },
        }
        new_snap3 = {
            "families": {
                "fam3": {"risk_level": "MEDIUM"},
            },
        }
        
        governor3 = build_decoy_family_drift_governor(old_snap3, new_snap3)
        assert governor3["drift_severity"] == "MINOR"
    
    def test_governor_major_severity_non_adjacent_bands(self):
        """Should return MAJOR for non-adjacent band changes."""
        # LOW -> HIGH
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "HIGH"},
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        assert governor["drift_severity"] == "MAJOR"
        
        # HIGH -> LOW
        old_snap2 = {
            "families": {
                "fam2": {"risk_level": "HIGH"},
            },
        }
        new_snap2 = {
            "families": {
                "fam2": {"risk_level": "LOW"},
            },
        }
        
        governor2 = build_decoy_family_drift_governor(old_snap2, new_snap2)
        assert governor2["drift_severity"] == "MAJOR"
    
    def test_governor_tracks_changed_families(self):
        """Should track all families with changed risk."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
                "fam2": {"risk_level": "MEDIUM"},
                "fam3": {"risk_level": "HIGH"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "MEDIUM"},  # Changed
                "fam2": {"risk_level": "MEDIUM"},  # Unchanged
                "fam3": {"risk_level": "LOW"},  # Changed
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        assert len(governor["families_with_changed_risk"]) == 2
        changed_fps = {f["fingerprint"] for f in governor["families_with_changed_risk"]}
        assert "fam1" in changed_fps
        assert "fam3" in changed_fps
    
    def test_governor_tracks_new_high_risk_families(self):
        """Should track families that became HIGH risk."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
                "fam2": {"risk_level": "MEDIUM"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "HIGH"},  # Became HIGH
                "fam2": {"risk_level": "MEDIUM"},  # Unchanged
                "fam3": {"risk_level": "HIGH"},  # New family, HIGH
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        assert len(governor["new_high_risk_families"]) == 2
        assert "fam1" in governor["new_high_risk_families"]
        assert "fam3" in governor["new_high_risk_families"]
        assert governor["new_high_risk_families"] == sorted(governor["new_high_risk_families"])
    
    def test_governor_neutral_notes_no_forbidden_words(self):
        """Neutral notes should not contain forbidden words."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "HIGH"},
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        for note in governor["neutral_notes"]:
            forbidden = check_forbidden_language(note)
            assert len(forbidden) == 0, f"Found forbidden words in note '{note}': {forbidden}"
    
    def test_governor_handles_new_families(self):
        """Should handle families present only in new snapshot."""
        old_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
            },
        }
        new_snap = {
            "families": {
                "fam1": {"risk_level": "LOW"},
                "fam2": {"risk_level": "HIGH"},  # New family
            },
        }
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        assert "fam2" in governor["new_high_risk_families"]
        assert governor["drift_severity"] == "NONE"  # No changes to existing families


class TestBuildDecoyUpliftPrescreen:
    """Tests for uplift pre-screen filter."""
    
    def test_prescreen_has_required_fields(self):
        """Prescreen should have all required fields."""
        drift_governor = {
            "drift_severity": "NONE",
        }
        slice_view = {
            "slice_confusability_status": "OK",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        required = {
            "status", "advisory_ok", "drift_severity",
            "slice_status", "advisory_notes"
        }
        assert required.issubset(set(prescreen.keys()))
    
    def test_prescreen_status_ok(self):
        """Should return OK when no concerns."""
        drift_governor = {
            "drift_severity": "NONE",
        }
        slice_view = {
            "slice_confusability_status": "OK",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "OK"
        assert prescreen["advisory_ok"] is True
    
    def test_prescreen_status_block_major_drift_and_hot(self):
        """Should return BLOCK for major drift + HOT slice."""
        drift_governor = {
            "drift_severity": "MAJOR",
        }
        slice_view = {
            "slice_confusability_status": "HOT",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "BLOCK"
        assert prescreen["advisory_ok"] is False
    
    def test_prescreen_status_attention_major_drift_only(self):
        """Should return ATTENTION for major drift without HOT slice."""
        drift_governor = {
            "drift_severity": "MAJOR",
        }
        slice_view = {
            "slice_confusability_status": "OK",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "ATTENTION"
        assert prescreen["advisory_ok"] is True  # Can proceed but review recommended
    
    def test_prescreen_status_attention_hot_slice_only(self):
        """Should return ATTENTION for HOT slice without major drift."""
        drift_governor = {
            "drift_severity": "NONE",
        }
        slice_view = {
            "slice_confusability_status": "HOT",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "ATTENTION"
        assert prescreen["advisory_ok"] is True
    
    def test_prescreen_status_attention_minor_drift_and_attention_slice(self):
        """Should return ATTENTION for minor drift + ATTENTION slice."""
        drift_governor = {
            "drift_severity": "MINOR",
        }
        slice_view = {
            "slice_confusability_status": "ATTENTION",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "ATTENTION"
        assert prescreen["advisory_ok"] is True
    
    def test_prescreen_status_attention_minor_drift_only(self):
        """Should return ATTENTION for minor drift alone."""
        drift_governor = {
            "drift_severity": "MINOR",
        }
        slice_view = {
            "slice_confusability_status": "OK",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "ATTENTION"
        assert prescreen["advisory_ok"] is True
    
    def test_prescreen_status_attention_slice_attention_only(self):
        """Should return ATTENTION for ATTENTION slice alone."""
        drift_governor = {
            "drift_severity": "NONE",
        }
        slice_view = {
            "slice_confusability_status": "ATTENTION",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["status"] == "ATTENTION"
        assert prescreen["advisory_ok"] is True
    
    def test_prescreen_advisory_notes_no_forbidden_words(self):
        """Advisory notes should not contain forbidden words."""
        drift_governor = {
            "drift_severity": "MAJOR",
        }
        slice_view = {
            "slice_confusability_status": "HOT",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        for note in prescreen["advisory_notes"]:
            forbidden = check_forbidden_language(note)
            assert len(forbidden) == 0, f"Found forbidden words in note '{note}': {forbidden}"
    
    def test_prescreen_preserves_drift_and_slice_status(self):
        """Should preserve drift severity and slice status in output."""
        drift_governor = {
            "drift_severity": "MINOR",
        }
        slice_view = {
            "slice_confusability_status": "ATTENTION",
        }
        
        prescreen = build_decoy_uplift_prescreen(drift_governor, slice_view)
        
        assert prescreen["drift_severity"] == "MINOR"
        assert prescreen["slice_status"] == "ATTENTION"


class TestDriftGovernorAndPrescreenIntegration:
    """Integration tests for drift governor and prescreen."""
    
    def test_full_workflow(self):
        """Test complete workflow from snapshots to prescreen."""
        # Build old and new snapshots
        old_contract = {
            "slice_name": "test",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.3, "difficulty_band": "easy"},
            },
        }
        new_contract = {
            "slice_name": "test",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.85, "difficulty_band": "hard"},
            },
        }
        
        old_snap = build_family_risk_snapshot(old_contract)
        new_snap = build_family_risk_snapshot(new_contract)
        
        # Build drift governor
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        
        # Build slice view
        view = build_slice_confusability_view(new_contract, new_snap)
        
        # Build prescreen
        prescreen = build_decoy_uplift_prescreen(governor, view)
        
        # Verify outputs
        assert governor["drift_severity"] in ("NONE", "MINOR", "MAJOR")
        assert prescreen["status"] in ("OK", "ATTENTION", "BLOCK")
        assert prescreen["advisory_ok"] in (True, False)
    
    def test_major_drift_hot_slice_blocks(self):
        """Major drift + HOT slice should result in BLOCK."""
        old_contract = {
            "slice_name": "test",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.2, "difficulty_band": "easy"},
            },
        }
        new_contract = {
            "slice_name": "test",
            "families": {
                "fam1": {"members": ["f1"], "avg_confusability": 0.85, "difficulty_band": "hard"},
            },
        }
        
        old_snap = build_family_risk_snapshot(old_contract)
        new_snap = build_family_risk_snapshot(new_contract)
        
        governor = build_decoy_family_drift_governor(old_snap, new_snap)
        view = build_slice_confusability_view(new_contract, new_snap)
        prescreen = build_decoy_uplift_prescreen(governor, view)
        
        # fam1 went from LOW to HIGH (MAJOR drift)
        # If slice is HOT, should BLOCK
        if view["slice_confusability_status"] == "HOT":
            assert prescreen["status"] == "BLOCK"
            assert prescreen["advisory_ok"] is False

