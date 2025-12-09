"""
Phase IV Governance Drift Cartographer Tests

PHASE IV — NOT RUN IN PHASE I
No uplift claims are made.
Deterministic execution guaranteed.

Tests for:
- Governance drift radar timeline
- Policy feed for MAAS/Global Health
- PR review hooks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.doc_sync_predictor import (
    build_governance_drift_radar,
    build_governance_review_hints_for_pr,
    build_governance_risk_budget,
    evaluate_governance_for_branch_protection,
    evaluate_governance_risk,
    extract_governance_alerts,
    render_governance_pr_comment,
    summarize_governance_radar_for_policy,
)


# ==============================================================================
# 19. GOVERNANCE DRIFT RADAR TIMELINE TESTS (6 tests)
# ==============================================================================


class TestGovernanceDriftRadarTimeline:
    """Tests for governance drift radar timeline."""

    @pytest.fixture
    def sample_risk_evaluations_improving(self) -> List[Dict[str, Any]]:
        """Create sample risk evaluations showing improving trend."""
        return [
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval1",
                "risk_band": "HIGH",
                "new_critical_terms": ["term1"],
                "risk_upgrades_count": 3,
                "risk_downgrades_count": 0,
            },
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval2",
                "risk_band": "MEDIUM",
                "new_critical_terms": [],
                "risk_upgrades_count": 1,
                "risk_downgrades_count": 1,
            },
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval3",
                "risk_band": "LOW",
                "new_critical_terms": [],
                "risk_upgrades_count": 0,
                "risk_downgrades_count": 2,
            },
        ]

    @pytest.fixture
    def sample_risk_evaluations_degrading(self) -> List[Dict[str, Any]]:
        """Create sample risk evaluations showing degrading trend."""
        return [
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval1",
                "risk_band": "LOW",
                "new_critical_terms": [],
                "risk_upgrades_count": 0,
                "risk_downgrades_count": 0,
            },
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval2",
                "risk_band": "MEDIUM",
                "new_critical_terms": [],
                "risk_upgrades_count": 1,
                "risk_downgrades_count": 0,
            },
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval3",
                "risk_band": "HIGH",
                "new_critical_terms": ["term1"],
                "risk_upgrades_count": 2,
                "risk_downgrades_count": 0,
            },
        ]

    @pytest.fixture
    def sample_risk_evaluations_stable(self) -> List[Dict[str, Any]]:
        """Create sample risk evaluations showing stable trend."""
        return [
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval1",
                "risk_band": "MEDIUM",
                "new_critical_terms": [],
                "risk_upgrades_count": 1,
                "risk_downgrades_count": 0,
            },
            {
                "schema_version": "1.0.0",
                "evaluation_id": "eval2",
                "risk_band": "MEDIUM",
                "new_critical_terms": [],
                "risk_upgrades_count": 0,
                "risk_downgrades_count": 1,
            },
        ]

    def test_radar_computes_total_runs(
        self, sample_risk_evaluations_improving: List[Dict[str, Any]]
    ) -> None:
        """Test that radar correctly counts total runs."""
        radar = build_governance_drift_radar(sample_risk_evaluations_improving)
        
        assert radar["total_runs"] == 3

    def test_radar_counts_high_risk_runs(
        self, sample_risk_evaluations_improving: List[Dict[str, Any]]
    ) -> None:
        """Test that radar counts runs with HIGH risk band."""
        radar = build_governance_drift_radar(sample_risk_evaluations_improving)
        
        assert radar["runs_with_high_risk"] == 1

    def test_radar_counts_new_critical_terms_runs(
        self, sample_risk_evaluations_improving: List[Dict[str, Any]]
    ) -> None:
        """Test that radar counts runs with new critical terms."""
        radar = build_governance_drift_radar(sample_risk_evaluations_improving)
        
        assert radar["runs_with_new_critical_terms"] == 1

    def test_radar_detects_improving_trend(
        self, sample_risk_evaluations_improving: List[Dict[str, Any]]
    ) -> None:
        """Test that radar detects improving trend."""
        radar = build_governance_drift_radar(sample_risk_evaluations_improving)
        
        assert radar["trend_status"] == "IMPROVING"

    def test_radar_detects_degrading_trend(
        self, sample_risk_evaluations_degrading: List[Dict[str, Any]]
    ) -> None:
        """Test that radar detects degrading trend."""
        radar = build_governance_drift_radar(sample_risk_evaluations_degrading)
        
        assert radar["trend_status"] == "DEGRADING"

    def test_radar_computes_max_consecutive_high_runs(
        self, sample_risk_evaluations_degrading: List[Dict[str, Any]]
    ) -> None:
        """Test that radar computes max consecutive HIGH runs."""
        radar = build_governance_drift_radar(sample_risk_evaluations_degrading)
        
        # Only one HIGH run, so max consecutive should be 1
        assert radar["max_consecutive_high_runs"] == 1

    def test_radar_handles_empty_evaluations(self) -> None:
        """Test that radar handles empty evaluation list."""
        radar = build_governance_drift_radar([])
        
        assert radar["total_runs"] == 0
        assert radar["trend_status"] == "STABLE"

    def test_radar_is_deterministic(
        self, sample_risk_evaluations_stable: List[Dict[str, Any]]
    ) -> None:
        """Test that radar output is deterministic."""
        radar1 = build_governance_drift_radar(sample_risk_evaluations_stable)
        radar2 = build_governance_drift_radar(sample_risk_evaluations_stable)
        
        assert radar1["radar_id"] == radar2["radar_id"]
        assert radar1["generated_at"] == radar2["generated_at"]


# ==============================================================================
# 20. POLICY FEED FOR MAAS TESTS (6 tests)
# ==============================================================================


class TestPolicyFeedForMAAS:
    """Tests for policy feed summary."""

    @pytest.fixture
    def sample_radar_hot(self) -> Dict[str, Any]:
        """Create sample radar with HOT status."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_hot_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 3,
            "runs_with_new_critical_terms": 2,
            "trend_status": "DEGRADING",
            "max_consecutive_high_runs": 3,
            "summary": "Radar analysis: 3 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_radar_attention(self) -> Dict[str, Any]:
        """Create sample radar with ATTENTION status."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_attention_456",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 3,
            "runs_with_high_risk": 1,
            "runs_with_new_critical_terms": 0,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 1,
            "summary": "Radar analysis: 1 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_radar_ok(self) -> Dict[str, Any]:
        """Create sample radar with OK status."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_ok_789",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 0,
            "runs_with_new_critical_terms": 0,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 0,
            "summary": "Radar analysis: No elevated risk indicators...",
        }

    def test_policy_feed_maps_to_hot_status(
        self, sample_radar_hot: Dict[str, Any]
    ) -> None:
        """Test that policy feed maps degrading trend to HOT."""
        policy = summarize_governance_radar_for_policy(sample_radar_hot)
        
        assert policy["status"] == "HOT"
        assert policy["policy_attention_required"] is True

    def test_policy_feed_maps_to_attention_status(
        self, sample_radar_attention: Dict[str, Any]
    ) -> None:
        """Test that policy feed maps high risk runs to ATTENTION."""
        policy = summarize_governance_radar_for_policy(sample_radar_attention)
        
        assert policy["status"] == "ATTENTION"
        assert policy["policy_attention_required"] is True

    def test_policy_feed_maps_to_ok_status(
        self, sample_radar_ok: Dict[str, Any]
    ) -> None:
        """Test that policy feed maps stable/no risk to OK."""
        policy = summarize_governance_radar_for_policy(sample_radar_ok)
        
        assert policy["status"] == "OK"
        assert policy["policy_attention_required"] is False

    def test_policy_feed_has_required_fields(
        self, sample_radar_ok: Dict[str, Any]
    ) -> None:
        """Test that policy feed has all required fields."""
        policy = summarize_governance_radar_for_policy(sample_radar_ok)
        
        required_fields = {
            "schema_version",
            "policy_summary_id",
            "generated_at",
            "policy_attention_required",
            "status",
            "key_terms_to_review",
            "notes",
        }
        assert required_fields.issubset(set(policy.keys()))
        assert policy["schema_version"] == "1.0.0"

    def test_policy_feed_notes_are_neutral(
        self, sample_radar_hot: Dict[str, Any]
    ) -> None:
        """Test that policy feed notes use neutral language."""
        policy = summarize_governance_radar_for_policy(sample_radar_hot)
        
        all_notes = " ".join(policy.get("notes", [])).lower()
        forbidden_terms = ["good", "bad", "healthy", "unhealthy", "better", "worse"]
        
        for term in forbidden_terms:
            assert term not in all_notes, f"Evaluative term '{term}' found in notes"

    def test_policy_feed_is_deterministic(
        self, sample_radar_ok: Dict[str, Any]
    ) -> None:
        """Test that policy feed is deterministic."""
        policy1 = summarize_governance_radar_for_policy(sample_radar_ok)
        policy2 = summarize_governance_radar_for_policy(sample_radar_ok)
        
        assert policy1["policy_summary_id"] == policy2["policy_summary_id"]
        assert policy1["generated_at"] == policy2["generated_at"]


# ==============================================================================
# 21. PR REVIEW HOOKS TESTS (6 tests)
# ==============================================================================


class TestPRReviewHooks:
    """Tests for PR review hints."""

    @pytest.fixture
    def sample_risk_eval_for_pr(self) -> Dict[str, Any]:
        """Create sample risk evaluation for PR testing."""
        return {
            "schema_version": "1.0.0",
            "evaluation_id": "eval_pr_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "risk_band": "HIGH",
            "new_critical_terms": ["volatile_term"],
            "risk_upgrades_count": 2,
            "risk_downgrades_count": 0,
            "summary": "Current state: 1 new critical term(s)...",
        }

    @pytest.fixture
    def sample_alerts_for_pr(self) -> List[Dict[str, Any]]:
        """Create sample alerts for PR testing."""
        return [
            {
                "term": "volatile_term",
                "old_risk": None,
                "new_risk": "critical",
                "alert_kind": "new_critical",
                "message": "Term 'volatile_term' added to watch list with critical risk level.",
            },
            {
                "term": "upgraded_term",
                "old_risk": "moderate",
                "new_risk": "high",
                "alert_kind": "upgraded",
                "message": "Risk level for 'upgraded_term' increased from moderate to high.",
            },
        ]

    @pytest.fixture
    def sample_files_touched(self) -> List[str]:
        """Create sample files touched in PR."""
        return [
            "docs/governance/volatile_term.md",
            "backend/api/upgraded_term.py",
            "tests/test_stable.py",
        ]

    def test_pr_hints_highlight_terms_in_files(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
        sample_files_touched: List[str],
    ) -> None:
        """Test that PR hints highlight terms that appear in touched files."""
        hints = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        
        # volatile_term should be highlighted (appears in file path)
        assert "volatile_term" in hints["highlight_terms"]
        # upgraded_term should be highlighted (appears in file path)
        assert "upgraded_term" in hints["highlight_terms"]

    def test_pr_hints_build_sections_to_review(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
        sample_files_touched: List[str],
    ) -> None:
        """Test that PR hints build sections_to_review."""
        hints = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        
        sections = hints["sections_to_review"]
        assert len(sections) > 0
        
        # Check that sections have required fields
        for section in sections:
            assert "file" in section
            assert "term" in section
            assert "reason" in section

    def test_pr_hints_summary_hint_for_highlighted_terms(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
        sample_files_touched: List[str],
    ) -> None:
        """Test that PR hints include summary hint."""
        hints = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        
        assert "summary_hint" in hints
        assert len(hints["summary_hint"]) > 0

    def test_pr_hints_empty_when_no_matching_files(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
    ) -> None:
        """Test that PR hints are minimal when no files match."""
        hints = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, ["unrelated_file.py"]
        )
        
        # Should still have structure but no highlighted terms
        assert "highlight_terms" in hints
        assert "sections_to_review" in hints

    def test_pr_hints_uses_neutral_language(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
        sample_files_touched: List[str],
    ) -> None:
        """Test that PR hints use neutral language."""
        hints = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        
        summary = hints.get("summary_hint", "").lower()
        all_reasons = " ".join(s.get("reason", "") for s in hints.get("sections_to_review", [])).lower()
        combined = (summary + " " + all_reasons).lower()
        
        forbidden_terms = ["fix", "wrong", "error", "mistake", "bad", "incorrect"]
        for term in forbidden_terms:
            assert term not in combined, f"Non-neutral term '{term}' found in hints"

    def test_pr_hints_is_deterministic(
        self,
        sample_risk_eval_for_pr: Dict[str, Any],
        sample_alerts_for_pr: List[Dict[str, Any]],
        sample_files_touched: List[str],
    ) -> None:
        """Test that PR hints are deterministic."""
        hints1 = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        hints2 = build_governance_review_hints_for_pr(
            sample_risk_eval_for_pr, sample_alerts_for_pr, sample_files_touched
        )
        
        assert hints1["review_hints_id"] == hints2["review_hints_id"]
        assert hints1["generated_at"] == hints2["generated_at"]
        assert hints1["highlight_terms"] == hints2["highlight_terms"]


# ==============================================================================
# 22. GOVERNANCE RISK BUDGET TESTS (6 tests)
# ==============================================================================


class TestGovernanceRiskBudget:
    """Tests for governance risk budgeting."""

    @pytest.fixture
    def sample_radar_below_limit(self) -> Dict[str, Any]:
        """Create sample radar below budget limits."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_below_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 1,
            "runs_with_new_critical_terms": 2,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 1,
            "summary": "Radar analysis: 1 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_radar_nearing_limit(self) -> Dict[str, Any]:
        """Create sample radar nearing budget limits."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_nearing_456",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 2,  # 2 of 3 = 66.7% (below 80% threshold, but let's test with 3)
            "runs_with_new_critical_terms": 4,  # 4 of 5 = 80% (at threshold)
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 2,
            "summary": "Radar analysis: 2 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_radar_exceeded(self) -> Dict[str, Any]:
        """Create sample radar exceeding budget limits."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_exceeded_789",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 5,  # Exceeds limit of 3
            "runs_with_new_critical_terms": 6,  # Exceeds limit of 5
            "trend_status": "DEGRADING",
            "max_consecutive_high_runs": 4,
            "summary": "Radar analysis: 5 run(s) with HIGH risk band...",
        }

    def test_budget_below_limits_status_ok(
        self, sample_radar_below_limit: Dict[str, Any]
    ) -> None:
        """Test that budget below limits produces OK status."""
        budget = build_governance_risk_budget(sample_radar_below_limit)
        
        assert budget["status"] == "OK"
        assert budget["budget_ok"] is True

    def test_budget_nearing_limit_status(
        self, sample_radar_nearing_limit: Dict[str, Any]
    ) -> None:
        """Test that budget at 80% threshold produces NEARING_LIMIT status."""
        budget = build_governance_risk_budget(sample_radar_nearing_limit)
        
        # 4 of 5 = 80%, should trigger NEARING_LIMIT
        assert budget["status"] == "NEARING_LIMIT"
        assert budget["budget_ok"] is True

    def test_budget_exceeded_status(
        self, sample_radar_exceeded: Dict[str, Any]
    ) -> None:
        """Test that budget exceeding limits produces EXCEEDED status."""
        budget = build_governance_risk_budget(sample_radar_exceeded)
        
        assert budget["status"] == "EXCEEDED"
        assert budget["budget_ok"] is False

    def test_budget_computes_remaining(
        self, sample_radar_below_limit: Dict[str, Any]
    ) -> None:
        """Test that budget correctly computes remaining budget."""
        budget = build_governance_risk_budget(sample_radar_below_limit)
        
        # 1 high run of 3 limit = 2 remaining
        assert budget["remaining_high_runs"] == 2
        # 2 critical terms runs of 5 limit = 3 remaining
        assert budget["remaining_new_critical_terms"] == 3

    def test_budget_has_neutral_notes(
        self, sample_radar_below_limit: Dict[str, Any]
    ) -> None:
        """Test that budget notes use neutral language."""
        budget = build_governance_risk_budget(sample_radar_below_limit)
        
        all_notes = " ".join(budget.get("neutral_notes", [])).lower()
        forbidden_terms = ["good", "bad", "healthy", "unhealthy", "better", "worse"]
        
        for term in forbidden_terms:
            assert term not in all_notes, f"Evaluative term '{term}' found in notes"

    def test_budget_is_deterministic(
        self, sample_radar_below_limit: Dict[str, Any]
    ) -> None:
        """Test that budget is deterministic."""
        budget1 = build_governance_risk_budget(sample_radar_below_limit)
        budget2 = build_governance_risk_budget(sample_radar_below_limit)
        
        assert budget1["budget_id"] == budget2["budget_id"]
        assert budget1["generated_at"] == budget2["generated_at"]


# ==============================================================================
# 23. BRANCH PROTECTION ADAPTER TESTS (6 tests)
# ==============================================================================


class TestBranchProtectionAdapter:
    """Tests for branch protection evaluation."""

    @pytest.fixture
    def sample_radar_for_branch(self) -> Dict[str, Any]:
        """Create sample radar for branch protection testing."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_branch_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 2,
            "runs_with_new_critical_terms": 1,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 1,
            "summary": "Radar analysis: 2 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_policy_hot(self) -> Dict[str, Any]:
        """Create sample policy summary with HOT status."""
        return {
            "schema_version": "1.0.0",
            "policy_summary_id": "policy_hot_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "policy_attention_required": True,
            "status": "HOT",
            "key_terms_to_review": [],
            "notes": ["Analyzed 5 risk evaluation run(s)", "3 run(s) classified as HIGH risk"],
        }

    @pytest.fixture
    def sample_policy_ok(self) -> Dict[str, Any]:
        """Create sample policy summary with OK status."""
        return {
            "schema_version": "1.0.0",
            "policy_summary_id": "policy_ok_456",
            "generated_at": "2025-01-15T12:00:00Z",
            "policy_attention_required": False,
            "status": "OK",
            "key_terms_to_review": [],
            "notes": ["No significant governance risk indicators detected"],
        }

    @pytest.fixture
    def sample_budget_exceeded(self) -> Dict[str, Any]:
        """Create sample budget with EXCEEDED status."""
        return {
            "schema_version": "1.0.0",
            "budget_id": "budget_exceeded_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "budget_ok": False,
            "remaining_high_runs": -2,
            "remaining_new_critical_terms": -1,
            "status": "EXCEEDED",
            "neutral_notes": ["High-risk runs: 5 of 3 limit", "New critical terms runs: 6 of 5 limit"],
        }

    @pytest.fixture
    def sample_budget_ok(self) -> Dict[str, Any]:
        """Create sample budget with OK status."""
        return {
            "schema_version": "1.0.0",
            "budget_id": "budget_ok_456",
            "generated_at": "2025-01-15T12:00:00Z",
            "budget_ok": True,
            "remaining_high_runs": 2,
            "remaining_new_critical_terms": 3,
            "status": "OK",
            "neutral_notes": ["High-risk runs: 1 of 3 limit", "New critical terms runs: 2 of 5 limit"],
        }

    def test_branch_protection_blocks_on_policy_hot(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_hot: Dict[str, Any],
        sample_budget_ok: Dict[str, Any],
    ) -> None:
        """Test that branch protection BLOCKS when policy status is HOT."""
        branch_eval = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_hot, sample_budget_ok
        )
        
        assert branch_eval["status"] == "BLOCK"
        assert branch_eval["branch_safe"] is False
        assert "Policy status is HOT" in branch_eval["blocking_reasons"]

    def test_branch_protection_blocks_on_budget_exceeded(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_ok: Dict[str, Any],
        sample_budget_exceeded: Dict[str, Any],
    ) -> None:
        """Test that branch protection BLOCKS when budget is exceeded."""
        branch_eval = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_ok, sample_budget_exceeded
        )
        
        assert branch_eval["status"] == "BLOCK"
        assert branch_eval["branch_safe"] is False
        assert "Risk budget exceeded" in branch_eval["blocking_reasons"]

    def test_branch_protection_warns_on_attention(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_ok: Dict[str, Any],
        sample_budget_ok: Dict[str, Any],
    ) -> None:
        """Test that branch protection allows when all clear."""
        # Create policy with ATTENTION status
        policy_attention = {
            "schema_version": "1.0.0",
            "policy_summary_id": "policy_attention_789",
            "generated_at": "2025-01-15T12:00:00Z",
            "policy_attention_required": True,
            "status": "ATTENTION",
            "key_terms_to_review": [],
            "notes": ["Policy status is ATTENTION"],
        }
        
        branch_eval = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, policy_attention, sample_budget_ok
        )
        
        assert branch_eval["status"] == "WARN"
        assert branch_eval["branch_safe"] is True

    def test_branch_protection_ok_when_all_clear(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_ok: Dict[str, Any],
        sample_budget_ok: Dict[str, Any],
    ) -> None:
        """Test that branch protection allows when all clear."""
        branch_eval = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_ok, sample_budget_ok
        )
        
        assert branch_eval["status"] == "OK"
        assert branch_eval["branch_safe"] is True
        assert len(branch_eval["blocking_reasons"]) == 0

    def test_branch_protection_has_required_fields(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_ok: Dict[str, Any],
        sample_budget_ok: Dict[str, Any],
    ) -> None:
        """Test that branch protection has all required fields."""
        branch_eval = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_ok, sample_budget_ok
        )
        
        required_fields = {
            "schema_version",
            "branch_protection_id",
            "generated_at",
            "branch_safe",
            "status",
            "blocking_reasons",
            "advisory_notes",
        }
        assert required_fields.issubset(set(branch_eval.keys()))
        assert branch_eval["schema_version"] == "1.0.0"

    def test_branch_protection_is_deterministic(
        self,
        sample_radar_for_branch: Dict[str, Any],
        sample_policy_ok: Dict[str, Any],
        sample_budget_ok: Dict[str, Any],
    ) -> None:
        """Test that branch protection is deterministic."""
        branch_eval1 = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_ok, sample_budget_ok
        )
        branch_eval2 = evaluate_governance_for_branch_protection(
            sample_radar_for_branch, sample_policy_ok, sample_budget_ok
        )
        
        assert branch_eval1["branch_protection_id"] == branch_eval2["branch_protection_id"]
        assert branch_eval1["generated_at"] == branch_eval2["generated_at"]


# ==============================================================================
# 24. PR COMMENT RENDERER TESTS (6 tests)
# ==============================================================================


class TestPRCommentRenderer:
    """Tests for PR comment rendering."""

    @pytest.fixture
    def sample_review_hints(self) -> Dict[str, Any]:
        """Create sample review hints for PR comment testing."""
        return {
            "schema_version": "1.0.0",
            "review_hints_id": "hints_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "highlight_terms": ["volatile_term", "upgraded_term"],
            "sections_to_review": [
                {
                    "file": "docs/governance/volatile_term.md",
                    "term": "volatile_term",
                    "reason": "Term 'volatile_term' recently added to watch list with critical risk level.",
                },
                {
                    "file": "backend/api/upgraded_term.py",
                    "term": "upgraded_term",
                    "reason": "Term 'upgraded_term' risk level increased from moderate to high.",
                },
            ],
            "summary_hint": "This PR touches 2 governance term(s) that may require terminology review: volatile_term, upgraded_term.",
        }

    @pytest.fixture
    def sample_radar_summary_stable(self) -> Dict[str, Any]:
        """Create sample radar summary with STABLE status."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_stable_123",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 1,
            "runs_with_new_critical_terms": 0,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 1,
            "summary": "Radar analysis: 1 run(s) with HIGH risk band...",
        }

    @pytest.fixture
    def sample_radar_summary_degrading(self) -> Dict[str, Any]:
        """Create sample radar summary with DEGRADING status."""
        return {
            "schema_version": "1.0.0",
            "radar_id": "radar_degrading_456",
            "generated_at": "2025-01-15T12:00:00Z",
            "total_runs": 5,
            "runs_with_high_risk": 3,
            "runs_with_new_critical_terms": 2,
            "trend_status": "DEGRADING",
            "max_consecutive_high_runs": 2,
            "summary": "Radar analysis: 3 run(s) with HIGH risk band...",
        }

    def test_pr_comment_includes_radar_status(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_stable: Dict[str, Any],
    ) -> None:
        """Test that PR comment includes radar status."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_stable)
        
        assert "STABLE" in comment
        assert "Radar Status" in comment

    def test_pr_comment_includes_highlighted_terms(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_stable: Dict[str, Any],
    ) -> None:
        """Test that PR comment includes highlighted terms."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_stable)
        
        assert "volatile_term" in comment
        assert "upgraded_term" in comment
        assert "Terms Requiring Attention" in comment

    def test_pr_comment_includes_sections_table(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_stable: Dict[str, Any],
    ) -> None:
        """Test that PR comment includes sections to review table."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_stable)
        
        assert "Sections to Review" in comment
        assert "| File | Term | Reason |" in comment
        assert "docs/governance/volatile_term.md" in comment

    def test_pr_comment_valid_markdown(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_stable: Dict[str, Any],
    ) -> None:
        """Test that PR comment is valid Markdown."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_stable)
        
        # Check for markdown headers
        assert "##" in comment
        # Check for markdown table
        assert "|" in comment
        # Check for code formatting
        assert "`" in comment

    def test_pr_comment_no_forbidden_language(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_stable: Dict[str, Any],
    ) -> None:
        """Test that PR comment uses no forbidden language."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_stable)
        comment_lower = comment.lower()
        
        forbidden_terms = ["fix", "wrong", "error", "mistake", "bad", "incorrect", "good", "better"]
        for term in forbidden_terms:
            assert term not in comment_lower, f"Non-neutral term '{term}' found in PR comment"

    def test_pr_comment_handles_degrading_trend(
        self,
        sample_review_hints: Dict[str, Any],
        sample_radar_summary_degrading: Dict[str, Any],
    ) -> None:
        """Test that PR comment handles DEGRADING trend status."""
        comment = render_governance_pr_comment(sample_review_hints, sample_radar_summary_degrading)
        
        assert "DEGRADING" in comment
        assert "📉" in comment  # Degrading emoji

