"""
Tests for Budget Observability Tools (Agent A5)

PHASE II — NOT USED IN PHASE I

Tests verify:
    1. Budget health classification logic (SAFE/TIGHT/STARVED/INVALID)
    2. Threshold boundaries for each classification
    3. Docs-friendly report generation
    4. Health JSON output format
    5. Markdown summary generation for CI

These tests ensure budget observability is purely read-only and does not
influence experiment execution.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.summarize_budget_usage import (
    BudgetSummary,
    BudgetHealthStatus,
    BudgetHealthResult,
    classify_budget_health,
    format_docs_report,
    format_health_json,
    format_markdown_summary,
    THRESHOLD_EXHAUSTED_SAFE,
    THRESHOLD_EXHAUSTED_TIGHT,
    THRESHOLD_TIMEOUT_SAFE,
    THRESHOLD_TIMEOUT_TIGHT,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def safe_summary() -> BudgetSummary:
    """Budget summary with SAFE health status."""
    return BudgetSummary(
        path="test/safe.jsonl",
        slice_name="slice_uplift_goal",
        mode="rfl",
        total_cycles=1000,
        budget_exhausted_count=5,  # 0.5% < 1% threshold
        max_candidates_hit_count=950,  # 95% (expected for RFL)
        timeout_abstentions_total=50,  # 0.05 avg/cycle < 0.1 threshold
        cycles_with_budget_field=1000,
    )


@pytest.fixture
def tight_summary() -> BudgetSummary:
    """Budget summary with TIGHT health status."""
    return BudgetSummary(
        path="test/tight.jsonl",
        slice_name="slice_uplift_sparse",
        mode="rfl",
        total_cycles=500,
        budget_exhausted_count=15,  # 3% in [1%, 5%] = TIGHT
        max_candidates_hit_count=400,  # 80%
        timeout_abstentions_total=200,  # 0.4 avg/cycle in [0.1, 1.0] = TIGHT
        cycles_with_budget_field=500,
    )


@pytest.fixture
def starved_summary() -> BudgetSummary:
    """Budget summary with STARVED health status."""
    return BudgetSummary(
        path="test/starved.jsonl",
        slice_name="slice_uplift_tree",
        mode="rfl",
        total_cycles=200,
        budget_exhausted_count=20,  # 10% > 5% threshold = STARVED
        max_candidates_hit_count=100,  # 50%
        timeout_abstentions_total=300,  # 1.5 avg/cycle > 1.0 threshold = STARVED
        cycles_with_budget_field=200,
    )


@pytest.fixture
def invalid_no_cycles() -> BudgetSummary:
    """Budget summary with no cycles (INVALID)."""
    return BudgetSummary(
        path="test/empty.jsonl",
        slice_name="slice_uplift_goal",
        mode="rfl",
        total_cycles=0,
        budget_exhausted_count=0,
        max_candidates_hit_count=0,
        timeout_abstentions_total=0,
        cycles_with_budget_field=0,
    )


@pytest.fixture
def invalid_no_budget_field() -> BudgetSummary:
    """Budget summary with cycles but no budget telemetry (INVALID)."""
    return BudgetSummary(
        path="test/no_budget.jsonl",
        slice_name="slice_uplift_goal",
        mode="rfl",
        total_cycles=100,
        budget_exhausted_count=0,
        max_candidates_hit_count=0,
        timeout_abstentions_total=0,
        cycles_with_budget_field=0,  # No budget telemetry
    )


# =============================================================================
# Test: Health Classification
# =============================================================================


class TestBudgetHealthClassification:
    """Tests for classify_budget_health() function."""

    def test_classify_safe(self, safe_summary: BudgetSummary):
        """Test SAFE classification."""
        result = classify_budget_health(safe_summary)
        
        assert result.status == BudgetHealthStatus.SAFE
        assert len(result.reasons) >= 3  # At least 3 metrics evaluated
        assert "SAFE" in result.reasons[0]  # First reason should mention SAFE
        assert result.metrics["budget_exhausted_pct"] < THRESHOLD_EXHAUSTED_SAFE

    def test_classify_tight(self, tight_summary: BudgetSummary):
        """Test TIGHT classification."""
        result = classify_budget_health(tight_summary)
        
        assert result.status == BudgetHealthStatus.TIGHT
        assert any("TIGHT" in r for r in result.reasons)

    def test_classify_starved(self, starved_summary: BudgetSummary):
        """Test STARVED classification."""
        result = classify_budget_health(starved_summary)
        
        assert result.status == BudgetHealthStatus.STARVED
        assert any("STARVED" in r for r in result.reasons)

    def test_classify_invalid_no_cycles(self, invalid_no_cycles: BudgetSummary):
        """Test INVALID classification when no cycles."""
        result = classify_budget_health(invalid_no_cycles)
        
        assert result.status == BudgetHealthStatus.INVALID
        assert "total_cycles=0" in result.reasons[0]

    def test_classify_invalid_no_budget_field(self, invalid_no_budget_field: BudgetSummary):
        """Test INVALID classification when no budget telemetry."""
        result = classify_budget_health(invalid_no_budget_field)
        
        assert result.status == BudgetHealthStatus.INVALID
        assert "cycles_with_budget_field=0" in result.reasons[0]


class TestBudgetHealthThresholds:
    """Tests for threshold boundaries."""

    def test_threshold_exhausted_boundary_safe(self):
        """Test boundary: budget_exhausted_pct just below SAFE threshold."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=9,  # 0.9% < 1%
            max_candidates_hit_count=900,
            timeout_abstentions_total=50,  # 0.05 avg
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.SAFE

    def test_threshold_exhausted_boundary_tight(self):
        """Test boundary: budget_exhausted_pct at SAFE threshold = TIGHT."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=10,  # 1.0% = TIGHT
            max_candidates_hit_count=900,
            timeout_abstentions_total=50,  # 0.05 avg
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.TIGHT

    def test_threshold_exhausted_boundary_starved(self):
        """Test boundary: budget_exhausted_pct above TIGHT threshold = STARVED."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=51,  # 5.1% > 5%
            max_candidates_hit_count=900,
            timeout_abstentions_total=50,  # 0.05 avg
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.STARVED

    def test_threshold_timeout_boundary_safe(self):
        """Test boundary: timeout_abstentions_avg just below SAFE threshold."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=5,  # 0.5%
            max_candidates_hit_count=900,
            timeout_abstentions_total=90,  # 0.09 avg < 0.1
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.SAFE

    def test_threshold_timeout_boundary_tight(self):
        """Test boundary: timeout_abstentions_avg at SAFE threshold = TIGHT."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=5,  # 0.5%
            max_candidates_hit_count=900,
            timeout_abstentions_total=100,  # 0.1 avg = TIGHT
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.TIGHT

    def test_threshold_timeout_boundary_starved(self):
        """Test boundary: timeout_abstentions_avg above TIGHT threshold = STARVED."""
        summary = BudgetSummary(
            path="test.jsonl",
            slice_name="test",
            mode="rfl",
            total_cycles=1000,
            budget_exhausted_count=5,  # 0.5%
            max_candidates_hit_count=900,
            timeout_abstentions_total=1100,  # 1.1 avg > 1.0
            cycles_with_budget_field=1000,
        )
        result = classify_budget_health(summary)
        assert result.status == BudgetHealthStatus.STARVED


class TestHealthResultFormat:
    """Tests for BudgetHealthResult.to_dict()."""

    def test_to_dict_structure(self, safe_summary: BudgetSummary):
        """Test to_dict() returns expected structure."""
        result = classify_budget_health(safe_summary)
        d = result.to_dict()
        
        assert "status" in d
        assert "reasons" in d
        assert "metrics" in d
        assert d["status"] == "SAFE"
        assert isinstance(d["reasons"], list)
        assert isinstance(d["metrics"], dict)

    def test_metrics_include_all_fields(self, safe_summary: BudgetSummary):
        """Test metrics include all expected fields."""
        result = classify_budget_health(safe_summary)
        metrics = result.metrics
        
        assert "budget_exhausted_pct" in metrics
        assert "max_candidates_hit_pct" in metrics
        assert "timeout_abstentions_avg" in metrics
        assert "total_cycles" in metrics


# =============================================================================
# Test: Docs-Friendly Report
# =============================================================================


class TestDocsReport:
    """Tests for format_docs_report()."""

    def test_docs_report_markdown_format(self, safe_summary: BudgetSummary):
        """Test docs report is valid Markdown."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        # Should contain Markdown elements
        assert "###" in report  # Header
        assert "|" in report  # Table
        assert "**Overall Health:" in report

    def test_docs_report_contains_slice_name(self, safe_summary: BudgetSummary):
        """Test docs report includes slice name."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        assert safe_summary.slice_name in report

    def test_docs_report_contains_metrics(self, safe_summary: BudgetSummary):
        """Test docs report includes metric values."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        # Should contain percentage values
        assert "%" in report
        assert "avg/cycle" in report

    def test_docs_report_contains_health_status(self, safe_summary: BudgetSummary):
        """Test docs report includes health status."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        assert health.status.value in report

    def test_docs_report_contains_emoji_safe(self, safe_summary: BudgetSummary):
        """Test SAFE status has ✅ emoji."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        assert "✅" in report

    def test_docs_report_contains_emoji_tight(self, tight_summary: BudgetSummary):
        """Test TIGHT status has ⚠️ emoji."""
        health = classify_budget_health(tight_summary)
        report = format_docs_report(tight_summary, health)
        
        assert "⚠️" in report

    def test_docs_report_contains_emoji_starved(self, starved_summary: BudgetSummary):
        """Test STARVED status has 🔥 emoji."""
        health = classify_budget_health(starved_summary)
        report = format_docs_report(starved_summary, health)
        
        assert "🔥" in report

    def test_docs_report_contains_hint(self, safe_summary: BudgetSummary):
        """Test docs report includes human-friendly hint."""
        health = classify_budget_health(safe_summary)
        report = format_docs_report(safe_summary, health)
        
        # Should contain hint marker
        assert "💡" in report or "Hint:" in report

    def test_docs_report_tight_hint_mentions_increase(self, tight_summary: BudgetSummary):
        """Test TIGHT hint suggests increasing budget."""
        health = classify_budget_health(tight_summary)
        report = format_docs_report(tight_summary, health)
        
        # Hint should mention increasing budget
        assert "increas" in report.lower() or "10" in report

    def test_docs_report_starved_hint_mentions_review(self, starved_summary: BudgetSummary):
        """Test STARVED hint suggests reviewing parameters."""
        health = classify_budget_health(starved_summary)
        report = format_docs_report(starved_summary, health)
        
        # Hint should mention review or tuning
        assert "review" in report.lower() or "tuning" in report.lower()


# =============================================================================
# Test: Health JSON Output
# =============================================================================


class TestHealthJson:
    """Tests for format_health_json()."""

    def test_health_json_valid(
        self,
        safe_summary: BudgetSummary,
        tight_summary: BudgetSummary,
    ):
        """Test health JSON is valid JSON."""
        summaries = [safe_summary, tight_summary]
        json_str = format_health_json(summaries)
        
        # Should be parseable JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_health_json_structure(
        self,
        safe_summary: BudgetSummary,
        tight_summary: BudgetSummary,
    ):
        """Test health JSON has expected structure."""
        summaries = [safe_summary, tight_summary]
        json_str = format_health_json(summaries)
        data = json.loads(json_str)
        
        assert "phase" in data
        assert "health_report" in data
        assert "aggregate" in data
        assert len(data["health_report"]) == 2

    def test_health_json_aggregate(
        self,
        safe_summary: BudgetSummary,
        tight_summary: BudgetSummary,
        starved_summary: BudgetSummary,
    ):
        """Test health JSON aggregate counts."""
        summaries = [safe_summary, tight_summary, starved_summary]
        json_str = format_health_json(summaries)
        data = json.loads(json_str)
        
        agg = data["aggregate"]
        assert agg["total_slices"] == 3
        assert agg["safe_count"] == 1
        assert agg["tight_count"] == 1
        assert agg["starved_count"] == 1


# =============================================================================
# Test: Markdown Summary for CI
# =============================================================================


class TestMarkdownSummary:
    """Tests for format_markdown_summary()."""

    def test_markdown_summary_valid(
        self,
        safe_summary: BudgetSummary,
        tight_summary: BudgetSummary,
    ):
        """Test Markdown summary is valid."""
        summaries = [safe_summary, tight_summary]
        md = format_markdown_summary(summaries)
        
        assert "##" in md  # Header
        assert "|" in md  # Table

    def test_markdown_summary_contains_emoji(
        self,
        safe_summary: BudgetSummary,
        starved_summary: BudgetSummary,
    ):
        """Test Markdown summary uses status emoji."""
        summaries = [safe_summary, starved_summary]
        md = format_markdown_summary(summaries)
        
        # Should contain emoji indicators
        assert "✅" in md or "⚠️" in md or "🔴" in md

    def test_markdown_summary_advisory_for_starved(
        self,
        starved_summary: BudgetSummary,
    ):
        """Test Markdown summary includes advisory for STARVED."""
        summaries = [starved_summary]
        md = format_markdown_summary(summaries)
        
        assert "Advisory" in md or "⚠️" in md


# =============================================================================
# Test: Read-Only Invariant
# =============================================================================


class TestReadOnlyInvariant:
    """Tests verifying observability tools are read-only."""

    def test_classify_does_not_modify_summary(self, safe_summary: BudgetSummary):
        """Test classify_budget_health does not modify input summary."""
        original_dict = safe_summary.to_dict()
        
        _ = classify_budget_health(safe_summary)
        
        # Summary should be unchanged
        assert safe_summary.to_dict() == original_dict

    def test_format_functions_are_pure(self, safe_summary: BudgetSummary):
        """Test format functions are pure (same input = same output)."""
        health = classify_budget_health(safe_summary)
        
        report1 = format_docs_report(safe_summary, health)
        report2 = format_docs_report(safe_summary, health)
        
        assert report1 == report2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

