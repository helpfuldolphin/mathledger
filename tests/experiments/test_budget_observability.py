"""
Minimal tests for budget observability.
"""

import pytest
from experiments.budget_observability import (
    BudgetSummary,
    BudgetHealthStatus,
    classify_budget_health,
    summarize_budget_from_logs,
)


class TestBudgetSummary:
    """Tests for BudgetSummary dataclass."""
    
    def test_summary_properties(self):
        """Test summary property calculations."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=5,
            max_candidates_hit_count=10,
            timeout_abstentions_total=20,
        )
        
        assert summary.budget_exhausted_pct == 5.0
        assert summary.max_candidates_hit_pct == 10.0
        assert summary.timeout_abstentions_avg == 0.2
    
    def test_summary_zero_cycles(self):
        """Test summary with zero cycles."""
        summary = BudgetSummary(
            total_cycles=0,
            budget_exhausted_count=0,
            max_candidates_hit_count=0,
            timeout_abstentions_total=0,
        )
        
        assert summary.budget_exhausted_pct == 0.0
        assert summary.timeout_abstentions_avg == 0.0
    
    def test_summary_to_dict(self):
        """Test JSON serialization."""
        summary = BudgetSummary(
            total_cycles=50,
            budget_exhausted_count=2,
            max_candidates_hit_count=3,
            timeout_abstentions_total=5,
        )
        
        d = summary.to_dict()
        assert d["total_cycles"] == 50
        assert d["budget_exhausted_pct"] == 4.0
        assert d["timeout_abstentions_avg"] == 0.1


class TestClassifyBudgetHealth:
    """Tests for budget health classification."""
    
    def test_classify_safe(self):
        """Test SAFE classification."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=0,  # 0% exhausted
            max_candidates_hit_count=0,
            timeout_abstentions_total=5,  # 0.05 avg
        )
        
        status = classify_budget_health(summary)
        assert status == BudgetHealthStatus.SAFE
    
    def test_classify_tight(self):
        """Test TIGHT classification."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=3,  # 3% exhausted (between 1% and 5%)
            max_candidates_hit_count=0,
            timeout_abstentions_total=0,
        )
        
        status = classify_budget_health(summary)
        assert status == BudgetHealthStatus.TIGHT
    
    def test_classify_starved_exhausted(self):
        """Test STARVED classification via exhausted budget."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=10,  # 10% exhausted (>= 5%)
            max_candidates_hit_count=0,
            timeout_abstentions_total=0,
        )
        
        status = classify_budget_health(summary)
        assert status == BudgetHealthStatus.STARVED
    
    def test_classify_starved_timeout(self):
        """Test STARVED classification via timeout abstentions."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=0,
            max_candidates_hit_count=0,
            timeout_abstentions_total=150,  # 1.5 avg (>= 1.0)
        )
        
        status = classify_budget_health(summary)
        assert status == BudgetHealthStatus.STARVED
    
    def test_classify_invalid(self):
        """Test INVALID classification (no data)."""
        summary = BudgetSummary(
            total_cycles=0,
            budget_exhausted_count=0,
            max_candidates_hit_count=0,
            timeout_abstentions_total=0,
        )
        
        status = classify_budget_health(summary)
        assert status == BudgetHealthStatus.INVALID


class TestSummarizeBudgetFromLogs:
    """Tests for log summarization."""
    
    def test_summarize_empty_logs(self):
        """Test summarizing empty logs."""
        result = summarize_budget_from_logs([])
        
        assert result["schema_version"] == "1.0.0"
        assert result["status"] == "INVALID"
        assert result["total_cycles"] == 0
    
    def test_summarize_budget_fields(self):
        """Test summarizing logs with budget fields."""
        log_lines = [
            '{"budget": {"budget_exhausted": true, "max_candidates_hit": false, "timeout_abstentions": 2}}',
            '{"budget": {"budget_exhausted": false, "max_candidates_hit": true, "timeout_abstentions": 0}}',
            '{"budget": {"budget_exhausted": false, "max_candidates_hit": false, "timeout_abstentions": 1}}',
        ]
        
        result = summarize_budget_from_logs(log_lines)
        
        assert result["total_cycles"] == 3
        assert result["summary"]["budget_exhausted_count"] == 1
        assert result["summary"]["max_candidates_hit_count"] == 1
        assert result["summary"]["timeout_abstentions_total"] == 3
        assert result["status"] in ["SAFE", "TIGHT", "STARVED"]
    
    def test_summarize_nested_metrics(self):
        """Test summarizing logs with nested budget in metrics."""
        log_lines = [
            '{"metrics": {"budget": {"budget_exhausted": true, "timeout_abstentions": 5}}}',
        ]
        
        result = summarize_budget_from_logs(log_lines)
        
        assert result["total_cycles"] == 1
        assert result["summary"]["budget_exhausted_count"] == 1
        assert result["summary"]["timeout_abstentions_total"] == 5
    
    def test_summarize_invalid_json(self):
        """Test handling invalid JSON lines."""
        log_lines = [
            '{"budget": {"budget_exhausted": true}}',
            'invalid json line',
            '{"budget": {"budget_exhausted": false}}',
        ]
        
        result = summarize_budget_from_logs(log_lines)
        
        assert result["total_cycles"] == 2  # Only valid JSON lines counted

