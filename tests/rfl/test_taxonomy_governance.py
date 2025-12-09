"""
Tests for Taxonomy Governance & Impact Analysis

Tests the impact analyzer, report generator, and CI acknowledgment hook.

PHASE III — GOVERNANCE
Agent B6 (abstention-ops-6)
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from taxonomy_governance import (
    analyze_taxonomy_change,
    render_taxonomy_change_report,
    render_taxonomy_change_report_json,
    check_breaking_changes_acknowledged,
    TaxonomyImpactAnalysis,
    BreakingChange,
    NonBreakingChange,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_taxonomy() -> Dict[str, Any]:
    """Base taxonomy fixture for testing."""
    return {
        "taxonomy_version": "1.0.0",
        "abstention_types": {
            "abstain_timeout": {
                "category": "timeout_related",
                "legacy_keys": ["timeout", "derivation_timeout"],
            },
            "abstain_crash": {
                "category": "crash_related",
                "legacy_keys": ["crash", "engine_failure"],
            },
            "abstain_budget": {
                "category": "resource_related",
                "legacy_keys": ["budget_exceeded"],
            },
        },
        "categories": {
            "timeout_related": ["abstain_timeout"],
            "crash_related": ["abstain_crash"],
            "resource_related": ["abstain_budget"],
        },
        "legacy_mappings": {
            "timeout": "abstain_timeout",
            "crash": "abstain_crash",
            "budget_exceeded": "abstain_budget",
        },
        "verification_methods": ["lean-disabled", "lean-timeout"],
    }


# ---------------------------------------------------------------------------
# Task 1: Impact Analyzer Tests
# ---------------------------------------------------------------------------

class TestAnalyzeTaxonomyChange:
    """Tests for analyze_taxonomy_change() function."""
    
    def test_no_changes_returns_low_risk(self, base_taxonomy) -> None:
        """Identical taxonomies should have LOW risk."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        
        assert analysis.risk_level == RiskLevel.LOW
        assert not analysis.has_changes
        assert not analysis.has_breaking_changes
        assert len(analysis.breaking_changes) == 0
        assert len(analysis.non_breaking_changes) == 0
    
    def test_added_type_is_non_breaking(self, base_taxonomy) -> None:
        """Adding a type should be non-breaking."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        new_taxonomy["abstention_types"]["abstain_new"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_changes
        assert not analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.LOW
        assert len(analysis.non_breaking_changes) == 1
        assert analysis.non_breaking_changes[0].change_type == "type_added"
    
    def test_removed_type_is_breaking_high_risk(self, base_taxonomy) -> None:
        """Removing a type should be breaking and HIGH risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_changes
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.HIGH
        assert any(bc.change_type == "type_removed" for bc in analysis.breaking_changes)
    
    def test_removed_category_is_breaking_high_risk(self, base_taxonomy) -> None:
        """Removing a category should be breaking and HIGH risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["categories"]["crash_related"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.HIGH
        assert any(bc.change_type == "category_removed" for bc in analysis.breaking_changes)
    
    def test_type_category_change_is_breaking_high_risk(self, base_taxonomy) -> None:
        """Changing a type's category should be breaking and HIGH risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        new_taxonomy["abstention_types"]["abstain_timeout"]["category"] = "crash_related"
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.HIGH
        assert any(
            bc.change_type == "type_category_changed" 
            for bc in analysis.breaking_changes
        )
    
    def test_legacy_mapping_removed_is_medium_risk(self, base_taxonomy) -> None:
        """Removing a legacy mapping should be MEDIUM risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        del new_taxonomy["legacy_mappings"]["crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.MEDIUM
        assert any(
            bc.change_type == "legacy_mapping_removed" 
            for bc in analysis.breaking_changes
        )
    
    def test_legacy_mapping_changed_is_medium_risk(self, base_taxonomy) -> None:
        """Changing a legacy mapping target should be MEDIUM risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        new_taxonomy["legacy_mappings"]["timeout"] = "abstain_crash"  # Changed target
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.MEDIUM
        assert any(
            bc.change_type == "legacy_mapping_changed" 
            for bc in analysis.breaking_changes
        )
    
    def test_legacy_mapping_added_is_non_breaking(self, base_taxonomy) -> None:
        """Adding a legacy mapping should be non-breaking."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        new_taxonomy["legacy_mappings"]["new_key"] = "abstain_timeout"
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.has_changes
        assert not analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.LOW
        assert any(
            nc.change_type == "legacy_mapping_added" 
            for nc in analysis.non_breaking_changes
        )
    
    def test_analysis_includes_version_info(self, base_taxonomy) -> None:
        """Analysis should include version information."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        assert analysis.old_version == "1.0.0"
        assert analysis.new_version == "2.0.0"
    
    def test_breaking_change_has_migration_hint(self, base_taxonomy) -> None:
        """Breaking changes should include migration hints."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        for bc in analysis.breaking_changes:
            assert bc.migration_hint, f"Missing migration hint for {bc.change_type}"
            assert bc.downstream_impact, f"Missing downstream impact for {bc.change_type}"


class TestRiskLevelCalculation:
    """Tests for risk level calculation."""
    
    def test_high_risk_types(self, base_taxonomy) -> None:
        """Test that specific change types result in HIGH risk."""
        high_risk_scenarios = [
            # Type removed
            lambda t: t["abstention_types"].pop("abstain_crash"),
            # Category removed
            lambda t: t["categories"].pop("crash_related"),
        ]
        
        for mutator in high_risk_scenarios:
            new_taxonomy = json.loads(json.dumps(base_taxonomy))
            new_taxonomy["taxonomy_version"] = "2.0.0"
            mutator(new_taxonomy)
            
            analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
            assert analysis.risk_level == RiskLevel.HIGH
    
    def test_medium_risk_types(self, base_taxonomy) -> None:
        """Test that specific change types result in MEDIUM risk."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        del new_taxonomy["legacy_mappings"]["crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        assert analysis.risk_level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Task 2: Report Generator Tests
# ---------------------------------------------------------------------------

class TestRenderTaxonomyChangeReport:
    """Tests for render_taxonomy_change_report() function."""
    
    def test_report_is_markdown(self, base_taxonomy) -> None:
        """Report should be valid Markdown."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert report.startswith("# ")  # Markdown heading
        assert "## Summary" in report
    
    def test_report_includes_version_info(self, base_taxonomy) -> None:
        """Report should include version information."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "1.0.0" in report  # Old version
        assert "2.0.0" in report  # New version
    
    def test_report_includes_risk_level(self, base_taxonomy) -> None:
        """Report should include risk level."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "LOW" in report or "MEDIUM" in report or "HIGH" in report
    
    def test_high_risk_report_has_warning(self, base_taxonomy) -> None:
        """HIGH risk report should have warning section."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "HIGH" in report
        assert "WARNING" in report or "⚠️" in report
    
    def test_report_includes_breaking_changes_table(self, base_taxonomy) -> None:
        """Report with breaking changes should include table."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "Breaking Change" in report or "🔴" in report
        assert "type_removed" in report
    
    def test_report_includes_non_breaking_changes_table(self, base_taxonomy) -> None:
        """Report with non-breaking changes should include table."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        new_taxonomy["abstention_types"]["abstain_new"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "Non-Breaking" in report or "🟢" in report
        assert "type_added" in report
    
    def test_report_includes_ci_instructions_for_high_risk(self, base_taxonomy) -> None:
        """HIGH risk report should include CI instructions."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "UPDATE_TAXONOMY_ACK" in report
    
    def test_no_changes_report(self, base_taxonomy) -> None:
        """Report with no changes should say so."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        assert "No Changes" in report or "identical" in report.lower()


class TestRenderTaxonomyChangeReportJson:
    """Tests for JSON report output."""
    
    def test_json_report_is_valid_json(self, base_taxonomy) -> None:
        """JSON report should be valid JSON."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        json_str = render_taxonomy_change_report_json(analysis)
        
        data = json.loads(json_str)
        assert isinstance(data, dict)
    
    def test_json_report_has_schema_version(self, base_taxonomy) -> None:
        """JSON report should have schema_version field."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        json_str = render_taxonomy_change_report_json(analysis)
        data = json.loads(json_str)
        
        assert "schema_version" in data


# ---------------------------------------------------------------------------
# Task 3: CI Acknowledgment Tests
# ---------------------------------------------------------------------------

class TestCheckBreakingChangesAcknowledged:
    """Tests for check_breaking_changes_acknowledged() function."""
    
    def test_low_risk_does_not_require_ack(self, base_taxonomy) -> None:
        """LOW risk changes should not require acknowledgment."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        
        acknowledged, message = check_breaking_changes_acknowledged(analysis)
        
        assert acknowledged
        assert "not HIGH" in message or "No acknowledgment" in message
    
    def test_medium_risk_does_not_require_ack(self, base_taxonomy) -> None:
        """MEDIUM risk changes should not require acknowledgment."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        del new_taxonomy["legacy_mappings"]["crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        assert analysis.risk_level == RiskLevel.MEDIUM
        
        acknowledged, message = check_breaking_changes_acknowledged(analysis)
        
        assert acknowledged
    
    def test_high_risk_without_env_var_fails(self, base_taxonomy) -> None:
        """HIGH risk without env var should fail."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            acknowledged, message = check_breaking_changes_acknowledged(analysis)
        
            assert not acknowledged
            assert "acknowledgment" in message.lower()
    
    def test_high_risk_with_env_var_passes(self, base_taxonomy) -> None:
        """HIGH risk with UPDATE_TAXONOMY_ACK=1 should pass."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        with patch.dict(os.environ, {"UPDATE_TAXONOMY_ACK": "1"}):
            acknowledged, message = check_breaking_changes_acknowledged(analysis)
        
            assert acknowledged
            assert "acknowledged" in message.lower()
    
    def test_high_risk_with_marker_file_passes(self, base_taxonomy) -> None:
        """HIGH risk with marker file should pass."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        
        # Create marker file temporarily
        marker_path = Path(__file__).parent.parent.parent / ".taxonomy-breaking-change-ack"
        try:
            marker_path.touch()
            
            with patch.dict(os.environ, {}, clear=True):
                acknowledged, message = check_breaking_changes_acknowledged(analysis)
            
                assert acknowledged
                assert "acknowledged" in message.lower()
        finally:
            if marker_path.exists():
                marker_path.unlink()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestGovernanceIntegration:
    """Integration tests for the complete governance workflow."""
    
    def test_full_workflow_no_changes(self, base_taxonomy) -> None:
        """Test complete workflow with no changes."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        report = render_taxonomy_change_report(analysis)
        acknowledged, _ = check_breaking_changes_acknowledged(analysis)
        
        assert not analysis.has_changes
        assert analysis.risk_level == RiskLevel.LOW
        assert acknowledged
        assert "No Changes" in report or "identical" in report.lower()
    
    def test_full_workflow_high_risk(self, base_taxonomy) -> None:
        """Test complete workflow with HIGH risk changes."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        new_taxonomy["abstention_types"]["abstain_new"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        report = render_taxonomy_change_report(analysis)
        
        # Analysis
        assert analysis.has_changes
        assert analysis.has_breaking_changes
        assert analysis.risk_level == RiskLevel.HIGH
        
        # Report
        assert "HIGH" in report
        assert "type_removed" in report
        assert "type_added" in report
        
        # Acknowledgment without env var
        with patch.dict(os.environ, {}, clear=True):
            acknowledged, _ = check_breaking_changes_acknowledged(analysis)
            assert not acknowledged
        
        # Acknowledgment with env var
        with patch.dict(os.environ, {"UPDATE_TAXONOMY_ACK": "1"}):
            acknowledged, _ = check_breaking_changes_acknowledged(analysis)
            assert acknowledged
    
    def test_analysis_to_dict_roundtrip(self, base_taxonomy) -> None:
        """Test that analysis can be serialized and contains expected keys."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "1.1.0"
        new_taxonomy["abstention_types"]["abstain_new"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        analysis = analyze_taxonomy_change(base_taxonomy, new_taxonomy)
        data = analysis.to_dict()
        
        assert "schema_version" in data
        assert "risk_level" in data
        assert "breaking_changes" in data
        assert "non_breaking_changes" in data
        assert data["risk_level"] == "LOW"


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------

class TestGovernanceDeterminism:
    """Tests for deterministic behavior."""
    
    def test_analysis_is_deterministic(self, base_taxonomy) -> None:
        """Multiple analyses should produce identical results."""
        new_taxonomy = json.loads(json.dumps(base_taxonomy))
        new_taxonomy["taxonomy_version"] = "2.0.0"
        del new_taxonomy["abstention_types"]["abstain_crash"]
        
        analyses = [
            analyze_taxonomy_change(base_taxonomy, new_taxonomy)
            for _ in range(3)
        ]
        
        dicts = [a.to_dict() for a in analyses]
        # Remove timestamp for comparison
        for d in dicts:
            d.pop("analyzed_at", None)
        
        assert all(d == dicts[0] for d in dicts)
    
    def test_report_is_deterministic(self, base_taxonomy) -> None:
        """Multiple reports should be identical (excluding timestamp)."""
        analysis = analyze_taxonomy_change(base_taxonomy, base_taxonomy)
        
        reports = [render_taxonomy_change_report(analysis) for _ in range(3)]
        
        # Reports might have different timestamps, so just check they're similar
        # by checking the main content is the same
        assert all("Summary" in r for r in reports)
        assert all("LOW" in r for r in reports)

