"""
Tests for Phase IV: Taxonomy Impact on Metrics, Docs, and Curriculum

Tests the impact analyzers and director panel for taxonomy changes.

PHASE IV — METRICS, DOCS, CURRICULUM IMPACT
Agent B6 (abstention-ops-6)
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from taxonomy_governance import (
    analyze_taxonomy_impact_on_metrics,
    analyze_taxonomy_alignment_with_docs_and_curriculum,
    build_taxonomy_director_panel,
    TaxonomyImpactAnalysis,
    BreakingChange,
    NonBreakingChange,
    RiskLevel,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_analysis_no_changes() -> TaxonomyImpactAnalysis:
    """Analysis with no changes."""
    return TaxonomyImpactAnalysis(
        old_version="1.0.0",
        new_version="1.0.0",
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture
def sample_analysis_type_removed() -> TaxonomyImpactAnalysis:
    """Analysis with a type removed."""
    return TaxonomyImpactAnalysis(
        old_version="1.0.0",
        new_version="2.0.0",
        risk_level=RiskLevel.HIGH,
        breaking_changes=[
            BreakingChange(
                change_type="type_removed",
                description="Abstention type 'abstain_crash' has been removed",
                affected_component="AbstentionType.ABSTAIN_CRASH",
                downstream_impact="Code referencing this type will fail",
                migration_hint="Replace with alternative type",
            ),
        ],
    )


@pytest.fixture
def sample_metrics_config() -> Dict[str, Any]:
    """Sample metrics configuration."""
    return {
        "metrics": {
            "abstention_rate": {
                "description": "Abstention rate by type",
                "aggregate_by": ["abstain_timeout", "abstain_crash"],
            },
            "category_breakdown": {
                "description": "Breakdown by semantic category",
                "categories": ["timeout_related", "crash_related"],
            },
            "throughput": {
                "description": "Proofs per hour",
                "independent": True,
            },
        }
    }


@pytest.fixture
def sample_vocab_index() -> Dict[str, Any]:
    """Sample vocabulary index from documentation."""
    return {
        "abstain_crash": ["docs/abstention_guide.md:45", "docs/api_reference.md:120"],
        "abstain_timeout": ["docs/abstention_guide.md:50"],
        "timeout_related": ["docs/categories.md:30"],
    }


@pytest.fixture
def sample_curriculum_manifest() -> Dict[str, Any]:
    """Sample curriculum manifest."""
    return {
        "version": 2,
        "systems": {
            "pl": {
                "slices": [
                    {
                        "name": "slice_easy",
                        "gates": {
                            "abstention": {
                                "max_rate_pct": 2.0,
                                "exclude_types": ["abstain_crash"],
                            },
                        },
                    },
                    {
                        "name": "slice_hard",
                        "gates": {
                            "abstention": {
                                "max_rate_pct": 5.0,
                            },
                        },
                    },
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Task 1: Metrics Impact Tests
# ---------------------------------------------------------------------------

class TestAnalyzeTaxonomyImpactOnMetrics:
    """Tests for analyze_taxonomy_impact_on_metrics()."""
    
    def test_no_changes_returns_ok(self, sample_analysis_no_changes) -> None:
        """No changes should result in OK status if all categories covered."""
        comprehensive_metrics = {
            "metrics": {
                "timeout_metrics": {"category": "timeout_related"},
                "crash_metrics": {"category": "crash_related"},
                "resource_metrics": {"category": "resource_related"},
                "oracle_metrics": {"category": "oracle_related"},
                "invalid_metrics": {"category": "invalid_related"},
            }
        }
        result = analyze_taxonomy_impact_on_metrics(
            sample_analysis_no_changes,
            comprehensive_metrics
        )
        
        assert result["status"] == "OK"
        assert result["affected_metric_kinds"] == []
    
    def test_removed_type_affects_metrics(self, sample_analysis_type_removed, sample_metrics_config) -> None:
        """Removed type should be detected in metrics."""
        result = analyze_taxonomy_impact_on_metrics(
            sample_analysis_type_removed,
            sample_metrics_config
        )
        
        assert result["status"] == "MISALIGNED"
        assert "abstention_rate" in result["affected_metric_kinds"]
    
    def test_non_covered_categories_detected(self, sample_analysis_no_changes) -> None:
        """Categories without metrics should be detected."""
        minimal_metrics = {
            "metrics": {
                "throughput": {"description": "Only throughput"},
            }
        }
        
        result = analyze_taxonomy_impact_on_metrics(
            sample_analysis_no_changes,
            minimal_metrics
        )
        
        assert result["status"] == "PARTIAL"
        assert len(result["non_covered_categories"]) > 0
    
    def test_all_categories_covered_returns_ok(self, sample_analysis_no_changes) -> None:
        """All categories covered should return OK."""
        comprehensive_metrics = {
            "metrics": {
                "timeout_metrics": {"category": "timeout_related"},
                "crash_metrics": {"category": "crash_related"},
                "resource_metrics": {"category": "resource_related"},
                "oracle_metrics": {"category": "oracle_related"},
                "invalid_metrics": {"category": "invalid_related"},
            }
        }
        
        result = analyze_taxonomy_impact_on_metrics(
            sample_analysis_no_changes,
            comprehensive_metrics
        )
        
        assert result["status"] == "OK"
        assert result["non_covered_categories"] == []


# ---------------------------------------------------------------------------
# Task 2: Docs & Curriculum Alignment Tests
# ---------------------------------------------------------------------------

class TestAnalyzeTaxonomyAlignmentWithDocsAndCurriculum:
    """Tests for analyze_taxonomy_alignment_with_docs_and_curriculum()."""
    
    def test_no_changes_returns_aligned(self, sample_analysis_no_changes, sample_vocab_index, sample_curriculum_manifest) -> None:
        """No changes should result in ALIGNED status."""
        result = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_no_changes,
            sample_vocab_index,
            sample_curriculum_manifest
        )
        
        assert result["alignment_status"] == "ALIGNED"
        assert result["missing_doc_updates"] == []
        assert result["slices_with_outdated_types"] == []
    
    def test_removed_type_detected_in_docs(self, sample_analysis_type_removed, sample_vocab_index, sample_curriculum_manifest) -> None:
        """Removed type should be detected in documentation."""
        result = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_type_removed,
            sample_vocab_index,
            sample_curriculum_manifest
        )
        
        assert result["alignment_status"] in {"PARTIAL", "OUT_OF_DATE"}
        assert len(result["missing_doc_updates"]) > 0
        assert any("abstain_crash" in update["term"].lower() for update in result["missing_doc_updates"])
    
    def test_removed_type_detected_in_curriculum(self, sample_analysis_type_removed, sample_vocab_index, sample_curriculum_manifest) -> None:
        """Removed type should be detected in curriculum slices."""
        result = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_type_removed,
            sample_vocab_index,
            sample_curriculum_manifest
        )
        
        assert len(result["slices_with_outdated_types"]) > 0
        assert any(s["slice_name"] == "slice_easy" for s in result["slices_with_outdated_types"])
    
    def test_empty_vocab_index_handled(self, sample_analysis_type_removed, sample_curriculum_manifest) -> None:
        """Empty vocab index should be handled gracefully."""
        result = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_type_removed,
            {},
            sample_curriculum_manifest
        )
        
        assert result["missing_doc_updates"] == []
        assert len(result["slices_with_outdated_types"]) > 0  # Still check curriculum
    
    def test_empty_curriculum_handled(self, sample_analysis_type_removed, sample_vocab_index) -> None:
        """Empty curriculum should be handled gracefully."""
        result = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_type_removed,
            sample_vocab_index,
            {}
        )
        
        assert result["slices_with_outdated_types"] == []
        assert len(result["missing_doc_updates"]) > 0  # Still check docs


# ---------------------------------------------------------------------------
# Task 3: Director Panel Tests
# ---------------------------------------------------------------------------

class TestBuildTaxonomyDirectorPanel:
    """Tests for build_taxonomy_director_panel()."""
    
    def test_green_status_all_ok(self) -> None:
        """All systems OK should result in GREEN."""
        impact_metrics = {"status": "OK", "affected_metric_kinds": [], "non_covered_categories": []}
        impact_docs = {"alignment_status": "ALIGNED", "missing_doc_updates": [], "slices_with_outdated_types": []}
        risk_analysis = {"risk_level": "LOW"}
        
        panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
        
        assert panel["status_light"] == "GREEN"
        assert panel["requires_ack"] is False
        assert "stable" in panel["headline"].lower()
    
    def test_yellow_status_partial_issues(self) -> None:
        """Partial issues should result in YELLOW."""
        impact_metrics = {"status": "PARTIAL", "affected_metric_kinds": [], "non_covered_categories": ["timeout_related"]}
        impact_docs = {"alignment_status": "PARTIAL", "missing_doc_updates": [{"term": "test"}], "slices_with_outdated_types": []}
        risk_analysis = {"risk_level": "MEDIUM"}
        
        panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
        
        assert panel["status_light"] == "YELLOW"
        assert panel["requires_ack"] is False
    
    def test_red_status_high_risk(self) -> None:
        """HIGH risk should result in RED."""
        impact_metrics = {"status": "MISALIGNED", "affected_metric_kinds": ["abstention_rate"], "non_covered_categories": []}
        impact_docs = {"alignment_status": "OUT_OF_DATE", "missing_doc_updates": [{"term": "test"}], "slices_with_outdated_types": [{"slice_name": "test"}]}
        risk_analysis = {"risk_level": "HIGH"}
        
        panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
        
        assert panel["status_light"] == "RED"
        assert panel["requires_ack"] is True
        assert "HIGH" in panel["headline"] or "risk" in panel["headline"].lower()
    
    def test_panel_includes_all_fields(self) -> None:
        """Panel should include all required fields."""
        impact_metrics = {"status": "OK", "affected_metric_kinds": [], "non_covered_categories": []}
        impact_docs = {"alignment_status": "ALIGNED", "missing_doc_updates": [], "slices_with_outdated_types": []}
        risk_analysis = {"risk_level": "LOW"}
        
        panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
        
        required_fields = ["status_light", "risk_level", "alignment_status", "headline", "requires_ack"]
        for field in required_fields:
            assert field in panel, f"Missing field: {field}"
    
    def test_requires_ack_only_for_high_risk(self) -> None:
        """requires_ack should only be True for HIGH risk."""
        for risk in ["LOW", "MEDIUM"]:
            risk_analysis = {"risk_level": risk}
            impact_metrics = {"status": "OK", "affected_metric_kinds": [], "non_covered_categories": []}
            impact_docs = {"alignment_status": "ALIGNED", "missing_doc_updates": [], "slices_with_outdated_types": []}
            
            panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
            assert panel["requires_ack"] is False, f"Risk {risk} should not require ack"
        
        risk_analysis = {"risk_level": "HIGH"}
        panel = build_taxonomy_director_panel(impact_metrics, impact_docs, risk_analysis)
        assert panel["requires_ack"] is True


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestPhase4Integration:
    """Integration tests for Phase IV functions."""
    
    def test_full_workflow_no_changes(self, sample_analysis_no_changes, sample_vocab_index, sample_curriculum_manifest) -> None:
        """Full workflow with no changes should be all green."""
        comprehensive_metrics = {
            "metrics": {
                "timeout_metrics": {"category": "timeout_related"},
                "crash_metrics": {"category": "crash_related"},
                "resource_metrics": {"category": "resource_related"},
                "oracle_metrics": {"category": "oracle_related"},
                "invalid_metrics": {"category": "invalid_related"},
            }
        }
        metrics_impact = analyze_taxonomy_impact_on_metrics(
            sample_analysis_no_changes,
            comprehensive_metrics
        )
        docs_impact = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_no_changes,
            sample_vocab_index,
            sample_curriculum_manifest
        )
        risk_analysis = sample_analysis_no_changes.to_dict()
        
        panel = build_taxonomy_director_panel(metrics_impact, docs_impact, risk_analysis)
        
        assert panel["status_light"] == "GREEN"
        assert panel["alignment_status"] == "ALIGNED"
        assert panel["requires_ack"] is False
    
    def test_full_workflow_with_breaking_changes(self, sample_analysis_type_removed, sample_metrics_config, sample_vocab_index, sample_curriculum_manifest) -> None:
        """Full workflow with breaking changes should show issues."""
        metrics_impact = analyze_taxonomy_impact_on_metrics(
            sample_analysis_type_removed,
            sample_metrics_config
        )
        docs_impact = analyze_taxonomy_alignment_with_docs_and_curriculum(
            sample_analysis_type_removed,
            sample_vocab_index,
            sample_curriculum_manifest
        )
        risk_analysis = sample_analysis_type_removed.to_dict()
        
        panel = build_taxonomy_director_panel(metrics_impact, docs_impact, risk_analysis)
        
        assert panel["status_light"] in {"YELLOW", "RED"}
        assert panel["requires_ack"] is True
        assert panel["affected_metric_kinds"] or panel["missing_doc_updates"] > 0 or panel["outdated_slices"] > 0

