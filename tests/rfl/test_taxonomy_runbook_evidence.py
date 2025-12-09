"""
Tests for Taxonomy Change Runbook & Evidence Capsule

Tests the runbook generation and evidence tile summarization functions.

PHASE IV Extension — RUNBOOK & EVIDENCE
Agent B6 (abstention-ops-6)
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from taxonomy_governance import (
    build_taxonomy_change_runbook,
    summarize_taxonomy_change_for_evidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_impact_metrics() -> Dict[str, Any]:
    """Minimal metrics impact (no issues)."""
    return {
        "status": "OK",
        "affected_metric_kinds": [],
        "non_covered_categories": [],
    }


@pytest.fixture
def minimal_impact_docs() -> Dict[str, Any]:
    """Minimal docs/curriculum impact (no issues)."""
    return {
        "alignment_status": "ALIGNED",
        "missing_doc_updates": [],
        "slices_with_outdated_types": [],
    }


@pytest.fixture
def impacted_metrics() -> Dict[str, Any]:
    """Metrics impact with issues."""
    return {
        "status": "MISALIGNED",
        "affected_metric_kinds": ["abstention_rate", "category_breakdown"],
        "non_covered_categories": ["timeout_related"],
    }


@pytest.fixture
def impacted_docs() -> Dict[str, Any]:
    """Docs/curriculum impact with issues."""
    return {
        "alignment_status": "OUT_OF_DATE",
        "missing_doc_updates": [
            {"term": "abstain_crash", "locations": ["docs/guide.md:45"]},
            {"term": "abstain_timeout", "locations": ["docs/api.md:120"]},
        ],
        "slices_with_outdated_types": [
            {"slice_name": "slice_easy", "outdated_type": "abstain_crash", "location": "gates"},
        ],
    }


@pytest.fixture
def minimal_analysis() -> Dict[str, Any]:
    """Minimal analysis (no changes)."""
    return {
        "old_version": "1.0.0",
        "new_version": "1.0.0",
        "risk_level": "LOW",
        "breaking_changes": [],
        "non_breaking_changes": [],
    }


@pytest.fixture
def breaking_analysis() -> Dict[str, Any]:
    """Analysis with breaking changes."""
    return {
        "old_version": "1.0.0",
        "new_version": "2.0.0",
        "risk_level": "HIGH",
        "breaking_changes": [
            {"change_type": "type_removed", "description": "Type removed"},
        ],
        "non_breaking_changes": [
            {"change_type": "type_added", "description": "Type added"},
        ],
    }


@pytest.fixture
def minimal_panel() -> Dict[str, Any]:
    """Minimal director panel (all green)."""
    return {
        "status_light": "GREEN",
        "risk_level": "LOW",
        "alignment_status": "ALIGNED",
        "headline": "Taxonomy stable and aligned",
        "requires_ack": False,
        "affected_metric_kinds": [],
        "missing_doc_updates": 0,
    }


@pytest.fixture
def red_panel() -> Dict[str, Any]:
    """Red status director panel."""
    return {
        "status_light": "RED",
        "risk_level": "HIGH",
        "alignment_status": "OUT_OF_DATE",
        "headline": "Risk: HIGH | Metrics: MISALIGNED | Alignment: OUT_OF_DATE",
        "requires_ack": True,
        "affected_metric_kinds": ["abstention_rate"],
        "missing_doc_updates": 2,
    }


# ---------------------------------------------------------------------------
# Task 1: Runbook Tests
# ---------------------------------------------------------------------------

class TestBuildTaxonomyChangeRunbook:
    """Tests for build_taxonomy_change_runbook()."""
    
    def test_runbook_has_schema_version(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Runbook must include schema_version."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        assert "schema_version" in runbook
        assert runbook["schema_version"] == "1.0.0"
    
    def test_runbook_has_steps(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Runbook must include ordered steps."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        assert "steps" in runbook
        assert isinstance(runbook["steps"], list)
        assert len(runbook["steps"]) > 0
    
    def test_steps_are_ordered(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Steps must be in order (consecutive ordering not required, just sorted)."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        orders = [step["order"] for step in runbook["steps"]]
        assert orders == sorted(orders), f"Steps not in order: {orders}"
        assert len(set(orders)) == len(orders), "Duplicate order values found"
    
    def test_steps_have_required_fields(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Each step must have id, description, status, order."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        for step in runbook["steps"]:
            assert "id" in step
            assert "description" in step
            assert "status" in step
            assert "order" in step
            assert step["status"] == "PENDING"  # All start as pending
    
    def test_first_step_is_update_code(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """First step should be updating taxonomy code."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        first_step = runbook["steps"][0]
        assert first_step["id"] == "update_taxonomy_code"
        assert first_step["order"] == 1
    
    def test_second_step_is_bump_version(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Second step should be bumping version."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        second_step = runbook["steps"][1]
        assert second_step["id"] == "bump_version"
        assert second_step["order"] == 2
    
    def test_blocking_items_when_metrics_affected(self, impacted_metrics, minimal_impact_docs) -> None:
        """Blocking items should include affected metrics."""
        runbook = build_taxonomy_change_runbook(impacted_metrics, minimal_impact_docs)
        
        assert len(runbook["blocking_items"]) > 0
        assert any("abstention_rate" in item for item in runbook["blocking_items"])
    
    def test_blocking_items_when_docs_affected(self, minimal_impact_metrics, impacted_docs) -> None:
        """Blocking items should include affected docs."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, impacted_docs)
        
        assert len(runbook["blocking_items"]) > 0
        assert any("Documentation" in item for item in runbook["blocking_items"])
        assert any("Curriculum" in item for item in runbook["blocking_items"])
    
    def test_advisory_items_for_non_covered_categories(self, impacted_metrics, minimal_impact_docs) -> None:
        """Advisory items should include non-covered categories."""
        runbook = build_taxonomy_change_runbook(impacted_metrics, minimal_impact_docs)
        
        assert len(runbook["advisory_items"]) > 0
        assert any("timeout_related" in item for item in runbook["advisory_items"])
    
    def test_runbook_includes_update_metrics_step_when_affected(self, impacted_metrics, minimal_impact_docs) -> None:
        """Runbook should include metrics update step when metrics are affected."""
        runbook = build_taxonomy_change_runbook(impacted_metrics, minimal_impact_docs)
        
        metrics_step = next((s for s in runbook["steps"] if s["id"] == "update_metrics_config"), None)
        assert metrics_step is not None
        assert "abstention_rate" in metrics_step["description"]
    
    def test_runbook_includes_update_docs_step_when_affected(self, minimal_impact_metrics, impacted_docs) -> None:
        """Runbook should include docs update step when docs are affected."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, impacted_docs)
        
        docs_step = next((s for s in runbook["steps"] if s["id"] == "update_documentation"), None)
        assert docs_step is not None
        assert "documentation" in docs_step["description"].lower()
    
    def test_runbook_includes_update_curriculum_step_when_affected(self, minimal_impact_metrics, impacted_docs) -> None:
        """Runbook should include curriculum update step when slices are affected."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, impacted_docs)
        
        curriculum_step = next((s for s in runbook["steps"] if s["id"] == "update_curriculum_slices"), None)
        assert curriculum_step is not None
        assert "slice_easy" in curriculum_step["description"]
    
    def test_runbook_is_deterministic(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Runbook should be deterministic (same input = same output)."""
        runbook1 = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        runbook2 = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        # Compare structure (excluding any timestamps)
        assert runbook1["schema_version"] == runbook2["schema_version"]
        assert len(runbook1["steps"]) == len(runbook2["steps"])
        assert runbook1["blocking_items"] == runbook2["blocking_items"]
        assert runbook1["advisory_items"] == runbook2["advisory_items"]
        
        # Compare step IDs and orders
        step_ids1 = [(s["id"], s["order"]) for s in runbook1["steps"]]
        step_ids2 = [(s["id"], s["order"]) for s in runbook2["steps"]]
        assert step_ids1 == step_ids2


# ---------------------------------------------------------------------------
# Task 2: Evidence Capsule Tests
# ---------------------------------------------------------------------------

class TestSummarizeTaxonomyChangeForEvidence:
    """Tests for summarize_taxonomy_change_for_evidence()."""
    
    def test_evidence_tile_has_change_magnitude(self, minimal_analysis, minimal_panel) -> None:
        """Evidence tile must include change_magnitude."""
        tile = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        
        assert "change_magnitude" in tile
        assert tile["change_magnitude"] in {"LOW", "MEDIUM", "HIGH"}
    
    def test_change_magnitude_low_for_no_changes(self, minimal_analysis, minimal_panel) -> None:
        """No changes should result in LOW magnitude."""
        tile = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        
        assert tile["change_magnitude"] == "LOW"
    
    def test_change_magnitude_high_for_high_risk(self, breaking_analysis, red_panel) -> None:
        """HIGH risk should result in HIGH magnitude."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        assert tile["change_magnitude"] == "HIGH"
    
    def test_evidence_tile_has_alignment_status(self, minimal_analysis, minimal_panel) -> None:
        """Evidence tile must include alignment_status."""
        tile = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        
        assert "alignment_status" in tile
        assert tile["alignment_status"] in {"ALIGNED", "PARTIAL", "OUT_OF_DATE"}
    
    def test_evidence_tile_has_metrics_impacted(self, breaking_analysis, red_panel) -> None:
        """Evidence tile must include metrics_impacted count."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        assert "metrics_impacted" in tile
        assert isinstance(tile["metrics_impacted"], int)
        assert tile["metrics_impacted"] >= 0
    
    def test_evidence_tile_has_docs_impacted(self, breaking_analysis, red_panel) -> None:
        """Evidence tile must include docs_impacted count."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        assert "docs_impacted" in tile
        assert isinstance(tile["docs_impacted"], int)
        assert tile["docs_impacted"] >= 0
    
    def test_evidence_tile_has_requires_ack(self, minimal_analysis, minimal_panel) -> None:
        """Evidence tile must include requires_ack."""
        tile = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        
        assert "requires_ack" in tile
        assert isinstance(tile["requires_ack"], bool)
    
    def test_evidence_tile_has_version_info(self, breaking_analysis, red_panel) -> None:
        """Evidence tile must include version_from and version_to."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        assert "version_from" in tile
        assert "version_to" in tile
        assert tile["version_from"] == "1.0.0"
        assert tile["version_to"] == "2.0.0"
    
    def test_evidence_tile_has_change_counts(self, breaking_analysis, red_panel) -> None:
        """Evidence tile must include breaking and non-breaking change counts."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        assert "breaking_changes_count" in tile
        assert "non_breaking_changes_count" in tile
        assert tile["breaking_changes_count"] == 1
        assert tile["non_breaking_changes_count"] == 1
    
    def test_evidence_tile_is_deterministic(self, minimal_analysis, minimal_panel) -> None:
        """Evidence tile should be deterministic."""
        tile1 = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        tile2 = summarize_taxonomy_change_for_evidence(minimal_analysis, minimal_panel)
        
        assert tile1 == tile2
    
    def test_evidence_tile_no_normative_language(self, breaking_analysis, red_panel) -> None:
        """Evidence tile should use descriptive language only (no 'should', 'must', etc.)."""
        tile = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        # Convert to string and check for normative words
        tile_str = json.dumps(tile).lower()
        normative_words = ["should", "must", "need", "ought"]
        
        # The tile itself shouldn't contain normative language in values
        # (descriptions are in runbook, not evidence tile)
        # Field names like "requires_ack" are allowed
        for word in normative_words:
            # Check that normative words aren't in the actual values
            # Split by colons to separate keys from values
            parts = tile_str.split(":")
            for part in parts[1:]:  # Skip keys, check values
                assert word not in part, f"Normative word '{word}' found in value: {part}"
    
    def test_change_magnitude_medium_for_partial_alignment(self) -> None:
        """PARTIAL alignment should result in MEDIUM magnitude."""
        analysis = {
            "old_version": "1.0.0",
            "new_version": "1.1.0",
            "risk_level": "MEDIUM",
            "breaking_changes": [],
            "non_breaking_changes": [],
        }
        panel = {
            "status_light": "YELLOW",
            "risk_level": "MEDIUM",
            "alignment_status": "PARTIAL",
            "headline": "Partial alignment",
            "requires_ack": False,
            "affected_metric_kinds": [],
            "missing_doc_updates": 1,
        }
        
        tile = summarize_taxonomy_change_for_evidence(analysis, panel)
        
        assert tile["change_magnitude"] == "MEDIUM"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestRunbookAndEvidenceIntegration:
    """Integration tests for runbook and evidence capsule."""
    
    def test_runbook_and_evidence_work_together(self, impacted_metrics, impacted_docs, breaking_analysis, red_panel) -> None:
        """Runbook and evidence tile should work together."""
        runbook = build_taxonomy_change_runbook(impacted_metrics, impacted_docs)
        evidence = summarize_taxonomy_change_for_evidence(breaking_analysis, red_panel)
        
        # Both should be valid
        assert runbook["schema_version"] == "1.0.0"
        assert evidence["change_magnitude"] in {"LOW", "MEDIUM", "HIGH"}
        
        # Runbook blocking items should align with evidence
        if evidence["change_magnitude"] == "HIGH":
            assert len(runbook["blocking_items"]) > 0
    
    def test_minimal_changes_produce_minimal_runbook(self, minimal_impact_metrics, minimal_impact_docs) -> None:
        """Minimal changes should produce a runbook with no blocking items."""
        runbook = build_taxonomy_change_runbook(minimal_impact_metrics, minimal_impact_docs)
        
        assert runbook["blocking_count"] == 0
        assert len(runbook["blocking_items"]) == 0
    
    def test_breaking_changes_produce_blocking_runbook(self, impacted_metrics, impacted_docs) -> None:
        """Breaking changes should produce blocking items."""
        runbook = build_taxonomy_change_runbook(impacted_metrics, impacted_docs)
        
        assert runbook["blocking_count"] > 0
        assert len(runbook["blocking_items"]) > 0

