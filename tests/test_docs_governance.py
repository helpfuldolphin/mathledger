"""
Tests for docs governance layer.

Validates:
- Governance snapshot building from lint outputs
- Uplift safety evaluation logic
- Evidence pack section generation
- Deterministic JSON output
"""

import pytest
import json
from pathlib import Path

from docs.docs_governance import (
    build_docs_governance_snapshot,
    evaluate_uplift_safety,
    build_docs_section_for_evidence_pack
)


class TestBuildDocsGovernanceSnapshot:
    """Test build_docs_governance_snapshot function."""
    
    def test_empty_reports(self):
        """Test with empty input reports."""
        snapshot = build_docs_governance_snapshot({}, {}, {})
        
        assert snapshot["schema_version"] == "1.0.0"
        assert snapshot["doc_count"] == 0
        assert snapshot["docs_with_invalid_snippets"] == []
        assert snapshot["docs_missing_phase_markers"] == []
        assert snapshot["docs_with_uplift_mentions_without_disclaimer"] == []
        assert snapshot["evidence_docs_covered"] == 0
    
    def test_snippet_report_processing(self):
        """Test processing of snippet validation report."""
        snippet_report = {
            "invalid_files": ["docs/U2_PORT_PLAN.md", "docs/PHASE2_RFL_UPLIFT_PLAN.md"]
        }
        
        snapshot = build_docs_governance_snapshot(snippet_report, {}, {})
        
        assert snapshot["docs_with_invalid_snippets"] == [
            "docs/PHASE2_RFL_UPLIFT_PLAN.md",
            "docs/U2_PORT_PLAN.md"
        ]
        # Should be sorted
        assert snapshot["docs_with_invalid_snippets"][0] < snapshot["docs_with_invalid_snippets"][1]
    
    def test_phase_marker_report_processing(self):
        """Test processing of phase marker report."""
        phase_marker_report = {
            "docs_missing_markers": ["docs/VSD_PHASE_2.md"],
            "docs_with_uplift_mentions_without_disclaimer": [
                "docs/U2_PORT_PLAN.md"
            ]
        }
        
        snapshot = build_docs_governance_snapshot({}, phase_marker_report, {})
        
        assert snapshot["docs_missing_phase_markers"] == ["docs/VSD_PHASE_2.md"]
        assert snapshot["docs_with_uplift_mentions_without_disclaimer"] == [
            "docs/U2_PORT_PLAN.md"
        ]
    
    def test_toc_index_processing(self, tmp_path):
        """Test processing of TOC index with evidence docs."""
        # Create some test docs
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Doc 1")
        doc2.write_text("# Doc 2")
        
        toc_index = {
            "evidence_docs": [
                str(doc1),
                str(doc2),
                str(tmp_path / "missing.md")  # Doesn't exist
            ]
        }
        
        snapshot = build_docs_governance_snapshot({}, {}, toc_index)
        
        # Only 2 docs exist on disk
        assert snapshot["evidence_docs_covered"] == 2
    
    def test_doc_count_aggregation(self):
        """Test that doc_count aggregates from all sources."""
        snippet_report = {
            "invalid_files": ["docs/A.md", "docs/B.md"]
        }
        phase_marker_report = {
            "docs_missing_markers": ["docs/B.md", "docs/C.md"],
            "docs_with_uplift_mentions_without_disclaimer": ["docs/D.md"]
        }
        toc_index = {
            "evidence_docs": ["docs/E.md", "docs/A.md"]  # A.md overlaps
        }
        
        snapshot = build_docs_governance_snapshot(
            snippet_report, phase_marker_report, toc_index
        )
        
        # Unique docs: A, B, C, D, E = 5
        assert snapshot["doc_count"] == 5
    
    def test_deterministic_sorting(self):
        """Test that output is deterministically sorted."""
        snippet_report = {
            "invalid_files": ["docs/Z.md", "docs/A.md", "docs/M.md"]
        }
        
        snapshot1 = build_docs_governance_snapshot(snippet_report, {}, {})
        snapshot2 = build_docs_governance_snapshot(snippet_report, {}, {})
        
        assert snapshot1["docs_with_invalid_snippets"] == snapshot2["docs_with_invalid_snippets"]
        assert snapshot1["docs_with_invalid_snippets"] == ["docs/A.md", "docs/M.md", "docs/Z.md"]


class TestEvaluateUpliftSafety:
    """Test evaluate_uplift_safety function."""
    
    def test_clean_snapshot(self):
        """Test with clean snapshot (no issues)."""
        snapshot = {
            "docs_with_invalid_snippets": [],
            "docs_missing_phase_markers": [],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        assert result["uplift_safe"] is True
        assert result["issues"] == []
        assert result["status"] == "OK"
    
    def test_invalid_snippets_warning(self):
        """Test that invalid snippets generate warnings."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/test.md"],
            "docs_missing_phase_markers": [],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        assert result["uplift_safe"] is True  # Only snippet issues
        assert len(result["issues"]) == 1
        assert "docs/test.md contains invalid code snippets" in result["issues"]
        assert result["status"] == "WARN"
    
    def test_missing_markers_warning(self):
        """Test that missing phase markers generate warnings."""
        snapshot = {
            "docs_with_invalid_snippets": [],
            "docs_missing_phase_markers": ["docs/phase2.md"],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        assert result["uplift_safe"] is True  # Only marker issues
        assert len(result["issues"]) == 1
        assert "docs/phase2.md missing required phase markers" in result["issues"]
        assert result["status"] == "WARN"
    
    def test_uplift_without_disclaimer_blocks(self):
        """Test that uplift mentions without disclaimer block status."""
        snapshot = {
            "docs_with_invalid_snippets": [],
            "docs_missing_phase_markers": [],
            "docs_with_uplift_mentions_without_disclaimer": ["docs/U2_PORT_PLAN.md"]
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        assert result["uplift_safe"] is False
        assert len(result["issues"]) == 1
        assert "docs/U2_PORT_PLAN.md mentions uplift but lacks disclaimer" in result["issues"]
        assert result["status"] == "BLOCK"
    
    def test_multiple_issues_prioritizes_block(self):
        """Test that BLOCK status takes priority over WARN."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/snippets.md"],
            "docs_missing_phase_markers": ["docs/markers.md"],
            "docs_with_uplift_mentions_without_disclaimer": ["docs/uplift.md"]
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        assert result["uplift_safe"] is False
        assert len(result["issues"]) == 3
        assert result["status"] == "BLOCK"
    
    def test_deterministic_issue_ordering(self):
        """Test that issues are deterministically ordered."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/Z.md", "docs/A.md"],
            "docs_missing_phase_markers": ["docs/M.md"],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        
        result1 = evaluate_uplift_safety(snapshot)
        result2 = evaluate_uplift_safety(snapshot)
        
        assert result1["issues"] == result2["issues"]
        # Issues should be sorted
        for i in range(len(result1["issues"]) - 1):
            assert result1["issues"][i] < result1["issues"][i + 1]


class TestBuildDocsSection:
    """Test build_docs_section_for_evidence_pack function."""
    
    def test_clean_evidence_section(self):
        """Test evidence section with no issues."""
        snapshot = {
            "docs_with_invalid_snippets": [],
            "docs_missing_phase_markers": [],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        uplift_safety = {
            "uplift_safe": True,
            "issues": [],
            "status": "OK"
        }
        
        section = build_docs_section_for_evidence_pack(snapshot, uplift_safety)
        
        assert section["docs_governance_ok"] is True
        assert section["docs_with_phase_issues"] == []
        assert section["docs_with_snippet_issues"] == []
        assert section["uplift_safe"] is True
    
    def test_evidence_section_with_issues(self):
        """Test evidence section with various issues."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/bad_snippet.md"],
            "docs_missing_phase_markers": ["docs/no_marker.md"],
            "docs_with_uplift_mentions_without_disclaimer": ["docs/bad_uplift.md"]
        }
        uplift_safety = {
            "uplift_safe": False,
            "issues": ["multiple issues"],
            "status": "BLOCK"
        }
        
        section = build_docs_section_for_evidence_pack(snapshot, uplift_safety)
        
        assert section["docs_governance_ok"] is False
        assert section["docs_with_phase_issues"] == ["docs/no_marker.md"]
        assert section["docs_with_snippet_issues"] == ["docs/bad_snippet.md"]
        assert section["uplift_safe"] is False
    
    def test_deterministic_output(self):
        """Test that evidence section output is deterministic."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/Z.md", "docs/A.md"],
            "docs_missing_phase_markers": ["docs/Y.md", "docs/B.md"],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        uplift_safety = {
            "uplift_safe": True,
            "issues": [],
            "status": "WARN"
        }
        
        section1 = build_docs_section_for_evidence_pack(snapshot, uplift_safety)
        section2 = build_docs_section_for_evidence_pack(snapshot, uplift_safety)
        
        assert section1 == section2
        # Lists should be sorted
        assert section1["docs_with_phase_issues"] == ["docs/B.md", "docs/Y.md"]
        assert section1["docs_with_snippet_issues"] == ["docs/A.md", "docs/Z.md"]


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from reports to evidence section."""
        snippet_report = {
            "invalid_files": ["docs/snippet_issue.md"]
        }
        phase_marker_report = {
            "docs_missing_markers": ["docs/marker_issue.md"],
            "docs_with_uplift_mentions_without_disclaimer": []
        }
        toc_index = {
            "evidence_docs": []
        }
        
        # Build snapshot
        snapshot = build_docs_governance_snapshot(
            snippet_report, phase_marker_report, toc_index
        )
        
        # Evaluate safety
        uplift_safety = evaluate_uplift_safety(snapshot)
        
        # Build evidence section
        evidence_section = build_docs_section_for_evidence_pack(
            snapshot, uplift_safety
        )
        
        # Verify end-to-end
        assert snapshot["doc_count"] == 2
        assert uplift_safety["status"] == "WARN"
        assert uplift_safety["uplift_safe"] is True
        assert evidence_section["docs_governance_ok"] is False
        assert len(evidence_section["docs_with_snippet_issues"]) == 1
        assert len(evidence_section["docs_with_phase_issues"]) == 1
    
    def test_json_serialization(self):
        """Test that all outputs are JSON-serializable."""
        snapshot = build_docs_governance_snapshot({}, {}, {})
        uplift_safety = evaluate_uplift_safety(snapshot)
        evidence_section = build_docs_section_for_evidence_pack(
            snapshot, uplift_safety
        )
        
        # Should not raise
        json.dumps(snapshot)
        json.dumps(uplift_safety)
        json.dumps(evidence_section)
    
    def test_deterministic_json_output(self):
        """Test that JSON output is deterministic."""
        snapshot = build_docs_governance_snapshot({}, {}, {})
        uplift_safety = evaluate_uplift_safety(snapshot)
        
        # Multiple serializations should be identical
        json1 = json.dumps(snapshot, sort_keys=True)
        json2 = json.dumps(snapshot, sort_keys=True)
        assert json1 == json2
        
        json3 = json.dumps(uplift_safety, sort_keys=True)
        json4 = json.dumps(uplift_safety, sort_keys=True)
        assert json3 == json4


class TestNormativeVocabularyAbsence:
    """Verify that uplift safety evaluation uses no normative vocabulary."""
    
    def test_no_normative_terms_in_issues(self):
        """Test that issue descriptions are neutral."""
        snapshot = {
            "docs_with_invalid_snippets": ["docs/test.md"],
            "docs_missing_phase_markers": ["docs/test2.md"],
            "docs_with_uplift_mentions_without_disclaimer": ["docs/test3.md"]
        }
        
        result = evaluate_uplift_safety(snapshot)
        
        # Check that issues don't contain normative terms
        normative_terms = ["bad", "wrong", "incorrect", "error", "must", "should"]
        for issue in result["issues"]:
            lower_issue = issue.lower()
            for term in normative_terms:
                assert term not in lower_issue, f"Normative term '{term}' found in: {issue}"
    
    def test_neutral_status_values(self):
        """Test that status values are neutral descriptors."""
        allowed_statuses = {"OK", "WARN", "BLOCK"}
        
        snapshots = [
            {},
            {"docs_with_invalid_snippets": ["test.md"]},
            {"docs_with_uplift_mentions_without_disclaimer": ["test.md"]}
        ]
        
        for snapshot in snapshots:
            result = evaluate_uplift_safety(snapshot)
            assert result["status"] in allowed_statuses
