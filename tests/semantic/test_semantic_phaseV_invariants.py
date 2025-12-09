"""
Tests for Phase V: Semantic Integrity Grid v2.

Tests semantic invariant checking, uplift pre-gate preview, and director tile.
"""

import unittest
from typing import Dict, Any

from experiments.semantic_consistency_audit import (
    # Phase IV components
    SemanticKnowledgeGraph,
    GraphEdge,
    build_semantic_alignment_index,
    analyze_semantic_risk,
    build_semantic_governance_snapshot,
    audit_semantic_contract,
    forecast_semantic_drift,
    # Phase V components
    InvariantStatus,
    check_semantic_invariants,
    UpliftSemanticStatus,
    preview_semantic_uplift_gate,
    build_semantic_uplift_director_tile,
    StatusLight,
)


# =============================================================================
# SEMANTIC INVARIANT CHECKER TESTS (Phase V - Task 1)
# =============================================================================

class TestInvariantStatus(unittest.TestCase):
    """Tests for InvariantStatus enum."""
    
    def test_invariant_status_values(self):
        """InvariantStatus should have OK, ATTENTION, BROKEN."""
        self.assertEqual(InvariantStatus.OK.value, "OK")
        self.assertEqual(InvariantStatus.ATTENTION.value, "ATTENTION")
        self.assertEqual(InvariantStatus.BROKEN.value, "BROKEN")


class TestCheckSemanticInvariants(unittest.TestCase):
    """Tests for check_semantic_invariants function."""
    
    def test_invariants_ok_when_all_satisfied(self):
        """Should return OK when all invariants are satisfied."""
        # Create graph with connected nodes
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
            ],
            edges=[
                GraphEdge(src="goal_hit", dst="density", weight=1.0, kind="cooccur"),
            ],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        
        taxonomy = {"terms": ["goal_hit", "density"]}
        curriculum_manifest = {"terms": ["goal_hit", "density"]}
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy=taxonomy,
            curriculum_manifest=curriculum_manifest,
        )
        
        self.assertEqual(result["invariant_status"], "OK")
        self.assertEqual(len(result["broken_invariants"]), 0)
    
    def test_invariant_curriculum_term_insufficient_appearances(self):
        """Should detect curriculum term appearing in <2 systems."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "goal_hit"}],
            edges=[
                GraphEdge(src="goal_hit", dst="other_term", weight=1.0, kind="cooccur"),
            ],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": ["orphan_term"],
            "taxonomy_terms_with_no_uses": [],
        }
        
        taxonomy = {"terms": []}  # orphan_term not in taxonomy
        curriculum_manifest = {"terms": ["orphan_term"]}
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy=taxonomy,
            curriculum_manifest=curriculum_manifest,
        )
        
        self.assertEqual(result["invariant_status"], "BROKEN")
        self.assertEqual(len(result["broken_invariants"]), 1)
        self.assertEqual(result["broken_invariants"][0]["invariant_type"], "curriculum_term_insufficient_appearances")
        self.assertIn("orphan_term", result["terms_involved"])
    
    def test_invariant_curriculum_term_appears_in_2_systems(self):
        """Should pass when curriculum term appears in ≥2 systems."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "goal_hit"}],
            edges=[
                GraphEdge(src="goal_hit", dst="other_term", weight=1.0, kind="cooccur"),
            ],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        
        taxonomy = {"terms": ["goal_hit"]}  # In taxonomy
        curriculum_manifest = {"terms": ["goal_hit"]}  # In curriculum and graph
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy=taxonomy,
            curriculum_manifest=curriculum_manifest,
        )
        
        # goal_hit appears in taxonomy (1) and graph (1) = 2, should pass
        self.assertEqual(result["invariant_status"], "OK")
    
    def test_invariant_taxonomy_term_unused_too_long(self):
        """Should detect taxonomy term unused for >N versions."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": ["unused_term"],
        }
        
        taxonomy = {
            "terms": ["unused_term"],
            "version_history": {
                "unused_term": {
                    "unused_versions": ["v1", "v2", "v3", "v4"],  # 4 versions > 3
                }
            }
        }
        
        result = check_semantic_invariants(
            alignment_index,
            taxonomy=taxonomy,
            max_unused_versions=3,
        )
        
        self.assertEqual(result["invariant_status"], "BROKEN")
        broken_inv = result["broken_invariants"][0]
        self.assertEqual(broken_inv["invariant_type"], "taxonomy_term_unused_too_long")
        self.assertIn("unused_term", result["terms_involved"])
    
    def test_invariant_taxonomy_term_unused_acceptable(self):
        """Should pass when taxonomy term unused for ≤N versions."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": ["unused_term"],
        }
        
        taxonomy = {
            "terms": ["unused_term"],
            "version_history": {
                "unused_term": {
                    "unused_versions": ["v1", "v2"],  # 2 versions ≤ 3
                }
            }
        }
        
        result = check_semantic_invariants(
            alignment_index,
            taxonomy=taxonomy,
            max_unused_versions=3,
        )
        
        # Should be ATTENTION (term unused but within limit)
        self.assertIn(result["invariant_status"], ["OK", "ATTENTION"])
    
    def test_invariant_isolated_graph_node(self):
        """Should detect isolated graph node (degree 0)."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "isolated_term"},
                {"canonical_form": "connected_term"},
            ],
            edges=[
                GraphEdge(src="connected_term", dst="other_term", weight=1.0, kind="cooccur"),
            ],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
        )
        
        self.assertEqual(result["invariant_status"], "ATTENTION")
        broken_inv = result["broken_invariants"][0]
        self.assertEqual(broken_inv["invariant_type"], "isolated_graph_node")
        self.assertIn("isolated_term", result["terms_involved"])
    
    def test_invariant_connected_graph_node(self):
        """Should pass when graph node has degree >0."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "connected_term"},
            ],
            edges=[
                GraphEdge(src="connected_term", dst="other_term", weight=1.0, kind="cooccur"),
            ],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
        )
        
        # connected_term has degree 1, should pass
        self.assertEqual(result["invariant_status"], "OK")
    
    def test_invariant_multiple_violations(self):
        """Should detect multiple invariant violations."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "isolated_term"},
            ],
            edges=[],  # No edges = isolated
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": ["orphan_term"],
            "taxonomy_terms_with_no_uses": ["unused_term"],
        }
        
        taxonomy = {"terms": []}
        curriculum_manifest = {"terms": ["orphan_term"]}
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy=taxonomy,
            curriculum_manifest=curriculum_manifest,
        )
        
        self.assertGreater(len(result["broken_invariants"]), 1)
        self.assertIn("orphan_term", result["terms_involved"])
        self.assertIn("isolated_term", result["terms_involved"])
    
    def test_invariant_neutral_notes_generated(self):
        """Should generate neutral explanatory notes."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "isolated_term"}],
            edges=[],
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
        )
        
        self.assertGreater(len(result["neutral_notes"]), 0)
        self.assertIsInstance(result["neutral_notes"], list)


# =============================================================================
# SEMANTIC UPLIFT PRE-GATE PREVIEW TESTS (Phase V - Task 2)
# =============================================================================

class TestUpliftSemanticStatus(unittest.TestCase):
    """Tests for UpliftSemanticStatus enum."""
    
    def test_uplift_semantic_status_values(self):
        """UpliftSemanticStatus should have OK, WARN, BLOCK."""
        self.assertEqual(UpliftSemanticStatus.OK.value, "OK")
        self.assertEqual(UpliftSemanticStatus.WARN.value, "WARN")
        self.assertEqual(UpliftSemanticStatus.BLOCK.value, "BLOCK")


class TestPreviewSemanticUpliftGate(unittest.TestCase):
    """Tests for preview_semantic_uplift_gate function."""
    
    def test_uplift_ok_when_all_checks_pass(self):
        """Should return OK when all semantic checks pass."""
        semantic_risk = {"status": "OK"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "OK")
        self.assertIn("passed", " ".join(preview["rationale"]).lower())
    
    def test_uplift_block_on_critical_risk(self):
        """Should BLOCK when semantic risk is CRITICAL."""
        semantic_risk = {"status": "CRITICAL"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "BLOCK")
        self.assertIn("CRITICAL", " ".join(preview["rationale"]).upper())
    
    def test_uplift_block_on_contract_breach(self):
        """Should BLOCK when contract status is BREACH."""
        semantic_risk = {"status": "OK"}
        contract_audit = {"contract_status": "BREACH"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "BLOCK")
        self.assertIn("BREACH", " ".join(preview["rationale"]).upper())
    
    def test_uplift_block_on_broken_invariants(self):
        """Should BLOCK when invariant status is BROKEN."""
        semantic_risk = {"status": "OK"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "BROKEN"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "BLOCK")
        self.assertIn("invariant", " ".join(preview["rationale"]).lower())
    
    def test_uplift_warn_on_attention_risk(self):
        """Should WARN when semantic risk is ATTENTION."""
        semantic_risk = {"status": "ATTENTION"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "WARN")
        self.assertIn("ATTENTION", " ".join(preview["rationale"]).upper())
    
    def test_uplift_warn_on_degrading_drift(self):
        """Should WARN when drift direction is DEGRADING."""
        semantic_risk = {"status": "OK"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "DEGRADING"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertEqual(preview["uplift_semantic_status"], "WARN")
        self.assertIn("degrading", " ".join(preview["rationale"]).lower())
    
    def test_uplift_rationale_includes_all_conditions(self):
        """Rationale should include explanations for all conditions."""
        semantic_risk = {"status": "ATTENTION"}
        contract_audit = {"contract_status": "ATTENTION"}
        drift_forecast = {"drift_direction": "DEGRADING"}
        invariant_check = {"invariant_status": "ATTENTION"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertGreater(len(preview["rationale"]), 1)
        rationale_text = " ".join(preview["rationale"]).lower()
        self.assertTrue("attention" in rationale_text or "warning" in rationale_text)
    
    def test_uplift_preview_effect_on_curriculum(self):
        """Should include preview effects on curriculum."""
        semantic_risk = {"status": "CRITICAL"}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        
        preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        self.assertIn("preview_effect_on_curriculum", preview)
        self.assertGreater(len(preview["preview_effect_on_curriculum"]), 0)
        self.assertIsInstance(preview["preview_effect_on_curriculum"], list)


# =============================================================================
# DIRECTOR TILE TESTS (Phase V - Task 3)
# =============================================================================

class TestBuildSemanticUpliftDirectorTile(unittest.TestCase):
    """Tests for build_semantic_uplift_director_tile function."""
    
    def test_tile_red_on_block(self):
        """Should show RED status light when uplift is BLOCKED."""
        semantic_risk = {"status": "CRITICAL", "high_risk_terms": ["term1"], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "BLOCK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertEqual(tile["status_light"], "RED")
        self.assertEqual(tile["semantic_uplift_status"], "BLOCK")
    
    def test_tile_yellow_on_warn(self):
        """Should show YELLOW status light when uplift is WARN."""
        semantic_risk = {"status": "ATTENTION", "high_risk_terms": [], "medium_risk_terms": ["term1"]}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "DEGRADING"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "WARN"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertEqual(tile["status_light"], "YELLOW")
        self.assertEqual(tile["semantic_uplift_status"], "WARN")
    
    def test_tile_green_on_ok(self):
        """Should show GREEN status light when uplift is OK."""
        semantic_risk = {"status": "OK", "high_risk_terms": [], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "OK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertEqual(tile["status_light"], "GREEN")
        self.assertEqual(tile["semantic_uplift_status"], "OK")
    
    def test_tile_top_risk_terms(self):
        """Should include top risk terms (high-risk first, then medium-risk)."""
        semantic_risk = {
            "status": "CRITICAL",
            "high_risk_terms": ["high1", "high2", "high3", "high4", "high5", "high6"],
            "medium_risk_terms": ["med1", "med2", "med3", "med4"],
        }
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "BLOCK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertIn("top_risk_terms", tile)
        self.assertGreater(len(tile["top_risk_terms"]), 0)
        # Should include high-risk terms first
        self.assertIn("high1", tile["top_risk_terms"])
        # Should limit to top 5 high-risk + top 3 medium-risk
        self.assertLessEqual(len(tile["top_risk_terms"]), 8)
    
    def test_tile_headline_block(self):
        """Headline should reflect BLOCK status."""
        semantic_risk = {"status": "CRITICAL", "high_risk_terms": ["term1"], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "BLOCK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertIn("headline", tile)
        headline_lower = tile["headline"].lower()
        self.assertTrue("block" in headline_lower or "blocked" in headline_lower)
    
    def test_tile_headline_warn(self):
        """Headline should reflect WARN status."""
        semantic_risk = {"status": "OK", "high_risk_terms": [], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "DEGRADING"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "WARN"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        headline_lower = tile["headline"].lower()
        self.assertTrue("warn" in headline_lower or "warning" in headline_lower or "review" in headline_lower)
    
    def test_tile_headline_ok(self):
        """Headline should reflect OK status."""
        semantic_risk = {"status": "OK", "high_risk_terms": [], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "OK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        headline_lower = tile["headline"].lower()
        self.assertTrue("passed" in headline_lower or "ok" in headline_lower or "check" in headline_lower)
    
    def test_tile_includes_all_fields(self):
        """Tile should include all required fields."""
        semantic_risk = {"status": "OK", "high_risk_terms": [], "medium_risk_terms": []}
        contract_audit = {"contract_status": "OK"}
        drift_forecast = {"drift_direction": "STABLE"}
        invariant_check = {"invariant_status": "OK"}
        uplift_preview = {"uplift_semantic_status": "OK"}
        
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        self.assertIn("status_light", tile)
        self.assertIn("semantic_uplift_status", tile)
        self.assertIn("top_risk_terms", tile)
        self.assertIn("headline", tile)


# =============================================================================
# PHASE V INTEGRATION TESTS
# =============================================================================

class TestPhaseVIntegration(unittest.TestCase):
    """Integration tests for Phase V workflow."""
    
    def test_full_phase_v_workflow(self):
        """Test complete Phase V workflow: invariants → preview → tile."""
        # Build graph
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
            ],
            edges=[
                GraphEdge(src="goal_hit", dst="density", weight=1.0, kind="cooccur"),
            ],
        )
        
        # Build alignment index
        alignment_index = build_semantic_alignment_index(
            {"status": "OK"},
            {"terms": ["goal_hit", "density"]},
            {"terms": ["goal_hit", "density"]},
            {"terms": ["goal_hit", "density"]},
            graph=graph,
        )
        
        # Build governance snapshot
        governance_snapshot = build_semantic_governance_snapshot(graph)
        
        # Analyze risk
        semantic_risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        # Audit contract
        contract_audit = audit_semantic_contract(
            alignment_index,
            {"terms": ["goal_hit", "density"]},
            {"terms": ["goal_hit", "density"]},
        )
        
        # Forecast drift
        drift_forecast = forecast_semantic_drift([
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
        ])
        
        # Check invariants
        invariant_check = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy={"terms": ["goal_hit", "density"]},
            curriculum_manifest={"terms": ["goal_hit", "density"]},
        )
        
        # Preview uplift gate
        uplift_preview = preview_semantic_uplift_gate(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
        )
        
        # Build director tile
        tile = build_semantic_uplift_director_tile(
            semantic_risk,
            contract_audit,
            drift_forecast,
            invariant_check,
            uplift_preview,
        )
        
        # Verify workflow produces valid outputs
        self.assertIn("invariant_status", invariant_check)
        self.assertIn("uplift_semantic_status", uplift_preview)
        self.assertIn("status_light", tile)
        self.assertIn("headline", tile)
    
    def test_invariant_checker_with_real_data(self):
        """Test invariant checker with realistic data structures."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
            ],
            edges=[],  # Isolated node
        )
        
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": ["orphan_term"],
            "taxonomy_terms_with_no_uses": ["unused_term"],
        }
        
        taxonomy = {
            "terms": ["unused_term"],
            "version_history": {
                "unused_term": {
                    "unused_versions": ["v1", "v2", "v3", "v4"],
                }
            }
        }
        
        curriculum_manifest = {"terms": ["orphan_term"]}
        
        result = check_semantic_invariants(
            alignment_index,
            graph=graph,
            taxonomy=taxonomy,
            curriculum_manifest=curriculum_manifest,
        )
        
        # Should detect multiple violations
        self.assertGreater(len(result["broken_invariants"]), 1)
        self.assertEqual(result["invariant_status"], "BROKEN")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

