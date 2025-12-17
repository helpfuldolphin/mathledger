"""
Tests for semantic integrity adapter.

Tests semantic integrity tile building, drift signal extraction, and global health integration.
"""

import json
import unittest
from typing import Any, Dict

from pathlib import Path
from tempfile import TemporaryDirectory

# Import reusable warning neutrality helpers (single source of truth)
from tests.helpers.warning_neutrality import assert_warning_neutral

from backend.health.semantic_integrity_adapter import (
    SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION,
    SEMANTIC_FOOTPRINT_SCHEMA_VERSION,
    SEMANTIC_SAFETY_PANEL_SCHEMA_VERSION,
    build_semantic_integrity_tile,
    extract_semantic_drift_signal,
    build_semantic_integrity_summary_for_p3,
    build_semantic_integrity_calibration_for_p4,
    build_first_light_semantic_footprint,
    emit_cal_exp_semantic_footprint,
    persist_cal_exp_semantic_footprint,
    build_semantic_safety_panel,
    extract_semantic_safety_panel_signal,
    semantic_safety_panel_for_alignment_view,
    attach_semantic_integrity_to_evidence,
    attach_semantic_safety_panel_to_evidence,
    summarize_semantic_integrity_for_uplift_council,
)


# =============================================================================
# SEMANTIC INTEGRITY TILE TESTS
# =============================================================================

class TestBuildSemanticIntegrityTile(unittest.TestCase):
    """Tests for build_semantic_integrity_tile function."""
    
    def test_tile_has_required_fields(self):
        """Tile should have all required fields."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": ["All semantic invariants are satisfied"],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": ["All semantic checks passed"],
            "preview_effect_on_curriculum": ["Uplift appears safe from semantic perspective"],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "Semantic uplift status: all checks passed",
        }
        
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        
        # Check required fields
        self.assertEqual(tile["schema_version"], SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION)
        self.assertEqual(tile["tile_type"], "semantic_integrity")
        self.assertIn("status_light", tile)
        self.assertIn("invariants_ok", tile)
        self.assertIn("uplift_semantic_status", tile)
        self.assertIn("broken_invariants", tile)
        self.assertIn("top_risk_terms", tile)
        self.assertIn("headline", tile)
        self.assertIn("notes", tile)
    
    def test_tile_deterministic(self):
        """Tile should be deterministic (same inputs produce same outputs)."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": ["All semantic invariants are satisfied"],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": ["All semantic checks passed"],
            "preview_effect_on_curriculum": ["Uplift appears safe"],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "Semantic uplift status: all checks passed",
        }
        
        tile1 = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        tile2 = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        
        self.assertEqual(tile1, tile2)
    
    def test_tile_json_safe(self):
        """Tile should be JSON-serializable."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": ["All semantic invariants are satisfied"],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": ["All semantic checks passed"],
            "preview_effect_on_curriculum": ["Uplift appears safe"],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "Semantic uplift status: all checks passed",
        }
        
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        
        # Should not raise
        json_str = json.dumps(tile)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION)
    
    def test_status_light_mapping(self):
        """Status light should be correctly mapped from director tile."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": [],
            "preview_effect_on_curriculum": [],
        }
        
        # Test GREEN
        director_tile_green = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile_green,
        )
        self.assertEqual(tile["status_light"], "GREEN")
        
        # Test YELLOW
        director_tile_yellow = {
            "status_light": "YELLOW",
            "semantic_uplift_status": "WARN",
            "top_risk_terms": ["term1"],
            "headline": "Review recommended",
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile_yellow,
        )
        self.assertEqual(tile["status_light"], "YELLOW")
        
        # Test RED
        director_tile_red = {
            "status_light": "RED",
            "semantic_uplift_status": "BLOCK",
            "top_risk_terms": ["term1", "term2"],
            "headline": "Uplift blocked",
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile_red,
        )
        self.assertEqual(tile["status_light"], "RED")
    
    def test_invariants_ok_mapping(self):
        """invariants_ok should be True when invariant_status is OK."""
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": [],
            "preview_effect_on_curriculum": [],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        
        # Test OK
        invariant_check_ok = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check_ok,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        self.assertTrue(tile["invariants_ok"])
        
        # Test ATTENTION
        invariant_check_attention = {
            "invariant_status": "ATTENTION",
            "broken_invariants": [{"invariant_type": "isolated_graph_node", "severity": "ATTENTION"}],
            "terms_involved": ["term1"],
            "neutral_notes": ["Some invariants have warnings"],
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check_attention,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        self.assertFalse(tile["invariants_ok"])
        
        # Test BROKEN
        invariant_check_broken = {
            "invariant_status": "BROKEN",
            "broken_invariants": [{"invariant_type": "curriculum_term_insufficient_appearances", "severity": "BROKEN"}],
            "terms_involved": ["term1"],
            "neutral_notes": ["Critical invariants violated"],
        }
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check_broken,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        self.assertFalse(tile["invariants_ok"])
    
    def test_notes_aggregation(self):
        """Notes should aggregate from all sources."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": ["Note 1", "Note 2"],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": ["Rationale 1", "Rationale 2"],
            "preview_effect_on_curriculum": ["Effect 1"],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        
        # Notes should contain all aggregated notes
        self.assertIn("Note 1", tile["notes"])
        self.assertIn("Note 2", tile["notes"])
        self.assertIn("Rationale 1", tile["notes"])
        self.assertIn("Rationale 2", tile["notes"])
        self.assertIn("Effect 1", tile["notes"])
        self.assertEqual(len(tile["notes"]), 5)
    
    def test_neutrality_of_strings(self):
        """All human-readable strings should be neutral (no emotional language)."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": ["All semantic invariants are satisfied"],
        }
        uplift_preview = {
            "uplift_semantic_status": "OK",
            "rationale": ["All semantic checks passed"],
            "preview_effect_on_curriculum": ["Uplift appears safe from semantic perspective"],
        }
        director_tile = {
            "status_light": "GREEN",
            "semantic_uplift_status": "OK",
            "top_risk_terms": [],
            "headline": "Semantic uplift status: all checks passed",
        }
        
        tile = build_semantic_integrity_tile(
            invariant_check=invariant_check,
            uplift_preview=uplift_preview,
            director_tile=director_tile,
        )
        
        # Check headline is neutral
        headline_lower = tile["headline"].lower()
        # Should not contain emotional language
        emotional_words = ["terrible", "awful", "amazing", "fantastic", "horrible", "excellent"]
        for word in emotional_words:
            self.assertNotIn(word, headline_lower)
        
        # Check notes are neutral
        for note in tile["notes"]:
            note_lower = note.lower()
            for word in emotional_words:
                self.assertNotIn(word, note_lower)
    
    def test_validation_missing_keys(self):
        """Should raise ValueError if required keys are missing."""
        # Missing invariant_check keys
        with self.assertRaises(ValueError):
            build_semantic_integrity_tile(
                invariant_check={},
                uplift_preview={"uplift_semantic_status": "OK", "rationale": [], "preview_effect_on_curriculum": []},
                director_tile={"status_light": "GREEN", "semantic_uplift_status": "OK", "top_risk_terms": [], "headline": ""},
            )
        
        # Missing uplift_preview keys
        with self.assertRaises(ValueError):
            build_semantic_integrity_tile(
                invariant_check={"invariant_status": "OK", "broken_invariants": [], "terms_involved": [], "neutral_notes": []},
                uplift_preview={},
                director_tile={"status_light": "GREEN", "semantic_uplift_status": "OK", "top_risk_terms": [], "headline": ""},
            )
        
        # Missing director_tile keys
        with self.assertRaises(ValueError):
            build_semantic_integrity_tile(
                invariant_check={"invariant_status": "OK", "broken_invariants": [], "terms_involved": [], "neutral_notes": []},
                uplift_preview={"uplift_semantic_status": "OK", "rationale": [], "preview_effect_on_curriculum": []},
                director_tile={},
            )


# =============================================================================
# DRIFT SIGNAL EXTRACTION TESTS
# =============================================================================

class TestExtractSemanticDriftSignal(unittest.TestCase):
    """Tests for extract_semantic_drift_signal function."""
    
    def test_signal_has_required_fields(self):
        """Signal should have all required fields."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        
        signal = extract_semantic_drift_signal(invariant_check)
        
        self.assertIn("drift_detected", signal)
        self.assertIn("severity", signal)
        self.assertIn("broken_invariant_count", signal)
        self.assertIn("terms_involved", signal)
        self.assertIn("critical_signals", signal)
    
    def test_drift_detected_ok(self):
        """drift_detected should be False when status is OK."""
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        
        signal = extract_semantic_drift_signal(invariant_check)
        
        self.assertFalse(signal["drift_detected"])
        self.assertEqual(signal["severity"], "OK")
        self.assertEqual(signal["broken_invariant_count"], 0)
    
    def test_drift_detected_attention(self):
        """drift_detected should be True when status is ATTENTION."""
        invariant_check = {
            "invariant_status": "ATTENTION",
            "broken_invariants": [
                {"invariant_type": "isolated_graph_node", "severity": "ATTENTION", "terms_involved": ["term1"]},
            ],
            "terms_involved": ["term1"],
            "neutral_notes": [],
        }
        
        signal = extract_semantic_drift_signal(invariant_check)
        
        self.assertTrue(signal["drift_detected"])
        self.assertEqual(signal["severity"], "ATTENTION")
        self.assertEqual(signal["broken_invariant_count"], 1)
        self.assertEqual(len(signal["critical_signals"]), 0)  # No BROKEN severity
    
    def test_drift_detected_broken(self):
        """drift_detected should be True when status is BROKEN."""
        invariant_check = {
            "invariant_status": "BROKEN",
            "broken_invariants": [
                {"invariant_type": "curriculum_term_insufficient_appearances", "severity": "BROKEN", "terms_involved": ["term1"]},
                {"invariant_type": "isolated_graph_node", "severity": "ATTENTION", "terms_involved": ["term2"]},
            ],
            "terms_involved": ["term1", "term2"],
            "neutral_notes": [],
        }
        
        signal = extract_semantic_drift_signal(invariant_check)
        
        self.assertTrue(signal["drift_detected"])
        self.assertEqual(signal["severity"], "BROKEN")
        self.assertEqual(signal["broken_invariant_count"], 2)
        self.assertEqual(len(signal["critical_signals"]), 1)
        self.assertEqual(signal["critical_signals"][0], "curriculum_term_insufficient_appearances")
    
    def test_critical_signals_extraction(self):
        """critical_signals should only include BROKEN severity invariants."""
        invariant_check = {
            "invariant_status": "BROKEN",
            "broken_invariants": [
                {"invariant_type": "type1", "severity": "BROKEN", "terms_involved": ["term1"]},
                {"invariant_type": "type2", "severity": "ATTENTION", "terms_involved": ["term2"]},
                {"invariant_type": "type3", "severity": "BROKEN", "terms_involved": ["term3"]},
            ],
            "terms_involved": ["term1", "term2", "term3"],
            "neutral_notes": [],
        }
        
        signal = extract_semantic_drift_signal(invariant_check)
        
        self.assertEqual(len(signal["critical_signals"]), 2)
        self.assertIn("type1", signal["critical_signals"])
        self.assertIn("type3", signal["critical_signals"])
        self.assertNotIn("type2", signal["critical_signals"])
    
    def test_validation_missing_keys(self):
        """Should raise ValueError if required keys are missing."""
        with self.assertRaises(ValueError):
            extract_semantic_drift_signal({})


# =============================================================================
# P3 BINDING TESTS
# =============================================================================

class TestBuildSemanticIntegritySummaryForP3(unittest.TestCase):
    """Tests for build_semantic_integrity_summary_for_p3 function."""
    
    def test_summary_has_required_fields(self):
        """Summary should have all required fields."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        
        summary = build_semantic_integrity_summary_for_p3(tile, invariant_check)
        
        self.assertIn("invariants_ok", summary)
        self.assertIn("broken_invariant_count", summary)
        self.assertIn("uplift_semantic_status", summary)
        self.assertIn("top_risk_terms", summary)
        self.assertIn("headline", summary)
    
    def test_summary_deterministic(self):
        """Summary should be deterministic."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "terms_involved": [],
            "neutral_notes": [],
        }
        
        summary1 = build_semantic_integrity_summary_for_p3(tile, invariant_check)
        summary2 = build_semantic_integrity_summary_for_p3(tile, invariant_check)
        
        self.assertEqual(summary1, summary2)
    
    def test_broken_invariant_count(self):
        """Should correctly count broken invariants."""
        tile = {
            "invariants_ok": False,
            "uplift_semantic_status": "WARN",
            "top_risk_terms": ["term1"],
            "headline": "Some issues detected",
        }
        invariant_check = {
            "invariant_status": "ATTENTION",
            "broken_invariants": [
                {"invariant_type": "type1", "severity": "ATTENTION"},
                {"invariant_type": "type2", "severity": "ATTENTION"},
            ],
            "terms_involved": ["term1"],
            "neutral_notes": [],
        }
        
        summary = build_semantic_integrity_summary_for_p3(tile, invariant_check)
        
        self.assertEqual(summary["broken_invariant_count"], 2)
        self.assertFalse(summary["invariants_ok"])


# =============================================================================
# P4 BINDING TESTS
# =============================================================================

class TestBuildSemanticIntegrityCalibrationForP4(unittest.TestCase):
    """Tests for build_semantic_integrity_calibration_for_p4 function."""
    
    def test_calibration_has_required_fields(self):
        """Calibration should have all required fields."""
        drift_signal = {
            "drift_detected": False,
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        calibration = build_semantic_integrity_calibration_for_p4(drift_signal)
        
        self.assertIn("drift_detected", calibration)
        self.assertIn("severity", calibration)
        self.assertIn("terms_involved", calibration)
        self.assertIn("critical_signals", calibration)
    
    def test_calibration_deterministic(self):
        """Calibration should be deterministic."""
        drift_signal = {
            "drift_detected": True,
            "severity": "BROKEN",
            "terms_involved": ["term1"],
            "critical_signals": ["type1"],
        }
        
        cal1 = build_semantic_integrity_calibration_for_p4(drift_signal)
        cal2 = build_semantic_integrity_calibration_for_p4(drift_signal)
        
        self.assertEqual(cal1, cal2)


# =============================================================================
# EVIDENCE HELPER TESTS
# =============================================================================

class TestAttachSemanticIntegrityToEvidence(unittest.TestCase):
    """Tests for attach_semantic_integrity_to_evidence function."""
    
    def test_evidence_shape(self):
        """Evidence should have correct shape."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
            "uplift_semantic_status": "OK",
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        result = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        
        self.assertIn("governance", result)
        self.assertIn("semantic_integrity", result["governance"])
        self.assertIn("invariants_ok", result["governance"]["semantic_integrity"])
        self.assertIn("broken_invariant_count", result["governance"]["semantic_integrity"])
        self.assertIn("severity", result["governance"]["semantic_integrity"])
        self.assertIn("terms_involved", result["governance"]["semantic_integrity"])
        self.assertIn("critical_signals", result["governance"]["semantic_integrity"])
    
    def test_evidence_deterministic(self):
        """Evidence attachment should be deterministic."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        result1 = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        result2 = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        
        self.assertEqual(result1, result2)
    
    def test_evidence_json_safe(self):
        """Evidence should be JSON-serializable."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        result = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        
        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)
    
    def test_evidence_non_mutation(self):
        """Should not mutate input evidence."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        original_evidence = dict(evidence)
        result = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        
        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should be different (has new governance key)
        self.assertNotEqual(result, original_evidence)
        self.assertIn("governance", result)
        self.assertNotIn("governance", original_evidence)


# =============================================================================
# COUNCIL SUMMARY TESTS
# =============================================================================

class TestSummarizeSemanticIntegrityForUpliftCouncil(unittest.TestCase):
    """Tests for summarize_semantic_integrity_for_uplift_council function."""
    
    def test_council_summary_ok(self):
        """Should return OK status when all checks pass."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        
        summary = summarize_semantic_integrity_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "OK")
        self.assertTrue(summary["invariants_ok"])
        self.assertEqual(summary["uplift_semantic_status"], "OK")
    
    def test_council_summary_warn(self):
        """Should return WARN status when uplift is WARN."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "WARN",
            "top_risk_terms": ["term1"],
            "headline": "Review recommended",
        }
        
        summary = summarize_semantic_integrity_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "WARN")
        self.assertTrue(summary["invariants_ok"])
        self.assertEqual(summary["uplift_semantic_status"], "WARN")
    
    def test_council_summary_block_on_uplift(self):
        """Should return BLOCK status when uplift is BLOCK."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "BLOCK",
            "top_risk_terms": ["term1", "term2"],
            "headline": "Uplift blocked",
        }
        
        summary = summarize_semantic_integrity_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "BLOCK")
        self.assertEqual(summary["uplift_semantic_status"], "BLOCK")
    
    def test_council_summary_block_on_invariants(self):
        """Should return BLOCK status when invariants_ok is False."""
        tile = {
            "invariants_ok": False,
            "uplift_semantic_status": "OK",
            "top_risk_terms": [],
            "headline": "Invariants violated",
        }
        
        summary = summarize_semantic_integrity_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "BLOCK")
        self.assertFalse(summary["invariants_ok"])
    
    def test_council_summary_top_risk_terms(self):
        """Should include top_risk_terms."""
        tile = {
            "invariants_ok": True,
            "uplift_semantic_status": "OK",
            "top_risk_terms": ["term1", "term2", "term3"],
            "headline": "All checks passed",
        }
        
        summary = summarize_semantic_integrity_for_uplift_council(tile)
        
        self.assertEqual(summary["top_risk_terms"], ["term1", "term2", "term3"])


# =============================================================================
# FIRST LIGHT SEMANTIC FOOTPRINT TESTS
# =============================================================================

# NOTE: The semantic footprint is designed as a compact safety badge for external
# reviewers examining evidence packs. It combines P3 (synthetic) and P4 (shadow)
# semantic integrity signals into a single, human-readable record.
#
# The footprint is NOT a direct gate - it is advisory only. The council still
# makes final decisions based on all available evidence. The footprint provides
# a clear view of semantic integrity across both experimental phases, allowing
# external reviewers to quickly assess the semantic safety posture using the
# 2×2 grid (P3 status × P4 severity).
#
# See docs/system_law/Semantic_Integrity_PhaseX.md for the full specification
# and interpretive guidance.

class TestBuildFirstLightSemanticFootprint(unittest.TestCase):
    """Tests for build_first_light_semantic_footprint function."""
    
    def test_footprint_has_required_fields(self):
        """Footprint should have all required fields."""
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
            "top_risk_terms": [],
            "headline": "All checks passed",
        }
        p4_calibration = {
            "drift_detected": False,
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        footprint = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        
        self.assertEqual(footprint["schema_version"], "1.0.0")
        self.assertIn("invariants_ok", footprint)
        self.assertIn("broken_invariant_count", footprint)
        self.assertIn("p3_uplift_semantic_status", footprint)
        self.assertIn("p4_severity", footprint)
        self.assertIn("terms_involved", footprint)
    
    def test_footprint_deterministic(self):
        """Footprint should be deterministic."""
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        footprint1 = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        footprint2 = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        
        self.assertEqual(footprint1, footprint2)
    
    def test_footprint_json_safe(self):
        """Footprint should be JSON-serializable."""
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        footprint = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        
        # Should not raise
        json_str = json.dumps(footprint)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0.0")
    
    def test_footprint_terms_limited_to_5(self):
        """terms_involved should be limited to top 5."""
        p3_summary = {
            "invariants_ok": False,
            "broken_invariant_count": 2,
            "uplift_semantic_status": "WARN",
        }
        p4_calibration = {
            "severity": "ATTENTION",
            "terms_involved": ["term1", "term2", "term3", "term4", "term5", "term6", "term7"],
        }
        
        footprint = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        
        self.assertEqual(len(footprint["terms_involved"]), 5)
        self.assertEqual(footprint["terms_involved"], ["term1", "term2", "term3", "term4", "term5"])
    
    def test_footprint_values_correct(self):
        """Footprint should correctly extract values from inputs."""
        p3_summary = {
            "invariants_ok": False,
            "broken_invariant_count": 3,
            "uplift_semantic_status": "BLOCK",
        }
        p4_calibration = {
            "severity": "BROKEN",
            "terms_involved": ["term1", "term2"],
        }
        
        footprint = build_first_light_semantic_footprint(p3_summary, p4_calibration)
        
        self.assertFalse(footprint["invariants_ok"])
        self.assertEqual(footprint["broken_invariant_count"], 3)
        self.assertEqual(footprint["p3_uplift_semantic_status"], "BLOCK")
        self.assertEqual(footprint["p4_severity"], "BROKEN")
        self.assertEqual(footprint["terms_involved"], ["term1", "term2"])


# =============================================================================
# EVIDENCE FOOTPRINT INTEGRATION TESTS
# =============================================================================

class TestAttachSemanticIntegrityToEvidenceWithFootprint(unittest.TestCase):
    """Tests for attach_semantic_integrity_to_evidence with footprint."""
    
    def test_evidence_includes_footprint_when_provided(self):
        """Evidence should include footprint when P3 and P4 data provided."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        result = attach_semantic_integrity_to_evidence(
            evidence, tile, drift_signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        self.assertIn("first_light_footprint", result["governance"]["semantic_integrity"])
        footprint = result["governance"]["semantic_integrity"]["first_light_footprint"]
        self.assertEqual(footprint["schema_version"], "1.0.0")
        self.assertTrue(footprint["invariants_ok"])
    
    def test_evidence_no_footprint_when_not_provided(self):
        """Evidence should not include footprint when P3/P4 data not provided."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        
        result = attach_semantic_integrity_to_evidence(evidence, tile, drift_signal)
        
        self.assertNotIn("first_light_footprint", result["governance"]["semantic_integrity"])
    
    def test_evidence_footprint_deterministic(self):
        """Evidence with footprint should be deterministic."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        result1 = attach_semantic_integrity_to_evidence(
            evidence, tile, drift_signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        result2 = attach_semantic_integrity_to_evidence(
            evidence, tile, drift_signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        self.assertEqual(result1, result2)
    
    def test_evidence_footprint_json_safe(self):
        """Evidence with footprint should be JSON-serializable."""
        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        result = attach_semantic_integrity_to_evidence(
            evidence, tile, drift_signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertIn("first_light_footprint", parsed["governance"]["semantic_integrity"])
    
    def test_evidence_footprint_non_mutation(self):
        """Should not mutate input evidence when footprint is added."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        tile = {
            "invariants_ok": True,
            "broken_invariants": [],
        }
        drift_signal = {
            "severity": "OK",
            "terms_involved": [],
            "critical_signals": [],
        }
        p3_summary = {
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "uplift_semantic_status": "OK",
        }
        p4_calibration = {
            "severity": "OK",
            "terms_involved": [],
        }
        
        original_evidence = dict(evidence)
        result = attach_semantic_integrity_to_evidence(
            evidence, tile, drift_signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should have footprint
        self.assertIn("first_light_footprint", result["governance"]["semantic_integrity"])


# =============================================================================
# CALIBRATION EXPERIMENT SEMANTIC FOOTPRINT TESTS
# =============================================================================

class TestEmitCalExpSemanticFootprint(unittest.TestCase):
    """Tests for emit_cal_exp_semantic_footprint function."""

    def test_footprint_has_required_fields(self):
        """Calibration footprint should have all required fields."""
        footprint = {
            "schema_version": "1.0.0",
            "invariants_ok": True,
            "broken_invariant_count": 0,
            "p3_uplift_semantic_status": "OK",
            "p4_severity": "OK",
            "terms_involved": [],
        }

        result = emit_cal_exp_semantic_footprint("CAL-EXP-1", footprint)

        self.assertEqual(result["schema_version"], SEMANTIC_FOOTPRINT_SCHEMA_VERSION)
        self.assertEqual(result["cal_id"], "CAL-EXP-1")
        self.assertEqual(result["p3_status"], "OK")
        self.assertEqual(result["p4_status"], "OK")
        self.assertEqual(result["broken_invariant_count"], 0)

    def test_footprint_extracts_p3_p4_status(self):
        """Should extract P3 and P4 status from footprint."""
        footprint = {
            "p3_uplift_semantic_status": "WARN",
            "p4_severity": "ATTENTION",
            "broken_invariant_count": 2,
        }

        result = emit_cal_exp_semantic_footprint("CAL-EXP-2", footprint)

        self.assertEqual(result["p3_status"], "WARN")
        self.assertEqual(result["p4_status"], "ATTENTION")
        self.assertEqual(result["broken_invariant_count"], 2)

    def test_footprint_json_safe(self):
        """Calibration footprint should be JSON-serializable."""
        footprint = {
            "p3_uplift_semantic_status": "OK",
            "p4_severity": "OK",
            "broken_invariant_count": 0,
        }

        result = emit_cal_exp_semantic_footprint("CAL-EXP-1", footprint)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["cal_id"], "CAL-EXP-1")

    def test_footprint_deterministic(self):
        """Calibration footprint should be deterministic."""
        footprint = {
            "p3_uplift_semantic_status": "OK",
            "p4_severity": "OK",
            "broken_invariant_count": 0,
        }

        result1 = emit_cal_exp_semantic_footprint("CAL-EXP-1", footprint)
        result2 = emit_cal_exp_semantic_footprint("CAL-EXP-1", footprint)

        self.assertEqual(result1, result2)


class TestPersistCalExpSemanticFootprint(unittest.TestCase):
    """Tests for persist_cal_exp_semantic_footprint function."""

    def test_persist_creates_file(self):
        """Should create footprint file in output directory."""
        footprint = {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-1",
            "p3_status": "OK",
            "p4_status": "OK",
            "broken_invariant_count": 0,
        }

        with TemporaryDirectory() as tmpdir:
            output_path = persist_cal_exp_semantic_footprint(
                footprint, Path(tmpdir)
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "semantic_footprint_CAL-EXP-1.json")

    def test_persist_creates_directory(self):
        """Should create output directory if it doesn't exist."""
        footprint = {
            "cal_id": "CAL-EXP-1",
            "p3_status": "OK",
            "p4_status": "OK",
            "broken_invariant_count": 0,
        }

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            output_path = persist_cal_exp_semantic_footprint(footprint, output_dir)

            self.assertTrue(output_dir.exists())
            self.assertTrue(output_path.exists())

    def test_persist_writes_valid_json(self):
        """Should write valid JSON to file."""
        footprint = {
            "cal_id": "CAL-EXP-1",
            "p3_status": "OK",
            "p4_status": "OK",
            "broken_invariant_count": 0,
        }

        with TemporaryDirectory() as tmpdir:
            output_path = persist_cal_exp_semantic_footprint(
                footprint, Path(tmpdir)
            )

            with open(output_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)

            self.assertEqual(parsed["cal_id"], "CAL-EXP-1")
            self.assertEqual(parsed["p3_status"], "OK")

    def test_persist_non_mutating(self):
        """Should not mutate input footprint."""
        footprint = {
            "cal_id": "CAL-EXP-1",
            "p3_status": "OK",
            "p4_status": "OK",
            "broken_invariant_count": 0,
        }
        original = dict(footprint)

        with TemporaryDirectory() as tmpdir:
            persist_cal_exp_semantic_footprint(footprint, Path(tmpdir))

        self.assertEqual(footprint, original)


class TestBuildSemanticSafetyPanel(unittest.TestCase):
    """Tests for build_semantic_safety_panel function."""

    def test_panel_has_required_fields(self):
        """Panel should have all required fields."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        self.assertEqual(panel["schema_version"], SEMANTIC_SAFETY_PANEL_SCHEMA_VERSION)
        self.assertEqual(panel["total_experiments"], 1)
        self.assertIn("grid_counts", panel)
        self.assertIn("experiments", panel)

    def test_panel_classifies_ok_ok(self):
        """Should classify OK×OK experiments correctly."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        self.assertEqual(panel["grid_counts"]["ok_ok"], 1)
        self.assertEqual(panel["grid_counts"]["ok_not_ok"], 0)
        self.assertEqual(panel["grid_counts"]["not_ok_ok"], 0)
        self.assertEqual(panel["grid_counts"]["not_ok_not_ok"], 0)
        self.assertEqual(panel["experiments"][0]["grid_bucket"], "OK×OK")

    def test_panel_classifies_all_buckets(self):
        """Should classify all grid buckets correctly."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 1,
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_status": "WARN",
                "p4_status": "OK",
                "broken_invariant_count": 2,
            },
            {
                "cal_id": "CAL-EXP-4",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 5,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        self.assertEqual(panel["grid_counts"]["ok_ok"], 1)
        self.assertEqual(panel["grid_counts"]["ok_not_ok"], 1)
        self.assertEqual(panel["grid_counts"]["not_ok_ok"], 1)
        self.assertEqual(panel["grid_counts"]["not_ok_not_ok"], 1)
        self.assertEqual(panel["total_experiments"], 4)

    def test_panel_json_safe(self):
        """Panel should be JSON-serializable."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        # Should not raise
        json_str = json.dumps(panel)
        self.assertIsInstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["total_experiments"], 1)

    def test_panel_deterministic(self):
        """Panel should be deterministic."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
        ]

        panel1 = build_semantic_safety_panel(footprints)
        panel2 = build_semantic_safety_panel(footprints)

        self.assertEqual(panel1, panel2)

    def test_panel_includes_experiment_details(self):
        """Panel should include per-experiment details."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "WARN",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 3,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        self.assertEqual(len(panel["experiments"]), 2)
        self.assertEqual(panel["experiments"][0]["cal_id"], "CAL-EXP-1")
        self.assertEqual(panel["experiments"][0]["grid_bucket"], "OK×OK")
        self.assertEqual(panel["experiments"][1]["cal_id"], "CAL-EXP-2")
        self.assertEqual(panel["experiments"][1]["grid_bucket"], "Not-OK×Not-OK")
        self.assertEqual(panel["experiments"][1]["broken_invariant_count"], 3)

    def test_panel_top_drivers_ranking(self):
        """Top drivers should be ranked by worst bucket, then broken count, then cal_id."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "OK",
                "broken_invariant_count": 0,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 1,
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_status": "WARN",
                "p4_status": "OK",
                "broken_invariant_count": 2,
            },
            {
                "cal_id": "CAL-EXP-4",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 5,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        # Top drivers should be: CAL-EXP-4 (not_ok_not_ok), then CAL-EXP-2/3 (ok_not_ok/not_ok_ok)
        # CAL-EXP-2 has lower broken count (1) than CAL-EXP-3 (2), so CAL-EXP-3 comes first
        self.assertEqual(len(panel["top_drivers"]), 3)
        self.assertEqual(panel["top_drivers"][0], "CAL-EXP-4")  # Worst bucket
        # Between CAL-EXP-2 and CAL-EXP-3, both are in middle buckets
        # CAL-EXP-3 has higher broken count (2) so comes before CAL-EXP-2 (1)
        self.assertIn("CAL-EXP-3", panel["top_drivers"])
        self.assertIn("CAL-EXP-2", panel["top_drivers"])

    def test_panel_top_drivers_tie_break_by_broken_count(self):
        """Top drivers should tie-break by broken_invariant_count (higher = worse)."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 1,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 3,
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 2,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        # All in same bucket (ok_not_ok), so ranked by broken count (descending)
        self.assertEqual(panel["top_drivers"], ["CAL-EXP-2", "CAL-EXP-3", "CAL-EXP-1"])

    def test_panel_top_drivers_tie_break_by_cal_id(self):
        """Top drivers should tie-break by cal_id (alphabetical) when broken counts equal."""
        footprints = [
            {
                "cal_id": "CAL-EXP-B",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 2,
            },
            {
                "cal_id": "CAL-EXP-A",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 2,
            },
            {
                "cal_id": "CAL-EXP-C",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 2,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        # All in same bucket with same broken count, so ranked alphabetically by cal_id
        self.assertEqual(panel["top_drivers"], ["CAL-EXP-A", "CAL-EXP-B", "CAL-EXP-C"])

    def test_panel_top_drivers_limited_to_3(self):
        """Top drivers should be limited to 3 even if more experiments exist."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 5,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 4,
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 3,
            },
            {
                "cal_id": "CAL-EXP-4",
                "p3_status": "BLOCK",
                "p4_status": "BROKEN",
                "broken_invariant_count": 2,
            },
        ]

        panel = build_semantic_safety_panel(footprints)

        # Should only include top 3
        self.assertEqual(len(panel["top_drivers"]), 3)
        self.assertEqual(panel["top_drivers"], ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"])

    def test_panel_top_drivers_deterministic(self):
        """Top drivers ranking should be deterministic."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_status": "OK",
                "p4_status": "ATTENTION",
                "broken_invariant_count": 1,
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_status": "WARN",
                "p4_status": "OK",
                "broken_invariant_count": 2,
            },
        ]

        panel1 = build_semantic_safety_panel(footprints)
        panel2 = build_semantic_safety_panel(footprints)

        self.assertEqual(panel1["top_drivers"], panel2["top_drivers"])


class TestExtractSemanticSafetyPanelSignal(unittest.TestCase):
    """Tests for extract_semantic_safety_panel_signal function."""

    def test_signal_has_required_fields(self):
        """Signal should have all required fields."""
        panel = {
            "schema_version": "1.0.0",
            "total_experiments": 3,
            "grid_counts": {
                "ok_ok": 2,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-3"],
            "experiments": [],
        }

        signal = extract_semantic_safety_panel_signal(panel)

        self.assertEqual(signal["ok_ok"], 2)
        self.assertEqual(signal["ok_not_ok"], 0)
        self.assertEqual(signal["not_ok_ok"], 0)
        self.assertEqual(signal["not_ok_not_ok"], 1)
        self.assertEqual(signal["top_drivers"], ["CAL-EXP-3"])

    def test_signal_json_safe(self):
        """Signal should be JSON-serializable."""
        panel = {
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
            "top_drivers": [],
        }

        signal = extract_semantic_safety_panel_signal(panel)

        # Should not raise
        json_str = json.dumps(signal)
        self.assertIsInstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["ok_ok"], 1)

    def test_signal_deterministic(self):
        """Signal extraction should be deterministic."""
        panel = {
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
            "top_drivers": ["CAL-EXP-1"],
        }

        signal1 = extract_semantic_safety_panel_signal(panel)
        signal2 = extract_semantic_safety_panel_signal(panel)

        self.assertEqual(signal1, signal2)


class TestAttachSemanticSafetyPanelToEvidence(unittest.TestCase):
    """Tests for attach_semantic_safety_panel_to_evidence function."""

    def test_evidence_includes_panel(self):
        """Evidence should include semantic safety panel."""
        evidence = {"timestamp": "2024-01-01"}
        panel = {
            "schema_version": "1.0.0",
            "total_experiments": 1,
            "grid_counts": {"ok_ok": 1, "ok_not_ok": 0, "not_ok_ok": 0, "not_ok_not_ok": 0},
            "experiments": [],
        }

        result = attach_semantic_safety_panel_to_evidence(evidence, panel)

        self.assertIn("semantic_safety_panel", result["governance"])
        self.assertEqual(result["governance"]["semantic_safety_panel"], panel)

    def test_evidence_panel_json_safe(self):
        """Evidence with panel should be JSON-serializable."""
        evidence = {"timestamp": "2024-01-01"}
        panel = {
            "total_experiments": 1,
            "grid_counts": {"ok_ok": 1},
            "experiments": [],
        }

        result = attach_semantic_safety_panel_to_evidence(evidence, panel)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        self.assertIn("semantic_safety_panel", parsed["governance"])

    def test_evidence_panel_non_mutation(self):
        """Should not mutate input evidence when panel is added."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        panel = {
            "total_experiments": 1,
            "grid_counts": {"ok_ok": 1},
            "experiments": [],
        }

        original_evidence = dict(evidence)
        result = attach_semantic_safety_panel_to_evidence(evidence, panel)

        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should have panel
        self.assertIn("semantic_safety_panel", result["governance"])

    def test_evidence_panel_deterministic(self):
        """Evidence with panel should be deterministic."""
        evidence = {"timestamp": "2024-01-01"}
        panel = {
            "total_experiments": 1,
            "grid_counts": {"ok_ok": 1},
            "top_drivers": [],
            "experiments": [],
        }

        result1 = attach_semantic_safety_panel_to_evidence(evidence, panel)
        result2 = attach_semantic_safety_panel_to_evidence(evidence, panel)

        self.assertEqual(result1, result2)

    def test_evidence_includes_signal(self):
        """Evidence should include signal under signals.semantic_safety_panel."""
        evidence = {"timestamp": "2024-01-01"}
        panel = {
            "total_experiments": 2,
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-2"],
            "experiments": [],
        }

        result = attach_semantic_safety_panel_to_evidence(evidence, panel)

        self.assertIn("signals", result)
        self.assertIn("semantic_safety_panel", result["signals"])
        signal = result["signals"]["semantic_safety_panel"]
        self.assertEqual(signal["ok_ok"], 1)
        self.assertEqual(signal["not_ok_not_ok"], 1)
        self.assertEqual(signal["top_drivers"], ["CAL-EXP-2"])

    def test_evidence_signal_json_safe(self):
        """Evidence signal should be JSON-serializable."""
        evidence = {"timestamp": "2024-01-01"}
        panel = {
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
            "top_drivers": [],
            "experiments": [],
        }

        result = attach_semantic_safety_panel_to_evidence(evidence, panel)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        self.assertIn("semantic_safety_panel", parsed["signals"])


class TestSemanticSafetyPanelForAlignmentView(unittest.TestCase):
    """Tests for semantic_safety_panel_for_alignment_view function."""

    def test_ggfl_output_has_required_fields(self):
        """GGFL output should have all required fields."""
        panel = {
            "schema_version": "1.0.0",
            "total_experiments": 3,
            "grid_counts": {
                "ok_ok": 2,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-3"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result["signal_type"], "SIG-SEM")
        self.assertIn(result["status"], ["ok", "warn"])
        self.assertFalse(result["conflict"])
        self.assertEqual(result["weight_hint"], "LOW")
        self.assertIsInstance(result["drivers"], list)
        self.assertIsInstance(result["summary"], str)

    def test_ggfl_status_warn_when_not_ok_not_ok(self):
        """GGFL status should be 'warn' when not_ok_not_ok > 0."""
        panel = {
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-2"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result["status"], "warn")

    def test_ggfl_status_ok_when_all_ok(self):
        """GGFL status should be 'ok' when all experiments are ok_ok."""
        panel = {
            "grid_counts": {
                "ok_ok": 3,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
            "top_drivers": [],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result["status"], "ok")

    def test_ggfl_conflict_always_false(self):
        """GGFL conflict should always be False."""
        panel = {
            "grid_counts": {
                "ok_ok": 0,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 3,
            },
            "top_drivers": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertFalse(result["conflict"])

    def test_ggfl_weight_hint_low(self):
        """GGFL weight_hint should always be 'LOW'."""
        panel = {
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result["weight_hint"], "LOW")

    def test_ggfl_drivers_limited_to_3(self):
        """GGFL drivers should be limited to top 3 and sorted deterministically."""
        panel = {
            "grid_counts": {
                "ok_ok": 0,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 5,
            },
            "top_drivers": ["CAL-EXP-5", "CAL-EXP-1", "CAL-EXP-3", "CAL-EXP-2", "CAL-EXP-4"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertLessEqual(len(result["drivers"]), 3)
        # Should be sorted deterministically
        self.assertEqual(result["drivers"], sorted(result["drivers"]))

    def test_ggfl_summary_neutral_single_sentence(self):
        """GGFL summary should be a neutral single sentence."""
        panel = {
            "total_experiments": 3,
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 2,
            },
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        summary = result["summary"]
        # Should be a single sentence (ends with period, no newlines)
        self.assertTrue(summary.endswith("."))
        # Use reusable helper (single source of truth for banned words)
        neutrality_result = assert_warning_neutral(summary)
        self.assertTrue(neutrality_result.passed, f"Summary neutrality check failed: {neutrality_result.message}")

    def test_ggfl_works_with_signal_format(self):
        """GGFL adapter should work with signal format (not just panel format)."""
        signal = {
            "ok_ok": 2,
            "ok_not_ok": 0,
            "not_ok_ok": 0,
            "not_ok_not_ok": 1,
            "top_drivers": ["CAL-EXP-3"],
        }

        result = semantic_safety_panel_for_alignment_view(signal)

        self.assertEqual(result["signal_type"], "SIG-SEM")
        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["drivers"], ["CAL-EXP-3"])
        self.assertEqual(result["drivers_reason_codes"], ["SEM-DRV-001:CAL-EXP-3"])

    def test_ggfl_drivers_reason_codes_deterministic_ordering(self):
        """GGFL drivers_reason_codes should have deterministic ordering (SEM-DRV-001, SEM-DRV-002, SEM-DRV-003)."""
        panel = {
            "grid_counts": {
                "ok_ok": 0,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 3,
            },
            "top_drivers": ["CAL-EXP-3", "CAL-EXP-1", "CAL-EXP-2"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        # Drivers should be sorted (human-readable)
        self.assertEqual(result["drivers"], ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"])
        # Reason codes should map deterministically: SEM-DRV-001 for first, SEM-DRV-002 for second, etc.
        self.assertEqual(len(result["drivers_reason_codes"]), 3)
        self.assertTrue(result["drivers_reason_codes"][0].startswith("SEM-DRV-001:"))
        self.assertTrue(result["drivers_reason_codes"][1].startswith("SEM-DRV-002:"))
        self.assertTrue(result["drivers_reason_codes"][2].startswith("SEM-DRV-003:"))
        # Verify cal_ids are included in reason codes
        self.assertIn("CAL-EXP-1", result["drivers_reason_codes"][0])
        self.assertIn("CAL-EXP-2", result["drivers_reason_codes"][1])
        self.assertIn("CAL-EXP-3", result["drivers_reason_codes"][2])

    def test_ggfl_drivers_reason_codes_limited_to_3(self):
        """GGFL drivers_reason_codes should be limited to 3 even if more drivers exist."""
        panel = {
            "grid_counts": {
                "ok_ok": 0,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 5,
            },
            "top_drivers": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3", "CAL-EXP-4", "CAL-EXP-5"],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        # Should only have 3 drivers and 3 reason codes
        self.assertEqual(len(result["drivers"]), 3)
        self.assertEqual(len(result["drivers_reason_codes"]), 3)
        # Should only include first 3 cal_ids
        self.assertIn("CAL-EXP-1", result["drivers"][0])
        self.assertIn("CAL-EXP-2", result["drivers"][1])
        self.assertIn("CAL-EXP-3", result["drivers"][2])
        self.assertNotIn("CAL-EXP-4", str(result["drivers"]))
        self.assertNotIn("CAL-EXP-5", str(result["drivers"]))

    def test_ggfl_drivers_reason_codes_empty_when_no_drivers(self):
        """GGFL drivers_reason_codes should be empty when no drivers present."""
        panel = {
            "grid_counts": {
                "ok_ok": 3,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 0,
            },
            "top_drivers": [],
        }

        result = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result["drivers"], [])
        self.assertEqual(result["drivers_reason_codes"], [])

    def test_ggfl_deterministic(self):
        """GGFL output should be deterministic."""
        panel = {
            "total_experiments": 2,
            "grid_counts": {
                "ok_ok": 1,
                "ok_not_ok": 0,
                "not_ok_ok": 0,
                "not_ok_not_ok": 1,
            },
            "top_drivers": ["CAL-EXP-2"],
        }

        result1 = semantic_safety_panel_for_alignment_view(panel)
        result2 = semantic_safety_panel_for_alignment_view(panel)

        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

