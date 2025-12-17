"""
Phase X: CI Smoke Test for Chronicle Governance Tile Serialization

This test verifies that the chronicle governance tile can be produced and serialized
without error. It does NOT test governance logic, recurrence projection semantics, or
invariant behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No recurrence projection or invariant checking logic is run
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock recurrence_projection, invariant_check, stability_scores
2. Call build_chronicle_governance_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Assert: Required fields present
6. Assert: Determinism across runs
7. Assert: Neutral tone in headline
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

import pytest


class TestChronicleGovernanceTileSerializes:
    """
    CI smoke tests for chronicle governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_chronicle_tile_serializes_without_error(self) -> None:
        """
        Verify chronicle governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        # 1. Create mock inputs
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": ["Low recurrence indicators"],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test explanation",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": ["All invariants satisfied."],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Stable",
            "evidence_fields": {},
            "composite_score": 0.8,
        }

        # 2. Call build_chronicle_governance_tile()
        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        # 3. Assert: isinstance(tile, dict)
        assert tile is not None, "Tile should not be None"
        assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

        # 4. Assert: json.dumps(tile) does not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_chronicle_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.chronicle_governance_adapter import (
            CHRONICLE_GOVERNANCE_TILE_SCHEMA_VERSION,
            build_chronicle_governance_tile,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "tile_type",
            "status_light",
            "recurrence_band",
            "projected_horizon_days",
            "invariants_ok",
            "highlighted_cases",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == CHRONICLE_GOVERNANCE_TILE_SCHEMA_VERSION

        # Verify tile_type
        assert tile["tile_type"] == "chronicle_governance"

    def test_chronicle_tile_deterministic(self) -> None:
        """Verify tile is deterministic across runs."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.5,
            "drivers": ["Test driver"],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "MEDIUM",
            "axes_contributing": ["alignment"],
            "headline": "Test",
        }

        tile1 = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        tile2 = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        # Should be identical
        assert tile1 == tile2, "Tile should be deterministic"

        # JSON serialization should also be identical
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2, "JSON serialization should be deterministic"

    def test_chronicle_tile_neutral_tone(self) -> None:
        """Verify headline uses neutral, descriptive language."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        headline = tile["headline"].lower()
        # Should not contain judgmental words
        assert "bad" not in headline
        assert "good" not in headline
        assert "problem" not in headline
        assert "error" not in headline
        assert "failure" not in headline

        # Should contain descriptive terms
        assert "status" in headline or "recurrence" in headline

    def test_chronicle_tile_status_light_logic(self) -> None:
        """Verify status light logic (RED on violations, YELLOW on high recurrence, GREEN otherwise)."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        # Test RED (invariants violated)
        tile_red = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.2,
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "VIOLATED",
                "broken_invariants": ["Test violation"],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "HIGH",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        assert tile_red["status_light"] == "RED"

        # Test YELLOW (high recurrence)
        tile_yellow = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.8,  # High recurrence
                "drivers": ["High causality"],
                "projected_recurrence_horizon": 15,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "MEDIUM",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        assert tile_yellow["status_light"] == "YELLOW"

        # Test GREEN (all nominal)
        tile_green = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.2,  # Low recurrence
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "HIGH",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        assert tile_green["status_light"] == "GREEN"

    def test_chronicle_tile_highlighted_cases_limit(self) -> None:
        """Verify highlighted_cases is limited to top 3."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.8,  # High recurrence
            "drivers": ["Driver 1", "Driver 2", "Driver 3", "Driver 4"],  # Many drivers
            "projected_recurrence_horizon": 15,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "VIOLATED",
            "broken_invariants": ["Violation 1", "Violation 2"],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "LOW",
            "axes_contributing": ["axis1", "axis2", "axis3"],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        # Should be limited to top 3
        assert len(tile["highlighted_cases"]) <= 3

    def test_extract_chronicle_drift_signal(self) -> None:
        """Verify extract_chronicle_drift_signal produces valid signal."""
        from backend.health.chronicle_governance_adapter import (
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.5,
            "drivers": [],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }

        signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Required fields
        assert "recurrence_likelihood" in signal
        assert "invariants_ok" in signal
        assert "band" in signal

        # Verify types
        assert isinstance(signal["recurrence_likelihood"], (int, float))
        assert isinstance(signal["invariants_ok"], bool)
        assert signal["band"] in ("LOW", "MEDIUM", "HIGH")

        # Verify serializable
        json_str = json.dumps(signal)
        assert len(json_str) > 0


class TestGlobalHealthSurfaceChronicleIntegration:
    """
    Tests for chronicle governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_without_chronicle(self) -> None:
        """Verify build works without chronicle inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "chronicle_governance" not in payload  # Should not be present

    def test_build_global_health_surface_with_chronicle(self) -> None:
        """Verify chronicle tile attached when all inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        payload = build_global_health_surface(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        assert isinstance(payload, dict)
        assert "chronicle_governance" in payload, "Chronicle tile should be present"

        chronicle_tile = payload["chronicle_governance"]
        assert isinstance(chronicle_tile, dict)
        assert "headline" in chronicle_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_chronicle_tile_does_not_affect_dynamics(self) -> None:
        """Verify chronicle tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        # Build without chronicle
        payload_without = build_global_health_surface()

        # Build with chronicle
        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        payload_with = build_global_health_surface(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]


class TestChronicleEvidenceAndCouncilIntegration(unittest.TestCase):
    """
    Tests for chronicle governance evidence attachment and council summary.

    SHADOW MODE: These tests verify evidence attachment and council summary only.
    """

    def test_attach_chronicle_governance_to_evidence(self) -> None:
        """Verify chronicle governance attaches to evidence pack correctly."""
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.5,
            "drivers": ["Test driver"],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "MEDIUM",
            "axes_contributing": ["alignment"],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence = {"evidence_type": "test"}
        result = attach_chronicle_governance_to_evidence(
            evidence, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Verify structure
        assert "governance" in result
        assert "chronicle" in result["governance"]
        
        chronicle = result["governance"]["chronicle"]
        assert "recurrence_likelihood" in chronicle
        assert "band" in chronicle
        assert "invariants_ok" in chronicle
        assert "highlighted_cases" in chronicle
        assert "status_light" in chronicle
        assert "recurrence_band" in chronicle
        assert "projected_horizon_days" in chronicle

        # Verify values
        assert chronicle["recurrence_likelihood"] == 0.5
        assert chronicle["band"] == "MEDIUM"
        assert chronicle["invariants_ok"] is True

        # Verify First Light annex is attached
        assert "first_light_annex" in chronicle
        annex = chronicle["first_light_annex"]
        assert "schema_version" in annex
        assert annex["schema_version"] == "1.0.0"
        assert "recurrence_likelihood" in annex
        assert "band" in annex
        assert "invariants_ok" in annex

    def test_attach_chronicle_governance_to_evidence_deterministic(self) -> None:
        """Verify evidence attachment is deterministic."""
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence1 = {"evidence_type": "test"}
        evidence2 = {"evidence_type": "test"}

        result1 = attach_chronicle_governance_to_evidence(
            evidence1, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )
        result2 = attach_chronicle_governance_to_evidence(
            evidence2, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Should be identical
        assert result1["governance"]["chronicle"] == result2["governance"]["chronicle"]

    def test_attach_chronicle_governance_to_evidence_json_serializable(self) -> None:
        """Verify attached evidence is JSON serializable."""
        import json
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence = {"evidence_type": "test"}
        result = attach_chronicle_governance_to_evidence(
            evidence, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Should serialize without error
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "chronicle" in parsed["governance"]
        assert "first_light_annex" in parsed["governance"]["chronicle"]

    def test_attach_chronicle_governance_to_evidence_non_mutation(self) -> None:
        """Verify original evidence dict is not mutated (returns new dict)."""
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence = {"evidence_type": "test"}
        original_keys = set(evidence.keys())
        
        result = attach_chronicle_governance_to_evidence(
            evidence, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Evidence should be modified in-place (same object reference)
        assert result is evidence
        # But should have new keys
        assert "governance" in evidence
        assert "first_light_annex" in evidence["governance"]["chronicle"]

    def test_summarize_chronicle_for_uplift_council_block(self) -> None:
        """Verify council summary maps to BLOCK correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
            summarize_chronicle_for_uplift_council,
        )

        # Test BLOCK: invariants violated
        tile_violated = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.2,
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "VIOLATED",
                "broken_invariants": ["Test violation"],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "HIGH",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        summary_violated = summarize_chronicle_for_uplift_council(tile_violated)
        assert summary_violated["council_status"] == "BLOCK"
        assert summary_violated["invariants_ok"] is False

        # Test BLOCK: high recurrence
        tile_high = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.8,  # High recurrence
                "drivers": ["High causality"],
                "projected_recurrence_horizon": 15,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "MEDIUM",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        summary_high = summarize_chronicle_for_uplift_council(tile_high)
        assert summary_high["council_status"] == "BLOCK"
        assert summary_high["recurrence_band"] == "HIGH"

    def test_summarize_chronicle_for_uplift_council_warn(self) -> None:
        """Verify council summary maps to WARN correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
            summarize_chronicle_for_uplift_council,
        )

        tile = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.5,  # Medium recurrence
                "drivers": ["Moderate causality"],
                "projected_recurrence_horizon": 22,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "MEDIUM",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        summary = summarize_chronicle_for_uplift_council(tile)

        assert summary["council_status"] == "WARN"
        assert summary["recurrence_band"] == "MEDIUM"
        assert summary["invariants_ok"] is True
        assert "rationale" in summary

    def test_summarize_chronicle_for_uplift_council_ok(self) -> None:
        """Verify council summary maps to OK correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
            summarize_chronicle_for_uplift_council,
        )

        tile = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.2,  # Low recurrence
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "HIGH",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        summary = summarize_chronicle_for_uplift_council(tile)

        assert summary["council_status"] == "OK"
        assert summary["recurrence_band"] == "LOW"
        assert summary["invariants_ok"] is True
        assert "rationale" in summary

    def test_summarize_chronicle_for_uplift_council_has_required_fields(self) -> None:
        """Verify council summary has all required fields."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
            summarize_chronicle_for_uplift_council,
        )

        tile = build_chronicle_governance_tile(
            recurrence_projection={
                "recurrence_likelihood": 0.3,
                "drivers": [],
                "projected_recurrence_horizon": 30,
                "neutral_explanation": "Test",
            },
            invariant_check={
                "invariant_status": "OK",
                "broken_invariants": [],
                "explanations": [],
            },
            stability_scores={
                "stability_band": "HIGH",
                "axes_contributing": [],
                "headline": "Test",
            },
        )
        summary = summarize_chronicle_for_uplift_council(tile)

        required_fields = ["council_status", "invariants_ok", "recurrence_band", "rationale"]
        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"

        # Verify council_status is valid
        assert summary["council_status"] in ("OK", "WARN", "BLOCK")

    # ========================================================================
    # First Light Recurrence Annex Tests
    # ========================================================================
    # The recurrence annex is designed to answer one specific question for
    # reviewers: "How likely is it that this class of failure will recur if
    # you rerun First Light?"
    #
    # The annex provides:
    # - recurrence_likelihood: Continuous probability [0.0, 1.0]
    # - band: Discrete classification (LOW/MEDIUM/HIGH)
    # - invariants_ok: Safety latch indicating structural integrity
    #
    # This helps reviewers distinguish between:
    # - Isolated incidents (low recurrence, invariants intact)
    # - Chronic patterns (high recurrence, may or may not violate invariants)
    # - Structural integrity concerns (invariants violated, regardless of recurrence)
    # ========================================================================

    def test_build_first_light_chronicle_annex(self) -> None:
        """Verify First Light chronicle annex is built correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_first_light_chronicle_annex,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.6,
            "drivers": ["Test driver"],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }

        annex = build_first_light_chronicle_annex(recurrence_projection, invariant_check)

        # Verify required fields
        assert "schema_version" in annex
        assert annex["schema_version"] == "1.0.0"
        assert "recurrence_likelihood" in annex
        assert "band" in annex
        assert "invariants_ok" in annex

        # Verify values
        assert annex["recurrence_likelihood"] == 0.6
        assert annex["band"] == "MEDIUM"  # 0.6 is in MEDIUM range
        assert annex["invariants_ok"] is True

    def test_build_first_light_chronicle_annex_band_mapping(self) -> None:
        """Verify annex band mapping (LOW/MEDIUM/HIGH) is correct."""
        from backend.health.chronicle_governance_adapter import (
            build_first_light_chronicle_annex,
        )

        # Test HIGH band
        annex_high = build_first_light_chronicle_annex(
            recurrence_projection={"recurrence_likelihood": 0.8},
            invariant_check={"invariant_status": "OK"},
        )
        assert annex_high["band"] == "HIGH"

        # Test MEDIUM band
        annex_medium = build_first_light_chronicle_annex(
            recurrence_projection={"recurrence_likelihood": 0.5},
            invariant_check={"invariant_status": "OK"},
        )
        assert annex_medium["band"] == "MEDIUM"

        # Test LOW band
        annex_low = build_first_light_chronicle_annex(
            recurrence_projection={"recurrence_likelihood": 0.2},
            invariant_check={"invariant_status": "OK"},
        )
        assert annex_low["band"] == "LOW"

    def test_build_first_light_chronicle_annex_invariants_ok(self) -> None:
        """Verify annex invariants_ok mapping is correct."""
        from backend.health.chronicle_governance_adapter import (
            build_first_light_chronicle_annex,
        )

        # Test invariants OK
        annex_ok = build_first_light_chronicle_annex(
            recurrence_projection={"recurrence_likelihood": 0.3},
            invariant_check={"invariant_status": "OK"},
        )
        assert annex_ok["invariants_ok"] is True

        # Test invariants violated
        annex_violated = build_first_light_chronicle_annex(
            recurrence_projection={"recurrence_likelihood": 0.3},
            invariant_check={"invariant_status": "VIOLATED"},
        )
        assert annex_violated["invariants_ok"] is False

    def test_first_light_annex_attached_to_evidence(self) -> None:
        """Verify First Light annex is attached when projection and check provided."""
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.4,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence = {"evidence_type": "test"}
        result = attach_chronicle_governance_to_evidence(
            evidence, tile, drift_signal,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        # Verify annex is present
        assert "first_light_annex" in result["governance"]["chronicle"]
        annex = result["governance"]["chronicle"]["first_light_annex"]
        assert annex["schema_version"] == "1.0.0"
        assert annex["recurrence_likelihood"] == 0.4
        assert annex["band"] == "MEDIUM"
        assert annex["invariants_ok"] is True

    def test_first_light_annex_not_attached_when_missing(self) -> None:
        """Verify First Light annex is NOT attached when projection/check not provided."""
        from backend.health.chronicle_governance_adapter import (
            attach_chronicle_governance_to_evidence,
            build_chronicle_governance_tile,
            extract_chronicle_drift_signal,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.3,
            "drivers": [],
            "projected_recurrence_horizon": 30,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "HIGH",
            "axes_contributing": [],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )
        drift_signal = extract_chronicle_drift_signal(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
        )

        evidence = {"evidence_type": "test"}
        result = attach_chronicle_governance_to_evidence(
            evidence, tile, drift_signal,
            # Not providing recurrence_projection and invariant_check
        )

        # Verify annex is NOT present
        assert "first_light_annex" not in result["governance"]["chronicle"]

    def test_first_light_annex_deterministic(self) -> None:
        """Verify First Light annex is deterministic across runs."""
        from backend.health.chronicle_governance_adapter import (
            build_first_light_chronicle_annex,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.5,
            "drivers": [],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }

        annex1 = build_first_light_chronicle_annex(recurrence_projection, invariant_check)
        annex2 = build_first_light_chronicle_annex(recurrence_projection, invariant_check)

        # Should be identical
        assert annex1 == annex2

        # JSON serialization should also be identical
        json1 = json.dumps(annex1, sort_keys=True)
        json2 = json.dumps(annex2, sort_keys=True)
        assert json1 == json2


class TestChronicleRiskRegister(unittest.TestCase):
    """
    Tests for CAL-EXP-level chronicle risk register.

    SHADOW MODE: These tests verify risk register aggregation and evidence attachment.
    """

    def test_build_cal_exp_recurrence_snapshot(self) -> None:
        """Verify calibration experiment recurrence snapshot is built correctly."""
        import tempfile
        from pathlib import Path
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
        )

        annex = {
            "schema_version": "1.0.0",
            "recurrence_likelihood": 0.6,
            "band": "MEDIUM",
            "invariants_ok": True,
        }

        snapshot = build_cal_exp_recurrence_snapshot("cal_001", annex)

        # Verify required fields
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == "1.0.0"
        assert "cal_id" in snapshot
        assert snapshot["cal_id"] == "cal_001"
        assert "recurrence_likelihood" in snapshot
        assert "band" in snapshot
        assert "invariants_ok" in snapshot
        assert "timestamp" in snapshot

        # Verify values
        assert snapshot["recurrence_likelihood"] == 0.6
        assert snapshot["band"] == "MEDIUM"
        assert snapshot["invariants_ok"] is True

    def test_build_cal_exp_recurrence_snapshot_emits_json(self) -> None:
        """Verify snapshot emits JSON file when output_dir provided."""
        import tempfile
        import json
        from pathlib import Path
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
        )

        annex = {
            "schema_version": "1.0.0",
            "recurrence_likelihood": 0.5,
            "band": "MEDIUM",
            "invariants_ok": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = build_cal_exp_recurrence_snapshot("cal_002", annex, output_dir=tmpdir)

            # Verify file was created
            file_path = Path(tmpdir) / "chronicle_recurrence_cal_002.json"
            assert file_path.exists()

            # Verify file contents
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            assert file_data["cal_id"] == "cal_002"
            assert file_data["recurrence_likelihood"] == 0.5
            assert file_data["band"] == "MEDIUM"

    def test_build_chronicle_risk_register(self) -> None:
        """Verify chronicle risk register is built correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Create test snapshots
        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_001", {
                "recurrence_likelihood": 0.2,
                "band": "LOW",
                "invariants_ok": True,
            }),
            build_cal_exp_recurrence_snapshot("cal_002", {
                "recurrence_likelihood": 0.5,
                "band": "MEDIUM",
                "invariants_ok": True,
            }),
            build_cal_exp_recurrence_snapshot("cal_003", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": False,  # High risk
            }),
            build_cal_exp_recurrence_snapshot("cal_004", {
                "recurrence_likelihood": 0.75,
                "band": "HIGH",
                "invariants_ok": True,  # High but invariants intact
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        # Verify required fields
        assert "schema_version" in register
        assert "total_calibrations" in register
        assert "band_counts" in register
        assert "high_risk_calibrations" in register
        assert "risk_summary" in register

        # Verify values
        assert register["total_calibrations"] == 4
        assert register["band_counts"]["LOW"] == 1
        assert register["band_counts"]["MEDIUM"] == 1
        assert register["band_counts"]["HIGH"] == 2
        assert "cal_003" in register["high_risk_calibrations"]
        assert "cal_004" not in register["high_risk_calibrations"]  # Invariants intact

    def test_build_chronicle_risk_register_empty(self) -> None:
        """Verify risk register handles empty snapshot list."""
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_risk_register,
        )

        register = build_chronicle_risk_register([])

        assert register["total_calibrations"] == 0
        assert register["band_counts"]["LOW"] == 0
        assert register["band_counts"]["MEDIUM"] == 0
        assert register["band_counts"]["HIGH"] == 0
        assert len(register["high_risk_calibrations"]) == 0
        assert "No calibration experiments" in register["risk_summary"]

    def test_build_chronicle_risk_register_classification(self) -> None:
        """Verify risk register correctly classifies high-risk calibrations."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Only HIGH + invariants_ok=False should be high risk
        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_high_violated", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": False,  # High risk
            }),
            build_cal_exp_recurrence_snapshot("cal_high_ok", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": True,  # Not high risk (invariants intact)
            }),
            build_cal_exp_recurrence_snapshot("cal_medium_violated", {
                "recurrence_likelihood": 0.5,
                "band": "MEDIUM",
                "invariants_ok": False,  # Not high risk (not HIGH band)
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        # Only cal_high_violated should be in high_risk_calibrations
        assert len(register["high_risk_calibrations"]) == 1
        assert "cal_high_violated" in register["high_risk_calibrations"]
        assert "cal_high_ok" not in register["high_risk_calibrations"]
        assert "cal_medium_violated" not in register["high_risk_calibrations"]

    def test_build_chronicle_risk_register_json_safe(self) -> None:
        """Verify risk register is JSON serializable."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_001", {
                "recurrence_likelihood": 0.3,
                "band": "LOW",
                "invariants_ok": True,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        # Should serialize without error
        json_str = json.dumps(register)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert "total_calibrations" in parsed
        assert parsed["total_calibrations"] == 1

    def test_attach_chronicle_risk_register_to_evidence(self) -> None:
        """Verify risk register attaches to evidence pack correctly."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
            attach_chronicle_risk_register_to_evidence,
        )

        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_001", {
                "recurrence_likelihood": 0.3,
                "band": "LOW",
                "invariants_ok": True,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)
        evidence = {"evidence_type": "test"}

        result = attach_chronicle_risk_register_to_evidence(evidence, register)

        # Verify structure
        assert "governance" in result
        assert "chronicle_risk_register" in result["governance"]

        risk_reg = result["governance"]["chronicle_risk_register"]
        assert "total_calibrations" in risk_reg
        assert "band_counts" in risk_reg
        assert "high_risk_calibrations" in risk_reg
        assert "risk_summary" in risk_reg

        # Verify evidence is modified in-place
        assert result is evidence

    def test_attach_chronicle_risk_register_non_mutation(self) -> None:
        """Verify risk register attachment doesn't mutate original register."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
            attach_chronicle_risk_register_to_evidence,
        )

        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_001", {
                "recurrence_likelihood": 0.3,
                "band": "LOW",
                "invariants_ok": True,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)
        register_copy = register.copy()
        evidence = {"evidence_type": "test"}

        attach_chronicle_risk_register_to_evidence(evidence, register)

        # Register should not be mutated
        assert register == register_copy

    def test_build_chronicle_risk_register_high_risk_details(self) -> None:
        """Verify high_risk_details list is included and deterministically ordered."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Create snapshots with high-risk calibrations in non-sorted order
        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_zebra", {
                "recurrence_likelihood": 0.9,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_alpha", {
                "recurrence_likelihood": 0.85,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_beta", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": False,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        # Verify high_risk_details exists
        assert "high_risk_details" in register
        details = register["high_risk_details"]
        assert len(details) == 3

        # Verify deterministic ordering (sorted by cal_id)
        cal_ids = [d["cal_id"] for d in details]
        assert cal_ids == ["cal_alpha", "cal_beta", "cal_zebra"]

        # Verify each detail entry has required fields
        for detail in details:
            assert "cal_id" in detail
            assert "recurrence_likelihood" in detail
            assert "invariants_ok" in detail
            assert "evidence_path_hint" in detail
            assert isinstance(detail["recurrence_likelihood"], float)
            assert isinstance(detail["invariants_ok"], bool)
            assert detail["invariants_ok"] is False  # All high-risk have invariants violated

    def test_build_chronicle_risk_register_evidence_path_hint(self) -> None:
        """Verify evidence_path_hint format is correct."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_test_001", {
                "recurrence_likelihood": 0.75,
                "band": "HIGH",
                "invariants_ok": False,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        details = register["high_risk_details"]
        assert len(details) == 1

        detail = details[0]
        expected_path = "calibration/chronicle_recurrence_cal_test_001.json"
        assert detail["evidence_path_hint"] == expected_path

    def test_build_chronicle_risk_register_high_risk_details_empty(self) -> None:
        """Verify high_risk_details is empty list when no high-risk calibrations."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        snapshots = [
            build_cal_exp_recurrence_snapshot("cal_001", {
                "recurrence_likelihood": 0.3,
                "band": "LOW",
                "invariants_ok": True,
            }),
        ]

        register = build_chronicle_risk_register(snapshots)

        assert "high_risk_details" in register
        assert register["high_risk_details"] == []

    def test_build_chronicle_risk_register_deterministic_ordering(self) -> None:
        """Verify high_risk_details ordering is deterministic across multiple calls."""
        from backend.health.chronicle_governance_adapter import (
            build_cal_exp_recurrence_snapshot,
            build_chronicle_risk_register,
        )

        # Create snapshots in different orders
        snapshots1 = [
            build_cal_exp_recurrence_snapshot("cal_c", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_a", {
                "recurrence_likelihood": 0.9,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_b", {
                "recurrence_likelihood": 0.85,
                "band": "HIGH",
                "invariants_ok": False,
            }),
        ]

        snapshots2 = [
            build_cal_exp_recurrence_snapshot("cal_b", {
                "recurrence_likelihood": 0.85,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_a", {
                "recurrence_likelihood": 0.9,
                "band": "HIGH",
                "invariants_ok": False,
            }),
            build_cal_exp_recurrence_snapshot("cal_c", {
                "recurrence_likelihood": 0.8,
                "band": "HIGH",
                "invariants_ok": False,
            }),
        ]

        register1 = build_chronicle_risk_register(snapshots1)
        register2 = build_chronicle_risk_register(snapshots2)

        # Details should be in same order regardless of input order
        details1 = register1["high_risk_details"]
        details2 = register2["high_risk_details"]

        cal_ids1 = [d["cal_id"] for d in details1]
        cal_ids2 = [d["cal_id"] for d in details2]

        assert cal_ids1 == cal_ids2 == ["cal_a", "cal_b", "cal_c"]

    def test_attach_chronicle_governance_to_calibration_report(self) -> None:
        """Verify chronicle governance attaches to calibration report correctly."""
        from backend.health.chronicle_calibration_binding import (
            attach_chronicle_governance_to_calibration_report,
        )
        from backend.health.chronicle_governance_adapter import (
            build_chronicle_governance_tile,
        )

        recurrence_projection = {
            "recurrence_likelihood": 0.5,
            "drivers": ["Test driver"],
            "projected_recurrence_horizon": 22,
            "neutral_explanation": "Test",
        }
        invariant_check = {
            "invariant_status": "OK",
            "broken_invariants": [],
            "explanations": [],
        }
        stability_scores = {
            "stability_band": "MEDIUM",
            "axes_contributing": ["alignment"],
            "headline": "Test",
        }

        tile = build_chronicle_governance_tile(
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            stability_scores=stability_scores,
        )

        calibration_report = {
            "schema_version": "1.0.0",
            "run_id": "test_run",
            "timing": {"start_time": "2025-01-01T00:00:00Z", "end_time": "2025-01-01T01:00:00Z", "cycles_observed": 100},
        }

        result = attach_chronicle_governance_to_calibration_report(
            calibration_report=calibration_report,
            recurrence_projection=recurrence_projection,
            invariant_check=invariant_check,
            tile=tile,
        )

        # Verify chronicle_governance section added
        assert "chronicle_governance" in result
        chronicle = result["chronicle_governance"]
        
        # Verify required fields
        assert "recurrence_likelihood" in chronicle
        assert "band" in chronicle
        assert "invariants_ok" in chronicle
        assert "drift_notes" in chronicle

        # Verify values
        assert chronicle["recurrence_likelihood"] == 0.5
        assert chronicle["band"] == "MEDIUM"
        assert chronicle["invariants_ok"] is True
        assert isinstance(chronicle["drift_notes"], list)

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

