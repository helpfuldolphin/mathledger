"""
Tests for Phase X: Drift Governance P4 Integration.

This test suite verifies:
  - Evidence attachment with drift governance
  - Council summary generation
  - Calibration report section building
  - P4 drift signal extraction (with poly_cause_detected)

All tests are marked as unit tests and do not require external dependencies.
"""

import json
from typing import Any, Dict

import pytest

import tempfile
from pathlib import Path

from backend.health.drift_tensor_adapter import (
    attach_drift_cluster_view_to_evidence,
    attach_drift_governance_to_evidence,
    build_drift_cluster_view,
    build_drift_governance_for_calibration_report,
    build_first_light_scenario_drift_map,
    emit_cal_exp_scenario_drift_map,
    extract_drift_signal_for_shadow,
    extract_scenario_drift_cluster_signal,
    summarize_drift_for_uplift_council,
)


class TestEvidenceAttachment:
    """Tests for attach_drift_governance_to_evidence function."""

    @pytest.mark.unit
    def test_attach_drift_governance_to_evidence_structure(self):
        """Evidence attachment has correct structure."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        governance_tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a"],
        }

        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        assert "governance" in enriched
        assert "drift" in enriched["governance"]

        drift_data = enriched["governance"]["drift"]
        assert "drift_band" in drift_data
        assert "tensor_norm" in drift_data
        assert "poly_cause_detected" in drift_data
        assert "highlighted_cases" in drift_data

    @pytest.mark.unit
    def test_attach_drift_governance_non_mutating(self):
        """Evidence attachment does not mutate input."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        governance_tile = {
            "status_light": "GREEN",
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "slices_with_poly_cause_drift": [],
            "headline": "No drift issues detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "poly_cause_detected": False,
            "slices_with_poly_cause_drift": [],
        }

        original_evidence = evidence.copy()
        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        # Original should be unchanged
        assert evidence == original_evidence
        assert "governance" not in evidence

        # Enriched should have governance
        assert "governance" in enriched
        assert enriched != evidence

    @pytest.mark.unit
    def test_attach_drift_governance_existing_governance(self):
        """Evidence attachment preserves existing governance data."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "governance": {
                "semantic_tda": {"status": "OK"},
            },
        }

        governance_tile = {
            "status_light": "GREEN",
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "slices_with_poly_cause_drift": [],
            "headline": "No drift issues detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "poly_cause_detected": False,
            "slices_with_poly_cause_drift": [],
        }

        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        assert "semantic_tda" in enriched["governance"]
        assert "drift" in enriched["governance"]
        assert enriched["governance"]["semantic_tda"]["status"] == "OK"

    @pytest.mark.unit
    def test_attach_drift_governance_serializable(self):
        """Enriched evidence is JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        governance_tile = {
            "status_light": "RED",
            "global_tensor_norm": 1.5,
            "risk_band": "HIGH",
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
            "headline": "High drift detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 1.5,
            "risk_band": "HIGH",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
        }

        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        # Should serialize without error
        json_str = json.dumps(enriched)
        assert len(json_str) > 0

        # Should round-trip
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "drift" in parsed["governance"]


class TestCouncilSummary:
    """Tests for summarize_drift_for_uplift_council function."""

    @pytest.mark.unit
    def test_summarize_drift_for_uplift_council_structure(self):
        """Council summary has correct structure."""
        tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.5,
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        summary = summarize_drift_for_uplift_council(tile)

        assert "status" in summary
        assert "drift_band" in summary
        assert "poly_cause_detected" in summary
        assert "implicated_slices" in summary
        assert "implicated_axes" in summary
        assert "summary" in summary

        assert summary["status"] in ("OK", "WARN", "BLOCK")
        assert summary["drift_band"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(summary["poly_cause_detected"], bool)
        assert isinstance(summary["implicated_slices"], list)

    @pytest.mark.unit
    def test_summarize_drift_ok_status(self):
        """Council summary returns OK for low drift."""
        tile = {
            "status_light": "GREEN",
            "global_tensor_norm": 0.1,
            "risk_band": "LOW",
            "slices_with_poly_cause_drift": [],
            "headline": "No drift issues detected.",
            "schema_version": "1.0.0",
        }

        summary = summarize_drift_for_uplift_council(tile)

        assert summary["status"] == "OK"
        assert summary["drift_band"] == "LOW"
        assert summary["poly_cause_detected"] is False

    @pytest.mark.unit
    def test_summarize_drift_warn_status(self):
        """Council summary returns WARN for medium drift."""
        tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.4,
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        summary = summarize_drift_for_uplift_council(tile)

        assert summary["status"] == "WARN"
        assert summary["drift_band"] == "MEDIUM"
        assert summary["poly_cause_detected"] is True

    @pytest.mark.unit
    def test_summarize_drift_block_status(self):
        """Council summary returns BLOCK for high drift."""
        tile = {
            "status_light": "RED",
            "global_tensor_norm": 1.5,
            "risk_band": "HIGH",
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
            "headline": "High drift detected.",
            "schema_version": "1.0.0",
        }

        summary = summarize_drift_for_uplift_council(tile)

        assert summary["status"] == "BLOCK"
        assert summary["drift_band"] == "HIGH"
        assert summary["poly_cause_detected"] is True

    @pytest.mark.unit
    def test_summarize_drift_block_medium_with_high_norm(self):
        """Council summary returns BLOCK for medium drift with high tensor norm."""
        tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.6,  # > 0.5 threshold
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        summary = summarize_drift_for_uplift_council(tile)

        assert summary["status"] == "BLOCK"
        assert summary["drift_band"] == "MEDIUM"


class TestCalibrationReport:
    """Tests for build_drift_governance_for_calibration_report function."""

    @pytest.mark.unit
    def test_build_drift_governance_for_calibration_report_structure(self):
        """Calibration report section has correct structure."""
        governance_tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a"],
        }

        report_section = build_drift_governance_for_calibration_report(
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        assert "global_tensor_norm" in report_section
        assert "drift_band" in report_section
        assert "poly_cause_detected" in report_section
        assert "highlighted_cases" in report_section

        assert report_section["drift_band"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(report_section["poly_cause_detected"], bool)
        assert isinstance(report_section["highlighted_cases"], list)

    @pytest.mark.unit
    def test_build_drift_governance_for_calibration_report_serializable(self):
        """Calibration report section is JSON serializable."""
        governance_tile = {
            "status_light": "RED",
            "global_tensor_norm": 1.5,
            "risk_band": "HIGH",
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
            "headline": "High drift detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 1.5,
            "risk_band": "HIGH",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
        }

        report_section = build_drift_governance_for_calibration_report(
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        json_str = json.dumps(report_section)
        assert len(json_str) > 0

        parsed = json.loads(json_str)
        assert parsed == report_section


class TestDriftSignalExtraction:
    """Tests for extract_drift_signal_for_shadow with poly_cause_detected."""

    @pytest.mark.unit
    def test_extract_drift_signal_includes_poly_cause_detected(self):
        """Drift signal includes poly_cause_detected field."""
        tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
            "global_tensor_norm": 0.583,
            "ranked_slices": ["slice_a"],
            "schema_version": "1.0.0",
        }

        poly_cause = {
            "poly_cause_detected": True,
            "cause_vectors": [
                {
                    "slice": "slice_a",
                    "axes": ["drift", "budget"],
                    "drift_scores": {"drift": 0.5, "budget": 0.3},
                }
            ],
            "risk_band": "MEDIUM",
            "notes": ["Slice slice_a: multiple axes show drift (budget, drift)"],
        }

        signal = extract_drift_signal_for_shadow(tensor=tensor, poly_cause=poly_cause)

        assert "poly_cause_detected" in signal
        assert signal["poly_cause_detected"] is True
        assert signal["risk_band"] == "MEDIUM"
        assert "slice_a" in signal["slices_with_poly_cause_drift"]

    @pytest.mark.unit
    def test_extract_drift_signal_no_poly_cause(self):
        """Drift signal correctly reports no poly-cause when none detected."""
        tensor = {
            "tensor": {
                "slice_a": {"drift": 0.2, "budget": 0.0, "metric": 0.0, "semantic": 0.0},
            },
            "global_tensor_norm": 0.2,
            "ranked_slices": ["slice_a"],
            "schema_version": "1.0.0",
        }

        poly_cause = {
            "poly_cause_detected": False,
            "cause_vectors": [],
            "risk_band": "LOW",
            "notes": [],
        }

        signal = extract_drift_signal_for_shadow(tensor=tensor, poly_cause=poly_cause)

        assert signal["poly_cause_detected"] is False
        assert signal["risk_band"] == "LOW"
        assert len(signal["slices_with_poly_cause_drift"]) == 0


class TestScenarioDriftMap:
    """Tests for build_first_light_scenario_drift_map function."""

    @pytest.mark.unit
    def test_build_first_light_scenario_drift_map_structure(self):
        """Scenario drift map has correct structure."""
        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a", "slice_b", "slice_c"],
        }

        scenario_map = build_first_light_scenario_drift_map(drift_signal)

        assert "schema_version" in scenario_map
        assert "risk_band" in scenario_map
        assert "tensor_norm" in scenario_map
        assert "poly_cause_detected" in scenario_map
        assert "slices_with_poly_cause" in scenario_map

        assert scenario_map["schema_version"] == "1.0.0"
        assert scenario_map["risk_band"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(scenario_map["poly_cause_detected"], bool)
        assert isinstance(scenario_map["slices_with_poly_cause"], list)

    @pytest.mark.unit
    def test_build_first_light_scenario_drift_map_limits_to_top_5(self):
        """Scenario drift map limits slices to top 5."""
        drift_signal = {
            "global_tensor_norm": 1.0,
            "risk_band": "HIGH",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": [
                "slice_a",
                "slice_b",
                "slice_c",
                "slice_d",
                "slice_e",
                "slice_f",
                "slice_g",
            ],
        }

        scenario_map = build_first_light_scenario_drift_map(drift_signal)

        assert len(scenario_map["slices_with_poly_cause"]) == 5
        assert "slice_a" in scenario_map["slices_with_poly_cause"]
        assert "slice_f" not in scenario_map["slices_with_poly_cause"]
        assert "slice_g" not in scenario_map["slices_with_poly_cause"]

    @pytest.mark.unit
    def test_build_first_light_scenario_drift_map_sorted(self):
        """Scenario drift map slices are sorted."""
        drift_signal = {
            "global_tensor_norm": 0.5,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_c", "slice_a", "slice_b"],
        }

        scenario_map = build_first_light_scenario_drift_map(drift_signal)

        assert scenario_map["slices_with_poly_cause"] == ["slice_a", "slice_b", "slice_c"]

    @pytest.mark.unit
    def test_build_first_light_scenario_drift_map_json_serializable(self):
        """Scenario drift map is JSON serializable."""
        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
        }

        scenario_map = build_first_light_scenario_drift_map(drift_signal)

        json_str = json.dumps(scenario_map)
        assert len(json_str) > 0

        parsed = json.loads(json_str)
        assert parsed == scenario_map

    @pytest.mark.unit
    def test_build_first_light_scenario_drift_map_deterministic(self):
        """Scenario drift map output is deterministic."""
        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_b", "slice_a"],
        }

        map1 = build_first_light_scenario_drift_map(drift_signal)
        map2 = build_first_light_scenario_drift_map(drift_signal)

        assert map1 == map2

    @pytest.mark.unit
    def test_attach_drift_governance_includes_scenario_map(self):
        """Evidence attachment includes first_light_scenario_map."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        governance_tile = {
            "status_light": "YELLOW",
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
            "headline": "Poly-cause patterns detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.707,
            "risk_band": "MEDIUM",
            "poly_cause_detected": True,
            "slices_with_poly_cause_drift": ["slice_a", "slice_b"],
        }

        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        assert "first_light_scenario_map" in enriched["governance"]["drift"]

        scenario_map = enriched["governance"]["drift"]["first_light_scenario_map"]
        assert "schema_version" in scenario_map
        assert "risk_band" in scenario_map
        assert "tensor_norm" in scenario_map
        assert "poly_cause_detected" in scenario_map
        assert "slices_with_poly_cause" in scenario_map

    @pytest.mark.unit
    def test_attach_drift_governance_scenario_map_non_mutating(self):
        """Evidence attachment with scenario map does not mutate input."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        governance_tile = {
            "status_light": "GREEN",
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "slices_with_poly_cause_drift": [],
            "headline": "No drift issues detected.",
            "schema_version": "1.0.0",
        }

        drift_signal = {
            "global_tensor_norm": 0.0,
            "risk_band": "LOW",
            "poly_cause_detected": False,
            "slices_with_poly_cause_drift": [],
        }

        original_evidence = evidence.copy()
        enriched = attach_drift_governance_to_evidence(
            evidence=evidence,
            governance_tile=governance_tile,
            drift_signal=drift_signal,
        )

        # Original should be unchanged
        assert evidence == original_evidence
        assert "governance" not in evidence

        # Enriched should have scenario map
        assert "first_light_scenario_map" in enriched["governance"]["drift"]


class TestCalExpScenarioDriftMapEmission:
    """Tests for emit_cal_exp_scenario_drift_map function."""

    @pytest.mark.unit
    def test_emit_cal_exp_scenario_drift_map_saves_file(self):
        """Scenario drift map is saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_map = {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.707,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            }

            output_path = emit_cal_exp_scenario_drift_map(
                cal_id="CAL-EXP-1",
                scenario_map=scenario_map,
                output_dir=Path(tmpdir),
            )

            assert output_path.exists()
            assert output_path.name == "scenario_drift_map_CAL-EXP-1.json"

    @pytest.mark.unit
    def test_emit_cal_exp_scenario_drift_map_json_content(self):
        """Saved JSON file contains correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_map = {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 1.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            }

            output_path = emit_cal_exp_scenario_drift_map(
                cal_id="CAL-EXP-2",
                scenario_map=scenario_map,
                output_dir=Path(tmpdir),
            )

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded == scenario_map
            assert loaded["risk_band"] == "HIGH"
            assert loaded["tensor_norm"] == 1.5


class TestDriftClusterView:
    """Tests for build_drift_cluster_view function."""

    @pytest.mark.unit
    def test_build_drift_cluster_view_structure(self):
        """Drift cluster view has correct structure."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_c"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        assert "schema_version" in cluster_view
        assert "slice_frequency" in cluster_view
        assert "high_risk_slices" in cluster_view
        assert "experiments_analyzed" in cluster_view
        assert "persistence_buckets" in cluster_view
        assert "persistence_score" in cluster_view

        assert cluster_view["schema_version"] == "1.0.0"
        assert cluster_view["experiments_analyzed"] == 2
        assert isinstance(cluster_view["slice_frequency"], dict)
        assert isinstance(cluster_view["high_risk_slices"], list)
        assert isinstance(cluster_view["persistence_buckets"], dict)
        assert isinstance(cluster_view["persistence_score"], float)

    @pytest.mark.unit
    def test_build_drift_cluster_view_frequency_counting(self):
        """Drift cluster view correctly counts slice frequency."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_c"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.6,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        # slice_a appears in all 3 experiments
        assert cluster_view["slice_frequency"]["slice_a"] == 3
        # slice_b and slice_c appear in 1 experiment each
        assert cluster_view["slice_frequency"]["slice_b"] == 1
        assert cluster_view["slice_frequency"]["slice_c"] == 1

    @pytest.mark.unit
    def test_build_drift_cluster_view_deterministic_ordering(self):
        """Drift cluster view has deterministic slice ordering."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_b", "slice_a"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_c"],
            },
        ]

        cluster_view1 = build_drift_cluster_view(maps)
        cluster_view2 = build_drift_cluster_view(maps)

        # Should be identical
        assert cluster_view1 == cluster_view2

        # slice_frequency should be sorted by count (desc), then name (asc)
        frequency_items = list(cluster_view1["slice_frequency"].items())
        assert frequency_items[0][0] == "slice_a"  # Count 2, first alphabetically
        assert frequency_items[0][1] == 2

    @pytest.mark.unit
    def test_build_drift_cluster_view_top_n_slices(self):
        """Drift cluster view limits high_risk_slices to top N."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": [f"slice_{i}" for i in range(10)],
            },
        ]

        cluster_view = build_drift_cluster_view(maps, top_n=5)

        assert len(cluster_view["high_risk_slices"]) == 5

    @pytest.mark.unit
    def test_build_drift_cluster_view_json_serializable(self):
        """Drift cluster view is JSON serializable."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        json_str = json.dumps(cluster_view)
        assert len(json_str) > 0

        parsed = json.loads(json_str)
        assert parsed == cluster_view

    @pytest.mark.unit
    def test_build_drift_cluster_view_empty_maps(self):
        """Drift cluster view handles empty maps list."""
        cluster_view = build_drift_cluster_view([])

        assert cluster_view["experiments_analyzed"] == 0
        assert cluster_view["slice_frequency"] == {}
        assert cluster_view["high_risk_slices"] == []


class TestDriftClusterViewEvidenceAttachment:
    """Tests for attach_drift_cluster_view_to_evidence function."""

    @pytest.mark.unit
    def test_attach_drift_cluster_view_to_evidence_structure(self):
        """Cluster view attachment has correct structure."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 3, "slice_b": 1},
            "high_risk_slices": ["slice_a", "slice_b"],
            "experiments_analyzed": 2,
        }

        enriched = attach_drift_cluster_view_to_evidence(
            evidence=evidence,
            cluster_view=cluster_view,
        )

        assert "governance" in enriched
        assert "scenario_drift_cluster_view" in enriched["governance"]

        attached_view = enriched["governance"]["scenario_drift_cluster_view"]
        assert attached_view == cluster_view

    @pytest.mark.unit
    def test_attach_drift_cluster_view_non_mutating(self):
        """Cluster view attachment does not mutate input."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 2},
            "high_risk_slices": ["slice_a"],
            "experiments_analyzed": 1,
        }

        original_evidence = evidence.copy()
        enriched = attach_drift_cluster_view_to_evidence(
            evidence=evidence,
            cluster_view=cluster_view,
        )

        # Original should be unchanged
        assert evidence == original_evidence
        assert "governance" not in evidence

        # Enriched should have cluster view
        assert "scenario_drift_cluster_view" in enriched["governance"]

    @pytest.mark.unit
    def test_attach_drift_cluster_view_preserves_existing_governance(self):
        """Cluster view attachment preserves existing governance data."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "governance": {
                "drift": {"drift_band": "MEDIUM"},
            },
        }

        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 1},
            "high_risk_slices": ["slice_a"],
            "experiments_analyzed": 1,
        }

        enriched = attach_drift_cluster_view_to_evidence(
            evidence=evidence,
            cluster_view=cluster_view,
        )

        assert "drift" in enriched["governance"]
        assert "scenario_drift_cluster_view" in enriched["governance"]
        assert enriched["governance"]["drift"]["drift_band"] == "MEDIUM"

    @pytest.mark.unit
    def test_attach_drift_cluster_view_serializable(self):
        """Enriched evidence with cluster view is JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"test": "value"},
        }

        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 3, "slice_b": 2},
            "high_risk_slices": ["slice_a", "slice_b"],
            "experiments_analyzed": 3,
        }

        enriched = attach_drift_cluster_view_to_evidence(
            evidence=evidence,
            cluster_view=cluster_view,
        )

        json_str = json.dumps(enriched)
        assert len(json_str) > 0

        parsed = json.loads(json_str)
        assert "scenario_drift_cluster_view" in parsed["governance"]


class TestDriftClusterViewPersistence:
    """Tests for persistence metrics in build_drift_cluster_view."""

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_buckets(self):
        """Persistence buckets correctly assign slices to appears_in_1/2/3."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_c"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.6,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        assert "persistence_buckets" in cluster_view
        buckets = cluster_view["persistence_buckets"]
        assert "appears_in_1" in buckets
        assert "appears_in_2" in buckets
        assert "appears_in_3" in buckets

        # slice_a appears in all 3 experiments
        assert "slice_a" in buckets["appears_in_3"]
        # slice_b and slice_c appear in 1 experiment each
        assert "slice_b" in buckets["appears_in_1"]
        assert "slice_c" in buckets["appears_in_1"]

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_buckets_sorted(self):
        """Persistence buckets are sorted for determinism."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_z", "slice_a"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_m"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        buckets = cluster_view["persistence_buckets"]
        # All buckets should be sorted
        assert buckets["appears_in_1"] == sorted(buckets["appears_in_1"])
        assert buckets["appears_in_2"] == sorted(buckets["appears_in_2"])
        assert buckets["appears_in_3"] == sorted(buckets["appears_in_3"])

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_score(self):
        """Persistence score is computed correctly."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.6,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        assert "persistence_score" in cluster_view
        # slice_a appears in all 3 experiments, so persistence_score = 3/3 = 1.0
        assert cluster_view["persistence_score"] == 1.0

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_score_mixed(self):
        """Persistence score correctly averages across slices."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a"],
            },
        ]

        cluster_view = build_drift_cluster_view(maps)

        # slice_a appears in 2/2 experiments = 1.0
        # slice_b appears in 1/2 experiments = 0.5
        # Mean = (1.0 + 0.5) / 2 = 0.75
        assert cluster_view["persistence_score"] == 0.75

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_score_deterministic(self):
        """Persistence score is deterministic across invocations."""
        maps = [
            {
                "schema_version": "1.0.0",
                "risk_band": "MEDIUM",
                "tensor_norm": 0.5,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_b"],
            },
            {
                "schema_version": "1.0.0",
                "risk_band": "HIGH",
                "tensor_norm": 0.8,
                "poly_cause_detected": True,
                "slices_with_poly_cause": ["slice_a", "slice_c"],
            },
        ]

        cluster_view1 = build_drift_cluster_view(maps)
        cluster_view2 = build_drift_cluster_view(maps)

        assert cluster_view1["persistence_score"] == cluster_view2["persistence_score"]

    @pytest.mark.unit
    def test_build_drift_cluster_view_persistence_score_empty(self):
        """Persistence score is 0.0 when no slices."""
        cluster_view = build_drift_cluster_view([])

        assert cluster_view["persistence_score"] == 0.0
        assert cluster_view["persistence_buckets"]["appears_in_1"] == []
        assert cluster_view["persistence_buckets"]["appears_in_2"] == []
        assert cluster_view["persistence_buckets"]["appears_in_3"] == []


class TestScenarioDriftClusterSignal:
    """Tests for extract_scenario_drift_cluster_signal function."""

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_structure(self):
        """Signal extraction has correct structure."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 3, "slice_b": 2},
            "high_risk_slices": ["slice_a", "slice_b"],
            "experiments_analyzed": 3,
            "persistence_buckets": {
                "appears_in_1": [],
                "appears_in_2": ["slice_b"],
                "appears_in_3": ["slice_a"],
            },
            "persistence_score": 0.833333,
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert "schema_version" in signal
        assert "mode" in signal
        assert "experiments_analyzed" in signal
        assert "high_risk_slices" in signal
        assert "persistence_score" in signal
        assert "drivers" in signal

        assert signal["schema_version"] == "1.0.0"
        assert signal["mode"] == "SHADOW"
        assert signal["experiments_analyzed"] == 3
        assert signal["high_risk_slices"] == ["slice_a", "slice_b"]
        assert signal["persistence_score"] == 0.833333
        assert isinstance(signal["drivers"], list)

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_defaults(self):
        """Signal extraction handles missing fields with defaults."""
        cluster_view = {}

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert signal["schema_version"] == "1.0.0"
        assert signal["mode"] == "SHADOW"
        assert signal["experiments_analyzed"] == 0
        assert signal["high_risk_slices"] == []
        assert signal["persistence_score"] == 0.0
        assert signal["drivers"] == []

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_drivers_persistence_high(self):
        """Signal includes DRIVER_PERSISTENCE_SCORE_HIGH when persistence_score >= 0.5."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 2},
            "high_risk_slices": [],
            "experiments_analyzed": 3,
            "persistence_buckets": {
                "appears_in_1": [],
                "appears_in_2": ["slice_a"],
                "appears_in_3": [],
            },
            "persistence_score": 0.666667,  # >= 0.5
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert "DRIVER_PERSISTENCE_SCORE_HIGH" in signal["drivers"]
        assert len(signal["drivers"]) == 1

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_drivers_high_risk_slices(self):
        """Signal includes DRIVER_HIGH_RISK_SLICES_PRESENT when slices present."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 1},
            "high_risk_slices": ["slice_a"],
            "experiments_analyzed": 1,
            "persistence_buckets": {
                "appears_in_1": ["slice_a"],
                "appears_in_2": [],
                "appears_in_3": [],
            },
            "persistence_score": 0.333333,  # < 0.5
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert "DRIVER_HIGH_RISK_SLICES_PRESENT" in signal["drivers"]
        assert len(signal["drivers"]) == 1

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_drivers_both(self):
        """Signal includes both drivers when both conditions are true."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 3},
            "high_risk_slices": ["slice_a"],
            "experiments_analyzed": 3,
            "persistence_buckets": {
                "appears_in_1": [],
                "appears_in_2": [],
                "appears_in_3": ["slice_a"],
            },
            "persistence_score": 1.0,  # >= 0.5
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert len(signal["drivers"]) == 2
        # Deterministic ordering: persistence first, slices second
        assert signal["drivers"][0] == "DRIVER_PERSISTENCE_SCORE_HIGH"
        assert signal["drivers"][1] == "DRIVER_HIGH_RISK_SLICES_PRESENT"

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_drivers_none(self):
        """Signal has empty drivers when neither condition is true."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 1},
            "high_risk_slices": [],
            "experiments_analyzed": 3,
            "persistence_buckets": {
                "appears_in_1": ["slice_a"],
                "appears_in_2": [],
                "appears_in_3": [],
            },
            "persistence_score": 0.333333,  # < 0.5
            "extraction_source": "MANIFEST",
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert signal["drivers"] == []

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_drivers_exact_codes(self):
        """Signal drivers contain only the two valid driver codes."""
        # Test all combinations
        test_cases = [
            {
                "persistence_score": 0.6,  # >= 0.5
                "high_risk_slices": [],
                "expected_drivers": ["DRIVER_PERSISTENCE_SCORE_HIGH"],
            },
            {
                "persistence_score": 0.3,  # < 0.5
                "high_risk_slices": ["slice_a"],
                "expected_drivers": ["DRIVER_HIGH_RISK_SLICES_PRESENT"],
            },
            {
                "persistence_score": 0.7,  # >= 0.5
                "high_risk_slices": ["slice_a", "slice_b"],
                "expected_drivers": [
                    "DRIVER_PERSISTENCE_SCORE_HIGH",
                    "DRIVER_HIGH_RISK_SLICES_PRESENT",
                ],
            },
            {
                "persistence_score": 0.2,  # < 0.5
                "high_risk_slices": [],
                "expected_drivers": [],
            },
        ]

        for case in test_cases:
            cluster_view = {
                "schema_version": "1.0.0",
                "slice_frequency": {"slice_a": 1},
                "high_risk_slices": case["high_risk_slices"],
                "experiments_analyzed": 3,
                "persistence_buckets": {
                    "appears_in_1": ["slice_a"],
                    "appears_in_2": [],
                    "appears_in_3": [],
                },
                "persistence_score": case["persistence_score"],
                "extraction_source": "MANIFEST",
            }

            signal = extract_scenario_drift_cluster_signal(cluster_view)

            assert signal["drivers"] == case["expected_drivers"]
            # Verify only valid driver codes are present
            valid_drivers = {
                "DRIVER_PERSISTENCE_SCORE_HIGH",
                "DRIVER_HIGH_RISK_SLICES_PRESENT",
            }
            assert all(driver in valid_drivers for driver in signal["drivers"])

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_extraction_source(self):
        """Signal includes extraction_source for provenance tracking."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 2},
            "high_risk_slices": ["slice_a"],
            "experiments_analyzed": 2,
            "persistence_buckets": {
                "appears_in_1": [],
                "appears_in_2": ["slice_a"],
                "appears_in_3": [],
            },
            "persistence_score": 1.0,
            "extraction_source": "MANIFEST",
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert "extraction_source" in signal
        assert signal["extraction_source"] == "MANIFEST"

    @pytest.mark.unit
    def test_extract_scenario_drift_cluster_signal_extraction_source_defaults(self):
        """Signal defaults extraction_source to MISSING if not provided."""
        cluster_view = {
            "schema_version": "1.0.0",
            "slice_frequency": {"slice_a": 1},
            "high_risk_slices": [],
            "experiments_analyzed": 1,
            "persistence_buckets": {
                "appears_in_1": ["slice_a"],
                "appears_in_2": [],
                "appears_in_3": [],
            },
            "persistence_score": 0.5,
        }

        signal = extract_scenario_drift_cluster_signal(cluster_view)

        assert signal["extraction_source"] == "MISSING"

