"""
Phase X: CI Tests for Semantic Drift Integration Functions

Tests for First-Light (P3), P4, Evidence, and Council integrations.

SHADOW MODE CONTRACT:
- These tests verify serialization and structural stability only
- No governance logic is tested
- No drift computation or real data processing is performed
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


class TestSemanticDriftP3Integration:
    """Tests for First-Light P3 integration."""

    def test_build_semantic_drift_summary_for_p3_has_required_fields(self) -> None:
        """Verify P3 summary contains all required fields."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_summary_for_p3,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a", "slice_b"],
            "projected_instability_count": 2,
            "gating_recommendation": "WARN",
            "recommendation_reasons": [],
            "headline": "Test headline.",
        }

        summary = build_semantic_drift_summary_for_p3(tile)

        required_fields = [
            "tensor_norm",
            "semantic_hotspots",
            "projected_instability_count",
            "status_light",
            "gating_recommendation",
        ]

        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"

        assert summary["tensor_norm"] == 1.5
        assert summary["semantic_hotspots"] == ["slice_a", "slice_b"]
        assert summary["projected_instability_count"] == 2
        assert summary["status_light"] == "YELLOW"
        assert summary["gating_recommendation"] == "WARN"

    def test_p3_summary_serializes(self) -> None:
        """Verify P3 summary is JSON-serializable."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_summary_for_p3,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = build_semantic_drift_summary_for_p3(tile)

        json_str = json.dumps(summary)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSemanticDriftP4Integration:
    """Tests for P4 calibration report integration."""

    def test_build_semantic_drift_calibration_for_p4_has_required_fields(self) -> None:
        """Verify P4 calibration contains all required fields."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_calibration_for_p4,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "RED",
            "tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "BLOCK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        calibration = build_semantic_drift_calibration_for_p4(tile)

        required_fields = [
            "tensor_norm",
            "hotspots",
            "regression_status",
            "projected_instability_count",
        ]

        for field in required_fields:
            assert field in calibration, f"Missing required field: {field}"

        assert calibration["tensor_norm"] == 2.0
        assert calibration["hotspots"] == ["slice_a"]
        assert calibration["regression_status"] == "REGRESSED"
        assert calibration["projected_instability_count"] == 1

    def test_p4_regression_status_mapping(self) -> None:
        """Verify regression_status maps correctly from gating_recommendation."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_calibration_for_p4,
        )

        # Test BLOCK → REGRESSED
        tile_block = {
            "schema_version": "1.0.0",
            "status_light": "RED",
            "tensor_norm": 2.0,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "BLOCK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }
        calibration_block = build_semantic_drift_calibration_for_p4(tile_block)
        assert calibration_block["regression_status"] == "REGRESSED"

        # Test WARN → ATTENTION
        tile_warn = tile_block.copy()
        tile_warn["gating_recommendation"] = "WARN"
        calibration_warn = build_semantic_drift_calibration_for_p4(tile_warn)
        assert calibration_warn["regression_status"] == "ATTENTION"

        # Test OK → STABLE
        tile_ok = tile_block.copy()
        tile_ok["gating_recommendation"] = "OK"
        calibration_ok = build_semantic_drift_calibration_for_p4(tile_ok)
        assert calibration_ok["regression_status"] == "STABLE"

    def test_p4_calibration_serializes(self) -> None:
        """Verify P4 calibration is JSON-serializable."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_calibration_for_p4,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        calibration = build_semantic_drift_calibration_for_p4(tile)

        json_str = json.dumps(calibration)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSemanticDriftEvidenceIntegration:
    """Tests for evidence attachment integration."""

    def test_attach_semantic_drift_to_evidence_creates_governance_section(self) -> None:
        """Verify evidence attachment creates governance section if missing."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01", "data": {}}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        result = attach_semantic_drift_to_evidence(evidence, tile)

        assert "governance" in result
        assert "semantic_drift" in result["governance"]

    def test_attach_semantic_drift_to_evidence_includes_all_fields(self) -> None:
        """Verify evidence attachment includes all required fields."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "WARN",
            "recommendation_reasons": ["Reason 1"],
            "headline": "Test headline.",
        }

        result = attach_semantic_drift_to_evidence(evidence, tile)

        semantic_drift = result["governance"]["semantic_drift"]
        assert semantic_drift["status_light"] == "YELLOW"
        assert semantic_drift["tensor_norm"] == 1.5
        assert semantic_drift["semantic_hotspots"] == ["slice_a"]
        assert semantic_drift["projected_instability_count"] == 1
        assert semantic_drift["gating_recommendation"] == "WARN"
        assert semantic_drift["headline"] == "Test headline."

    def test_attach_semantic_drift_to_evidence_with_drift_signal(self) -> None:
        """Verify evidence attachment includes drift_signal when provided."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }
        drift_signal = {
            "status_light": "GREEN",
            "gating_recommendation": "OK",
            "semantic_hotspots": [],
        }

        result = attach_semantic_drift_to_evidence(evidence, tile, drift_signal)

        semantic_drift = result["governance"]["semantic_drift"]
        assert "drift_signal" in semantic_drift
        assert semantic_drift["drift_signal"]["status_light"] == "GREEN"

    def test_evidence_attachment_modifies_in_place(self) -> None:
        """Verify evidence attachment modifies the input dict in-place."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        result = attach_semantic_drift_to_evidence(evidence, tile)

        # Should be the same object (modified in-place)
        assert result is evidence
        assert "governance" in evidence


class TestSemanticDriftCouncilIntegration:
    """Tests for uplift council adapter integration."""

    def test_summarize_semantic_drift_for_uplift_council_has_required_fields(
        self,
    ) -> None:
        """Verify council summary contains all required fields."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "RED",
            "tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a", "slice_b"],
            "projected_instability_count": 2,
            "gating_recommendation": "BLOCK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)

        required_fields = [
            "status",
            "semantic_hotspots",
            "tensor_norm",
            "gating_recommendation",
            "advisory",
        ]

        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"

        assert summary["status"] == "BLOCK"
        assert summary["semantic_hotspots"] == ["slice_a", "slice_b"]
        assert summary["tensor_norm"] == 2.0

    def test_council_status_mapping_block(self) -> None:
        """Verify council status maps BLOCK correctly."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "RED",
            "tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "BLOCK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)
        assert summary["status"] == "BLOCK"
        assert "blocking" in summary["advisory"].lower() or "block" in summary["advisory"].lower()

    def test_council_status_mapping_warn(self) -> None:
        """Verify council status maps WARN correctly."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "WARN",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)
        assert summary["status"] == "WARN"
        assert "attention" in summary["advisory"].lower()

    def test_council_status_mapping_ok(self) -> None:
        """Verify council status maps OK correctly."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)
        assert summary["status"] == "OK"
        assert "stable" in summary["advisory"].lower()

    def test_council_hotspots_sorted(self) -> None:
        """Verify council summary sorts hotspots deterministically."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": ["zebra", "alpha", "beta"],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)
        assert summary["semantic_hotspots"] == sorted(tile["semantic_hotspots"])

    def test_council_summary_serializes(self) -> None:
        """Verify council summary is JSON-serializable."""
        from backend.health.semantic_drift_adapter import (
            summarize_semantic_drift_for_uplift_council,
        )

        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        summary = summarize_semantic_drift_for_uplift_council(tile)

        json_str = json.dumps(summary)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSemanticDriftFailureShelf:
    """Tests for First-Light failure shelf builder.
    
    Note: The shelf is meant as a triage list for auditors, not a gate.
    """

    def test_build_semantic_drift_failure_shelf_has_required_fields(self) -> None:
        """Verify failure shelf contains all required fields."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_failure_shelf_for_first_light,
        )

        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a", "slice_b", "slice_c"],
            "projected_instability_count": 2,
            "status_light": "YELLOW",
            "gating_recommendation": "WARN",
        }

        p4_calibration = {
            "tensor_norm": 2.0,
            "hotspots": ["slice_a", "slice_d"],
            "regression_status": "REGRESSED",
            "projected_instability_count": 1,
        }

        shelf = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )

        required_fields = [
            "schema_version",
            "p3_tensor_norm",
            "p4_tensor_norm",
            "semantic_hotspots",
            "regression_status",
        ]

        for field in required_fields:
            assert field in shelf, f"Missing required field: {field}"

        assert shelf["schema_version"] == "1.0.0"
        assert shelf["p3_tensor_norm"] == 1.5
        assert shelf["p4_tensor_norm"] == 2.0
        assert shelf["regression_status"] == "REGRESSED"

    def test_failure_shelf_limits_hotspots_to_top_5(self) -> None:
        """Verify failure shelf limits semantic hotspots to top 5."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_failure_shelf_for_first_light,
        )

        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": [
                "slice_a",
                "slice_b",
                "slice_c",
                "slice_d",
                "slice_e",
                "slice_f",
                "slice_g",
            ],
            "projected_instability_count": 0,
            "status_light": "GREEN",
            "gating_recommendation": "OK",
        }

        p4_calibration = {
            "tensor_norm": 0.5,
            "hotspots": [],
            "regression_status": "STABLE",
            "projected_instability_count": 0,
        }

        shelf = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )

        assert len(shelf["semantic_hotspots"]) == 5
        assert shelf["semantic_hotspots"] == [
            "slice_a",
            "slice_b",
            "slice_c",
            "slice_d",
            "slice_e",
        ]

    def test_failure_shelf_handles_empty_hotspots(self) -> None:
        """Verify failure shelf handles empty hotspots list."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_failure_shelf_for_first_light,
        )

        p3_summary = {
            "tensor_norm": 0.0,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "status_light": "GREEN",
            "gating_recommendation": "OK",
        }

        p4_calibration = {
            "tensor_norm": 0.0,
            "hotspots": [],
            "regression_status": "STABLE",
            "projected_instability_count": 0,
        }

        shelf = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )

        assert shelf["semantic_hotspots"] == []
        assert shelf["p3_tensor_norm"] == 0.0
        assert shelf["p4_tensor_norm"] == 0.0

    def test_failure_shelf_serializes(self) -> None:
        """Verify failure shelf is JSON-serializable."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_failure_shelf_for_first_light,
        )

        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "status_light": "YELLOW",
            "gating_recommendation": "WARN",
        }

        p4_calibration = {
            "tensor_norm": 2.0,
            "hotspots": ["slice_a"],
            "regression_status": "ATTENTION",
            "projected_instability_count": 1,
        }

        shelf = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )

        json_str = json.dumps(shelf)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0.0"

    def test_failure_shelf_deterministic(self) -> None:
        """Verify failure shelf produces deterministic output."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_failure_shelf_for_first_light,
        )

        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a", "slice_b"],
            "projected_instability_count": 2,
            "status_light": "YELLOW",
            "gating_recommendation": "WARN",
        }

        p4_calibration = {
            "tensor_norm": 2.0,
            "hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
            "projected_instability_count": 1,
        }

        shelf1 = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )
        shelf2 = build_semantic_drift_failure_shelf_for_first_light(
            p3_summary, p4_calibration
        )

        assert shelf1 == shelf2, "Shelf output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(shelf1, sort_keys=True)
        json2 = json.dumps(shelf2, sort_keys=True)
        assert json1 == json2

    def test_attach_semantic_drift_to_evidence_includes_failure_shelf(self) -> None:
        """Verify evidence attachment includes failure shelf when P3/P4 provided."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "YELLOW",
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "WARN",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a", "slice_b"],
            "projected_instability_count": 2,
            "status_light": "YELLOW",
            "gating_recommendation": "WARN",
        }

        p4_calibration = {
            "tensor_norm": 2.0,
            "hotspots": ["slice_a"],
            "regression_status": "ATTENTION",
            "projected_instability_count": 1,
        }

        result = attach_semantic_drift_to_evidence(
            evidence, tile, p3_summary=p3_summary, p4_calibration=p4_calibration
        )

        semantic_drift = result["governance"]["semantic_drift"]
        assert "first_light_failure_shelf" in semantic_drift

        shelf = semantic_drift["first_light_failure_shelf"]
        assert shelf["schema_version"] == "1.0.0"
        assert shelf["p3_tensor_norm"] == 1.5
        assert shelf["p4_tensor_norm"] == 2.0
        assert shelf["regression_status"] == "ATTENTION"

    def test_attach_semantic_drift_to_evidence_optional_failure_shelf(self) -> None:
        """Verify failure shelf is optional in evidence attachment."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_drift_to_evidence,
        )

        evidence = {"timestamp": "2024-01-01"}
        tile = {
            "schema_version": "1.0.0",
            "status_light": "GREEN",
            "tensor_norm": 0.5,
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Test.",
        }

        # Attach without P3/P4 (no failure shelf)
        result = attach_semantic_drift_to_evidence(evidence, tile)

        semantic_drift = result["governance"]["semantic_drift"]
        assert "first_light_failure_shelf" not in semantic_drift

        # Attach with only P3 (no failure shelf)
        p3_summary = {
            "tensor_norm": 1.5,
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "status_light": "YELLOW",
            "gating_recommendation": "WARN",
        }

        evidence2 = {"timestamp": "2024-01-01"}
        result2 = attach_semantic_drift_to_evidence(
            evidence2, tile, p3_summary=p3_summary
        )

        semantic_drift2 = result2["governance"]["semantic_drift"]
        assert "first_light_failure_shelf" not in semantic_drift2

        # Attach with both P3 and P4 (failure shelf included)
        p4_calibration = {
            "tensor_norm": 2.0,
            "hotspots": ["slice_a"],
            "regression_status": "ATTENTION",
            "projected_instability_count": 1,
        }

        evidence3 = {"timestamp": "2024-01-01"}
        result3 = attach_semantic_drift_to_evidence(
            evidence3, tile, p3_summary=p3_summary, p4_calibration=p4_calibration
        )

        semantic_drift3 = result3["governance"]["semantic_drift"]
        assert "first_light_failure_shelf" in semantic_drift3


class TestSemanticDriftTriageIndex:
    """Tests for semantic failure triage index builder."""

    def test_emit_cal_exp_semantic_failure_shelf_writes_file(self, tmp_path) -> None:
        """Verify shelf emitter writes file correctly."""
        from backend.health.semantic_drift_adapter import (
            emit_cal_exp_semantic_failure_shelf,
        )

        shelf = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 1.5,
            "p4_tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a", "slice_b"],
            "regression_status": "REGRESSED",
        }

        output_path = emit_cal_exp_semantic_failure_shelf(
            "CAL-EXP-1", shelf, output_dir=tmp_path
        )

        assert output_path.exists()
        assert output_path.name == "semantic_failure_shelf_CAL-EXP-1.json"

        # Verify file contents
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["cal_id"] == "CAL-EXP-1"
        assert data["p3_tensor_norm"] == 1.5
        assert data["p4_tensor_norm"] == 2.0
        assert data["semantic_hotspots"] == ["slice_a", "slice_b"]
        assert data["regression_status"] == "REGRESSED"

    def test_emit_cal_exp_semantic_failure_shelf_creates_directory(self, tmp_path) -> None:
        """Verify shelf emitter creates output directory if missing."""
        from backend.health.semantic_drift_adapter import (
            emit_cal_exp_semantic_failure_shelf,
        )

        shelf = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.5,
            "p4_tensor_norm": 0.5,
            "semantic_hotspots": [],
            "regression_status": "STABLE",
        }

        output_dir = tmp_path / "calibration"
        output_path = emit_cal_exp_semantic_failure_shelf(
            "CAL-EXP-2", shelf, output_dir=output_dir
        )

        assert output_dir.exists()
        assert output_path.exists()

    def test_build_semantic_failure_triage_index_has_required_fields(self) -> None:
        """Verify triage index contains all required fields."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a", "slice_b"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_c"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)

        required_fields = ["schema_version", "items", "total_shelves", "neutral_notes"]
        for field in required_fields:
            assert field in index, f"Missing required field: {field}"

        assert index["schema_version"] == "1.0.0"
        assert index["total_shelves"] == 2
        assert len(index["items"]) == 2

    def test_triage_index_ranks_by_severity(self) -> None:
        """Verify triage index ranks items by severity (REGRESSED > ATTENTION > STABLE)."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-3",
                "p3_tensor_norm": 0.5,
                "p4_tensor_norm": 0.5,
                "semantic_hotspots": [],
                "regression_status": "STABLE",
            },
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)

        items = index["items"]
        assert len(items) == 3

        # First item should be REGRESSED
        assert items[0]["regression_status"] == "REGRESSED"
        assert items[0]["cal_id"] == "CAL-EXP-1"

        # Second item should be ATTENTION
        assert items[1]["regression_status"] == "ATTENTION"
        assert items[1]["cal_id"] == "CAL-EXP-2"

        # Third item should be STABLE
        assert items[2]["regression_status"] == "STABLE"
        assert items[2]["cal_id"] == "CAL-EXP-3"

    def test_triage_index_truncates_to_max_items(self) -> None:
        """Verify triage index truncates to max_items."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": f"CAL-EXP-{i}",
                "p3_tensor_norm": float(i),
                "p4_tensor_norm": float(i),
                "semantic_hotspots": [],
                "regression_status": "REGRESSED",
            }
            for i in range(15)
        ]

        index = build_semantic_failure_triage_index(shelves, max_items=10)

        assert len(index["items"]) == 10
        assert index["total_shelves"] == 15

    def test_triage_index_handles_empty_shelves(self) -> None:
        """Verify triage index handles empty shelves list."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        index = build_semantic_failure_triage_index([])

        assert index["items"] == []
        assert index["total_shelves"] == 0
        assert len(index["neutral_notes"]) > 0

    def test_triage_index_deterministic(self) -> None:
        """Verify triage index produces deterministic output."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index1 = build_semantic_failure_triage_index(shelves)
        index2 = build_semantic_failure_triage_index(shelves)

        assert index1 == index2, "Triage index output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(index1, sort_keys=True)
        json2 = json.dumps(index2, sort_keys=True)
        assert json1 == json2

    def test_triage_index_serializes(self) -> None:
        """Verify triage index is JSON-serializable."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 1.5,
                "p4_tensor_norm": 2.0,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)

        json_str = json.dumps(index)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0.0"

    def test_triage_index_non_mutating(self) -> None:
        """Verify triage index builder does not mutate input shelves."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelf1 = {
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 1.5,
            "p4_tensor_norm": 2.0,
            "semantic_hotspots": ["slice_a"],
            "regression_status": "REGRESSED",
        }

        shelf2 = {
            "cal_id": "CAL-EXP-2",
            "p3_tensor_norm": 1.0,
            "p4_tensor_norm": 1.5,
            "semantic_hotspots": ["slice_b"],
            "regression_status": "ATTENTION",
        }

        shelves = [shelf1.copy(), shelf2.copy()]
        original_shelves = [s.copy() for s in shelves]

        index = build_semantic_failure_triage_index(shelves)

        # Verify original shelves unchanged
        assert shelves == original_shelves

    def test_attach_semantic_failure_triage_index_to_evidence(self) -> None:
        """Verify triage index attachment to evidence."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_failure_triage_index_to_evidence,
            build_semantic_failure_triage_index,
        )

        evidence = {"timestamp": "2024-01-01"}
        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
        ]

        triage_index = build_semantic_failure_triage_index(shelves)
        result = attach_semantic_failure_triage_index_to_evidence(evidence, triage_index)

        assert "governance" in result
        assert "semantic_failure_triage_index" in result["governance"]

        attached_index = result["governance"]["semantic_failure_triage_index"]
        assert attached_index["schema_version"] == "1.0.0"
        assert len(attached_index["items"]) == 1

    def test_attach_triage_index_modifies_in_place(self) -> None:
        """Verify triage index attachment modifies evidence in-place."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_failure_triage_index_to_evidence,
            build_semantic_failure_triage_index,
        )

        evidence = {"timestamp": "2024-01-01"}
        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 1.5,
                "p4_tensor_norm": 2.0,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
        ]

        triage_index = build_semantic_failure_triage_index(shelves)
        result = attach_semantic_failure_triage_index_to_evidence(evidence, triage_index)

        # Should be the same object (modified in-place)
        assert result is evidence
        assert "semantic_failure_triage_index" in evidence["governance"]

    def test_triage_index_items_include_shelf_path_hint(self) -> None:
        """Verify triage index items include shelf_path_hint field."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)

        items = index["items"]
        assert len(items) == 2

        # Verify shelf_path_hint is present and correct
        assert "shelf_path_hint" in items[0]
        assert items[0]["shelf_path_hint"] == "calibration/semantic_failure_shelf_CAL-EXP-1.json"
        assert items[1]["shelf_path_hint"] == "calibration/semantic_failure_shelf_CAL-EXP-2.json"

    def test_extract_semantic_failure_triage_index_signal_has_required_fields(self) -> None:
        """Verify signal extraction produces required fields."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b", "slice_c"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)
        signal = extract_semantic_failure_triage_index_signal(index)

        required_fields = ["total_items", "top5"]
        for field in required_fields:
            assert field in signal, f"Missing required field: {field}"

        assert signal["total_items"] == 2
        assert len(signal["top5"]) == 2

    def test_signal_extraction_top5_structure(self) -> None:
        """Verify top5 items have correct structure."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a", "slice_b"],
                "regression_status": "REGRESSED",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)
        signal = extract_semantic_failure_triage_index_signal(index)

        top5 = signal["top5"]
        assert len(top5) == 1

        item = top5[0]
        required_fields = ["cal_id", "regression_status", "combined_tensor_norm", "hotspots_count"]
        for field in required_fields:
            assert field in item, f"Missing required field in top5 item: {field}"

        assert item["cal_id"] == "CAL-EXP-1"
        assert item["regression_status"] == "REGRESSED"
        assert item["combined_tensor_norm"] == 4.5
        assert item["hotspots_count"] == 2

    def test_signal_extraction_advisory_warning_on_regressed(self) -> None:
        """Verify advisory warning is present when any item has REGRESSED status."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)
        signal = extract_semantic_failure_triage_index_signal(index)

        assert "advisory_warning" in signal
        assert "REGRESSED" in signal["advisory_warning"]
        assert "1" in signal["advisory_warning"]  # Should mention 1 regressed experiment

    def test_signal_extraction_no_advisory_warning_when_stable(self) -> None:
        """Verify no advisory warning when all items are STABLE."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 0.5,
                "p4_tensor_norm": 0.5,
                "semantic_hotspots": [],
                "regression_status": "STABLE",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)
        signal = extract_semantic_failure_triage_index_signal(index)

        assert "advisory_warning" not in signal

    def test_signal_extraction_top5_truncates_to_5(self) -> None:
        """Verify top5 truncates to 5 items even if more exist."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": f"CAL-EXP-{i}",
                "p3_tensor_norm": float(i),
                "p4_tensor_norm": float(i),
                "semantic_hotspots": [],
                "regression_status": "REGRESSED",
            }
            for i in range(10)
        ]

        index = build_semantic_failure_triage_index(shelves, max_items=10)
        signal = extract_semantic_failure_triage_index_signal(index)

        assert signal["total_items"] == 10
        assert len(signal["top5"]) == 5  # Should truncate to 5

    def test_signal_extraction_deterministic(self) -> None:
        """Verify signal extraction produces deterministic output."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
            extract_semantic_failure_triage_index_signal,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index = build_semantic_failure_triage_index(shelves)
        signal1 = extract_semantic_failure_triage_index_signal(index)
        signal2 = extract_semantic_failure_triage_index_signal(index)

        assert signal1 == signal2, "Signal extraction should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(signal1, sort_keys=True)
        json2 = json.dumps(signal2, sort_keys=True)
        assert json1 == json2

    def test_attach_triage_index_includes_signal(self) -> None:
        """Verify attaching triage index also attaches signal."""
        from backend.health.semantic_drift_adapter import (
            attach_semantic_failure_triage_index_to_evidence,
            build_semantic_failure_triage_index,
        )

        evidence = {"timestamp": "2024-01-01"}
        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
        ]

        triage_index = build_semantic_failure_triage_index(shelves)
        result = attach_semantic_failure_triage_index_to_evidence(evidence, triage_index)

        # Verify both governance and signals are attached
        assert "governance" in result
        assert "semantic_failure_triage_index" in result["governance"]
        assert "signals" in result
        assert "semantic_failure_triage_index" in result["signals"]

        signal = result["signals"]["semantic_failure_triage_index"]
        assert "total_items" in signal
        assert "top5" in signal
        assert "advisory_warning" in signal  # Should have warning for REGRESSED

    def test_triage_index_with_hints_still_deterministic(self) -> None:
        """Verify triage index with shelf_path_hint is still deterministic."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_failure_triage_index,
        )

        shelves = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_tensor_norm": 2.0,
                "p4_tensor_norm": 2.5,
                "semantic_hotspots": ["slice_a"],
                "regression_status": "REGRESSED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_tensor_norm": 1.0,
                "p4_tensor_norm": 1.5,
                "semantic_hotspots": ["slice_b"],
                "regression_status": "ATTENTION",
            },
        ]

        index1 = build_semantic_failure_triage_index(shelves)
        index2 = build_semantic_failure_triage_index(shelves)

        assert index1 == index2, "Triage index with hints should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(index1, sort_keys=True)
        json2 = json.dumps(index2, sort_keys=True)
        assert json1 == json2

