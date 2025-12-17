"""Tests for performance governance integration with evidence pack.

Tests cover:
- Evidence pack attachment function
- JSON serialization safety
- Shape consistency
- Read-only contract
"""

import json
import pytest
from experiments.verify_perf_equivalence import (
    build_perf_joint_governance_view,
    build_uplift_perf_governance_tile,
    attach_perf_governance_tile_to_evidence,
    attach_perf_governance_tile,
)


class TestEvidencePackIntegration:
    """Test performance governance integration with evidence pack."""

    def test_attach_perf_tile_to_evidence(self):
        """Test attaching perf tile to evidence payload."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [{"run_id": "run1", "status": "PASS"}],
            "release_risk_level": "LOW",
        }
        budget_trend = {"budget_risk": "LOW"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": [],
        }

        result = attach_perf_governance_tile_to_evidence(evidence, perf_joint_view)

        assert "governance" in result
        assert "uplift_perf" in result["governance"]
        assert result["governance"]["uplift_perf"]["tile_type"] == "uplift_perf_governance"

    def test_attach_perf_tile_read_only(self):
        """Test that attach_perf_governance_tile is read-only."""
        perf_joint_view = {
            "perf_risk": "LOW",
            "budget_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
            "summary_note": "test",
        }

        tile = build_uplift_perf_governance_tile(perf_joint_view)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": [],
        }

        result = attach_perf_governance_tile(evidence, tile)

        # Original evidence should be unchanged
        assert "governance" not in evidence

        # Result should have tile attached
        assert "governance" in result
        assert "uplift_perf" in result["governance"]
        assert result["governance"]["uplift_perf"] == tile

    def test_attach_perf_tile_additive(self):
        """Test that attach_perf_governance_tile is additive (preserves existing data)."""
        perf_joint_view = {
            "perf_risk": "LOW",
            "budget_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
            "summary_note": "test",
        }

        tile = build_uplift_perf_governance_tile(perf_joint_view)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": [],
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }

        result = attach_perf_governance_tile(evidence, tile)

        # Should preserve existing governance data
        assert "other_tile" in result["governance"]
        assert result["governance"]["other_tile"]["status"] == "OK"

        # Should add perf tile
        assert "uplift_perf" in result["governance"]

    def test_evidence_pack_json_serializable(self):
        """Test that evidence pack with perf tile is JSON serializable."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [{"run_id": "run1", "status": "PASS"}],
            "release_risk_level": "LOW",
        }
        budget_trend = {"budget_risk": "LOW"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": [],
        }

        result = attach_perf_governance_tile_to_evidence(evidence, perf_joint_view)

        # Should serialize to JSON
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to round-trip
        parsed = json.loads(json_str)
        assert parsed == result
        assert "governance" in parsed
        assert "uplift_perf" in parsed["governance"]

    def test_evidence_pack_shape_consistent(self):
        """Test that evidence pack maintains consistent shape."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [{"run_id": "run1", "status": "PASS"}],
            "release_risk_level": "LOW",
        }
        budget_trend = {"budget_risk": "LOW"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        evidence = {
            "version": "1.0.0",
            "experiment_id": "test-123",
            "artifacts": [],
        }

        result = attach_perf_governance_tile_to_evidence(evidence, perf_joint_view)

        # Check required keys
        assert "governance" in result
        assert "uplift_perf" in result["governance"]

        tile = result["governance"]["uplift_perf"]
        assert "tile_type" in tile
        assert "schema_version" in tile
        assert "status" in tile
        assert "perf_risk" in tile
        assert "slices_with_regressions" in tile
        assert "slices_blocking_uplift" in tile
        assert "headline" in tile
        assert "notes" in tile

    def test_minimal_evidence_pack(self):
        """Test creating minimal evidence pack with perf tile."""
        perf_joint_view = {
            "perf_risk": "LOW",
            "budget_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
            "summary_note": "Performance governance: nominal",
        }

        evidence = {}
        result = attach_perf_governance_tile_to_evidence(evidence, perf_joint_view)

        assert "governance" in result
        assert "uplift_perf" in result["governance"]
        assert result["governance"]["uplift_perf"]["status"] == "OK"

