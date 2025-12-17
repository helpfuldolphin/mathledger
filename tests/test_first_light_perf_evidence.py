"""Tests for First Light performance evidence integration.

Tests cover:
- First Light perf summary generation
- Evidence pack attachment
- JSON serialization safety
- Structure validation
- Read-only contract

WORKED EXAMPLE — How to Read a Perf Summary:

Given a synthetic perf_joint_view:
```python
perf_joint_view = {
    "perf_risk": "HIGH",
    "budget_risk": "HIGH",
    "slices_with_regressions": ["slice_uplift_goal", "slice_uplift_sparse"],
    "slices_blocking_uplift": ["slice_uplift_goal"],
    "summary_note": "High performance risk detected; High budget risk detected; Performance blocking uplift on 1 slice(s)",
}
```

The summary produced by build_first_light_perf_summary() will be:
```json
{
    "schema_version": "1.0.0",
    "perf_risk": "HIGH",
    "slices_with_regressions": ["slice_uplift_goal", "slice_uplift_sparse"],
    "slices_blocking_uplift": ["slice_uplift_goal"]
}
```

Interpretation for External Reviewers:
- perf_risk: "HIGH" indicates the performance subsystem recommends caution
- slices_with_regressions: Two slices showed performance regressions during the run
- slices_blocking_uplift: One slice (slice_uplift_goal) has performance issues that,
  combined with budget risk, are blocking uplift decisions

An external reviewer should read this as: "The performance subsystem would recommend
caution here. Slice 'slice_uplift_goal' has both performance regressions and budget
risk, which together block uplift. Slice 'slice_uplift_sparse' has regressions but
not blocking uplift (budget is OK)."

Note: This is advisory only. The Uplift Council aggregates this with budget and
metrics dimensions to make the final uplift decision. This summary provides
cross-check context for evidence review.
"""

import json
import pytest
from experiments.verify_perf_equivalence import (
    build_perf_joint_governance_view,
    build_first_light_perf_summary,
    attach_first_light_perf_summary,
)


class TestFirstLightPerfSummary:
    """Test First Light performance summary generation."""

    def test_build_first_light_perf_summary_low_risk(self):
        """Test building summary with LOW risk."""
        perf_joint_view = {
            "perf_risk": "LOW",
            "budget_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
            "summary_note": "Performance governance: nominal",
        }

        summary = build_first_light_perf_summary(perf_joint_view)

        assert summary["schema_version"] == "1.0.0"
        assert summary["perf_risk"] == "LOW"
        assert summary["slices_with_regressions"] == []
        assert summary["slices_blocking_uplift"] == []

    def test_build_first_light_perf_summary_high_risk(self):
        """Test building summary with HIGH risk and blocking slices."""
        perf_joint_view = {
            "perf_risk": "HIGH",
            "budget_risk": "HIGH",
            "slices_with_regressions": ["slice_a", "slice_b", "all_slices"],
            "slices_blocking_uplift": ["slice_a"],
            "summary_note": "High performance risk detected",
        }

        summary = build_first_light_perf_summary(perf_joint_view)

        assert summary["perf_risk"] == "HIGH"
        assert "slice_a" in summary["slices_with_regressions"]
        assert "slice_b" in summary["slices_with_regressions"]
        assert "all_slices" not in summary["slices_with_regressions"]  # Filtered out
        assert summary["slices_blocking_uplift"] == ["slice_a"]

    def test_build_first_light_perf_summary_filters_all_slices(self):
        """Test that 'all_slices' marker is filtered out."""
        perf_joint_view = {
            "perf_risk": "MEDIUM",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a", "all_slices"],
            "slices_blocking_uplift": ["all_slices"],
            "summary_note": "test",
        }

        summary = build_first_light_perf_summary(perf_joint_view)

        assert "all_slices" not in summary["slices_with_regressions"]
        assert "all_slices" not in summary["slices_blocking_uplift"]
        assert "slice_a" in summary["slices_with_regressions"]

    def test_build_first_light_perf_summary_missing_key(self):
        """Test that missing required key raises error."""
        with pytest.raises(Exception, match="missing required key"):
            build_first_light_perf_summary({})


class TestFirstLightEvidenceIntegration:
    """Test First Light evidence pack integration."""

    def test_attach_first_light_perf_summary(self):
        """Test attaching perf summary to evidence pack."""
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
        perf_summary = build_first_light_perf_summary(perf_joint_view)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        assert "governance" in result
        assert "uplift_perf_summary" in result["governance"]
        assert result["governance"]["uplift_perf_summary"] == perf_summary

    def test_attach_first_light_perf_summary_read_only(self):
        """Test that attach_first_light_perf_summary is read-only."""
        perf_summary = {
            "schema_version": "1.0.0",
            "perf_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
        }

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        # Original evidence should be unchanged
        assert "governance" not in evidence

        # Result should have summary attached
        assert "governance" in result
        assert "uplift_perf_summary" in result["governance"]

    def test_attach_first_light_perf_summary_additive(self):
        """Test that attach_first_light_perf_summary is additive (preserves existing data)."""
        perf_summary = {
            "schema_version": "1.0.0",
            "perf_risk": "LOW",
            "slices_with_regressions": [],
            "slices_blocking_uplift": [],
        }

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        # Should preserve existing governance data
        assert "other_tile" in result["governance"]
        assert result["governance"]["other_tile"]["status"] == "OK"

        # Should add perf summary
        assert "uplift_perf_summary" in result["governance"]

    def test_evidence_pack_json_serializable(self):
        """Test that evidence pack with perf summary is JSON serializable."""
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
        perf_summary = build_first_light_perf_summary(perf_joint_view)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        # Should serialize to JSON
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to round-trip
        parsed = json.loads(json_str)
        assert parsed == result
        assert "governance" in parsed
        assert "uplift_perf_summary" in parsed["governance"]

    def test_evidence_pack_structure_consistent(self):
        """Test that evidence pack maintains consistent structure."""
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
        perf_summary = build_first_light_perf_summary(perf_joint_view)

        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        # Check required keys
        assert "governance" in result
        assert "uplift_perf_summary" in result["governance"]

        summary = result["governance"]["uplift_perf_summary"]
        assert "schema_version" in summary
        assert "perf_risk" in summary
        assert "slices_with_regressions" in summary
        assert "slices_blocking_uplift" in summary

    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic perf_joint_view."""
        # Synthetic perf joint view (as might come from First Light analysis)
        perf_joint_view = {
            "perf_risk": "MEDIUM",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_blocking_uplift": [],
            "summary_note": "Performance monitoring: regressions detected on 2 slices",
        }

        # Build summary
        perf_summary = build_first_light_perf_summary(perf_joint_view)

        # Attach to evidence
        evidence = {
            "version": "1.0.0",
            "experiment_id": "first_light_001",
            "artifacts": [],
        }

        result = attach_first_light_perf_summary(evidence, perf_summary)

        # Verify structure
        assert result["governance"]["uplift_perf_summary"]["perf_risk"] == "MEDIUM"
        assert len(result["governance"]["uplift_perf_summary"]["slices_with_regressions"]) == 2
        assert result["governance"]["uplift_perf_summary"]["slices_blocking_uplift"] == []

        # Verify JSON serializable
        json.dumps(result)

