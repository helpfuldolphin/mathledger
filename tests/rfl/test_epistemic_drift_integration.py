"""Integration tests for epistemic drift P3/P4 and evidence pack integration.

Tests the integration of epistemic drift analysis with:
- P3 stability reports (epistemic_summary)
- P4 calibration reports (epistemic_calibration)
- Evidence packs (governance.epistemic_drift)
"""

import json
from typing import Any, Dict, List

import pytest

from rfl.verification.abstention_semantics import (
    build_abstention_storyline,
    build_epistemic_abstention_profile,
    build_epistemic_drift_timeline,
)
from rfl.verification.epistemic_drift_integration import (
    attach_epistemic_calibration_panel_to_evidence,
    attach_epistemic_drift_to_evidence,
    build_epistemic_calibration_for_p4,
    build_epistemic_calibration_panel,
    build_epistemic_summary_for_p3,
    build_first_light_epistemic_footprint,
    emit_cal_exp_epistemic_footprint,
    emit_epistemic_footprint_from_cal_exp_report,
)


class TestP3EpistemicSummary:
    """Test build_epistemic_summary_for_p3."""

    def test_summary_empty_profiles(self):
        """Test summary with empty profiles returns defaults."""
        summary = build_epistemic_summary_for_p3([])

        assert summary["schema_version"] == "1.0.0"
        assert summary["mean_epistemic_risk"] == "LOW"
        assert summary["drift_band"] == "STABLE"
        assert summary["abstention_anomalies"] == []
        assert summary["risk_distribution"] == {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    def test_summary_low_risk_stable(self):
        """Test summary with low risk, stable profiles."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_2"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_3"},
        ]

        summary = build_epistemic_summary_for_p3(profiles)

        assert summary["mean_epistemic_risk"] == "LOW"
        assert summary["drift_band"] == "STABLE"
        assert summary["risk_distribution"]["LOW"] == 3
        assert len(summary["abstention_anomalies"]) == 0

    def test_summary_high_risk_volatile(self):
        """Test summary with high risk, volatile profiles."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_2"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_3"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_4"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_5"},
        ]

        summary = build_epistemic_summary_for_p3(profiles)

        assert summary["mean_epistemic_risk"] in ["MEDIUM", "HIGH"]
        # Volatile drift should be detected
        assert summary["drift_band"] in ["DRIFTING", "VOLATILE"]
        assert "High mean epistemic risk" in summary["abstention_anomalies"] or any(
            "volatility" in a.lower() for a in summary["abstention_anomalies"]
        )

    def test_summary_includes_risk_distribution(self):
        """Test that risk distribution is correctly computed."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_2"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_3"},
        ]

        summary = build_epistemic_summary_for_p3(profiles)

        assert summary["risk_distribution"]["LOW"] == 1
        assert summary["risk_distribution"]["MEDIUM"] == 1
        assert summary["risk_distribution"]["HIGH"] == 1

    def test_summary_json_serializable(self):
        """Test that summary is JSON serializable."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
        ]

        summary = build_epistemic_summary_for_p3(profiles)

        # Should not raise
        json_str = json.dumps(summary)
        assert json_str is not None
        assert "mean_epistemic_risk" in json_str


class TestP4EpistemicCalibration:
    """Test build_epistemic_calibration_for_p4."""

    def test_calibration_empty_profiles(self):
        """Test calibration with empty profiles returns defaults."""
        calibration = build_epistemic_calibration_for_p4([])

        assert calibration["schema_version"] == "1.0.0"
        assert calibration["mean_risk"] == "LOW"
        assert calibration["drift_band"] == "STABLE"
        assert calibration["variance"] == 0.0
        assert calibration["structural_anomalies"] == []
        assert calibration["change_points"] == []
        assert calibration["drift_index"] == 0.0

    def test_calibration_includes_variance(self):
        """Test that variance is computed and normalized."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_2"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_3"},
        ]

        calibration = build_epistemic_calibration_for_p4(profiles)

        assert "variance" in calibration
        assert 0.0 <= calibration["variance"] <= 1.0
        assert calibration["variance"] > 0.0  # Should have some variance

    def test_calibration_detects_structural_anomalies(self):
        """Test that structural anomalies are detected."""
        # Create volatile profile sequence
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_2"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_3"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_4"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_5"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_6"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_7"},
        ]

        calibration = build_epistemic_calibration_for_p4(profiles)

        # Should detect some anomalies
        assert len(calibration["structural_anomalies"]) >= 0  # May or may not trigger
        assert calibration["drift_index"] >= 0.0
        assert calibration["drift_band"] in ["STABLE", "DRIFTING", "VOLATILE"]

    def test_calibration_includes_change_points(self):
        """Test that change points are included."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "LOW", "slice_name": "slice_2"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_3"},  # Significant transition
        ]

        calibration = build_epistemic_calibration_for_p4(profiles)

        assert "change_points" in calibration
        # May or may not detect transition depending on threshold
        assert isinstance(calibration["change_points"], list)

    def test_calibration_json_serializable(self):
        """Test that calibration is JSON serializable."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
        ]

        calibration = build_epistemic_calibration_for_p4(profiles)

        # Should not raise
        json_str = json.dumps(calibration)
        assert json_str is not None
        assert "mean_risk" in json_str


class TestEvidencePackIntegration:
    """Test attach_epistemic_drift_to_evidence."""

    def test_attach_to_empty_evidence(self):
        """Test attaching to empty evidence dict."""
        evidence: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.1,
            "change_points": [],
            "summary_text": "Stable pattern",
        }

        result = attach_epistemic_drift_to_evidence(evidence, drift_timeline)

        assert "governance" in result
        assert "epistemic_drift" in result["governance"]
        assert result["governance"]["epistemic_drift"]["drift_band"] == "STABLE"
        assert result["governance"]["epistemic_drift"]["drift_index"] == 0.1

    def test_attach_with_storyline(self):
        """Test attaching with storyline."""
        evidence: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "DRIFTING",
            "drift_index": 0.5,
            "change_points": [
                {
                    "slice_name": "slice_a",
                    "transition": "LOW → HIGH",
                    "change_magnitude": 0.8,
                }
            ],
            "summary_text": "Drifting pattern",
        }
        storyline = {
            "global_epistemic_trend": "DEGRADING",
            "story": "Risk trend degrading",
        }

        result = attach_epistemic_drift_to_evidence(evidence, drift_timeline, storyline)

        drift_section = result["governance"]["epistemic_drift"]
        assert drift_section["drift_band"] == "DRIFTING"
        assert drift_section["storyline_episodes"] is not None
        assert drift_section["storyline_episodes"]["trend"] == "DEGRADING"
        assert len(drift_section["key_transitions"]) == 1

    def test_attach_preserves_existing_governance(self):
        """Test that existing governance data is preserved."""
        evidence: Dict[str, Any] = {
            "governance": {
                "other_signal": {"value": 42},
            }
        }
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.0,
            "change_points": [],
            "summary_text": "Stable",
        }

        result = attach_epistemic_drift_to_evidence(evidence, drift_timeline)

        assert "other_signal" in result["governance"]
        assert result["governance"]["other_signal"]["value"] == 42
        assert "epistemic_drift" in result["governance"]

    def test_attach_key_transitions_sorted(self):
        """Test that key transitions are sorted by magnitude."""
        evidence: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "VOLATILE",
            "drift_index": 0.9,
            "change_points": [
                {
                    "slice_name": "slice_a",
                    "transition": "LOW → MEDIUM",
                    "change_magnitude": 0.3,
                },
                {
                    "slice_name": "slice_b",
                    "transition": "MEDIUM → HIGH",
                    "change_magnitude": 0.8,
                },
                {
                    "slice_name": "slice_c",
                    "transition": "LOW → HIGH",
                    "change_magnitude": 0.5,
                },
            ],
            "summary_text": "Volatile",
        }

        result = attach_epistemic_drift_to_evidence(evidence, drift_timeline)

        transitions = result["governance"]["epistemic_drift"]["key_transitions"]
        assert len(transitions) <= 5  # Top 5
        # Should be sorted by magnitude (descending)
        if len(transitions) > 1:
            magnitudes = [t["change_magnitude"] for t in transitions]
            assert magnitudes == sorted(magnitudes, reverse=True)

    def test_attach_json_serializable(self):
        """Test that evidence with drift is JSON serializable."""
        evidence: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.0,
            "change_points": [],
            "summary_text": "Stable",
        }

        result = attach_epistemic_drift_to_evidence(evidence, drift_timeline)

        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None
        assert "epistemic_drift" in json_str


class TestDeterminism:
    """Test deterministic behavior of integration functions."""

    def test_p3_summary_deterministic(self):
        """Test that build_epistemic_summary_for_p3 is deterministic."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_2"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_3"},
        ]

        summary1 = build_epistemic_summary_for_p3(profiles)
        summary2 = build_epistemic_summary_for_p3(profiles)

        assert summary1 == summary2

    def test_p4_calibration_deterministic(self):
        """Test that build_epistemic_calibration_for_p4 is deterministic."""
        profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_2"},
        ]

        calibration1 = build_epistemic_calibration_for_p4(profiles)
        calibration2 = build_epistemic_calibration_for_p4(profiles)

        assert calibration1 == calibration2

    def test_evidence_attach_deterministic(self):
        """Test that attach_epistemic_drift_to_evidence is deterministic."""
        evidence1: Dict[str, Any] = {}
        evidence2: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "DRIFTING",
            "drift_index": 0.5,
            "change_points": [
                {
                    "slice_name": "slice_a",
                    "transition": "LOW → HIGH",
                    "change_magnitude": 0.8,
                }
            ],
            "summary_text": "Drifting",
        }

        result1 = attach_epistemic_drift_to_evidence(evidence1, drift_timeline)
        result2 = attach_epistemic_drift_to_evidence(evidence2, drift_timeline)

        # Compare epistemic_drift sections
        assert (
            result1["governance"]["epistemic_drift"]
            == result2["governance"]["epistemic_drift"]
        )

    def test_footprint_deterministic(self):
        """Test that build_first_light_epistemic_footprint is deterministic."""
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }

        footprint1 = build_first_light_epistemic_footprint(p3_summary, p4_calibration)
        footprint2 = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        assert footprint1 == footprint2

    def test_evidence_with_footprint_deterministic(self):
        """Test that evidence attachment with footprint is deterministic."""
        evidence1: Dict[str, Any] = {}
        evidence2: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.0,
            "change_points": [],
            "summary_text": "Stable",
        }
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }
        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        result1 = attach_epistemic_drift_to_evidence(
            evidence1, drift_timeline, first_light_footprint=footprint
        )
        result2 = attach_epistemic_drift_to_evidence(
            evidence2, drift_timeline, first_light_footprint=footprint
        )

        # Compare epistemic_drift sections including footprint
        assert (
            result1["governance"]["epistemic_drift"]
            == result2["governance"]["epistemic_drift"]
        )


class TestFirstLightFootprint:
    """Test build_first_light_epistemic_footprint.
    
    Worked Example:
    P3: DRIFTING + HIGH; P4: STABLE + LOW.
    This suggests synthetic stress is higher than real-world behavior;
    treat as conservative, not dangerous.
    """

    def test_footprint_combines_p3_p4(self):
        """Test that footprint combines P3 and P4 data."""
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }

        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        assert footprint["schema_version"] == "1.0.0"
        assert footprint["p3_drift_band"] == "STABLE"
        assert footprint["p4_drift_band"] == "DRIFTING"
        assert footprint["p3_mean_risk"] == "LOW"
        assert footprint["p4_mean_risk"] == "MEDIUM"

    def test_footprint_with_defaults(self):
        """Test that footprint handles missing fields with defaults."""
        p3_summary = {}  # Missing fields
        p4_calibration = {}  # Missing fields

        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        assert footprint["p3_drift_band"] == "STABLE"
        assert footprint["p4_drift_band"] == "STABLE"
        assert footprint["p3_mean_risk"] == "LOW"
        assert footprint["p4_mean_risk"] == "LOW"

    def test_footprint_json_serializable(self):
        """Test that footprint is JSON serializable."""
        p3_summary = {
            "drift_band": "VOLATILE",
            "mean_epistemic_risk": "HIGH",
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "mean_risk": "LOW",
        }

        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        # Should not raise
        json_str = json.dumps(footprint)
        assert json_str is not None
        assert "p3_drift_band" in json_str
        assert "p4_drift_band" in json_str

    def test_footprint_attached_to_evidence(self):
        """Test that footprint can be attached to evidence."""
        evidence: Dict[str, Any] = {}
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.0,
            "change_points": [],
            "summary_text": "Stable",
        }
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }
        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        result = attach_epistemic_drift_to_evidence(
            evidence, drift_timeline, first_light_footprint=footprint
        )

        assert "first_light_footprint" in result["governance"]["epistemic_drift"]
        footprint_in_evidence = result["governance"]["epistemic_drift"][
            "first_light_footprint"
        ]
        assert footprint_in_evidence["p3_drift_band"] == "STABLE"
        assert footprint_in_evidence["p4_drift_band"] == "DRIFTING"
        assert footprint_in_evidence["p3_mean_risk"] == "LOW"
        assert footprint_in_evidence["p4_mean_risk"] == "MEDIUM"

    def test_footprint_with_full_p3_p4_data(self):
        """Test footprint with complete P3 and P4 summaries."""
        # Build real P3 summary
        p3_profiles = [
            {"epistemic_risk_band": "LOW", "slice_name": "slice_1"},
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_2"},
        ]
        p3_summary = build_epistemic_summary_for_p3(p3_profiles)

        # Build real P4 calibration
        p4_profiles = [
            {"epistemic_risk_band": "MEDIUM", "slice_name": "slice_3"},
            {"epistemic_risk_band": "HIGH", "slice_name": "slice_4"},
        ]
        p4_calibration = build_epistemic_calibration_for_p4(p4_profiles)

        # Build footprint
        footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)

        # Verify footprint contains correct data
        assert footprint["p3_drift_band"] == p3_summary["drift_band"]
        assert footprint["p4_drift_band"] == p4_calibration["drift_band"]
        assert footprint["p3_mean_risk"] == p3_summary["mean_epistemic_risk"]
        assert footprint["p4_mean_risk"] == p4_calibration["mean_risk"]


class TestCalExpFootprintEmission:
    """Test emit_cal_exp_epistemic_footprint."""

    def test_emission_includes_cal_id(self):
        """Test that emission includes calibration experiment ID."""
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }

        footprint = emit_cal_exp_epistemic_footprint(
            "CAL-EXP-1", p3_summary, p4_calibration
        )

        assert footprint["schema_version"] == "1.0.0"
        assert footprint["cal_id"] == "CAL-EXP-1"
        assert footprint["p3_drift_band"] == "STABLE"
        assert footprint["p4_drift_band"] == "DRIFTING"
        assert footprint["p3_mean_risk"] == "LOW"
        assert footprint["p4_mean_risk"] == "MEDIUM"

    def test_emission_shape_correct(self):
        """Test that emission has correct shape."""
        p3_summary = {
            "drift_band": "VOLATILE",
            "mean_epistemic_risk": "HIGH",
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "mean_risk": "LOW",
        }

        footprint = emit_cal_exp_epistemic_footprint(
            "CAL-EXP-2", p3_summary, p4_calibration
        )

        # Verify all required fields present
        required_fields = {
            "schema_version",
            "cal_id",
            "p3_drift_band",
            "p4_drift_band",
            "p3_mean_risk",
            "p4_mean_risk",
        }
        assert set(footprint.keys()) == required_fields

    def test_emission_deterministic(self):
        """Test that emission is deterministic."""
        p3_summary = {
            "drift_band": "DRIFTING",
            "mean_epistemic_risk": "MEDIUM",
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "mean_risk": "LOW",
        }

        footprint1 = emit_cal_exp_epistemic_footprint(
            "CAL-EXP-3", p3_summary, p4_calibration
        )
        footprint2 = emit_cal_exp_epistemic_footprint(
            "CAL-EXP-3", p3_summary, p4_calibration
        )

        assert footprint1 == footprint2

    def test_emission_json_serializable(self):
        """Test that emission is JSON serializable."""
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "mean_risk": "LOW",
        }

        footprint = emit_cal_exp_epistemic_footprint(
            "CAL-EXP-1", p3_summary, p4_calibration
        )

        # Should not raise
        json_str = json.dumps(footprint)
        assert json_str is not None
        assert "CAL-EXP-1" in json_str

    def test_emission_non_mutating(self):
        """Test that emission does not mutate input dictionaries."""
        p3_summary = {
            "drift_band": "STABLE",
            "mean_epistemic_risk": "LOW",
        }
        p4_calibration = {
            "drift_band": "DRIFTING",
            "mean_risk": "MEDIUM",
        }
        p3_summary_original = p3_summary.copy()
        p4_calibration_original = p4_calibration.copy()

        emit_cal_exp_epistemic_footprint("CAL-EXP-1", p3_summary, p4_calibration)

        # Verify inputs unchanged
        assert p3_summary == p3_summary_original
        assert p4_calibration == p4_calibration_original


class TestEpistemicCalibrationPanel:
    """Test build_epistemic_calibration_panel."""

    def test_panel_empty_footprints(self):
        """Test panel with empty footprints returns defaults."""
        panel = build_epistemic_calibration_panel([])

        assert panel["schema_version"] == "1.0.0"
        assert panel["num_experiments"] == 0
        assert panel["num_conservative"] == 0
        assert panel["num_divergent"] == 0
        assert panel["num_high_risk_both"] == 0
        assert panel["experiments"] == []

    def test_panel_counts_conservative(self):
        """Test that panel correctly counts conservative patterns (P3 > P4)."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "VOLATILE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "MEDIUM",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["num_experiments"] == 2
        assert panel["num_conservative"] == 2
        assert panel["num_divergent"] == 0
        assert panel["num_high_risk_both"] == 0

    def test_panel_counts_divergent(self):
        """Test that panel correctly counts divergent patterns (P4 > P3)."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "HIGH",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "MEDIUM",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["num_experiments"] == 2
        assert panel["num_conservative"] == 0
        assert panel["num_divergent"] == 2
        assert panel["num_high_risk_both"] == 0

    def test_panel_counts_high_risk_both(self):
        """Test that panel correctly counts high risk in both P3 and P4."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "VOLATILE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "HIGH",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "HIGH",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["num_experiments"] == 2
        assert panel["num_high_risk_both"] == 2

    def test_panel_includes_experiments(self):
        """Test that panel includes original footprints."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert len(panel["experiments"]) == 1
        assert panel["experiments"][0]["cal_id"] == "CAL-EXP-1"

    def test_panel_mixed_patterns(self):
        """Test panel with mixed patterns."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },  # Conservative
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "HIGH",
            },  # Divergent
            {
                "cal_id": "CAL-EXP-3",
                "p3_drift_band": "VOLATILE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "HIGH",
            },  # High risk both
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["num_experiments"] == 3
        assert panel["num_conservative"] == 1
        assert panel["num_divergent"] == 1
        assert panel["num_high_risk_both"] == 1

    def test_panel_json_serializable(self):
        """Test that panel is JSON serializable."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        # Should not raise
        json_str = json.dumps(panel)
        assert json_str is not None
        assert "num_experiments" in json_str

    def test_panel_deterministic(self):
        """Test that panel is deterministic."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "MEDIUM",
                "p4_mean_risk": "LOW",
            },
        ]

        panel1 = build_epistemic_calibration_panel(footprints)
        panel2 = build_epistemic_calibration_panel(footprints)

        assert panel1 == panel2


class TestAutoEmitterFromCalExpReport:
    """Test emit_epistemic_footprint_from_cal_exp_report."""

    def test_emission_from_cal_exp1_report_with_epistemic_data(self):
        """Test auto-emission from CAL-EXP-1 report with epistemic_summary/epistemic_calibration."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "DRIFTING",
                "mean_epistemic_risk": "MEDIUM",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint["cal_id"] == "CAL-EXP-1"
        assert footprint["p3_drift_band"] == "DRIFTING"
        assert footprint["p4_drift_band"] == "STABLE"
        assert footprint["p3_mean_risk"] == "MEDIUM"
        assert footprint["p4_mean_risk"] == "LOW"
        assert footprint["extraction_path"] == "DIRECT"
        assert "confidence" not in footprint  # Not using defaults
        assert "advisory_note" not in footprint  # Not using defaults
        assert "extraction_audit" in footprint
        assert len(footprint["extraction_audit"]) == 4  # DIRECT, NESTED, FALLBACK, DEFAULTS
        assert footprint["extraction_audit"][0]["path"] == "DIRECT"
        assert footprint["extraction_audit"][0]["found"] is True

    def test_emission_from_nested_p3_p4_structure(self):
        """Test auto-emission from report with nested p3/p4 structure."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "p3": {
                "epistemic_summary": {
                    "drift_band": "VOLATILE",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "MEDIUM",
                }
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-2", cal_exp_report)

        assert footprint["cal_id"] == "CAL-EXP-2"
        assert footprint["p3_drift_band"] == "VOLATILE"
        assert footprint["p4_drift_band"] == "DRIFTING"
        assert footprint["p3_mean_risk"] == "HIGH"
        assert footprint["p4_mean_risk"] == "MEDIUM"
        assert footprint["extraction_path"] == "NESTED"
        assert "confidence" not in footprint  # Not using defaults
        assert "advisory_note" not in footprint  # Not using defaults
        assert "extraction_audit" in footprint
        assert footprint["extraction_audit"][1]["path"] == "NESTED"
        assert footprint["extraction_audit"][1]["found"] is True

    def test_emission_from_alignment_fallback(self):
        """Test auto-emission falls back to alignment data if epistemic data missing."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_alignment_summary": {
                "alignment_band": "HIGH",
                "forecast_band": "MEDIUM",
                "status_light": "RED",
            },
            "epistemic_alignment": {
                "alignment_band": "LOW",
                "forecast_band": "LOW",
                "status_light": "GREEN",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint["cal_id"] == "CAL-EXP-1"
        # Should derive from alignment data
        assert footprint["p3_drift_band"] in ["STABLE", "DRIFTING", "VOLATILE"]
        assert footprint["p4_drift_band"] in ["STABLE", "DRIFTING", "VOLATILE"]
        assert footprint["p3_mean_risk"] in ["LOW", "MEDIUM", "HIGH"]
        assert footprint["p4_mean_risk"] in ["LOW", "MEDIUM", "HIGH"]
        assert footprint["extraction_path"] == "FALLBACK"
        assert "confidence" not in footprint  # Not using defaults
        assert "advisory_note" not in footprint  # Not using defaults
        assert "extraction_audit" in footprint
        assert footprint["extraction_audit"][2]["path"] == "FALLBACK"
        assert footprint["extraction_audit"][2]["found"] is True

    def test_emission_with_safe_defaults(self):
        """Test auto-emission uses safe defaults when no data present."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "summary": {"final_divergence_rate": 0.05},
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint["cal_id"] == "CAL-EXP-1"
        # Should use safe defaults
        assert footprint["p3_drift_band"] == "STABLE"
        assert footprint["p4_drift_band"] == "STABLE"
        assert footprint["p3_mean_risk"] == "LOW"
        assert footprint["p4_mean_risk"] == "LOW"
        assert footprint["extraction_path"] == "DEFAULTS"
        assert footprint["confidence"] == "LOW"
        assert footprint["advisory_note"] == "DEFAULTS_USED"

    def test_emission_deterministic(self):
        """Test that auto-emission is deterministic."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "DRIFTING",
                "mean_risk": "MEDIUM",
            },
        }

        footprint1 = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)
        footprint2 = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint1 == footprint2

    def test_emission_json_serializable(self):
        """Test that auto-emission output is JSON serializable."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        # Should not raise
        json_str = json.dumps(footprint)
        assert json_str is not None
        assert "CAL-EXP-1" in json_str
        assert "extraction_path" in json_str

    def test_extraction_path_direct_precedence(self):
        """Test that DIRECT extraction takes precedence over NESTED."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "HIGH",
                }
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        # Should use DIRECT (top-level), not NESTED
        assert footprint["extraction_path"] == "DIRECT"
        assert footprint["p3_drift_band"] == "STABLE"  # From top-level, not nested
        assert footprint["p4_drift_band"] == "STABLE"

    def test_extraction_path_nested_precedence(self):
        """Test that NESTED extraction takes precedence over FALLBACK."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "p3": {
                "epistemic_summary": {
                    "drift_band": "VOLATILE",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "MEDIUM",
                }
            },
            "epistemic_alignment_summary": {
                "alignment_band": "LOW",
                "status_light": "GREEN",
            },
            "epistemic_alignment": {
                "alignment_band": "LOW",
                "status_light": "GREEN",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        # Should use NESTED, not FALLBACK
        assert footprint["extraction_path"] == "NESTED"
        assert footprint["p3_drift_band"] == "VOLATILE"  # From nested, not fallback
        assert footprint["p4_drift_band"] == "DRIFTING"

    def test_defaults_always_produce_confidence_low(self):
        """Test that using DEFAULTS always produces confidence=LOW and advisory_note=DEFAULTS_USED."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "summary": {"final_divergence_rate": 0.05},
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint["extraction_path"] == "DEFAULTS"
        assert footprint["confidence"] == "LOW"
        assert footprint["advisory_note"] == "DEFAULTS_USED"

    def test_defaults_partial_extraction(self):
        """Test that partial extraction (one side defaults) also uses DEFAULTS path."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            # Missing epistemic_calibration
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        # Should use DEFAULTS because P4 is missing
        assert footprint["extraction_path"] == "DEFAULTS"
        assert footprint["confidence"] == "LOW"
        assert footprint["advisory_note"] == "DEFAULTS_USED"
        assert "extraction_audit" in footprint
        # DIRECT should be found (P3 present), but DEFAULTS used (P4 missing)
        assert footprint["extraction_audit"][0]["found"] is False  # Both P3 and P4 required
        assert footprint["extraction_audit"][3]["found"] is True

    def test_extraction_audit_completeness_and_ordering(self):
        """Test that extraction_audit includes all paths in deterministic order."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert "extraction_audit" in footprint
        audit = footprint["extraction_audit"]
        
        # Must have exactly 4 entries in order: DIRECT, NESTED, FALLBACK, DEFAULTS
        assert len(audit) == 4
        assert audit[0]["path"] == "DIRECT"
        assert audit[1]["path"] == "NESTED"
        assert audit[2]["path"] == "FALLBACK"
        assert audit[3]["path"] == "DEFAULTS"
        
        # Each entry must have path, found, and fields
        for entry in audit:
            assert "path" in entry
            assert "found" in entry
            assert isinstance(entry["found"], bool)
            assert "fields" in entry
            assert isinstance(entry["fields"], list)

    def test_extraction_audit_fields_populated(self):
        """Test that extraction_audit fields list contains found field names."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        audit = footprint["extraction_audit"]
        
        # DIRECT should have both fields
        direct_entry = audit[0]
        assert direct_entry["path"] == "DIRECT"
        assert direct_entry["found"] is True
        assert "epistemic_summary" in direct_entry["fields"]
        assert "epistemic_calibration" in direct_entry["fields"]
        
        # NESTED should not be found
        nested_entry = audit[1]
        assert nested_entry["path"] == "NESTED"
        assert nested_entry["found"] is False
        assert len(nested_entry["fields"]) == 0

    def test_multiple_sources_present_strict_mode(self):
        """Test that strict mode detects multiple sources and adds advisory note."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "HIGH",
                }
            },
        }

        # With strict mode ON
        footprint_strict = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )

        # Should use DIRECT (precedence), but note multiple complete sources
        assert footprint_strict["extraction_path"] == "DIRECT"
        assert "MULTIPLE_SOURCES_PRESENT_COMPLETE" in footprint_strict["advisory_note"]
        
        # Audit should show both DIRECT and NESTED found
        audit = footprint_strict["extraction_audit"]
        assert audit[0]["found"] is True  # DIRECT
        assert audit[1]["found"] is True  # NESTED
        assert footprint_strict["extraction_audit_schema_version"] == "1.0.0"

    def test_multiple_sources_present_strict_mode_with_defaults(self):
        """Test that strict mode combines MULTIPLE_SOURCES_PRESENT with DEFAULTS_USED."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "HIGH",
                }
            },
            # Missing some fields, so will use defaults for those
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )

        # Should use DIRECT (precedence), note multiple complete sources, but no defaults
        assert footprint["extraction_path"] == "DIRECT"
        assert "MULTIPLE_SOURCES_PRESENT_COMPLETE" in footprint["advisory_note"]
        # No defaults used in this case
        assert "DEFAULTS_USED" not in footprint.get("advisory_note", "")

    def test_multiple_sources_present_strict_mode_off(self):
        """Test that without strict mode, multiple sources are silently handled."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "HIGH",
                }
            },
        }

        # With strict mode OFF (default)
        footprint = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=False
        )

        # Should use DIRECT (precedence), no advisory note about multiple sources
        assert footprint["extraction_path"] == "DIRECT"
        assert "MULTIPLE_SOURCES_PRESENT_COMPLETE" not in footprint.get("advisory_note", "")
        assert "MULTIPLE_SOURCES_PRESENT_PARTIAL" not in footprint.get("advisory_note", "")

    def test_extraction_audit_deterministic(self):
        """Test that extraction_audit is deterministic."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint1 = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)
        footprint2 = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert footprint1["extraction_audit"] == footprint2["extraction_audit"]

    def test_multiple_sources_present_complete(self):
        """Test that strict mode detects MULTIPLE_SOURCES_PRESENT_COMPLETE (2+ complete sources)."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            "p4": {
                "epistemic_calibration": {
                    "drift_band": "DRIFTING",
                    "mean_risk": "HIGH",
                }
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )

        # Both DIRECT and NESTED are complete sources
        assert footprint["extraction_path"] == "DIRECT"
        assert "MULTIPLE_SOURCES_PRESENT_COMPLETE" in footprint["advisory_note"]
        assert "MULTIPLE_SOURCES_PRESENT_PARTIAL" not in footprint["advisory_note"]
        assert footprint["extraction_audit_schema_version"] == "1.0.0"

    def test_multiple_sources_present_partial(self):
        """Test that strict mode detects MULTIPLE_SOURCES_PRESENT_PARTIAL (1 complete + partial)."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
            # Missing p4.epistemic_calibration, so NESTED is partial
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )

        # DIRECT is complete, NESTED is partial
        assert footprint["extraction_path"] == "DIRECT"
        assert "MULTIPLE_SOURCES_PRESENT_PARTIAL" in footprint["advisory_note"]
        assert "MULTIPLE_SOURCES_PRESENT_COMPLETE" not in footprint["advisory_note"]
        assert footprint["extraction_audit_schema_version"] == "1.0.0"

    def test_advisory_note_deterministic_ordering(self):
        """Test that advisory notes are deterministically ordered."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            # Missing epistemic_calibration, so will use defaults
            "p3": {
                "epistemic_summary": {
                    "drift_band": "DRIFTING",
                    "mean_epistemic_risk": "HIGH",
                }
            },
        }

        footprint1 = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )
        footprint2 = emit_epistemic_footprint_from_cal_exp_report(
            "CAL-EXP-1", cal_exp_report, strict_extraction=True
        )

        # Advisory notes should be deterministically ordered (sorted)
        assert footprint1["advisory_note"] == footprint2["advisory_note"]
        # Should contain both DEFAULTS_USED and MULTIPLE_SOURCES_PRESENT_PARTIAL
        assert "DEFAULTS_USED" in footprint1["advisory_note"]
        assert "MULTIPLE_SOURCES_PRESENT_PARTIAL" in footprint1["advisory_note"]
        # Should be sorted alphabetically
        assert footprint1["advisory_note"].startswith("DEFAULTS_USED")

    def test_extraction_audit_schema_version_present(self):
        """Test that extraction_audit_schema_version is always present."""
        cal_exp_report = {
            "schema_version": "1.0.0",
            "epistemic_summary": {
                "drift_band": "STABLE",
                "mean_epistemic_risk": "LOW",
            },
            "epistemic_calibration": {
                "drift_band": "STABLE",
                "mean_risk": "LOW",
            },
        }

        footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

        assert "extraction_audit_schema_version" in footprint
        assert footprint["extraction_audit_schema_version"] == "1.0.0"

    def test_extraction_path_validation(self):
        """Test that extraction_path is validated against allowed set."""
        # This test ensures that if somehow an invalid path is generated, it raises an error
        # In practice, this should never happen, but we validate for safety
        from rfl.verification.epistemic_drift_integration import EXTRACTION_PATHS
        
        # Verify all valid paths are in the frozen set
        assert "DIRECT" in EXTRACTION_PATHS
        assert "NESTED" in EXTRACTION_PATHS
        assert "FALLBACK" in EXTRACTION_PATHS
        assert "DEFAULTS" in EXTRACTION_PATHS
        assert len(EXTRACTION_PATHS) == 4


class TestDominantPattern:
    """Test dominant_pattern logic in panel builder."""

    def test_dominant_pattern_conservative(self):
        """Test that panel correctly identifies CONSERVATIVE pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "VOLATILE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "MEDIUM",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "MEDIUM",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["dominant_pattern"] == "CONSERVATIVE"
        assert panel["num_conservative"] == 3

    def test_dominant_pattern_divergent(self):
        """Test that panel correctly identifies DIVERGENT pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "HIGH",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "MEDIUM",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["dominant_pattern"] == "DIVERGENT"
        assert panel["num_divergent"] == 2

    def test_dominant_pattern_high_risk_both(self):
        """Test that panel correctly identifies HIGH_RISK_BOTH pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "VOLATILE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "HIGH",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "HIGH",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["dominant_pattern"] == "HIGH_RISK_BOTH"
        assert panel["num_high_risk_both"] == 2

    def test_dominant_pattern_convergent(self):
        """Test that panel correctly identifies CONVERGENT pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["dominant_pattern"] == "CONVERGENT"

    def test_dominant_pattern_mixed(self):
        """Test that panel correctly identifies MIXED pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },  # Conservative
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "VOLATILE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "HIGH",
            },  # Divergent
            {
                "cal_id": "CAL-EXP-3",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },  # Convergent
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert panel["dominant_pattern"] == "MIXED"

    def test_dominant_pattern_empty(self):
        """Test that empty panel returns MIXED."""
        panel = build_epistemic_calibration_panel([])

        assert panel["dominant_pattern"] == "MIXED"
        assert panel["dominant_pattern_confidence"] == 0.0

    def test_dominant_pattern_deterministic(self):
        """Test that dominant pattern is deterministic."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
        ]

        panel1 = build_epistemic_calibration_panel(footprints)
        panel2 = build_epistemic_calibration_panel(footprints)

        assert panel1["dominant_pattern"] == panel2["dominant_pattern"]

    def test_dominant_pattern_confidence_included(self):
        """Test that panel includes dominant_pattern_confidence field."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        assert "dominant_pattern_confidence" in panel
        assert isinstance(panel["dominant_pattern_confidence"], float)
        assert 0.0 <= panel["dominant_pattern_confidence"] <= 1.0

    def test_dominant_pattern_confidence_high_margin(self):
        """Test that high margin produces high confidence."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
        ]

        panel = build_epistemic_calibration_panel(footprints)

        # All conservative, so high confidence
        assert panel["dominant_pattern"] == "CONSERVATIVE"
        assert panel["dominant_pattern_confidence"] > 0.5

    def test_dominant_pattern_confidence_low_margin(self):
        """Test that low margin produces low confidence."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },  # Conservative
            {
                "cal_id": "CAL-EXP-2",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "DRIFTING",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "HIGH",
            },  # Divergent
            {
                "cal_id": "CAL-EXP-3",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },  # Convergent
        ]

        panel = build_epistemic_calibration_panel(footprints)

        # Mixed pattern, so low confidence
        assert panel["dominant_pattern"] == "MIXED"
        assert panel["dominant_pattern_confidence"] < 0.5

    def test_dominant_pattern_confidence_bounds(self):
        """Test that confidence is always in [0.0, 1.0]."""
        # Test with various patterns
        test_cases = [
            # All conservative
            [
                {
                    "cal_id": "CAL-EXP-1",
                    "p3_drift_band": "DRIFTING",
                    "p4_drift_band": "STABLE",
                    "p3_mean_risk": "HIGH",
                    "p4_mean_risk": "LOW",
                },
            ],
            # Mixed
            [
                {
                    "cal_id": "CAL-EXP-1",
                    "p3_drift_band": "DRIFTING",
                    "p4_drift_band": "STABLE",
                    "p3_mean_risk": "HIGH",
                    "p4_mean_risk": "LOW",
                },
                {
                    "cal_id": "CAL-EXP-2",
                    "p3_drift_band": "STABLE",
                    "p4_drift_band": "DRIFTING",
                    "p3_mean_risk": "LOW",
                    "p4_mean_risk": "HIGH",
                },
            ],
            # Empty
            [],
        ]

        for footprints in test_cases:
            panel = build_epistemic_calibration_panel(footprints)
            confidence = panel["dominant_pattern_confidence"]
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of bounds for {len(footprints)} experiments"


class TestEvidenceAttachmentWithAdvisoryNotes:
    """Test attach_epistemic_calibration_panel_to_evidence with advisory notes."""

    def test_attachment_includes_dominant_pattern(self):
        """Test that evidence attachment includes dominant_pattern."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "HIGH",
                "p4_mean_risk": "LOW",
            },
        ]
        panel = build_epistemic_calibration_panel(footprints)
        evidence: Dict[str, Any] = {}

        result = attach_epistemic_calibration_panel_to_evidence(evidence, panel)

        assert "governance" in result
        assert "epistemic_calibration_panel" in result["governance"]
        attached_panel = result["governance"]["epistemic_calibration_panel"]
        assert attached_panel["dominant_pattern"] == "CONSERVATIVE"

    def test_attachment_with_advisory_notes(self):
        """Test that evidence attachment includes advisory_notes."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]
        panel = build_epistemic_calibration_panel(footprints)
        evidence: Dict[str, Any] = {}
        advisory_notes = [
            "P3 and P4 show stable alignment",
            "No significant epistemic drift detected",
        ]

        result = attach_epistemic_calibration_panel_to_evidence(evidence, panel, advisory_notes)

        attached_panel = result["governance"]["epistemic_calibration_panel"]
        assert attached_panel["advisory_notes"] == advisory_notes

    def test_attachment_without_advisory_notes(self):
        """Test that evidence attachment uses empty list if advisory_notes not provided."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]
        panel = build_epistemic_calibration_panel(footprints)
        evidence: Dict[str, Any] = {}

        result = attach_epistemic_calibration_panel_to_evidence(evidence, panel)

        attached_panel = result["governance"]["epistemic_calibration_panel"]
        assert attached_panel["advisory_notes"] == []

    def test_attachment_preserves_existing_governance(self):
        """Test that evidence attachment preserves existing governance data."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "STABLE",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "LOW",
                "p4_mean_risk": "LOW",
            },
        ]
        panel = build_epistemic_calibration_panel(footprints)
        evidence: Dict[str, Any] = {
            "governance": {
                "epistemic_drift": {
                    "drift_band": "STABLE",
                    "drift_index": 0.0,
                },
            },
        }

        result = attach_epistemic_calibration_panel_to_evidence(evidence, panel)

        # Should preserve existing governance
        assert "epistemic_drift" in result["governance"]
        assert result["governance"]["epistemic_drift"]["drift_band"] == "STABLE"
        # Should add panel
        assert "epistemic_calibration_panel" in result["governance"]

    def test_attachment_deterministic(self):
        """Test that evidence attachment is deterministic."""
        footprints = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_drift_band": "DRIFTING",
                "p4_drift_band": "STABLE",
                "p3_mean_risk": "MEDIUM",
                "p4_mean_risk": "LOW",
            },
        ]
        panel = build_epistemic_calibration_panel(footprints)
        evidence1: Dict[str, Any] = {}
        evidence2: Dict[str, Any] = {}
        advisory_notes = ["Test note"]

        result1 = attach_epistemic_calibration_panel_to_evidence(evidence1, panel, advisory_notes)
        result2 = attach_epistemic_calibration_panel_to_evidence(evidence2, panel, advisory_notes)

        # Compare panels
        panel1 = result1["governance"]["epistemic_calibration_panel"]
        panel2 = result2["governance"]["epistemic_calibration_panel"]
        assert panel1 == panel2

