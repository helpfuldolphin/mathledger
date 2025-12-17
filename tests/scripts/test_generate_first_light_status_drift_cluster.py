"""
Tests for scenario drift cluster signal integration in generate_first_light_status.

SHADOW MODE: All tests verify observational behavior only.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.generate_first_light_status import generate_status


class TestScenarioDriftClusterSignalIntegration:
    """Tests for scenario drift cluster signal extraction and integration."""

    @pytest.mark.unit
    def test_scenario_drift_cluster_signal_from_manifest(self):
        """Signal is extracted from manifest governance section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with drift cluster view
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
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
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "signals" in status
            assert status["signals"] is not None
            assert "scenario_drift_cluster" in status["signals"]
            
            cluster_signal = status["signals"]["scenario_drift_cluster"]
            assert cluster_signal["schema_version"] == "1.0.0"
            assert cluster_signal["mode"] == "SHADOW"
            assert cluster_signal["experiments_analyzed"] == 3
            assert cluster_signal["high_risk_slices"] == ["slice_a", "slice_b"]
            assert cluster_signal["persistence_score"] == 0.833333
            assert "drivers" in cluster_signal
            assert "extraction_source" in cluster_signal
            assert cluster_signal["extraction_source"] == "MANIFEST"

    @pytest.mark.unit
    def test_scenario_drift_cluster_signal_from_evidence_fallback(self):
        """Signal is extracted from evidence.json when not in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest without drift cluster view
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {},
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            # Create evidence.json with drift cluster view
            evidence = {
                "governance": {
                    "scenario_drift_cluster_view": {
                        "schema_version": "1.0.0",
                        "slice_frequency": {"slice_c": 2},
                        "high_risk_slices": ["slice_c"],
                        "experiments_analyzed": 2,
                        "persistence_buckets": {
                            "appears_in_1": [],
                            "appears_in_2": ["slice_c"],
                            "appears_in_3": [],
                        },
                        "persistence_score": 1.0,
                    },
                },
            }
            
            (evidence_dir / "evidence.json").write_text(json.dumps(evidence))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "signals" in status
            assert status["signals"] is not None
            assert "scenario_drift_cluster" in status["signals"]
            
            cluster_signal = status["signals"]["scenario_drift_cluster"]
            assert cluster_signal["experiments_analyzed"] == 2
            assert cluster_signal["high_risk_slices"] == ["slice_c"]
            assert cluster_signal["persistence_score"] == 1.0

    @pytest.mark.unit
    def test_scenario_drift_cluster_signal_missing_safe(self):
        """Status generation succeeds when drift cluster view is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest without drift cluster view
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {},
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            # Status should be generated successfully
            assert "signals" in status
            
            # Signal should not be present if view is missing
            if status["signals"]:
                assert "scenario_drift_cluster" not in status["signals"]

    @pytest.mark.unit
    def test_scenario_drift_cluster_warning_high_persistence_score(self):
        """Warning is generated when persistence_score >= 0.5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with high persistence score
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
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
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "warnings" in status
            warnings = status["warnings"]
            
            # Should have exactly one warning about persistence score
            cluster_warnings = [
                w for w in warnings
                if "Scenario drift cluster" in w
            ]
            assert len(cluster_warnings) == 1  # One warning max
            
            warning = cluster_warnings[0]
            assert "persistence_score" in warning
            assert "0.667" in warning or "0.666" in warning
            assert "DRIVER_PERSISTENCE_SCORE_HIGH" in warning

    @pytest.mark.unit
    def test_scenario_drift_cluster_warning_high_risk_slices(self):
        """Warning is generated when high_risk_slices is non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with high-risk slices
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
                        "schema_version": "1.0.0",
                        "slice_frequency": {"slice_a": 1, "slice_b": 1},
                        "high_risk_slices": ["slice_a", "slice_b"],  # Non-empty
                        "experiments_analyzed": 1,
                        "persistence_buckets": {
                            "appears_in_1": ["slice_a", "slice_b"],
                            "appears_in_2": [],
                            "appears_in_3": [],
                        },
                        "persistence_score": 0.3,  # < 0.5
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "warnings" in status
            warnings = status["warnings"]
            
            # Should have exactly one warning about high-risk slices
            cluster_warnings = [
                w for w in warnings
                if "Scenario drift cluster" in w
            ]
            assert len(cluster_warnings) == 1  # One warning max
            
            warning = cluster_warnings[0]
            assert "slices:" in warning
            assert "slice_a" in warning
            assert "slice_b" in warning
            assert "DRIVER_HIGH_RISK_SLICES_PRESENT" in warning

    @pytest.mark.unit
    def test_scenario_drift_cluster_warning_both_conditions(self):
        """Warning includes both conditions when both are true."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with both high persistence and high-risk slices
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
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
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "warnings" in status
            warnings = status["warnings"]
            
            # Should have exactly one warning with both conditions
            cluster_warnings = [
                w for w in warnings
                if "Scenario drift cluster" in w
            ]
            assert len(cluster_warnings) == 1  # One warning max
            
            warning = cluster_warnings[0]
            assert "persistence_score" in warning
            assert "slices:" in warning
            assert "slice_a" in warning
            assert "DRIVER_PERSISTENCE_SCORE_HIGH" in warning
            assert "DRIVER_HIGH_RISK_SLICES_PRESENT" in warning

    @pytest.mark.unit
    def test_scenario_drift_cluster_no_warning_low_conditions(self):
        """No warning when persistence_score < 0.5 and no high-risk slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with low persistence and no high-risk slices
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
                        "schema_version": "1.0.0",
                        "slice_frequency": {"slice_a": 1},
                        "high_risk_slices": [],  # Empty
                        "experiments_analyzed": 3,
                        "persistence_buckets": {
                            "appears_in_1": ["slice_a"],
                            "appears_in_2": [],
                            "appears_in_3": [],
                        },
                        "persistence_score": 0.333333,  # < 0.5
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            assert "signals" in status
            assert status["signals"] is not None
            assert "scenario_drift_cluster" in status["signals"]
            
            signal = status["signals"]["scenario_drift_cluster"]
            assert signal["drivers"] == []  # No drivers when conditions are low
            
            # Should not have warning
            warnings = status.get("warnings", [])
            cluster_warnings = [
                w for w in warnings
                if "Scenario drift cluster" in w
            ]
            assert len(cluster_warnings) == 0

    @pytest.mark.unit
    def test_scenario_drift_cluster_signal_deterministic(self):
        """Signal extraction is deterministic across invocations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p3_dir = Path(tmpdir) / "p3"
            p4_dir = Path(tmpdir) / "p4"
            evidence_dir = Path(tmpdir) / "evidence"
            
            p3_dir.mkdir()
            p4_dir.mkdir()
            evidence_dir.mkdir()
            
            # Create minimal P3/P4 artifacts
            p3_run = p3_dir / "fl_test"
            p3_run.mkdir()
            (p3_run / "synthetic_raw.jsonl").write_text("", encoding="utf-8")
            (p3_run / "stability_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "metrics": {"success_rate": 1.0, "omega_occupancy": 0.95},
            }), encoding="utf-8")
            (p3_run / "red_flag_matrix.json").write_text("{}", encoding="utf-8")
            (p3_run / "metrics_windows.json").write_text("{}", encoding="utf-8")
            (p3_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            (p3_run / "run_config.json").write_text("{}", encoding="utf-8")
            
            p4_run = p4_dir / "p4_test"
            p4_run.mkdir()
            (p4_run / "real_cycles.jsonl").write_text("", encoding="utf-8")
            (p4_run / "twin_predictions.jsonl").write_text("", encoding="utf-8")
            (p4_run / "calibration_report.json").write_text(json.dumps({
                "schema_version": "1.0.0",
                "twin_success_accuracy": 0.9,
            }), encoding="utf-8")
            (p4_run / "divergence_log.jsonl").write_text("", encoding="utf-8")
            (p4_run / "divergence_distribution.json").write_text("{}", encoding="utf-8")
            (p4_run / "tda_metrics.json").write_text("{}", encoding="utf-8")
            
            # Create manifest with drift cluster view
            manifest = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "file_count": 2,
                "shadow_mode_compliance": {
                    "all_divergence_logged_only": True,
                    "no_governance_modification": True,
                    "no_abort_enforcement": True,
                },
                "governance": {
                    "scenario_drift_cluster_view": {
                        "schema_version": "1.0.0",
                        "slice_frequency": {"slice_a": 2, "slice_b": 1},
                        "high_risk_slices": ["slice_a", "slice_b"],
                        "experiments_analyzed": 2,
                        "persistence_buckets": {
                            "appears_in_1": ["slice_b"],
                            "appears_in_2": ["slice_a"],
                            "appears_in_3": [],
                        },
                        "persistence_score": 0.75,
                    },
                },
                "files": [],
            }
            
            (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
            
            status1 = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            status2 = generate_status(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_dir,
            )
            
            signal1 = status1["signals"]["scenario_drift_cluster"]
            signal2 = status2["signals"]["scenario_drift_cluster"]
            
            assert signal1 == signal2
            # Verify drivers are deterministic
            assert signal1["drivers"] == signal2["drivers"]
            # Verify extraction_source is present
            assert signal1["extraction_source"] == "MANIFEST"
            assert signal2["extraction_source"] == "MANIFEST"
            # Verify drivers are deterministic
            assert signal1["drivers"] == signal2["drivers"]

