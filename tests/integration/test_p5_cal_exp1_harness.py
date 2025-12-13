"""
P5 CAL-EXP-1 Calibration Harness Integration Tests

These tests verify the scientific integrity of the CAL-EXP-1 calibration
infrastructure. They do NOT enforce thresholds - only shape, determinism,
and SHADOW MODE compliance.

=============================================================================
ARCHITECT'S GUIDE: What These Tests Protect
=============================================================================

Each test is designed to protect a specific scientific property:

1. test_cal_exp1_runs_and_produces_metrics
   - PROTECTS: Output schema integrity
   - QUESTION: Does the harness produce correctly-structured calibration data?
   - If this fails: Plotting and analysis scripts will break

2. test_cal_exp1_window_metrics_structure
   - PROTECTS: Per-window metric completeness
   - QUESTION: Does each window contain all required P5 metrics?
   - If this fails: Time-series analysis will be incomplete

3. test_cal_exp1_replay_equivalence
   - PROTECTS: Reproducibility spine
   - QUESTION: Does TRACE replay produce identical metrics to SYNTHETIC?
   - If this fails: Non-determinism exists in the pipeline

4. test_cal_exp1_shadow_mode_preserved
   - PROTECTS: SHADOW MODE contract
   - QUESTION: Are all verdicts marked as SHADOW_ONLY?
   - If this fails: Accidental enforcement could occur

5. test_cal_exp1_pattern_classification_shape
   - PROTECTS: Pattern tag validity
   - QUESTION: Do pattern tags use only valid vocabulary?
   - If this fails: Regime classification is broken

6. test_cal_exp1_summary_aggregation
   - PROTECTS: Summary metric consistency
   - QUESTION: Are summary metrics computed from window metrics?
   - If this fails: Quick-view summaries are unreliable

=============================================================================
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestCalExp1HarnessIntegrity:
    """
    Test CAL-EXP-1 harness produces valid calibration artifacts.

    SCIENTIFIC PURPOSE: Ensure calibration infrastructure is reliable
    before using it for parameter tuning decisions.
    """

    @pytest.fixture
    def synthetic_adapter(self):
        """Create synthetic telemetry adapter for testing."""
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapter,
            AdapterMode,
        )
        return RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
            mode=AdapterMode.SYNTHETIC,
        )

    def test_cal_exp1_runs_and_produces_metrics(self, synthetic_adapter, tmp_path):
        """
        Test that CAL-EXP-1 produces correctly-structured output files.

        SCIENTIFIC QUESTION: Does the harness produce calibration data
        that can be consumed by analysis tools?

        PROTECTS: Output schema integrity. If this fails, downstream
        plotting and analysis scripts will break.
        """
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=40,  # 2 windows of 20 cycles
            window_size=20,
            seed=42,
            run_id="test_metrics_structure",
        )

        result = runner.run()

        # Verify metrics structure
        metrics = result.to_metrics_dict()
        assert metrics["schema_version"] == "1.0.0"
        assert metrics["mode"] == "SHADOW"
        assert metrics["total_cycles"] == 40
        assert metrics["window_size"] == 20
        assert "windows" in metrics
        assert len(metrics["windows"]) == 2  # 40 cycles / 20 per window

        # Verify summary structure
        summary = result.to_summary_dict()
        assert summary["schema_version"] == "1.0.0"
        assert summary["mode"] == "SHADOW"
        assert "summary" in summary
        assert "provisional_verdict" in summary
        assert summary["provisional_verdict"]["enforcement"] == "SHADOW_ONLY"

    def test_cal_exp1_window_metrics_structure(self, synthetic_adapter):
        """
        Test that each window contains all required P5 metrics.

        SCIENTIFIC QUESTION: Does each observation window contain the
        complete set of metrics needed for P5 analysis?

        PROTECTS: Per-window metric completeness. If this fails,
        time-series analysis will be missing data points.

        P5 BLUEPRINT MAPPING:
        - divergence_rate → P5-M1
        - mean_delta_p → P5-M2 proxy
        - delta_bias → P5-M2
        - delta_variance → P5-M3
        - phase_lag_xcorr → P5-M4
        - pattern_tag → P5-M5
        """
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=60,  # 3 windows
            window_size=20,
            seed=42,
        )

        result = runner.run()

        # Check each window has all required fields
        required_fields = [
            "window_index",
            "window_start",
            "window_end",
            "cycles_in_window",
            "divergence_count",
            "divergence_rate",      # P5-M1
            "mean_delta_p",         # P5-M2 proxy
            "delta_bias",           # P5-M2
            "delta_variance",       # P5-M3
            "phase_lag_xcorr",      # P5-M4
            "pattern_tag",          # P5-M5
        ]

        for window in result.windows:
            window_dict = window.to_dict()
            for field in required_fields:
                assert field in window_dict, f"Window missing field: {field}"

            # Verify numeric fields are bounded
            assert 0.0 <= window_dict["divergence_rate"] <= 1.0
            assert window_dict["delta_variance"] >= 0.0
            assert window_dict["cycles_in_window"] > 0

    def test_cal_exp1_replay_equivalence(self, tmp_path):
        """
        Test that TRACE replay produces identical metrics to SYNTHETIC run.

        SCIENTIFIC QUESTION: Is the calibration pipeline deterministic?
        Can we reproduce results from recorded traces?

        PROTECTS: Reproducibility spine. If this fails, there is
        non-determinism that will make calibration results unreliable.

        This is the P5 REPRODUCIBILITY CONTRACT. All calibration runs
        must be replayable from trace for scientific verification.
        """
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapter,
            RealTelemetryAdapterConfig,
            AdapterMode,
            write_trace_jsonl,
        )
        from scripts.run_p5_cal_exp1 import CalExp1Runner, verify_trace_replay

        seed = 12345
        cycles = 40
        window_size = 20

        # Run SYNTHETIC mode
        synthetic_adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=seed,
            mode=AdapterMode.SYNTHETIC,
        )

        synthetic_runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=cycles,
            window_size=window_size,
            seed=seed,
            run_id="synthetic_run",
        )
        synthetic_result = synthetic_runner.run()

        # Write trace
        trace_path = tmp_path / "test_trace.jsonl"
        snapshots = synthetic_runner.get_adapter_snapshots()
        write_trace_jsonl(snapshots, str(trace_path))

        # Run TRACE mode replay
        trace_adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=seed,
            trace_path=str(trace_path),
            mode=AdapterMode.TRACE,
        )

        trace_runner = CalExp1Runner(
            adapter=trace_adapter,
            total_cycles=cycles,
            window_size=window_size,
            seed=seed,
            run_id="trace_replay",
        )
        trace_result = trace_runner.run()

        # Verify equivalence
        check = verify_trace_replay(synthetic_result, trace_result)

        # The key assertion: replay MUST produce identical metrics
        assert check["verdict"] == "PASS", (
            f"Replay verification failed. Deltas: {check.get('deltas', {})}"
        )
        assert check["overall_pass"] is True

    def test_cal_exp1_shadow_mode_preserved(self, synthetic_adapter):
        """
        Test that all outputs are marked SHADOW_ONLY with no enforcement.

        SCIENTIFIC QUESTION: Does the calibration infrastructure maintain
        proper separation from production governance?

        PROTECTS: SHADOW MODE contract. If this fails, calibration
        verdicts could accidentally trigger production enforcement.

        INVARIANT: All verdicts must have enforcement="SHADOW_ONLY"
        """
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=40,
            window_size=20,
            seed=42,
        )

        result = runner.run()

        # Check mode markers
        assert result.mode == "SHADOW"

        # Check verdict enforcement
        summary = result.to_summary_dict()
        assert summary["provisional_verdict"]["enforcement"] == "SHADOW_ONLY"
        assert "_note" in summary["provisional_verdict"]
        assert "observational" in summary["provisional_verdict"]["_note"].lower()

        # Metrics should also declare SHADOW mode
        metrics = result.to_metrics_dict()
        assert metrics["mode"] == "SHADOW"

    def test_cal_exp1_pattern_classification_shape(self, synthetic_adapter):
        """
        Test that pattern tags use only valid vocabulary.

        SCIENTIFIC QUESTION: Are regime classifications using the
        expected pattern vocabulary?

        PROTECTS: Pattern tag validity. If this fails, downstream
        regime analysis will encounter unexpected values.

        VALID PATTERNS: NONE, DRIFT, SPIKE, OSCILLATION
        """
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=100,  # More cycles for pattern diversity
            window_size=20,
            seed=42,
        )

        result = runner.run()

        valid_patterns = {"NONE", "DRIFT", "SPIKE", "OSCILLATION"}

        for window in result.windows:
            assert window.pattern_tag in valid_patterns, (
                f"Invalid pattern tag: {window.pattern_tag}"
            )

        # Pattern progression should only contain valid patterns
        for pattern in result.pattern_progression:
            assert pattern in valid_patterns

    def test_cal_exp1_summary_aggregation(self, synthetic_adapter):
        """
        Test that summary metrics are correctly computed from windows.

        SCIENTIFIC QUESTION: Are the quick-view summary metrics
        mathematically consistent with the per-window data?

        PROTECTS: Summary metric consistency. If this fails,
        quick assessments will not match detailed analysis.
        """
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        runner = CalExp1Runner(
            adapter=synthetic_adapter,
            total_cycles=60,  # 3 windows
            window_size=20,
            seed=42,
        )

        result = runner.run()

        # Final metrics should match last window
        assert result.final_divergence_rate == result.windows[-1].divergence_rate
        assert result.final_delta_bias == result.windows[-1].delta_bias

        # Mean divergence should be average of all windows
        expected_mean = sum(w.divergence_rate for w in result.windows) / len(result.windows)
        assert abs(result.mean_divergence_over_run - expected_mean) < 1e-10

        # Pattern progression should match window pattern_tags
        assert result.pattern_progression == [w.pattern_tag for w in result.windows]


class TestCalExp1OutputFiles:
    """
    Test CAL-EXP-1 file output format and content.

    SCIENTIFIC PURPOSE: Ensure output files are suitable for
    external analysis tools and archive requirements.
    """

    def test_metrics_file_json_serializable(self, tmp_path):
        """
        Test that metrics can be written and read as valid JSON.

        SCIENTIFIC QUESTION: Can calibration data be archived and
        shared with external analysis tools?

        PROTECTS: Data portability. If this fails, results cannot
        be saved or loaded for later analysis.
        """
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapter,
            AdapterMode,
        )
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
            mode=AdapterMode.SYNTHETIC,
        )

        runner = CalExp1Runner(
            adapter=adapter,
            total_cycles=40,
            window_size=20,
            seed=42,
        )

        result = runner.run()

        # Write to file
        metrics_path = tmp_path / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result.to_metrics_dict(), f, indent=2)

        # Read back and verify
        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        assert loaded["schema_version"] == "1.0.0"
        assert loaded["window_count"] == len(result.windows)
        assert len(loaded["windows"]) == len(result.windows)

    def test_replay_check_file_structure(self, tmp_path):
        """
        Test that replay check file has correct structure.

        SCIENTIFIC QUESTION: Does the reproducibility check produce
        machine-readable verification results?

        PROTECTS: Verification automation. If this fails, CI/CD
        cannot automatically verify reproducibility.
        """
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapter,
            AdapterMode,
            write_trace_jsonl,
        )
        from scripts.run_p5_cal_exp1 import CalExp1Runner, verify_trace_replay

        seed = 99999
        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=seed,
            mode=AdapterMode.SYNTHETIC,
        )

        runner = CalExp1Runner(
            adapter=adapter,
            total_cycles=40,
            window_size=20,
            seed=seed,
        )
        result1 = runner.run()

        # Write trace and replay
        trace_path = tmp_path / "trace.jsonl"
        write_trace_jsonl(runner.get_adapter_snapshots(), str(trace_path))

        trace_adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=seed,
            trace_path=str(trace_path),
            mode=AdapterMode.TRACE,
        )

        runner2 = CalExp1Runner(
            adapter=trace_adapter,
            total_cycles=40,
            window_size=20,
            seed=seed,
        )
        result2 = runner2.run()

        check = verify_trace_replay(result1, result2)

        # Verify check structure
        required_fields = [
            "schema_version",
            "mode",
            "check_type",
            "timestamp",
            "synthetic_run_id",
            "trace_run_id",
            "tolerance",
            "overall_pass",
            "window_checks",
            "deltas",
            "verdict",
        ]

        for field in required_fields:
            assert field in check, f"Replay check missing field: {field}"

        assert check["check_type"] == "trace_replay_equivalence"
        assert check["mode"] == "SHADOW"
        assert check["verdict"] in ("PASS", "FAIL")


class TestCalExp1AdapterConfigIntegration:
    """
    Test CAL-EXP-1 with adapter configuration files.

    SCIENTIFIC PURPOSE: Ensure harness can be configured via
    external config files for reproducible experiment setup.
    """

    def test_cal_exp1_with_config_file(self, tmp_path):
        """
        Test that CAL-EXP-1 can be run with adapter config JSON.

        SCIENTIFIC QUESTION: Can experiments be configured via
        files for reproducible setup?

        PROTECTS: Configuration reproducibility. If this fails,
        experiments cannot be reliably reproduced from config.
        """
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapter,
            RealTelemetryAdapterConfig,
        )
        from scripts.run_p5_cal_exp1 import CalExp1Runner

        # Create config file
        config_path = tmp_path / "test_config.json"
        config_data = {
            "mode": "synthetic",
            "runner_type": "u2",
            "slice_name": "arithmetic_simple",
            "seed": 54321,
            "source_label": "TEST_CONFIG",
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load config and create adapter
        config = RealTelemetryAdapterConfig.from_json_file(str(config_path))
        adapter = RealTelemetryAdapter.from_config(config)

        # Run calibration
        runner = CalExp1Runner(
            adapter=adapter,
            total_cycles=40,
            window_size=20,
            seed=config.seed,
        )

        result = runner.run()

        # Verify run completed
        assert result.total_cycles == 40
        assert len(result.windows) == 2
        assert result.mode == "SHADOW"


class TestCalExp1EvidencePackHook:
    """
    Test CAL-EXP-1 evidence pack integration.

    SCIENTIFIC PURPOSE: Ensure CAL-EXP-1 data flows correctly into
    evidence pack while maintaining SHADOW MODE invariants.
    """

    def test_evidence_pack_loads_cal_exp1_report(self, tmp_path):
        """
        Test that evidence pack builder loads cal_exp1_report.json.

        SCIENTIFIC QUESTION: Does the evidence pack include CAL-EXP-1
        metrics when present?

        PROTECTS: Evidence pack completeness for P5 calibration data.
        If this fails, CAL-EXP-1 results won't be archived.
        """
        from backend.topology.first_light.evidence_pack import EvidencePackBuilder

        # Create run directory with cal_exp1_report.json
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "windows": [
                {"divergence_rate": 0.95, "pattern_tag": "NONE"},
                {"divergence_rate": 1.0, "pattern_tag": "OSCILLATION"},
            ],
            "summary": {
                "final_divergence_rate": 1.0,
                "final_delta_bias": -0.025,
            },
        }
        with open(run_dir / "cal_exp1_report.json", "w") as f:
            json.dump(report, f)

        builder = EvidencePackBuilder(validate_schemas=False)
        summary = builder._load_cal_exp1_summary(run_dir)

        assert summary is not None
        assert summary["final_divergence_rate"] == 1.0
        assert summary["final_delta_bias"] == -0.025
        assert summary["pattern_tag"] == "OSCILLATION"
        assert summary["note"] == "SHADOW advisory only; no gating."

    def test_evidence_pack_missing_cal_exp1_not_error(self, tmp_path):
        """
        Test that missing cal_exp1_report.json is NOT an error.

        SCIENTIFIC QUESTION: Does the evidence pack gracefully handle
        runs without CAL-EXP-1 data?

        PROTECTS: Non-blocking behavior. CAL-EXP-1 is optional;
        missing it should not break evidence pack generation.
        """
        from backend.topology.first_light.evidence_pack import EvidencePackBuilder

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        builder = EvidencePackBuilder(validate_schemas=False)
        summary = builder._load_cal_exp1_summary(run_dir)

        # Should return None, not raise an error
        assert summary is None

    def test_evidence_pack_cal_exp1_preserves_shadow_mode(self, tmp_path):
        """
        Test that CAL-EXP-1 hook preserves SHADOW MODE markers.

        SCIENTIFIC QUESTION: Does the evidence pack maintain SHADOW
        MODE contract when including CAL-EXP-1 data?

        PROTECTS: SHADOW MODE integrity. If this fails, CAL-EXP-1
        data could accidentally be interpreted as gating.

        INVARIANT: All CAL-EXP-1 data must include "SHADOW advisory" note.
        """
        from backend.topology.first_light.evidence_pack import EvidencePackBuilder

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "windows": [{"divergence_rate": 0.5, "pattern_tag": "DRIFT"}],
            "summary": {"final_divergence_rate": 0.5, "final_delta_bias": 0.01},
        }
        with open(run_dir / "cal_exp1_report.json", "w") as f:
            json.dump(report, f)

        builder = EvidencePackBuilder(validate_schemas=False)
        summary = builder._load_cal_exp1_summary(run_dir)

        # SHADOW MODE contract: must include advisory note
        assert summary is not None
        assert "SHADOW" in summary.get("note", "")
        assert "no gating" in summary.get("note", "").lower()


class TestCalExp1StatusHook:
    """
    Test CAL-EXP-1 status script integration.

    SCIENTIFIC PURPOSE: Ensure CAL-EXP-1 data flows correctly into
    first_light_status.json while maintaining SHADOW MODE invariants.
    """

    def test_status_loads_cal_exp1_summary(self, tmp_path):
        """
        Test that status script loads cal_exp1_report.json.

        SCIENTIFIC QUESTION: Does the status JSON include CAL-EXP-1
        calibration metrics when present?

        PROTECTS: Status completeness for P5 calibration data.
        """
        from scripts.generate_first_light_status import load_cal_exp1_summary

        # Create run directory with cal_exp1_report.json
        run_dir = tmp_path / "cal_exp1_test"
        run_dir.mkdir()

        report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "windows": [
                {"divergence_rate": 0.85, "pattern_tag": "NONE"},
                {"divergence_rate": 0.95, "pattern_tag": "DRIFT"},
            ],
            "summary": {
                "final_divergence_rate": 0.95,
                "final_delta_bias": -0.015,
            },
        }
        with open(run_dir / "cal_exp1_report.json", "w") as f:
            json.dump(report, f)

        summary = load_cal_exp1_summary(tmp_path)

        assert summary is not None
        assert summary["final_divergence_rate"] == 0.95
        assert summary["final_delta_bias"] == -0.015
        assert summary["pattern_tag"] == "DRIFT"
        assert summary["schema_version"] == "1.0.0"

    def test_status_missing_cal_exp1_not_error(self, tmp_path):
        """
        Test that missing cal_exp1_report.json is NOT an error.

        SCIENTIFIC QUESTION: Does the status script gracefully handle
        runs without CAL-EXP-1 data?

        PROTECTS: Non-blocking behavior. CAL-EXP-1 is optional.
        """
        from scripts.generate_first_light_status import load_cal_exp1_summary

        summary = load_cal_exp1_summary(tmp_path)

        # Should return None, not raise an error
        assert summary is None

    def test_status_cal_exp1_is_non_mutating(self, tmp_path):
        """
        Test that status CAL-EXP-1 hook does not modify input files.

        SCIENTIFIC QUESTION: Does the status hook read-only access
        CAL-EXP-1 data without side effects?

        PROTECTS: Non-mutating behavior. Status generation must
        not alter any calibration artifacts.
        """
        from scripts.generate_first_light_status import load_cal_exp1_summary
        import hashlib

        # Create run directory with cal_exp1_report.json
        run_dir = tmp_path / "cal_exp1_immutable"
        run_dir.mkdir()

        report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "windows": [{"divergence_rate": 0.9, "pattern_tag": "NONE"}],
            "summary": {"final_divergence_rate": 0.9, "final_delta_bias": 0.0},
        }
        report_path = run_dir / "cal_exp1_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        # Compute hash before
        with open(report_path, "rb") as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()

        # Load summary (should not mutate)
        summary = load_cal_exp1_summary(tmp_path)

        # Compute hash after
        with open(report_path, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()

        # File must be unchanged
        assert hash_before == hash_after
        assert summary is not None
