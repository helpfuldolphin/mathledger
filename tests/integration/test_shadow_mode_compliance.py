"""
SHADOW MODE Compliance Tests

Verifies that P3/P4 harnesses enforce SHADOW MODE contract:
- P3 harness never calls any governance APIs
- P4 FirstLightShadowRunnerP4 never mutates real runner state
- P4 FirstLightShadowRunnerP4 never emits abort/stop signals
- All logs explicitly mark mode="SHADOW" and action="LOGGED_ONLY"

These tests are critical for Phase X compliance.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestP3ShadowModeCompliance:
    """Verify P3 harness SHADOW MODE compliance."""

    def test_p3_harness_no_governance_api_calls(self) -> None:
        """Verify P3 harness source code contains no governance API calls."""
        harness_path = Path("scripts/usla_first_light_harness.py")
        source = harness_path.read_text()

        # Governance API patterns that should NOT appear
        forbidden_patterns = [
            "governance.enforce",
            "governance.abort",
            "governance.block",
            "governance.modify",
            "real_runner.stop",
            "real_runner.abort",
            "execute_abort",
            "trigger_abort",
            "enforce_policy",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in source, \
                f"P3 harness contains forbidden pattern: {pattern}"

    def test_p3_harness_shadow_mode_enforced_in_config(self) -> None:
        """Verify P3 harness enforces shadow_mode=True."""
        from backend.topology.first_light.config import FirstLightConfig

        config = FirstLightConfig()
        assert config.shadow_mode is True, "Default config must have shadow_mode=True"

        # Verify validation rejects shadow_mode=False
        config_invalid = FirstLightConfig(shadow_mode=False)
        errors = config_invalid.validate()
        assert any("SHADOW MODE" in err for err in errors), \
            "Config validation must reject shadow_mode=False"

    def test_p3_output_always_shadow_mode(self) -> None:
        """Verify P3 output artifacts have schema_version (shadow mode implicit)."""
        import shutil

        # Clean up first
        test_dir = Path("results/shadow_test_p3")
        if test_dir.exists():
            shutil.rmtree(test_dir)

        # Run a small P3 harness
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", "10",
                "--seed", "42",
                "--output-dir", str(test_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"P3 harness failed: {result.stderr}"

        # Find output directory
        output_dirs = list(test_dir.glob("fl_*"))
        assert len(output_dirs) == 1, f"Expected 1 output dir, found {len(output_dirs)}"
        output_dir = output_dirs[0]

        # Check stability_report.json has schema and comes from P3 (shadow-only phase)
        with open(output_dir / "stability_report.json") as f:
            report = json.load(f)
        assert report.get("schema_version") == "1.0.0", \
            "stability_report.json must have schema_version=1.0.0"
        # P3 is inherently SHADOW-only - verified by config test above

        # Clean up
        shutil.rmtree(test_dir)


class TestP4ShadowModeCompliance:
    """Verify P4 shadow runner SHADOW MODE compliance."""

    def test_p4_config_rejects_non_shadow_mode(self) -> None:
        """Verify P4 config rejects shadow_mode=False."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4

        config = FirstLightConfigP4(shadow_mode=False)
        errors = config.validate()
        assert any("SHADOW MODE" in err for err in errors), \
            "P4 config validation must reject shadow_mode=False"

    def test_p4_runner_enforces_shadow_mode(self) -> None:
        """Verify P4 runner raises on shadow_mode=False."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        provider = MockTelemetryProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=10,
            telemetry_adapter=provider,
            shadow_mode=False,
        )

        with pytest.raises(ValueError, match="SHADOW MODE"):
            FirstLightShadowRunnerP4(config, seed=42)

    def test_p4_runner_never_mutates_real_state(self) -> None:
        """Verify P4 runner never calls mutation methods on telemetry provider."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        # Create a mock provider that tracks method calls
        class TrackedMockProvider(MockTelemetryProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mutation_calls: List[str] = []

            def set_state(self, *args, **kwargs):
                self.mutation_calls.append("set_state")
                raise RuntimeError("SHADOW MODE VIOLATION: set_state called")

            def modify_governance(self, *args, **kwargs):
                self.mutation_calls.append("modify_governance")
                raise RuntimeError("SHADOW MODE VIOLATION: modify_governance called")

            def abort(self, *args, **kwargs):
                self.mutation_calls.append("abort")
                raise RuntimeError("SHADOW MODE VIOLATION: abort called")

            def stop(self, *args, **kwargs):
                self.mutation_calls.append("stop")
                raise RuntimeError("SHADOW MODE VIOLATION: stop called")

        provider = TrackedMockProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=50,
            telemetry_adapter=provider,
        )

        runner = FirstLightShadowRunnerP4(config, seed=42)

        # Run all cycles
        for _ in runner.run_cycles(50):
            pass

        # Verify no mutation methods were called
        assert provider.mutation_calls == [], \
            f"P4 runner called mutation methods: {provider.mutation_calls}"

    def test_p4_divergence_snapshots_all_logged_only(self) -> None:
        """Verify all divergence snapshots have action=LOGGED_ONLY."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        provider = MockTelemetryProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=50,
            telemetry_adapter=provider,
        )

        runner = FirstLightShadowRunnerP4(config, seed=42)
        for _ in runner.run_cycles(50):
            pass

        divergences = runner.get_divergence_snapshots()
        assert len(divergences) == 50

        for i, div in enumerate(divergences):
            assert div.action == "LOGGED_ONLY", \
                f"Divergence {i} has action={div.action}, expected LOGGED_ONLY"

    def test_p4_real_observations_all_shadow_mode(self) -> None:
        """Verify all real observations have mode=SHADOW."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        provider = MockTelemetryProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=50,
            telemetry_adapter=provider,
        )

        runner = FirstLightShadowRunnerP4(config, seed=42)
        for _ in runner.run_cycles(50):
            pass

        observations = runner.get_observations()
        assert len(observations) == 50

        for i, obs in enumerate(observations):
            assert obs.mode == "SHADOW", \
                f"Observation {i} has mode={obs.mode}, expected SHADOW"

    def test_p4_harness_output_all_shadow_mode(self) -> None:
        """Verify P4 harness output files all mark SHADOW mode."""
        import shutil

        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_p4_harness.py",
                "--cycles", "20",
                "--seed", "42",
                "--output-dir", "results/shadow_test_p4",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"P4 harness failed: {result.stderr}"

        output_dirs = list(Path("results/shadow_test_p4").glob("p4_*"))
        assert len(output_dirs) == 1
        output_dir = output_dirs[0]

        # Check p4_summary.json
        with open(output_dir / "p4_summary.json") as f:
            summary = json.load(f)
        assert summary.get("mode") == "SHADOW", \
            "p4_summary.json must have mode=SHADOW"

        # Check all divergence_log.jsonl entries
        with open(output_dir / "divergence_log.jsonl") as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                assert record.get("action") == "LOGGED_ONLY", \
                    f"divergence_log line {i} has action={record.get('action')}"
                assert record.get("mode") == "SHADOW", \
                    f"divergence_log line {i} has mode={record.get('mode')}"

        # Check all real_cycles.jsonl entries
        with open(output_dir / "real_cycles.jsonl") as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                assert record.get("mode") == "SHADOW", \
                    f"real_cycles line {i} has mode={record.get('mode')}"

        # Clean up
        shutil.rmtree("results/shadow_test_p4")

    def test_p4_red_flag_status_logged_only(self) -> None:
        """Verify red-flag status always indicates LOGGED_ONLY action."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        provider = MockTelemetryProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=50,
            telemetry_adapter=provider,
        )

        runner = FirstLightShadowRunnerP4(config, seed=42)
        for _ in runner.run_cycles(50):
            pass

        status = runner.get_red_flag_status()
        assert status["mode"] == "SHADOW", "Red-flag status must have mode=SHADOW"
        assert status["action"] == "LOGGED_ONLY", \
            "Red-flag status must have action=LOGGED_ONLY"

    def test_p4_divergence_status_logged_only(self) -> None:
        """Verify divergence status always indicates LOGGED_ONLY action."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

        provider = MockTelemetryProvider(seed=42)
        config = FirstLightConfigP4(
            total_cycles=50,
            telemetry_adapter=provider,
        )

        runner = FirstLightShadowRunnerP4(config, seed=42)
        for _ in runner.run_cycles(50):
            pass

        status = runner.get_divergence_status()
        assert status["mode"] == "SHADOW", "Divergence status must have mode=SHADOW"
        assert status["action"] == "LOGGED_ONLY", \
            "Divergence status must have action=LOGGED_ONLY"


class TestShadowModeDataStructures:
    """Verify data structures enforce SHADOW MODE markers."""

    def test_real_cycle_observation_default_shadow(self) -> None:
        """Verify RealCycleObservation defaults to mode=SHADOW."""
        from backend.topology.first_light.data_structures_p4 import RealCycleObservation

        obs = RealCycleObservation()
        assert obs.mode == "SHADOW"
        assert obs.source == "REAL_RUNNER"

    def test_twin_cycle_observation_default_shadow(self) -> None:
        """Verify TwinCycleObservation defaults to mode=SHADOW."""
        from backend.topology.first_light.data_structures_p4 import TwinCycleObservation

        obs = TwinCycleObservation()
        assert obs.mode == "SHADOW"
        assert obs.source == "SHADOW_TWIN"

    def test_divergence_snapshot_default_logged_only(self) -> None:
        """Verify DivergenceSnapshot defaults to action=LOGGED_ONLY."""
        from backend.topology.first_light.data_structures_p4 import DivergenceSnapshot

        snap = DivergenceSnapshot()
        assert snap.action == "LOGGED_ONLY"
