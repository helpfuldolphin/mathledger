"""
CAL-EXP-1 Hygiene Import Tests

These tests verify that all modules required for CAL-EXP-1 replication
are available from a clean checkout. If any of these tests fail, the
repository is broken for reproducibility.

SAVE TO REPO: YES
"""

import importlib
import sys
from pathlib import Path

import pytest


class TestCalExp1HarnessImports:
    """Test that CAL-EXP-1 harness entry points exist and can be imported."""

    def test_run_p5_cal_exp1_exists(self):
        """Verify scripts/run_p5_cal_exp1.py exists."""
        harness_path = Path("scripts/run_p5_cal_exp1.py")
        assert harness_path.exists(), (
            f"HYGIENE FAILURE: {harness_path} does not exist. "
            "This file must be tracked in git for CAL-EXP-1 replication."
        )

    def test_first_light_cal_exp1_warm_start_exists(self):
        """Verify scripts/first_light_cal_exp1_warm_start.py exists."""
        harness_path = Path("scripts/first_light_cal_exp1_warm_start.py")
        assert harness_path.exists(), (
            f"HYGIENE FAILURE: {harness_path} does not exist. "
            "This file must be tracked in git for CAL-EXP-1 replication."
        )

    def test_first_light_cal_exp1_runtime_stability_exists(self):
        """Verify scripts/first_light_cal_exp1_runtime_stability.py exists."""
        harness_path = Path("scripts/first_light_cal_exp1_runtime_stability.py")
        assert harness_path.exists(), (
            f"HYGIENE FAILURE: {harness_path} does not exist. "
            "This file must be tracked in git for CAL-EXP-1 replication."
        )


class TestCalExp1BackendModuleImports:
    """Test that backend modules required by CAL-EXP-1 can be imported."""

    def test_experiments_u2_runtime_module_exists(self):
        """Verify experiments/u2/runtime/ module exists."""
        runtime_init = Path("experiments/u2/runtime/__init__.py")
        assert runtime_init.exists(), (
            f"HYGIENE FAILURE: {runtime_init} does not exist. "
            "The experiments.u2.runtime module must be tracked for CAL-EXP-1."
        )

    def test_experiments_u2_runtime_can_import(self):
        """Verify experiments.u2.runtime can be imported."""
        try:
            from experiments.u2 import runtime
        except ImportError as e:
            pytest.fail(
                f"HYGIENE FAILURE: Cannot import experiments.u2.runtime: {e}. "
                "This module must be tracked in git."
            )

    def test_experiments_u2_runtime_exports_build_runtime_health_snapshot(self):
        """Verify build_runtime_health_snapshot is exported."""
        try:
            from experiments.u2.runtime import build_runtime_health_snapshot
        except ImportError as e:
            pytest.fail(
                f"HYGIENE FAILURE: Cannot import build_runtime_health_snapshot: {e}. "
                "This function is required by first_light_cal_exp1_runtime_stability.py."
            )

    def test_backend_telemetry_module_exists(self):
        """Verify backend/telemetry/ module exists."""
        telemetry_init = Path("backend/telemetry/__init__.py")
        assert telemetry_init.exists(), (
            f"HYGIENE FAILURE: {telemetry_init} does not exist. "
            "The backend.telemetry module must be tracked for CAL-EXP-1."
        )

    def test_backend_telemetry_can_import(self):
        """Verify backend.telemetry can be imported."""
        try:
            import backend.telemetry
        except ImportError as e:
            pytest.fail(
                f"HYGIENE FAILURE: Cannot import backend.telemetry: {e}. "
                "This module must be tracked in git."
            )


class TestCalExp1ConfigurationFiles:
    """Test that configuration files required by CAL-EXP-1 exist."""

    def test_p5_synthetic_config_exists(self):
        """Verify config/p5_synthetic.json exists."""
        config_path = Path("config/p5_synthetic.json")
        assert config_path.exists(), (
            f"HYGIENE FAILURE: {config_path} does not exist. "
            "This config file is required by run_p5_cal_exp1.py --adapter-config."
        )


class TestCalExp1TrackedDependencies:
    """Test that core tracked dependencies are present."""

    def test_backend_topology_first_light_exists(self):
        """Verify backend/topology/first_light/ is tracked."""
        first_light_init = Path("backend/topology/first_light/__init__.py")
        assert first_light_init.exists(), (
            "HYGIENE FAILURE: backend/topology/first_light/ not present."
        )

    def test_backend_tda_exists(self):
        """Verify backend/tda/ is tracked."""
        tda_init = Path("backend/tda/__init__.py")
        assert tda_init.exists(), (
            "HYGIENE FAILURE: backend/tda/ not present."
        )

    def test_backend_ledger_monotone_guard_exists(self):
        """Verify backend/ledger/monotone_guard.py is tracked."""
        guard_path = Path("backend/ledger/monotone_guard.py")
        assert guard_path.exists(), (
            "HYGIENE FAILURE: backend/ledger/monotone_guard.py not present. "
            "This module is required by cal_exp_1_harness.py."
        )


class TestCalExp1TestDirectories:
    """Test that test directories required for validation exist."""

    def test_tests_telemetry_exists(self):
        """Verify tests/telemetry/ directory exists."""
        telemetry_tests = Path("tests/telemetry")
        assert telemetry_tests.exists() and telemetry_tests.is_dir(), (
            f"HYGIENE FAILURE: {telemetry_tests} does not exist. "
            "This test directory must be tracked for CAL-EXP-1 validation."
        )

    def test_tests_first_light_exists(self):
        """Verify tests/first_light/ directory exists."""
        first_light_tests = Path("tests/first_light")
        assert first_light_tests.exists() and first_light_tests.is_dir(), (
            f"HYGIENE FAILURE: {first_light_tests} does not exist. "
            "This test directory must be tracked."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
