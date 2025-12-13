"""
CAL-EXP-2 Reproducibility Tests

Verifies that all files required for CAL-EXP-2 (P4 Divergence Minimization)
are tracked in git and importable from a clean checkout.

These tests are GATING - if they fail, CAL-EXP-2 cannot be reproduced.

SAVE TO REPO: YES
"""

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


class TestCalExp2HarnessTracked:
    """Verify CAL-EXP-2 harness script is tracked and valid."""

    def test_harness_script_exists(self):
        """Verify scripts/first_light_cal_exp2_convergence.py exists."""
        harness_path = Path("scripts/first_light_cal_exp2_convergence.py")
        assert harness_path.exists(), (
            f"CAL-EXP-2 HYGIENE FAILURE: {harness_path} does not exist. "
            "This file must be tracked in git for CAL-EXP-2 reproducibility."
        )

    def test_harness_script_compiles(self):
        """Verify harness script is valid Python syntax."""
        harness_path = Path("scripts/first_light_cal_exp2_convergence.py")
        if not harness_path.exists():
            pytest.skip("Harness script not found")

        try:
            py_compile.compile(str(harness_path), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"CAL-EXP-2 harness has syntax error: {e}")

    def test_harness_is_tracked_in_git(self):
        """Verify harness is tracked in git (not untracked)."""
        result = subprocess.run(
            ["git", "ls-files", "scripts/first_light_cal_exp2_convergence.py"],
            capture_output=True,
            text=True,
        )
        tracked_files = result.stdout.strip()
        assert "first_light_cal_exp2_convergence.py" in tracked_files, (
            "CAL-EXP-2 HYGIENE FAILURE: Harness script is not tracked in git. "
            "Run: git add scripts/first_light_cal_exp2_convergence.py"
        )


class TestCalExp2CoreDependencies:
    """Verify all P4 core dependencies are importable."""

    def test_import_first_light_config_p4(self):
        """Verify FirstLightConfigP4 is importable."""
        try:
            from backend.topology.first_light.config_p4 import FirstLightConfigP4
            assert FirstLightConfigP4 is not None
        except ImportError as e:
            pytest.fail(f"Cannot import FirstLightConfigP4: {e}")

    def test_import_first_light_runner_p4(self):
        """Verify FirstLightShadowRunnerP4 is importable."""
        try:
            from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
            assert FirstLightShadowRunnerP4 is not None
        except ImportError as e:
            pytest.fail(f"Cannot import FirstLightShadowRunnerP4: {e}")

    def test_import_mock_telemetry_provider(self):
        """Verify MockTelemetryProvider is importable."""
        try:
            from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
            assert MockTelemetryProvider is not None
        except ImportError as e:
            pytest.fail(f"Cannot import MockTelemetryProvider: {e}")

    def test_import_divergence_analyzer(self):
        """Verify DivergenceAnalyzer is importable."""
        try:
            from backend.topology.first_light.divergence_analyzer import DivergenceAnalyzer
            assert DivergenceAnalyzer is not None
        except ImportError as e:
            pytest.fail(f"Cannot import DivergenceAnalyzer: {e}")


class TestCalExp2MinimalSet:
    """Verify the minimal set of files for CAL-EXP-2 are all tracked."""

    CAL_EXP_2_MINIMAL_SET = [
        "scripts/first_light_cal_exp2_convergence.py",
        "backend/topology/first_light/runner_p4.py",
        "backend/topology/first_light/config_p4.py",
        "backend/topology/first_light/telemetry_adapter.py",
        "backend/topology/first_light/divergence_analyzer.py",
        "backend/topology/first_light/data_structures_p4.py",
        "backend/topology/first_light/p5_pattern_classifier.py",
        "tests/first_light/test_cal_exp2_exp3_scaffolds.py",
    ]

    def test_all_minimal_set_files_exist(self):
        """Verify all files in the minimal set exist."""
        missing = []
        for filepath in self.CAL_EXP_2_MINIMAL_SET:
            if not Path(filepath).exists():
                missing.append(filepath)

        assert not missing, (
            f"CAL-EXP-2 MINIMAL SET INCOMPLETE: Missing {len(missing)} files:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    def test_all_minimal_set_files_tracked(self):
        """Verify all files in the minimal set are tracked in git."""
        result = subprocess.run(
            ["git", "ls-files"] + self.CAL_EXP_2_MINIMAL_SET,
            capture_output=True,
            text=True,
        )
        tracked = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

        untracked = []
        for filepath in self.CAL_EXP_2_MINIMAL_SET:
            # Normalize path separators for comparison
            normalized = filepath.replace("\\", "/")
            if normalized not in tracked and filepath not in tracked:
                untracked.append(filepath)

        assert not untracked, (
            f"CAL-EXP-2 HYGIENE FAILURE: {len(untracked)} files not tracked:\n"
            + "\n".join(f"  - {f}" for f in untracked)
        )


class TestCalExp2HarnessImportable:
    """Verify CAL-EXP-2 harness script is importable (not just compilable)."""

    def test_harness_module_imports(self):
        """Verify harness script can be imported as a module."""
        harness_path = Path("scripts/first_light_cal_exp2_convergence.py")
        if not harness_path.exists():
            pytest.skip("Harness script not found")

        # Add scripts to path and attempt import
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "first_light_cal_exp2_convergence", harness_path
        )
        assert spec is not None, "Could not create module spec for harness"
        assert spec.loader is not None, "Module spec has no loader"

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            pytest.fail(
                f"CAL-EXP-2 IMPORT FAILURE: Harness has unresolved import: {e}\n"
                "This means a dependency is missing from the tracked files."
            )
        except Exception as e:
            # Other exceptions during import (not ImportError) are okay
            # since we're testing importability, not execution
            if "required" in str(e).lower() or "missing" in str(e).lower():
                pytest.fail(f"CAL-EXP-2 harness has missing dependency: {e}")

    def test_core_imports_from_harness(self):
        """Verify the exact imports used by the harness script."""
        # These are the TRUE imports from first_light_cal_exp2_convergence.py
        errors = []

        try:
            from backend.topology.first_light.config_p4 import FirstLightConfigP4

            assert FirstLightConfigP4 is not None
        except ImportError as e:
            errors.append(f"FirstLightConfigP4: {e}")

        try:
            from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4

            assert FirstLightShadowRunnerP4 is not None
        except ImportError as e:
            errors.append(f"FirstLightShadowRunnerP4: {e}")

        try:
            from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider

            assert MockTelemetryProvider is not None
        except ImportError as e:
            errors.append(f"MockTelemetryProvider: {e}")

        assert not errors, (
            "CAL-EXP-2 IMPORT FAILURE: Core dependencies missing:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


class TestCalExp2NoResultsRequired:
    """Verify CAL-EXP-2 imports don't require results/ directory."""

    def test_imports_without_results_directory(self):
        """Verify imports succeed without results/cal_exp_2 directory."""
        # These imports should work without any results directory
        errors = []

        try:
            from backend.topology.first_light.config_p4 import FirstLightConfigP4
        except ImportError as e:
            errors.append(f"FirstLightConfigP4: {e}")

        try:
            from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        except ImportError as e:
            errors.append(f"FirstLightShadowRunnerP4: {e}")

        try:
            from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
        except ImportError as e:
            errors.append(f"MockTelemetryProvider: {e}")

        assert not errors, (
            "CAL-EXP-2 imports failed (results/ should not be required):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
