"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Feature Flags Tests
==============================

Tests for the runtime feature flag system in experiments.u2.runtime.

This module verifies:
- Registry contents are stable and deterministic
- Flag getters/setters work correctly
- CLI prints flags in sorted order
- Defaults preserve current behavior
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestFeatureFlagRegistry(unittest.TestCase):
    """Tests for the feature flag registry."""

    def test_registry_is_dict(self) -> None:
        """FEATURE_FLAGS should be a dict."""
        from experiments.u2.runtime import FEATURE_FLAGS
        self.assertIsInstance(FEATURE_FLAGS, dict)

    def test_registry_has_expected_flags(self) -> None:
        """Registry should contain the documented flags."""
        from experiments.u2.runtime import FEATURE_FLAGS
        
        expected_flags = {
            "u2.use_cycle_orchestrator",
            "u2.enable_extra_telemetry",
            "u2.strict_input_validation",
            "u2.trace_hash_chain",
        }
        
        actual_flags = set(FEATURE_FLAGS.keys())
        
        missing = expected_flags - actual_flags
        self.assertEqual(missing, set(), f"Missing flags: {missing}")

    def test_registry_is_deterministic(self) -> None:
        """Registry should return same flags on repeated access."""
        from experiments.u2.runtime import FEATURE_FLAGS
        
        keys1 = sorted(FEATURE_FLAGS.keys())
        keys2 = sorted(FEATURE_FLAGS.keys())
        
        self.assertEqual(keys1, keys2)

    def test_all_flags_have_required_fields(self) -> None:
        """Each flag should have name, default, description, stability."""
        from experiments.u2.runtime import FEATURE_FLAGS, RuntimeFeatureFlag
        
        for name, flag in FEATURE_FLAGS.items():
            self.assertIsInstance(flag, RuntimeFeatureFlag)
            self.assertEqual(flag.name, name)
            self.assertIsNotNone(flag.default)
            self.assertIsInstance(flag.description, str)
            self.assertGreater(len(flag.description), 10)
            self.assertIsNotNone(flag.stability)


class TestFeatureFlagStability(unittest.TestCase):
    """Tests for FeatureFlagStability enum."""

    def test_stability_values(self) -> None:
        """Stability enum should have expected values."""
        from experiments.u2.runtime import FeatureFlagStability
        
        self.assertEqual(FeatureFlagStability.STABLE.value, "stable")
        self.assertEqual(FeatureFlagStability.BETA.value, "beta")
        self.assertEqual(FeatureFlagStability.EXPERIMENTAL.value, "experimental")


class TestRuntimeFeatureFlagDataclass(unittest.TestCase):
    """Tests for RuntimeFeatureFlag dataclass."""

    def test_create_flag(self) -> None:
        """Should be able to create a RuntimeFeatureFlag."""
        from experiments.u2.runtime import RuntimeFeatureFlag, FeatureFlagStability
        
        flag = RuntimeFeatureFlag(
            name="test.flag",
            default=True,
            description="A test flag for unit testing.",
            stability=FeatureFlagStability.EXPERIMENTAL,
        )
        
        self.assertEqual(flag.name, "test.flag")
        self.assertTrue(flag.default)
        self.assertEqual(flag.stability, FeatureFlagStability.EXPERIMENTAL)

    def test_to_dict(self) -> None:
        """to_dict() should return JSON-serializable dict."""
        from experiments.u2.runtime import RuntimeFeatureFlag, FeatureFlagStability
        
        flag = RuntimeFeatureFlag(
            name="test.flag",
            default=False,
            description="Test description.",
            stability=FeatureFlagStability.BETA,
        )
        
        d = flag.to_dict()
        
        self.assertEqual(d["name"], "test.flag")
        self.assertFalse(d["default"])
        self.assertEqual(d["description"], "Test description.")
        self.assertEqual(d["stability"], "beta")
        
        # Should be JSON serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    def test_flag_is_frozen(self) -> None:
        """RuntimeFeatureFlag should be immutable (frozen)."""
        from experiments.u2.runtime import RuntimeFeatureFlag, FeatureFlagStability
        
        flag = RuntimeFeatureFlag(
            name="test.flag",
            default=True,
            description="Test.",
            stability=FeatureFlagStability.STABLE,
        )
        
        with self.assertRaises(Exception):  # FrozenInstanceError
            flag.default = False  # type: ignore


class TestGetFeatureFlag(unittest.TestCase):
    """Tests for get_feature_flag function."""

    def setUp(self) -> None:
        """Reset flag overrides before each test."""
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        """Reset flag overrides after each test."""
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_get_registered_flag_returns_default(self) -> None:
        """get_feature_flag should return registry default."""
        from experiments.u2.runtime import get_feature_flag
        
        # u2.use_cycle_orchestrator defaults to True
        value = get_feature_flag("u2.use_cycle_orchestrator")
        self.assertTrue(value)

    def test_get_unknown_flag_returns_none(self) -> None:
        """get_feature_flag should return None for unknown flags."""
        from experiments.u2.runtime import get_feature_flag
        
        value = get_feature_flag("unknown.flag")
        self.assertIsNone(value)

    def test_get_unknown_flag_with_default(self) -> None:
        """get_feature_flag should use provided default for unknown flags."""
        from experiments.u2.runtime import get_feature_flag
        
        value = get_feature_flag("unknown.flag", default="fallback")
        self.assertEqual(value, "fallback")


class TestSetFeatureFlag(unittest.TestCase):
    """Tests for set_feature_flag function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_set_flag_overrides_default(self) -> None:
        """set_feature_flag should override the default value."""
        from experiments.u2.runtime import get_feature_flag, set_feature_flag
        
        # Default is False
        self.assertFalse(get_feature_flag("u2.enable_extra_telemetry"))
        
        # Override to True
        set_feature_flag("u2.enable_extra_telemetry", True)
        self.assertTrue(get_feature_flag("u2.enable_extra_telemetry"))

    def test_set_unknown_flag_raises(self) -> None:
        """set_feature_flag should raise for unknown flags."""
        from experiments.u2.runtime import set_feature_flag
        
        with self.assertRaises(ValueError) as cm:
            set_feature_flag("unknown.flag", True)
        
        self.assertIn("Unknown feature flag", str(cm.exception))


class TestResetFeatureFlags(unittest.TestCase):
    """Tests for reset_feature_flags function."""

    def test_reset_clears_overrides(self) -> None:
        """reset_feature_flags should clear all overrides."""
        from experiments.u2.runtime import (
            get_feature_flag,
            set_feature_flag,
            reset_feature_flags,
        )
        
        # Override a flag
        set_feature_flag("u2.enable_extra_telemetry", True)
        self.assertTrue(get_feature_flag("u2.enable_extra_telemetry"))
        
        # Reset
        reset_feature_flags()
        
        # Should be back to default
        self.assertFalse(get_feature_flag("u2.enable_extra_telemetry"))


class TestListFeatureFlags(unittest.TestCase):
    """Tests for list_feature_flags function."""

    def test_list_returns_copy(self) -> None:
        """list_feature_flags should return a copy of the registry."""
        from experiments.u2.runtime import list_feature_flags, FEATURE_FLAGS
        
        flags = list_feature_flags()
        
        # Should be equal content
        self.assertEqual(set(flags.keys()), set(FEATURE_FLAGS.keys()))
        
        # Should be a copy (modifying shouldn't affect original)
        flags["test.new"] = None  # type: ignore
        self.assertNotIn("test.new", FEATURE_FLAGS)


class TestFeatureFlagsDefaultsBehavior(unittest.TestCase):
    """Tests that default flag values preserve current behavior."""

    def test_use_cycle_orchestrator_default_true(self) -> None:
        """u2.use_cycle_orchestrator should default to True (INV-RUN-1)."""
        from experiments.u2.runtime import get_feature_flag
        self.assertTrue(get_feature_flag("u2.use_cycle_orchestrator"))

    def test_enable_extra_telemetry_default_false(self) -> None:
        """u2.enable_extra_telemetry should default to False (no behavior change)."""
        from experiments.u2.runtime import get_feature_flag
        self.assertFalse(get_feature_flag("u2.enable_extra_telemetry"))

    def test_strict_input_validation_default_true(self) -> None:
        """u2.strict_input_validation should default to True (current behavior)."""
        from experiments.u2.runtime import get_feature_flag
        self.assertTrue(get_feature_flag("u2.strict_input_validation"))

    def test_trace_hash_chain_default_false(self) -> None:
        """u2.trace_hash_chain should default to False (opt-in feature)."""
        from experiments.u2.runtime import get_feature_flag
        self.assertFalse(get_feature_flag("u2.trace_hash_chain"))


class TestFeatureFlagsCLI(unittest.TestCase):
    """Tests for --show-feature-flags CLI option."""

    def test_show_feature_flags_exits_zero(self) -> None:
        """--show-feature-flags should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "experiments/u2_runtime_inspect.py", "--show-feature-flags"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

    def test_show_feature_flags_json_valid(self) -> None:
        """--show-feature-flags --json should return valid JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-feature-flags",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        self.assertIn("feature_flags", data)
        self.assertIn("total_count", data)
        self.assertIn("runtime_version", data)

    def test_show_feature_flags_sorted_order(self) -> None:
        """Feature flags should be listed in sorted order."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-feature-flags",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        
        flag_names = [f["name"] for f in data["feature_flags"]]
        self.assertEqual(flag_names, sorted(flag_names))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

