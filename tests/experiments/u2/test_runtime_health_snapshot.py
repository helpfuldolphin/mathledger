"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Health Snapshot Tests
================================

Tests for the runtime health snapshot, flag policy validation,
and global health hook in experiments.u2.runtime.

This module verifies:
- Health snapshot contract is deterministic and JSON-serializable
- Flag policy guards work correctly for dev/ci/prod environments
- Global health summary produces correct status
"""

from __future__ import annotations

import json
import unittest


class TestBuildRuntimeHealthSnapshot(unittest.TestCase):
    """Tests for build_runtime_health_snapshot function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_snapshot_has_required_fields(self) -> None:
        """Health snapshot should contain all required fields."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        snapshot = build_runtime_health_snapshot()
        
        required_fields = [
            "schema_version",
            "runtime_version",
            "active_flags",
            "flag_stabilities",
            "config_valid",
            "config_errors",
        ]
        
        for field in required_fields:
            self.assertIn(field, snapshot, f"Missing field: {field}")

    def test_snapshot_is_json_serializable(self) -> None:
        """Health snapshot must be JSON-serializable."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        snapshot = build_runtime_health_snapshot()
        
        # Should not raise
        json_str = json.dumps(snapshot)
        self.assertIsInstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        self.assertEqual(parsed["runtime_version"], snapshot["runtime_version"])

    def test_snapshot_is_deterministic(self) -> None:
        """Same inputs should produce same snapshot."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        s1 = build_runtime_health_snapshot()
        s2 = build_runtime_health_snapshot()
        
        self.assertEqual(s1["schema_version"], s2["schema_version"])
        self.assertEqual(s1["runtime_version"], s2["runtime_version"])
        self.assertEqual(s1["active_flags"], s2["active_flags"])

    def test_snapshot_reflects_flag_overrides(self) -> None:
        """Snapshot should reflect current flag values including overrides."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            set_feature_flag,
            get_feature_flag,
        )
        
        # Check default
        snapshot1 = build_runtime_health_snapshot()
        self.assertFalse(snapshot1["active_flags"]["u2.enable_extra_telemetry"])
        
        # Override flag
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        snapshot2 = build_runtime_health_snapshot()
        self.assertTrue(snapshot2["active_flags"]["u2.enable_extra_telemetry"])

    def test_snapshot_includes_stabilities(self) -> None:
        """Snapshot should include stability for each flag."""
        from experiments.u2.runtime import build_runtime_health_snapshot, FEATURE_FLAGS
        
        snapshot = build_runtime_health_snapshot()
        
        for name in FEATURE_FLAGS:
            self.assertIn(name, snapshot["flag_stabilities"])
            self.assertIn(snapshot["flag_stabilities"][name], ["stable", "beta", "experimental"])

    def test_snapshot_config_valid_without_path(self) -> None:
        """Config should be valid when no path provided."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        snapshot = build_runtime_health_snapshot()
        
        self.assertTrue(snapshot["config_valid"])
        self.assertEqual(snapshot["config_errors"], [])

    def test_snapshot_config_invalid_for_missing_file(self) -> None:
        """Config should be invalid for non-existent file."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        snapshot = build_runtime_health_snapshot(config_path="nonexistent_12345.yaml")
        
        self.assertFalse(snapshot["config_valid"])
        self.assertTrue(len(snapshot["config_errors"]) > 0)

    def test_snapshot_schema_version_is_string(self) -> None:
        """Schema version should be a string."""
        from experiments.u2.runtime import build_runtime_health_snapshot
        
        snapshot = build_runtime_health_snapshot()
        
        self.assertIsInstance(snapshot["schema_version"], str)
        # Should be semver-like
        parts = snapshot["schema_version"].split(".")
        self.assertEqual(len(parts), 3)


class TestValidateFlagPolicy(unittest.TestCase):
    """Tests for validate_flag_policy function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_invalid_env_raises_error(self) -> None:
        """Invalid environment context should raise ValueError."""
        from experiments.u2.runtime import validate_flag_policy
        
        with self.assertRaises(ValueError) as cm:
            validate_flag_policy("invalid")
        
        self.assertIn("Invalid env_context", str(cm.exception))

    def test_valid_envs_accepted(self) -> None:
        """Valid environment contexts should be accepted."""
        from experiments.u2.runtime import validate_flag_policy
        
        for env in ["dev", "ci", "prod"]:
            result = validate_flag_policy(env)
            self.assertIn("policy_ok", result)
            self.assertEqual(result["env_context"], env)

    def test_default_flags_pass_in_all_envs(self) -> None:
        """Default flag values should pass policy in all environments."""
        from experiments.u2.runtime import validate_flag_policy
        
        for env in ["dev", "ci", "prod"]:
            result = validate_flag_policy(env)
            self.assertTrue(result["policy_ok"], f"Failed in {env}: {result['violations']}")
            self.assertEqual(result["violations"], [])

    def test_beta_flag_on_in_dev_allowed(self) -> None:
        """BETA flags can be ON in dev environment."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.trace_hash_chain", True)  # BETA flag
        
        result = validate_flag_policy("dev")
        self.assertTrue(result["policy_ok"])

    def test_beta_flag_on_in_ci_violation(self) -> None:
        """BETA flags ON in ci should produce violation."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.trace_hash_chain", True)  # BETA flag
        
        result = validate_flag_policy("ci")
        self.assertFalse(result["policy_ok"])
        self.assertEqual(len(result["violations"]), 1)
        self.assertEqual(result["violations"][0]["flag_name"], "u2.trace_hash_chain")

    def test_beta_flag_on_in_prod_violation(self) -> None:
        """BETA flags ON in prod should produce violation."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.trace_hash_chain", True)  # BETA flag
        
        result = validate_flag_policy("prod")
        self.assertFalse(result["policy_ok"])

    def test_experimental_flag_on_in_dev_allowed(self) -> None:
        """EXPERIMENTAL flags can be ON in dev environment."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.enable_extra_telemetry", True)  # EXPERIMENTAL flag
        
        result = validate_flag_policy("dev")
        self.assertTrue(result["policy_ok"])

    def test_experimental_flag_on_in_ci_violation(self) -> None:
        """EXPERIMENTAL flags ON in ci should produce violation."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.enable_extra_telemetry", True)  # EXPERIMENTAL flag
        
        result = validate_flag_policy("ci")
        self.assertFalse(result["policy_ok"])
        self.assertEqual(result["violations"][0]["flag_name"], "u2.enable_extra_telemetry")

    def test_experimental_flag_on_in_prod_violation(self) -> None:
        """EXPERIMENTAL flags ON in prod should produce violation."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.enable_extra_telemetry", True)  # EXPERIMENTAL flag
        
        result = validate_flag_policy("prod")
        self.assertFalse(result["policy_ok"])

    def test_stable_flags_allowed_everywhere(self) -> None:
        """STABLE flags can be toggled in any environment."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        # Toggle a stable flag
        set_feature_flag("u2.strict_input_validation", False)
        
        for env in ["dev", "ci", "prod"]:
            result = validate_flag_policy(env)
            self.assertTrue(result["policy_ok"], f"STABLE flag blocked in {env}")

    def test_violation_includes_reason(self) -> None:
        """Violations should include human-readable reason."""
        from experiments.u2.runtime import validate_flag_policy, set_feature_flag
        
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        result = validate_flag_policy("prod")
        
        self.assertTrue(len(result["violations"]) > 0)
        violation = result["violations"][0]
        self.assertIn("reason", violation)
        self.assertIn("EXPERIMENTAL", violation["reason"])
        self.assertIn("prod", violation["reason"])


class TestSummarizeRuntimeForGlobalHealth(unittest.TestCase):
    """Tests for summarize_runtime_for_global_health function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_summary_has_required_fields(self) -> None:
        """Summary should contain all required fields."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        required_fields = [
            "runtime_ok",
            "flag_policy_ok",
            "beta_flags_active",
            "experimental_flags_active",
            "status",
        ]
        
        for field in required_fields:
            self.assertIn(field, summary, f"Missing field: {field}")

    def test_status_ok_when_all_good(self) -> None:
        """Status should be OK when everything is nominal."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        self.assertEqual(summary["status"], "OK")
        self.assertTrue(summary["runtime_ok"])
        self.assertTrue(summary["flag_policy_ok"])

    def test_status_warn_with_beta_flags(self) -> None:
        """Status should be WARN when beta flags are active."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
            set_feature_flag,
        )
        
        set_feature_flag("u2.trace_hash_chain", True)  # BETA flag
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")  # Allowed in dev
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        self.assertEqual(summary["status"], "WARN")
        self.assertIn("u2.trace_hash_chain", summary["beta_flags_active"])

    def test_status_block_with_policy_violation_in_prod(self) -> None:
        """Status should be BLOCK when policy violations in prod."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
            set_feature_flag,
        )
        
        set_feature_flag("u2.enable_extra_telemetry", True)  # EXPERIMENTAL flag
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")  # Violation in prod
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        self.assertEqual(summary["status"], "BLOCK")
        self.assertFalse(summary["flag_policy_ok"])

    def test_status_block_with_invalid_config(self) -> None:
        """Status should be BLOCK when config is invalid."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
        )
        
        snapshot = build_runtime_health_snapshot(config_path="nonexistent.yaml")
        policy = validate_flag_policy("dev")
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        self.assertEqual(summary["status"], "BLOCK")
        self.assertFalse(summary["runtime_ok"])

    def test_experimental_flags_tracked(self) -> None:
        """Experimental flags should be tracked in summary."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            summarize_runtime_for_global_health,
            set_feature_flag,
        )
        
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        summary = summarize_runtime_for_global_health(snapshot, policy)
        
        self.assertIn("u2.enable_extra_telemetry", summary["experimental_flags_active"])


class TestFlagPolicyViolationDataclass(unittest.TestCase):
    """Tests for FlagPolicyViolation dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should return JSON-serializable dict."""
        from experiments.u2.runtime import FlagPolicyViolation
        
        v = FlagPolicyViolation(
            flag_name="test.flag",
            stability="experimental",
            current_value=True,
            reason="Test reason",
        )
        
        d = v.to_dict()
        
        self.assertEqual(d["flag_name"], "test.flag")
        self.assertEqual(d["stability"], "experimental")
        self.assertEqual(d["current_value"], True)
        self.assertEqual(d["reason"], "Test reason")
        
        # Should be JSON serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


class TestHealthSnapshotConstants(unittest.TestCase):
    """Tests for health snapshot constants."""

    def test_schema_version_constant(self) -> None:
        """Schema version constant should be defined."""
        from experiments.u2.runtime import HEALTH_SNAPSHOT_SCHEMA_VERSION
        
        self.assertIsInstance(HEALTH_SNAPSHOT_SCHEMA_VERSION, str)
        self.assertTrue(len(HEALTH_SNAPSHOT_SCHEMA_VERSION) > 0)

    def test_valid_env_contexts_constant(self) -> None:
        """Valid env contexts constant should be defined."""
        from experiments.u2.runtime import VALID_ENV_CONTEXTS
        
        self.assertIn("dev", VALID_ENV_CONTEXTS)
        self.assertIn("ci", VALID_ENV_CONTEXTS)
        self.assertIn("prod", VALID_ENV_CONTEXTS)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

