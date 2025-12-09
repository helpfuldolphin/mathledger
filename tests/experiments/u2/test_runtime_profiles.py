"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Profile Tests
========================

Tests for runtime profiles, fail-safe modes, and director console integration
in experiments.u2.runtime.

This module verifies:
- Runtime profiles can be loaded and evaluated
- Profile violations are correctly detected
- Fail-safe actions are derived correctly
- Director console panel is built correctly
"""

from __future__ import annotations

import json
import unittest


class TestRuntimeProfileDataclass(unittest.TestCase):
    """Tests for RuntimeProfile dataclass."""

    def test_create_profile(self) -> None:
        """Should be able to create a RuntimeProfile."""
        from experiments.u2.runtime import RuntimeProfile
        
        profile = RuntimeProfile(
            name="test-profile",
            description="Test profile",
            expected_env_context="dev",
            required_flags={"u2.use_cycle_orchestrator"},
        )
        
        self.assertEqual(profile.name, "test-profile")
        self.assertEqual(profile.expected_env_context, "dev")

    def test_invalid_env_context_raises(self) -> None:
        """Invalid env_context should raise ValueError."""
        from experiments.u2.runtime import RuntimeProfile
        
        with self.assertRaises(ValueError) as cm:
            RuntimeProfile(
                name="test",
                description="Test",
                expected_env_context="invalid",
            )
        
        self.assertIn("Invalid expected_env_context", str(cm.exception))

    def test_conflicting_required_and_forbidden_raises(self) -> None:
        """Flags cannot be both required and forbidden."""
        from experiments.u2.runtime import RuntimeProfile
        
        with self.assertRaises(ValueError) as cm:
            RuntimeProfile(
                name="test",
                description="Test",
                expected_env_context="dev",
                required_flags={"u2.use_cycle_orchestrator"},
                forbidden_flags={"u2.use_cycle_orchestrator"},
            )
        
        self.assertIn("cannot be both required and forbidden", str(cm.exception))

    def test_forbidden_combinations_with_unknown_flags_raises(self) -> None:
        """Forbidden combinations must reference valid flags."""
        from experiments.u2.runtime import RuntimeProfile
        
        with self.assertRaises(ValueError) as cm:
            RuntimeProfile(
                name="test",
                description="Test",
                expected_env_context="dev",
                forbidden_combinations=[("unknown.flag", "another.unknown")],
            )
        
        self.assertIn("references unknown flags", str(cm.exception))


class TestLoadRuntimeProfile(unittest.TestCase):
    """Tests for load_runtime_profile function."""

    def test_load_existing_profile(self) -> None:
        """Should load an existing profile."""
        from experiments.u2.runtime import load_runtime_profile
        
        profile = load_runtime_profile("dev-default")
        
        self.assertEqual(profile.name, "dev-default")
        self.assertEqual(profile.expected_env_context, "dev")

    def test_load_nonexistent_profile_raises(self) -> None:
        """Loading nonexistent profile should raise ValueError."""
        from experiments.u2.runtime import load_runtime_profile
        
        with self.assertRaises(ValueError) as cm:
            load_runtime_profile("nonexistent-profile")
        
        self.assertIn("Unknown runtime profile", str(cm.exception))

    def test_all_builtin_profiles_loadable(self) -> None:
        """All built-in profiles should be loadable."""
        from experiments.u2.runtime import load_runtime_profile, RUNTIME_PROFILES
        
        for name in RUNTIME_PROFILES.keys():
            profile = load_runtime_profile(name)
            self.assertEqual(profile.name, name)


class TestEvaluateRuntimeProfile(unittest.TestCase):
    """Tests for evaluate_runtime_profile function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_profile_passes_when_all_checks_pass(self) -> None:
        """Profile should pass when all checks pass."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertTrue(result["profile_ok"])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["violations"], [])

    def test_profile_fails_when_required_flag_off(self) -> None:
        """Profile should fail when required flag is OFF."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Turn off required flag
        set_feature_flag("u2.use_cycle_orchestrator", False)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertFalse(result["profile_ok"])
        self.assertIn("Required flag", result["violations"][0])

    def test_profile_fails_when_forbidden_flag_on(self) -> None:
        """Profile should fail when forbidden flag is ON."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Turn on forbidden flag
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")
        profile = load_runtime_profile("prod-hardened")
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertFalse(result["profile_ok"])
        self.assertIn("Forbidden flag", result["violations"][0])

    def test_profile_fails_on_env_mismatch(self) -> None:
        """Profile should fail when environment doesn't match."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")  # Wrong env
        profile = load_runtime_profile("dev-default")  # Expects dev
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertFalse(result["profile_ok"])
        self.assertIn("Environment mismatch", result["violations"][0])

    def test_profile_status_block_in_prod_with_violations(self) -> None:
        """Profile status should be BLOCK in prod with violations."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")
        profile = load_runtime_profile("prod-hardened")
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertEqual(result["status"], "BLOCK")

    def test_profile_status_warn_in_dev_with_violations(self) -> None:
        """Profile status should be WARN in dev with violations."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Turn off required flag
        set_feature_flag("u2.use_cycle_orchestrator", False)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        
        result = evaluate_runtime_profile(profile, snapshot, policy)
        
        self.assertEqual(result["status"], "WARN")


class TestDeriveRuntimeFailSafeAction(unittest.TestCase):
    """Tests for derive_runtime_fail_safe_action function."""

    def test_action_allow_when_profile_ok(self) -> None:
        """Action should be ALLOW when profile passes."""
        from experiments.u2.runtime import derive_runtime_fail_safe_action
        
        profile_eval = {
            "profile_ok": True,
            "violations": [],
            "status": "OK",
        }
        
        result = derive_runtime_fail_safe_action(profile_eval)
        
        self.assertEqual(result["action"], "ALLOW")
        self.assertIn("passed", result["reason"])

    def test_action_no_run_with_critical_violations(self) -> None:
        """Action should be NO_RUN with critical violations."""
        from experiments.u2.runtime import derive_runtime_fail_safe_action
        
        profile_eval = {
            "profile_ok": False,
            "violations": ["Required flag 'u2.use_cycle_orchestrator' is OFF"],
            "status": "BLOCK",
        }
        
        result = derive_runtime_fail_safe_action(profile_eval)
        
        self.assertEqual(result["action"], "NO_RUN")
        self.assertIn("Critical", result["reason"])

    def test_action_no_run_with_forbidden_flag(self) -> None:
        """Action should be NO_RUN when forbidden flag is ON."""
        from experiments.u2.runtime import derive_runtime_fail_safe_action
        
        profile_eval = {
            "profile_ok": False,
            "violations": ["Forbidden flag 'u2.enable_extra_telemetry' is ON"],
            "status": "BLOCK",
        }
        
        result = derive_runtime_fail_safe_action(profile_eval)
        
        self.assertEqual(result["action"], "NO_RUN")

    def test_action_safe_degrade_with_non_critical(self) -> None:
        """Action should be SAFE_DEGRADE with non-critical violations."""
        from experiments.u2.runtime import derive_runtime_fail_safe_action
        
        profile_eval = {
            "profile_ok": False,
            "violations": ["Optional feature X is not enabled"],
            "status": "WARN",
        }
        
        result = derive_runtime_fail_safe_action(profile_eval)
        
        self.assertEqual(result["action"], "SAFE_DEGRADE")
        self.assertIn("Non-critical", result["reason"])

    def test_action_no_run_with_block_status(self) -> None:
        """Action should be NO_RUN when status is BLOCK."""
        from experiments.u2.runtime import derive_runtime_fail_safe_action
        
        profile_eval = {
            "profile_ok": False,
            "violations": ["Some violation"],
            "status": "BLOCK",
        }
        
        result = derive_runtime_fail_safe_action(profile_eval)
        
        self.assertEqual(result["action"], "NO_RUN")


class TestBuildRuntimeDirectorPanel(unittest.TestCase):
    """Tests for build_runtime_director_panel function."""

    def setUp(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def tearDown(self) -> None:
        from experiments.u2.runtime import reset_feature_flags
        reset_feature_flags()

    def test_panel_has_required_fields(self) -> None:
        """Panel should contain all required fields."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        panel = build_runtime_director_panel(snapshot, policy)
        
        required_fields = [
            "runtime_version",
            "env_context",
            "status_light",
            "action",
            "key_violations",
        ]
        
        for field in required_fields:
            self.assertIn(field, panel, f"Missing field: {field}")

    def test_panel_is_json_serializable(self) -> None:
        """Panel must be JSON-serializable."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        panel = build_runtime_director_panel(snapshot, policy)
        
        # Should not raise
        json_str = json.dumps(panel)
        self.assertIsInstance(json_str, str)

    def test_status_light_green_when_ok(self) -> None:
        """Status light should be GREEN when status is OK."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        panel = build_runtime_director_panel(snapshot, policy)
        
        self.assertEqual(panel["status_light"], "GREEN")

    def test_status_light_yellow_when_warn(self) -> None:
        """Status light should be YELLOW when status is WARN."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Create a WARN condition
        set_feature_flag("u2.trace_hash_chain", True)  # BETA flag
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval)
        
        # Should be YELLOW due to beta flag active
        self.assertEqual(panel["status_light"], "YELLOW")

    def test_status_light_red_when_block(self) -> None:
        """Status light should be RED when status is BLOCK."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Create a BLOCK condition
        set_feature_flag("u2.enable_extra_telemetry", True)  # EXPERIMENTAL
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")
        profile = load_runtime_profile("prod-hardened")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval)
        
        self.assertEqual(panel["status_light"], "RED")

    def test_action_allows_when_profile_ok(self) -> None:
        """Action should be ALLOW when profile passes."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
            derive_runtime_fail_safe_action,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        fail_safe = derive_runtime_fail_safe_action(profile_eval)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval, fail_safe)
        
        self.assertEqual(panel["action"], "ALLOW")

    def test_action_no_run_with_critical_violations(self) -> None:
        """Action should be NO_RUN with critical violations."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
            derive_runtime_fail_safe_action,
            set_feature_flag,
        )
        
        # Create critical violation
        set_feature_flag("u2.use_cycle_orchestrator", False)  # Required flag OFF
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")
        profile = load_runtime_profile("prod-hardened")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        fail_safe = derive_runtime_fail_safe_action(profile_eval)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval, fail_safe)
        
        self.assertEqual(panel["action"], "NO_RUN")

    def test_key_violations_limited_to_top_3(self) -> None:
        """Key violations should be limited to top 3."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
            set_feature_flag,
        )
        
        # Create multiple violations
        set_feature_flag("u2.use_cycle_orchestrator", False)
        set_feature_flag("u2.enable_extra_telemetry", True)
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("prod")
        profile = load_runtime_profile("prod-hardened")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval)
        
        self.assertLessEqual(len(panel["key_violations"]), 3)

    def test_profile_name_included_when_provided(self) -> None:
        """Profile name should be included when profile_eval is provided."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
            load_runtime_profile,
            evaluate_runtime_profile,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        profile = load_runtime_profile("dev-default")
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy)
        
        panel = build_runtime_director_panel(snapshot, policy, profile_eval)
        
        self.assertEqual(panel["profile_name"], "dev-default")

    def test_panel_without_profile_eval(self) -> None:
        """Panel should work without profile_eval."""
        from experiments.u2.runtime import (
            build_runtime_health_snapshot,
            validate_flag_policy,
            build_runtime_director_panel,
        )
        
        snapshot = build_runtime_health_snapshot()
        policy = validate_flag_policy("dev")
        panel = build_runtime_director_panel(snapshot, policy)
        
        self.assertIsNone(panel["profile_name"])
        self.assertIn("status_light", panel)


class TestBuiltinRuntimeProfiles(unittest.TestCase):
    """Tests for built-in runtime profiles."""

    def test_dev_default_profile(self) -> None:
        """dev-default profile should have correct settings."""
        from experiments.u2.runtime import load_runtime_profile
        
        profile = load_runtime_profile("dev-default")
        
        self.assertEqual(profile.expected_env_context, "dev")
        self.assertIn("u2.use_cycle_orchestrator", profile.required_flags)

    def test_ci_strict_profile(self) -> None:
        """ci-strict profile should forbid experimental flags."""
        from experiments.u2.runtime import load_runtime_profile
        
        profile = load_runtime_profile("ci-strict")
        
        self.assertEqual(profile.expected_env_context, "ci")
        self.assertIn("u2.enable_extra_telemetry", profile.forbidden_flags)

    def test_prod_hardened_profile(self) -> None:
        """prod-hardened profile should only allow stable flags."""
        from experiments.u2.runtime import load_runtime_profile
        
        profile = load_runtime_profile("prod-hardened")
        
        self.assertEqual(profile.expected_env_context, "prod")
        self.assertIn("u2.enable_extra_telemetry", profile.forbidden_flags)
        self.assertIn("u2.trace_hash_chain", profile.forbidden_flags)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

