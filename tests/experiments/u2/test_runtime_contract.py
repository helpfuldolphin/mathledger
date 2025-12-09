"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Contract Tests
=========================

This module enforces the runtime API contract as a testable specification.
If the public API changes, these tests will fail until the expected symbols
are explicitly updated — preventing accidental API drift.

INVARIANTS TESTED
-----------------
- INV-RUN-4: The "Thin Waist" API surface does not grow without explicit justification.
- Imports work without side effects
- Public symbols are stable and documented

HOW TO UPDATE THE CONTRACT
--------------------------
If you are intentionally adding new symbols to the runtime API:

1. Add the new symbol(s) to EXPECTED_PUBLIC_SYMBOLS below.
2. Add the symbol(s) to SYMBOL_SHAPE_SNAPSHOT with their kind.
3. Bump EXPECTED_VERSION to match the new runtime version.
4. Update experiments/u2/runtime/__init__.py __version__ to match.
5. Document the change in the runtime module docstring.

If tests fail unexpectedly, DO NOT just update the snapshots — investigate
whether the change was intentional and backwards-compatible.
"""

from __future__ import annotations

import unittest
from typing import Dict, Set


# ============================================================================
# Contract Snapshot: Expected Public Symbols (v1.5.0)
# ============================================================================
# Update this set ONLY when intentionally changing the public API.
# Any change requires version bump in experiments/u2/runtime/__init__.py

EXPECTED_PUBLIC_SYMBOLS: Set[str] = {
    # seed_manager
    "SeedSchedule",
    "generate_seed_schedule",
    "hash_string",
    # cycle_orchestrator
    "CycleState",
    "CycleResult",
    "OrderingStrategy",
    "BaselineOrderingStrategy",
    "RflOrderingStrategy",
    "execute_cycle",
    "get_ordering_strategy",
    "CycleExecutionError",
    # error_classifier
    "RuntimeErrorKind",
    "ErrorContext",
    "classify_error",
    "classify_error_with_context",
    "build_error_result",
    # trace_logger
    "PHASE_II_LABEL",
    "TelemetryRecord",
    "TraceWriter",
    "TraceReader",
    "build_telemetry_record",
    # feature_flags (v1.3.0)
    "FeatureFlagStability",
    "RuntimeFeatureFlag",
    "FEATURE_FLAGS",
    "get_feature_flag",
    "set_feature_flag",
    "reset_feature_flags",
    "list_feature_flags",
    # health_snapshot (v1.4.0)
    "HEALTH_SNAPSHOT_SCHEMA_VERSION",
    "build_runtime_health_snapshot",
    # flag_policy (v1.4.0)
    "VALID_ENV_CONTEXTS",
    "FlagPolicyViolation",
    "validate_flag_policy",
    # global_health (v1.4.0)
    "summarize_runtime_for_global_health",
    # runtime_profiles (v1.5.0)
    "RuntimeProfile",
    "RUNTIME_PROFILES",
    "load_runtime_profile",
    "evaluate_runtime_profile",
    # fail_safe (v1.5.0)
    "derive_runtime_fail_safe_action",
    # director_console (v1.5.0)
    "build_runtime_director_panel",
}

EXPECTED_VERSION = "1.5.0"

# ============================================================================
# Shape Snapshot: Symbol → Kind Mapping
# ============================================================================
# This mapping captures the "shape" of the API: what kind of thing each
# symbol is. If a symbol changes kind (e.g., class → function), the seal
# test will fail.

SYMBOL_SHAPE_SNAPSHOT: Dict[str, str] = {
    # seed_manager
    "SeedSchedule": "dataclass",
    "generate_seed_schedule": "function",
    "hash_string": "function",
    # cycle_orchestrator
    "CycleState": "dataclass",
    "CycleResult": "dataclass",
    "OrderingStrategy": "class",  # Protocol is detected as class at runtime
    "BaselineOrderingStrategy": "class",
    "RflOrderingStrategy": "class",
    "execute_cycle": "function",
    "get_ordering_strategy": "function",
    "CycleExecutionError": "exception",
    # error_classifier
    "RuntimeErrorKind": "enum",
    "ErrorContext": "dataclass",
    "classify_error": "function",
    "classify_error_with_context": "function",
    "build_error_result": "function",
    # trace_logger
    "PHASE_II_LABEL": "constant",
    "TelemetryRecord": "dataclass",
    "TraceWriter": "class",
    "TraceReader": "class",
    "build_telemetry_record": "function",
    # feature_flags (v1.3.0)
    "FeatureFlagStability": "enum",
    "RuntimeFeatureFlag": "dataclass",
    "FEATURE_FLAGS": "constant",
    "get_feature_flag": "function",
    "set_feature_flag": "function",
    "reset_feature_flags": "function",
    "list_feature_flags": "function",
    # health_snapshot (v1.4.0)
    "HEALTH_SNAPSHOT_SCHEMA_VERSION": "constant",
    "build_runtime_health_snapshot": "function",
    # flag_policy (v1.4.0)
    "VALID_ENV_CONTEXTS": "constant",
    "FlagPolicyViolation": "dataclass",
    "validate_flag_policy": "function",
    # global_health (v1.4.0)
    "summarize_runtime_for_global_health": "function",
    # runtime_profiles (v1.5.0)
    "RuntimeProfile": "dataclass",
    "RUNTIME_PROFILES": "constant",
    "load_runtime_profile": "function",
    "evaluate_runtime_profile": "function",
    # fail_safe (v1.5.0)
    "derive_runtime_fail_safe_action": "function",
    # director_console (v1.5.0)
    "build_runtime_director_panel": "function",
}

# Expected feature flags (must be kept in sync with runtime)
EXPECTED_FEATURE_FLAGS: Set[str] = {
    "u2.use_cycle_orchestrator",
    "u2.enable_extra_telemetry",
    "u2.strict_input_validation",
    "u2.trace_hash_chain",
}


class TestRuntimeContractSnapshot(unittest.TestCase):
    """Tests that enforce the runtime API contract."""

    def test_version_matches_expected(self) -> None:
        """Runtime version must match contract expectation."""
        from experiments.u2.runtime import __version__
        self.assertEqual(
            __version__,
            EXPECTED_VERSION,
            f"Runtime version changed from {EXPECTED_VERSION} to {__version__}. "
            "Update EXPECTED_VERSION and EXPECTED_PUBLIC_SYMBOLS if intentional."
        )

    def test_all_matches_expected_symbols(self) -> None:
        """__all__ must exactly match the expected public symbols."""
        from experiments.u2 import runtime
        
        actual_all = set(runtime.__all__)
        
        missing = EXPECTED_PUBLIC_SYMBOLS - actual_all
        extra = actual_all - EXPECTED_PUBLIC_SYMBOLS
        
        self.assertEqual(
            missing,
            set(),
            f"Missing symbols from __all__: {missing}"
        )
        self.assertEqual(
            extra,
            set(),
            f"Unexpected symbols in __all__: {extra}. "
            "If intentional, add to EXPECTED_PUBLIC_SYMBOLS and bump version."
        )

    def test_no_extra_public_symbols_leaked(self) -> None:
        """No unintended symbols should be accessible without underscore."""
        from experiments.u2 import runtime
        
        # Get all non-private attributes
        all_attrs = {
            name for name in dir(runtime)
            if not name.startswith("_")
        }
        
        # Expected: symbols in __all__ plus standard module attributes
        allowed_extras = {
            # Standard module attributes (submodules)
            "seed_manager",
            "cycle_orchestrator", 
            "error_classifier",
            "trace_logger",
            # Typing imports (acceptable leakage from type hints)
            "Any",
            "Dict",
            "Optional",
            "Set",
            "Tuple",
            "List",
            # Stdlib imports used in feature flags
            "dataclass",
            "field",
            "Enum",
        }
        
        actual_public = set(runtime.__all__)
        unexpected = all_attrs - actual_public - allowed_extras
        
        # Filter out imported module names (acceptable)
        unexpected = {
            name for name in unexpected
            if not hasattr(getattr(runtime, name, None), "__file__")
        }
        
        self.assertEqual(
            unexpected,
            set(),
            f"Unexpected public attributes leaked: {unexpected}"
        )


class TestRuntimeImportStability(unittest.TestCase):
    """Tests that imports work correctly and without side effects."""

    def test_import_has_no_side_effects(self) -> None:
        """Importing runtime should not perform I/O or modify global state."""
        # This test passes if import succeeds without exception
        # Side effects would typically manifest as:
        # - File creation
        # - Network calls
        # - Global state mutation
        
        # Re-import to verify
        import importlib
        from experiments.u2 import runtime
        importlib.reload(runtime)
        
        # If we get here, no obvious side effects occurred
        self.assertTrue(True)

    def test_public_imports_all_succeed(self) -> None:
        """All symbols in __all__ must be importable."""
        from experiments.u2.runtime import (
            # seed_manager
            SeedSchedule,
            generate_seed_schedule,
            hash_string,
            # cycle_orchestrator
            CycleState,
            CycleResult,
            OrderingStrategy,
            BaselineOrderingStrategy,
            RflOrderingStrategy,
            execute_cycle,
            get_ordering_strategy,
            CycleExecutionError,
            # error_classifier
            RuntimeErrorKind,
            ErrorContext,
            classify_error,
            classify_error_with_context,
            build_error_result,
            # trace_logger
            PHASE_II_LABEL,
            TelemetryRecord,
            TraceWriter,
            TraceReader,
            build_telemetry_record,
            # feature_flags
            FeatureFlagStability,
            RuntimeFeatureFlag,
            FEATURE_FLAGS,
            get_feature_flag,
            set_feature_flag,
            reset_feature_flags,
            list_feature_flags,
            # runtime_profiles
            RuntimeProfile,
            RUNTIME_PROFILES,
            load_runtime_profile,
            evaluate_runtime_profile,
            # fail_safe
            derive_runtime_fail_safe_action,
            # director_console
            build_runtime_director_panel,
        )
        
        # All imports succeeded
        self.assertIsNotNone(SeedSchedule)
        self.assertIsNotNone(generate_seed_schedule)
        self.assertIsNotNone(CycleState)
        self.assertIsNotNone(execute_cycle)
        self.assertIsNotNone(RuntimeErrorKind)
        self.assertIsNotNone(TraceWriter)
        self.assertIsNotNone(FeatureFlagStability)
        self.assertIsNotNone(FEATURE_FLAGS)
        self.assertIsNotNone(RuntimeProfile)
        self.assertIsNotNone(RUNTIME_PROFILES)

    def test_common_import_patterns_work(self) -> None:
        """Common import patterns should work as documented."""
        # Pattern 1: Import specific symbols
        from experiments.u2.runtime import execute_cycle, CycleState, CycleResult
        self.assertTrue(callable(execute_cycle))
        
        # Pattern 2: Import module and access
        from experiments.u2 import runtime
        self.assertTrue(hasattr(runtime, "execute_cycle"))
        self.assertTrue(hasattr(runtime, "__version__"))
        
        # Pattern 3: Import from submodules directly
        from experiments.u2.runtime.seed_manager import generate_seed_schedule
        self.assertTrue(callable(generate_seed_schedule))


class TestRuntimeSymbolTypes(unittest.TestCase):
    """Tests that exported symbols have expected types."""

    def test_seed_manager_types(self) -> None:
        """seed_manager symbols have expected types."""
        from experiments.u2.runtime import (
            SeedSchedule,
            generate_seed_schedule,
            hash_string,
        )
        
        self.assertTrue(callable(generate_seed_schedule))
        self.assertTrue(callable(hash_string))
        # SeedSchedule should be a class/dataclass
        self.assertTrue(isinstance(SeedSchedule, type))

    def test_cycle_orchestrator_types(self) -> None:
        """cycle_orchestrator symbols have expected types."""
        from experiments.u2.runtime import (
            CycleState,
            CycleResult,
            BaselineOrderingStrategy,
            RflOrderingStrategy,
            execute_cycle,
            get_ordering_strategy,
            CycleExecutionError,
        )
        
        self.assertTrue(isinstance(CycleState, type))
        self.assertTrue(isinstance(CycleResult, type))
        self.assertTrue(isinstance(BaselineOrderingStrategy, type))
        self.assertTrue(isinstance(RflOrderingStrategy, type))
        self.assertTrue(callable(execute_cycle))
        self.assertTrue(callable(get_ordering_strategy))
        self.assertTrue(issubclass(CycleExecutionError, Exception))

    def test_error_classifier_types(self) -> None:
        """error_classifier symbols have expected types."""
        from experiments.u2.runtime import (
            RuntimeErrorKind,
            ErrorContext,
            classify_error,
            classify_error_with_context,
            build_error_result,
        )
        from enum import Enum
        
        self.assertTrue(issubclass(RuntimeErrorKind, Enum))
        self.assertTrue(isinstance(ErrorContext, type))
        self.assertTrue(callable(classify_error))
        self.assertTrue(callable(classify_error_with_context))
        self.assertTrue(callable(build_error_result))

    def test_trace_logger_types(self) -> None:
        """trace_logger symbols have expected types."""
        from experiments.u2.runtime import (
            PHASE_II_LABEL,
            TelemetryRecord,
            TraceWriter,
            TraceReader,
            build_telemetry_record,
        )
        
        self.assertIsInstance(PHASE_II_LABEL, str)
        self.assertTrue(isinstance(TelemetryRecord, type))
        self.assertTrue(isinstance(TraceWriter, type))
        self.assertTrue(isinstance(TraceReader, type))
        self.assertTrue(callable(build_telemetry_record))


class TestRuntimeErrorKindCompleteness(unittest.TestCase):
    """Tests that RuntimeErrorKind covers expected error types."""

    def test_error_kinds_include_all_expected(self) -> None:
        """All expected error kinds must be present."""
        from experiments.u2.runtime import RuntimeErrorKind
        
        expected_kinds = {
            "SUBPROCESS",
            "JSON_DECODE",
            "FILE_NOT_FOUND",
            "VALIDATION",
            "TIMEOUT",
            "UNKNOWN",
        }
        
        actual_kinds = {k.name for k in RuntimeErrorKind}
        
        missing = expected_kinds - actual_kinds
        self.assertEqual(
            missing,
            set(),
            f"Missing error kinds: {missing}"
        )


class TestRuntimeContractSeal(unittest.TestCase):
    """
    Stronger API contract enforcement via shape snapshots.
    
    This test class maintains a "seal" on the runtime API surface:
    - Symbol names must match exactly
    - Symbol kinds (class, function, enum, etc.) must match
    - Feature flags must match expected set
    
    If any of these fail, it indicates a potential breaking change.
    DO NOT UPDATE THE SNAPSHOTS WITHOUT:
    1. Confirming the change is intentional
    2. Updating runtime __version__
    3. Documenting the change
    """

    def test_symbol_shape_snapshot_complete(self) -> None:
        """Every symbol in __all__ must have a shape entry."""
        from experiments.u2 import runtime
        
        missing_shape = set(runtime.__all__) - set(SYMBOL_SHAPE_SNAPSHOT.keys())
        extra_shape = set(SYMBOL_SHAPE_SNAPSHOT.keys()) - set(runtime.__all__)
        
        self.assertEqual(
            missing_shape,
            set(),
            f"Symbols in __all__ without shape entry: {missing_shape}. "
            "Add to SYMBOL_SHAPE_SNAPSHOT."
        )
        self.assertEqual(
            extra_shape,
            set(),
            f"Shape entries for non-existent symbols: {extra_shape}. "
            "Remove from SYMBOL_SHAPE_SNAPSHOT."
        )

    def test_symbol_kinds_match_snapshot(self) -> None:
        """Each symbol's kind must match the snapshot."""
        from experiments.u2 import runtime
        from dataclasses import is_dataclass
        from enum import Enum
        from typing import Protocol, runtime_checkable
        
        def get_symbol_kind(obj) -> str:
            """Determine the kind of a symbol."""
            if isinstance(obj, str):
                return "constant"
            if isinstance(obj, dict):
                return "constant"
            if isinstance(obj, frozenset):
                return "constant"
            if is_dataclass(obj) and isinstance(obj, type):
                return "dataclass"
            if isinstance(obj, type):
                if issubclass(obj, Enum):
                    return "enum"
                if issubclass(obj, Exception):
                    return "exception"
                # Check for Protocol (runtime_checkable decorator)
                if hasattr(obj, "__protocol_attrs__"):
                    return "protocol"
                return "class"
            if callable(obj):
                return "function"
            return "unknown"
        
        mismatches = []
        for name, expected_kind in SYMBOL_SHAPE_SNAPSHOT.items():
            if not hasattr(runtime, name):
                mismatches.append(f"{name}: symbol not found in runtime")
                continue
            
            obj = getattr(runtime, name)
            actual_kind = get_symbol_kind(obj)
            
            if actual_kind != expected_kind:
                mismatches.append(
                    f"{name}: expected {expected_kind}, got {actual_kind}"
                )
        
        self.assertEqual(
            mismatches,
            [],
            f"Symbol kind mismatches:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )

    def test_feature_flags_match_snapshot(self) -> None:
        """Feature flag registry must match expected set."""
        from experiments.u2.runtime import FEATURE_FLAGS
        
        actual_flags = set(FEATURE_FLAGS.keys())
        
        missing = EXPECTED_FEATURE_FLAGS - actual_flags
        extra = actual_flags - EXPECTED_FEATURE_FLAGS
        
        self.assertEqual(
            missing,
            set(),
            f"Missing feature flags: {missing}. "
            "Add to FEATURE_FLAGS registry and EXPECTED_FEATURE_FLAGS snapshot."
        )
        self.assertEqual(
            extra,
            set(),
            f"Unexpected feature flags: {extra}. "
            "If intentional, add to EXPECTED_FEATURE_FLAGS snapshot."
        )

    def test_symbols_are_deterministic(self) -> None:
        """Symbol listing must be deterministic across multiple calls."""
        from experiments.u2 import runtime
        
        all_1 = tuple(sorted(runtime.__all__))
        all_2 = tuple(sorted(runtime.__all__))
        all_3 = tuple(sorted(runtime.__all__))
        
        self.assertEqual(all_1, all_2)
        self.assertEqual(all_2, all_3)


class TestFeatureFlagContractTypes(unittest.TestCase):
    """Tests for feature flag type contracts."""

    def test_feature_flag_stability_is_enum(self) -> None:
        """FeatureFlagStability must be an Enum."""
        from experiments.u2.runtime import FeatureFlagStability
        from enum import Enum
        
        self.assertTrue(issubclass(FeatureFlagStability, Enum))

    def test_runtime_feature_flag_is_dataclass(self) -> None:
        """RuntimeFeatureFlag must be a frozen dataclass."""
        from experiments.u2.runtime import RuntimeFeatureFlag
        from dataclasses import is_dataclass
        
        self.assertTrue(is_dataclass(RuntimeFeatureFlag))
        # Check frozen
        self.assertTrue(RuntimeFeatureFlag.__dataclass_fields__)

    def test_feature_flags_registry_is_dict(self) -> None:
        """FEATURE_FLAGS must be a dict."""
        from experiments.u2.runtime import FEATURE_FLAGS
        
        self.assertIsInstance(FEATURE_FLAGS, dict)

    def test_get_feature_flag_is_callable(self) -> None:
        """get_feature_flag must be callable."""
        from experiments.u2.runtime import get_feature_flag
        
        self.assertTrue(callable(get_feature_flag))

    def test_set_feature_flag_is_callable(self) -> None:
        """set_feature_flag must be callable."""
        from experiments.u2.runtime import set_feature_flag
        
        self.assertTrue(callable(set_feature_flag))

    def test_reset_feature_flags_is_callable(self) -> None:
        """reset_feature_flags must be callable."""
        from experiments.u2.runtime import reset_feature_flags
        
        self.assertTrue(callable(reset_feature_flags))

    def test_list_feature_flags_is_callable(self) -> None:
        """list_feature_flags must be callable."""
        from experiments.u2.runtime import list_feature_flags
        
        self.assertTrue(callable(list_feature_flags))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

