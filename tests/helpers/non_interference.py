"""
Non-Interference Assertion Helpers for SHADOW MODE Signal Isolation.

Provides reusable assertion functions for verifying:
- Signal presence/absence only affects expected keys
- Warning list changes are bounded (delta <= N)
- Warning ordering is preserved for base warnings
- Adapter functions are pure (no side effects on input)

Reusable across all SHADOW MODE signal adapters and governance artifacts.

SHADOW MODE CONTRACT:
- All functions are observational verification only
- No enforcement, no gating
- Used for regression tripwires

Usage:
    from tests.helpers.non_interference import (
        pytest_assert_only_keys_changed,
        pytest_assert_warning_delta_at_most_one,
        pytest_assert_adapter_is_pure,
        pytest_assert_output_excludes_keys,
    )

Path Convention
---------------
All ``allowed_paths`` use dot-separated notation for nested dict keys.

Supported patterns:
- Exact match: ``"governance.budget_risk"`` matches only that key
- Wildcard suffix: ``"governance.budget_risk.*"`` matches key and all children
- Parent inference: If child is allowed, parent is implicitly allowed

Examples::

    # Example 1: Allow only budget_risk and its children to change
    allowed_paths = ["governance.budget_risk.*"]
    # Matches: governance.budget_risk, governance.budget_risk.calibration_reference
    # Also allows: governance (parent of allowed child)

    # Example 2: Allow specific signal keys in status output
    allowed_paths = [
        "signals.budget_calibration.*",
        "signals.identity_preflight.*",
    ]
    # Matches: signals.budget_calibration, signals.budget_calibration.fp_rate
    # Does NOT match: signals.divergence

    # Example 3: Allow exact key only (no children)
    allowed_paths = ["warnings"]
    # Matches: warnings
    # Does NOT match: warnings.count (if warnings were a dict)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


@dataclass
class NonInterferenceResult:
    """Result of a non-interference check."""

    passed: bool
    message: str
    violations: List[str] = field(default_factory=list)


# =============================================================================
# Key Path Helpers
# =============================================================================

def _get_nested_keys(
    obj: Dict[str, Any],
    prefix: str = "",
) -> Set[str]:
    """
    Get all nested key paths from a dict.

    Args:
        obj: Dict to extract keys from
        prefix: Current path prefix (for recursion)

    Returns:
        Set of dot-separated key paths (e.g., {"governance.budget_risk", "governance.divergence"})
    """
    keys: Set[str] = set()
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        keys.add(path)
        if isinstance(value, dict):
            keys.update(_get_nested_keys(value, path))
    return keys


def _path_matches(path: str, allowed_patterns: List[str]) -> bool:
    """
    Check if a path matches any allowed pattern.

    Uses dot-separated path convention (see module docstring for full spec).

    Matching rules:
    - Exact match: ``"governance.budget_risk"`` matches only that path
    - Wildcard suffix: ``"governance.budget_risk.*"`` matches path and children
    - Parent inference: If ``"governance.budget_risk.*"`` is allowed,
      ``"governance"`` also matches (parent of allowed child)

    Args:
        path: Dot-separated key path to check (e.g., "governance.divergence")
        allowed_patterns: List of allowed path patterns using dot notation.
            Use ``"key.*"`` suffix for wildcard matching.

    Returns:
        True if path matches any pattern, False otherwise

    Examples::

        >>> _path_matches("governance.budget_risk", ["governance.budget_risk.*"])
        True
        >>> _path_matches("governance", ["governance.budget_risk.*"])
        True  # Parent inference
        >>> _path_matches("governance.divergence", ["governance.budget_risk.*"])
        False
    """
    for pattern in allowed_patterns:
        if pattern.endswith(".*"):
            # Prefix match
            prefix = pattern[:-2]
            if path == prefix or path.startswith(prefix + "."):
                return True
            # Parent match: if allowed_path is "governance.budget_risk.*",
            # then "governance" is also allowed (parent of allowed child)
            if prefix.startswith(path + "."):
                return True
        else:
            # Exact match
            if path == pattern:
                return True
            # Parent match for exact patterns too
            if pattern.startswith(path + "."):
                return True
    return False


# =============================================================================
# Core Non-Interference Assertions
# =============================================================================

def assert_only_keys_changed(
    before: Dict[str, Any],
    after: Dict[str, Any],
    allowed_paths: List[str],
) -> NonInterferenceResult:
    """
    Assert that only allowed key paths differ between two dicts.

    SHADOW MODE: Observational verification only.

    This verifies that adding a signal (e.g., budget_risk) does not
    accidentally modify unrelated keys (e.g., divergence metrics).

    Uses dot-separated path convention (see module docstring for full spec).

    Args:
        before: Dict before signal addition
        after: Dict after signal addition
        allowed_paths: List of allowed path patterns using dot notation.
            - Exact: ``"governance.budget_risk"``
            - Wildcard: ``"governance.budget_risk.*"`` (matches children)

    Returns:
        NonInterferenceResult with pass/fail and list of unexpected changes.
        Violations use format: ``"Unexpected key added: path.to.key"``

    Example::

        >>> before = {"governance": {"divergence": {"rate": 0.08}}}
        >>> after = {"governance": {"divergence": {"rate": 0.08}, "budget_risk": {...}}}
        >>> result = assert_only_keys_changed(before, after, ["governance.budget_risk.*"])
        >>> assert result.passed
    """
    keys_before = _get_nested_keys(before)
    keys_after = _get_nested_keys(after)

    added_keys = keys_after - keys_before
    removed_keys = keys_before - keys_after

    violations: List[str] = []

    # Check added keys (sorted for deterministic output)
    for key in sorted(added_keys):
        if not _path_matches(key, allowed_paths):
            violations.append(f"Unexpected key added: {key}")

    # Check removed keys (sorted for deterministic output)
    for key in sorted(removed_keys):
        if not _path_matches(key, allowed_paths):
            violations.append(f"Unexpected key removed: {key}")

    # Check modified values for shared keys (not in allowed paths)
    shared_keys = keys_before & keys_after
    for key in sorted(shared_keys):
        if _path_matches(key, allowed_paths):
            continue  # Allowed to change

        # Get values at this path
        val_before = _get_value_at_path(before, key)
        val_after = _get_value_at_path(after, key)

        if val_before != val_after:
            # Truncate long values for readability
            before_str = _truncate_repr(val_before, max_len=60)
            after_str = _truncate_repr(val_after, max_len=60)
            violations.append(f"Unexpected value change at {key}: {before_str} -> {after_str}")

    if violations:
        return NonInterferenceResult(
            passed=False,
            message=f"{len(violations)} unexpected change(s) detected",
            violations=violations,
        )

    return NonInterferenceResult(
        passed=True,
        message="Only allowed keys changed",
    )


def _get_value_at_path(obj: Dict[str, Any], path: str) -> Any:
    """Get value at a dot-separated path."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _truncate_repr(value: Any, max_len: int = 60) -> str:
    """
    Get truncated repr of a value for readable error messages.

    Args:
        value: Value to represent
        max_len: Maximum length before truncation

    Returns:
        String representation, truncated with "..." if too long
    """
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def assert_warning_delta_at_most(
    before: List[str],
    after: List[str],
    max_delta: int = 1,
) -> NonInterferenceResult:
    """
    Assert that warning list changes by at most N lines.

    SHADOW MODE: Observational verification only.

    This verifies that a signal adds a bounded number of warnings
    and does not remove or reorder existing warnings.

    Args:
        before: Warning list before signal addition
        after: Warning list after signal addition
        max_delta: Maximum allowed increase in warning count (default: 1)

    Returns:
        NonInterferenceResult with pass/fail and violation details

    Example:
        >>> before = ["Schema OK", "P5 baseline set"]
        >>> after = ["Schema OK", "P5 baseline set", "Budget: DEFER"]
        >>> result = assert_warning_delta_at_most(before, after, max_delta=1)
        >>> assert result.passed
    """
    violations: List[str] = []

    delta = len(after) - len(before)

    # Check delta bound
    if delta > max_delta:
        violations.append(f"Warning count increased by {delta}, max allowed is {max_delta}")

    if delta < 0:
        violations.append(f"Warning count decreased by {abs(delta)} (warnings removed)")

    # Check ordering preserved (before warnings should appear in same order at start of after)
    if len(before) > 0:
        for i, warning in enumerate(before):
            if i >= len(after):
                violations.append(f"Base warning [{i}] missing from result")
            elif after[i] != warning:
                violations.append(
                    f"Base warning [{i}] order changed: expected {warning!r}, got {after[i]!r}"
                )

    if violations:
        return NonInterferenceResult(
            passed=False,
            message=f"Warning delta check failed",
            violations=violations,
        )

    return NonInterferenceResult(
        passed=True,
        message=f"Warning delta is {delta} (max {max_delta}), ordering preserved",
    )


def assert_warning_delta_at_most_one(
    before: List[str],
    after: List[str],
) -> NonInterferenceResult:
    """
    Convenience wrapper: assert warning list changes by at most 1 line.

    SHADOW MODE: Observational verification only.

    Args:
        before: Warning list before signal addition
        after: Warning list after signal addition

    Returns:
        NonInterferenceResult with pass/fail
    """
    return assert_warning_delta_at_most(before, after, max_delta=1)


def assert_adapter_is_pure(
    adapter_fn: Callable[[Dict[str, Any]], Any],
    input_ref: Dict[str, Any],
) -> NonInterferenceResult:
    """
    Assert that an adapter function does not modify its input.

    SHADOW MODE: Observational verification only.

    This verifies that adapter functions are pure (no side effects).

    Args:
        adapter_fn: Function to test
        input_ref: Input dict to pass to adapter

    Returns:
        NonInterferenceResult with pass/fail

    Example:
        >>> from backend.topology.first_light.budget_binding import budget_calibration_for_alignment_view
        >>> ref = {"enablement_recommendation": "ENABLE", "overall_pass": True}
        >>> result = assert_adapter_is_pure(budget_calibration_for_alignment_view, ref)
        >>> assert result.passed
    """
    # Deep copy before call
    input_before = copy.deepcopy(input_ref)

    # Call adapter
    try:
        _ = adapter_fn(input_ref)
    except Exception as e:
        return NonInterferenceResult(
            passed=False,
            message=f"Adapter raised exception: {e}",
            violations=[str(e)],
        )

    # Check input unchanged
    if input_ref != input_before:
        return NonInterferenceResult(
            passed=False,
            message="Adapter modified input reference",
            violations=["Input was mutated by adapter call"],
        )

    return NonInterferenceResult(
        passed=True,
        message="Adapter is pure (input unchanged)",
    )


def assert_output_excludes_keys(
    output: Dict[str, Any],
    excluded_keys: Set[str],
) -> NonInterferenceResult:
    """
    Assert that adapter output does not contain excluded keys.

    SHADOW MODE: Observational verification only.

    This verifies signal isolation (e.g., budget adapter output
    should not contain divergence-related keys).

    Args:
        output: Adapter output dict
        excluded_keys: Set of key names that must not appear

    Returns:
        NonInterferenceResult with pass/fail and overlap list

    Example:
        >>> output = {"alignment": "healthy", "status": "ok", "mode": "SHADOW"}
        >>> divergence_keys = {"divergence", "p5_divergence", "divergence_rate"}
        >>> result = assert_output_excludes_keys(output, divergence_keys)
        >>> assert result.passed
    """
    output_keys = _get_nested_keys(output)
    output_key_names = {k.split(".")[-1] for k in output_keys}

    overlap = output_key_names & excluded_keys

    if overlap:
        return NonInterferenceResult(
            passed=False,
            message=f"Output contains {len(overlap)} excluded key(s)",
            violations=[f"Excluded key present: {k}" for k in sorted(overlap)],
        )

    return NonInterferenceResult(
        passed=True,
        message="Output excludes all forbidden keys",
    )


# =============================================================================
# Pytest Assertion Helpers
# =============================================================================

def pytest_assert_only_keys_changed(
    before: Dict[str, Any],
    after: Dict[str, Any],
    allowed_paths: List[str],
    context: str = "",
) -> None:
    """
    Pytest assertion that only allowed keys changed.

    Raises AssertionError with detailed info on failure.

    Args:
        before: Dict before signal addition
        after: Dict after signal addition
        allowed_paths: List of allowed path patterns
        context: Optional context string for error message
    """
    result = assert_only_keys_changed(before, after, allowed_paths)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        violations_str = "\n".join(f"  - {v}" for v in result.violations)
        raise AssertionError(
            f"Non-interference check failed{ctx}:\n"
            f"  {result.message}\n"
            f"Violations:\n{violations_str}"
        )


def pytest_assert_warning_delta_at_most_one(
    before: List[str],
    after: List[str],
    context: str = "",
) -> None:
    """
    Pytest assertion that warning delta is at most 1.

    Raises AssertionError with detailed info on failure.

    Args:
        before: Warning list before
        after: Warning list after
        context: Optional context string for error message
    """
    result = assert_warning_delta_at_most_one(before, after)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        violations_str = "\n".join(f"  - {v}" for v in result.violations)
        raise AssertionError(
            f"Warning delta check failed{ctx}:\n"
            f"  {result.message}\n"
            f"Violations:\n{violations_str}"
        )


def pytest_assert_adapter_is_pure(
    adapter_fn: Callable[[Dict[str, Any]], Any],
    input_ref: Dict[str, Any],
    context: str = "",
) -> None:
    """
    Pytest assertion that adapter is pure.

    Raises AssertionError with detailed info on failure.

    Args:
        adapter_fn: Function to test
        input_ref: Input dict to pass to adapter
        context: Optional context string for error message
    """
    result = assert_adapter_is_pure(adapter_fn, input_ref)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        raise AssertionError(
            f"Adapter purity check failed{ctx}:\n"
            f"  {result.message}"
        )


def pytest_assert_output_excludes_keys(
    output: Dict[str, Any],
    excluded_keys: Set[str],
    context: str = "",
) -> None:
    """
    Pytest assertion that output excludes certain keys.

    Raises AssertionError with detailed info on failure.

    Args:
        output: Adapter output dict
        excluded_keys: Set of key names that must not appear
        context: Optional context string for error message
    """
    result = assert_output_excludes_keys(output, excluded_keys)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        violations_str = "\n".join(f"  - {v}" for v in result.violations)
        raise AssertionError(
            f"Key exclusion check failed{ctx}:\n"
            f"  {result.message}\n"
            f"Violations:\n{violations_str}"
        )
