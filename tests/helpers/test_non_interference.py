"""
Unit tests for tests/helpers/non_interference.py

Verifies the helper functions work correctly for:
- Key path matching (exact, prefix, parent)
- Warning delta assertions
- Adapter purity assertions
- Key exclusion assertions
"""

from __future__ import annotations

import pytest

from tests.helpers.non_interference import (
    assert_only_keys_changed,
    assert_warning_delta_at_most,
    assert_warning_delta_at_most_one,
    assert_adapter_is_pure,
    assert_output_excludes_keys,
    _path_matches,
    _get_nested_keys,
)


class TestPathMatching:
    """Tests for path matching logic."""

    def test_exact_match(self) -> None:
        """Test exact path matching."""
        assert _path_matches("governance.budget_risk", ["governance.budget_risk"])
        assert not _path_matches("governance.divergence", ["governance.budget_risk"])

    def test_prefix_match(self) -> None:
        """Test prefix path matching with wildcard."""
        assert _path_matches("governance.budget_risk", ["governance.budget_risk.*"])
        assert _path_matches("governance.budget_risk.calibration", ["governance.budget_risk.*"])
        assert not _path_matches("governance.divergence", ["governance.budget_risk.*"])

    def test_parent_match(self) -> None:
        """Test parent path is allowed when child is in allowed list."""
        # If "governance.budget_risk.*" is allowed, "governance" should match
        # because it's a parent of the allowed path
        assert _path_matches("governance", ["governance.budget_risk.*"])

    def test_no_match(self) -> None:
        """Test paths that should not match."""
        assert not _path_matches("other.path", ["governance.budget_risk.*"])
        assert not _path_matches("governance_extra", ["governance.*"])


class TestGetNestedKeys:
    """Tests for nested key extraction."""

    def test_flat_dict(self) -> None:
        """Test key extraction from flat dict."""
        obj = {"a": 1, "b": 2}
        keys = _get_nested_keys(obj)
        assert keys == {"a", "b"}

    def test_nested_dict(self) -> None:
        """Test key extraction from nested dict."""
        obj = {"a": {"b": {"c": 1}}}
        keys = _get_nested_keys(obj)
        assert keys == {"a", "a.b", "a.b.c"}


class TestAssertOnlyKeysChanged:
    """Tests for assert_only_keys_changed."""

    def test_no_changes(self) -> None:
        """Test when dicts are identical."""
        before = {"a": 1}
        after = {"a": 1}
        result = assert_only_keys_changed(before, after, [])
        assert result.passed

    def test_allowed_key_added(self) -> None:
        """Test when an allowed key is added."""
        before = {"a": 1}
        after = {"a": 1, "b": 2}
        result = assert_only_keys_changed(before, after, ["b"])
        assert result.passed

    def test_unexpected_key_added(self) -> None:
        """Test when an unexpected key is added."""
        before = {"a": 1}
        after = {"a": 1, "b": 2}
        result = assert_only_keys_changed(before, after, [])
        assert not result.passed
        assert "Unexpected key added: b" in result.violations

    def test_nested_allowed_key(self) -> None:
        """Test with nested allowed paths."""
        before = {"governance": {"divergence": {"rate": 0.08}}}
        after = {
            "governance": {
                "divergence": {"rate": 0.08},
                "budget_risk": {"calibration": "DEFER"},
            }
        }
        result = assert_only_keys_changed(before, after, ["governance.budget_risk.*"])
        assert result.passed


class TestWarningDelta:
    """Tests for warning delta assertions."""

    def test_no_change(self) -> None:
        """Test when warnings are identical."""
        before = ["warn1", "warn2"]
        after = ["warn1", "warn2"]
        result = assert_warning_delta_at_most_one(before, after)
        assert result.passed

    def test_one_added(self) -> None:
        """Test when one warning is added."""
        before = ["warn1"]
        after = ["warn1", "warn2"]
        result = assert_warning_delta_at_most_one(before, after)
        assert result.passed

    def test_two_added_fails(self) -> None:
        """Test when two warnings are added (exceeds max)."""
        before = ["warn1"]
        after = ["warn1", "warn2", "warn3"]
        result = assert_warning_delta_at_most_one(before, after)
        assert not result.passed

    def test_order_changed_fails(self) -> None:
        """Test when base warning order is changed."""
        before = ["warn1", "warn2"]
        after = ["warn2", "warn1"]
        result = assert_warning_delta_at_most_one(before, after)
        assert not result.passed

    def test_custom_max_delta(self) -> None:
        """Test with custom max delta."""
        before = ["warn1"]
        after = ["warn1", "warn2", "warn3"]
        result = assert_warning_delta_at_most(before, after, max_delta=2)
        assert result.passed


class TestAdapterPurity:
    """Tests for adapter purity assertions."""

    def test_pure_adapter(self) -> None:
        """Test with a pure adapter function."""
        def pure_adapter(ref):
            return {"result": ref.get("value", 0) * 2}

        result = assert_adapter_is_pure(pure_adapter, {"value": 5})
        assert result.passed

    def test_impure_adapter(self) -> None:
        """Test with an impure adapter that modifies input."""
        def impure_adapter(ref):
            ref["modified"] = True
            return {"result": "done"}

        result = assert_adapter_is_pure(impure_adapter, {"value": 5})
        assert not result.passed


class TestOutputExcludesKeys:
    """Tests for output key exclusion assertions."""

    def test_no_excluded_keys(self) -> None:
        """Test when output has no excluded keys."""
        output = {"alignment": "healthy", "status": "ok"}
        excluded = {"divergence", "error"}
        result = assert_output_excludes_keys(output, excluded)
        assert result.passed

    def test_has_excluded_key(self) -> None:
        """Test when output has an excluded key."""
        output = {"alignment": "healthy", "divergence": 0.05}
        excluded = {"divergence", "error"}
        result = assert_output_excludes_keys(output, excluded)
        assert not result.passed
        assert "divergence" in str(result.violations)
