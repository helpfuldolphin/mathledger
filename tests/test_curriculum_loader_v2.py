"""
PHASE II — NOT USED IN PHASE I

Unit tests for CurriculumLoaderV2 and related classes.
"""

import pytest
from typing import Dict, Any

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    CurriculumLoadError,
    UnknownMetricKindError,
    SuccessMetricSpec,
    UpliftSlice,
    DegenerateCheckWarning,
)
from experiments.slice_success_metrics import METRIC_KINDS, is_valid_metric_kind


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def make_minimal_config() -> Dict[str, Any]:
    """Create a minimal valid curriculum config."""
    return {
        "version": "2.0",
        "slices": {
            "test_slice": {
                "description": "A test slice",
                "items": ["1+1", "2+2", "3+3"],
                "atoms": 3,
                "depth_min": 1,
                "depth_max": 3,
                "total_max": 100,
                "formula_pool": 50,
                "prereg_hash": "abc123",
                "verifier": "truth_table",
                "success_metric": {
                    "kind": "sparse_success",
                    "thresholds": {"min_verified": 10},
                },
            },
        },
    }


def make_multi_slice_config() -> Dict[str, Any]:
    """Create a config with multiple slices using different metrics."""
    return {
        "version": "2.0",
        "slices": {
            "slice_a": {
                "description": "Slice A with sparse_success",
                "items": ["a", "b", "c"],
                "atoms": 3,
                "depth_min": 1,
                "depth_max": 2,
                "total_max": 50,
                "formula_pool": 25,
                "success_metric": {
                    "kind": "sparse_success",
                    "thresholds": {"min_verified": 5},
                },
            },
            "slice_b": {
                "description": "Slice B with goal_hit",
                "items": ["x", "y", "z"],
                "atoms": 3,
                "depth_min": 2,
                "depth_max": 4,
                "total_max": 100,
                "formula_pool": 50,
                "success_metric": {
                    "kind": "goal_hit",
                    "thresholds": {"min_total_verified": 3},
                    "target_hashes": ["h1", "h2", "h3", "h4", "h5"],
                },
            },
            "slice_c_no_metric": {
                "description": "Slice C without a metric",
                "items": ["p", "q"],
                "atoms": 2,
                "depth_min": 0,
                "depth_max": 1,
                "total_max": 20,
            },
        },
    }


# ---------------------------------------------------------------------------
# SuccessMetricSpec Tests
# ---------------------------------------------------------------------------


class TestSuccessMetricSpec:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for SuccessMetricSpec dataclass.
    """

    def test_create_valid_metric_spec(self):
        """Can create a valid SuccessMetricSpec."""
        spec = SuccessMetricSpec(
            kind="sparse_success",
            thresholds={"min_verified": 10},
        )
        assert spec.kind == "sparse_success"
        assert spec.thresholds["min_verified"] == 10
        assert spec.target_hashes is None

    def test_create_metric_with_target_hashes(self):
        """Can create a metric spec with target hashes."""
        spec = SuccessMetricSpec(
            kind="goal_hit",
            thresholds={"min_total_verified": 3},
            target_hashes={"h1", "h2", "h3"},
        )
        assert spec.kind == "goal_hit"
        assert len(spec.target_hashes) == 3
        assert "h1" in spec.target_hashes

    def test_invalid_metric_kind_raises(self):
        """Creating a spec with unknown metric kind raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SuccessMetricSpec(kind="invalid_kind")
        assert "Unknown metric kind" in str(exc_info.value)
        assert "invalid_kind" in str(exc_info.value)

    def test_from_dict(self):
        """Can create spec from dictionary."""
        data = {
            "kind": "multi_goal",
            "thresholds": {"count": 5},
            "target_hashes": ["a", "b", "c"],
        }
        spec = SuccessMetricSpec.from_dict(data)
        assert spec.kind == "multi_goal"
        assert spec.thresholds["count"] == 5
        assert spec.target_hashes == {"a", "b", "c"}

    def test_from_dict_missing_kind_raises(self):
        """from_dict raises when kind is missing."""
        with pytest.raises(ValueError) as exc_info:
            SuccessMetricSpec.from_dict({"thresholds": {}})
        assert "requires 'kind'" in str(exc_info.value)

    def test_to_dict(self):
        """Can convert spec to dictionary."""
        spec = SuccessMetricSpec(
            kind="goal_hit",
            thresholds={"min_total_verified": 2},
            target_hashes={"z", "a", "m"},
        )
        d = spec.to_dict()
        assert d["kind"] == "goal_hit"
        assert d["thresholds"]["min_total_verified"] == 2
        # target_hashes should be sorted
        assert d["target_hashes"] == ["a", "m", "z"]

    def test_get_required_params(self):
        """get_required_params returns correct params for kind."""
        spec = SuccessMetricSpec(kind="chain_success")
        params = spec.get_required_params()
        assert "dependency_graph" in params
        assert "chain_target_hash" in params
        assert "min_chain_length" in params

    def test_get_description(self):
        """get_description returns kind description."""
        spec = SuccessMetricSpec(kind="sparse_success")
        desc = spec.get_description()
        assert "minimum count" in desc.lower()

    def test_frozen_immutability(self):
        """SuccessMetricSpec is immutable (frozen)."""
        spec = SuccessMetricSpec(kind="sparse_success")
        with pytest.raises(AttributeError):
            spec.kind = "other_kind"


# ---------------------------------------------------------------------------
# UpliftSlice Tests
# ---------------------------------------------------------------------------


class TestUpliftSlice:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for UpliftSlice dataclass.
    """

    def test_create_uplift_slice(self):
        """Can create a basic UpliftSlice."""
        slice_obj = UpliftSlice(
            name="test",
            description="Test slice",
            items=["a", "b"],
            atoms=2,
            depth_min=1,
            depth_max=3,
            total_max=100,
        )
        assert slice_obj.name == "test"
        assert slice_obj.atoms == 2
        assert slice_obj.depth_max == 3

    def test_from_dict_minimal(self):
        """Can create from minimal dict."""
        data = {
            "description": "Minimal slice",
            "items": ["x", "y", "z"],
        }
        slice_obj = UpliftSlice.from_dict("minimal", data)
        assert slice_obj.name == "minimal"
        assert slice_obj.description == "Minimal slice"
        assert len(slice_obj.items) == 3
        # Defaults
        assert slice_obj.atoms == 3  # defaults to len(items)
        assert slice_obj.verifier == "truth_table"

    def test_from_dict_full(self):
        """Can create from full dict."""
        data = {
            "description": "Full slice",
            "items": ["1", "2", "3"],
            "atoms": 5,
            "depth_min": 2,
            "depth_max": 10,
            "total_max": 500,
            "formula_pool": 200,
            "prereg_hash": "xyz789",
            "verifier": "lean",
            "success_metric": {
                "kind": "sparse_success",
                "thresholds": {"min_verified": 50},
            },
            "extra_field": "extra_value",
        }
        slice_obj = UpliftSlice.from_dict("full", data)
        assert slice_obj.name == "full"
        assert slice_obj.atoms == 5
        assert slice_obj.depth_max == 10
        assert slice_obj.verifier == "lean"
        assert slice_obj.prereg_hash == "xyz789"
        assert slice_obj.success_metric.kind == "sparse_success"
        assert slice_obj.metadata.get("extra_field") == "extra_value"

    def test_to_dict(self):
        """Can convert slice to dict."""
        slice_obj = UpliftSlice(
            name="test",
            description="Test",
            atoms=3,
            depth_min=1,
            depth_max=5,
            total_max=100,
            formula_pool=50,
            success_metric=SuccessMetricSpec(
                kind="sparse_success",
                thresholds={"min_verified": 10},
            ),
        )
        d = slice_obj.to_dict()
        assert d["name"] == "test"
        assert d["atoms"] == 3
        assert d["success_metric"]["kind"] == "sparse_success"


# ---------------------------------------------------------------------------
# CurriculumLoaderV2 Tests
# ---------------------------------------------------------------------------


class TestCurriculumLoaderV2:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for CurriculumLoaderV2 class.
    """

    def test_load_minimal_config(self):
        """Can load a minimal config."""
        config = make_minimal_config()
        loader = CurriculumLoaderV2(config)
        assert loader.version == "2.0"
        assert len(loader.slices) == 1
        assert "test_slice" in loader.slices

    def test_get_slice(self):
        """Can get a slice by name."""
        config = make_minimal_config()
        loader = CurriculumLoaderV2(config)
        slice_obj = loader.get_slice("test_slice")
        assert slice_obj.name == "test_slice"
        assert slice_obj.atoms == 3

    def test_get_slice_not_found_raises(self):
        """get_slice raises KeyError for unknown slice."""
        config = make_minimal_config()
        loader = CurriculumLoaderV2(config)
        with pytest.raises(KeyError) as exc_info:
            loader.get_slice("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert "test_slice" in str(exc_info.value)  # shows available

    def test_get_slice_names(self):
        """get_slice_names returns all slice names."""
        config = make_multi_slice_config()
        loader = CurriculumLoaderV2(config)
        names = loader.get_slice_names()
        assert set(names) == {"slice_a", "slice_b", "slice_c_no_metric"}

    def test_get_slices_by_metric_kind(self):
        """get_slices_by_metric_kind filters correctly."""
        config = make_multi_slice_config()
        loader = CurriculumLoaderV2(config)

        sparse = loader.get_slices_by_metric_kind("sparse_success")
        assert len(sparse) == 1
        assert sparse[0].name == "slice_a"

        goal_hit = loader.get_slices_by_metric_kind("goal_hit")
        assert len(goal_hit) == 1
        assert goal_hit[0].name == "slice_b"

        chain = loader.get_slices_by_metric_kind("chain_success")
        assert len(chain) == 0

    def test_get_metric_kinds_in_use(self):
        """get_metric_kinds_in_use returns correct mapping."""
        config = make_multi_slice_config()
        loader = CurriculumLoaderV2(config)
        usage = loader.get_metric_kinds_in_use()

        assert "sparse_success" in usage
        assert "slice_a" in usage["sparse_success"]
        assert "goal_hit" in usage
        assert "slice_b" in usage["goal_hit"]
        # slice_c has no metric, so shouldn't appear
        assert "slice_c_no_metric" not in str(usage)

    def test_unknown_metric_kind_raises_at_load(self):
        """Loading config with unknown metric kind raises error."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_slice": {
                    "description": "Bad",
                    "success_metric": {
                        "kind": "totally_invalid_metric_kind",
                    },
                },
            },
        }
        with pytest.raises(UnknownMetricKindError) as exc_info:
            CurriculumLoaderV2(config)
        assert "bad_slice" in str(exc_info.value)
        assert "totally_invalid_metric_kind" in str(exc_info.value)
        # Should list available kinds
        assert "sparse_success" in str(exc_info.value)

    def test_validation_can_be_disabled(self):
        """Validation can be disabled, but SuccessMetricSpec still validates kind."""
        config = {
            "version": "2.0",
            "slices": {
                "experimental": {
                    "description": "Experimental",
                    "success_metric": {
                        "kind": "future_metric",  # Not registered yet
                    },
                },
            },
        }
        # Even with validate_metrics=False, SuccessMetricSpec validates kind in __post_init__
        # This is caught and converted to UnknownMetricKindError
        with pytest.raises(UnknownMetricKindError):
            CurriculumLoaderV2(config, validate_metrics=False)

    def test_to_summary_dict(self):
        """to_summary_dict returns expected structure."""
        config = make_multi_slice_config()
        loader = CurriculumLoaderV2(config)
        summary = loader.to_summary_dict()

        assert summary["version"] == "2.0"
        assert summary["slice_count"] == 3
        assert "slices" in summary
        assert "metric_kinds_in_use" in summary


# ---------------------------------------------------------------------------
# Non-Degenerate Defaults Check Tests
# ---------------------------------------------------------------------------


class TestNonDegenerateChecks:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for verify_non_degenerate_defaults.
    """

    def test_valid_config_passes(self):
        """A valid config produces no warnings."""
        config = make_minimal_config()
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert len(warnings) == 0

    def test_depth_min_greater_than_max_error(self):
        """depth_min > depth_max produces an error."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_depth": {
                    "description": "Bad depth",
                    "depth_min": 10,
                    "depth_max": 5,
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert len(warnings) == 1
        assert warnings[0].severity == "error"
        assert "depth_ordering" in warnings[0].check

    def test_zero_atoms_with_items_warning(self):
        """atoms=0 with items produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "zero_atoms": {
                    "description": "Zero atoms",
                    "items": ["a", "b", "c"],
                    "atoms": 0,
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "atoms_nonzero" for w in warnings)

    def test_zero_total_max_with_items_warning(self):
        """total_max=0 with items produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "zero_total": {
                    "description": "Zero total",
                    "items": ["a", "b"],
                    "atoms": 2,
                    "total_max": 0,
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "total_max_nonzero" for w in warnings)

    def test_sparse_threshold_exceeds_total_max_warning(self):
        """min_verified > total_max produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_sparse": {
                    "description": "Bad sparse",
                    "atoms": 3,
                    "total_max": 10,
                    "success_metric": {
                        "kind": "sparse_success",
                        "thresholds": {"min_verified": 100},  # > total_max
                    },
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "sparse_threshold" for w in warnings)

    def test_goal_threshold_exceeds_pool_warning(self):
        """min_total_verified > formula_pool produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_goal": {
                    "description": "Bad goal",
                    "items": ["a", "b", "c"],
                    "atoms": 3,
                    "formula_pool": 5,
                    "success_metric": {
                        "kind": "goal_hit",
                        "thresholds": {"min_total_verified": 10},  # > pool
                    },
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "goal_threshold" for w in warnings)

    def test_goal_threshold_exceeds_target_count_warning(self):
        """min_total_verified > len(target_hashes) produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_target": {
                    "description": "Bad target",
                    "atoms": 3,
                    "formula_pool": 100,
                    "success_metric": {
                        "kind": "goal_hit",
                        "thresholds": {"min_total_verified": 10},
                        "target_hashes": ["h1", "h2", "h3"],  # only 3
                    },
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "goal_target_count" for w in warnings)

    def test_chain_threshold_exceeds_depth_warning(self):
        """min_chain_length > depth_max produces a warning."""
        config = {
            "version": "2.0",
            "slices": {
                "bad_chain": {
                    "description": "Bad chain",
                    "atoms": 3,
                    "depth_max": 5,
                    "success_metric": {
                        "kind": "chain_success",
                        "thresholds": {"min_chain_length": 10},  # > depth_max
                    },
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        warnings = loader.verify_non_degenerate_defaults()
        assert any(w.check == "chain_threshold" for w in warnings)


# ---------------------------------------------------------------------------
# METRIC_KINDS Registry Tests
# ---------------------------------------------------------------------------


class TestMetricKindsRegistry:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for the METRIC_KINDS registry in slice_success_metrics.
    """

    def test_all_expected_kinds_registered(self):
        """Expected metric kinds are registered."""
        expected = {"goal_hit", "sparse_success", "chain_success", "multi_goal"}
        assert expected.issubset(set(METRIC_KINDS.keys()))

    def test_is_valid_metric_kind(self):
        """is_valid_metric_kind returns correct values."""
        assert is_valid_metric_kind("sparse_success") is True
        assert is_valid_metric_kind("goal_hit") is True
        assert is_valid_metric_kind("invalid") is False

    def test_registry_structure(self):
        """Registry entries have correct structure."""
        for kind, (required, optional, desc) in METRIC_KINDS.items():
            assert isinstance(required, tuple)
            assert isinstance(optional, tuple)
            assert isinstance(desc, str)
            assert len(desc) > 0


# ---------------------------------------------------------------------------
# DegenerateCheckWarning Tests
# ---------------------------------------------------------------------------


class TestDegenerateCheckWarning:
    """
    PHASE II — NOT USED IN PHASE I
    Tests for DegenerateCheckWarning.
    """

    def test_str_format(self):
        """__str__ produces expected format."""
        w = DegenerateCheckWarning(
            slice_name="test_slice",
            check="test_check",
            message="Something is wrong",
            severity="warning",
        )
        s = str(w)
        assert "[WARNING]" in s
        assert "test_slice" in s
        assert "test_check" in s
        assert "Something is wrong" in s

    def test_error_severity_format(self):
        """Error severity is formatted correctly."""
        w = DegenerateCheckWarning(
            slice_name="bad",
            check="depth",
            message="Bad depth",
            severity="error",
        )
        assert "[ERROR]" in str(w)
