# PHASE II — NOT USED IN PHASE I
"""
Unit tests for curriculum_loader_v2.py
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.curriculum_loader_v2 import (
    CurriculumConfigError,
    CurriculumLoaderV2,
    SuccessMetricSpec,
    UpliftSlice,
    VALID_METRIC_KINDS,
    _canonical_metric_kind,
    _compute_deterministic_hash,
    get_metric_function,
)


class TestSuccessMetricSpec(unittest.TestCase):
    """Tests for SuccessMetricSpec dataclass."""

    def test_valid_kinds(self):
        """All valid metric kinds should be accepted."""
        for kind in ["goal_hit", "sparse", "chain_length", "multi_goal", "density"]:
            spec = SuccessMetricSpec(kind=kind)
            # density is an alias for sparse
            expected = "sparse" if kind == "density" else kind
            self.assertEqual(spec.kind, expected)

    def test_invalid_kind_raises(self):
        """Invalid metric kind should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SuccessMetricSpec(kind="invalid_metric")
        self.assertIn("Invalid success_metric.kind", str(ctx.exception))

    def test_from_dict_basic(self):
        """from_dict should parse basic metric spec."""
        data = {
            "kind": "goal_hit",
            "parameters": {"min_total_verified": 3},
        }
        spec = SuccessMetricSpec.from_dict(data, "test_slice")
        self.assertEqual(spec.kind, "goal_hit")
        self.assertEqual(spec.thresholds.get("min_total_verified"), 3)

    def test_from_dict_with_target_hashes(self):
        """from_dict should parse target_hashes."""
        data = {
            "kind": "goal_hit",
            "parameters": {
                "target_hashes": ["h1", "h2", "h3"],
            },
        }
        spec = SuccessMetricSpec.from_dict(data, "test_slice")
        self.assertEqual(spec.target_hashes, frozenset(["h1", "h2", "h3"]))

    def test_from_dict_missing_kind_raises(self):
        """from_dict should raise if kind is missing."""
        with self.assertRaises(ValueError) as ctx:
            SuccessMetricSpec.from_dict({}, "test_slice")
        self.assertIn("missing success_metric.kind", str(ctx.exception))

    def test_to_dict_roundtrip(self):
        """to_dict should produce serializable dict."""
        spec = SuccessMetricSpec(
            kind="chain_length",
            thresholds={"min_chain_length": 5},
            target_hashes=frozenset(["h1"]),
            parameters={"chain_target_hash": "h10"},
        )
        d = spec.to_dict()
        self.assertEqual(d["kind"], "chain_length")
        self.assertEqual(d["thresholds"]["min_chain_length"], 5)
        self.assertIn("h1", d["target_hashes"])
        # Should be JSON serializable
        json.dumps(d)

    def test_density_alias(self):
        """density should be canonicalized to sparse."""
        spec = SuccessMetricSpec(kind="density")
        self.assertEqual(spec.kind, "sparse")


class TestUpliftSlice(unittest.TestCase):
    """Tests for UpliftSlice dataclass."""

    def test_from_dict_basic(self):
        """from_dict should parse basic slice config."""
        data = {
            "description": "Test slice",
            "items": ["a", "b", "c"],
            "prereg_hash": "abc123",
            "success_metric": {
                "kind": "sparse",
                "parameters": {"min_verified": 2},
            },
        }
        slice_obj = UpliftSlice.from_dict("test_slice", data)
        self.assertEqual(slice_obj.name, "test_slice")
        self.assertEqual(slice_obj.description, "Test slice")
        self.assertEqual(slice_obj.items, ("a", "b", "c"))
        self.assertEqual(slice_obj.prereg_hash, "abc123")
        self.assertEqual(slice_obj.success_metric.kind, "sparse")
        # config_hash should be a hex string
        self.assertEqual(len(slice_obj.config_hash), 64)

    def test_from_dict_without_success_metric(self):
        """from_dict should infer default success_metric."""
        data = {
            "description": "No metric slice",
            "items": ["x", "y"],
        }
        slice_obj = UpliftSlice.from_dict("infer_slice", data)
        # Should default to sparse
        self.assertEqual(slice_obj.success_metric.kind, "sparse")

    def test_deterministic_hash(self):
        """Same config should produce same hash."""
        data = {
            "description": "Hash test",
            "items": ["1", "2", "3"],
            "prereg_hash": "test",
        }
        slice1 = UpliftSlice.from_dict("hash_test", data)
        slice2 = UpliftSlice.from_dict("hash_test", data)
        self.assertEqual(slice1.config_hash, slice2.config_hash)

    def test_hash_changes_with_content(self):
        """Different content should produce different hash."""
        data1 = {"description": "A", "items": ["1"]}
        data2 = {"description": "B", "items": ["1"]}
        slice1 = UpliftSlice.from_dict("s1", data1)
        slice2 = UpliftSlice.from_dict("s1", data2)
        self.assertNotEqual(slice1.config_hash, slice2.config_hash)

    def test_to_dict(self):
        """to_dict should produce serializable dict."""
        data = {
            "description": "Serialize test",
            "items": ["a"],
            "success_metric": {"kind": "multi_goal"},
        }
        slice_obj = UpliftSlice.from_dict("ser_test", data)
        d = slice_obj.to_dict()
        self.assertEqual(d["name"], "ser_test")
        self.assertIn("success_metric", d)
        # Should be JSON serializable
        json.dumps(d)


class TestCurriculumLoaderV2(unittest.TestCase):
    """Tests for CurriculumLoaderV2 class."""

    def _create_temp_yaml(self, config: dict) -> Path:
        """Helper to create a temp YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            return Path(f.name)

    def test_load_basic_config(self):
        """Should load basic valid config."""
        config = {
            "version": 2.0,
            "slices": {
                "slice_a": {
                    "description": "Slice A",
                    "items": ["item1", "item2"],
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        self.assertEqual(len(loader.list_slices()), 1)
        self.assertEqual(loader.list_slices()[0].name, "slice_a")

    def test_load_from_yaml_path(self):
        """Should load from YAML file path."""
        config = {
            "version": 2.0,
            "slices": {
                "yaml_slice": {
                    "description": "YAML test",
                    "items": ["x"],
                },
            },
        }
        path = self._create_temp_yaml(config)
        try:
            loader = CurriculumLoaderV2.from_yaml_path(path)
            self.assertEqual(loader.slice_names, ["yaml_slice"])
        finally:
            path.unlink()

    def test_from_default_phase2_config(self):
        """Should load default Phase II config."""
        loader = CurriculumLoaderV2.from_default_phase2_config()
        # Should have at least one slice
        self.assertGreater(len(loader.list_slices()), 0)
        # All slices should have valid metric kinds
        for slice_obj in loader.list_slices():
            self.assertIn(slice_obj.success_metric.kind, VALID_METRIC_KINDS)

    def test_list_slices_sorted(self):
        """list_slices should return slices sorted by name."""
        config = {
            "version": 2.0,
            "slices": {
                "zebra": {"items": ["z"]},
                "alpha": {"items": ["a"]},
                "middle": {"items": ["m"]},
            },
        }
        loader = CurriculumLoaderV2(config)
        names = [s.name for s in loader.list_slices()]
        self.assertEqual(names, ["alpha", "middle", "zebra"])

    def test_get_slice(self):
        """get_slice should return specific slice."""
        config = {
            "version": 2.0,
            "slices": {
                "target": {"items": ["t"]},
            },
        }
        loader = CurriculumLoaderV2(config)
        slice_obj = loader.get_slice("target")
        self.assertEqual(slice_obj.name, "target")

    def test_get_slice_not_found(self):
        """get_slice should raise KeyError for missing slice."""
        config = {"version": 2.0, "slices": {"exists": {"items": ["e"]}}}
        loader = CurriculumLoaderV2(config)
        with self.assertRaises(KeyError):
            loader.get_slice("missing")

    def test_get_success_metric_spec(self):
        """get_success_metric_spec should return metric spec."""
        config = {
            "version": 2.0,
            "slices": {
                "metric_slice": {
                    "items": ["m"],
                    "success_metric": {"kind": "chain_length"},
                },
            },
        }
        loader = CurriculumLoaderV2(config)
        spec = loader.get_success_metric_spec("metric_slice")
        self.assertEqual(spec.kind, "chain_length")

    def test_missing_version_raises(self):
        """Should raise error for missing version."""
        config = {"slices": {"s": {"items": []}}}
        with self.assertRaises(CurriculumConfigError) as ctx:
            CurriculumLoaderV2(config)
        self.assertIn("version", str(ctx.exception).lower())

    def test_unsupported_version_raises(self):
        """Should raise error for unsupported version."""
        config = {"version": 99.0, "slices": {"s": {"items": []}}}
        with self.assertRaises(CurriculumConfigError):
            CurriculumLoaderV2(config)

    def test_empty_slices_raises(self):
        """Should raise error for empty slices."""
        config = {"version": 2.0, "slices": {}}
        with self.assertRaises(CurriculumConfigError) as ctx:
            CurriculumLoaderV2(config)
        self.assertIn("No slices", str(ctx.exception))

    def test_invalid_metric_kind_raises(self):
        """Should raise error at load time for invalid metric kind."""
        config = {
            "version": 2.0,
            "slices": {
                "bad_metric": {
                    "items": ["b"],
                    "success_metric": {"kind": "invalid_kind"},
                },
            },
        }
        with self.assertRaises((CurriculumConfigError, ValueError)):
            CurriculumLoaderV2(config)


class TestCrossValidation(unittest.TestCase):
    """Tests for cross-validation with slice_success_metrics."""

    def test_all_valid_kinds_have_functions(self):
        """All valid metric kinds should map to actual functions."""
        for kind in ["goal_hit", "sparse", "chain_length", "multi_goal"]:
            func = get_metric_function(kind)
            self.assertTrue(callable(func))

    def test_invalid_kind_raises(self):
        """Invalid kind should raise ValueError."""
        with self.assertRaises(ValueError):
            get_metric_function("nonexistent")


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions."""

    def test_canonical_metric_kind(self):
        """_canonical_metric_kind should normalize aliases."""
        self.assertEqual(_canonical_metric_kind("density"), "sparse")
        self.assertEqual(_canonical_metric_kind("goal_hit"), "goal_hit")
        self.assertEqual(_canonical_metric_kind("sparse"), "sparse")

    def test_deterministic_hash(self):
        """_compute_deterministic_hash should be deterministic."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        h1 = _compute_deterministic_hash(data)
        h2 = _compute_deterministic_hash(data)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # SHA-256 hex

    def test_deterministic_hash_order_independent(self):
        """Hash should be same regardless of dict key order."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        h1 = _compute_deterministic_hash(data1)
        h2 = _compute_deterministic_hash(data2)
        self.assertEqual(h1, h2)


class TestDeterminism(unittest.TestCase):
    """Tests for hash determinism requirements."""

    def test_slice_hash_determinism(self):
        """UpliftSlice hash must be deterministic across multiple loads."""
        config = {
            "version": 2.0,
            "slices": {
                "det_test": {
                    "description": "Determinism test",
                    "items": ["x", "y", "z"],
                    "prereg_hash": "abc",
                    "success_metric": {"kind": "goal_hit"},
                },
            },
        }
        hashes = []
        for _ in range(10):
            loader = CurriculumLoaderV2(config)
            hashes.append(loader.get_slice("det_test").config_hash)

        # All hashes must be identical
        self.assertTrue(all(h == hashes[0] for h in hashes))


if __name__ == "__main__":
    unittest.main()
