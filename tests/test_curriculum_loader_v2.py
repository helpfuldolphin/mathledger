# PHASE II — NOT USED IN PHASE I
"""
Unit tests for curriculum_loader_v2.

Tests cover:
- Loading Phase II curriculum
- Validation of slice parameters (atoms, depth bounds, max_candidates)
- Validation of success_metric.kind
- Deterministic hashing
- Fail-fast on malformed YAML
- Rejection of Phase I curriculum files
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from experiments.curriculum_loader_v2 import (
    VALID_SUCCESS_METRIC_KINDS,
    CurriculumLoadError,
    CurriculumLoader,
    UpliftSlice,
    ValidationError,
    _compute_deterministic_hash,
    hash_curriculum_config,
    hash_slice_config,
    load_all_slices,
    load_phase2_curriculum,
    load_slice,
)


class TestUpliftSlice(unittest.TestCase):
    """Test the UpliftSlice dataclass."""

    def test_creation(self):
        """Test basic UpliftSlice creation."""
        slice_obj = UpliftSlice(
            name="test_slice",
            params={"atoms": 3, "depth_max": 4},
            success_metric={"kind": "goal_hit"},
            uplift_spec={"description": "Test slice"},
        )
        self.assertEqual(slice_obj.name, "test_slice")
        self.assertEqual(slice_obj.params["atoms"], 3)
        self.assertEqual(slice_obj.success_metric["kind"], "goal_hit")

    def test_to_dict(self):
        """Test to_dict conversion."""
        slice_obj = UpliftSlice(
            name="test",
            params={"atoms": 2},
            success_metric={"kind": "density"},
        )
        d = slice_obj.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["params"]["atoms"], 2)

    def test_default_uplift_spec(self):
        """Test that uplift_spec defaults to empty dict."""
        slice_obj = UpliftSlice(
            name="test",
            params={},
            success_metric={},
        )
        self.assertEqual(slice_obj.uplift_spec, {})


class TestDeterministicHash(unittest.TestCase):
    """Test deterministic hashing functionality."""

    def test_identical_input_identical_hash(self):
        """Same input produces same hash."""
        data = {"name": "test", "params": {"atoms": 3}}
        hash1 = _compute_deterministic_hash(data)
        hash2 = _compute_deterministic_hash(data)
        self.assertEqual(hash1, hash2)

    def test_different_input_different_hash(self):
        """Different input produces different hash."""
        data1 = {"name": "test1"}
        data2 = {"name": "test2"}
        hash1 = _compute_deterministic_hash(data1)
        hash2 = _compute_deterministic_hash(data2)
        self.assertNotEqual(hash1, hash2)

    def test_key_order_independence(self):
        """Hash is independent of dict key order in source."""
        # Python dicts maintain insertion order, but our hash uses sort_keys
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        hash1 = _compute_deterministic_hash(data1)
        hash2 = _compute_deterministic_hash(data2)
        self.assertEqual(hash1, hash2)

    def test_nested_dict_determinism(self):
        """Hash is deterministic for nested structures."""
        data = {
            "outer": {
                "inner": {"value": 1},
                "list": [1, 2, 3],
            }
        }
        hashes = [_compute_deterministic_hash(data) for _ in range(10)]
        self.assertTrue(all(h == hashes[0] for h in hashes))

    def test_hash_slice_config_determinism(self):
        """hash_slice_config produces deterministic output."""
        slice_obj = UpliftSlice(
            name="test",
            params={"atoms": 5},
            success_metric={"kind": "goal_hit"},
        )
        hashes = [hash_slice_config(slice_obj) for _ in range(10)]
        self.assertTrue(all(h == hashes[0] for h in hashes))


class TestLoadPhase2Curriculum(unittest.TestCase):
    """Test loading Phase II curriculum YAML."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("version: 2.0\nslices:\n  test_slice:\n    items: [1, 2, 3]\n")
            f.flush()
            path = Path(f.name)

        try:
            config = load_phase2_curriculum(path)
            self.assertEqual(config["version"], 2.0)
            self.assertIn("slices", config)
        finally:
            path.unlink()

    def test_fail_fast_on_missing_file(self):
        """Test fail-fast on missing file."""
        with self.assertRaises(CurriculumLoadError) as ctx:
            load_phase2_curriculum(Path("/nonexistent/path.yaml"))
        self.assertIn("not found", str(ctx.exception))

    def test_fail_fast_on_malformed_yaml(self):
        """Test fail-fast on malformed YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content:\n  - bad indentation\n[broken")
            f.flush()
            path = Path(f.name)

        try:
            with self.assertRaises(CurriculumLoadError) as ctx:
                load_phase2_curriculum(path)
            self.assertIn("parse", str(ctx.exception).lower())
        finally:
            path.unlink()

    def test_fail_fast_on_empty_yaml(self):
        """Test fail-fast on empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            with self.assertRaises(CurriculumLoadError) as ctx:
                load_phase2_curriculum(path)
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            path.unlink()

    def test_reject_phase1_curriculum(self):
        """Test rejection of Phase I curriculum files."""
        # Create a temp file with Phase I path pattern
        with tempfile.TemporaryDirectory() as tmpdir:
            phase1_path = Path(tmpdir) / "config" / "curriculum.yaml"
            phase1_path.parent.mkdir(parents=True)
            phase1_path.write_text("version: 1\n")

            with self.assertRaises(CurriculumLoadError) as ctx:
                load_phase2_curriculum(phase1_path)
            self.assertIn("Phase I", str(ctx.exception))
            self.assertIn("Phase II only", str(ctx.exception))


class TestValidation(unittest.TestCase):
    """Test validation of slice parameters."""

    def _create_yaml_with_slice(self, slice_content: str) -> Path:
        """Helper to create a YAML file with a test slice."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(f"version: 2.0\nslices:\n  test_slice:\n{slice_content}")
            f.flush()
            return Path(f.name)

    def test_valid_atoms(self):
        """Test validation of valid atoms parameter."""
        path = self._create_yaml_with_slice("    atoms: 3\n")
        try:
            slice_obj = load_slice("test_slice", config_path=path)
            self.assertEqual(slice_obj.params.get("atoms"), 3)
        finally:
            path.unlink()

    def test_invalid_atoms_negative(self):
        """Test validation fails for negative atoms."""
        path = self._create_yaml_with_slice("    atoms: -1\n")
        try:
            with self.assertRaises(ValidationError) as ctx:
                load_slice("test_slice", config_path=path)
            self.assertIn("atoms", str(ctx.exception))
        finally:
            path.unlink()

    def test_invalid_atoms_zero(self):
        """Test validation fails for zero atoms."""
        path = self._create_yaml_with_slice("    atoms: 0\n")
        try:
            with self.assertRaises(ValidationError) as ctx:
                load_slice("test_slice", config_path=path)
            self.assertIn("atoms", str(ctx.exception))
        finally:
            path.unlink()

    def test_valid_depth_bounds(self):
        """Test validation of valid depth bounds."""
        path = self._create_yaml_with_slice("    depth_min: 2\n    depth_max: 5\n")
        try:
            slice_obj = load_slice("test_slice", config_path=path)
            self.assertEqual(slice_obj.params.get("depth_min"), 2)
            self.assertEqual(slice_obj.params.get("depth_max"), 5)
        finally:
            path.unlink()

    def test_invalid_depth_min_greater_than_max(self):
        """Test validation fails when depth_min > depth_max."""
        path = self._create_yaml_with_slice("    depth_min: 10\n    depth_max: 5\n")
        try:
            with self.assertRaises(ValidationError) as ctx:
                load_slice("test_slice", config_path=path)
            self.assertIn("depth_min", str(ctx.exception))
            self.assertIn("depth_max", str(ctx.exception))
        finally:
            path.unlink()

    def test_valid_max_candidates(self):
        """Test validation of valid max_candidates."""
        path = self._create_yaml_with_slice("    max_candidates: 50\n")
        try:
            slice_obj = load_slice("test_slice", config_path=path)
            self.assertEqual(slice_obj.params.get("max_candidates"), 50)
        finally:
            path.unlink()

    def test_invalid_max_candidates_zero(self):
        """Test validation fails for zero max_candidates."""
        path = self._create_yaml_with_slice("    max_candidates: 0\n")
        try:
            with self.assertRaises(ValidationError) as ctx:
                load_slice("test_slice", config_path=path)
            self.assertIn("max_candidates", str(ctx.exception))
        finally:
            path.unlink()


class TestSuccessMetricValidation(unittest.TestCase):
    """Test validation of success_metric configuration."""

    def _create_yaml_with_metric(self, metric_content: str) -> Path:
        """Helper to create a YAML file with a success metric."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            content = f"version: 2.0\nslices:\n  test_slice:\n    success_metric:\n{metric_content}"
            f.write(content)
            f.flush()
            return Path(f.name)

    def test_valid_success_metric_kinds(self):
        """Test all valid success metric kinds are accepted."""
        for kind in VALID_SUCCESS_METRIC_KINDS:
            path = self._create_yaml_with_metric(f"      kind: {kind}\n")
            try:
                slice_obj = load_slice("test_slice", config_path=path)
                self.assertEqual(slice_obj.success_metric.get("kind"), kind)
            finally:
                path.unlink()

    def test_invalid_success_metric_kind(self):
        """Test invalid success metric kind is rejected."""
        path = self._create_yaml_with_metric("      kind: invalid_kind\n")
        try:
            with self.assertRaises(ValidationError) as ctx:
                load_slice("test_slice", config_path=path)
            self.assertIn("success_metric.kind", str(ctx.exception))
            self.assertIn("invalid_kind", str(ctx.exception))
        finally:
            path.unlink()

    def test_empty_success_metric_allowed(self):
        """Test empty success metric is allowed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("version: 2.0\nslices:\n  test_slice:\n    items: [1, 2]\n")
            f.flush()
            path = Path(f.name)

        try:
            slice_obj = load_slice("test_slice", config_path=path)
            self.assertEqual(slice_obj.success_metric, {})
        finally:
            path.unlink()


class TestSliceLoading(unittest.TestCase):
    """Test slice loading functionality."""

    def test_load_nonexistent_slice(self):
        """Test loading a non-existent slice fails."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("version: 2.0\nslices:\n  real_slice:\n    items: [1]\n")
            f.flush()
            path = Path(f.name)

        try:
            with self.assertRaises(CurriculumLoadError) as ctx:
                load_slice("nonexistent", config_path=path)
            self.assertIn("not found", str(ctx.exception))
            self.assertIn("real_slice", str(ctx.exception))
        finally:
            path.unlink()

    def test_load_all_slices(self):
        """Test loading all slices from config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""version: 2.0
slices:
  slice_a:
    items: [1, 2]
  slice_b:
    items: [3, 4]
  slice_c:
    items: [5, 6]
""")
            f.flush()
            path = Path(f.name)

        try:
            slices = load_all_slices(config_path=path)
            self.assertEqual(len(slices), 3)
            # Should be sorted by name
            self.assertEqual(slices[0].name, "slice_a")
            self.assertEqual(slices[1].name, "slice_b")
            self.assertEqual(slices[2].name, "slice_c")
        finally:
            path.unlink()


class TestCurriculumLoader(unittest.TestCase):
    """Test the CurriculumLoader class."""

    def setUp(self):
        """Create a temporary YAML file for testing."""
        self.tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        self.tmpfile.write("""version: 2.0
slices:
  test_slice_1:
    items: [1, 2, 3]
    description: "Test slice 1"
  test_slice_2:
    atoms: 4
    depth_max: 5
    success_metric:
      kind: goal_hit
""")
        self.tmpfile.flush()
        self.path = Path(self.tmpfile.name)

    def tearDown(self):
        """Clean up temp file."""
        self.path.unlink()

    def test_lazy_loading(self):
        """Test that config is loaded lazily."""
        loader = CurriculumLoader(self.path)
        self.assertIsNone(loader._config)
        # Access config triggers loading
        _ = loader.config
        self.assertIsNotNone(loader._config)

    def test_config_hash_determinism(self):
        """Test config hash is deterministic."""
        loader1 = CurriculumLoader(self.path)
        loader2 = CurriculumLoader(self.path)
        self.assertEqual(loader1.config_hash, loader2.config_hash)

    def test_get_slice(self):
        """Test getting a specific slice."""
        loader = CurriculumLoader(self.path)
        slice_obj = loader.get_slice("test_slice_1")
        self.assertEqual(slice_obj.name, "test_slice_1")
        self.assertEqual(slice_obj.params["items"], [1, 2, 3])

    def test_get_slice_hash(self):
        """Test getting slice hash."""
        loader = CurriculumLoader(self.path)
        hash1 = loader.get_slice_hash("test_slice_1")
        hash2 = loader.get_slice_hash("test_slice_1")
        self.assertEqual(hash1, hash2)

    def test_list_slice_names(self):
        """Test listing slice names."""
        loader = CurriculumLoader(self.path)
        names = loader.list_slice_names()
        self.assertEqual(names, ["test_slice_1", "test_slice_2"])

    def test_get_all_slices(self):
        """Test getting all slices."""
        loader = CurriculumLoader(self.path)
        slices = loader.get_all_slices()
        self.assertEqual(len(slices), 2)

    def test_slice_caching(self):
        """Test that slices are cached after loading."""
        loader = CurriculumLoader(self.path)
        slice1 = loader.get_slice("test_slice_1")
        slice2 = loader.get_slice("test_slice_1")
        self.assertIs(slice1, slice2)


class TestRealCurriculum(unittest.TestCase):
    """Test against the real Phase II curriculum file."""

    @unittest.skipUnless(
        Path("config/curriculum_uplift_phase2.yaml").exists(),
        "Real curriculum file not present",
    )
    def test_load_real_curriculum(self):
        """Test loading the real Phase II curriculum."""
        loader = CurriculumLoader()
        slices = loader.get_all_slices()
        self.assertGreater(len(slices), 0)

    @unittest.skipUnless(
        Path("config/curriculum_uplift_phase2.yaml").exists(),
        "Real curriculum file not present",
    )
    def test_real_curriculum_determinism(self):
        """Test determinism with real curriculum."""
        loader1 = CurriculumLoader()
        loader2 = CurriculumLoader()
        self.assertEqual(loader1.config_hash, loader2.config_hash)

        for name in loader1.list_slice_names():
            hash1 = loader1.get_slice_hash(name)
            hash2 = loader2.get_slice_hash(name)
            self.assertEqual(hash1, hash2, f"Slice {name} hash mismatch")


class TestNonStrictMode(unittest.TestCase):
    """Test non-strict validation mode."""

    def test_non_strict_allows_invalid_atoms(self):
        """Test non-strict mode allows invalid atoms."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("version: 2.0\nslices:\n  test:\n    atoms: -1\n")
            f.flush()
            path = Path(f.name)

        try:
            # strict=False should not raise
            slice_obj = load_slice("test", config_path=path, strict=False)
            self.assertEqual(slice_obj.params.get("atoms"), -1)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
