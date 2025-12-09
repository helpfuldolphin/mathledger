# PHASE II — NOT USED IN PHASE I
# File: tests/experiments/test_curriculum_loader_v2.py
"""
Tests for CurriculumLoaderV2 and Phase II uplift slice configurations.

These tests verify:
1. The curriculum file loads without errors
2. All four uplift slices are present and valid
3. Each slice has the expected configuration matching docs
4. Hash computation is deterministic and stable
5. Validation catches invalid configurations
6. Monotonicity checks work correctly

Reference Documents:
- docs/PHASE2_RFL_UPLIFT_PLAN.md (expected slice parameters)
- experiments/slice_success_metrics.py (valid metric kinds)
"""

import unittest
import os
import yaml
import hashlib
import json
import tempfile
from typing import Dict, Any

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    VALID_SUCCESS_METRIC_KINDS,
    REQUIRED_SLICE_FIELDS,
    REQUIRED_PARAMETER_FIELDS,
    FormulaPoolIntegrityResult,
    SuccessMetricValidationResult,
)


class TestCurriculumLoaderV2Basic(unittest.TestCase):
    """Basic loader functionality tests using mock data."""

    def setUp(self):
        """Set up a dummy curriculum file for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_dir, "test_curriculum.yaml")
        self.valid_test_data: Dict[str, Any] = {
            'version': '2.1.0',
            'slices': {
                'test_slice_1': {
                    'description': 'A test slice.',
                    'uplift': {'phase': 'II', 'not_allowed_in_phase_I': True},
                    'parameters': {
                        'atoms': 4,
                        'depth_min': 2,
                        'depth_max': 5,
                        'breadth_max': 40,
                        'total_max': 200,
                        'formula_pool': 16,
                        'axiom_instances': 24,
                    },
                    'success_metric': {
                        'kind': 'goal_hit',
                        'parameters': {'min_goal_hits': 1}
                    },
                    'budget': {'max_candidates_per_cycle': 50},
                    'formula_pool_entries': ['p', 'q', 'p->q']
                }
            }
        }
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.valid_test_data, f)

        self.loader = CurriculumLoaderV2(filepath=self.test_config_path)

    def tearDown(self):
        """Clean up the dummy curriculum file."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        os.rmdir(self.test_dir)

    def test_load_valid_file(self):
        """Test that a valid curriculum file loads without errors."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.get_version(), '2.1.0')

    def test_list_slices(self):
        """Test that list_slices returns slice names."""
        slices = self.loader.list_slices()
        self.assertEqual(slices, ['test_slice_1'])

    def test_get_slice_config(self):
        """Test that get_slice_config returns the correct dictionary."""
        config = self.loader.get_slice_config('test_slice_1')
        self.assertEqual(config['description'], 'A test slice.')
        self.assertEqual(config['budget']['max_candidates_per_cycle'], 50)

    def test_get_parameters(self):
        """Test that get_parameters returns the parameters block."""
        params = self.loader.get_parameters('test_slice_1')
        self.assertEqual(params['atoms'], 4)
        self.assertEqual(params['depth_max'], 5)

    def test_get_success_metric_config(self):
        """Test retrieval of the success_metric block."""
        metric_config = self.loader.get_success_metric_config('test_slice_1')
        self.assertEqual(metric_config['kind'], 'goal_hit')
        self.assertEqual(metric_config['parameters']['min_goal_hits'], 1)

    def test_get_slice_config_not_found(self):
        """Test that get_slice_config raises KeyError for a non-existent slice."""
        with self.assertRaises(KeyError):
            self.loader.get_slice_config('non_existent_slice')

    def test_hash_slice_config_determinism(self):
        """Test that hashing is deterministic."""
        hash1 = self.loader.hash_slice_config('test_slice_1')
        hash2 = self.loader.hash_slice_config('test_slice_1')
        self.assertEqual(hash1, hash2)

        # Also check against a known hash value to guard against changes
        slice_config = self.loader.get_slice_config('test_slice_1')
        canonical_json = json.dumps(slice_config, sort_keys=True)
        known_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        self.assertEqual(hash1, known_hash)


class TestCurriculumLoaderV2Validation(unittest.TestCase):
    """Tests for validation error handling."""

    def setUp(self):
        """Set up temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        for fname in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, fname))
        os.rmdir(self.test_dir)

    def _write_and_load(self, data: Dict[str, Any]) -> CurriculumLoaderV2:
        """Helper to write YAML and attempt loading."""
        path = os.path.join(self.test_dir, "test.yaml")
        with open(path, 'w') as f:
            yaml.dump(data, f)
        return CurriculumLoaderV2(filepath=path)

    def test_validation_failure_missing_version(self):
        """Test that missing version raises ValueError."""
        invalid_data = {'slices': {}}
        with self.assertRaisesRegex(ValueError, "must contain a 'version' field"):
            self._write_and_load(invalid_data)

    def test_validation_failure_wrong_version(self):
        """Test that version 1.x raises ValueError."""
        invalid_data = {'version': '1.0', 'slices': {}}
        with self.assertRaisesRegex(ValueError, "Phase II curriculum requires version 2.x"):
            self._write_and_load(invalid_data)

    def test_validation_failure_missing_slices(self):
        """Test that missing 'slices' block raises ValueError."""
        invalid_data = {'version': '2.0'}
        with self.assertRaisesRegex(ValueError, "must contain a 'slices' block"):
            self._write_and_load(invalid_data)

    def test_validation_failure_missing_key_in_slice(self):
        """Test that a slice with missing keys raises ValueError."""
        invalid_data = {
            'version': '2.0',
            'slices': {
                'bad_slice': {
                    'description': 'Missing fields',
                    # Missing: uplift, parameters, success_metric, budget, formula_pool_entries
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "missing required key"):
            self._write_and_load(invalid_data)

    def test_validation_failure_invalid_metric_kind(self):
        """Test that invalid success_metric.kind raises ValueError."""
        invalid_data = {
            'version': '2.0',
            'slices': {
                'bad_slice': {
                    'description': 'Bad metric kind',
                    'uplift': {'phase': 'II'},
                    'parameters': {
                        'atoms': 4,
                        'depth_min': 2,
                        'depth_max': 5,
                        'breadth_max': 40,
                        'total_max': 200,
                        'formula_pool': 16,
                        'axiom_instances': 24,
                    },
                    'success_metric': {'kind': 'invalid_metric'},
                    'budget': {'max_candidates_per_cycle': 40},
                    'formula_pool_entries': ['p']
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid success_metric.kind"):
            self._write_and_load(invalid_data)

    def test_validation_failure_empty_formula_pool(self):
        """Test that empty formula_pool_entries raises ValueError."""
        invalid_data = {
            'version': '2.0',
            'slices': {
                'bad_slice': {
                    'description': 'Empty pool',
                    'uplift': {'phase': 'II'},
                    'parameters': {
                        'atoms': 4,
                        'depth_min': 2,
                        'depth_max': 5,
                        'breadth_max': 40,
                        'total_max': 200,
                        'formula_pool': 16,
                        'axiom_instances': 24,
                    },
                    'success_metric': {'kind': 'goal_hit'},
                    'budget': {'max_candidates_per_cycle': 40},
                    'formula_pool_entries': []
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self._write_and_load(invalid_data)


class TestRealCurriculumFile(unittest.TestCase):
    """Tests against the actual curriculum_uplift_phase2.yaml file."""

    @classmethod
    def setUpClass(cls):
        """Load the real curriculum file once for all tests."""
        cls.loader = CurriculumLoaderV2()

    def test_load_real_file(self):
        """Test that the real curriculum file loads successfully."""
        self.assertIsNotNone(self.loader)
        self.assertTrue(self.loader.get_version().startswith('2.'))

    def test_exactly_four_slices(self):
        """Test that exactly four uplift slices are defined."""
        slices = self.loader.list_slices()
        self.assertEqual(len(slices), 4)

    def test_expected_slice_names(self):
        """Test that all expected slice names are present."""
        expected_slices = [
            'slice_uplift_goal',
            'slice_uplift_sparse',
            'slice_uplift_tree',
            'slice_uplift_dependency',
        ]
        actual_slices = self.loader.list_slices()
        for expected in expected_slices:
            self.assertIn(expected, actual_slices,
                          f"Expected slice '{expected}' not found in curriculum")


class TestSliceUpliftGoal(unittest.TestCase):
    """Tests for slice_uplift_goal configuration."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()
        cls.config = cls.loader.get_slice_config('slice_uplift_goal')
        cls.params = cls.loader.get_parameters('slice_uplift_goal')
        cls.metric = cls.loader.get_success_metric_config('slice_uplift_goal')

    def test_description_present(self):
        """Test that description is non-empty."""
        self.assertTrue(len(self.config['description']) > 0)
        self.assertIn('goal', self.config['description'].lower())

    def test_atoms_matches_doc(self):
        """Test atoms=4 as specified in PHASE2_RFL_UPLIFT_PLAN.md."""
        self.assertEqual(self.params['atoms'], 4)

    def test_depth_range(self):
        """Test depth range 2-5 as specified in docs."""
        self.assertEqual(self.params['depth_min'], 2)
        self.assertEqual(self.params['depth_max'], 5)

    def test_budget_matches_doc(self):
        """Test max 40 candidates/cycle as specified in docs."""
        budget = self.loader.get_budget('slice_uplift_goal')
        self.assertEqual(budget['max_candidates_per_cycle'], 40)

    def test_success_metric_kind(self):
        """Test success_metric.kind is 'goal_hit'."""
        self.assertEqual(self.metric['kind'], 'goal_hit')

    def test_success_metric_parameters(self):
        """Test success_metric has expected parameters."""
        self.assertIn('parameters', self.metric)
        self.assertIn('min_goal_hits', self.metric['parameters'])
        self.assertIn('min_total_verified', self.metric['parameters'])

    def test_formula_pool_non_empty(self):
        """Test formula pool has entries."""
        pool = self.loader.get_formula_pool('slice_uplift_goal')
        self.assertGreater(len(pool), 0)

    def test_hash_stability(self):
        """Test that hash is deterministic (compute twice, compare)."""
        h1 = self.loader.hash_slice_config('slice_uplift_goal')
        h2 = self.loader.hash_slice_config('slice_uplift_goal')
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # SHA256 hex length


class TestSliceUpliftSparse(unittest.TestCase):
    """Tests for slice_uplift_sparse configuration."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()
        cls.config = cls.loader.get_slice_config('slice_uplift_sparse')
        cls.params = cls.loader.get_parameters('slice_uplift_sparse')
        cls.metric = cls.loader.get_success_metric_config('slice_uplift_sparse')

    def test_description_present(self):
        """Test that description is non-empty."""
        self.assertTrue(len(self.config['description']) > 0)
        self.assertIn('sparse', self.config['description'].lower())

    def test_atoms_matches_doc(self):
        """Test atoms=5 as specified in PHASE2_RFL_UPLIFT_PLAN.md."""
        self.assertEqual(self.params['atoms'], 5)

    def test_depth_range(self):
        """Test depth range 3-7 as specified in docs."""
        self.assertEqual(self.params['depth_min'], 3)
        self.assertEqual(self.params['depth_max'], 7)

    def test_budget_matches_doc(self):
        """Test max 40 candidates/cycle as specified in docs."""
        budget = self.loader.get_budget('slice_uplift_sparse')
        self.assertEqual(budget['max_candidates_per_cycle'], 40)

    def test_success_metric_kind(self):
        """Test success_metric.kind is 'sparse_success'."""
        self.assertEqual(self.metric['kind'], 'sparse_success')

    def test_success_metric_parameters(self):
        """Test success_metric has expected parameters."""
        self.assertIn('parameters', self.metric)
        self.assertIn('min_verified', self.metric['parameters'])
        # Per docs: verified >= 5
        self.assertEqual(self.metric['parameters']['min_verified'], 5)

    def test_larger_than_goal_slice(self):
        """Test that sparse slice is more complex than goal slice."""
        goal_params = self.loader.get_parameters('slice_uplift_goal')
        # More atoms
        self.assertGreater(self.params['atoms'], goal_params['atoms'])
        # Higher depth range
        self.assertGreater(self.params['depth_max'], goal_params['depth_max'])


class TestSliceUpliftTree(unittest.TestCase):
    """Tests for slice_uplift_tree configuration."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()
        cls.config = cls.loader.get_slice_config('slice_uplift_tree')
        cls.params = cls.loader.get_parameters('slice_uplift_tree')
        cls.metric = cls.loader.get_success_metric_config('slice_uplift_tree')

    def test_description_present(self):
        """Test that description is non-empty."""
        self.assertTrue(len(self.config['description']) > 0)
        self.assertIn('chain', self.config['description'].lower())

    def test_atoms_matches_doc(self):
        """Test atoms=4 as specified in PHASE2_RFL_UPLIFT_PLAN.md."""
        self.assertEqual(self.params['atoms'], 4)

    def test_depth_range(self):
        """Test depth range 2-6 as specified in docs."""
        self.assertEqual(self.params['depth_min'], 2)
        self.assertEqual(self.params['depth_max'], 6)

    def test_budget_matches_doc(self):
        """Test max 30 candidates/cycle as specified in docs."""
        budget = self.loader.get_budget('slice_uplift_tree')
        self.assertEqual(budget['max_candidates_per_cycle'], 30)

    def test_success_metric_kind(self):
        """Test success_metric.kind is 'chain_success'."""
        self.assertEqual(self.metric['kind'], 'chain_success')

    def test_success_metric_parameters(self):
        """Test success_metric has expected parameters."""
        self.assertIn('parameters', self.metric)
        self.assertIn('min_chain_length', self.metric['parameters'])
        # Per docs: proof_depth >= 3
        self.assertEqual(self.metric['parameters']['min_chain_length'], 3)

    def test_tighter_budget_than_goal(self):
        """Test that tree slice has tighter budget than goal slice."""
        goal_budget = self.loader.get_budget('slice_uplift_goal')
        tree_budget = self.loader.get_budget('slice_uplift_tree')
        self.assertLess(
            tree_budget['max_candidates_per_cycle'],
            goal_budget['max_candidates_per_cycle']
        )


class TestSliceUpliftDependency(unittest.TestCase):
    """Tests for slice_uplift_dependency configuration."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()
        cls.config = cls.loader.get_slice_config('slice_uplift_dependency')
        cls.params = cls.loader.get_parameters('slice_uplift_dependency')
        cls.metric = cls.loader.get_success_metric_config('slice_uplift_dependency')

    def test_description_present(self):
        """Test that description is non-empty."""
        self.assertTrue(len(self.config['description']) > 0)
        self.assertIn('multi-goal', self.config['description'].lower())

    def test_atoms_matches_doc(self):
        """Test atoms=5 as specified in PHASE2_RFL_UPLIFT_PLAN.md."""
        self.assertEqual(self.params['atoms'], 5)

    def test_depth_range(self):
        """Test depth range 2-6 as specified in docs."""
        self.assertEqual(self.params['depth_min'], 2)
        self.assertEqual(self.params['depth_max'], 6)

    def test_budget_matches_doc(self):
        """Test max 40 candidates/cycle as specified in docs."""
        budget = self.loader.get_budget('slice_uplift_dependency')
        self.assertEqual(budget['max_candidates_per_cycle'], 40)

    def test_success_metric_kind(self):
        """Test success_metric.kind is 'multi_goal_success'."""
        self.assertEqual(self.metric['kind'], 'multi_goal_success')

    def test_success_metric_parameters(self):
        """Test success_metric has expected parameters."""
        self.assertIn('parameters', self.metric)
        # Should specify the number of required goals
        self.assertIn('required_goal_count', self.metric['parameters'])

    def test_more_atoms_than_tree(self):
        """Test that dependency slice has more atoms than tree slice."""
        tree_params = self.loader.get_parameters('slice_uplift_tree')
        self.assertGreater(self.params['atoms'], tree_params['atoms'])


class TestMonotonicity(unittest.TestCase):
    """Tests for slice complexity monotonicity."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_monotonicity_check_runs(self):
        """Test that monotonicity check executes without error."""
        warnings = self.loader.validate_monotonicity()
        # This returns a list of warnings (could be empty)
        self.assertIsInstance(warnings, list)

    def test_no_severe_violations(self):
        """Test that there are no unexpected monotonicity violations."""
        warnings = self.loader.validate_monotonicity()
        # Log any warnings for visibility
        for w in warnings:
            print(f"Monotonicity warning: {w}")
        # We allow some warnings as slices may be in different families
        # But we shouldn't have more than 2 (different families crossing)
        self.assertLessEqual(len(warnings), 2,
                             f"Too many monotonicity violations: {warnings}")


class TestAllSlicesHaveValidMetrics(unittest.TestCase):
    """Test that all slices use valid success metric kinds."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_all_metrics_valid(self):
        """Test that every slice has a valid metric kind."""
        for slice_name in self.loader.list_slices():
            metric = self.loader.get_success_metric_config(slice_name)
            self.assertIn(
                metric['kind'],
                VALID_SUCCESS_METRIC_KINDS,
                f"Slice '{slice_name}' has invalid metric kind: {metric['kind']}"
            )


class TestHashStabilityAcrossSlices(unittest.TestCase):
    """Test hash stability and uniqueness across all slices."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_all_hashes_unique(self):
        """Test that each slice produces a unique hash."""
        hashes = {}
        for slice_name in self.loader.list_slices():
            h = self.loader.hash_slice_config(slice_name)
            self.assertNotIn(h, hashes.values(),
                             f"Hash collision between slices")
            hashes[slice_name] = h

    def test_hashes_are_sha256(self):
        """Test that all hashes are valid SHA256 hex strings."""
        for slice_name in self.loader.list_slices():
            h = self.loader.hash_slice_config(slice_name)
            self.assertEqual(len(h), 64, "SHA256 hex should be 64 chars")
            # Verify it's valid hex
            int(h, 16)  # Will raise if not valid hex


# =============================================================================
# NEW TESTS: Diagnostics, Integrity, Hash Determinism, Metric Alignment
# =============================================================================

class TestDescribeSlice(unittest.TestCase):
    """Tests for describe_slice() introspection utility."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_describe_slice_returns_string(self):
        """Test that describe_slice returns a non-empty string."""
        description = self.loader.describe_slice('slice_uplift_goal')
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 100)

    def test_describe_slice_contains_key_info(self):
        """Test that description contains essential information."""
        description = self.loader.describe_slice('slice_uplift_goal')
        # Should contain slice name
        self.assertIn('slice_uplift_goal', description)
        # Should contain parameter info
        self.assertIn('atoms', description)
        self.assertIn('depth_min', description)
        self.assertIn('depth_max', description)
        # Should contain metric info
        self.assertIn('goal_hit', description)
        # Should contain hash
        self.assertIn('Config Hash', description)

    def test_describe_all_slices(self):
        """Test that all slices can be described without error."""
        for slice_name in self.loader.list_slices():
            description = self.loader.describe_slice(slice_name)
            self.assertIsInstance(description, str)
            self.assertIn(slice_name, description)


class TestValidateSuccessMetric(unittest.TestCase):
    """Tests for validate_success_metric() introspection utility."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_goal_hit_metric_valid(self):
        """Test that slice_uplift_goal has valid metric params."""
        result = self.loader.validate_success_metric('slice_uplift_goal')
        self.assertTrue(result.valid)
        self.assertEqual(result.metric_kind, 'goal_hit')
        self.assertEqual(len(result.missing_params), 0)
        self.assertEqual(len(result.unknown_params), 0)

    def test_sparse_success_metric_valid(self):
        """Test that slice_uplift_sparse has valid metric params."""
        result = self.loader.validate_success_metric('slice_uplift_sparse')
        self.assertTrue(result.valid)
        self.assertEqual(result.metric_kind, 'sparse_success')
        self.assertIn('min_verified', result.param_values)

    def test_chain_success_metric_valid(self):
        """Test that slice_uplift_tree has valid metric params."""
        result = self.loader.validate_success_metric('slice_uplift_tree')
        self.assertTrue(result.valid)
        self.assertEqual(result.metric_kind, 'chain_success')
        self.assertIn('min_chain_length', result.param_values)

    def test_multi_goal_metric_valid(self):
        """Test that slice_uplift_dependency has valid metric params."""
        result = self.loader.validate_success_metric('slice_uplift_dependency')
        self.assertTrue(result.valid)
        self.assertEqual(result.metric_kind, 'multi_goal_success')
        self.assertIn('required_goal_count', result.param_values)


class TestValidateFormulaPoolIntegrity(unittest.TestCase):
    """Tests for validate_formula_pool_integrity() introspection utility."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_no_duplicate_formulas_in_goal_slice(self):
        """Test that slice_uplift_goal has no duplicate formulas."""
        result = self.loader.validate_formula_pool_integrity('slice_uplift_goal')
        self.assertEqual(len(result.duplicate_formulas), 0,
                         f"Found duplicates: {result.duplicate_formulas}")

    def test_all_formulas_normalize_cleanly(self):
        """Test that all formulas in all slices normalize without errors."""
        for slice_name in self.loader.list_slices():
            result = self.loader.validate_formula_pool_integrity(slice_name)
            self.assertEqual(len(result.normalization_errors), 0,
                             f"Slice '{slice_name}' has normalization errors: "
                             f"{result.normalization_errors}")

    def test_no_hash_collisions(self):
        """Test that no hash collisions exist in any slice."""
        for slice_name in self.loader.list_slices():
            result = self.loader.validate_formula_pool_integrity(slice_name)
            self.assertEqual(len(result.hash_collisions), 0,
                             f"Slice '{slice_name}' has hash collisions: "
                             f"{result.hash_collisions}")

    def test_integrity_result_contains_hashes(self):
        """Test that integrity result includes normalized hashes."""
        result = self.loader.validate_formula_pool_integrity('slice_uplift_goal')
        self.assertIsInstance(result.normalized_hashes, dict)
        self.assertGreater(len(result.normalized_hashes), 0)
        # Each entry should be (normalized_form, hash)
        for formula, (normalized, formula_hash) in result.normalized_hashes.items():
            self.assertIsInstance(normalized, str)
            self.assertEqual(len(formula_hash), 64)  # SHA256 hex


class TestValidateAll(unittest.TestCase):
    """Tests for validate_all() comprehensive validation."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_validate_all_returns_dict(self):
        """Test that validate_all returns a properly structured dict."""
        result = self.loader.validate_all()
        self.assertIsInstance(result, dict)
        self.assertIn('valid', result)
        self.assertIn('version', result)
        self.assertIn('slice_count', result)
        self.assertIn('slices', result)
        self.assertIn('monotonicity_warnings', result)

    def test_validate_all_overall_valid(self):
        """Test that the real curriculum passes all validation."""
        result = self.loader.validate_all()
        self.assertTrue(result['valid'],
                        f"Validation failed with issues in slices: "
                        f"{[(k, v['issues']) for k, v in result['slices'].items() if v['issues']]}")

    def test_validate_all_has_all_slices(self):
        """Test that validate_all covers all slices."""
        result = self.loader.validate_all()
        self.assertEqual(result['slice_count'], 4)
        for slice_name in self.loader.list_slices():
            self.assertIn(slice_name, result['slices'])


class TestHashDeterminismAdvanced(unittest.TestCase):
    """Advanced tests for hash determinism and stability."""

    def setUp(self):
        """Create fresh loader instances for each test."""
        self.loader1 = CurriculumLoaderV2()
        self.loader2 = CurriculumLoaderV2()

    def test_hash_identical_across_loader_instances(self):
        """Test that hashes are identical across different loader instances."""
        for slice_name in self.loader1.list_slices():
            h1 = self.loader1.hash_slice_config(slice_name)
            h2 = self.loader2.hash_slice_config(slice_name)
            self.assertEqual(h1, h2,
                             f"Hash mismatch for '{slice_name}' across loader instances")

    def test_to_json_deterministic(self):
        """Test that to_json produces identical output across calls."""
        json1 = self.loader1.to_json()
        json2 = self.loader2.to_json()
        self.assertEqual(json1, json2)

    def test_json_parseable(self):
        """Test that to_json output is valid JSON."""
        json_str = self.loader1.to_json()
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertIn('version', parsed)
        self.assertIn('slices', parsed)


class TestMetricSliceAlignment(unittest.TestCase):
    """Tests for success metric and slice alignment."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CurriculumLoaderV2()

    def test_goal_slice_uses_goal_metric(self):
        """Test that goal slice uses goal_hit metric."""
        metric = self.loader.get_success_metric_config('slice_uplift_goal')
        self.assertEqual(metric['kind'], 'goal_hit')

    def test_sparse_slice_uses_sparse_metric(self):
        """Test that sparse slice uses sparse_success metric."""
        metric = self.loader.get_success_metric_config('slice_uplift_sparse')
        self.assertEqual(metric['kind'], 'sparse_success')

    def test_tree_slice_uses_chain_metric(self):
        """Test that tree slice uses chain_success metric."""
        metric = self.loader.get_success_metric_config('slice_uplift_tree')
        self.assertEqual(metric['kind'], 'chain_success')

    def test_dependency_slice_uses_multi_goal_metric(self):
        """Test that dependency slice uses multi_goal_success metric."""
        metric = self.loader.get_success_metric_config('slice_uplift_dependency')
        self.assertEqual(metric['kind'], 'multi_goal_success')

    def test_all_slices_have_metric_parameters(self):
        """Test that all slices have parameters for their metrics."""
        for slice_name in self.loader.list_slices():
            metric = self.loader.get_success_metric_config(slice_name)
            self.assertIn('parameters', metric,
                          f"Slice '{slice_name}' missing metric parameters")
            self.assertIsInstance(metric['parameters'], dict)


if __name__ == '__main__':
    unittest.main()
