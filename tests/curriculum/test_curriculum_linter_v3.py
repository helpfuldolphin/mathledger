# PHASE II — NOT USED IN PHASE I
# File: tests/curriculum/test_curriculum_linter_v3.py
"""
Test suite for the Curriculum Linter V3.
"""
import unittest
import os
import yaml
import json
import subprocess
import sys
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments'))
from curriculum_linter_v3 import CurriculumLinter, Violation

CANONICAL_VALID_SLICE = {
    'description': 'A valid test slice.',
    'uplift': {'phase': 'II', 'experiment_family': 'U2', 'not_allowed_in_phase_I': True},
    'parameters': {
        'atoms': 4, 'depth_min': 2, 'depth_max': 5, 'breadth_max': 40, 'total_max': 200,
        'formula_pool': 16, 'axiom_instances': 24, 'timeout_s': 1.0, 'lean_timeout_s': 0.0
    },
    'success_metric': {'kind': 'goal_hit', 'parameters': {'min_goal_hits': 1, 'min_total_verified': 3}},
    'budget': {'max_candidates_per_cycle': 40, 'max_cycles_per_run': 500},
    'formula_pool_entries': ['p', 'q']
}

class BaseLinterTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join("tests", "curriculum", "temp")
        os.makedirs(self.test_dir, exist_ok=True)
        self.curriculum_path = os.path.join(self.test_dir, "test_curriculum.yaml")

    def tearDown(self):
        if os.path.exists(self.curriculum_path): os.remove(self.curriculum_path)
        if os.path.exists(self.test_dir) and not os.listdir(self.test_dir): os.rmdir(self.test_dir)

    def get_linter_from_data(self, slices_dict):
        return CurriculumLinter(curriculum_data={'version': '2.1.0', 'slices': slices_dict})

    def write_curriculum(self, data):
        with open(self.curriculum_path, 'w') as f: yaml.dump(data, f)

    def assertHasViolation(self, linter, rule_id, slice_name='bad_slice'):
        self.assertTrue(
            any(v.rule_id == rule_id and v.slice_name == slice_name for v in linter.violations),
            f"Expected to find violation {rule_id} for slice {slice_name}, but not found in {linter.violations}"
        )

class TestValidCurriculum(BaseLinterTest):
    def test_linter_on_real_curriculum(self):
        linter = CurriculumLinter(curriculum_path="config/curriculum_uplift_phase2.yaml")
        self.assertTrue(linter.lint(), f"Linter found unexpected violations: {linter.violations}")

class TestStructuralInvariants(BaseLinterTest):
    def test_inv_s_01_depth_min_max(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE)
        slice_data['parameters']['depth_min'] = 6; slice_data['parameters']['depth_max'] = 5
        linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
        self.assertHasViolation(linter, 'INV-S-01')

    def test_inv_s_02_non_negative_params(self):
        for key in ['depth_min', 'depth_max', 'breadth_max', 'total_max']:
            with self.subTest(key=key):
                slice_data = copy.deepcopy(CANONICAL_VALID_SLICE)
                slice_data['parameters'][key] = -1
                linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
                self.assertHasViolation(linter, 'INV-S-02')

    def test_inv_s_03_unknown_metric_kind(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['success_metric']['kind'] = 'non_existent'
        linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
        self.assertHasViolation(linter, 'INV-S-03')
        
    def test_inv_s_04_mismatched_metric_params(self):
        schemas = CurriculumLinter().metric_schemas
        for kind in schemas:
            with self.subTest(kind=kind):
                slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['success_metric']['kind'] = kind
                slice_data['success_metric']['parameters'] = {'wrong_param': 1}
                linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
                self.assertHasViolation(linter, 'INV-S-04')

    def test_inv_s_05_empty_formula_pool(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['formula_pool_entries'] = []
        linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
        self.assertHasViolation(linter, 'INV-S-05')

class TestForbiddenPatterns(BaseLinterTest):
    def test_forbid_03_phase_not_ii(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['uplift']['phase'] = 'I'
        linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
        self.assertHasViolation(linter, 'FORBID-03')

    def test_forbid_04_min_chain_length(self):
        for length in [0, 1]:
            with self.subTest(length=length):
                slice_data = copy.deepcopy(CANONICAL_VALID_SLICE)
                slice_data['success_metric'] = {'kind': 'chain_success', 'parameters': {'min_chain_length': length}}
                linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
                self.assertHasViolation(linter, 'FORBID-04')

class TestDAGGeneration(BaseLinterTest):
    def test_dag_generation(self):
        linter = CurriculumLinter(curriculum_path="config/curriculum_uplift_phase2.yaml"); linter.lint()
        dag = linter.generate_dag()
        self.assertEqual(len(dag['nodes']), 4)
        edges = {(e['source'], e['target']) for e in dag['edges']}
        expected_edges = {
            ('slice_uplift_goal', 'slice_uplift_sparse'), ('slice_uplift_goal', 'slice_uplift_tree'),
            ('slice_uplift_goal', 'slice_uplift_dependency'), ('slice_uplift_tree', 'slice_uplift_sparse'),
            ('slice_uplift_tree', 'slice_uplift_dependency'), ('slice_uplift_sparse', 'slice_uplift_dependency')}
        self.assertEqual(edges, expected_edges)

    def test_dag_output_file_creation(self):
        script_path = os.path.join("experiments", "curriculum_linter_v3.py")
        dag_output_path = os.path.join(self.test_dir, "test_dag.json")
        self.write_curriculum({'version': '2.1', 'slices': {'s1': CANONICAL_VALID_SLICE}})
        # Use a valid curriculum so the script exits 0
        result = subprocess.run([sys.executable, script_path, "--curriculum", self.curriculum_path, "--dag-output", dag_output_path], capture_output=True)
        self.assertEqual(result.returncode, 0, f"Linter script failed unexpectedly: {result.stderr.decode()}")
        self.assertTrue(os.path.exists(dag_output_path))
        os.remove(dag_output_path)

class TestCIMode(BaseLinterTest):
    def setUp(self):
        super().setUp(); self.script_path = os.path.join("experiments", "curriculum_linter_v3.py")

    def test_ci_mode_success(self):
        self.write_curriculum({'version': '2.1', 'slices': {'s1': CANONICAL_VALID_SLICE}})
        result = subprocess.run([sys.executable, self.script_path, "--ci", "--curriculum", self.curriculum_path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"STDERR: {result.stderr}")
        self.assertIn("CI Mode: All checks passed.", result.stdout)

    def test_ci_mode_failure(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['parameters']['depth_min'] = 10
        self.write_curriculum({'version': '2.1', 'slices': {'bad_slice': slice_data}})
        result = subprocess.run([sys.executable, self.script_path, "--ci", "--curriculum", self.curriculum_path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("CI Mode: 1 violation(s) found.", result.stderr)

    def test_non_ci_mode_failure_exit_code_is_zero(self):
        slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); slice_data['parameters']['depth_min'] = 10
        self.write_curriculum({'version': '2.1', 'slices': {'bad_slice': slice_data}})
        result = subprocess.run([sys.executable, self.script_path, "--curriculum", self.curriculum_path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "Script should exit 0 in non-CI mode even with violations")
        self.assertIn("Found 1 violation(s)", result.stdout)

class TestGranularViolations(BaseLinterTest):
    def test_schema_violations_are_caught(self):
        schemas = CurriculumLinter().metric_schemas
        for kind, params in schemas.items():
            for param_to_miss in params:
                 with self.subTest(kind=kind, missing=param_to_miss):
                    slice_data = copy.deepcopy(CANONICAL_VALID_SLICE)
                    slice_data['success_metric']['kind'] = kind
                    valid_params = {p:1 for p in params}; del valid_params[param_to_miss]
                    slice_data['success_metric']['parameters'] = valid_params
                    linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
                    self.assertHasViolation(linter, 'INV-S-04')

    def test_missing_top_level_keys_are_caught(self):
        for key_to_miss in ['description', 'uplift', 'parameters', 'success_metric', 'budget', 'formula_pool_entries']:
            with self.subTest(missing=key_to_miss):
                slice_data = copy.deepcopy(CANONICAL_VALID_SLICE); del slice_data[key_to_miss]
                linter = self.get_linter_from_data({'bad_slice': slice_data}); self.assertFalse(linter.lint())
                self.assertHasViolation(linter, 'SCHEMA-FAIL')
                
    def test_loader_failure_is_handled(self):
        linter = CurriculumLinter(curriculum_path="non_existent_file.yaml")
        self.assertFalse(linter.lint()); self.assertEqual(len(linter.violations), 1)
        self.assertHasViolation(linter, 'LOAD-FAIL', slice_name='GLOBAL')
        
    def test_no_slices_key_is_handled(self):
        linter = self.get_linter_from_data({}) # No 'slices' key
        self.assertTrue(linter.lint()) # Linting an empty set of slices is valid
        self.assertEqual(len(linter.violations), 0)

if __name__ == '__main__':
    # Reduced verbosity for CI
    unittest.main(verbosity=2)