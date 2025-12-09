# PHASE II — NOT USED IN PHASE I
# File: tests/curriculum/test_drift_oracle.py
import unittest, os, yaml, subprocess, sys, copy, shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.curriculum_linter_v3 import build_drift_report

BASE_CURRICULUM = {
    'version': '2.1.0',
    'slices': {
        'slice_1': {
            'description': 'Base slice', 'uplift': {'phase': 'II'},
            'parameters': {'atoms': 4, 'depth_max': 5},
            'success_metric': {'kind': 'goal_hit', 'parameters': {'min_goal_hits': 1}},
        }
    }
}

class TestDriftOracle(unittest.TestCase):
    def setUp(self):
        """Set up a fresh git repo for each test."""
        self.repo_dir = os.path.abspath("tests/curriculum/temp_git_repo")
        if os.path.exists(self.repo_dir):
            shutil.rmtree(self.repo_dir, ignore_errors=True)
        os.makedirs(self.repo_dir, exist_ok=True)
        
        subprocess.check_call(['git', 'init', '-b', 'main'], cwd=self.repo_dir)
        subprocess.check_call(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_dir)
        subprocess.check_call(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_dir)
        self._commit("README.md", "init", "Initial commit")

    def tearDown(self):
        """Clean up the git repo."""
        shutil.rmtree(self.repo_dir, ignore_errors=True)

    def _commit(self, filename, content, message):
        path = os.path.join(self.repo_dir, filename)
        with open(path, 'w') as f:
            if isinstance(content, dict): yaml.dump(content, f)
            else: f.write(content)
        subprocess.check_call(['git', 'add', path], cwd=self.repo_dir)
        subprocess.check_call(['git', 'commit', '--allow-empty', '-m', message], cwd=self.repo_dir)
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo_dir, text=True).strip()

    def test_no_drift(self):
        sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Initial")
        report = build_drift_report(sha, sha, "curriculum.yaml", self.repo_dir)
        self.assertEqual(report['summary']['total_drifts_detected'], 0)

    def test_parametric_drift_warning(self):
        base_sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Base")
        mod = copy.deepcopy(BASE_CURRICULUM)
        mod['slices']['slice_1']['parameters']['depth_max'] = 6
        head_sha = self._commit("curriculum.yaml", mod, "Change depth")
        report = build_drift_report(base_sha, head_sha, "curriculum.yaml", self.repo_dir)
        self.assertEqual(report['summary']['total_drifts_detected'], 1)
        self.assertEqual(report['drifts'][0]['severity'], 'WARNING')
        self.assertEqual(report['drifts'][0]['parameter'], 'parameters.depth_max')

    def test_semantic_drift_critical(self):
        base_sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Base")
        mod = copy.deepcopy(BASE_CURRICULUM)
        mod['slices']['slice_1']['success_metric']['kind'] = 'chain_success'
        head_sha = self._commit("curriculum.yaml", mod, "Change metric")
        report = build_drift_report(base_sha, head_sha, "curriculum.yaml", self.repo_dir)
        self.assertEqual(report['summary']['total_drifts_detected'], 1)
        self.assertEqual(report['drifts'][0]['severity'], 'CRITICAL')
        self.assertEqual(report['drifts'][0]['parameter'], 'success_metric.kind')

    def test_slice_removed_critical(self):
        base_sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Base")
        mod = copy.deepcopy(BASE_CURRICULUM); del mod['slices']['slice_1']
        head_sha = self._commit("curriculum.yaml", mod, "Remove slice")
        report = build_drift_report(base_sha, head_sha, "curriculum.yaml", self.repo_dir)
        self.assertEqual(report['summary']['total_drifts_detected'], 1)
        self.assertEqual(report['drifts'][0]['severity'], 'CRITICAL')

    def test_slice_added_warning(self):
        base_sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Base")
        mod = copy.deepcopy(BASE_CURRICULUM); mod['slices']['slice_2'] = {}
        head_sha = self._commit("curriculum.yaml", mod, "Add slice")
        report = build_drift_report(base_sha, head_sha, "curriculum.yaml", self.repo_dir)
        self.assertEqual(report['summary']['total_drifts_detected'], 1)
        self.assertEqual(report['drifts'][0]['severity'], 'WARNING')

    def test_ci_mode_exit_codes(self):
        script = os.path.abspath(os.path.join("experiments", "curriculum_linter_v3.py"))
        base_sha = self._commit("curriculum.yaml", BASE_CURRICULUM, "Base")

        # CRITICAL change
        crit_mod = copy.deepcopy(BASE_CURRICULUM); del crit_mod['slices']['slice_1']
        head_crit = self._commit("curriculum.yaml", crit_mod, "Crit mod")
        res_crit = subprocess.run([sys.executable, script, "--drift-check", "--ci", "--base-commit", base_sha, "--head-commit", head_crit, "--curriculum", "curriculum.yaml"], cwd=self.repo_dir)
        self.assertEqual(res_crit.returncode, 1, "Should exit 1 on CRITICAL drift")

        # WARNING change
        warn_mod = copy.deepcopy(BASE_CURRICULUM); warn_mod['slices']['slice_1']['parameters']['depth_max'] = 10
        head_warn = self._commit("curriculum.yaml", warn_mod, "Warn mod")
        res_warn = subprocess.run([sys.executable, script, "--drift-check", "--ci", "--base-commit", base_sha, "--head-commit", head_warn, "--curriculum", "curriculum.yaml"], cwd=self.repo_dir)
        self.assertEqual(res_warn.returncode, 0, "Should exit 0 on WARNING-only drift")

if __name__ == '__main__':
    unittest.main(verbosity=2)