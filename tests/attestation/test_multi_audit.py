"""
Tests for multi-experiment audit runner.

PHASE II — NOT USED IN PHASE I
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.audit_uplift_u2_all import (
    discover_experiments,
    aggregate_results,
    audit_all_experiments,
    format_multi_audit_json,
    format_multi_audit_markdown,
)


class TestExperimentDiscovery(unittest.TestCase):
    """Test experiment directory discovery."""
    
    def setUp(self):
        """Create temporary directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_discover_single_experiment(self):
        """Test discovery of a single experiment."""
        exp_dir = self.root / "exp1"
        exp_dir.mkdir()
        (exp_dir / "manifest.json").touch()
        
        experiments = discover_experiments(self.root)
        
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0], exp_dir)
    
    def test_discover_multiple_experiments(self):
        """Test discovery of multiple experiments."""
        # Create 3 experiment directories
        for i in range(1, 4):
            exp_dir = self.root / f"exp{i}"
            exp_dir.mkdir()
            (exp_dir / "manifest.json").touch()
        
        experiments = discover_experiments(self.root)
        
        self.assertEqual(len(experiments), 3)
    
    def test_discover_nested_experiments(self):
        """Test discovery of experiments in nested directories."""
        # Create nested structure
        nested = self.root / "level1" / "level2"
        nested.mkdir(parents=True)
        (nested / "manifest.json").touch()
        
        # Also create one at root level
        root_exp = self.root / "exp_root"
        root_exp.mkdir()
        (root_exp / "manifest.json").touch()
        
        experiments = discover_experiments(self.root)
        
        self.assertEqual(len(experiments), 2)
        self.assertIn(nested, experiments)
        self.assertIn(root_exp, experiments)
    
    def test_discover_no_experiments(self):
        """Test discovery when no experiments exist."""
        # Empty directory
        experiments = discover_experiments(self.root)
        
        self.assertEqual(len(experiments), 0)
    
    def test_discover_ignores_non_experiment_dirs(self):
        """Test that directories without manifest.json are ignored."""
        # Create directory without manifest
        no_manifest = self.root / "not_an_experiment"
        no_manifest.mkdir()
        (no_manifest / "some_file.txt").touch()
        
        # Create actual experiment
        exp_dir = self.root / "real_exp"
        exp_dir.mkdir()
        (exp_dir / "manifest.json").touch()
        
        experiments = discover_experiments(self.root)
        
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0], exp_dir)


class TestResultAggregation(unittest.TestCase):
    """Test aggregation of multiple audit results."""
    
    def test_aggregate_all_passed(self):
        """Test aggregation when all experiments pass."""
        results = [
            {"status": "PASS", "findings": []},
            {"status": "PASS", "findings": []},
            {"status": "PASS", "findings": []},
        ]
        
        summary = aggregate_results(results)
        
        self.assertEqual(summary["total_experiments"], 3)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["missing"], 0)
    
    def test_aggregate_mixed_statuses(self):
        """Test aggregation with mixed statuses."""
        results = [
            {"status": "PASS", "findings": []},
            {"status": "FAIL", "findings": [{"category": "log_file", "severity": "ERROR"}]},
            {"status": "MISSING", "findings": [{"category": "manifest", "severity": "MISSING"}]},
            {"status": "PASS", "findings": []},
        ]
        
        summary = aggregate_results(results)
        
        self.assertEqual(summary["total_experiments"], 4)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["missing"], 1)
    
    def test_aggregate_findings_by_category(self):
        """Test that findings are grouped by category."""
        results = [
            {
                "status": "FAIL",
                "findings": [
                    {"category": "log_file", "severity": "ERROR"},
                    {"category": "cycle_count", "severity": "ERROR"},
                ]
            },
            {
                "status": "FAIL",
                "findings": [
                    {"category": "log_file", "severity": "ERROR"},
                    {"category": "hash_mismatch", "severity": "ERROR"},
                ]
            },
        ]
        
        summary = aggregate_results(results)
        
        self.assertEqual(summary["findings_by_category"]["log_file"], 2)
        self.assertEqual(summary["findings_by_category"]["cycle_count"], 1)
        self.assertEqual(summary["findings_by_category"]["hash_mismatch"], 1)


class TestMultiAuditExecution(unittest.TestCase):
    """Test end-to-end multi-audit execution."""
    
    def setUp(self):
        """Create temporary directory with test experiments."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_valid_experiment(self, name: str, cycles: int = 5):
        """Helper to create a valid experiment."""
        exp_dir = self.root / name
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": cycles,
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Create matching log file
        log_path = exp_dir / f"uplift_u2_test_slice_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(cycles):
                f.write(json.dumps({"cycle": i}) + "\n")
        
        return exp_dir
    
    def _create_invalid_experiment(self, name: str):
        """Helper to create an invalid experiment (empty log)."""
        exp_dir = self.root / name
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 5,
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Create empty log file
        log_path = exp_dir / f"uplift_u2_test_slice_baseline.jsonl"
        log_path.touch()
        
        return exp_dir
    
    def test_audit_all_valid_experiments(self):
        """Test auditing multiple valid experiments."""
        self._create_valid_experiment("exp1", cycles=3)
        self._create_valid_experiment("exp2", cycles=5)
        
        results, summary = audit_all_experiments(self.root, quiet=True)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["missing"], 0)
    
    def test_audit_mixed_valid_invalid(self):
        """Test auditing mixture of valid and invalid experiments."""
        self._create_valid_experiment("exp_good")
        self._create_invalid_experiment("exp_bad")
        
        results, summary = audit_all_experiments(self.root, quiet=True)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
    
    def test_audit_empty_directory(self):
        """Test auditing directory with no experiments."""
        results, summary = audit_all_experiments(self.root, quiet=True)
        
        self.assertEqual(len(results), 0)
        self.assertEqual(summary["total_experiments"], 0)


class TestReportFormats(unittest.TestCase):
    """Test report formatting."""
    
    def test_json_report_structure(self):
        """Test JSON report has correct structure."""
        root = Path("/tmp/test_root")
        results = [
            {"experiment_dir": "/tmp/test_root/exp1", "status": "PASS", "findings": []},
        ]
        summary = {"total_experiments": 1, "passed": 1, "failed": 0, "missing": 0}
        
        json_report = format_multi_audit_json(root, results, summary)
        
        self.assertIn("root_dir", json_report)
        self.assertIn("summary", json_report)
        self.assertIn("experiments", json_report)
        self.assertEqual(len(json_report["experiments"]), 1)
    
    def test_markdown_report_contains_summary(self):
        """Test Markdown report includes summary section."""
        root = Path("/tmp/test_root")
        results = [
            {"experiment_dir": "/tmp/test_root/exp1", "status": "PASS", "findings": []},
        ]
        summary = {
            "total_experiments": 1,
            "passed": 1,
            "failed": 0,
            "missing": 0,
            "findings_by_category": {}
        }
        
        md_report = format_multi_audit_markdown(root, results, summary)
        
        self.assertIn("## Summary", md_report)
        self.assertIn("Total Experiments", md_report)
        self.assertIn("Passed", md_report)
        self.assertIn("## Experiment Details", md_report)
    
    def test_markdown_report_shows_failures(self):
        """Test Markdown report highlights failures."""
        root = Path("/tmp/test_root")
        results = [
            {
                "experiment_dir": "/tmp/test_root/exp1",
                "status": "FAIL",
                "findings": [{"severity": "ERROR", "category": "test", "message": "Test error"}]
            },
        ]
        summary = {
            "total_experiments": 1,
            "passed": 0,
            "failed": 1,
            "missing": 0,
            "findings_by_category": {"test": 1}
        }
        
        md_report = format_multi_audit_markdown(root, results, summary)
        
        self.assertIn("FAIL", md_report)
        self.assertIn("❌", md_report)
        self.assertIn("Test error", md_report)


class TestExitCodeSemantics(unittest.TestCase):
    """Test exit code behavior via subprocess."""
    
    def setUp(self):
        """Create temporary directory with test experiments."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_valid_experiment(self, name: str):
        """Helper to create a valid experiment."""
        exp_dir = self.root / name
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 2,
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        log_path = exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            f.write('{"cycle": 0}\n{"cycle": 1}\n')
    
    def _create_missing_manifest_experiment(self, name: str):
        """Helper to create experiment with missing manifest."""
        exp_dir = self.root / name
        exp_dir.mkdir()
        # No manifest file
    
    def test_exit_code_0_all_pass(self):
        """Test exit code 0 when all experiments pass."""
        self._create_valid_experiment("exp1")
        self._create_valid_experiment("exp2")
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_uplift_u2_all.py"
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["python", str(script_path), str(self.root), "--quiet"],
            capture_output=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        self.assertEqual(result.returncode, 0)
    
    def test_exit_code_2_missing_artifacts(self):
        """Test exit code 2 when experiments have missing artifacts."""
        self._create_valid_experiment("exp_good")
        self._create_missing_manifest_experiment("exp_missing")
        
        # The discovery will only find exp_good since exp_missing has no manifest
        # So we need to create one with manifest but missing log
        exp_bad = self.root / "exp_bad"
        exp_bad.mkdir()
        with open(exp_bad / "manifest.json", 'w') as f:
            json.dump({"slice": "test", "mode": "baseline", "cycles": 5}, f)
        # No log file created
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_uplift_u2_all.py"
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["python", str(script_path), str(self.root), "--quiet"],
            capture_output=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        self.assertEqual(result.returncode, 2)
    
    def test_exit_code_1_failures(self):
        """Test exit code 1 when experiments fail integrity checks."""
        # Create experiment with cycle count mismatch
        exp_dir = self.root / "exp_fail"
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 10,  # Declare 10 cycles
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # But only write 3 cycles
        log_path = exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(3):
                f.write(f'{{"cycle": {i}}}\n')
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_uplift_u2_all.py"
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["python", str(script_path), str(self.root), "--quiet"],
            capture_output=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
    unittest.main()
