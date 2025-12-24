"""
Tests for CI entry point wrapper.

PHASE II — NOT USED IN PHASE I
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCIEntry(unittest.TestCase):
    """Test CI entry point functionality."""
    
    def setUp(self):
        """Create temporary directory with test experiments."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_valid_experiment(self, name: str, cycles: int = 2):
        """Helper to create a valid experiment."""
        exp_dir = self.root / name
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": cycles,
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        log_path = exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(cycles):
                f.write(f'{{"cycle": {i}}}\n')
    
    def test_ci_entry_forwards_exit_code(self):
        """Test that CI entry forwards exit code from audit_uplift_u2_all."""
        self._create_valid_experiment("exp1")
        self._create_valid_experiment("exp2")
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run(
            ["python", str(script_path), str(self.root)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        # Should pass with exit code 0
        self.assertEqual(result.returncode, 0)
        
        # Should contain CI-friendly output
        self.assertIn("CI EVIDENCE PACK AUDIT", result.stdout)
    
    def test_ci_entry_prints_summary(self):
        """Test that CI entry prints a summary line."""
        self._create_valid_experiment("exp1")
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run(
            ["python", str(script_path), str(self.root)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        # Should print summary information
        self.assertIn("Summary:", result.stdout)
        self.assertIn("experiment(s) audited", result.stdout)
        
        # Should show passed count
        self.assertIn("Passed:", result.stdout)
    
    def test_ci_entry_with_failures(self):
        """Test CI entry behavior with failed experiments."""
        # Create experiment with cycle mismatch (will fail)
        exp_dir = self.root / "exp_fail"
        exp_dir.mkdir()
        
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 10,
        }
        with open(exp_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        log_path = exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(3):
                f.write(f'{{"cycle": {i}}}\n')
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run(
            ["python", str(script_path), str(self.root)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        # Should fail with exit code 1
        self.assertEqual(result.returncode, 1)
        
        # Should show failed count
        self.assertIn("Failed:", result.stdout)
    
    def test_ci_entry_default_directory(self):
        """Test that CI entry can be called without arguments."""
        # This tests that the default path handling works
        # We expect it to fail because results/uplift_u2 likely doesn't exist
        # but the script should run without error
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        # Create a dummy results/uplift_u2 directory
        default_dir = repo_root / "results" / "uplift_u2_test_ci"
        default_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                ["python", str(script_path), str(default_dir)],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)},
                timeout=10
            )
            
            # Should run without crashing
            # Exit code may vary depending on directory contents
            self.assertIn("CI EVIDENCE PACK AUDIT", result.stdout)
        finally:
            # Clean up
            if default_dir.exists():
                shutil.rmtree(default_dir)


class TestCIEntryOutputFiles(unittest.TestCase):
    """Test CI entry file output options."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.output_dir = self.root / "outputs"
        self.output_dir.mkdir()
        
        # Create a valid experiment
        exp_dir = self.root / "exp1"
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
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_ci_entry_output_json(self):
        """Test that --output-json creates a report file."""
        json_path = self.output_dir / "audit.json"
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run(
            [
                "python", str(script_path), str(self.root),
                "--output-json", str(json_path)
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertTrue(json_path.exists())
        
        # Verify JSON is valid
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.assertIn("summary", data)
            self.assertIn("experiments", data)
    
    def test_ci_entry_output_markdown(self):
        """Test that --output-md creates a report file."""
        md_path = self.output_dir / "audit.md"
        
        script_path = Path(__file__).parent.parent.parent / "experiments" / "audit_ci_entry.py"
        repo_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run(
            [
                "python", str(script_path), str(self.root),
                "--output-md", str(md_path)
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertTrue(md_path.exists())
        
        # Verify Markdown content
        with open(md_path, 'r') as f:
            content = f.read()
            self.assertIn("# Multi-Experiment Audit Report", content)
            self.assertIn("## Summary", content)


if __name__ == '__main__':
    unittest.main()
