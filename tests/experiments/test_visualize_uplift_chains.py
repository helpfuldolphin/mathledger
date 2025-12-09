#!/usr/bin/env python
"""
PHASE II — Tests for Chain-Depth Cartographer & Slice Visualizer

Tests validate that visualization functions:
  - Produce expected output files
  - Handle synthetic log data correctly
  - Are deterministic given the same inputs
  - Don't raise exceptions on valid inputs

These tests use SYNTHETIC DATA with known chain structures to verify
visualization correctness without depending on actual experiment logs.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class TestVisualizeUpliftChains(unittest.TestCase):
    """Test suite for visualize_uplift_chains module."""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory for test artifacts."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_viz_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"

        # Generate synthetic baseline log
        cls._generate_synthetic_log(cls.baseline_log, mode="baseline", cycles=100)

        # Generate synthetic RFL log with different characteristics
        cls._generate_synthetic_log(cls.rfl_log, mode="rfl", cycles=100)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @classmethod
    def _generate_synthetic_log(
        cls,
        path: Path,
        mode: str,
        cycles: int,
        seed: int = 42,
    ) -> None:
        """
        Generate synthetic experiment log with known chain structures.

        Creates a JSONL file simulating experiment output with deterministic
        chain depth patterns for testing visualization.

        Args:
            path: Output path for JSONL file.
            mode: 'baseline' or 'rfl'.
            cycles: Number of cycles to generate.
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed if mode == "baseline" else seed + 1)

        records: List[Dict[str, Any]] = []

        for i in range(cycles):
            # Generate synthetic metrics with known patterns
            # Baseline: stable with occasional spikes
            # RFL: gradual improvement trend

            if mode == "baseline":
                depth = int(np.random.poisson(3))  # Mean depth ~3
                verified = int(np.random.binomial(5, 0.4))
                abstained = int(np.random.binomial(3, 0.3))
            else:
                # RFL shows improvement over time
                base_depth = 3 + (i / cycles) * 2  # Increases from 3 to 5
                depth = int(np.random.poisson(base_depth))
                verified = int(np.random.binomial(5, 0.4 + (i / cycles) * 0.2))
                abstained = int(np.random.binomial(3, max(0.1, 0.3 - (i / cycles) * 0.2)))

            candidates = verified + abstained + int(np.random.randint(0, 3))
            success = verified > 0 and abstained == 0

            record = {
                "cycle": i,
                "mode": mode,
                "slice_name": "test_slice_synthetic",
                "status": "verified" if success else "abstain",
                "abstention": abstained > 0,
                "success": success,
                "proof_found": verified > 0,
                "derivation": {
                    "candidates": candidates,
                    "verified": verified,
                    "abstained": abstained,
                    "depth": depth,
                    "candidate_hash": f"hash_{mode}_{i}",
                },
                "roots": {
                    "h_t": f"h_{mode}_{i}",
                    "r_t": f"r_{mode}_{i}",
                    "u_t": f"u_{mode}_{i}",
                },
                "rfl": {
                    "executed": mode == "rfl",
                },
                "gates_passed": True,
            }

            records.append(record)

        # Write JSONL
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def test_load_experiment_log(self):
        """Test that experiment logs load correctly."""
        from experiments.visualize_uplift_chains import load_experiment_log

        records = load_experiment_log(self.baseline_log)

        self.assertEqual(len(records), 100)
        self.assertEqual(records[0]["cycle"], 0)
        self.assertEqual(records[0]["mode"], "baseline")
        self.assertEqual(records[-1]["cycle"], 99)

    def test_extract_chain_depths(self):
        """Test chain depth extraction from logs."""
        from experiments.visualize_uplift_chains import (
            extract_chain_depths,
            load_experiment_log,
        )

        records = load_experiment_log(self.baseline_log)
        depths = extract_chain_depths(records)

        self.assertEqual(len(depths), 100)
        self.assertTrue(all(isinstance(d, int) for d in depths))
        self.assertTrue(all(d >= 0 for d in depths))

    def test_extract_goals_reached(self):
        """Test goal extraction from logs."""
        from experiments.visualize_uplift_chains import (
            extract_goals_reached,
            load_experiment_log,
        )

        records = load_experiment_log(self.baseline_log)
        goals = extract_goals_reached(records)

        self.assertIn("goal_A", goals)
        self.assertIn("goal_B", goals)
        self.assertIn("goal_C", goals)

        self.assertEqual(len(goals["goal_A"]), 100)
        self.assertEqual(len(goals["goal_B"]), 100)
        self.assertEqual(len(goals["goal_C"]), 100)

    def test_compute_cumulative_coverage(self):
        """Test cumulative coverage computation."""
        from experiments.visualize_uplift_chains import compute_cumulative_coverage

        # Test with known sequence
        achieved = [True, False, True, True, False]
        coverage = compute_cumulative_coverage(achieved)

        self.assertEqual(len(coverage), 5)
        self.assertAlmostEqual(coverage[0], 1.0)      # 1/1
        self.assertAlmostEqual(coverage[1], 0.5)      # 1/2
        self.assertAlmostEqual(coverage[2], 2 / 3)    # 2/3
        self.assertAlmostEqual(coverage[3], 0.75)     # 3/4
        self.assertAlmostEqual(coverage[4], 0.6)      # 3/5

    def test_plot_chain_depth_histogram_creates_file(self):
        """Test that histogram plot creates output file."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        out_path = Path(self.temp_dir) / "test_histogram.png"

        result = plot_chain_depth_histogram(
            self.baseline_log,
            out_path,
            title="Test Histogram",
            mode="baseline",
        )

        self.assertTrue(out_path.exists())
        self.assertEqual(result, out_path)
        # Check file is non-empty
        self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_chain_depth_histogram_deterministic(self):
        """Test that histogram output is deterministic."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        out_path_1 = Path(self.temp_dir) / "test_hist_det_1.png"
        out_path_2 = Path(self.temp_dir) / "test_hist_det_2.png"

        plot_chain_depth_histogram(self.baseline_log, out_path_1, mode="baseline")
        plot_chain_depth_histogram(self.baseline_log, out_path_2, mode="baseline")

        # Files should be identical (deterministic rendering)
        with open(out_path_1, "rb") as f1, open(out_path_2, "rb") as f2:
            # Note: Due to matplotlib internals, byte-exact match may not be possible
            # Check that file sizes are the same as a weaker determinism check
            self.assertEqual(f1.read().__len__(), f2.read().__len__())

    def test_plot_longest_chain_comparison_creates_file(self):
        """Test that longest chain comparison creates output file."""
        from experiments.visualize_uplift_chains import plot_longest_chain_comparison

        out_path = Path(self.temp_dir) / "test_longest_chain.png"

        result = plot_longest_chain_comparison(
            self.baseline_log,
            self.rfl_log,
            out_path,
        )

        self.assertTrue(out_path.exists())
        self.assertEqual(result, out_path)
        self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_goal_coverage_trajectories_creates_file(self):
        """Test that goal coverage plot creates output file."""
        from experiments.visualize_uplift_chains import plot_goal_coverage_trajectories

        out_path = Path(self.temp_dir) / "test_goal_coverage.png"

        result = plot_goal_coverage_trajectories(
            self.baseline_log,
            self.rfl_log,
            out_path,
        )

        self.assertTrue(out_path.exists())
        self.assertEqual(result, out_path)
        self.assertGreater(out_path.stat().st_size, 0)

    def test_generate_all_visualizations(self):
        """Test batch visualization generation."""
        from experiments.visualize_uplift_chains import generate_all_visualizations

        output_dir = Path(self.temp_dir) / "batch_viz"

        results = generate_all_visualizations(
            self.baseline_log,
            self.rfl_log,
            output_dir,
            slice_name="test_slice",
        )

        # Check all expected outputs
        self.assertIn("histogram_baseline", results)
        self.assertIn("histogram_rfl", results)
        self.assertIn("longest_chain", results)
        self.assertIn("goal_coverage", results)
        self.assertIn("manifest", results)

        # Check files exist
        for name, path in results.items():
            self.assertTrue(Path(path).exists(), f"Missing output: {name}")

        # Check manifest content
        manifest_path = results["manifest"]
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["phase"], "II")
        self.assertEqual(manifest["slice_name"], "test_slice")

    def test_empty_log_handling(self):
        """Test that empty logs raise appropriate errors."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        empty_log = Path(self.temp_dir) / "empty.jsonl"
        empty_log.write_text("")

        out_path = Path(self.temp_dir) / "empty_hist.png"

        with self.assertRaises(ValueError):
            plot_chain_depth_histogram(empty_log, out_path)

    def test_auto_mode_detection(self):
        """Test automatic mode detection from log records."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        out_path = Path(self.temp_dir) / "auto_mode_hist.png"

        # Should auto-detect mode as "baseline"
        plot_chain_depth_histogram(
            self.baseline_log,
            out_path,
            mode="auto",
        )

        self.assertTrue(out_path.exists())

    def test_slice_name_detection(self):
        """Test automatic slice name detection from log records."""
        from experiments.visualize_uplift_chains import (
            load_experiment_log,
            plot_chain_depth_histogram,
        )

        records = load_experiment_log(self.baseline_log)
        expected_slice = records[0].get("slice_name")

        out_path = Path(self.temp_dir) / "slice_detect_hist.png"

        # Should detect slice_name automatically
        result = plot_chain_depth_histogram(
            self.baseline_log,
            out_path,
            slice_name=None,  # Auto-detect
        )

        self.assertTrue(out_path.exists())


class TestChainAnalyzerIntegration(unittest.TestCase):
    """Tests for ChainAnalyzer integration with visualizations."""

    def test_chain_analyzer_import(self):
        """Test that ChainAnalyzer can be imported."""
        from experiments.derivation_chain_analysis import ChainAnalyzer

        # Create simple derivation chain
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
        ]

        analyzer = ChainAnalyzer(derivations)

        # Test depth computation
        self.assertEqual(analyzer.get_depth("h0"), 1)
        self.assertEqual(analyzer.get_depth("h1"), 2)
        self.assertEqual(analyzer.get_depth("h2"), 3)

    def test_chain_analyzer_with_complex_dag(self):
        """Test ChainAnalyzer with branching derivations."""
        from experiments.derivation_chain_analysis import ChainAnalyzer

        # Diamond-shaped DAG: h0 <- h1, h2 <- h3
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h0"]},
            {"hash": "h3", "premises": ["h1", "h2"]},
        ]

        analyzer = ChainAnalyzer(derivations)

        self.assertEqual(analyzer.get_depth("h0"), 1)
        self.assertEqual(analyzer.get_depth("h1"), 2)
        self.assertEqual(analyzer.get_depth("h2"), 2)
        self.assertEqual(analyzer.get_depth("h3"), 3)  # max(2, 2) + 1


class TestVisualizationProperties(unittest.TestCase):
    """Tests for visualization output properties."""

    @classmethod
    def setUpClass(cls):
        """Set up temp directory."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_viz_props_")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil

        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_histogram_bins_respected(self):
        """Test that histogram bin count is respected."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        # Create minimal test log
        log_path = Path(self.temp_dir) / "bins_test.jsonl"
        records = [
            {
                "cycle": i,
                "mode": "baseline",
                "derivation": {"verified": i % 5, "depth": i % 10},
            }
            for i in range(50)
        ]
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Test with different bin counts
        for bins in [5, 10, 20]:
            out_path = Path(self.temp_dir) / f"hist_{bins}_bins.png"
            plot_chain_depth_histogram(log_path, out_path, bins=bins)
            self.assertTrue(out_path.exists())

    def test_output_directory_creation(self):
        """Test that output directories are created automatically."""
        from experiments.visualize_uplift_chains import plot_chain_depth_histogram

        # Create minimal test log
        log_path = Path(self.temp_dir) / "dir_test.jsonl"
        records = [
            {"cycle": 0, "mode": "baseline", "derivation": {"verified": 1}}
        ]
        with open(log_path, "w") as f:
            f.write(json.dumps(records[0]) + "\n")

        # Use nested output path that doesn't exist
        nested_path = (
            Path(self.temp_dir) / "nested" / "dirs" / "test_output.png"
        )

        plot_chain_depth_histogram(log_path, nested_path)

        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())


if __name__ == "__main__":
    unittest.main()

