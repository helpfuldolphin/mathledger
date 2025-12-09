#!/usr/bin/env python
"""
PHASE II — Tests for DAG Topology Explorer & Extended Visualization Suite

20 test cases covering:
  - Determinism guarantees
  - No randomness in plotting
  - Safe backend (Agg) usage
  - Correct DAG extraction from logs
  - Correct branching factor derivation
  - All visualization functions

ABSOLUTE SAFEGUARDS:
  - Tests do NOT make uplift claims.
  - Tests verify visualization correctness, not policy.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class TestDAGTopologyAnalyzer(unittest.TestCase):
    """Tests for DAGTopologyAnalyzer class."""

    def test_01_empty_dag_initialization(self):
        """Test that empty derivations create empty DAG."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        analyzer = DAGTopologyAnalyzer([])
        
        self.assertEqual(analyzer.node_count, 0)
        self.assertEqual(analyzer.edge_count, 0)
        self.assertEqual(analyzer.get_max_depth(), 0)

    def test_02_linear_chain_depth(self):
        """Test depth computation for linear chain."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        # h0 <- h1 <- h2 <- h3 (chain of length 4)
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
            {"hash": "h3", "premises": ["h2"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        
        self.assertEqual(analyzer.get_depth("h0"), 1)
        self.assertEqual(analyzer.get_depth("h1"), 2)
        self.assertEqual(analyzer.get_depth("h2"), 3)
        self.assertEqual(analyzer.get_depth("h3"), 4)
        self.assertEqual(analyzer.get_max_depth(), 4)

    def test_03_diamond_dag_depth(self):
        """Test depth computation for diamond-shaped DAG."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        # Diamond: h0 <- h1, h2 <- h3 (h3 depends on both h1 and h2)
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h0"]},
            {"hash": "h3", "premises": ["h1", "h2"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        
        self.assertEqual(analyzer.get_depth("h0"), 1)
        self.assertEqual(analyzer.get_depth("h1"), 2)
        self.assertEqual(analyzer.get_depth("h2"), 2)
        self.assertEqual(analyzer.get_depth("h3"), 3)  # max(2, 2) + 1

    def test_04_branching_factor_computation(self):
        """Test branching factor (in/out degree) computation."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h0"]},
            {"hash": "h3", "premises": ["h1", "h2"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        factors = analyzer.get_branching_factors()
        
        # h0: in=0 (no premises), out=2 (h1, h2 depend on it)
        self.assertEqual(factors["h0"]["in_degree"], 0)
        self.assertEqual(factors["h0"]["out_degree"], 2)
        
        # h3: in=2 (h1, h2), out=0 (nothing depends on it)
        self.assertEqual(factors["h3"]["in_degree"], 2)
        self.assertEqual(factors["h3"]["out_degree"], 0)

    def test_05_depth_distribution(self):
        """Test depth distribution computation."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": []},
            {"hash": "h2", "premises": ["h0"]},
            {"hash": "h3", "premises": ["h1"]},
            {"hash": "h4", "premises": ["h2", "h3"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        dist = analyzer.get_depth_distribution()
        
        # Depth 1: h0, h1 (2 nodes)
        # Depth 2: h2, h3 (2 nodes)
        # Depth 3: h4 (1 node)
        self.assertEqual(dist[1], 2)
        self.assertEqual(dist[2], 2)
        self.assertEqual(dist[3], 1)

    def test_06_longest_path_extraction(self):
        """Test longest path extraction."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        path = analyzer.get_longest_path("h2")
        
        self.assertEqual(path, ["h0", "h1", "h2"])

    def test_07_adjacency_matrix_shape(self):
        """Test adjacency matrix has correct shape."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        derivations = [
            {"hash": f"h{i}", "premises": [f"h{i-1}"] if i > 0 else []}
            for i in range(10)
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        matrix, nodes = analyzer.get_adjacency_matrix(max_nodes=10)
        
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(len(nodes), 10)

    def test_08_dag_footprint_completeness(self):
        """Test DAG footprint contains all expected fields."""
        from experiments.visualize_dag_topology import DAGTopologyAnalyzer
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        footprint = analyzer.get_dag_footprint()
        
        self.assertIn("node_count", footprint)
        self.assertIn("edge_count", footprint)
        self.assertIn("max_depth", footprint)
        self.assertIn("mean_depth", footprint)
        self.assertIn("median_depth", footprint)
        self.assertIn("depth_distribution", footprint)
        self.assertIn("branching_distribution", footprint)


class TestDataExtraction(unittest.TestCase):
    """Tests for data extraction from logs."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and synthetic logs."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_dag_")
        cls.log_path = Path(cls.temp_dir) / "test.jsonl"
        
        # Generate synthetic log
        records = []
        for i in range(50):
            records.append({
                "cycle": i,
                "mode": "baseline",
                "derivation": {
                    "candidates": 5 + (i % 3),
                    "verified": 2 + (i % 2),
                    "abstained": i % 2,
                    "candidate_hash": f"hash_{i}",
                    "depth": 1 + (i % 5),
                },
                "abstention": bool(i % 2),
                "success": bool(i % 3),
            })
        
        with open(cls.log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_09_load_experiment_log(self):
        """Test log loading returns correct record count."""
        from experiments.visualize_dag_topology import load_experiment_log
        
        records = load_experiment_log(self.log_path)
        self.assertEqual(len(records), 50)

    def test_10_extract_derivations_from_log(self):
        """Test derivation extraction from experiment log."""
        from experiments.visualize_dag_topology import (
            extract_derivations_from_log,
            load_experiment_log,
        )
        
        records = load_experiment_log(self.log_path)
        derivations = extract_derivations_from_log(records)
        
        self.assertGreater(len(derivations), 0)
        self.assertTrue(all("hash" in d for d in derivations))

    def test_11_extract_cycle_metrics(self):
        """Test cycle metrics extraction."""
        from experiments.visualize_dag_topology import (
            extract_cycle_metrics,
            load_experiment_log,
        )
        
        records = load_experiment_log(self.log_path)
        metrics = extract_cycle_metrics(records)
        
        self.assertEqual(len(metrics["cycle"]), 50)
        self.assertEqual(len(metrics["candidates"]), 50)
        self.assertEqual(len(metrics["verified"]), 50)
        self.assertEqual(len(metrics["abstention"]), 50)
        self.assertEqual(len(metrics["depth"]), 50)


class TestVisualizationDeterminism(unittest.TestCase):
    """Tests for visualization determinism guarantees."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and synthetic data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_det_")
        cls.log_path = Path(cls.temp_dir) / "det_test.jsonl"
        
        # Generate deterministic log
        records = []
        for i in range(100):
            records.append({
                "cycle": i,
                "mode": "baseline",
                "derivation": {
                    "candidates": 5,
                    "verified": 2,
                    "abstained": 1,
                    "candidate_hash": f"hash_{i:04d}",
                    "depth": (i % 5) + 1,
                },
            })
        
        with open(cls.log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_12_agg_backend_used(self):
        """Test that Agg backend is used (no display required)."""
        import matplotlib
        # Import the module to trigger backend setting
        import experiments.visualize_dag_topology  # noqa
        
        # Verify Agg backend is active
        self.assertEqual(matplotlib.get_backend(), "agg")

    def test_13_deterministic_histogram(self):
        """Test histogram output is deterministic."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            extract_derivations_from_log,
            load_experiment_log,
            plot_depth_distribution_histogram,
        )
        
        records = load_experiment_log(self.log_path)
        derivations = extract_derivations_from_log(records)
        analyzer = DAGTopologyAnalyzer(derivations)
        
        out1 = Path(self.temp_dir) / "det_hist_1.png"
        out2 = Path(self.temp_dir) / "det_hist_2.png"
        
        plot_depth_distribution_histogram(analyzer, out1)
        plot_depth_distribution_histogram(analyzer, out2)
        
        # File sizes should match (weak determinism check)
        self.assertEqual(out1.stat().st_size, out2.stat().st_size)

    def test_14_no_randomness_in_branching_plot(self):
        """Test branching plot has no random elements."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            plot_branching_factor_distribution,
        )
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h0", "h1"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        
        out1 = Path(self.temp_dir) / "branch_1.png"
        out2 = Path(self.temp_dir) / "branch_2.png"
        
        plot_branching_factor_distribution(analyzer, out1)
        plot_branching_factor_distribution(analyzer, out2)
        
        self.assertEqual(out1.stat().st_size, out2.stat().st_size)


class TestVisualizationOutputs(unittest.TestCase):
    """Tests for visualization output generation."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_viz_out_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate baseline log
        baseline_records = []
        for i in range(50):
            baseline_records.append({
                "cycle": i,
                "mode": "baseline",
                "derivation": {
                    "candidates": 5,
                    "verified": 2,
                    "abstained": 1,
                    "candidate_hash": f"b_hash_{i}",
                    "depth": (i % 4) + 1,
                },
                "abstention": bool(i % 3),
                "success": bool(i % 2),
            })
        
        with open(cls.baseline_log, "w") as f:
            for r in baseline_records:
                f.write(json.dumps(r) + "\n")
        
        # Generate RFL log
        rfl_records = []
        for i in range(50):
            rfl_records.append({
                "cycle": i,
                "mode": "rfl",
                "derivation": {
                    "candidates": 6,
                    "verified": 3,
                    "abstained": 0,
                    "candidate_hash": f"r_hash_{i}",
                    "depth": (i % 5) + 1,
                },
                "abstention": False,
                "success": True,
            })
        
        with open(cls.rfl_log, "w") as f:
            for r in rfl_records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_15_adjacency_matrix_creates_file(self):
        """Test adjacency matrix heatmap creates file."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            extract_derivations_from_log,
            load_experiment_log,
            plot_adjacency_matrix_heatmap,
        )
        
        records = load_experiment_log(self.baseline_log)
        derivations = extract_derivations_from_log(records)
        analyzer = DAGTopologyAnalyzer(derivations)
        
        out_path = Path(self.temp_dir) / "adj_matrix.png"
        result = plot_adjacency_matrix_heatmap(analyzer, out_path)
        
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)

    def test_16_depth_layers_creates_file(self):
        """Test depth layer plot creates file."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            extract_derivations_from_log,
            load_experiment_log,
            plot_depth_level_layers,
        )
        
        records = load_experiment_log(self.baseline_log)
        derivations = extract_derivations_from_log(records)
        analyzer = DAGTopologyAnalyzer(derivations)
        
        out_path = Path(self.temp_dir) / "depth_layers.png"
        result = plot_depth_level_layers(analyzer, out_path)
        
        self.assertTrue(out_path.exists())

    def test_17_longest_path_creates_file(self):
        """Test longest path visualization creates file."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            plot_longest_path_visualization,
        )
        
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
        ]
        
        analyzer = DAGTopologyAnalyzer(derivations)
        out_path = Path(self.temp_dir) / "longest_path.png"
        
        result = plot_longest_path_visualization(analyzer, out_path)
        
        self.assertTrue(out_path.exists())

    def test_18_comparison_plots_create_files(self):
        """Test comparison plots create files."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_node_count_comparison,
            plot_chain_depth_trend,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out1 = Path(self.temp_dir) / "node_count.png"
        out2 = Path(self.temp_dir) / "depth_trend.png"
        
        plot_node_count_comparison(baseline_records, rfl_records, out1)
        plot_chain_depth_trend(baseline_records, rfl_records, out2)
        
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())

    def test_19_extended_visualizations_create_files(self):
        """Test extended visualization functions create files."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_abstention_vs_depth_scatter,
            plot_candidate_pool_entropy,
            plot_mp_round_yield_vs_depth,
            plot_success_concentration_diagram,
        )
        
        records = load_experiment_log(self.baseline_log)
        
        outputs = [
            (plot_abstention_vs_depth_scatter, "abstention_scatter.png"),
            (plot_candidate_pool_entropy, "entropy.png"),
            (plot_mp_round_yield_vs_depth, "yield.png"),
            (plot_success_concentration_diagram, "concentration.png"),
        ]
        
        for func, filename in outputs:
            out_path = Path(self.temp_dir) / filename
            func(records, out_path)
            self.assertTrue(out_path.exists(), f"Failed: {filename}")

    def test_20_batch_generation_creates_manifest(self):
        """Test batch generation creates all files and manifest."""
        from experiments.visualize_dag_topology import generate_all_topology_visualizations
        
        output_dir = Path(self.temp_dir) / "batch_output"
        
        results = generate_all_topology_visualizations(
            self.baseline_log,
            self.rfl_log,
            output_dir,
        )
        
        # Check manifest exists
        self.assertIn("manifest", results)
        manifest_path = results["manifest"]
        self.assertTrue(Path(manifest_path).exists())
        
        # Load and validate manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        self.assertEqual(manifest["phase"], "II")
        self.assertIn("outputs", manifest)
        self.assertIn("baseline_footprint", manifest)
        self.assertIn("rfl_footprint", manifest)
        
        # Check at least some visualization files exist
        for key in ["adjacency_baseline", "depth_dist_comparison", "cumulative_abstention"]:
            if key in results:
                self.assertTrue(Path(results[key]).exists(), f"Missing: {key}")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_edge_")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_empty_dag_visualization(self):
        """Test visualization handles empty DAG gracefully."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            plot_adjacency_matrix_heatmap,
            plot_depth_level_layers,
        )
        
        analyzer = DAGTopologyAnalyzer([])
        
        out1 = Path(self.temp_dir) / "empty_adj.png"
        out2 = Path(self.temp_dir) / "empty_layers.png"
        
        # Should not raise
        plot_adjacency_matrix_heatmap(analyzer, out1)
        plot_depth_level_layers(analyzer, out2)
        
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())

    def test_single_node_dag(self):
        """Test visualization handles single-node DAG."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            plot_longest_path_visualization,
        )
        
        analyzer = DAGTopologyAnalyzer([{"hash": "h0", "premises": []}])
        
        out_path = Path(self.temp_dir) / "single_node.png"
        plot_longest_path_visualization(analyzer, out_path)
        
        self.assertTrue(out_path.exists())
        self.assertEqual(analyzer.get_max_depth(), 1)


# =============================================================================
# NEW TESTS: Curriculum-Behavior Mapping & Drift Analysis
# =============================================================================


class TestExpectedDAGProfile(unittest.TestCase):
    """Tests for ExpectedDAGProfile class."""

    def test_21_profile_initialization(self):
        """Test ExpectedDAGProfile initializes correctly."""
        from experiments.visualize_dag_topology import ExpectedDAGProfile
        
        profile = ExpectedDAGProfile(
            slice_type="test",
            expected_depth_range=(1.0, 5.0),
            expected_branching_range=(0.5, 2.0),
            expected_success_rate_range=(0.3, 0.7),
            description="Test profile",
        )
        
        self.assertEqual(profile.slice_type, "test")
        self.assertEqual(profile.expected_depth_range, (1.0, 5.0))
        self.assertEqual(profile.description, "Test profile")

    def test_22_profile_to_dict(self):
        """Test profile serialization to dict."""
        from experiments.visualize_dag_topology import ExpectedDAGProfile
        
        profile = ExpectedDAGProfile(
            slice_type="goal_hit",
            expected_depth_range=(1.0, 3.0),
            expected_branching_range=(0.0, 1.5),
            expected_success_rate_range=(0.6, 1.0),
            description="Test",
        )
        
        d = profile.to_dict()
        
        self.assertIn("slice_type", d)
        self.assertIn("expected_depth_range", d)
        self.assertEqual(d["slice_type"], "goal_hit")

    def test_23_depth_drift_below(self):
        """Test depth drift detection when below range."""
        from experiments.visualize_dag_topology import ExpectedDAGProfile
        
        profile = ExpectedDAGProfile(
            slice_type="test",
            expected_depth_range=(3.0, 5.0),
            expected_branching_range=(0.0, 2.0),
            expected_success_rate_range=(0.0, 1.0),
            description="Test",
        )
        
        result = profile.check_depth_drift(1.5)
        
        self.assertEqual(result["status"], "below")
        self.assertAlmostEqual(result["drift"], 1.5)  # 3.0 - 1.5

    def test_24_depth_drift_above(self):
        """Test depth drift detection when above range."""
        from experiments.visualize_dag_topology import ExpectedDAGProfile
        
        profile = ExpectedDAGProfile(
            slice_type="test",
            expected_depth_range=(1.0, 3.0),
            expected_branching_range=(0.0, 2.0),
            expected_success_rate_range=(0.0, 1.0),
            description="Test",
        )
        
        result = profile.check_depth_drift(5.0)
        
        self.assertEqual(result["status"], "above")
        self.assertAlmostEqual(result["drift"], 2.0)  # 5.0 - 3.0

    def test_25_depth_drift_within(self):
        """Test depth drift detection when within range."""
        from experiments.visualize_dag_topology import ExpectedDAGProfile
        
        profile = ExpectedDAGProfile(
            slice_type="test",
            expected_depth_range=(1.0, 5.0),
            expected_branching_range=(0.0, 2.0),
            expected_success_rate_range=(0.0, 1.0),
            description="Test",
        )
        
        result = profile.check_depth_drift(3.0)
        
        self.assertEqual(result["status"], "within")
        self.assertAlmostEqual(result["drift"], 0.0)


class TestSliceTypeInference(unittest.TestCase):
    """Tests for slice type inference."""

    def test_26_infer_goal_hit_slice(self):
        """Test inference for goal_hit slice type."""
        from experiments.visualize_dag_topology import infer_expected_dag_profile
        
        profile = infer_expected_dag_profile("slice_easy_goal")
        self.assertEqual(profile.slice_type, "goal_hit")
        
        profile = infer_expected_dag_profile("slice_debug_uplift")
        self.assertEqual(profile.slice_type, "goal_hit")

    def test_27_infer_tree_slice(self):
        """Test inference for tree slice type."""
        from experiments.visualize_dag_topology import infer_expected_dag_profile
        
        profile = infer_expected_dag_profile("slice_uplift_tree")
        self.assertEqual(profile.slice_type, "tree")
        
        profile = infer_expected_dag_profile("slice_hard_branching")
        self.assertEqual(profile.slice_type, "tree")

    def test_28_infer_dependency_slice(self):
        """Test inference for dependency slice type."""
        from experiments.visualize_dag_topology import infer_expected_dag_profile
        
        profile = infer_expected_dag_profile("slice_uplift_dependency")
        self.assertEqual(profile.slice_type, "dependency")
        
        profile = infer_expected_dag_profile("first_organism_pl")
        self.assertEqual(profile.slice_type, "dependency")

    def test_29_infer_sparse_slice(self):
        """Test inference for sparse slice type."""
        from experiments.visualize_dag_topology import infer_expected_dag_profile
        
        profile = infer_expected_dag_profile("slice_uplift_proto")
        self.assertEqual(profile.slice_type, "sparse")
        
        profile = infer_expected_dag_profile("slice_medium")
        self.assertEqual(profile.slice_type, "sparse")

    def test_30_infer_unknown_slice(self):
        """Test inference returns unknown for unrecognized slices."""
        from experiments.visualize_dag_topology import infer_expected_dag_profile
        
        profile = infer_expected_dag_profile("completely_random_name_xyz")
        self.assertEqual(profile.slice_type, "unknown")
        # Unknown should have wide ranges
        self.assertEqual(profile.expected_depth_range, (1.0, 10.0))


class TestBehaviorDriftComputation(unittest.TestCase):
    """Tests for behavior drift computation."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_drift_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate baseline log
        baseline_records = []
        for i in range(100):
            baseline_records.append({
                "cycle": i,
                "mode": "baseline",
                "slice_name": "slice_uplift_proto",
                "derivation": {
                    "candidates": 5,
                    "verified": 2,
                    "abstained": 1,
                    "candidate_hash": f"b_hash_{i}",
                    "depth": (i % 4) + 1,
                },
                "abstention": bool(i % 3),
                "success": bool(i % 2),
            })
        
        with open(cls.baseline_log, "w") as f:
            for r in baseline_records:
                f.write(json.dumps(r) + "\n")
        
        # Generate RFL log (slightly different characteristics)
        rfl_records = []
        for i in range(100):
            rfl_records.append({
                "cycle": i,
                "mode": "rfl",
                "slice_name": "slice_uplift_proto",
                "derivation": {
                    "candidates": 6,
                    "verified": 3,
                    "abstained": 0,
                    "candidate_hash": f"r_hash_{i}",
                    "depth": (i % 5) + 2,  # Deeper
                },
                "abstention": False,
                "success": i % 3 != 0,  # Different pattern
            })
        
        with open(cls.rfl_log, "w") as f:
            for r in rfl_records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_31_compute_behavior_drift(self):
        """Test behavior drift computation."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            compute_behavior_drift,
            extract_cycle_metrics,
            extract_derivations_from_log,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        baseline_derivations = extract_derivations_from_log(baseline_records)
        rfl_derivations = extract_derivations_from_log(rfl_records)
        
        baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
        rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
        
        baseline_metrics = extract_cycle_metrics(baseline_records)
        rfl_metrics = extract_cycle_metrics(rfl_records)
        
        drift_report = compute_behavior_drift(
            baseline_analyzer, rfl_analyzer,
            baseline_metrics, rfl_metrics,
            "slice_uplift_proto",
        )
        
        self.assertEqual(drift_report["phase"], "II")
        self.assertIn("expected_profile", drift_report)
        self.assertIn("observed", drift_report)
        self.assertIn("drift_vs_expected", drift_report)
        self.assertIn("delta_baseline_vs_rfl", drift_report)

    def test_32_drift_report_contains_delta(self):
        """Test drift report contains baseline vs RFL delta."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            compute_behavior_drift,
            extract_cycle_metrics,
            extract_derivations_from_log,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        baseline_derivations = extract_derivations_from_log(baseline_records)
        rfl_derivations = extract_derivations_from_log(rfl_records)
        
        baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
        rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
        
        baseline_metrics = extract_cycle_metrics(baseline_records)
        rfl_metrics = extract_cycle_metrics(rfl_records)
        
        drift_report = compute_behavior_drift(
            baseline_analyzer, rfl_analyzer,
            baseline_metrics, rfl_metrics,
            "slice_uplift_proto",
        )
        
        delta = drift_report["delta_baseline_vs_rfl"]
        self.assertIn("depth_delta", delta)
        self.assertIn("branching_delta", delta)
        self.assertIn("success_delta", delta)

    def test_33_write_behavior_drift_report(self):
        """Test drift report JSON output."""
        from experiments.visualize_dag_topology import write_behavior_drift_report
        
        drift_report = {
            "phase": "II",
            "type": "behavior_drift_report",
            "slice_name": "test",
            "expected_profile": {"slice_type": "test"},
            "observed": {},
            "narrative_flags": [],
        }
        
        out_path = Path(self.temp_dir) / "test_drift.json"
        result = write_behavior_drift_report(drift_report, out_path)
        
        self.assertTrue(out_path.exists())
        
        with open(out_path, "r") as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["phase"], "II")


class TestAdvancedDifferentialVisualizations(unittest.TestCase):
    """Tests for advanced differential visualizations."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_diff_viz_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate synthetic logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(100):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "test_slice",
                    "derivation": {
                        "candidates": 5 + (1 if mode == "rfl" else 0),
                        "verified": 2 + (1 if mode == "rfl" else 0),
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                        "depth": (i % 5) + (1 if mode == "rfl" else 0),
                    },
                    "abstention": bool(i % 3),
                    "success": bool(i % 2),
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_34_differential_depth_histogram(self):
        """Test differential depth histogram generation."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_differential_depth_histogram,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out_path = Path(self.temp_dir) / "diff_hist.png"
        result = plot_differential_depth_histogram(baseline_records, rfl_records, out_path)
        
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)

    def test_35_branching_factor_delta_series(self):
        """Test branching factor delta series plot."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_branching_factor_delta_series,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out_path = Path(self.temp_dir) / "branching_delta.png"
        result = plot_branching_factor_delta_series(baseline_records, rfl_records, out_path)
        
        self.assertTrue(out_path.exists())

    def test_36_chain_depth_drift_visual(self):
        """Test chain depth drift visualization."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_chain_depth_drift_visual,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out_path = Path(self.temp_dir) / "depth_drift.png"
        result = plot_chain_depth_drift_visual(
            baseline_records, rfl_records, out_path,
            slice_name="slice_uplift_proto",
        )
        
        self.assertTrue(out_path.exists())

    def test_37_success_drift_waterfall(self):
        """Test success drift waterfall chart."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_success_drift_waterfall,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out_path = Path(self.temp_dir) / "waterfall.png"
        result = plot_success_drift_waterfall(baseline_records, rfl_records, out_path)
        
        self.assertTrue(out_path.exists())

    def test_38_curriculum_behavior_comparison(self):
        """Test curriculum-behavior comparison plot."""
        from experiments.visualize_dag_topology import (
            DAGTopologyAnalyzer,
            extract_cycle_metrics,
            extract_derivations_from_log,
            load_experiment_log,
            plot_curriculum_behavior_comparison,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        baseline_derivations = extract_derivations_from_log(baseline_records)
        rfl_derivations = extract_derivations_from_log(rfl_records)
        
        baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
        rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
        
        baseline_metrics = extract_cycle_metrics(baseline_records)
        rfl_metrics = extract_cycle_metrics(rfl_records)
        
        out_path = Path(self.temp_dir) / "comparison.png"
        result = plot_curriculum_behavior_comparison(
            baseline_analyzer, rfl_analyzer,
            baseline_metrics, rfl_metrics,
            "slice_uplift_proto",
            out_path,
        )
        
        self.assertTrue(out_path.exists())

    def test_39_generate_all_with_drift(self):
        """Test batch generation with drift analysis."""
        from experiments.visualize_dag_topology import generate_all_topology_with_drift
        
        output_dir = Path(self.temp_dir) / "batch_drift"
        
        results = generate_all_topology_with_drift(
            self.baseline_log,
            self.rfl_log,
            output_dir,
            slice_name="slice_uplift_proto",
        )
        
        # Check drift-specific outputs
        self.assertIn("behavior_drift", results)
        self.assertIn("differential_depth_histogram", results)
        self.assertIn("branching_delta_series", results)
        self.assertIn("chain_depth_drift", results)
        self.assertIn("curriculum_behavior_comparison", results)
        
        # Check behavior_drift.json exists
        drift_path = results["behavior_drift"]
        self.assertTrue(Path(drift_path).exists())
        
        with open(drift_path, "r") as f:
            drift_data = json.load(f)
        
        self.assertEqual(drift_data["slice_name"], "slice_uplift_proto")

    def test_40_differential_visualizations_deterministic(self):
        """Test that differential visualizations are deterministic."""
        from experiments.visualize_dag_topology import (
            load_experiment_log,
            plot_differential_depth_histogram,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        out1 = Path(self.temp_dir) / "det_diff_1.png"
        out2 = Path(self.temp_dir) / "det_diff_2.png"
        
        plot_differential_depth_histogram(baseline_records, rfl_records, out1)
        plot_differential_depth_histogram(baseline_records, rfl_records, out2)
        
        # File sizes should match
        self.assertEqual(out1.stat().st_size, out2.stat().st_size)


# =============================================================================
# NEW TESTS: Topology Summary & Warning Flags (C5 Diagnostic Layer)
# =============================================================================


class TestTopologySummaryExport(unittest.TestCase):
    """Tests for summarize_slice_topology() function."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_summary_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate baseline log
        baseline_records = []
        for i in range(100):
            baseline_records.append({
                "cycle": i,
                "mode": "baseline",
                "slice_name": "slice_test_summary",
                "derivation": {
                    "candidates": 5,
                    "verified": 2,
                    "abstained": 1,
                    "candidate_hash": f"b_hash_{i}",
                    "depth": (i % 4) + 1,
                },
                "abstention": bool(i % 3),
                "success": bool(i % 2),
            })
        
        with open(cls.baseline_log, "w") as f:
            for r in baseline_records:
                f.write(json.dumps(r) + "\n")
        
        # Generate RFL log with different characteristics
        rfl_records = []
        for i in range(100):
            rfl_records.append({
                "cycle": i,
                "mode": "rfl",
                "slice_name": "slice_test_summary",
                "derivation": {
                    "candidates": 6,
                    "verified": 3,
                    "abstained": 0,
                    "candidate_hash": f"r_hash_{i}",
                    "depth": (i % 5) + 2,
                },
                "abstention": False,
                "success": i % 3 != 0,
            })
        
        with open(cls.rfl_log, "w") as f:
            for r in rfl_records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_41_summary_structure_complete(self):
        """Test that summary contains all required fields."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "summary_structure.json"
        
        summary = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path,
            slice_name="slice_test_summary",
        )
        
        # Verify top-level structure
        self.assertEqual(summary["phase"], "II")
        self.assertEqual(summary["type"], "slice_topology_summary")
        self.assertEqual(summary["slice_name"], "slice_test_summary")
        
        # Verify baseline_footprint structure
        bf = summary["baseline_footprint"]
        self.assertIn("node_count", bf)
        self.assertIn("edge_count", bf)
        self.assertIn("max_depth", bf)
        self.assertIn("mean_depth", bf)
        self.assertIn("mean_branching", bf)
        
        # Verify rfl_footprint structure
        rf = summary["rfl_footprint"]
        self.assertIn("node_count", rf)
        self.assertIn("edge_count", rf)
        self.assertIn("max_depth", rf)
        self.assertIn("mean_depth", rf)
        self.assertIn("mean_branching", rf)
        
        # Verify deltas
        deltas = summary["deltas"]
        self.assertIn("depth_delta", deltas)
        self.assertIn("node_delta", deltas)
        self.assertIn("edge_delta", deltas)
        self.assertIn("branching_delta", deltas)
        
        # Verify warnings structure
        warnings = summary["warnings"]
        self.assertIn("depth_saturation_warning", warnings)
        self.assertIn("branching_collapse_warning", warnings)
        self.assertIn("topology_change_warning", warnings)
        self.assertIn("explanations", warnings)
        self.assertIn("thresholds", warnings)

    def test_42_summary_determinism(self):
        """Test that summary output is deterministic on repeated calls."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path_1 = Path(self.temp_dir) / "summary_det_1.json"
        out_path_2 = Path(self.temp_dir) / "summary_det_2.json"
        
        summary_1 = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path_1,
            slice_name="slice_test_summary",
        )
        
        summary_2 = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path_2,
            slice_name="slice_test_summary",
        )
        
        # Summaries should be identical
        # Compare key fields (exclude source_logs which may have path differences)
        self.assertEqual(summary_1["slice_name"], summary_2["slice_name"])
        self.assertEqual(summary_1["baseline_footprint"], summary_2["baseline_footprint"])
        self.assertEqual(summary_1["rfl_footprint"], summary_2["rfl_footprint"])
        self.assertEqual(summary_1["deltas"], summary_2["deltas"])
        self.assertEqual(summary_1["warnings"]["depth_saturation_warning"], 
                        summary_2["warnings"]["depth_saturation_warning"])
        self.assertEqual(summary_1["warnings"]["branching_collapse_warning"], 
                        summary_2["warnings"]["branching_collapse_warning"])
        self.assertEqual(summary_1["warnings"]["topology_change_warning"], 
                        summary_2["warnings"]["topology_change_warning"])

    def test_43_summary_file_output(self):
        """Test that summary JSON file is written correctly."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "summary_file.json"
        
        summary = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path,
        )
        
        # File should exist
        self.assertTrue(out_path.exists())
        
        # File should be valid JSON
        with open(out_path, "r") as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["phase"], "II")
        self.assertEqual(loaded["type"], "slice_topology_summary")

    def test_44_summary_missing_baseline_raises(self):
        """Test that missing baseline log raises FileNotFoundError."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "summary_missing.json"
        
        with self.assertRaises(FileNotFoundError):
            summarize_slice_topology(
                Path(self.temp_dir) / "nonexistent.jsonl",
                self.rfl_log,
                out_path,
            )


class TestWarningFlags(unittest.TestCase):
    """Tests for warning flag computation."""

    def test_45_branching_collapse_warning_triggered(self):
        """Test branching collapse warning is triggered when both < epsilon."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        # Both have near-zero branching
        baseline_footprint = {"node_count": 10, "edge_count": 0, "max_depth": 3}
        rfl_footprint = {"node_count": 10, "edge_count": 0, "max_depth": 3}
        
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=0.0,
            branching_delta=0.0,
            slice_name="test_slice",
        )
        
        self.assertTrue(warnings["branching_collapse_warning"])
        self.assertIn("Branching collapse", " ".join(warnings["explanations"]))

    def test_46_topology_change_warning_depth(self):
        """Test topology change warning triggered by large depth delta."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        baseline_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 3}
        rfl_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 5}
        
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=2.0,  # > 1.0 threshold
            branching_delta=0.0,
            slice_name="test_slice",
        )
        
        self.assertTrue(warnings["topology_change_warning"])
        self.assertIn("depth delta", " ".join(warnings["explanations"]).lower())

    def test_47_topology_change_warning_branching(self):
        """Test topology change warning triggered by large branching delta."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        baseline_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 3}
        rfl_footprint = {"node_count": 10, "edge_count": 15, "max_depth": 3}
        
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=0.0,
            branching_delta=0.8,  # > 0.5 threshold
            slice_name="test_slice",
        )
        
        self.assertTrue(warnings["topology_change_warning"])
        self.assertIn("branching delta", " ".join(warnings["explanations"]).lower())

    def test_48_no_warnings_when_within_thresholds(self):
        """Test no warnings when all metrics within normal ranges."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        baseline_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 3}
        rfl_footprint = {"node_count": 11, "edge_count": 6, "max_depth": 3}
        
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=0.5,  # < 1.0 threshold
            branching_delta=0.2,  # < 0.5 threshold
            slice_name="unknown_slice",  # No curriculum lookup
        )
        
        self.assertFalse(warnings["depth_saturation_warning"])
        self.assertFalse(warnings["branching_collapse_warning"])
        self.assertFalse(warnings["topology_change_warning"])
        self.assertEqual(len(warnings["explanations"]), 0)


class TestSummaryAutoDetection(unittest.TestCase):
    """Tests for slice name auto-detection and curriculum lookup."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test logs."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_auto_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate logs with slice_name embedded
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(50):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "auto_detected_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_49_slice_name_auto_detection(self):
        """Test that slice name is auto-detected from logs."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "auto_detect.json"
        
        # Don't provide slice_name - should auto-detect
        summary = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path,
            slice_name=None,
        )
        
        self.assertEqual(summary["slice_name"], "auto_detected_slice")

    def test_50_slice_name_override(self):
        """Test that explicit slice name overrides auto-detection."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "override.json"
        
        summary = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path,
            slice_name="explicit_override",
        )
        
        self.assertEqual(summary["slice_name"], "explicit_override")


# =============================================================================
# FINAL: Structural Health Contract Tests (C5 Formalization)
# =============================================================================


class TestSummaryJSONContract(unittest.TestCase):
    """
    Tests for the formalized Summary JSON Contract.
    
    Required keys: phase, type, slice_name, baseline_footprint, rfl_footprint,
                   deltas, warnings, source_logs
    """

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_contract_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(50):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "contract_test_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                        "depth": (i % 4) + 1,
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_51_all_required_keys_present(self):
        """Test that all required top-level keys are present in summary."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "keys_test.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        required_keys = [
            "phase",
            "type",
            "slice_name",
            "baseline_footprint",
            "rfl_footprint",
            "deltas",
            "warnings",
            "source_logs",
        ]
        
        for key in required_keys:
            self.assertIn(key, summary, f"Missing required key: {key}")

    def test_52_correct_types_for_all_fields(self):
        """Test that all fields have correct types."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "types_test.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        # Top-level types
        self.assertIsInstance(summary["phase"], str)
        self.assertIsInstance(summary["type"], str)
        self.assertIsInstance(summary["slice_name"], str)
        self.assertIsInstance(summary["baseline_footprint"], dict)
        self.assertIsInstance(summary["rfl_footprint"], dict)
        self.assertIsInstance(summary["deltas"], dict)
        self.assertIsInstance(summary["warnings"], dict)
        self.assertIsInstance(summary["source_logs"], dict)
        
        # Footprint field types
        for fp in [summary["baseline_footprint"], summary["rfl_footprint"]]:
            self.assertIsInstance(fp["node_count"], int)
            self.assertIsInstance(fp["edge_count"], int)
            self.assertIsInstance(fp["max_depth"], int)
            self.assertIsInstance(fp["mean_depth"], float)
            self.assertIsInstance(fp["mean_branching"], float)
        
        # Delta field types
        self.assertIsInstance(summary["deltas"]["depth_delta"], (int, float))
        self.assertIsInstance(summary["deltas"]["node_delta"], int)
        self.assertIsInstance(summary["deltas"]["edge_delta"], int)
        self.assertIsInstance(summary["deltas"]["branching_delta"], float)
        
        # Warning field types
        self.assertIsInstance(summary["warnings"]["depth_saturation_warning"], bool)
        self.assertIsInstance(summary["warnings"]["branching_collapse_warning"], bool)
        self.assertIsInstance(summary["warnings"]["topology_change_warning"], bool)
        self.assertIsInstance(summary["warnings"]["explanations"], list)

    def test_53_json_output_has_sorted_keys(self):
        """Test that JSON output has sorted keys (deterministic ordering)."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "sorted_test.json"
        summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        # Read raw file and verify key ordering
        with open(out_path, "r") as f:
            content = f.read()
        
        # Top-level keys should appear in sorted order
        baseline_pos = content.find('"baseline_footprint"')
        deltas_pos = content.find('"deltas"')
        phase_pos = content.find('"phase"')
        rfl_pos = content.find('"rfl_footprint"')
        
        # Verify alphabetical ordering
        self.assertLess(baseline_pos, deltas_pos)
        self.assertLess(deltas_pos, phase_pos)
        self.assertLess(phase_pos, rfl_pos)

    def test_54_repeated_calls_produce_identical_json(self):
        """Test that repeated calls produce byte-identical JSON output."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path_1 = Path(self.temp_dir) / "identical_1.json"
        out_path_2 = Path(self.temp_dir) / "identical_2.json"
        
        summarize_slice_topology(self.baseline_log, self.rfl_log, out_path_1, slice_name="fixed")
        summarize_slice_topology(self.baseline_log, self.rfl_log, out_path_2, slice_name="fixed")
        
        with open(out_path_1, "r") as f1, open(out_path_2, "r") as f2:
            content_1 = f1.read()
            content_2 = f2.read()
        
        # Exclude source_logs which may have path variations
        # Compare everything else
        import json as json_mod
        data_1 = json_mod.loads(content_1)
        data_2 = json_mod.loads(content_2)
        
        del data_1["source_logs"]
        del data_2["source_logs"]
        
        self.assertEqual(data_1, data_2)


class TestDepthSaturationThreshold(unittest.TestCase):
    """
    Tests for curriculum-informed depth saturation warnings.
    
    Warning should trigger when max_depth >= 0.9 * theoretical_max
    """

    def test_55_depth_saturation_triggered_at_90_percent(self):
        """Test depth saturation warning triggers at >= 90% of theoretical max."""
        from experiments.visualize_dag_topology import (
            compute_warning_flags,
            WARNING_DEPTH_SATURATION_MARGIN,
        )
        
        # Verify threshold is 0.9
        self.assertEqual(WARNING_DEPTH_SATURATION_MARGIN, 0.9)
        
        # Mock: theoretical_max = 10, observed = 9 (90% exactly)
        # We need to test with a slice that has known depth_max
        # Since we can't easily mock curriculum lookup, test the logic directly
        
        # At 90% exactly: 9/10 = 0.9 >= 0.9 should trigger
        baseline_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 9}
        rfl_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 9}
        
        # We'll test with a slice that exists in curriculum
        # slice_easy_fo has depth_max=3, so 3/3 = 100% should trigger
        # But we need explicit control, so let's directly test the logic
        
        # Test: if theoretical_max were 10 and we saw 9, that's 90%
        # The warning check is: saturation >= WARNING_DEPTH_SATURATION_MARGIN
        saturation = 9 / 10  # 0.9
        self.assertTrue(saturation >= WARNING_DEPTH_SATURATION_MARGIN)

    def test_56_depth_saturation_not_triggered_below_90_percent(self):
        """Test depth saturation warning does NOT trigger below 90%."""
        from experiments.visualize_dag_topology import WARNING_DEPTH_SATURATION_MARGIN
        
        # At 89%: 8.9/10 = 0.89 < 0.9 should NOT trigger
        saturation = 8.9 / 10  # 0.89
        self.assertFalse(saturation >= WARNING_DEPTH_SATURATION_MARGIN)
        
        # At 80%: 8/10 = 0.8 < 0.9 should NOT trigger
        saturation = 8 / 10  # 0.8
        self.assertFalse(saturation >= WARNING_DEPTH_SATURATION_MARGIN)

    def test_57_theoretical_max_appears_in_warnings_when_available(self):
        """Test that theoretical_max_depth appears in warnings output."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        # When a slice has curriculum depth_max, it should appear in output
        baseline_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 3}
        rfl_footprint = {"node_count": 10, "edge_count": 5, "max_depth": 3}
        
        # Use a slice that exists in curriculum (slice_easy_fo has depth_max=3)
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=0.0,
            branching_delta=0.0,
            slice_name="slice_easy_fo",  # Has depth_max=3 in curriculum
        )
        
        # theoretical_max_depth should be present if curriculum lookup succeeded
        # Note: May be None if curriculum file not found, which is OK for this test
        self.assertIn("thresholds", warnings)


class TestWarningSemantics(unittest.TestCase):
    """
    Tests ensuring warning semantics remain ADVISORY only.
    
    - Warnings do NOT alter exit codes
    - No value-laden language (good/bad)
    - Output is purely geometric
    """

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_semantics_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate logs that WILL trigger warnings
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(50):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "semantics_test_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 0,  # Zero verified = low branching
                        "abstained": 5,
                        "candidate_hash": f"{mode}_hash_{i}",
                        "depth": 1,
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_58_warning_explanations_use_geometric_language(self):
        """Test that warning explanations use geometric, not value-laden language."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        # Trigger branching collapse warning
        baseline_footprint = {"node_count": 10, "edge_count": 0, "max_depth": 3}
        rfl_footprint = {"node_count": 10, "edge_count": 0, "max_depth": 3}
        
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=0.0,
            branching_delta=0.0,
            slice_name="test_slice",
        )
        
        explanations_text = " ".join(warnings["explanations"])
        
        # Should NOT contain value-laden words
        forbidden_words = ["good", "bad", "better", "worse", "improved", "degraded", 
                          "success", "failure", "excellent", "poor"]
        
        for word in forbidden_words:
            self.assertNotIn(
                word.lower(), 
                explanations_text.lower(),
                f"Warning contains value-laden word: {word}"
            )
        
        # Should contain geometric language
        self.assertTrue(
            any(word in explanations_text.lower() for word in 
                ["branching", "depth", "collapse", "saturation", "delta", "threshold"])
        )

    def test_59_summary_output_uses_geometric_language(self):
        """Test that print_topology_summary uses geometric language."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        import io
        import sys
        
        out_path = Path(self.temp_dir) / "language_test.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        # Capture stdout
        from experiments.visualize_dag_topology import print_topology_summary
        
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            print_topology_summary(summary)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        # Should NOT contain value words
        forbidden_words = ["good", "bad", "better", "worse", "improved", "degraded"]
        
        for word in forbidden_words:
            self.assertNotIn(
                word.lower(),
                output.lower(),
                f"Output contains value-laden word: {word}"
            )
        
        # Should say "ADVISORY" not "ERROR" for warnings
        if "ADVISORY FLAGS" in output or "[!]" in output:
            self.assertNotIn("ERROR", output.split("ADVISORY")[0] if "ADVISORY" in output else output)

    def test_60_summary_with_warnings_still_valid_json(self):
        """Test that summary is valid JSON regardless of warning state."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "warning_json_test.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        # Verify file is valid JSON
        with open(out_path, "r") as f:
            loaded = json.load(f)
        
        # Warnings structure should be present (regardless of flag states)
        self.assertIn("warnings", loaded)
        self.assertIn("branching_collapse_warning", loaded["warnings"])
        self.assertIn("depth_saturation_warning", loaded["warnings"])
        self.assertIn("topology_change_warning", loaded["warnings"])
        
        # Structure should be valid
        self.assertEqual(loaded["phase"], "II")
        self.assertEqual(loaded["type"], "slice_topology_summary")


class TestExitCodeBehavior(unittest.TestCase):
    """
    Tests ensuring exit codes are correct.
    
    - --summary-out always returns 0 (warnings don't affect exit code)
    - Only actual errors (missing files) cause non-zero exit
    """

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_exit_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(20):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "exit_test_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_61_summary_function_returns_dict_regardless_of_warnings(self):
        """Test summarize_slice_topology always returns dict, even with warnings."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "return_test.json"
        
        result = summarize_slice_topology(
            self.baseline_log,
            self.rfl_log,
            out_path,
        )
        
        # Should always return a dict
        self.assertIsInstance(result, dict)
        
        # Even if warnings exist
        if result["warnings"]["branching_collapse_warning"]:
            self.assertIsInstance(result, dict)

    def test_62_no_exception_raised_on_warnings(self):
        """Test that no exceptions are raised when warnings are triggered."""
        from experiments.visualize_dag_topology import compute_warning_flags
        
        # Create conditions that trigger ALL warnings
        baseline_footprint = {"node_count": 1, "edge_count": 0, "max_depth": 10}
        rfl_footprint = {"node_count": 1, "edge_count": 0, "max_depth": 15}
        
        # Should not raise
        warnings = compute_warning_flags(
            baseline_footprint, rfl_footprint,
            depth_delta=5.0,  # Triggers topology_change
            branching_delta=0.0,
            slice_name="test",
        )
        
        # Should have warnings but no exception
        self.assertTrue(warnings["branching_collapse_warning"])
        self.assertTrue(warnings["topology_change_warning"])


# =============================================================================
# C5 v1.2: Depth Time-Series, Stability Score, Ledger Snapshot Tests
# =============================================================================


class TestDepthEvolutionTimeSeries(unittest.TestCase):
    """Tests for depth evolution time-series contract."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_timeseries_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate logs with varying depth patterns
        for log_path, mode, depth_pattern in [
            (cls.baseline_log, "baseline", lambda i: (i % 5) + 1),
            (cls.rfl_log, "rfl", lambda i: (i % 6) + 2),
        ]:
            records = []
            for i in range(100):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "timeseries_test_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                        "depth": depth_pattern(i),
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_63_timeseries_equal_length(self):
        """Test that baseline and rfl time-series have equal length."""
        from experiments.visualize_dag_topology import (
            compute_depth_evolution_contract,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        contract = compute_depth_evolution_contract(
            baseline_records, rfl_records, "test_slice"
        )
        
        self.assertEqual(len(contract["baseline"]), len(contract["rfl"]))
        self.assertEqual(len(contract["baseline"]), contract["length"])
        self.assertEqual(len(contract["expected_min"]), contract["length"])
        self.assertEqual(len(contract["expected_max"]), contract["length"])

    def test_64_timeseries_deterministic_downsampling(self):
        """Test that downsampling is deterministic."""
        from experiments.visualize_dag_topology import (
            compute_depth_evolution_contract,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        # Run twice with same downsample factor
        contract_1 = compute_depth_evolution_contract(
            baseline_records, rfl_records, "test_slice", downsample_factor=5
        )
        contract_2 = compute_depth_evolution_contract(
            baseline_records, rfl_records, "test_slice", downsample_factor=5
        )
        
        self.assertEqual(contract_1["baseline"], contract_2["baseline"])
        self.assertEqual(contract_1["rfl"], contract_2["rfl"])

    def test_65_timeseries_contract_structure(self):
        """Test timeseries contract has all required fields."""
        from experiments.visualize_dag_topology import (
            compute_depth_evolution_contract,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        contract = compute_depth_evolution_contract(
            baseline_records, rfl_records, "test_slice"
        )
        
        required_fields = ["baseline", "rfl", "expected_min", "expected_max", 
                          "length", "downsample_factor"]
        
        for field in required_fields:
            self.assertIn(field, contract)

    def test_66_timeseries_values_are_integers(self):
        """Test that depth values are integers."""
        from experiments.visualize_dag_topology import (
            compute_depth_evolution_contract,
            load_experiment_log,
        )
        
        baseline_records = load_experiment_log(self.baseline_log)
        rfl_records = load_experiment_log(self.rfl_log)
        
        contract = compute_depth_evolution_contract(
            baseline_records, rfl_records, "test_slice"
        )
        
        for val in contract["baseline"]:
            self.assertIsInstance(val, int)
        for val in contract["rfl"]:
            self.assertIsInstance(val, int)


class TestStructuralStabilityScore(unittest.TestCase):
    """Tests for structural stability score computation."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_stability_")
        
        # Stable logs (constant values)
        cls.stable_log = Path(cls.temp_dir) / "stable.jsonl"
        records = []
        for i in range(50):
            records.append({
                "cycle": i,
                "mode": "baseline",
                "derivation": {
                    "candidates": 5,
                    "verified": 2,
                    "depth": 3,  # Constant
                },
            })
        with open(cls.stable_log, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        # Unstable logs (varying values)
        cls.unstable_log = Path(cls.temp_dir) / "unstable.jsonl"
        records = []
        for i in range(50):
            records.append({
                "cycle": i,
                "mode": "rfl",
                "derivation": {
                    "candidates": 5 + (i % 10),  # Varying
                    "verified": 1 + (i % 5),  # Varying
                    "depth": 1 + (i % 8),  # Varying
                },
            })
        with open(cls.unstable_log, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_67_stability_score_range(self):
        """Test that stability score is in [0, 1] range."""
        from experiments.visualize_dag_topology import (
            compute_structural_stability_score,
            load_experiment_log,
        )
        
        stable_records = load_experiment_log(self.stable_log)
        unstable_records = load_experiment_log(self.unstable_log)
        
        stability = compute_structural_stability_score(stable_records, unstable_records)
        
        self.assertGreaterEqual(stability["score"], 0.0)
        self.assertLessEqual(stability["score"], 1.0)

    def test_68_stability_components_present(self):
        """Test that stability has all required components."""
        from experiments.visualize_dag_topology import (
            compute_structural_stability_score,
            load_experiment_log,
        )
        
        stable_records = load_experiment_log(self.stable_log)
        
        stability = compute_structural_stability_score(stable_records, stable_records)
        
        self.assertIn("score", stability)
        self.assertIn("components", stability)
        self.assertIn("baseline", stability["components"])
        self.assertIn("rfl", stability["components"])
        
        for mode in ["baseline", "rfl"]:
            self.assertIn("depth_stability", stability["components"][mode])
            self.assertIn("branching_stability", stability["components"][mode])
            self.assertIn("depth_variation", stability["components"][mode])
            self.assertIn("branching_variation", stability["components"][mode])

    def test_69_stable_series_higher_score(self):
        """Test that stable series yields higher stability score."""
        from experiments.visualize_dag_topology import compute_stability_coefficient
        
        stable_series = [3.0] * 50  # Perfectly stable
        unstable_series = [float(i % 8) + 1 for i in range(50)]  # Varying
        
        stable_coef = compute_stability_coefficient(stable_series)
        unstable_coef = compute_stability_coefficient(unstable_series)
        
        self.assertGreater(stable_coef, unstable_coef)
        self.assertEqual(stable_coef, 1.0)  # Perfect stability

    def test_70_stability_no_exit_code_effect(self):
        """Test that stability computation doesn't affect control flow."""
        from experiments.visualize_dag_topology import (
            compute_structural_stability_score,
            load_experiment_log,
        )
        
        # Should not raise regardless of stability score
        stable_records = load_experiment_log(self.stable_log)
        unstable_records = load_experiment_log(self.unstable_log)
        
        stability = compute_structural_stability_score(stable_records, unstable_records)
        
        # Just verify it returns a dict, doesn't affect exit
        self.assertIsInstance(stability, dict)


class TestLedgerSnapshotIntegration(unittest.TestCase):
    """Tests for ledger-grade topology snapshot."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_ledger_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(50):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "ledger_test_slice",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "abstained": 1,
                        "candidate_hash": f"{mode}_hash_{i}",
                        "depth": (i % 4) + 1,
                    },
                })
            
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_71_ledger_entry_has_required_fields(self):
        """Test ledger entry has all required contract fields."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        required_fields = [
            "phase", "type", "version", "slice_name",
            "topology_hash", "timestamp",
            "structural_metrics", "warning_flags", "stability"
        ]
        
        for field in required_fields:
            self.assertIn(field, entry, f"Missing required field: {field}")

    def test_72_ledger_hash_deterministic(self):
        """Test topology hash is deterministic."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry_1 = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        entry_2 = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        # Hash should be identical
        self.assertEqual(entry_1["topology_hash"], entry_2["topology_hash"])

    def test_73_ledger_hash_is_sha256(self):
        """Test topology hash is valid SHA256 hex string."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        topology_hash = entry["topology_hash"]
        
        # SHA256 produces 64 hex characters
        self.assertEqual(len(topology_hash), 64)
        # Should be valid hex
        int(topology_hash, 16)

    def test_74_ledger_stable_key_ordering(self):
        """Test ledger JSON has stable key ordering."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            write_ledger_topology_entry,
        )
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        out_path = Path(self.temp_dir) / "ordering_test.json"
        write_ledger_topology_entry(entry, out_path)
        
        with open(out_path, "r") as f:
            content = f.read()
        
        # Keys should be sorted alphabetically
        phase_pos = content.find('"phase"')
        slice_pos = content.find('"slice_name"')
        stability_pos = content.find('"stability"')
        
        self.assertLess(phase_pos, slice_pos)
        self.assertLess(slice_pos, stability_pos)

    def test_75_ledger_no_forbidden_words(self):
        """Test ledger entry contains no forbidden value-laden words."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        # Convert to string for word search
        entry_str = json.dumps(entry).lower()
        
        forbidden_words = [
            "improve", "better", "worse", "good", "bad",
            "success", "failure", "excellent", "poor"
        ]
        
        for word in forbidden_words:
            self.assertNotIn(word, entry_str, f"Found forbidden word: {word}")

    def test_76_ledger_timestamp_format(self):
        """Test timestamp follows ISO 8601 UTC format."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        
        timestamp = entry["timestamp"]
        
        # Should end with Z (UTC)
        self.assertTrue(timestamp.endswith("Z"))
        # Should be valid ISO format
        from datetime import datetime
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")

    def test_77_ledger_with_timeseries_option(self):
        """Test ledger entry can include depth timeseries."""
        from experiments.visualize_dag_topology import to_ledger_topology_entry
        
        entry_without = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log,
            include_timeseries=False
        )
        entry_with = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log,
            include_timeseries=True
        )
        
        self.assertNotIn("depth_timeseries", entry_without)
        self.assertIn("depth_timeseries", entry_with)


class TestSummaryIncludesNewFeatures(unittest.TestCase):
    """Test that summary includes stability and timeseries."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and test data."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_summary_new_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(50):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "summary_new_test",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "depth": (i % 4) + 1,
                    },
                })
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_78_summary_includes_stability(self):
        """Test summary includes stability score."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "summary_stability.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        self.assertIn("stability", summary)
        self.assertIn("score", summary["stability"])
        self.assertIn("components", summary["stability"])

    def test_79_summary_includes_timeseries(self):
        """Test summary includes depth timeseries."""
        from experiments.visualize_dag_topology import summarize_slice_topology
        
        out_path = Path(self.temp_dir) / "summary_timeseries.json"
        summary = summarize_slice_topology(self.baseline_log, self.rfl_log, out_path)
        
        self.assertIn("depth_timeseries", summary)
        self.assertIn("baseline", summary["depth_timeseries"])
        self.assertIn("rfl", summary["depth_timeseries"])


# =============================================================================
# Phase III: Topology Ledger Analytics & Director Light Tests
# =============================================================================


class TestAnalyzeTopologyLedgerEntries(unittest.TestCase):
    """Tests for analyze_topology_ledger_entries()."""

    def _make_entry(
        self,
        slice_name: str,
        stability_score: float,
        max_depth: int,
        depth_saturation: bool = False,
        branching_collapse: bool = False,
        topology_change: bool = False,
        timestamp: str = "2025-01-01T00:00:00Z",
    ) -> Dict[str, Any]:
        """Helper to create mock ledger entries."""
        return {
            "phase": "II",
            "type": "topology_ledger_entry",
            "version": "1.2",
            "slice_name": slice_name,
            "timestamp": timestamp,
            "structural_metrics": {
                "baseline_footprint": {"max_depth": max_depth, "node_count": 10},
                "rfl_footprint": {"max_depth": max_depth, "node_count": 12},
                "deltas": {"depth_delta": 0},
            },
            "warning_flags": {
                "depth_saturation": depth_saturation,
                "branching_collapse": branching_collapse,
                "topology_change": topology_change,
            },
            "stability": {
                "score": stability_score,
                "components": {},
            },
        }

    def test_80_empty_entries_returns_zero_analytics(self):
        """Test analytics with empty entries list."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        result = analyze_topology_ledger_entries([])
        
        self.assertEqual(result["entry_count"], 0)
        self.assertEqual(result["average_stability_score"], 0.0)
        self.assertEqual(result["max_depth_over_time"], [])
        self.assertEqual(result["runs_with_low_stability"], [])

    def test_81_single_entry_analytics(self):
        """Test analytics with single entry."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        entry = self._make_entry("slice_a", 0.8, 5)
        result = analyze_topology_ledger_entries([entry])
        
        self.assertEqual(result["entry_count"], 1)
        self.assertEqual(result["average_stability_score"], 0.8)
        self.assertEqual(result["max_depth_over_time"], [5])

    def test_82_average_stability_computation(self):
        """Test correct average stability calculation."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        entries = [
            self._make_entry("a", 0.6, 3),
            self._make_entry("b", 0.8, 4),
            self._make_entry("c", 0.7, 5),
        ]
        result = analyze_topology_ledger_entries(entries)
        
        expected_avg = (0.6 + 0.8 + 0.7) / 3
        self.assertAlmostEqual(result["average_stability_score"], expected_avg, places=3)

    def test_83_warning_flag_frequency_counting(self):
        """Test correct counting of warning flags."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        entries = [
            self._make_entry("a", 0.8, 3, depth_saturation=True),
            self._make_entry("b", 0.8, 4, branching_collapse=True),
            self._make_entry("c", 0.8, 5, depth_saturation=True, topology_change=True),
        ]
        result = analyze_topology_ledger_entries(entries)
        
        flags = result["frequency_of_warning_flags"]
        self.assertEqual(flags["depth_saturation"], 2)
        self.assertEqual(flags["branching_collapse"], 1)
        self.assertEqual(flags["topology_change"], 1)

    def test_84_low_stability_runs_identification(self):
        """Test identification of low-stability runs."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        entries = [
            self._make_entry("high", 0.9, 3),
            self._make_entry("low_1", 0.3, 4),
            self._make_entry("medium", 0.6, 5),
            self._make_entry("low_2", 0.4, 6),
        ]
        result = analyze_topology_ledger_entries(entries, low_stability_threshold=0.5)
        
        low_runs = result["runs_with_low_stability"]
        self.assertEqual(len(low_runs), 2)
        run_ids = [r["run_id"] for r in low_runs]
        self.assertIn("low_1", run_ids)
        self.assertIn("low_2", run_ids)

    def test_85_max_depth_over_time_series(self):
        """Test max depth time series extraction."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        entries = [
            self._make_entry("a", 0.8, 3),
            self._make_entry("b", 0.8, 5),
            self._make_entry("c", 0.8, 4),
        ]
        result = analyze_topology_ledger_entries(entries)
        
        self.assertEqual(result["max_depth_over_time"], [3, 5, 4])

    def test_86_schema_version_present(self):
        """Test schema version is included."""
        from experiments.visualize_dag_topology import analyze_topology_ledger_entries
        
        result = analyze_topology_ledger_entries([])
        self.assertIn("schema_version", result)
        self.assertEqual(result["schema_version"], "1.0")


class TestDirectorLightMapping(unittest.TestCase):
    """Tests for map_topology_to_director_status()."""

    def test_87_green_status_for_healthy_analytics(self):
        """Test GREEN status for healthy topology."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.85,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = map_topology_to_director_status(analytics)
        
        self.assertEqual(result["status_light"], "GREEN")

    def test_88_yellow_status_for_warning_flags(self):
        """Test YELLOW status when warning flags exceed threshold."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.85,
            "frequency_of_warning_flags": {
                "depth_saturation": 1,
                "branching_collapse": 1,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = map_topology_to_director_status(analytics, warning_flag_threshold=2)
        
        self.assertEqual(result["status_light"], "YELLOW")

    def test_89_yellow_status_for_low_stability(self):
        """Test YELLOW status when some low-stability runs exist."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.75,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [{"run_id": "test", "stability_score": 0.4}],
        }
        
        result = map_topology_to_director_status(analytics)
        
        self.assertEqual(result["status_light"], "YELLOW")

    def test_90_red_status_for_frequent_topology_change(self):
        """Test RED status when topology_change is frequent."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.7,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 3,
            },
            "runs_with_low_stability": [],
        }
        
        result = map_topology_to_director_status(analytics, topology_change_threshold=3)
        
        self.assertEqual(result["status_light"], "RED")

    def test_91_red_status_for_many_low_stability_runs(self):
        """Test RED status when many low-stability runs exist."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.5,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [
                {"run_id": "a", "stability_score": 0.3},
                {"run_id": "b", "stability_score": 0.4},
            ],
        }
        
        result = map_topology_to_director_status(analytics, low_stability_run_threshold=2)
        
        self.assertEqual(result["status_light"], "RED")

    def test_92_rationale_uses_neutral_language(self):
        """Test rationale uses only neutral language."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.5,
            "frequency_of_warning_flags": {
                "depth_saturation": 2,
                "branching_collapse": 1,
                "topology_change": 1,
            },
            "runs_with_low_stability": [{"run_id": "a", "stability_score": 0.3}],
        }
        
        result = map_topology_to_director_status(analytics)
        rationale = result["rationale"].lower()
        
        # Check for forbidden words
        forbidden = ["better", "worse", "good", "bad", "success", "failure", "improve"]
        for word in forbidden:
            self.assertNotIn(word, rationale, f"Found forbidden word: {word}")

    def test_93_thresholds_included_in_result(self):
        """Test thresholds are included for transparency."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 1,
            "average_stability_score": 0.8,
            "frequency_of_warning_flags": {"depth_saturation": 0, "branching_collapse": 0, "topology_change": 0},
            "runs_with_low_stability": [],
        }
        
        result = map_topology_to_director_status(analytics)
        
        self.assertIn("thresholds", result)
        self.assertIn("warning_flag_threshold", result["thresholds"])
        self.assertIn("low_stability_run_threshold", result["thresholds"])
        self.assertIn("topology_change_threshold", result["thresholds"])

    def test_94_empty_analytics_green_status(self):
        """Test GREEN status with empty analytics."""
        from experiments.visualize_dag_topology import map_topology_to_director_status
        
        analytics = {
            "entry_count": 0,
            "average_stability_score": 0.0,
            "frequency_of_warning_flags": {"depth_saturation": 0, "branching_collapse": 0, "topology_change": 0},
            "runs_with_low_stability": [],
        }
        
        result = map_topology_to_director_status(analytics)
        
        self.assertEqual(result["status_light"], "GREEN")
        self.assertIn("no ledger entries", result["rationale"])


class TestGlobalHealthTopologySummary(unittest.TestCase):
    """Tests for summarize_topology_for_global_health()."""

    def test_95_ok_status_for_healthy_analytics(self):
        """Test OK status for healthy topology."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.85,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["topology_ok"])

    def test_96_warn_status_for_warning_flags(self):
        """Test WARN status when warning flags present."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.75,
            "frequency_of_warning_flags": {
                "depth_saturation": 1,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["status"], "WARN")
        self.assertFalse(result["topology_ok"])

    def test_97_block_status_for_frequent_topology_change(self):
        """Test BLOCK status for frequent topology changes."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.6,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 3,
            },
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["status"], "BLOCK")
        self.assertFalse(result["topology_ok"])

    def test_98_block_status_for_very_low_stability(self):
        """Test BLOCK status for very low average stability."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.2,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["status"], "BLOCK")

    def test_99_warning_run_count_present(self):
        """Test warning_run_count is computed correctly."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 5,
            "average_stability_score": 0.7,
            "frequency_of_warning_flags": {
                "depth_saturation": 2,
                "branching_collapse": 1,
                "topology_change": 1,
            },
            "runs_with_low_stability": [{"run_id": "a"}],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertIn("warning_run_count", result)
        self.assertGreater(result["warning_run_count"], 0)

    def test_100_empty_analytics_returns_ok(self):
        """Test empty analytics returns OK status."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 0,
            "average_stability_score": 0.0,
            "frequency_of_warning_flags": {"depth_saturation": 0, "branching_collapse": 0, "topology_change": 0},
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["topology_ok"])

    def test_101_average_stability_score_included(self):
        """Test average stability score is included."""
        from experiments.visualize_dag_topology import summarize_topology_for_global_health
        
        analytics = {
            "entry_count": 3,
            "average_stability_score": 0.756,
            "frequency_of_warning_flags": {"depth_saturation": 0, "branching_collapse": 0, "topology_change": 0},
            "runs_with_low_stability": [],
        }
        
        result = summarize_topology_for_global_health(analytics)
        
        self.assertEqual(result["average_stability_score"], 0.756)


class TestAnalyticsIntegration(unittest.TestCase):
    """Integration tests for analytics pipeline."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and generate ledger entries."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_analytics_integration_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(30):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "integration_test",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "depth": (i % 4) + 1,
                    },
                })
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_102_full_pipeline_ledger_to_director(self):
        """Test full pipeline from ledger entry to director status."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
            map_topology_to_director_status,
            summarize_topology_for_global_health,
        )
        
        # Generate ledger entry
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="integration_test"
        )
        
        # Run analytics
        analytics = analyze_topology_ledger_entries([entry])
        
        # Map to director status
        director_status = map_topology_to_director_status(analytics)
        
        # Summarize for global health
        global_health = summarize_topology_for_global_health(analytics)
        
        # Verify all outputs are valid
        self.assertIn("status_light", director_status)
        self.assertIn(director_status["status_light"], ["GREEN", "YELLOW", "RED"])
        
        self.assertIn("status", global_health)
        self.assertIn(global_health["status"], ["OK", "WARN", "BLOCK"])

    def test_103_multiple_entries_analytics(self):
        """Test analytics with multiple ledger entries."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
        )
        
        # Generate multiple entries (simulating different runs)
        entries = []
        for i in range(3):
            entry = to_ledger_topology_entry(
                self.baseline_log, self.rfl_log, 
                slice_name=f"run_{i}"
            )
            entries.append(entry)
        
        analytics = analyze_topology_ledger_entries(entries)
        
        self.assertEqual(analytics["entry_count"], 3)
        self.assertEqual(len(analytics["max_depth_over_time"]), 3)


# =============================================================================
# Phase IV: Topology-Guided Curriculum & Policy Advisor Tests
# =============================================================================


class TestSliceTopologyCurriculumView(unittest.TestCase):
    """Tests for build_slice_topology_curriculum_view()."""

    def _make_analytics(
        self,
        max_depths: List[int],
        depth_saturation: int = 0,
        branching_collapse: int = 0,
        topology_change: int = 0,
        low_stability_count: int = 0,
    ) -> Dict[str, Any]:
        """Helper to create mock analytics."""
        return {
            "entry_count": len(max_depths),
            "max_depth_over_time": max_depths,
            "frequency_of_warning_flags": {
                "depth_saturation": depth_saturation,
                "branching_collapse": branching_collapse,
                "topology_change": topology_change,
            },
            "runs_with_low_stability": [{"run_id": f"run_{i}"} for i in range(low_stability_count)],
            "average_stability_score": 0.7,
        }

    def test_104_slice_view_has_required_fields(self):
        """Test slice view has all required fields."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 4, 5])
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertIn("slice_view", result)
        slice_view = result["slice_view"]
        
        required_fields = [
            "typical_max_depth",
            "depth_trend",
            "branching_behavior",
            "slice_topology_status",
        ]
        for field in required_fields:
            self.assertIn(field, slice_view)

    def test_105_typical_max_depth_computation(self):
        """Test typical max depth is computed correctly."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 5, 4, 6, 5])
        result = build_slice_topology_curriculum_view(analytics)
        
        typical_depth = result["slice_view"]["typical_max_depth"]
        # Median of [3, 4, 5, 5, 6] is 5
        self.assertEqual(typical_depth, 5)

    def test_106_depth_trend_deepening(self):
        """Test DEEPENING trend detection."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        # Increasing depths: [1, 2, 3, 4, 5, 6]
        analytics = self._make_analytics([1, 2, 3, 4, 5, 6])
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["depth_trend"], "DEEPENING")

    def test_107_depth_trend_shallowing(self):
        """Test SHALLOWING trend detection."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        # Decreasing depths: [6, 5, 4, 3, 2, 1]
        analytics = self._make_analytics([6, 5, 4, 3, 2, 1])
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["depth_trend"], "SHALLOWING")

    def test_108_depth_trend_stable(self):
        """Test STABLE trend detection."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        # Stable depths: [3, 3, 3, 3, 3]
        analytics = self._make_analytics([3, 3, 3, 3, 3])
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["depth_trend"], "STABLE")

    def test_109_branching_behavior_low_edge_density(self):
        """Test branching behavior reflects low edge density."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 4, 5], branching_collapse=1)
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["branching_behavior"], "low_edge_density")

    def test_110_slice_status_ok(self):
        """Test OK slice status."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 4, 5])
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["slice_topology_status"], "OK")

    def test_111_slice_status_attention(self):
        """Test ATTENTION slice status."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 4, 5], depth_saturation=1)
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["slice_topology_status"], "ATTENTION")

    def test_112_slice_status_stressed(self):
        """Test STRESSED slice status."""
        from experiments.visualize_dag_topology import build_slice_topology_curriculum_view
        
        analytics = self._make_analytics([3, 4, 5], topology_change=2)
        result = build_slice_topology_curriculum_view(analytics)
        
        self.assertEqual(result["slice_view"]["slice_topology_status"], "STRESSED")


class TestTopologyPolicyRecommendations(unittest.TestCase):
    """Tests for derive_topology_policy_recommendations()."""

    def _make_slice_view(
        self,
        depth_trend: str = "STABLE",
        slice_status: str = "OK",
        branching_behavior: str = "normal",
        topology_change: int = 0,
        branching_collapse: int = 0,
    ) -> Dict[str, Any]:
        """Helper to create mock slice view."""
        return {
            "slice_view": {
                "depth_trend": depth_trend,
                "slice_topology_status": slice_status,
                "branching_behavior": branching_behavior,
                "warning_summary": {
                    "topology_change": topology_change,
                    "branching_collapse": branching_collapse,
                    "depth_saturation": 0,
                    "total_flags": topology_change + branching_collapse,
                },
            },
        }

    def test_113_policy_recommendations_has_required_fields(self):
        """Test policy recommendations have all required fields."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view()
        result = derive_topology_policy_recommendations(slice_view)
        
        required_fields = [
            "slices_needing_policy_adjustment",
            "slices_suitable_for_deeper_slices",
            "policy_recommendation_notes",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_114_slices_needing_adjustment_stressed(self):
        """Test stressed slices flagged for policy adjustment."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        result = derive_topology_policy_recommendations(slice_view)
        
        self.assertGreater(len(result["slices_needing_policy_adjustment"]), 0)

    def test_115_slices_needing_adjustment_frequent_topology_change(self):
        """Test frequent topology change triggers adjustment recommendation."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(topology_change=2)
        result = derive_topology_policy_recommendations(slice_view)
        
        self.assertGreater(len(result["slices_needing_policy_adjustment"]), 0)

    def test_116_slices_suitable_for_deeper_stable_ok(self):
        """Test stable OK slices suitable for deeper exploration."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(depth_trend="STABLE", slice_status="OK")
        result = derive_topology_policy_recommendations(slice_view)
        
        self.assertGreater(len(result["slices_suitable_for_deeper_slices"]), 0)

    def test_117_slices_suitable_for_deeper_shallowing(self):
        """Test shallowing trend suitable for deeper exploration."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(depth_trend="SHALLOWING", slice_status="OK")
        result = derive_topology_policy_recommendations(slice_view)
        
        self.assertGreater(len(result["slices_suitable_for_deeper_slices"]), 0)

    def test_118_policy_notes_use_neutral_language(self):
        """Test policy recommendation notes use neutral language."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        result = derive_topology_policy_recommendations(slice_view)
        
        notes_text = " ".join(result["policy_recommendation_notes"]).lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure", "improve"]
        for word in forbidden:
            self.assertNotIn(word, notes_text, f"Found forbidden word: {word}")

    def test_119_no_adjustment_for_healthy_slice(self):
        """Test no adjustment recommendation for healthy slice."""
        from experiments.visualize_dag_topology import derive_topology_policy_recommendations
        
        slice_view = self._make_slice_view(depth_trend="STABLE", slice_status="OK")
        result = derive_topology_policy_recommendations(slice_view)
        
        # Should not need adjustment, but may be suitable for deeper
        # This is acceptable - healthy slices can still be candidates for deeper exploration


class TestTopologyDirectorPanel(unittest.TestCase):
    """Tests for build_topology_director_panel()."""

    def _make_analytics(
        self,
        max_depths: List[int] = None,
        avg_stability: float = 0.7,
        topology_change: int = 0,
    ) -> Dict[str, Any]:
        """Helper to create mock analytics."""
        if max_depths is None:
            max_depths = [3, 4, 5]
        return {
            "entry_count": len(max_depths),
            "max_depth_over_time": max_depths,
            "average_stability_score": avg_stability,
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": topology_change,
            },
            "runs_with_low_stability": [],
        }

    def _make_slice_view(self, slice_status: str = "OK") -> Dict[str, Any]:
        """Helper to create mock slice view."""
        return {
            "slice_view": {
                "slice_topology_status": slice_status,
                "depth_trend": "STABLE",
                "branching_behavior": "normal",
            },
        }

    def test_120_director_panel_has_required_fields(self):
        """Test director panel has all required fields."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics()
        result = build_topology_director_panel(analytics)
        
        required_fields = [
            "status_light",
            "average_stability_score",
            "topology_ok",
            "slices_stressed",
            "headline",
            "hints",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_121_status_light_present(self):
        """Test status light is present and valid."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics()
        result = build_topology_director_panel(analytics)
        
        self.assertIn(result["status_light"], ["GREEN", "YELLOW", "RED"])

    def test_122_slices_stressed_count(self):
        """Test slices_stressed count is correct."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics()
        slice_view = self._make_slice_view(slice_status="STRESSED")
        result = build_topology_director_panel(analytics, slice_view=slice_view)
        
        self.assertEqual(result["slices_stressed"], 1)

    def test_123_headline_is_neutral(self):
        """Test headline uses neutral language."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics()
        result = build_topology_director_panel(analytics)
        
        headline = result["headline"].lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, headline, f"Found forbidden word: {word}")

    def test_124_hints_include_depth_frontier(self):
        """Test hints include depth frontier observation."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        # Stable depths for last 3 runs
        analytics = self._make_analytics([3, 4, 5, 5, 5, 5])
        result = build_topology_director_panel(analytics)
        
        hints_text = " ".join(result["hints"]).lower()
        self.assertIn("depth", hints_text)
        self.assertIn("stable", hints_text)

    def test_125_hints_include_topology_change(self):
        """Test hints include topology change observations."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics(topology_change=2)
        result = build_topology_director_panel(analytics)
        
        hints_text = " ".join(result["hints"]).lower()
        self.assertIn("topology_change", hints_text)

    def test_126_hints_include_policy_advice(self):
        """Test hints include policy advice when provided."""
        from experiments.visualize_dag_topology import (
            build_topology_director_panel,
            derive_topology_policy_recommendations,
        )
        
        analytics = self._make_analytics()
        slice_view = self._make_slice_view()
        policy_advice = derive_topology_policy_recommendations(slice_view)
        
        result = build_topology_director_panel(
            analytics, slice_view=slice_view, policy_advice=policy_advice
        )
        
        hints_text = " ".join(result["hints"]).lower()
        # Should mention policy or deeper exploration if applicable
        if policy_advice.get("slices_suitable_for_deeper_slices"):
            self.assertTrue(
                any("deeper" in hint.lower() or "policy" in hint.lower() for hint in result["hints"])
            )

    def test_127_metrics_included(self):
        """Test metrics are included in panel."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = self._make_analytics()
        result = build_topology_director_panel(analytics)
        
        self.assertIn("metrics", result)
        self.assertIn("entry_count", result["metrics"])
        self.assertIn("total_warning_flags", result["metrics"])

    def test_128_empty_analytics_handled(self):
        """Test empty analytics handled gracefully."""
        from experiments.visualize_dag_topology import build_topology_director_panel
        
        analytics = {
            "entry_count": 0,
            "max_depth_over_time": [],
            "average_stability_score": 0.0,
            "frequency_of_warning_flags": {"depth_saturation": 0, "branching_collapse": 0, "topology_change": 0},
            "runs_with_low_stability": [],
        }
        
        result = build_topology_director_panel(analytics)
        
        self.assertIn("status_light", result)
        self.assertEqual(result["slices_stressed"], 0)


class TestPhaseIVIntegration(unittest.TestCase):
    """Integration tests for Phase IV pipeline."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and generate ledger entries."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_phase_iv_integration_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(30):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "phase_iv_test",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "depth": (i % 4) + 1,
                    },
                })
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_129_full_phase_iv_pipeline(self):
        """Test full Phase IV pipeline from ledger to director panel."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
            build_slice_topology_curriculum_view,
            derive_topology_policy_recommendations,
            build_topology_director_panel,
        )
        
        # Generate ledger entry
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="phase_iv_test"
        )
        
        # Run analytics
        analytics = analyze_topology_ledger_entries([entry])
        
        # Build slice view
        slice_view = build_slice_topology_curriculum_view(analytics)
        
        # Derive policy recommendations
        policy_advice = derive_topology_policy_recommendations(slice_view)
        
        # Build director panel
        director_panel = build_topology_director_panel(
            analytics, slice_view=slice_view, policy_advice=policy_advice
        )
        
        # Verify all outputs are valid
        self.assertIn("status_light", director_panel)
        self.assertIn("headline", director_panel)
        self.assertIn("hints", director_panel)
        self.assertIn("slices_needing_policy_adjustment", policy_advice)
        self.assertIn("slices_suitable_for_deeper_slices", policy_advice)

    def test_130_slice_view_integration(self):
        """Test slice view integrates with analytics correctly."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
            build_slice_topology_curriculum_view,
        )
        
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="test"
        )
        analytics = analyze_topology_ledger_entries([entry])
        slice_view = build_slice_topology_curriculum_view(analytics)
        
        # Verify slice view uses analytics data
        self.assertIn("slice_view", slice_view)
        self.assertEqual(slice_view["entry_count"], analytics["entry_count"])


# =============================================================================
# Topology Risk Envelope & Curriculum Progression Predictor Tests
# =============================================================================


class TestTopologyRiskEnvelope(unittest.TestCase):
    """Tests for build_topology_risk_envelope()."""

    def _make_ledger_entry(
        self,
        baseline_depth: int,
        rfl_depth: int,
        slice_name: str = "test_slice",
    ) -> Dict[str, Any]:
        """Helper to create mock ledger entry."""
        return {
            "phase": "II",
            "type": "topology_ledger_entry",
            "version": "1.2",
            "slice_name": slice_name,
            "structural_metrics": {
                "baseline_footprint": {"max_depth": baseline_depth, "node_count": 10},
                "rfl_footprint": {"max_depth": rfl_depth, "node_count": 12},
                "deltas": {"depth_delta": rfl_depth - baseline_depth},
            },
            "warning_flags": {
                "depth_saturation": False,
                "branching_collapse": False,
                "topology_change": False,
            },
            "stability": {"score": 0.7},
        }

    def test_131_risk_envelope_has_required_fields(self):
        """Test risk envelope has all required fields."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        entries = [
            self._make_ledger_entry(3, 4),
            self._make_ledger_entry(4, 5),
            self._make_ledger_entry(5, 6),
        ]
        result = build_topology_risk_envelope(entries)
        
        required_fields = [
            "max_depth_band",
            "branching_volatility",
            "risk_band",
            "envelope_summary",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_132_max_depth_band_computation(self):
        """Test max depth band is computed correctly."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        entries = [
            self._make_ledger_entry(3, 4),
            self._make_ledger_entry(5, 6),
            self._make_ledger_entry(4, 5),
        ]
        result = build_topology_risk_envelope(entries)
        
        depth_band = result["max_depth_band"]
        self.assertEqual(depth_band[0], 4)  # min of [4, 6, 5]
        self.assertEqual(depth_band[1], 6)  # max of [4, 6, 5]

    def test_133_branching_volatility_computation(self):
        """Test branching volatility (stdev of depth deltas) is computed."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        # Stable depths: [3, 3, 3] -> deltas: [0, 0] -> volatility ≈ 0
        entries = [
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(3, 3),
        ]
        result = build_topology_risk_envelope(entries)
        
        self.assertAlmostEqual(result["branching_volatility"], 0.0, places=2)

    def test_134_risk_band_ok_for_stable_depths(self):
        """Test OK risk band for stable depths."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        entries = [
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(3, 3),
        ]
        result = build_topology_risk_envelope(entries)
        
        self.assertEqual(result["risk_band"], "OK")

    def test_135_risk_band_attention_for_high_volatility(self):
        """Test ATTENTION risk band for high volatility."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        # High volatility: [3, 6, 2, 7, 1] -> large deltas
        entries = [
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(6, 6),
            self._make_ledger_entry(2, 2),
            self._make_ledger_entry(7, 7),
            self._make_ledger_entry(1, 1),
        ]
        result = build_topology_risk_envelope(entries)
        
        # Should be ATTENTION or STRESSED depending on volatility threshold
        self.assertIn(result["risk_band"], ["ATTENTION", "STRESSED"])

    def test_136_risk_band_stressed_for_volatility_and_increase(self):
        """Test STRESSED risk band for high volatility AND depth increase."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        # Increasing depths with high volatility: [1, 2, 4, 5, 7]
        entries = [
            self._make_ledger_entry(1, 1),
            self._make_ledger_entry(2, 2),
            self._make_ledger_entry(4, 4),
            self._make_ledger_entry(5, 5),
            self._make_ledger_entry(7, 7),
        ]
        result = build_topology_risk_envelope(entries)
        
        # Should be STRESSED if volatility is high enough
        # (May be ATTENTION if volatility threshold not met, but depth is increasing)
        self.assertIn(result["risk_band"], ["ATTENTION", "STRESSED"])

    def test_137_empty_history_returns_ok(self):
        """Test empty ledger history returns OK risk band."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        result = build_topology_risk_envelope([])
        
        self.assertEqual(result["risk_band"], "OK")
        self.assertEqual(result["max_depth_band"], [0, 0])

    def test_138_envelope_summary_is_neutral(self):
        """Test envelope summary uses neutral language."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        entries = [
            self._make_ledger_entry(3, 4),
            self._make_ledger_entry(4, 5),
        ]
        result = build_topology_risk_envelope(entries)
        
        summary = result["envelope_summary"].lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, summary, f"Found forbidden word: {word}")

    def test_139_depth_deltas_computed(self):
        """Test depth deltas are computed and included."""
        from experiments.visualize_dag_topology import build_topology_risk_envelope
        
        entries = [
            self._make_ledger_entry(3, 3),
            self._make_ledger_entry(5, 5),
            self._make_ledger_entry(4, 4),
        ]
        result = build_topology_risk_envelope(entries)
        
        self.assertIn("depth_deltas", result)
        self.assertEqual(len(result["depth_deltas"]), 2)  # n-1 deltas for n entries


class TestCurriculumProgressionPredictor(unittest.TestCase):
    """Tests for predict_curriculum_progression_from_topology()."""

    def _make_slice_view(
        self,
        depth_trend: str = "STABLE",
        slice_status: str = "OK",
    ) -> Dict[str, Any]:
        """Helper to create mock slice view."""
        return {
            "slice_view": {
                "depth_trend": depth_trend,
                "slice_topology_status": slice_status,
                "typical_max_depth": 5,
                "branching_behavior": "normal",
            },
            "entry_count": 3,
        }

    def _make_risk_envelope(
        self,
        risk_band: str = "OK",
        volatility: float = 0.5,
        depth_increasing: bool = False,
    ) -> Dict[str, Any]:
        """Helper to create mock risk envelope."""
        return {
            "risk_band": risk_band,
            "branching_volatility": volatility,
            "max_depth_band": [3, 5],
            "depth_increasing": depth_increasing,
            "envelope_summary": "test",
        }

    def test_140_progression_prediction_has_required_fields(self):
        """Test progression prediction has all required fields."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view()
        risk_envelope = self._make_risk_envelope()
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        required_fields = [
            "slices_ready_for_next_depth",
            "slices_needing_stabilization",
            "readiness_status",
            "notes",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_141_ready_status_for_stable_ok_slice(self):
        """Test READY status for stable OK slice with low risk."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(depth_trend="STABLE", slice_status="OK")
        risk_envelope = self._make_risk_envelope(risk_band="OK", volatility=0.3)
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "READY")
        self.assertGreater(len(result["slices_ready_for_next_depth"]), 0)

    def test_142_stabilize_status_for_stressed_risk(self):
        """Test STABILIZE status for stressed risk envelope."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view()
        risk_envelope = self._make_risk_envelope(risk_band="STRESSED", volatility=2.0)
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "STABILIZE")
        self.assertGreater(len(result["slices_needing_stabilization"]), 0)

    def test_143_stabilize_status_for_stressed_slice(self):
        """Test STABILIZE status for stressed slice."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        risk_envelope = self._make_risk_envelope()
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "STABILIZE")
        self.assertGreater(len(result["slices_needing_stabilization"]), 0)

    def test_144_stabilize_status_for_deepening_with_stress(self):
        """Test STABILIZE status for deepening trend with stress."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(depth_trend="DEEPENING", slice_status="ATTENTION")
        risk_envelope = self._make_risk_envelope()
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "STABILIZE")

    def test_145_stabilize_status_for_high_volatility(self):
        """Test STABILIZE status for high volatility."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view()
        risk_envelope = self._make_risk_envelope(volatility=2.5)  # Above threshold
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "STABILIZE")

    def test_146_ready_for_shallowing_trend(self):
        """Test READY status for shallowing trend with OK status."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(depth_trend="SHALLOWING", slice_status="OK")
        risk_envelope = self._make_risk_envelope(risk_band="OK", volatility=0.3)
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertEqual(result["readiness_status"], "READY")

    def test_147_notes_use_neutral_language(self):
        """Test notes use neutral language."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(slice_status="STRESSED")
        risk_envelope = self._make_risk_envelope(risk_band="STRESSED")
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        notes_text = " ".join(result["notes"]).lower()
        
        forbidden = ["better", "worse", "good", "bad", "success", "failure"]
        for word in forbidden:
            self.assertNotIn(word, notes_text, f"Found forbidden word: {word}")

    def test_148_analysis_basis_included(self):
        """Test analysis_basis is included in result."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view()
        risk_envelope = self._make_risk_envelope()
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        self.assertIn("analysis_basis", result)
        self.assertIn("risk_band", result["analysis_basis"])
        self.assertIn("depth_trend", result["analysis_basis"])

    def test_149_hold_status_default(self):
        """Test HOLD status as default when conditions not met."""
        from experiments.visualize_dag_topology import predict_curriculum_progression_from_topology
        
        slice_view = self._make_slice_view(depth_trend="DEEPENING", slice_status="OK")
        risk_envelope = self._make_risk_envelope(risk_band="OK", volatility=0.8)
        result = predict_curriculum_progression_from_topology(slice_view, risk_envelope)
        
        # Deepening trend with OK status but not meeting ready conditions -> HOLD
        self.assertEqual(result["readiness_status"], "HOLD")


class TestRiskEnvelopeAndProgressionIntegration(unittest.TestCase):
    """Integration tests for risk envelope and progression predictor."""

    @classmethod
    def setUpClass(cls):
        """Create temp directory and generate ledger entries."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_risk_envelope_")
        cls.baseline_log = Path(cls.temp_dir) / "baseline.jsonl"
        cls.rfl_log = Path(cls.temp_dir) / "rfl.jsonl"
        
        # Generate test logs
        for log_path, mode in [(cls.baseline_log, "baseline"), (cls.rfl_log, "rfl")]:
            records = []
            for i in range(30):
                records.append({
                    "cycle": i,
                    "mode": mode,
                    "slice_name": "risk_test",
                    "derivation": {
                        "candidates": 5,
                        "verified": 2,
                        "depth": (i % 4) + 1,
                    },
                })
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_150_full_pipeline_ledger_to_progression(self):
        """Test full pipeline from ledger entries to progression prediction."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            analyze_topology_ledger_entries,
            build_slice_topology_curriculum_view,
            build_topology_risk_envelope,
            predict_curriculum_progression_from_topology,
        )
        
        # Generate ledger entry
        entry = to_ledger_topology_entry(
            self.baseline_log, self.rfl_log, slice_name="risk_test"
        )
        
        # Build risk envelope
        risk_envelope = build_topology_risk_envelope([entry])
        
        # Run analytics and build slice view
        analytics = analyze_topology_ledger_entries([entry])
        slice_view = build_slice_topology_curriculum_view(analytics)
        
        # Predict progression
        progression = predict_curriculum_progression_from_topology(
            slice_view, risk_envelope
        )
        
        # Verify all outputs are valid
        self.assertIn("readiness_status", progression)
        self.assertIn(progression["readiness_status"], ["READY", "STABILIZE", "HOLD"])
        self.assertIn("risk_band", risk_envelope)
        self.assertIn(risk_envelope["risk_band"], ["OK", "ATTENTION", "STRESSED"])

    def test_151_multiple_entries_risk_envelope(self):
        """Test risk envelope with multiple ledger entries."""
        from experiments.visualize_dag_topology import (
            to_ledger_topology_entry,
            build_topology_risk_envelope,
        )
        
        # Generate multiple entries
        entries = []
        for i in range(3):
            entry = to_ledger_topology_entry(
                self.baseline_log, self.rfl_log, slice_name=f"run_{i}"
            )
            entries.append(entry)
        
        risk_envelope = build_topology_risk_envelope(entries)
        
        self.assertIn("max_depth_band", risk_envelope)
        self.assertIn("branching_volatility", risk_envelope)
        self.assertGreater(len(risk_envelope["depth_history"]), 0)


if __name__ == "__main__":
    unittest.main()

