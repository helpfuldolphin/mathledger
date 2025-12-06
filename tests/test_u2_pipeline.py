"""
PHASE II -- NOT USED IN PHASE I
Unit tests for u2_pipeline module.
"""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.u2_pipeline import (
    # Constants
    PHASE_II_LABEL,
    DEFAULT_CONFIG_PATH,
    # Curriculum Loader V2
    SliceConfig,
    CurriculumConfigV2,
    load_curriculum_v2,
    # Feature Extraction + Scoring
    CandidateFeatures,
    PolicyWeights,
    compute_item_hash,
    extract_features,
    score_candidate,
    extract_and_score_candidates,
    # Success Metrics
    SuccessMetricConfig,
    SuccessMetricResult,
    evaluate_success_metric,
    # Attestation Bindings
    CycleAttestation,
    compute_attestation_hash,
    create_cycle_attestation,
    # Manifest Generator
    DebugArtifact,
    PairedRunManifest,
    compute_slice_config_hash,
    compute_ht_series_hash,
    generate_seed_schedule,
    create_paired_manifest,
    save_manifest,
    save_debug_artifacts,
)


class TestCurriculumLoaderV2(unittest.TestCase):
    """Tests for curriculum_loader_v2."""

    def test_load_default_config(self):
        """Should load the default curriculum config file."""
        if not DEFAULT_CONFIG_PATH.exists():
            self.skipTest("Default config file not found")
        
        config = load_curriculum_v2()
        self.assertIsInstance(config, CurriculumConfigV2)
        self.assertTrue(len(config.slices) > 0)
        self.assertTrue(len(config.config_hash) == 64)  # SHA256 hex

    def test_get_slice(self):
        """Should retrieve a specific slice by name."""
        if not DEFAULT_CONFIG_PATH.exists():
            self.skipTest("Default config file not found")
        
        config = load_curriculum_v2()
        slice_config = config.get_slice("arithmetic_simple")
        self.assertEqual(slice_config.name, "arithmetic_simple")
        self.assertTrue(len(slice_config.items) > 0)

    def test_get_slice_not_found(self):
        """Should raise KeyError for non-existent slice."""
        if not DEFAULT_CONFIG_PATH.exists():
            self.skipTest("Default config file not found")
        
        config = load_curriculum_v2()
        with self.assertRaises(KeyError):
            config.get_slice("non_existent_slice")

    def test_slice_config_to_dict(self):
        """SliceConfig should serialize to dict correctly."""
        slice_cfg = SliceConfig(
            name="test_slice",
            description="Test description",
            items=["a", "b", "c"],
            prereg_hash="abc123",
            metadata={"extra": "value"},
        )
        result = slice_cfg.to_dict()
        self.assertEqual(result["name"], "test_slice")
        self.assertEqual(result["items"], ["a", "b", "c"])
        self.assertEqual(result["prereg_hash"], "abc123")


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction and scoring."""

    def test_compute_item_hash_deterministic(self):
        """Item hash should be deterministic."""
        hash1 = compute_item_hash("1 + 1")
        hash2 = compute_item_hash("1 + 1")
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)  # Truncated SHA256

    def test_compute_item_hash_different_items(self):
        """Different items should have different hashes."""
        hash1 = compute_item_hash("1 + 1")
        hash2 = compute_item_hash("2 + 2")
        self.assertNotEqual(hash1, hash2)

    def test_extract_features(self):
        """Should extract features from an item."""
        features = extract_features("1 + 1")
        self.assertEqual(features.item, "1 + 1")
        self.assertEqual(features.length, 5)
        self.assertGreater(features.complexity_estimate, 0.0)
        self.assertEqual(features.success_history_score, 0.0)

    def test_extract_features_with_history(self):
        """Should use success history when provided."""
        item_hash = compute_item_hash("1 + 1")
        history = {item_hash: 0.75}
        features = extract_features("1 + 1", history)
        self.assertEqual(features.success_history_score, 0.75)

    def test_score_candidate(self):
        """Should compute score from features and weights."""
        features = CandidateFeatures(
            item="test",
            item_hash="abc",
            length=10,
            complexity_estimate=0.5,
            success_history_score=0.8,
        )
        weights = PolicyWeights(
            length_weight=1.0,
            complexity_weight=2.0,
            success_history_weight=3.0,
        )
        score = score_candidate(features, weights)
        expected = 1.0 * 10 + 2.0 * 0.5 + 3.0 * 0.8
        self.assertEqual(score, expected)

    def test_extract_and_score_candidates_sorting(self):
        """Should return candidates sorted by score descending."""
        items = ["1 + 1", "2 + 2 + 3 + 4 + 5"]  # Short vs long
        weights = PolicyWeights(length_weight=-1.0, complexity_weight=0.0, success_history_weight=0.0)
        
        results = extract_and_score_candidates(items, weights, None)
        
        # Shorter item should have higher score (less negative)
        self.assertEqual(results[0][0], "1 + 1")
        self.assertGreater(results[0][2], results[1][2])


class TestSuccessMetrics(unittest.TestCase):
    """Tests for success metric evaluation."""

    def test_sparse_metric_success(self):
        """Sparse metric should pass when verified >= min_verified."""
        config = SuccessMetricConfig(
            metric_type="sparse",
            parameters={"min_verified": 5},
        )
        result = evaluate_success_metric(config, verified_count=7, attempted_count=10)
        self.assertTrue(result.success)
        self.assertEqual(result.metric_value, 7.0)

    def test_sparse_metric_failure(self):
        """Sparse metric should fail when verified < min_verified."""
        config = SuccessMetricConfig(
            metric_type="sparse",
            parameters={"min_verified": 5},
        )
        result = evaluate_success_metric(config, verified_count=3, attempted_count=10)
        self.assertFalse(result.success)
        self.assertEqual(result.metric_value, 3.0)

    def test_goal_hit_metric(self):
        """Goal hit metric should count verified targets."""
        config = SuccessMetricConfig(
            metric_type="goal_hit",
            parameters={"target_hashes": ["h1", "h2", "h3"], "min_total_verified": 2},
        )
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "other"}]
        result = evaluate_success_metric(
            config, verified_statements=statements
        )
        self.assertTrue(result.success)
        self.assertEqual(result.metric_value, 2.0)

    def test_multi_goal_metric(self):
        """Multi-goal metric should check all required goals."""
        config = SuccessMetricConfig(
            metric_type="multi_goal",
            parameters={"required_goal_hashes": ["h1", "h2"]},
        )
        result = evaluate_success_metric(
            config, verified_hashes={"h1", "h2", "h3"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.metric_value, 2.0)

    def test_unknown_metric_type(self):
        """Unknown metric type should return failure."""
        config = SuccessMetricConfig(
            metric_type="unknown_type",
            parameters={},
        )
        result = evaluate_success_metric(config)
        self.assertFalse(result.success)
        self.assertIn("Unknown metric type", result.details.get("error", ""))


class TestAttestationBindings(unittest.TestCase):
    """Tests for attestation bindings."""

    def test_compute_attestation_hash_deterministic(self):
        """Attestation hash should be deterministic."""
        data = {"key": "value", "num": 42}
        hash1 = compute_attestation_hash(data)
        hash2 = compute_attestation_hash(data)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # Full SHA256

    def test_compute_attestation_hash_order_independent(self):
        """Attestation hash should be key-order independent."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        self.assertEqual(compute_attestation_hash(data1), compute_attestation_hash(data2))

    def test_create_cycle_attestation(self):
        """Should create valid cycle attestation."""
        attestation = create_cycle_attestation(
            cycle_index=0,
            slice_name="test",
            mode="baseline",
            seed=42,
            item="1 + 1",
            result="2",
            success=True,
        )
        self.assertEqual(attestation.cycle_index, 0)
        self.assertEqual(attestation.slice_name, "test")
        self.assertTrue(attestation.success)
        self.assertEqual(len(attestation.attestation_hash), 64)
        self.assertEqual(len(attestation.item_hash), 16)

    def test_cycle_attestation_deterministic(self):
        """Same inputs should produce same attestation hash."""
        att1 = create_cycle_attestation(0, "test", "baseline", 42, "1+1", "2", True)
        att2 = create_cycle_attestation(0, "test", "baseline", 42, "1+1", "2", True)
        self.assertEqual(att1.attestation_hash, att2.attestation_hash)


class TestManifestGenerator(unittest.TestCase):
    """Tests for manifest generation."""

    def test_generate_seed_schedule_deterministic(self):
        """Seed schedule should be deterministic."""
        seeds1 = generate_seed_schedule(42, 10)
        seeds2 = generate_seed_schedule(42, 10)
        self.assertEqual(seeds1, seeds2)
        self.assertEqual(len(seeds1), 10)

    def test_generate_seed_schedule_different_seeds(self):
        """Different initial seeds should produce different schedules."""
        seeds1 = generate_seed_schedule(42, 5)
        seeds2 = generate_seed_schedule(43, 5)
        self.assertNotEqual(seeds1, seeds2)

    def test_compute_slice_config_hash(self):
        """Should compute deterministic hash for slice config."""
        slice_cfg = SliceConfig(
            name="test",
            description="desc",
            items=["a", "b"],
            prereg_hash="hash",
        )
        hash1 = compute_slice_config_hash(slice_cfg)
        hash2 = compute_slice_config_hash(slice_cfg)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_compute_ht_series_hash(self):
        """Should compute deterministic hash for telemetry series."""
        ht_series = [
            {"cycle": 0, "item": "a", "success": True},
            {"cycle": 1, "item": "b", "success": False},
        ]
        hash1 = compute_ht_series_hash(ht_series)
        hash2 = compute_ht_series_hash(ht_series)
        self.assertEqual(hash1, hash2)

    def test_create_paired_manifest(self):
        """Should create valid paired manifest."""
        slice_cfg = SliceConfig(
            name="test",
            description="desc",
            items=["a", "b"],
            prereg_hash="prereg123",
        )
        baseline_ht = [{"cycle": 0, "success": True}]
        rfl_ht = [{"cycle": 0, "success": True}]
        
        manifest = create_paired_manifest(
            experiment_id="exp_001",
            slice_config=slice_cfg,
            cycles=10,
            initial_seed=42,
            baseline_log_path="/path/baseline.jsonl",
            rfl_log_path="/path/rfl.jsonl",
            baseline_ht_series=baseline_ht,
            rfl_ht_series=rfl_ht,
        )
        
        self.assertEqual(manifest.experiment_id, "exp_001")
        self.assertEqual(manifest.slice_name, "test")
        self.assertEqual(manifest.cycles, 10)
        self.assertEqual(manifest.initial_seed, 42)
        self.assertEqual(manifest.prereg_hash, "prereg123")
        self.assertIsNone(manifest.delta_p_placeholder)  # NO INTERPRETATION
        self.assertEqual(len(manifest.deterministic_seed_schedule), 10)

    def test_save_manifest(self):
        """Should save manifest to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = PairedRunManifest(
                label=PHASE_II_LABEL,
                experiment_id="test",
                slice_name="test",
                cycles=5,
                initial_seed=42,
                slice_config_hash="abc",
                prereg_hash="def",
                baseline_log="/path/base.jsonl",
                rfl_log="/path/rfl.jsonl",
                baseline_ht_hash="ghi",
                rfl_ht_hash="jkl",
                delta_p_placeholder=None,
                created_at="2025-01-01T00:00:00Z",
                deterministic_seed_schedule=[1, 2, 3, 4, 5],
            )
            
            output_path = Path(tmpdir) / "manifest.json"
            save_manifest(manifest, output_path)
            
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                data = json.load(f)
            self.assertEqual(data["experiment_id"], "test")
            self.assertIsNone(data["delta_p_placeholder"])

    def test_save_debug_artifacts(self):
        """Should save debug artifacts to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = [
                DebugArtifact(
                    cycle_index=0,
                    candidate_ordering_trace=[{"rank": 0, "item": "a"}],
                    feature_vectors=[{"item": "a", "length": 1}],
                    policy_weights={"w1": 0.5},
                    success_metric_evaluation={"success": True},
                ),
                DebugArtifact(
                    cycle_index=1,
                    candidate_ordering_trace=[{"rank": 0, "item": "b"}],
                    feature_vectors=[{"item": "b", "length": 1}],
                    policy_weights={"w1": 0.6},
                    success_metric_evaluation={"success": False},
                ),
            ]
            
            output_path = Path(tmpdir) / "debug.jsonl"
            save_debug_artifacts(artifacts, output_path)
            
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["cycle_index"], 0)


class TestDeterminism(unittest.TestCase):
    """Tests to verify determinism guarantees."""

    def test_full_pipeline_deterministic(self):
        """Full scoring pipeline should be deterministic."""
        items = ["1 + 1", "2 * 3", "(4 + 5) * 2"]
        weights = PolicyWeights(length_weight=-0.1, complexity_weight=0.5, success_history_weight=1.0)
        
        # Run multiple times
        results = [
            extract_and_score_candidates(items, weights, None)
            for _ in range(10)
        ]
        
        # All results should be identical
        first_ordering = [(item, score) for item, _, score in results[0]]
        for r in results[1:]:
            ordering = [(item, score) for item, _, score in r]
            self.assertEqual(ordering, first_ordering)

    def test_attestation_chain_deterministic(self):
        """Attestation chain should be deterministic."""
        attestations1 = [
            create_cycle_attestation(i, "test", "rfl", 42 + i, f"item{i}", f"result{i}", i % 2 == 0)
            for i in range(5)
        ]
        attestations2 = [
            create_cycle_attestation(i, "test", "rfl", 42 + i, f"item{i}", f"result{i}", i % 2 == 0)
            for i in range(5)
        ]
        
        for a1, a2 in zip(attestations1, attestations2):
            self.assertEqual(a1.attestation_hash, a2.attestation_hash)


class TestReturnTypes(unittest.TestCase):
    """Tests to ensure return types are consistent."""

    def test_success_metric_result_types(self):
        """SuccessMetricResult should have correct types."""
        config = SuccessMetricConfig(metric_type="sparse", parameters={"min_verified": 1})
        result = evaluate_success_metric(config, verified_count=1, attempted_count=1)
        
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.metric_value, float)
        self.assertIsInstance(result.config, SuccessMetricConfig)
        self.assertIsInstance(result.details, dict)

    def test_candidate_features_types(self):
        """CandidateFeatures should have correct types."""
        features = extract_features("1 + 1")
        
        self.assertIsInstance(features.item, str)
        self.assertIsInstance(features.item_hash, str)
        self.assertIsInstance(features.length, int)
        self.assertIsInstance(features.complexity_estimate, float)
        self.assertIsInstance(features.success_history_score, float)

    def test_cycle_attestation_types(self):
        """CycleAttestation should have correct types."""
        att = create_cycle_attestation(0, "test", "baseline", 42, "item", "result", True)
        
        self.assertIsInstance(att.cycle_index, int)
        self.assertIsInstance(att.slice_name, str)
        self.assertIsInstance(att.mode, str)
        self.assertIsInstance(att.seed, int)
        self.assertIsInstance(att.item, str)
        self.assertIsInstance(att.item_hash, str)
        self.assertIsInstance(att.result, str)
        self.assertIsInstance(att.success, bool)
        self.assertIsInstance(att.attestation_hash, str)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
