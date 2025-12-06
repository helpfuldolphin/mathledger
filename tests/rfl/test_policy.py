# PHASE II — NOT USED IN PHASE I
"""
Tests for RFL Policy Module
===========================

Tests for policy dynamics, safety guards, and determinism.

Test Protocol:
    1. Verify determinism: same seed + same config => same trajectory
    2. Verify safety guards: overlarge updates get clamped
    3. Verify warm-start: loading from file produces identical behavior
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rfl.policy.update import (
    PolicyStateSnapshot,
    PolicyUpdater,
    LearningScheduleConfig,
    summarize_policy_state,
    init_cold_start,
    init_from_file,
)
from rfl.policy.features import FeatureVector, SLICE_FEATURE_MASKS, extract_features
from rfl.policy.scoring import PolicyScorer, score_candidates


class TestLearningScheduleConfig:
    """Tests for LearningScheduleConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LearningScheduleConfig()
        assert config.learning_rate == 0.01
        assert config.decay_factor == 0.999
        assert config.min_learning_rate == 0.0001
        assert config.max_weight_norm_l2 == 10.0
        assert config.max_abs_weight == 5.0

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict are inverse operations."""
        config = LearningScheduleConfig(
            learning_rate=0.05,
            decay_factor=0.99,
            max_weight_norm_l2=8.0,
        )
        d = config.to_dict()
        loaded = LearningScheduleConfig.from_dict(d)
        assert loaded.learning_rate == config.learning_rate
        assert loaded.decay_factor == config.decay_factor
        assert loaded.max_weight_norm_l2 == config.max_weight_norm_l2


class TestPolicyStateSnapshot:
    """Tests for PolicyStateSnapshot."""

    def test_to_dict_sorted_keys(self):
        """Test that to_dict produces sorted keys for determinism."""
        state = PolicyStateSnapshot(
            slice_name="test",
            weights={"z": 1.0, "a": 2.0, "m": 3.0},
        )
        d = state.to_dict()
        keys = list(d["weights"].keys())
        assert keys == sorted(keys), "Weights should be sorted"

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        state = PolicyStateSnapshot(
            slice_name="test_slice",
            weights={"feat1": 0.5, "feat2": -0.3},
            update_count=10,
            learning_rate=0.005,
            seed=12345,
            clamped=True,
            clamp_count=2,
        )
        json_str = state.to_json()
        loaded = PolicyStateSnapshot.from_json(json_str)
        
        assert loaded.slice_name == state.slice_name
        assert loaded.weights == state.weights
        assert loaded.update_count == state.update_count
        assert loaded.learning_rate == state.learning_rate
        assert loaded.seed == state.seed
        assert loaded.clamped == state.clamped
        assert loaded.clamp_count == state.clamp_count


class TestSummarizePolicyState:
    """Tests for summarize_policy_state function."""

    def test_empty_weights(self):
        """Test summary with no weights."""
        state = PolicyStateSnapshot(slice_name="test", weights={})
        summary = summarize_policy_state(state)
        
        assert summary["slice_name"] == "test"
        assert summary["weight_norm_l1"] == 0.0
        assert summary["weight_norm_l2"] == 0.0
        assert summary["nonzero_weights"] == 0

    def test_norms_computed_correctly(self):
        """Test L1 and L2 norms are computed correctly."""
        state = PolicyStateSnapshot(
            slice_name="test",
            weights={"a": 3.0, "b": 4.0},  # 3-4-5 triangle
        )
        summary = summarize_policy_state(state)
        
        assert summary["weight_norm_l1"] == pytest.approx(7.0)
        assert summary["weight_norm_l2"] == pytest.approx(5.0)
        assert summary["nonzero_weights"] == 2

    def test_deterministic_across_calls(self):
        """Test summary is identical across multiple calls."""
        state = PolicyStateSnapshot(
            slice_name="test",
            weights={"x": 1.0, "y": 2.0, "z": 3.0},
        )
        summaries = [summarize_policy_state(state) for _ in range(10)]
        
        for s in summaries[1:]:
            assert s == summaries[0], "Summary should be deterministic"


class TestInitColdStart:
    """Tests for init_cold_start function."""

    def test_creates_zero_weights(self):
        """Test cold start creates empty weights."""
        schedule = LearningScheduleConfig()
        state = init_cold_start("test_slice", schedule, seed=42)
        
        assert state.slice_name == "test_slice"
        assert state.weights == {}
        assert state.update_count == 0
        assert state.seed == 42
        assert state.phase == "II"

    def test_uses_schedule_learning_rate(self):
        """Test cold start uses learning rate from schedule."""
        schedule = LearningScheduleConfig(learning_rate=0.05)
        state = init_cold_start("test", schedule)
        
        assert state.learning_rate == 0.05


class TestInitFromFile:
    """Tests for init_from_file function."""

    def test_loads_json_file(self):
        """Test loading from JSON file."""
        state = PolicyStateSnapshot(
            slice_name="loaded_slice",
            weights={"w1": 0.5},
            update_count=5,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            with open(path, "w") as f:
                json.dump(state.to_dict(), f)
            
            loaded = init_from_file(path)
            assert loaded.slice_name == "loaded_slice"
            assert loaded.weights == {"w1": 0.5}
            assert loaded.update_count == 5

    def test_loads_jsonl_last_line(self):
        """Test loading from JSONL file (takes last line)."""
        state1 = PolicyStateSnapshot(slice_name="old", update_count=1)
        state2 = PolicyStateSnapshot(slice_name="newer", update_count=5)
        state3 = PolicyStateSnapshot(slice_name="newest", update_count=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "states.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(state1.to_dict()) + "\n")
                f.write(json.dumps(state2.to_dict()) + "\n")
                f.write(json.dumps(state3.to_dict()) + "\n")
            
            loaded = init_from_file(path)
            assert loaded.slice_name == "newest"
            assert loaded.update_count == 10

    def test_file_not_found(self):
        """Test raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            init_from_file(Path("/nonexistent/path.json"))

    def test_empty_file_raises(self):
        """Test raises ValueError for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.write_text("")
            
            with pytest.raises(ValueError, match="Empty"):
                init_from_file(path)


class TestPolicyUpdater:
    """Tests for PolicyUpdater class."""

    def test_single_update(self):
        """Test single feature update."""
        schedule = LearningScheduleConfig(learning_rate=0.1)
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        new_state = updater.update("feat1", gradient=1.0)
        
        assert "feat1" in new_state.weights
        assert new_state.weights["feat1"] == pytest.approx(0.1)  # lr * gradient
        assert new_state.update_count == 1

    def test_learning_rate_decay(self):
        """Test learning rate decays after update."""
        schedule = LearningScheduleConfig(
            learning_rate=0.1,
            decay_factor=0.9,
            min_learning_rate=0.001,
        )
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        state1 = updater.update("feat", 1.0)
        assert state1.learning_rate == pytest.approx(0.09)  # 0.1 * 0.9
        
        state2 = updater.update("feat", 1.0)
        assert state2.learning_rate == pytest.approx(0.081)  # 0.09 * 0.9

    def test_learning_rate_floor(self):
        """Test learning rate does not go below minimum."""
        schedule = LearningScheduleConfig(
            learning_rate=0.01,
            decay_factor=0.1,  # Aggressive decay
            min_learning_rate=0.005,
        )
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        state1 = updater.update("feat", 1.0)
        assert state1.learning_rate >= 0.005

    def test_l2_norm_clamping(self):
        """Test L2 norm clamping rescales weights."""
        schedule = LearningScheduleConfig(
            learning_rate=10.0,  # Large learning rate to trigger clamping
            max_weight_norm_l2=5.0,
        )
        state = PolicyStateSnapshot(
            slice_name="test",
            weights={"a": 3.0, "b": 4.0},  # L2 norm = 5.0
            learning_rate=10.0,
        )
        updater = PolicyUpdater(schedule, state)
        
        # Apply large update that would exceed norm
        new_state = updater.update("a", gradient=10.0)  # Would add 100 to a
        
        # Verify norm is clamped
        weights = new_state.weights
        l2 = np.sqrt(sum(v**2 for v in weights.values()))
        assert l2 <= schedule.max_weight_norm_l2 + 1e-6
        assert new_state.clamped

    def test_per_weight_clipping(self):
        """Test per-weight clipping to max_abs_weight."""
        schedule = LearningScheduleConfig(
            learning_rate=100.0,  # Very large
            max_abs_weight=5.0,
            max_weight_norm_l2=1000.0,  # Don't trigger L2 clamping
        )
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        # Apply huge update
        new_state = updater.update("feat", gradient=1.0)  # Would be 100.0
        
        # Verify clipped to max
        assert abs(new_state.weights["feat"]) <= schedule.max_abs_weight
        assert new_state.clamped

    def test_small_update_not_clamped(self):
        """Test small updates are not clamped."""
        schedule = LearningScheduleConfig(
            learning_rate=0.001,
            max_weight_norm_l2=10.0,
            max_abs_weight=5.0,
        )
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        new_state = updater.update("feat", gradient=1.0)
        
        assert not new_state.clamped
        assert new_state.clamp_count == 0

    def test_batch_update(self):
        """Test batch update applies all gradients."""
        schedule = LearningScheduleConfig(learning_rate=0.1)
        state = init_cold_start("test", schedule)
        updater = PolicyUpdater(schedule, state)
        
        gradients = {"a": 1.0, "b": -1.0, "c": 0.5}
        new_state = updater.batch_update(gradients)
        
        assert new_state.weights["a"] == pytest.approx(0.1)
        assert new_state.weights["b"] == pytest.approx(-0.1)
        assert new_state.weights["c"] == pytest.approx(0.05)
        assert new_state.update_count == 1

    def test_determinism_same_seed(self):
        """Test same seed produces identical updates."""
        schedule = LearningScheduleConfig(learning_rate=0.1)
        
        results = []
        for _ in range(5):
            state = init_cold_start("test", schedule, seed=12345)
            updater = PolicyUpdater(schedule, state)
            
            for i in range(10):
                state = updater.update(f"feat{i % 3}", gradient=float(i))
            
            results.append(state.to_dict())
        
        for r in results[1:]:
            assert r == results[0], "Same seed should produce identical results"

    def test_clamping_deterministic(self):
        """Test clamping behavior is deterministic."""
        schedule = LearningScheduleConfig(
            learning_rate=100.0,
            max_weight_norm_l2=5.0,
            max_abs_weight=3.0,
        )
        
        results = []
        for _ in range(5):
            state = init_cold_start("test", schedule, seed=42)
            updater = PolicyUpdater(schedule, state)
            
            for i in range(5):
                state = updater.update("feat", gradient=1.0)
            
            results.append(state.to_dict())
        
        for r in results[1:]:
            assert r == results[0], "Clamping should be deterministic"


class TestFeatureVector:
    """Tests for FeatureVector."""

    def test_to_array(self):
        """Test conversion to numpy array."""
        fv = FeatureVector(length=10, depth=3, atom_count=2, connective_count=1)
        arr = fv.to_array()
        
        assert len(arr) == 5  # Default features
        assert arr[0] == 10.0  # length

    def test_to_dict_sorted_raw(self):
        """Test raw dict is sorted in output."""
        fv = FeatureVector(raw={"z": 1.0, "a": 2.0})
        d = fv.to_dict()
        keys = list(d["raw"].keys())
        assert keys == sorted(keys)


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_basic_extraction(self):
        """Test basic feature extraction."""
        fv = extract_features("p -> q")
        
        assert fv.length == 6
        assert fv.atom_count == 2  # p, q
        assert fv.connective_count >= 1  # ->

    def test_depth_estimation(self):
        """Test depth is estimated from parentheses."""
        fv_flat = extract_features("p -> q")
        fv_nested = extract_features("((p -> q) -> r)")
        
        assert fv_nested.depth > fv_flat.depth

    def test_deterministic(self):
        """Test extraction is deterministic."""
        text = "p -> (q & r)"
        results = [extract_features(text).to_dict() for _ in range(10)]
        
        for r in results[1:]:
            assert r == results[0]


class TestPolicyScorer:
    """Tests for PolicyScorer."""

    def test_score_empty_weights(self):
        """Test scoring with no weights returns 0."""
        scorer = PolicyScorer(weights={})
        fv = FeatureVector(length=10, depth=5)
        
        score = scorer.score(fv)
        assert score == 0.0

    def test_score_with_weights(self):
        """Test scoring with weights."""
        scorer = PolicyScorer(
            weights={"length": 0.1, "depth": 0.5},
            slice_name="default",
        )
        fv = FeatureVector(length=10, depth=4)
        
        score = scorer.score(fv)
        expected = 0.1 * 10 + 0.5 * 4  # 1.0 + 2.0 = 3.0
        assert score == pytest.approx(expected)


class TestScoreCandidates:
    """Tests for score_candidates function."""

    def test_ranking_order(self):
        """Test candidates are ranked by score."""
        scorer = PolicyScorer(weights={"length": -0.1}, slice_name="default")
        candidates = ["short", "medium text", "this is a longer text"]
        
        ranked = score_candidates(candidates, scorer)
        
        # Shorter should score higher (negative weight on length)
        assert ranked[0][0] == "short"

    def test_deterministic_ranking(self):
        """Test ranking is deterministic."""
        scorer = PolicyScorer(weights={"length": 0.1}, slice_name="default")
        candidates = ["a", "bb", "ccc", "dddd"]
        
        results = [score_candidates(candidates, scorer) for _ in range(10)]
        
        for r in results[1:]:
            assert r == results[0]


class TestWarmStartDeterminism:
    """Tests for warm-start determinism."""

    def test_warm_start_produces_same_trajectory(self):
        """Test warm-start from file produces identical trajectory."""
        schedule = LearningScheduleConfig(learning_rate=0.1)
        
        # Run 1: Full trajectory
        state1 = init_cold_start("test", schedule, seed=42)
        updater1 = PolicyUpdater(schedule, state1)
        for i in range(10):
            state1 = updater1.update(f"feat{i % 3}", float(i))
        
        # Save state at step 5
        state_mid = init_cold_start("test", schedule, seed=42)
        updater_mid = PolicyUpdater(schedule, state_mid)
        for i in range(5):
            state_mid = updater_mid.update(f"feat{i % 3}", float(i))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            with open(path, "w") as f:
                json.dump(state_mid.to_dict(), f)
            
            # Run 2: Load from checkpoint and continue
            loaded_state = init_from_file(path)
            updater2 = PolicyUpdater(schedule, loaded_state)
            for i in range(5, 10):
                loaded_state = updater2.update(f"feat{i % 3}", float(i))
        
        # Final states should match
        assert state1.weights == loaded_state.weights
        assert state1.update_count == loaded_state.update_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
