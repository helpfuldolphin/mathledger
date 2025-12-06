"""
Tests for RFL Policy module.

Validates:
- Policy update mechanics
- Safety guards (L2 clamping, per-weight clipping)
- Determinism
- Telemetry snapshots
- Feature telemetry (TASK 1)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from rfl.policy import (
    PolicyUpdater,
    PolicyState,
    PolicyStateSnapshot,
    init_cold_start,
    init_from_file,
    summarize_policy_state,
    save_policy_state,
)
from rfl.policy.features import extract_features, get_feature_names
from rfl.policy.rewards import compute_reward


class TestPolicyBasics:
    """Test basic policy operations."""
    
    def test_cold_start_initialization(self):
        """Test cold start creates zero-initialized policy."""
        feature_names = ["feat_a", "feat_b", "feat_c"]
        state = init_cold_start(feature_names, seed=42)
        
        assert state.step == 0
        assert state.total_reward == 0.0
        assert state.update_count == 0
        assert len(state.weights) == 3
        assert all(w == 0.0 for w in state.weights.values())
    
    def test_policy_update_basic(self):
        """Test basic policy update without safety violations."""
        updater = PolicyUpdater(learning_rate=0.1, seed=42)
        state = init_cold_start(["feat_a", "feat_b"], seed=42)
        
        features = {"feat_a": 1.0, "feat_b": 2.0}
        reward = 1.0
        
        new_state = updater.update(state, features, reward)
        
        # Check update applied: w_new = w_old + lr * reward * feature
        assert new_state.weights["feat_a"] == pytest.approx(0.1)
        assert new_state.weights["feat_b"] == pytest.approx(0.2)
        assert new_state.step == 1
        assert new_state.update_count == 1
        assert new_state.total_reward == 1.0
    
    def test_policy_update_negative_reward(self):
        """Test policy update with negative reward."""
        updater = PolicyUpdater(learning_rate=0.1, seed=42)
        state = init_cold_start(["feat_a"], seed=42)
        
        features = {"feat_a": 1.0}
        reward = -1.0
        
        new_state = updater.update(state, features, reward)
        
        assert new_state.weights["feat_a"] == pytest.approx(-0.1)
        assert new_state.total_reward == -1.0


class TestSafetyGuards:
    """Test safety guard mechanisms."""
    
    def test_per_weight_clipping(self):
        """Test per-weight clipping prevents individual weights from exploding."""
        updater = PolicyUpdater(
            learning_rate=1.0,
            max_abs_weight=2.0,
            max_weight_norm_l2=100.0,  # High enough to not trigger
            seed=42
        )
        
        # Start with a weight near the limit
        state = PolicyState(weights={"feat_a": 1.5}, seed=42)
        
        # Apply large positive update
        features = {"feat_a": 10.0}
        reward = 1.0
        
        new_state = updater.update(state, features, reward)
        
        # Weight should be clipped to max_abs_weight
        assert new_state.weights["feat_a"] == pytest.approx(2.0)
        assert abs(new_state.weights["feat_a"]) <= updater.max_abs_weight
    
    def test_per_weight_clipping_negative(self):
        """Test per-weight clipping works for negative weights."""
        updater = PolicyUpdater(
            learning_rate=1.0,
            max_abs_weight=2.0,
            max_weight_norm_l2=100.0,
            seed=42
        )
        
        state = PolicyState(weights={"feat_a": -1.5}, seed=42)
        features = {"feat_a": 10.0}
        reward = -1.0
        
        new_state = updater.update(state, features, reward)
        
        # Weight should be clipped to -max_abs_weight
        assert new_state.weights["feat_a"] == pytest.approx(-2.0)
    
    def test_l2_norm_clamping(self):
        """Test L2 norm clamping prevents unbounded weight growth."""
        updater = PolicyUpdater(
            learning_rate=1.0,
            max_weight_norm_l2=5.0,
            max_abs_weight=100.0,  # High enough to not trigger
            seed=42
        )
        
        # Create state with weights that would exceed L2 norm
        state = PolicyState(
            weights={"feat_a": 3.0, "feat_b": 3.0, "feat_c": 3.0},
            seed=42
        )
        
        features = {"feat_a": 1.0, "feat_b": 1.0, "feat_c": 1.0}
        reward = 2.0
        
        new_state = updater.update(state, features, reward)
        
        # Compute L2 norm
        weights_array = np.array(list(new_state.weights.values()))
        l2_norm = np.linalg.norm(weights_array)
        
        # Should be clamped to max_weight_norm_l2
        assert l2_norm <= updater.max_weight_norm_l2 + 1e-6  # Small epsilon for float precision
    
    def test_l2_norm_preserves_direction(self):
        """Test L2 clamping preserves weight direction."""
        updater = PolicyUpdater(
            learning_rate=1.0,
            max_weight_norm_l2=2.0,
            max_abs_weight=100.0,
            seed=42
        )
        
        state = PolicyState(
            weights={"feat_a": 2.0, "feat_b": 2.0},
            seed=42
        )
        
        # Get direction before update
        weights_before = np.array([state.weights["feat_a"], state.weights["feat_b"]])
        
        features = {"feat_a": 1.0, "feat_b": 1.0}
        reward = 1.0
        
        new_state = updater.update(state, features, reward)
        weights_after = np.array([new_state.weights["feat_a"], new_state.weights["feat_b"]])
        
        # Direction should be preserved (parallel vectors)
        # Normalize and check dot product
        dir_before = weights_before / np.linalg.norm(weights_before)
        dir_after = weights_after / np.linalg.norm(weights_after)
        
        dot_product = np.dot(dir_before, dir_after)
        assert dot_product > 0.99  # Nearly parallel


class TestDeterminism:
    """Test deterministic behavior of policy updates."""
    
    def test_same_seed_same_initialization(self):
        """Test same seed produces same initialization."""
        feature_names = ["a", "b", "c"]
        
        state1 = init_cold_start(feature_names, seed=12345)
        state2 = init_cold_start(feature_names, seed=12345)
        
        assert state1.weights == state2.weights
        assert state1.seed == state2.seed
    
    def test_same_seed_same_updates(self):
        """Test same seed produces same update sequence."""
        updater1 = PolicyUpdater(learning_rate=0.1, seed=42)
        updater2 = PolicyUpdater(learning_rate=0.1, seed=42)
        
        state1 = init_cold_start(["feat_a", "feat_b"], seed=42)
        state2 = init_cold_start(["feat_a", "feat_b"], seed=42)
        
        features = {"feat_a": 1.5, "feat_b": 2.5}
        reward = 1.0
        
        new_state1 = updater1.update(state1, features, reward)
        new_state2 = updater2.update(state2, features, reward)
        
        assert new_state1.weights == new_state2.weights
        assert new_state1.step == new_state2.step
        assert new_state1.total_reward == new_state2.total_reward


class TestTelemetry:
    """Test policy telemetry and snapshots."""
    
    def test_summarize_policy_state_basic(self):
        """Test basic policy state summary."""
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": -2.0, "feat_c": 3.0},
            step=10,
            total_reward=5.0,
            seed=42
        )
        
        snapshot = summarize_policy_state(state, include_feature_telemetry=False)
        
        assert snapshot.step == 10
        assert snapshot.num_features == 3
        assert snapshot.total_reward == 5.0
        assert snapshot.l2_norm == pytest.approx(np.sqrt(1**2 + 2**2 + 3**2))
        assert snapshot.l1_norm == pytest.approx(6.0)
        assert snapshot.max_abs_weight == pytest.approx(3.0)
        assert snapshot.mean_weight == pytest.approx((1.0 - 2.0 + 3.0) / 3.0)
        assert snapshot.feature_telemetry is None
    
    def test_summarize_with_feature_telemetry(self):
        """Test feature telemetry computation (TASK 1)."""
        state = PolicyState(
            weights={
                "feat_a": 5.0,
                "feat_b": 3.0,
                "feat_c": -2.0,
                "feat_d": -4.0,
                "feat_e": 0.0,
            },
            step=5,
            total_reward=10.0,
            seed=42
        )
        
        snapshot = summarize_policy_state(
            state,
            include_feature_telemetry=True,
            top_k=3
        )
        
        assert snapshot.feature_telemetry is not None
        
        # Check top-K positive
        top_pos = snapshot.feature_telemetry.top_k_positive
        assert len(top_pos) == 3
        assert top_pos[0] == ("feat_a", 5.0)
        assert top_pos[1] == ("feat_b", 3.0)
        assert top_pos[2] == ("feat_e", 0.0)
        
        # Check top-K negative
        top_neg = snapshot.feature_telemetry.top_k_negative
        assert len(top_neg) == 3
        assert top_neg[0] == ("feat_d", -4.0)
        assert top_neg[1] == ("feat_c", -2.0)
        
        # Check sparsity (4 non-zero out of 5)
        assert snapshot.feature_telemetry.sparsity == pytest.approx(0.8)
    
    def test_feature_telemetry_top_k_ties(self):
        """Test behavior when multiple features share identical weights."""
        state = PolicyState(
            weights={
                "feat_a": 1.0,
                "feat_b": 1.0,  # Tie with feat_a
                "feat_c": 1.0,  # Tie with feat_a and feat_b
                "feat_d": 0.5,
            },
            step=1,
            seed=42
        )
        
        snapshot = summarize_policy_state(
            state,
            include_feature_telemetry=True,
            top_k=2
        )
        
        # Should return top-k even with ties (deterministic ordering from dict)
        assert len(snapshot.feature_telemetry.top_k_positive) == 2
        
        # All top-2 should have weight 1.0
        for name, weight in snapshot.feature_telemetry.top_k_positive:
            assert weight == pytest.approx(1.0)
    
    def test_feature_telemetry_determinism(self):
        """Test feature telemetry is deterministic."""
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0},
            step=1,
            seed=42
        )
        
        snapshot1 = summarize_policy_state(state, include_feature_telemetry=True, top_k=2)
        snapshot2 = summarize_policy_state(state, include_feature_telemetry=True, top_k=2)
        
        assert snapshot1.feature_telemetry.top_k_positive == snapshot2.feature_telemetry.top_k_positive
        assert snapshot1.feature_telemetry.top_k_negative == snapshot2.feature_telemetry.top_k_negative
        assert snapshot1.feature_telemetry.sparsity == snapshot2.feature_telemetry.sparsity


class TestSerialization:
    """Test policy serialization and loading."""
    
    def test_save_and_load_policy(self):
        """Test save and load round-trip."""
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=5,
            total_reward=10.0,
            update_count=5,
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "policy.json"
            save_policy_state(state, filepath)
            
            loaded_state = init_from_file(filepath)
            
            assert loaded_state.weights == state.weights
            assert loaded_state.step == state.step
            assert loaded_state.total_reward == state.total_reward
            assert loaded_state.update_count == state.update_count
            assert loaded_state.seed == state.seed
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            init_from_file(Path("/nonexistent/path.json"))
    
    def test_snapshot_serialization(self):
        """Test snapshot can be serialized to dict/JSON."""
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=5,
            total_reward=10.0,
            seed=42
        )
        
        snapshot = summarize_policy_state(state, include_feature_telemetry=True, top_k=1)
        snapshot_dict = snapshot.to_dict()
        
        # Check basic fields
        assert "step" in snapshot_dict
        assert "l2_norm" in snapshot_dict
        assert "feature_telemetry" in snapshot_dict
        
        # Should be JSON serializable
        json_str = json.dumps(snapshot_dict)
        assert len(json_str) > 0


class TestFeatures:
    """Test feature extraction."""
    
    def test_extract_basic_features(self):
        """Test basic feature extraction from formula."""
        formula = "p -> q"
        features = extract_features(formula)
        
        assert "formula_length" in features.features
        assert "num_atoms" in features.features
        assert "num_implications" in features.features
        assert features.features["num_implications"] == 1.0
        assert features.features["bias"] == 1.0
    
    def test_extract_complex_features(self):
        """Test feature extraction from complex formula."""
        formula = "(p & q) -> (r | ~s)"
        features = extract_features(formula)
        
        assert features.features["num_atoms"] == 4.0  # p, q, r, s
        assert features.features["num_conjunctions"] == 1.0
        assert features.features["num_disjunctions"] == 1.0
        assert features.features["num_negations"] == 1.0
        assert features.features["num_implications"] == 1.0
        assert features.features["tree_depth"] >= 1.0


class TestRewards:
    """Test reward computation."""
    
    def test_compute_reward_success(self):
        """Test reward for successful proof."""
        reward_signal = compute_reward(success=True, abstained=False)
        
        assert reward_signal.reward == 1.0
        assert reward_signal.metadata["outcome"] == "success"
    
    def test_compute_reward_failure(self):
        """Test reward for failed proof."""
        reward_signal = compute_reward(success=False, abstained=False)
        
        assert reward_signal.reward == -1.0
        assert reward_signal.metadata["outcome"] == "failure"
    
    def test_compute_reward_abstention(self):
        """Test reward for abstention."""
        reward_signal = compute_reward(success=False, abstained=True)
        
        assert reward_signal.reward == 0.0
        assert reward_signal.metadata["outcome"] == "abstention"
    
    def test_compute_reward_with_chain_bonus(self):
        """Test reward with chain length bonus."""
        reward_signal = compute_reward(
            success=True,
            abstained=False,
            chain_length=10,
            bonus_for_short_proof=True
        )
        
        # Base reward 1.0 - 0.01 * 10 = 0.9
        assert reward_signal.reward == pytest.approx(0.9)
        assert reward_signal.metadata["bonus_applied"] is True
