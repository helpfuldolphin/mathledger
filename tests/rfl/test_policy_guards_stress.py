"""
Policy Guardrail Stress Tests (TASK 3)

Validates safety guards under adversarial/extreme conditions:
- L2 norm clamping under high learning rates
- Per-weight clipping under repeated large updates
- Long sequences of updates
- Directionality preservation during clamping

All tests are deterministic and fast (<1s).
"""

import pytest
import numpy as np

from rfl.policy import PolicyUpdater, PolicyState, init_cold_start


class TestL2ClampingStress:
    """Stress tests for L2 norm clamping."""
    
    def test_high_learning_rate_repeated_updates(self):
        """Test L2 clamping with very high learning rate over many steps."""
        updater = PolicyUpdater(
            learning_rate=5.0,  # Very high
            max_weight_norm_l2=10.0,
            max_abs_weight=100.0,  # High enough to not interfere
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b", "feat_c"], seed=42)
        
        # Apply 100 steps of positive updates
        for _ in range(100):
            features = {"feat_a": 1.0, "feat_b": 1.0, "feat_c": 1.0}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            # Check L2 norm never exceeds limit
            weights_array = np.array(list(state.weights.values()))
            l2_norm = np.linalg.norm(weights_array)
            assert l2_norm <= updater.max_weight_norm_l2 + 1e-6
    
    def test_explosive_gradient_sequence(self):
        """Test L2 clamping with synthetic explosive gradients."""
        updater = PolicyUpdater(
            learning_rate=2.0,
            max_weight_norm_l2=5.0,
            max_abs_weight=50.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b"], seed=42)
        
        # Apply increasingly large feature values
        for i in range(50):
            feature_magnitude = 1.0 + i * 0.5  # Growing
            features = {"feat_a": feature_magnitude, "feat_b": feature_magnitude}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            weights_array = np.array(list(state.weights.values()))
            l2_norm = np.linalg.norm(weights_array)
            assert l2_norm <= updater.max_weight_norm_l2 + 1e-6
    
    def test_alternating_large_updates(self):
        """Test L2 clamping with alternating positive/negative large updates."""
        updater = PolicyUpdater(
            learning_rate=3.0,
            max_weight_norm_l2=8.0,
            max_abs_weight=100.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b", "feat_c"], seed=42)
        
        # Alternate between positive and negative rewards
        for i in range(100):
            features = {"feat_a": 5.0, "feat_b": 5.0, "feat_c": 5.0}
            reward = 1.0 if i % 2 == 0 else -1.0
            state = updater.update(state, features, reward)
            
            weights_array = np.array(list(state.weights.values()))
            l2_norm = np.linalg.norm(weights_array)
            assert l2_norm <= updater.max_weight_norm_l2 + 1e-6
    
    def test_l2_clamping_preserves_direction(self):
        """Test that L2 clamping preserves weight vector direction."""
        updater = PolicyUpdater(
            learning_rate=10.0,  # Very high to trigger clamping
            max_weight_norm_l2=3.0,
            max_abs_weight=100.0,
            seed=42
        )
        
        # Start with known direction
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0, "feat_c": 2.0},
            seed=42
        )
        
        # Record initial direction (normalized)
        weights_before = np.array([1.0, 2.0, 2.0])
        dir_before = weights_before / np.linalg.norm(weights_before)
        
        # Apply large update that should trigger clamping
        features = {"feat_a": 1.0, "feat_b": 1.0, "feat_c": 1.0}
        reward = 1.0
        new_state = updater.update(state, features, reward)
        
        # Get new direction
        weights_after = np.array([
            new_state.weights["feat_a"],
            new_state.weights["feat_b"],
            new_state.weights["feat_c"]
        ])
        dir_after = weights_after / np.linalg.norm(weights_after)
        
        # Directions should be nearly parallel (dot product close to 1)
        # Since we're adding positive to all, direction should be preserved
        dot_product = np.dot(dir_before, dir_after)
        assert dot_product > 0.9  # Allow some drift from update before clamping


class TestPerWeightClippingStress:
    """Stress tests for per-weight clipping."""
    
    def test_repeated_extreme_updates_single_feature(self):
        """Test per-weight clipping with repeated extreme updates to one feature."""
        updater = PolicyUpdater(
            learning_rate=1.0,
            max_weight_norm_l2=100.0,  # High to not interfere
            max_abs_weight=3.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b"], seed=42)
        
        # Repeatedly update feat_a with extreme values
        for _ in range(200):
            features = {"feat_a": 100.0, "feat_b": 0.1}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            # feat_a should be clipped
            assert abs(state.weights["feat_a"]) <= updater.max_abs_weight + 1e-6
            assert abs(state.weights["feat_b"]) <= updater.max_abs_weight + 1e-6
    
    def test_all_features_extreme_updates(self):
        """Test per-weight clipping when all features receive extreme updates."""
        updater = PolicyUpdater(
            learning_rate=2.0,
            max_weight_norm_l2=1000.0,
            max_abs_weight=5.0,
            seed=42
        )
        
        num_features = 10
        feature_names = [f"feat_{i}" for i in range(num_features)]
        state = init_cold_start(feature_names, seed=42)
        
        # Apply extreme updates to all features
        for _ in range(100):
            features = {name: 50.0 for name in feature_names}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            # All weights should be clipped
            for weight in state.weights.values():
                assert abs(weight) <= updater.max_abs_weight + 1e-6
    
    def test_negative_extreme_updates(self):
        """Test per-weight clipping with extreme negative updates."""
        updater = PolicyUpdater(
            learning_rate=5.0,
            max_weight_norm_l2=100.0,
            max_abs_weight=2.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a"], seed=42)
        
        # Apply extreme negative updates
        for _ in range(50):
            features = {"feat_a": 10.0}
            reward = -1.0
            state = updater.update(state, features, reward)
            
            # Should be clipped to -max_abs_weight
            assert state.weights["feat_a"] >= -updater.max_abs_weight - 1e-6
            assert state.weights["feat_a"] <= updater.max_abs_weight + 1e-6


class TestCombinedGuardrailsStress:
    """Stress tests for combined L2 and per-weight guardrails."""
    
    def test_both_guardrails_under_stress(self):
        """Test both L2 and per-weight limits under extreme conditions."""
        updater = PolicyUpdater(
            learning_rate=3.0,
            max_weight_norm_l2=10.0,
            max_abs_weight=4.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b", "feat_c", "feat_d"], seed=42)
        
        # Apply 500 random-ish updates
        for i in range(500):
            # Vary features and rewards
            features = {
                "feat_a": 2.0 + (i % 5),
                "feat_b": 3.0 - (i % 3),
                "feat_c": 1.0 + (i % 7),
                "feat_d": 4.0
            }
            reward = 1.0 if i % 3 != 0 else -1.0
            state = updater.update(state, features, reward)
            
            # Check both invariants
            weights_array = np.array(list(state.weights.values()))
            l2_norm = np.linalg.norm(weights_array)
            
            assert l2_norm <= updater.max_weight_norm_l2 + 1e-6
            for weight in state.weights.values():
                assert abs(weight) <= updater.max_abs_weight + 1e-6
    
    def test_which_guardrail_triggers_first(self):
        """Test that appropriate guardrail triggers based on configuration."""
        # Case 1: Per-weight should trigger first
        updater1 = PolicyUpdater(
            learning_rate=1.0,
            max_weight_norm_l2=100.0,  # Very high
            max_abs_weight=2.0,        # Low
            seed=42
        )
        
        state1 = init_cold_start(["feat_a"], seed=42)
        
        for _ in range(10):
            features = {"feat_a": 10.0}
            reward = 1.0
            state1 = updater1.update(state1, features, reward)
        
        # Weight should be at per-weight limit, not L2 limit
        assert abs(state1.weights["feat_a"]) == pytest.approx(updater1.max_abs_weight)
        
        # Case 2: L2 should trigger first (with multiple features)
        updater2 = PolicyUpdater(
            learning_rate=2.0,
            max_weight_norm_l2=5.0,    # Low
            max_abs_weight=10.0,       # High
            seed=42
        )
        
        state2 = init_cold_start(["feat_a", "feat_b", "feat_c"], seed=42)
        
        for _ in range(20):
            features = {"feat_a": 1.0, "feat_b": 1.0, "feat_c": 1.0}
            reward = 1.0
            state2 = updater2.update(state2, features, reward)
        
        # L2 norm should be at limit
        weights_array = np.array(list(state2.weights.values()))
        l2_norm = np.linalg.norm(weights_array)
        assert l2_norm == pytest.approx(updater2.max_weight_norm_l2, abs=1e-5)


class TestDeterminismUnderStress:
    """Verify determinism is maintained under stress conditions."""
    
    def test_stress_sequence_is_deterministic(self):
        """Test same seed produces same results under stress."""
        def run_stress_sequence(seed):
            updater = PolicyUpdater(
                learning_rate=5.0,
                max_weight_norm_l2=7.0,
                max_abs_weight=3.0,
                seed=seed
            )
            
            state = init_cold_start(["feat_a", "feat_b"], seed=seed)
            
            for i in range(100):
                features = {"feat_a": 10.0, "feat_b": 5.0}
                reward = 1.0 if i % 2 == 0 else -1.0
                state = updater.update(state, features, reward)
            
            return state
        
        state1 = run_stress_sequence(12345)
        state2 = run_stress_sequence(12345)
        
        # Should be identical
        assert state1.weights == state2.weights
        assert state1.step == state2.step
        assert state1.total_reward == state2.total_reward
    
    def test_different_seeds_produce_different_results(self):
        """Sanity check: different seeds should give different results."""
        def run_stress_sequence(seed):
            updater = PolicyUpdater(
                learning_rate=5.0,
                max_weight_norm_l2=7.0,
                max_abs_weight=3.0,
                seed=seed
            )
            
            state = init_cold_start(["feat_a"], seed=seed)
            
            for _ in range(10):
                features = {"feat_a": 10.0}
                reward = 1.0
                state = updater.update(state, features, reward)
            
            return state
        
        state1 = run_stress_sequence(111)
        state2 = run_stress_sequence(222)
        
        # With different seeds, results could theoretically differ
        # But in this deterministic update (no randomness in update itself),
        # they should actually be the same. This test documents that.
        # If we add stochastic exploration later, this would change.
        assert state1.weights == state2.weights


class TestInvariantsUnderStress:
    """Test invariants are maintained under adversarial conditions."""
    
    def test_no_nan_or_inf_under_extreme_values(self):
        """Test that extreme values don't produce NaN or Inf."""
        updater = PolicyUpdater(
            learning_rate=1e10,  # Extreme learning rate
            max_weight_norm_l2=10.0,
            max_abs_weight=5.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b"], seed=42)
        
        for _ in range(10):
            features = {"feat_a": 1e6, "feat_b": 1e6}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            # Check no NaN or Inf
            for weight in state.weights.values():
                assert not np.isnan(weight)
                assert not np.isinf(weight)
    
    def test_monotonic_step_counter(self):
        """Test step counter always increases."""
        updater = PolicyUpdater(seed=42)
        state = init_cold_start(["feat_a"], seed=42)
        
        for i in range(100):
            features = {"feat_a": 1.0}
            reward = 1.0
            state = updater.update(state, features, reward)
            
            # Step should equal iteration + 1
            assert state.step == i + 1
    
    def test_total_reward_accumulation(self):
        """Test total reward accumulates correctly under stress."""
        updater = PolicyUpdater(seed=42)
        state = init_cold_start(["feat_a"], seed=42)
        
        expected_total = 0.0
        rewards = [1.0, -1.0, 0.0, 1.0, -1.0] * 20  # 100 steps
        
        for reward in rewards:
            features = {"feat_a": 1.0}
            state = updater.update(state, features, reward)
            expected_total += reward
        
        assert state.total_reward == pytest.approx(expected_total)


class TestPerformance:
    """Test performance characteristics (still fast)."""
    
    def test_stress_test_completes_quickly(self):
        """Test that stress tests complete in reasonable time."""
        import time
        
        updater = PolicyUpdater(
            learning_rate=5.0,
            max_weight_norm_l2=10.0,
            max_abs_weight=5.0,
            seed=42
        )
        
        state = init_cold_start(["feat_a", "feat_b", "feat_c"], seed=42)
        
        start_time = time.time()
        
        # Run 1000 updates
        for i in range(1000):
            features = {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0}
            reward = 1.0 if i % 2 == 0 else -1.0
            state = updater.update(state, features, reward)
        
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second (generous bound)
        assert elapsed < 1.0
