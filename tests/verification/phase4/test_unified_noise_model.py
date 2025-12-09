"""
Test Suite for UnifiedNoiseModel MVP

Tests for base + heavy-tail + adaptive noise regimes.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import unittest
from backend.verification.noise_models.unified_model_mvp import (
    UnifiedNoiseModelMVP,
    UnifiedNoiseConfigMVP,
    BaseNoiseConfig,
    HeavyTailConfig,
    AdaptiveNoiseConfig,
)


class TestUnifiedNoiseModelMVP(unittest.TestCase):
    """Test suite for UnifiedNoiseModel MVP."""
    
    def test_deterministic_noise(self):
        """Test that identical seeds produce identical noise."""
        config = UnifiedNoiseConfigMVP(
            base_noise=BaseNoiseConfig(
                timeout_rate=0.5,
                spurious_fail_rate=0.3,
                spurious_pass_rate=0.1,
            ),
        )
        
        # Create two models with same seed
        model1 = UnifiedNoiseModelMVP(config, master_seed=12345)
        model2 = UnifiedNoiseModelMVP(config, master_seed=12345)
        
        # Test 100 items
        for i in range(100):
            item = f"Module.{i}"
            
            # Step cycles
            model1.step_cycle()
            model2.step_cycle()
            
            # Noise decisions should be identical
            self.assertEqual(
                model1.should_timeout(item),
                model2.should_timeout(item),
            )
            self.assertEqual(
                model1.should_spurious_fail(item),
                model2.should_spurious_fail(item),
            )
            self.assertEqual(
                model1.should_spurious_pass(item),
                model2.should_spurious_pass(item),
            )
    
    def test_noise_rate_accuracy(self):
        """Test that empirical noise rates match configured rates."""
        config = UnifiedNoiseConfigMVP(
            base_noise=BaseNoiseConfig(
                timeout_rate=0.1,
                spurious_fail_rate=0.05,
                spurious_pass_rate=0.02,
            ),
        )
        
        model = UnifiedNoiseModelMVP(config, master_seed=12345)
        
        # Sample 10,000 items
        n_samples = 10000
        timeout_count = 0
        fail_count = 0
        pass_count = 0
        
        for i in range(n_samples):
            item = f"Module.{i}"
            model.step_cycle()
            
            if model.should_timeout(item):
                timeout_count += 1
            if model.should_spurious_fail(item):
                fail_count += 1
            if model.should_spurious_pass(item):
                pass_count += 1
        
        # Check empirical rates (within 1%)
        empirical_timeout_rate = timeout_count / n_samples
        empirical_fail_rate = fail_count / n_samples
        empirical_pass_rate = pass_count / n_samples
        
        self.assertAlmostEqual(empirical_timeout_rate, 0.1, delta=0.01)
        self.assertAlmostEqual(empirical_fail_rate, 0.05, delta=0.01)
        self.assertAlmostEqual(empirical_pass_rate, 0.02, delta=0.01)
    
    def test_heavy_tail_timeout_duration(self):
        """Test heavy-tail timeout duration sampling."""
        config = UnifiedNoiseConfigMVP(
            base_noise=BaseNoiseConfig(timeout_rate=1.0),
            heavy_tail=HeavyTailConfig(
                enabled=True,
                pi=0.5,
                lambda_fast=0.1,
                alpha=1.5,
                x_min=100.0,
            ),
        )
        
        model = UnifiedNoiseModelMVP(config, master_seed=12345)
        
        # Sample durations
        durations = []
        for i in range(1000):
            item = f"Module.{i}"
            model.step_cycle()
            duration = model.sample_timeout_duration(item)
            durations.append(duration)
        
        # Should have some very large durations (Pareto tail)
        max_duration = max(durations)
        self.assertGreater(max_duration, 1000.0)  # At least one large timeout
    
    def test_adaptive_noise_adjustment(self):
        """Test adaptive noise adjustment based on policy confidence."""
        config = UnifiedNoiseConfigMVP(
            base_noise=BaseNoiseConfig(timeout_rate=0.1),
            adaptive=AdaptiveNoiseConfig(
                enabled=True,
                gamma=0.5,
            ),
        )
        
        model = UnifiedNoiseModelMVP(config, master_seed=12345)
        
        # High confidence (policy_prob = 0.9) should increase noise rate
        meta_high_conf = {"policy_prob": 0.9}
        
        # Low confidence (policy_prob = 0.5) should not change noise rate
        meta_low_conf = {"policy_prob": 0.5}
        
        # Sample 1000 items with high confidence
        high_conf_count = 0
        for i in range(1000):
            item = f"Module.{i}"
            model.step_cycle()
            if model.should_timeout(item, meta_high_conf):
                high_conf_count += 1
        
        # Sample 1000 items with low confidence
        model2 = UnifiedNoiseModelMVP(config, master_seed=12345)
        low_conf_count = 0
        for i in range(1000):
            item = f"Module.{i}"
            model2.step_cycle()
            if model2.should_timeout(item, meta_low_conf):
                low_conf_count += 1
        
        # High confidence should have higher noise rate
        self.assertGreater(high_conf_count, low_conf_count)


if __name__ == "__main__":
    unittest.main()
