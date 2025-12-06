"""
Standalone Test Runner — Run Noise Model Tests Without Pytest

This script runs the noise model tests without requiring pytest,
allowing validation in environments where pytest is not available.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test functions (without pytest decorators)
from backend.verification.error_codes import (
    VerifierErrorCode,
    VerifierOutcome,
    VerifierTier,
    verified_outcome,
    timeout_outcome,
    spurious_fail_outcome,
)
from backend.verification.noise_sampler import (
    NoiseSampler,
    NoiseConfig,
    TimeoutDistributionConfig,
    create_noise_sampler,
)
from backend.verification.noisy_lean_wrapper import NoisyLeanWrapper
from backend.verification.tier_router import (
    VerifierTierRouter,
    EscalationPolicy,
    create_tier_router,
)
from backend.lean_mode import mock_lean_build


class TestRunner:
    """Simple test runner without pytest."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        try:
            print(f"Running: {test_name}...", end=" ")
            test_func()
            print("✓ PASS")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAIL")
            self.failed += 1
            self.errors.append((test_name, str(e)))
        except Exception as e:
            print(f"✗ ERROR")
            self.failed += 1
            self.errors.append((test_name, f"Exception: {e}\n{traceback.format_exc()}"))
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        if self.errors:
            print("\nFailures:")
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")
        
        return self.failed == 0


# ==================== Test Functions ====================

def test_identical_seeds_produce_identical_noise():
    """Verify that identical seeds produce identical noise signatures."""
    seed = 42
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.1,
        spurious_fail_rate=0.05,
        spurious_pass_rate=0.02,
        timeout_distribution=TimeoutDistributionConfig.uniform(500, 1500),
    )
    
    sampler1 = NoiseSampler(config, seed)
    sampler2 = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(100)]
    
    for ctx in contexts:
        assert sampler1.should_timeout(ctx) == sampler2.should_timeout(ctx)
        assert sampler1.should_spurious_fail(ctx) == sampler2.should_spurious_fail(ctx)
        assert sampler1.should_spurious_pass(ctx) == sampler2.should_spurious_pass(ctx)


def test_noise_rates_match_config():
    """Verify that empirical noise rates match configured rates."""
    seed = 12345
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.1,
        spurious_fail_rate=0.05,
        spurious_pass_rate=0.02,
        timeout_distribution=TimeoutDistributionConfig.uniform(500, 1500),
    )
    sampler = NoiseSampler(config, seed)
    
    n_samples = 10000
    contexts = [f"context_{i}" for i in range(n_samples)]
    
    timeout_count = sum(sampler.should_timeout(ctx) for ctx in contexts)
    fail_count = sum(sampler.should_spurious_fail(ctx) for ctx in contexts)
    pass_count = sum(sampler.should_spurious_pass(ctx) for ctx in contexts)
    
    timeout_rate = timeout_count / n_samples
    fail_rate = fail_count / n_samples
    pass_rate = pass_count / n_samples
    
    assert abs(timeout_rate - 0.1) < 0.01, f"Timeout rate {timeout_rate} != 0.1"
    assert abs(fail_rate - 0.05) < 0.01, f"Fail rate {fail_rate} != 0.05"
    assert abs(pass_rate - 0.02) < 0.01, f"Pass rate {pass_rate} != 0.02"


def test_zero_noise_config():
    """Verify that zero noise rates produce no noise."""
    seed = 99999
    config = NoiseConfig.no_noise()
    sampler = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(1000)]
    
    assert not any(sampler.should_timeout(ctx) for ctx in contexts)
    assert not any(sampler.should_spurious_fail(ctx) for ctx in contexts)
    assert not any(sampler.should_spurious_pass(ctx) for ctx in contexts)


def test_noisy_wrapper_determinism():
    """Verify that noisy wrapper produces deterministic outcomes."""
    seed = 42
    config = NoiseConfig.default_balanced()
    sampler = NoiseSampler(config, seed)
    
    wrapper1 = NoisyLeanWrapper(mock_lean_build, sampler, VerifierTier.BALANCED)
    wrapper2 = NoisyLeanWrapper(mock_lean_build, sampler, VerifierTier.BALANCED)
    
    contexts = [f"context_{i}" for i in range(20)]
    
    for ctx in contexts:
        outcome1 = wrapper1.verify("TestModule", ctx, timeout=60)
        outcome2 = wrapper2.verify("TestModule", ctx, timeout=60)
        
        assert outcome1.error_code == outcome2.error_code
        assert outcome1.success == outcome2.success
        assert outcome1.noise_injected == outcome2.noise_injected


def test_tier_router_determinism():
    """Verify that tier router is deterministic."""
    seed = 99999
    
    router1 = create_tier_router(mock_lean_build, seed)
    router2 = create_tier_router(mock_lean_build, seed)
    
    contexts = [f"context_{i}" for i in range(10)]
    
    for ctx in contexts:
        outcome1 = router1.verify_with_escalation("TestModule", ctx)
        outcome2 = router2.verify_with_escalation("TestModule", ctx)
        
        assert outcome1.error_code == outcome2.error_code
        assert outcome1.success == outcome2.success
        assert outcome1.attempt_count == outcome2.attempt_count


def test_error_code_enum_stability():
    """Verify that error code enum values are stable."""
    assert VerifierErrorCode.VERIFIED.value == "VERIFIED"
    assert VerifierErrorCode.PROOF_INVALID.value == "PROOF_INVALID"
    assert VerifierErrorCode.VERIFIER_TIMEOUT.value == "VERIFIER_TIMEOUT"


def test_outcome_serialization():
    """Verify that outcomes can be serialized and deserialized."""
    outcome = verified_outcome(
        duration_ms=123.45,
        tier=VerifierTier.BALANCED,
        attempt_count=1,
        metadata={"test": "data"},
    )
    
    outcome_dict = outcome.to_dict()
    outcome_restored = VerifierOutcome.from_dict(outcome_dict)
    
    assert outcome_restored.error_code == outcome.error_code
    assert outcome_restored.success == outcome.success
    assert outcome_restored.duration_ms == outcome.duration_ms


def test_end_to_end_determinism():
    """Verify end-to-end determinism from seed to outcome."""
    seed = 123456
    
    router1 = create_tier_router(mock_lean_build, seed)
    outcomes1 = [
        router1.verify_with_escalation("TestModule", f"context_{i}")
        for i in range(20)
    ]
    
    router2 = create_tier_router(mock_lean_build, seed)
    outcomes2 = [
        router2.verify_with_escalation("TestModule", f"context_{i}")
        for i in range(20)
    ]
    
    for i, (o1, o2) in enumerate(zip(outcomes1, outcomes2)):
        assert o1.error_code == o2.error_code, f"Mismatch at index {i}"
        assert o1.success == o2.success, f"Mismatch at index {i}"
        assert o1.noise_injected == o2.noise_injected, f"Mismatch at index {i}"


# ==================== Main ====================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Noise Model Test Suite — Standalone Runner")
    print("=" * 70 + "\n")
    
    runner = TestRunner()
    
    # Run all tests
    tests = [
        (test_identical_seeds_produce_identical_noise, "test_identical_seeds_produce_identical_noise"),
        (test_noise_rates_match_config, "test_noise_rates_match_config"),
        (test_zero_noise_config, "test_zero_noise_config"),
        (test_noisy_wrapper_determinism, "test_noisy_wrapper_determinism"),
        (test_tier_router_determinism, "test_tier_router_determinism"),
        (test_error_code_enum_stability, "test_error_code_enum_stability"),
        (test_outcome_serialization, "test_outcome_serialization"),
        (test_end_to_end_determinism, "test_end_to_end_determinism"),
    ]
    
    for test_func, test_name in tests:
        runner.run_test(test_func, test_name)
    
    # Print summary
    success = runner.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
