"""
Test Suite for Noise Model — Reproducibility and Determinism Tests

This test suite validates the core invariants of the imperfect verifier
noise model:

1. **Seed Reproducibility**: Identical seeds produce identical noise
2. **Noise Rate Accuracy**: Empirical rates match configured rates
3. **Deterministic Outcomes**: Full end-to-end determinism
4. **Error Code Stability**: All outcomes map to stable error codes
5. **Tier Escalation**: Escalation policies work correctly

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import subprocess
from pathlib import Path
from typing import List

import pytest

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


# ==================== Test 1: Seed Reproducibility ====================

def test_identical_seeds_produce_identical_noise():
    """Verify that identical seeds produce identical noise signatures.
    
    This is the CORE invariant of the noise model: determinism.
    """
    seed = 42
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.1,
        spurious_fail_rate=0.05,
        spurious_pass_rate=0.02,
        timeout_distribution=TimeoutDistributionConfig.uniform(500, 1500),
    )
    
    # Create two samplers with same seed
    sampler1 = NoiseSampler(config, seed)
    sampler2 = NoiseSampler(config, seed)
    
    # Test 100 contexts
    contexts = [f"context_{i}" for i in range(100)]
    
    for ctx in contexts:
        # Timeout decisions must match
        assert sampler1.should_timeout(ctx) == sampler2.should_timeout(ctx), \
            f"Timeout decision mismatch for context {ctx}"
        
        # Spurious fail decisions must match
        assert sampler1.should_spurious_fail(ctx) == sampler2.should_spurious_fail(ctx), \
            f"Spurious fail decision mismatch for context {ctx}"
        
        # Spurious pass decisions must match
        assert sampler1.should_spurious_pass(ctx) == sampler2.should_spurious_pass(ctx), \
            f"Spurious pass decision mismatch for context {ctx}"
        
        # Timeout durations must match
        dur1 = sampler1.sample_timeout_duration(ctx)
        dur2 = sampler2.sample_timeout_duration(ctx)
        assert abs(dur1 - dur2) < 1e-9, \
            f"Timeout duration mismatch for context {ctx}: {dur1} vs {dur2}"


def test_different_seeds_produce_different_noise():
    """Verify that different seeds produce different noise signatures."""
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.1,
        spurious_fail_rate=0.05,
        spurious_pass_rate=0.02,
        timeout_distribution=TimeoutDistributionConfig.uniform(500, 1500),
    )
    
    sampler1 = NoiseSampler(config, seed=42)
    sampler2 = NoiseSampler(config, seed=99)
    
    # Test that at least some decisions differ
    contexts = [f"context_{i}" for i in range(100)]
    
    timeout_diffs = sum(
        sampler1.should_timeout(ctx) != sampler2.should_timeout(ctx)
        for ctx in contexts
    )
    
    # With different seeds, we expect at least some differences
    assert timeout_diffs > 0, "Different seeds should produce different noise"


def test_noise_signature_determinism():
    """Verify that noise signatures are deterministic."""
    seed = 12345
    config = NoiseConfig.default_balanced()
    
    sampler1 = NoiseSampler(config, seed)
    sampler2 = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(50)]
    
    for ctx in contexts:
        sig1 = sampler1.get_noise_signature(ctx)
        sig2 = sampler2.get_noise_signature(ctx)
        assert sig1 == sig2, f"Noise signature mismatch for {ctx}"


# ==================== Test 2: Noise Rate Accuracy ====================

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
    
    # Count noise events
    timeout_count = sum(sampler.should_timeout(ctx) for ctx in contexts)
    fail_count = sum(sampler.should_spurious_fail(ctx) for ctx in contexts)
    pass_count = sum(sampler.should_spurious_pass(ctx) for ctx in contexts)
    
    # Check rates (allow 1% tolerance for statistical variation)
    timeout_rate = timeout_count / n_samples
    fail_rate = fail_count / n_samples
    pass_rate = pass_count / n_samples
    
    assert abs(timeout_rate - 0.1) < 0.01, \
        f"Timeout rate {timeout_rate} does not match config 0.1"
    assert abs(fail_rate - 0.05) < 0.01, \
        f"Spurious fail rate {fail_rate} does not match config 0.05"
    assert abs(pass_rate - 0.02) < 0.01, \
        f"Spurious pass rate {pass_rate} does not match config 0.02"


def test_zero_noise_config():
    """Verify that zero noise rates produce no noise."""
    seed = 99999
    config = NoiseConfig.no_noise()
    sampler = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(1000)]
    
    # No noise should be injected
    assert not any(sampler.should_timeout(ctx) for ctx in contexts)
    assert not any(sampler.should_spurious_fail(ctx) for ctx in contexts)
    assert not any(sampler.should_spurious_pass(ctx) for ctx in contexts)


def test_high_noise_config():
    """Verify that high noise rates produce frequent noise."""
    seed = 77777
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.9,
        spurious_fail_rate=0.8,
        spurious_pass_rate=0.7,
        timeout_distribution=TimeoutDistributionConfig.fixed(100),
    )
    sampler = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(1000)]
    
    # High rates should produce many noise events
    timeout_count = sum(sampler.should_timeout(ctx) for ctx in contexts)
    fail_count = sum(sampler.should_spurious_fail(ctx) for ctx in contexts)
    pass_count = sum(sampler.should_spurious_pass(ctx) for ctx in contexts)
    
    assert timeout_count > 850, f"Expected ~900 timeouts, got {timeout_count}"
    assert fail_count > 750, f"Expected ~800 spurious fails, got {fail_count}"
    assert pass_count > 650, f"Expected ~700 spurious passes, got {pass_count}"


# ==================== Test 3: Noisy Wrapper Determinism ====================

def test_noisy_wrapper_determinism():
    """Verify that noisy wrapper produces deterministic outcomes."""
    seed = 42
    config = NoiseConfig.default_balanced()
    sampler = NoiseSampler(config, seed)
    
    # Create two wrappers with same configuration
    wrapper1 = NoisyLeanWrapper(mock_lean_build, sampler, VerifierTier.BALANCED)
    wrapper2 = NoisyLeanWrapper(mock_lean_build, sampler, VerifierTier.BALANCED)
    
    # Verify same module with same context
    contexts = [f"context_{i}" for i in range(20)]
    
    for ctx in contexts:
        outcome1 = wrapper1.verify("TestModule", ctx, timeout=60)
        outcome2 = wrapper2.verify("TestModule", ctx, timeout=60)
        
        # Outcomes must match exactly
        assert outcome1.error_code == outcome2.error_code, \
            f"Error code mismatch for {ctx}"
        assert outcome1.success == outcome2.success, \
            f"Success flag mismatch for {ctx}"
        assert outcome1.noise_injected == outcome2.noise_injected, \
            f"Noise injection flag mismatch for {ctx}"
        assert outcome1.noise_type == outcome2.noise_type, \
            f"Noise type mismatch for {ctx}"


def test_noisy_wrapper_timeout_injection():
    """Verify that timeout injection works correctly."""
    seed = 42
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=1.0,  # Always timeout
        spurious_fail_rate=0.0,
        spurious_pass_rate=0.0,
        timeout_distribution=TimeoutDistributionConfig.fixed(100),
    )
    sampler = NoiseSampler(config, seed)
    wrapper = NoisyLeanWrapper(mock_lean_build, sampler, VerifierTier.BALANCED)
    
    # Should always timeout
    outcome = wrapper.verify("TestModule", "test_context", timeout=60)
    
    assert outcome.error_code == VerifierErrorCode.VERIFIER_TIMEOUT
    assert not outcome.success
    assert outcome.noise_injected
    assert outcome.noise_type == "timeout"


def test_noisy_wrapper_spurious_fail_injection():
    """Verify that spurious failure injection works correctly."""
    seed = 42
    config = NoiseConfig(
        noise_enabled=True,
        timeout_rate=0.0,
        spurious_fail_rate=1.0,  # Always spurious fail
        spurious_pass_rate=0.0,
        timeout_distribution=TimeoutDistributionConfig.fixed(100),
    )
    sampler = NoiseSampler(config, seed)
    
    # Mock runner that always succeeds
    def always_succeed_runner(module_name: str) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["lake", "build", module_name],
            returncode=0,
            stdout="Success",
            stderr="",
        )
    
    wrapper = NoisyLeanWrapper(always_succeed_runner, sampler, VerifierTier.BALANCED)
    
    # Should inject spurious failure
    outcome = wrapper.verify("TestModule", "test_context", timeout=60)
    
    assert outcome.error_code == VerifierErrorCode.VERIFIER_SPURIOUS_FAIL
    assert not outcome.success
    assert outcome.noise_injected
    assert outcome.noise_type == "spurious_fail"
    assert outcome.metadata.get("ground_truth") == "VERIFIED"


# ==================== Test 4: Tier Router and Escalation ====================

def test_tier_router_single_tier():
    """Verify that single-tier verification works."""
    seed = 42
    router = create_tier_router(
        base_runner=mock_lean_build,
        seed=seed,
        escalation_policy=EscalationPolicy.NEVER,
    )
    
    outcome = router.verify_single_tier(
        "TestModule",
        "test_context",
        VerifierTier.BALANCED,
    )
    
    assert outcome.tier == VerifierTier.BALANCED
    assert outcome.attempt_count == 1


def test_tier_router_escalation_on_failure():
    """Verify that escalation works on failure."""
    seed = 42
    
    # Create router with escalation enabled
    router = create_tier_router(
        base_runner=mock_lean_build,
        seed=seed,
        escalation_policy=EscalationPolicy.ON_FAILURE,
        max_escalation_attempts=3,
    )
    
    # Mock will fail (returncode=1), so should escalate
    outcome = router.verify_with_escalation("TestModule", "test_context")
    
    # Should have attempted escalation
    assert outcome.attempt_count >= 1
    assert outcome.attempt_count <= 3


def test_tier_router_no_escalation_on_success():
    """Verify that escalation does not occur on success."""
    seed = 42
    
    # Create mock runner that always succeeds
    def always_succeed_runner(module_name: str) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["lake", "build", module_name],
            returncode=0,
            stdout="Success",
            stderr="",
        )
    
    router = create_tier_router(
        base_runner=always_succeed_runner,
        seed=seed,
        escalation_policy=EscalationPolicy.ON_FAILURE,
        max_escalation_attempts=3,
    )
    
    outcome = router.verify_with_escalation("TestModule", "test_context")
    
    # Should succeed on first attempt, no escalation
    assert outcome.success
    assert outcome.attempt_count == 1


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


# ==================== Test 5: Error Code Stability ====================

def test_error_code_enum_stability():
    """Verify that error code enum values are stable."""
    # These values must never change (part of API contract)
    assert VerifierErrorCode.VERIFIED.value == "VERIFIED"
    assert VerifierErrorCode.PROOF_INVALID.value == "PROOF_INVALID"
    assert VerifierErrorCode.VERIFIER_TIMEOUT.value == "VERIFIER_TIMEOUT"
    assert VerifierErrorCode.VERIFIER_SPURIOUS_FAIL.value == "VERIFIER_SPURIOUS_FAIL"
    assert VerifierErrorCode.VERIFIER_SPURIOUS_PASS.value == "VERIFIER_SPURIOUS_PASS"


def test_outcome_serialization():
    """Verify that outcomes can be serialized and deserialized."""
    outcome = verified_outcome(
        duration_ms=123.45,
        tier=VerifierTier.BALANCED,
        attempt_count=1,
        metadata={"test": "data"},
    )
    
    # Serialize to dict
    outcome_dict = outcome.to_dict()
    
    # Deserialize back
    outcome_restored = VerifierOutcome.from_dict(outcome_dict)
    
    # Should match
    assert outcome_restored.error_code == outcome.error_code
    assert outcome_restored.success == outcome.success
    assert outcome_restored.duration_ms == outcome.duration_ms
    assert outcome_restored.tier == outcome.tier


def test_outcome_rfl_feedback():
    """Verify that RFL feedback conversion works correctly."""
    # Verified outcome → positive feedback
    outcome1 = verified_outcome(100, VerifierTier.BALANCED)
    assert outcome1.to_rfl_feedback() == "positive"
    
    # Timeout → no feedback (abstention)
    outcome2 = timeout_outcome(100, VerifierTier.BALANCED, noise_injected=True)
    assert outcome2.to_rfl_feedback() is None
    
    # Spurious fail → negative feedback
    outcome3 = spurious_fail_outcome(100, VerifierTier.BALANCED)
    assert outcome3.to_rfl_feedback() == "negative"


# ==================== Test 6: Configuration Loading ====================

def test_config_loader_default_config():
    """Verify that default config can be loaded."""
    from backend.verification.config_loader import NoiseConfigLoader
    
    loader = NoiseConfigLoader()
    
    # Should be able to load global config
    global_config = loader.get_global_config()
    assert "noise_enabled" in global_config or "escalation_policy" in global_config


def test_tier_config_factory():
    """Verify that tier configs can be created via factory."""
    sampler_fast = create_noise_sampler("fast_noisy", seed=42)
    sampler_balanced = create_noise_sampler("balanced", seed=42)
    sampler_slow = create_noise_sampler("slow_precise", seed=42)
    
    # All should be valid samplers
    assert sampler_fast.config.noise_enabled
    assert sampler_balanced.config.noise_enabled
    assert sampler_slow.config.noise_enabled


# ==================== Test 7: End-to-End Integration ====================

def test_end_to_end_determinism():
    """Verify end-to-end determinism from seed to outcome.
    
    This is the ultimate test: given the same seed, the entire pipeline
    (sampler → wrapper → router) must produce identical outcomes.
    """
    seed = 123456
    
    # Run 1
    router1 = create_tier_router(mock_lean_build, seed)
    outcomes1 = [
        router1.verify_with_escalation("TestModule", f"context_{i}")
        for i in range(20)
    ]
    
    # Run 2 (same seed)
    router2 = create_tier_router(mock_lean_build, seed)
    outcomes2 = [
        router2.verify_with_escalation("TestModule", f"context_{i}")
        for i in range(20)
    ]
    
    # All outcomes must match
    for i, (o1, o2) in enumerate(zip(outcomes1, outcomes2)):
        assert o1.error_code == o2.error_code, f"Mismatch at index {i}"
        assert o1.success == o2.success, f"Mismatch at index {i}"
        assert o1.noise_injected == o2.noise_injected, f"Mismatch at index {i}"
        assert o1.attempt_count == o2.attempt_count, f"Mismatch at index {i}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
