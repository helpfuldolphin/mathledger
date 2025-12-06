"""
U2 Integration Tests — End-to-End Tests with Noisy Verifier

This test suite validates the integration of the noisy verifier regime
with the U2 runner, ensuring:

1. **U2 Determinism**: Full U2 runs are deterministic with noise
2. **Noise Telemetry**: Noise events are logged correctly
3. **RFL Robustness**: RFL updates handle noise gracefully

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from pathlib import Path
from typing import Dict, Any

import pytest

from backend.verification.u2_integration import (
    create_noisy_execute_fn,
    outcome_to_rfl_feedback,
    should_update_rfl_policy,
)
from backend.verification.u2_runner_stub import U2Runner, U2Config
from backend.verification.error_codes import VerifierErrorCode


# ==================== Test 1: Noisy Execute Function ====================

def test_noisy_execute_fn_determinism():
    """Verify that noisy execute function is deterministic."""
    slice_name = "test_slice"
    master_seed = 42
    
    # Create two execute functions with same seed
    execute_fn1 = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    execute_fn2 = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    # Execute same items with same seeds
    items = [f"item_{i}" for i in range(10)]
    
    for item in items:
        seed = hash(item) % (2**31)
        
        success1, result1 = execute_fn1(item, seed)
        success2, result2 = execute_fn2(item, seed)
        
        # Results must match
        assert success1 == success2, f"Success mismatch for {item}"
        assert result1.get("outcome") == result2.get("outcome"), \
            f"Outcome mismatch for {item}"


def test_noisy_execute_fn_without_noise():
    """Verify that execute function works without noise."""
    slice_name = "test_slice"
    master_seed = 42
    
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=False,
        use_escalation=False,
    )
    
    # Should work without errors
    success, result = execute_fn("test_item", 12345)
    
    # Result should have outcome field
    assert "outcome" in result


def test_noisy_execute_fn_with_escalation():
    """Verify that execute function uses escalation when enabled."""
    slice_name = "test_slice"
    master_seed = 42
    
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    # Execute multiple items
    items = [f"item_{i}" for i in range(20)]
    
    for item in items:
        seed = hash(item) % (2**31)
        success, result = execute_fn(item, seed)
        
        # Result should have attempt_count field
        assert "attempt_count" in result, f"Missing attempt_count for {item}"


# ==================== Test 2: U2 Runner Integration ====================

def test_u2_runner_with_noise():
    """Verify that U2 runner works with noisy execute function."""
    slice_name = "test_slice"
    master_seed = 42
    
    # Create noisy execute function
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    # Create U2 config
    config = U2Config(
        experiment_id="test_experiment",
        slice_name=slice_name,
        mode="rfl",
        total_cycles=5,
        master_seed=master_seed,
        snapshot_interval=0,
        snapshot_dir=Path("/tmp/snapshots"),
        output_dir=Path("/tmp/output"),
        slice_config={"items": [f"item_{i}" for i in range(5)]},
        execute_fn=execute_fn,
    )
    
    # Run U2 runner
    runner = U2Runner(config)
    results = runner.run()
    
    # Should have results
    assert len(results) > 0
    
    # Get summary
    summary = runner.get_summary()
    assert "total_cycles" in summary
    assert "success_rate" in summary
    assert "noise_injection_rate" in summary


def test_u2_runner_determinism():
    """Verify that U2 runner produces deterministic results.
    
    This is the ultimate integration test: two U2 runs with the same
    seed must produce identical results.
    """
    slice_name = "test_slice"
    master_seed = 99999
    
    def run_u2(seed: int) -> Dict[str, Any]:
        """Helper to run U2 and return summary."""
        execute_fn = create_noisy_execute_fn(
            slice_name=slice_name,
            master_seed=seed,
            noise_enabled=True,
            use_escalation=True,
        )
        
        config = U2Config(
            experiment_id="test_experiment",
            slice_name=slice_name,
            mode="rfl",
            total_cycles=10,
            master_seed=seed,
            snapshot_interval=0,
            snapshot_dir=Path("/tmp/snapshots"),
            output_dir=Path("/tmp/output"),
            slice_config={"items": [f"item_{i}" for i in range(5)]},
            execute_fn=execute_fn,
        )
        
        runner = U2Runner(config)
        results = runner.run()
        
        return runner.get_summary()
    
    # Run 1
    summary1 = run_u2(master_seed)
    
    # Run 2 (same seed)
    summary2 = run_u2(master_seed)
    
    # Summaries must match
    assert summary1["total_cycles"] == summary2["total_cycles"]
    assert summary1["successes"] == summary2["successes"]
    assert summary1["failures"] == summary2["failures"]
    assert summary1["noise_injected_count"] == summary2["noise_injected_count"]
    assert summary1["noise_by_type"] == summary2["noise_by_type"]


def test_u2_runner_different_seeds():
    """Verify that different seeds produce different results."""
    slice_name = "test_slice"
    
    def run_u2(seed: int) -> Dict[str, Any]:
        """Helper to run U2 and return summary."""
        execute_fn = create_noisy_execute_fn(
            slice_name=slice_name,
            master_seed=seed,
            noise_enabled=True,
            use_escalation=True,
        )
        
        config = U2Config(
            experiment_id="test_experiment",
            slice_name=slice_name,
            mode="rfl",
            total_cycles=10,
            master_seed=seed,
            snapshot_interval=0,
            snapshot_dir=Path("/tmp/snapshots"),
            output_dir=Path("/tmp/output"),
            slice_config={"items": [f"item_{i}" for i in range(5)]},
            execute_fn=execute_fn,
        )
        
        runner = U2Runner(config)
        results = runner.run()
        
        return runner.get_summary()
    
    # Run with different seeds
    summary1 = run_u2(seed=42)
    summary2 = run_u2(seed=99)
    
    # Results should differ (at least in noise events)
    # Note: Due to randomness, there's a small chance they could match,
    # but with 50 total cycles, this is extremely unlikely
    assert (
        summary1["noise_injected_count"] != summary2["noise_injected_count"] or
        summary1["noise_by_type"] != summary2["noise_by_type"]
    ), "Different seeds should produce different noise patterns"


# ==================== Test 3: RFL Feedback Integration ====================

def test_rfl_feedback_conversion():
    """Verify that RFL feedback conversion works correctly."""
    from backend.verification.error_codes import (
        verified_outcome,
        proof_invalid_outcome,
        timeout_outcome,
        spurious_fail_outcome,
        spurious_pass_outcome,
    )
    from backend.verification.error_codes import VerifierTier
    
    # Verified → positive
    outcome1 = verified_outcome(100, VerifierTier.BALANCED)
    assert outcome_to_rfl_feedback(outcome1) == "positive"
    
    # Invalid → negative
    outcome2 = proof_invalid_outcome(100, VerifierTier.BALANCED)
    assert outcome_to_rfl_feedback(outcome2) == "negative"
    
    # Timeout → abstention (None)
    outcome3 = timeout_outcome(100, VerifierTier.BALANCED, noise_injected=True)
    assert outcome_to_rfl_feedback(outcome3) is None
    
    # Spurious fail → negative (with noise flag)
    outcome4 = spurious_fail_outcome(100, VerifierTier.BALANCED)
    assert outcome_to_rfl_feedback(outcome4) == "negative"
    
    # Spurious pass → positive (with noise flag)
    outcome5 = spurious_pass_outcome(100, VerifierTier.BALANCED)
    assert outcome_to_rfl_feedback(outcome5) == "positive"


def test_rfl_policy_update_decision():
    """Verify that RFL policy update decisions are correct."""
    from backend.verification.error_codes import (
        verified_outcome,
        timeout_outcome,
        abstention_outcome,
    )
    from backend.verification.error_codes import VerifierTier
    
    # Verified → should update
    outcome1 = verified_outcome(100, VerifierTier.BALANCED)
    assert should_update_rfl_policy(outcome1)
    
    # Timeout → should NOT update (abstention)
    outcome2 = timeout_outcome(100, VerifierTier.BALANCED, noise_injected=True)
    assert not should_update_rfl_policy(outcome2)
    
    # Abstention → should NOT update
    outcome3 = abstention_outcome(VerifierTier.MOCK, reason="mock_mode")
    assert not should_update_rfl_policy(outcome3)


# ==================== Test 4: Noise Telemetry ====================

def test_noise_telemetry_logging():
    """Verify that noise events are logged with full telemetry."""
    slice_name = "test_slice"
    master_seed = 42
    
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    # Execute multiple items
    items = [f"item_{i}" for i in range(50)]
    
    noise_events = []
    for item in items:
        seed = hash(item) % (2**31)
        success, result = execute_fn(item, seed)
        
        if result.get("noise_injected"):
            noise_events.append({
                "item": item,
                "noise_type": result.get("noise_type"),
                "tier": result.get("tier"),
                "attempt_count": result.get("attempt_count"),
            })
    
    # Should have some noise events
    assert len(noise_events) > 0, "Expected some noise events in 50 items"
    
    # All noise events should have required fields
    for event in noise_events:
        assert "noise_type" in event
        assert "tier" in event
        assert "attempt_count" in event


# ==================== Test 5: Error Handling ====================

def test_execute_fn_error_handling():
    """Verify that execute function handles errors gracefully."""
    slice_name = "test_slice"
    master_seed = 42
    
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=master_seed,
        noise_enabled=True,
        use_escalation=True,
    )
    
    # Execute with various inputs
    test_cases = [
        ("normal_item", 12345),
        ("", 12345),  # Empty item
        ("item_with_special_chars_!@#$%", 12345),
    ]
    
    for item, seed in test_cases:
        try:
            success, result = execute_fn(item, seed)
            # Should not raise exception
            assert "outcome" in result
        except Exception as e:
            pytest.fail(f"Execute function raised exception for {item}: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
