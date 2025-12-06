"""
U2 Integration — Noisy Verifier Integration for U2 Runtime

This module provides integration points for the noisy verifier regime
into the U2 uplift experiment runner. It wraps the execution function
with noise injection and telemetry.

Design Principles:
- Minimal changes to existing U2 runner code
- Drop-in replacement for execute_fn
- Full telemetry for noise injection
- Backward compatible with non-noisy mode

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from rfl.prng import DeterministicPRNG, int_to_hex_seed

from backend.lean_mode import get_build_runner, LeanMode
from backend.verification.config_loader import (
    NoiseConfigLoader,
    load_noise_config_for_tier,
    load_escalation_config,
)
from backend.verification.error_codes import VerifierOutcome, VerifierTier
from backend.verification.noise_sampler import NoiseSampler
from backend.verification.tier_router import VerifierTierRouter, create_tier_router


def create_noisy_execute_fn(
    slice_name: str,
    master_seed: int,
    noise_enabled: bool = True,
    use_escalation: bool = True,
) -> Callable[[str, int], Tuple[bool, Any]]:
    """
    Create execution function with noise injection for U2 runner.
    
    This is a drop-in replacement for create_execute_fn() in run_uplift_u2.py
    that adds deterministic noise injection to verifier calls.
    
    Args:
        slice_name: Name of the experiment slice
        master_seed: Master random seed for noise generation
        noise_enabled: Whether to enable noise injection
        use_escalation: Whether to use tier escalation
    
    Returns:
        Execution function compatible with U2 runner
    """
    # Initialize tier router if noise is enabled
    tier_router: Optional[VerifierTierRouter] = None
    
    if noise_enabled:
        try:
            # Load noise configuration
            config_loader = NoiseConfigLoader()
            escalation_policy, max_attempts = load_escalation_config()
            
            # Create base Lean runner
            base_runner = get_build_runner(LeanMode.MOCK)  # Use MOCK for now
            
            # Create tier router
            tier_router = create_tier_router(
                base_runner=base_runner,
                seed=master_seed,
                escalation_policy=escalation_policy,
                max_escalation_attempts=max_attempts,
            )
            
            print(f"INFO: Noise injection enabled for slice '{slice_name}'")
            print(f"      Escalation policy: {escalation_policy}")
            print(f"      Max attempts: {max_attempts}")
        
        except Exception as e:
            print(f"WARNING: Failed to initialize noise injection: {e}")
            print(f"         Falling back to non-noisy execution")
            tier_router = None
    
    def execute_fn(item: str, seed: int) -> Tuple[bool, Any]:
        """Execute item with optional noise injection."""
        
        # Build context string for noise seeding
        context = f"slice_{slice_name}_item_{item}_seed_{seed}"
        
        # If noise is enabled and tier router is available, use noisy path
        if tier_router is not None:
            return _execute_with_noise(
                item=item,
                seed=seed,
                context=context,
                tier_router=tier_router,
                use_escalation=use_escalation,
            )
        
        # Otherwise, fall back to original non-noisy execution
        return _execute_without_noise(item, seed, slice_name)
    
    return execute_fn


def _execute_with_noise(
    item: str,
    seed: int,
    context: str,
    tier_router: VerifierTierRouter,
    use_escalation: bool,
) -> Tuple[bool, Any]:
    """Execute item with noise injection using tier router.
    
    Args:
        item: Item to verify
        seed: Random seed for this item
        context: Context string for noise seeding
        tier_router: Configured tier router
        use_escalation: Whether to use tier escalation
    
    Returns:
        Tuple of (success, result_dict)
    """
    # For now, use mock module name (in real integration, this would be derived from item)
    module_name = f"Item_{hash(item) % 10000}"
    
    # Verify with or without escalation
    if use_escalation:
        outcome = tier_router.verify_with_escalation(module_name, context)
    else:
        outcome = tier_router.verify_single_tier(module_name, context, VerifierTier.BALANCED)
    
    # Convert outcome to U2-compatible result format
    result = _outcome_to_result_dict(outcome, item)
    
    # Return success flag and result dict
    return outcome.success, result


def _execute_without_noise(
    item: str,
    seed: int,
    slice_name: str,
) -> Tuple[bool, Any]:
    """Execute item without noise injection (original behavior).
    
    This is the fallback path that matches the original create_execute_fn()
    behavior from run_uplift_u2.py.
    
    Args:
        item: Item to verify
        seed: Random seed for this item
        slice_name: Slice name
    
    Returns:
        Tuple of (success, result_dict)
    """
    success = False
    result: Dict[str, Any] = {}
    
    try:
        # Find the run_fo_cycles.py script relative to this module
        script_dir = Path(__file__).parent.parent.parent / "experiments"
        substrate_script = script_dir / "run_fo_cycles.py"
        
        # Check if substrate script exists
        if not substrate_script.exists():
            # Fall back to mock execution for testing
            mock_prng = DeterministicPRNG(int_to_hex_seed(seed))
            mock_rng = mock_prng.for_path("mock_execution", slice_name, str(seed))
            success = mock_rng.random() > 0.5
            result = {"outcome": "VERIFIED" if success else "FAILED", "mock": True}
            return success, result
        
        cmd = [
            sys.executable,
            str(substrate_script),
            "--item",
            item,
            "--seed",
            str(seed),
        ]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            timeout=60,
        )
        
        result = json.loads(proc.stdout)
        if result.get("outcome") == "VERIFIED":
            success = True
    
    except subprocess.TimeoutExpired:
        result = {"error": "timeout", "item": item}
        success = False
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            result = {"error": str(e), "stdout": e.stdout, "stderr": e.stderr}
        else:
            result = {"error": str(e)}
        success = False
    
    return success, result


def _outcome_to_result_dict(outcome: VerifierOutcome, item: str) -> Dict[str, Any]:
    """Convert VerifierOutcome to U2-compatible result dict.
    
    Args:
        outcome: Verifier outcome from noisy wrapper
        item: Original item string
    
    Returns:
        Result dict compatible with U2 runner expectations
    """
    result = {
        "outcome": outcome.error_code.value,
        "success": outcome.success,
        "duration_ms": outcome.duration_ms,
        "tier": outcome.tier.value,
        "noise_injected": outcome.noise_injected,
        "noise_type": outcome.noise_type,
        "attempt_count": outcome.attempt_count,
        "item": item,
    }
    
    # Add metadata
    result.update(outcome.metadata)
    
    return result


# ==================== RFL Feedback Integration ====================

def outcome_to_rfl_feedback(outcome: VerifierOutcome) -> Optional[str]:
    """Convert verifier outcome to RFL feedback signal.
    
    This function implements the RFL feedback policy for noisy verifiers:
    - VERIFIED → positive feedback
    - PROOF_INVALID → negative feedback
    - TIMEOUT → no feedback (abstention)
    - SPURIOUS_FAIL → negative feedback with caution flag
    - SPURIOUS_PASS → positive feedback with caution flag
    
    Args:
        outcome: Verifier outcome
    
    Returns:
        "positive", "negative", or None (abstention)
    """
    return outcome.to_rfl_feedback()


def should_update_rfl_policy(outcome: VerifierOutcome) -> bool:
    """Determine if outcome should trigger RFL policy update.
    
    Policy updates are triggered by:
    - Successful verification (positive feedback)
    - Failed verification (negative feedback)
    
    But NOT by:
    - Timeouts (abstention)
    - Abstentions (mock mode)
    - Resource constraints
    
    Args:
        outcome: Verifier outcome
    
    Returns:
        True if RFL policy should be updated, False otherwise
    """
    # No update on abstention or timeout
    if outcome.is_abstention() or outcome.is_timeout():
        return False
    
    # No update on resource constraints
    if outcome.error_code.is_resource_constraint():
        return False
    
    # Update on success or failure
    return True


def get_rfl_feedback_metadata(outcome: VerifierOutcome) -> Dict[str, Any]:
    """Extract RFL-relevant metadata from verifier outcome.
    
    Args:
        outcome: Verifier outcome
    
    Returns:
        Metadata dict for RFL logging
    """
    return {
        "error_code": outcome.error_code.value,
        "tier": outcome.tier.value,
        "noise_injected": outcome.noise_injected,
        "noise_type": outcome.noise_type,
        "attempt_count": outcome.attempt_count,
        "duration_ms": outcome.duration_ms,
    }
