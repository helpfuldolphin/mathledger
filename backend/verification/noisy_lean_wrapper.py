"""
Noisy Lean Wrapper — Deterministic Noise Injection for Lean Verifier

This module wraps the Lean verifier with deterministic noise injection,
implementing the imperfect verifier regime for Phase II. All noise is
seeded and reproducible.

Design Principles:
- Wraps existing Lean build runners (mock, dry_run, full)
- Injects noise deterministically based on NoiseSampler
- Returns VerifierOutcome with full telemetry
- Preserves ground truth in metadata for analysis

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

import subprocess
import time
from typing import Callable, Optional

from backend.verification.error_codes import (
    VerifierErrorCode,
    VerifierOutcome,
    VerifierTier,
    verified_outcome,
    proof_invalid_outcome,
    timeout_outcome,
    spurious_fail_outcome,
    spurious_pass_outcome,
    abstention_outcome,
)
from backend.verification.noise_sampler import NoiseSampler

# Type alias for build runner functions
BuildRunner = Callable[[str], subprocess.CompletedProcess[str]]


class NoisyLeanWrapper:
    """Wraps Lean verifier with deterministic noise injection.
    
    This class implements the imperfect verifier regime by:
    1. Running the base Lean verifier to get ground truth
    2. Deterministically injecting noise based on NoiseSampler
    3. Returning VerifierOutcome with full telemetry
    
    Usage:
        from backend.lean_mode import get_build_runner, LeanMode
        
        base_runner = get_build_runner(LeanMode.FULL)
        sampler = NoiseSampler(config, seed=42)
        wrapper = NoisyLeanWrapper(base_runner, sampler, VerifierTier.BALANCED)
        
        outcome = wrapper.verify("MyModule", "cycle_1_item_3", timeout=60)
    """
    
    def __init__(
        self,
        base_runner: BuildRunner,
        noise_sampler: NoiseSampler,
        tier: VerifierTier,
    ):
        """Initialize noisy wrapper.
        
        Args:
            base_runner: Base Lean build runner (from lean_mode.py)
            noise_sampler: Deterministic noise sampler
            tier: Verifier tier for this wrapper
        """
        self.base_runner = base_runner
        self.noise_sampler = noise_sampler
        self.tier = tier
    
    def verify(
        self,
        module_name: str,
        context: str,
        timeout: int = 90,
    ) -> VerifierOutcome:
        """Execute verification with deterministic noise injection.
        
        This method implements the full noise injection pipeline:
        1. Check for timeout injection (pre-execution)
        2. Run base verifier to get ground truth
        3. Check for spurious failure/pass injection (post-execution)
        4. Return VerifierOutcome with full telemetry
        
        Args:
            module_name: Lean module name to verify
            context: Unique context string for noise seeding (e.g., "cycle_1_item_3")
            timeout: Timeout in seconds (not currently enforced, for future use)
        
        Returns:
            VerifierOutcome with error code, success flag, and telemetry
        """
        start_time = time.time()
        
        # ==================== Phase 1: Check for Timeout Injection ====================
        
        if self.noise_sampler.should_timeout(context):
            # Inject timeout: simulate timeout duration and return early
            timeout_duration = self.noise_sampler.sample_timeout_duration(context)
            time.sleep(timeout_duration)
            
            duration_ms = (time.time() - start_time) * 1000
            return timeout_outcome(
                duration_ms=duration_ms,
                tier=self.tier,
                attempt_count=1,
                noise_injected=True,
                metadata={
                    "simulated_timeout_s": timeout_duration,
                    "context": context,
                    "module_name": module_name,
                },
            )
        
        # ==================== Phase 2: Run Base Verifier ====================
        
        try:
            result = self.base_runner(module_name)
        except Exception as e:
            # Internal error in verifier
            duration_ms = (time.time() - start_time) * 1000
            return VerifierOutcome(
                error_code=VerifierErrorCode.VERIFIER_INTERNAL_ERROR,
                success=False,
                duration_ms=duration_ms,
                tier=self.tier,
                noise_injected=False,
                noise_type=None,
                attempt_count=1,
                metadata={
                    "error": str(e),
                    "context": context,
                    "module_name": module_name,
                },
            )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # ==================== Phase 3: Determine Ground Truth ====================
        
        ground_truth_success = (result.returncode == 0)
        
        # Check for abstention signatures (mock mode)
        if "LEAN_MOCK_ABSTAIN" in result.stderr:
            return abstention_outcome(
                tier=self.tier,
                reason="mock_mode",
                metadata={
                    "context": context,
                    "module_name": module_name,
                    "stdout": result.stdout[:200],
                    "stderr": result.stderr[:200],
                },
            )
        
        # ==================== Phase 4: Check for Spurious Failures ====================
        
        if ground_truth_success and self.noise_sampler.should_spurious_fail(context):
            # Inject spurious failure: ground truth is VERIFIED, but we report FAIL
            return spurious_fail_outcome(
                duration_ms=duration_ms,
                tier=self.tier,
                attempt_count=1,
                metadata={
                    "ground_truth": "VERIFIED",
                    "context": context,
                    "module_name": module_name,
                    "returncode": result.returncode,
                },
            )
        
        # ==================== Phase 5: Check for Spurious Passes ====================
        
        if not ground_truth_success and self.noise_sampler.should_spurious_pass(context):
            # Inject spurious pass: ground truth is FAILED, but we report SUCCESS
            return spurious_pass_outcome(
                duration_ms=duration_ms,
                tier=self.tier,
                attempt_count=1,
                metadata={
                    "ground_truth": "FAILED",
                    "context": context,
                    "module_name": module_name,
                    "returncode": result.returncode,
                    "stderr": result.stderr[:200],
                },
            )
        
        # ==================== Phase 6: No Noise, Return Ground Truth ====================
        
        if ground_truth_success:
            return verified_outcome(
                duration_ms=duration_ms,
                tier=self.tier,
                attempt_count=1,
                metadata={
                    "context": context,
                    "module_name": module_name,
                    "returncode": result.returncode,
                },
            )
        else:
            return proof_invalid_outcome(
                duration_ms=duration_ms,
                tier=self.tier,
                attempt_count=1,
                metadata={
                    "context": context,
                    "module_name": module_name,
                    "returncode": result.returncode,
                    "stderr": result.stderr[:500],
                },
            )
    
    def verify_batch(
        self,
        module_names: list[str],
        context_prefix: str,
        timeout: int = 90,
    ) -> list[VerifierOutcome]:
        """Verify a batch of modules with noise injection.
        
        Args:
            module_names: List of Lean module names to verify
            context_prefix: Prefix for context strings (e.g., "cycle_1")
            timeout: Timeout in seconds per module
        
        Returns:
            List of VerifierOutcomes, one per module
        """
        outcomes = []
        for i, module_name in enumerate(module_names):
            context = f"{context_prefix}_item_{i}"
            outcome = self.verify(module_name, context, timeout)
            outcomes.append(outcome)
        return outcomes


# ==================== Wrapper Factory ====================

def create_noisy_lean_wrapper(
    base_runner: BuildRunner,
    tier: VerifierTier,
    noise_sampler: NoiseSampler,
) -> NoisyLeanWrapper:
    """Factory function to create a noisy Lean wrapper.
    
    Args:
        base_runner: Base Lean build runner
        tier: Verifier tier
        noise_sampler: Configured noise sampler
    
    Returns:
        NoisyLeanWrapper instance
    """
    return NoisyLeanWrapper(base_runner, noise_sampler, tier)
