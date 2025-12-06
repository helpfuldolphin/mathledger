"""
Verifier Tier Router — Mixed-Verifier Routing with Escalation

This module implements tier-based routing for verifier calls with automatic
escalation on failure. Different tiers trade off speed vs. accuracy:
- FAST_NOISY: High noise, low latency
- BALANCED: Medium noise, medium latency
- SLOW_PRECISE: Low noise, high latency

Design Principles:
- Automatic escalation on timeout/failure
- Configurable escalation policies
- Full telemetry for tier transitions
- Deterministic behavior (seeded noise per tier)

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

from backend.verification.error_codes import VerifierOutcome, VerifierTier
from backend.verification.noise_sampler import NoiseSampler, create_noise_sampler
from backend.verification.noisy_lean_wrapper import (
    BuildRunner,
    NoisyLeanWrapper,
    create_noisy_lean_wrapper,
)


class EscalationPolicy:
    """Policy for tier escalation on verification failure.
    
    Escalation policies determine when and how to escalate from one tier
    to the next after a verification failure.
    """
    
    ON_FAILURE = "on_failure"
    """Escalate on any verification failure."""
    
    ON_TIMEOUT = "on_timeout"
    """Escalate only on timeout."""
    
    ON_NOISE = "on_noise"
    """Escalate only on noise-injected failures."""
    
    NEVER = "never"
    """Never escalate (single-tier only)."""
    
    ALWAYS = "always"
    """Always escalate regardless of outcome (for testing)."""


class VerifierTierRouter:
    """Routes verification requests to appropriate tier with escalation.
    
    This class manages multiple verifier tiers and implements automatic
    escalation policies. Each tier has its own noise sampler and wrapper.
    
    Usage:
        router = VerifierTierRouter(base_runner, seed=42)
        outcome = router.verify_with_escalation("MyModule", "cycle_1_item_3")
    """
    
    def __init__(
        self,
        base_runner: BuildRunner,
        seed: int,
        escalation_policy: str = EscalationPolicy.ON_FAILURE,
        max_escalation_attempts: int = 3,
        tier_configs: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize tier router.
        
        Args:
            base_runner: Base Lean build runner (shared across tiers)
            seed: Master seed for noise generation
            escalation_policy: Policy for tier escalation
            max_escalation_attempts: Maximum number of escalation attempts
            tier_configs: Optional tier-specific configurations
        """
        self.base_runner = base_runner
        self.seed = seed
        self.escalation_policy = escalation_policy
        self.max_escalation_attempts = max_escalation_attempts
        
        # Initialize tier wrappers
        self.tiers = self._init_tiers(tier_configs or {})
        
        # Default tier sequence for escalation
        self.tier_sequence = [
            VerifierTier.FAST_NOISY,
            VerifierTier.BALANCED,
            VerifierTier.SLOW_PRECISE,
        ]
    
    def _init_tiers(self, tier_configs: Dict[str, Dict]) -> Dict[VerifierTier, NoisyLeanWrapper]:
        """Initialize verifier wrappers for each tier.
        
        Args:
            tier_configs: Tier-specific configuration overrides
        
        Returns:
            Dict mapping VerifierTier to NoisyLeanWrapper
        """
        tiers = {}
        
        for tier in [VerifierTier.FAST_NOISY, VerifierTier.BALANCED, VerifierTier.SLOW_PRECISE]:
            # Create noise sampler for this tier
            # Use tier name as part of seed derivation for independence
            tier_seed = hash((self.seed, tier.value)) % (2**31)
            sampler = create_noise_sampler(tier.value, tier_seed)
            
            # Create noisy wrapper
            wrapper = create_noisy_lean_wrapper(self.base_runner, tier, sampler)
            tiers[tier] = wrapper
        
        return tiers
    
    def _get_timeout_for_tier(self, tier: VerifierTier) -> int:
        """Get timeout in seconds for a given tier.
        
        Args:
            tier: Verifier tier
        
        Returns:
            Timeout in seconds
        """
        timeout_map = {
            VerifierTier.FAST_NOISY: 30,
            VerifierTier.BALANCED: 60,
            VerifierTier.SLOW_PRECISE: 120,
            VerifierTier.MOCK: 10,
        }
        return timeout_map.get(tier, 90)
    
    def _should_escalate(self, outcome: VerifierOutcome) -> bool:
        """Determine if outcome should trigger escalation.
        
        Args:
            outcome: Verifier outcome from current tier
        
        Returns:
            True if escalation should occur, False otherwise
        """
        if self.escalation_policy == EscalationPolicy.NEVER:
            return False
        
        if self.escalation_policy == EscalationPolicy.ALWAYS:
            return True
        
        if self.escalation_policy == EscalationPolicy.ON_FAILURE:
            return not outcome.success
        
        if self.escalation_policy == EscalationPolicy.ON_TIMEOUT:
            return outcome.is_timeout()
        
        if self.escalation_policy == EscalationPolicy.ON_NOISE:
            return outcome.noise_injected and not outcome.success
        
        # Default: escalate on failure
        return not outcome.success
    
    def verify_with_escalation(
        self,
        module_name: str,
        context: str,
        initial_tier: Optional[VerifierTier] = None,
        max_attempts: Optional[int] = None,
    ) -> VerifierOutcome:
        """Verify with automatic tier escalation on failure.
        
        This method implements the escalation pipeline:
        1. Start with initial tier (default: FAST_NOISY)
        2. If failure and escalation policy satisfied, escalate to next tier
        3. Repeat until success or max attempts reached
        4. Return final outcome with attempt count
        
        Args:
            module_name: Lean module name to verify
            context: Unique context string for noise seeding
            initial_tier: Initial tier to start with (default: FAST_NOISY)
            max_attempts: Override for max escalation attempts
        
        Returns:
            Final VerifierOutcome after escalation
        """
        if initial_tier is None:
            initial_tier = VerifierTier.FAST_NOISY
        
        if max_attempts is None:
            max_attempts = self.max_escalation_attempts
        
        # Build tier sequence starting from initial tier
        try:
            start_idx = self.tier_sequence.index(initial_tier)
        except ValueError:
            start_idx = 0
        
        tier_sequence = self.tier_sequence[start_idx:start_idx + max_attempts]
        
        # Escalation loop
        for attempt, tier in enumerate(tier_sequence, start=1):
            wrapper = self.tiers[tier]
            timeout = self._get_timeout_for_tier(tier)
            
            # Add attempt number to context for unique noise per attempt
            attempt_context = f"{context}_attempt_{attempt}"
            
            outcome = wrapper.verify(module_name, attempt_context, timeout)
            
            # Update attempt count in outcome
            outcome = dataclasses.replace(outcome, attempt_count=attempt)
            
            # Add escalation metadata
            metadata = dict(outcome.metadata)
            metadata["tier"] = tier.value
            metadata["escalation_attempt"] = attempt
            metadata["max_attempts"] = max_attempts
            outcome = dataclasses.replace(outcome, metadata=metadata)
            
            # Check if we should escalate
            if outcome.success or attempt == max_attempts:
                # Success or final attempt, return outcome
                return outcome
            
            if not self._should_escalate(outcome):
                # Policy says don't escalate, return outcome
                return outcome
            
            # Log escalation (for telemetry)
            print(f"INFO: Escalating from {tier.value} to next tier after {outcome.error_code.value}")
        
        # Should not reach here, but return last outcome as fallback
        return outcome
    
    def verify_single_tier(
        self,
        module_name: str,
        context: str,
        tier: VerifierTier,
    ) -> VerifierOutcome:
        """Verify using a single tier without escalation.
        
        Args:
            module_name: Lean module name to verify
            context: Unique context string for noise seeding
            tier: Specific tier to use
        
        Returns:
            VerifierOutcome from specified tier
        """
        wrapper = self.tiers[tier]
        timeout = self._get_timeout_for_tier(tier)
        return wrapper.verify(module_name, context, timeout)
    
    def verify_batch(
        self,
        module_names: list[str],
        context_prefix: str,
        use_escalation: bool = True,
    ) -> list[VerifierOutcome]:
        """Verify a batch of modules with optional escalation.
        
        Args:
            module_names: List of Lean module names to verify
            context_prefix: Prefix for context strings (e.g., "cycle_1")
            use_escalation: Whether to use escalation (default: True)
        
        Returns:
            List of VerifierOutcomes, one per module
        """
        outcomes = []
        for i, module_name in enumerate(module_names):
            context = f"{context_prefix}_item_{i}"
            if use_escalation:
                outcome = self.verify_with_escalation(module_name, context)
            else:
                outcome = self.verify_single_tier(module_name, context, VerifierTier.BALANCED)
            outcomes.append(outcome)
        return outcomes


# ==================== Router Factory ====================

def create_tier_router(
    base_runner: BuildRunner,
    seed: int,
    escalation_policy: str = EscalationPolicy.ON_FAILURE,
    max_escalation_attempts: int = 3,
) -> VerifierTierRouter:
    """Factory function to create a tier router.
    
    Args:
        base_runner: Base Lean build runner
        seed: Master seed for noise generation
        escalation_policy: Policy for tier escalation
        max_escalation_attempts: Maximum number of escalation attempts
    
    Returns:
        VerifierTierRouter instance
    """
    return VerifierTierRouter(
        base_runner=base_runner,
        seed=seed,
        escalation_policy=escalation_policy,
        max_escalation_attempts=max_escalation_attempts,
    )
