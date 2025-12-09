"""
Noise Sampler for Telemetry Runtime

Determines whether to inject noise based on noise configuration and PRNG.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

import hashlib
from typing import Dict, Any, Tuple
from backend.verification.error_codes import VerifierTier
from rfl.prng import DeterministicPRNG, int_to_hex_seed


def should_inject_noise_decision(
    module_name: str,
    context: str,
    master_seed: int,
    noise_config: Dict[str, Any],
    tier: VerifierTier,
) -> Tuple[bool, Dict[str, Any]]:
    """Determine if noise should be injected for this verification.
    
    Uses hierarchical PRNG seeding to ensure deterministic noise injection.
    
    Args:
        module_name: Lean module name
        context: Context string for seeding
        master_seed: Master random seed
        noise_config: Noise configuration dict
        tier: Verifier tier
    
    Returns:
        Tuple of (should_inject, noise_decision_dict)
    """
    
    # Get tier-specific noise rates
    tier_name = tier.value.lower()
    tier_config = noise_config.get(tier_name, {})
    
    timeout_rate = tier_config.get("timeout_rate", 0.0)
    spurious_fail_rate = tier_config.get("spurious_fail_rate", 0.0)
    spurious_pass_rate = tier_config.get("spurious_pass_rate", 0.0)
    
    # Create hierarchical PRNG
    master_prng = DeterministicPRNG(int_to_hex_seed(master_seed))
    context_prng = master_prng.for_path("noise_injection", context, module_name, tier_name)
    
    # Sample noise decision
    noise_type_prng = context_prng.for_path("noise_type")
    u = noise_type_prng.random()
    
    # Determine noise type
    if u < timeout_rate:
        noise_type = "timeout"
    elif u < timeout_rate + spurious_fail_rate:
        noise_type = "spurious_fail"
    elif u < timeout_rate + spurious_fail_rate + spurious_pass_rate:
        noise_type = "spurious_pass"
    else:
        # No noise
        return False, {}
    
    # Package decision
    noise_decision = {
        "noise_type": noise_type,
        "timeout_rate": timeout_rate,
        "spurious_fail_rate": spurious_fail_rate,
        "spurious_pass_rate": spurious_pass_rate,
        "u": u,
        "ground_truth": "UNKNOWN",  # Will be filled in by runtime
    }
    
    return True, noise_decision
