"""
RFL Integration with Noise-Robust Updates

Implements noise-robust RFL policy updates that account for verifier epistemic uncertainty.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from typing import Dict, Any
from backend.verification.error_codes import VerifierErrorCode


def compute_expected_value(
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
) -> float:
    """Compute expected value under verifier epistemic uncertainty.
    
    Uses Bayes' theorem to compute posterior probability of ground truth
    validity given verifier outcome and noise rates.
    
    Mathematical Formula:
        V_expected = 2 * P(y = VALID | o) - 1
    
    where P(y = VALID | o) is computed via Bayes' theorem:
        P(y = VALID | o = VERIFIED) = (1 - θ_sf - θ_t) / ((1 - θ_sf - θ_t) + θ_sp)
        P(y = VALID | o = FAILED) = θ_sf / (θ_sf + (1 - θ_sp - θ_t))
        P(y = VALID | o = TIMEOUT) = 0.5 (no information)
    
    Args:
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
    
    Returns:
        Expected value in [-1, 1]
    """
    
    if outcome == VerifierErrorCode.VERIFIED:
        # P(y = VALID | o = VERIFIED)
        numerator = 1 - theta_spurious_fail - theta_timeout
        denominator = numerator + theta_spurious_pass
        
        if denominator == 0:
            return 0.0
        
        p_valid_given_verified = numerator / denominator
        return 2 * p_valid_given_verified - 1
    
    elif outcome == VerifierErrorCode.PROOF_INVALID:
        # P(y = VALID | o = FAILED)
        numerator = theta_spurious_fail
        denominator = theta_spurious_fail + (1 - theta_spurious_pass - theta_timeout)
        
        if denominator == 0:
            return 0.0
        
        p_valid_given_failed = numerator / denominator
        return 2 * p_valid_given_failed - 1
    
    elif outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        # No information from timeout
        return 0.0
    
    else:
        # Other outcomes (errors, abstentions) treated as no information
        return 0.0


def apply_bias_correction(
    value_expected: float,
    theta_spurious_pass: float,
) -> float:
    """Apply bias correction to ensure RFL-stability.
    
    Corrects for bias introduced by spurious passes.
    
    Mathematical Formula:
        V_corrected = V_expected / (1 - 2θ_sp)
    
    Args:
        value_expected: Expected value before correction
        theta_spurious_pass: Spurious pass rate
    
    Returns:
        Bias-corrected value
    """
    
    # Avoid division by zero
    if theta_spurious_pass >= 0.5:
        # Degenerate case: too much noise
        return value_expected
    
    # Bias correction factor
    correction_factor = 1 - 2 * theta_spurious_pass
    
    # Apply correction
    value_corrected = value_expected / correction_factor
    
    # Clamp to [-1, 1]
    value_corrected = max(-1.0, min(1.0, value_corrected))
    
    return value_corrected


def update_rfl_policy_noisy(
    policy_weights: Dict[str, float],
    item: str,
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
    learning_rate: float,
) -> Dict[str, float]:
    """Update RFL policy with noise-robust update.
    
    Args:
        policy_weights: Current policy weights
        item: Item that was verified
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
        learning_rate: Learning rate
    
    Returns:
        Updated policy weights
    """
    
    # Compute expected value
    value_expected = compute_expected_value(
        outcome=outcome,
        theta_spurious_fail=theta_spurious_fail,
        theta_spurious_pass=theta_spurious_pass,
        theta_timeout=theta_timeout,
    )
    
    # Apply bias correction
    value_corrected = apply_bias_correction(
        value_expected=value_expected,
        theta_spurious_pass=theta_spurious_pass,
    )
    
    # Abstention handling: skip update if timeout
    if outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        return policy_weights  # No update
    
    # Compute gradient (simplified: assume log-linear policy)
    grad = value_corrected  # ∇log π(item) ≈ 1 for log-linear
    
    # Update weight
    new_weights = policy_weights.copy()
    new_weights[item] = new_weights.get(item, 0.0) + learning_rate * grad
    
    return new_weights
