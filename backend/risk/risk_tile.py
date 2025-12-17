"""Core logic for calculating risk metrics and generating risk tiles."""

from typing import Dict, Union

def normalize_metric(value: float, threshold: float, operator: str) -> float:
    """
    Normalizes a metric's value to a risk score between 0 (no risk) and 1 (max risk).

    Args:
        value: The measured value of the metric.
        threshold: The gating threshold for the metric.
        operator: The comparison operator (e.g., '>', '<').

    Returns:
        A normalized risk score.
    """
    if operator in ('>', '>='):
        # Higher values are better, so risk increases as value falls below threshold
        if threshold == 0: return 1.0 # Avoid division by zero
        score = 1 - (value / threshold)
    elif operator in ('<', '<='):
        # Lower values are better, so risk increases as value exceeds threshold
        if threshold == 0: return 1.0 if value > 0 else 0.0
        score = (value / threshold) - 1
    else:
        # For '==', any deviation is considered max risk.
        # This is a simplification; a more nuanced approach might use a tolerance.
        return 0.0 if value == threshold else 1.0

    return max(0, min(1, score)) # Clamp the score between 0 and 1

def compute_overall_risk(normalized_scores: Dict[str, float]) -> float:
    """
    Computes the overall risk score from a dictionary of normalized metric scores.

    Args:
        normalized_scores: A dictionary mapping metric IDs to their normalized scores.

    Returns:
        The overall weighted risk score.
    """
    weights = {
        'delta_p': 0.15,
        'rsi': 0.30,
        'omega': 0.25,
        'tda': 0.10,
        'divergence': 0.20,
    }
    
    overall_risk = 0.0
    total_weight = 0.0

    for metric_id, score in normalized_scores.items():
        weight = weights.get(metric_id)
        if weight is not None:
            overall_risk += score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return overall_risk / total_weight


def map_to_band(overall_risk: float) -> str:
    """
    Maps an overall risk score to a qualitative risk band.

    Args:
        overall_risk: The calculated overall risk score.

    Returns:
        The corresponding risk band ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    """
    if overall_risk < 0.10:
        return 'LOW'
    if 0.10 <= overall_risk < 0.40:
        return 'MEDIUM'
    if 0.40 <= overall_risk < 0.70:
        return 'HIGH'
    return 'CRITICAL'
