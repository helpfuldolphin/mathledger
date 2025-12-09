"""
Statistical Fitting Functions for Noise Calibration

Implements MLE estimation, confidence intervals, and distribution fitting.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from scipy import stats


def fit_bernoulli_rate(
    n_success: int,
    n_total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """Fit Bernoulli rate with Wilson confidence interval.
    
    Args:
        n_success: Number of successes
        n_total: Total number of trials
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        Tuple of (rate, (lower_ci, upper_ci))
    """
    
    if n_total == 0:
        return 0.0, (0.0, 0.0)
    
    # MLE
    p = n_success / n_total
    
    # Wilson confidence interval
    ci = wilson_confidence_interval(n_success, n_total, confidence_level)
    
    return p, ci


def wilson_confidence_interval(
    n_success: int,
    n_total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion.
    
    Args:
        n_success: Number of successes
        n_total: Total number of trials
        confidence_level: Confidence level
    
    Returns:
        Tuple of (lower, upper)
    """
    
    if n_total == 0:
        return (0.0, 0.0)
    
    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return (lower, upper)


def fit_timeout_distribution(
    durations: List[float],
) -> Dict[str, Any]:
    """Fit timeout duration distribution using AIC model selection.
    
    Tries exponential, gamma, lognormal, and Weibull distributions.
    
    Args:
        durations: List of timeout durations in milliseconds
    
    Returns:
        Dict with best distribution, parameters, and AIC
    """
    
    if not durations:
        return None
    
    durations = np.array(durations)
    
    # Fit candidate distributions
    candidates = []
    
    # Exponential
    try:
        params_exp = stats.expon.fit(durations)
        aic_exp = _compute_aic(durations, stats.expon, params_exp)
        candidates.append(("exponential", params_exp, aic_exp))
    except:
        pass
    
    # Gamma
    try:
        params_gamma = stats.gamma.fit(durations)
        aic_gamma = _compute_aic(durations, stats.gamma, params_gamma)
        candidates.append(("gamma", params_gamma, aic_gamma))
    except:
        pass
    
    # Lognormal
    try:
        params_lognorm = stats.lognorm.fit(durations)
        aic_lognorm = _compute_aic(durations, stats.lognorm, params_lognorm)
        candidates.append(("lognormal", params_lognorm, aic_lognorm))
    except:
        pass
    
    # Weibull
    try:
        params_weibull = stats.weibull_min.fit(durations)
        aic_weibull = _compute_aic(durations, stats.weibull_min, params_weibull)
        candidates.append(("weibull", params_weibull, aic_weibull))
    except:
        pass
    
    if not candidates:
        return None
    
    # Select best by AIC
    best_dist, best_params, best_aic = min(candidates, key=lambda x: x[2])
    
    return {
        "distribution": best_dist,
        "parameters": list(best_params),
        "aic": best_aic,
        "mean": np.mean(durations),
        "std": np.std(durations),
        "median": np.median(durations),
    }


def _compute_aic(
    data: np.ndarray,
    distribution,
    params: Tuple,
) -> float:
    """Compute Akaike Information Criterion (AIC).
    
    Args:
        data: Observed data
        distribution: scipy.stats distribution
        params: Distribution parameters
    
    Returns:
        AIC value
    """
    
    # Log-likelihood
    log_likelihood = np.sum(distribution.logpdf(data, *params))
    
    # Number of parameters
    k = len(params)
    
    # AIC = 2k - 2 ln(L)
    aic = 2 * k - 2 * log_likelihood
    
    return aic
