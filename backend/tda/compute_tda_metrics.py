import json
import time
import uuid
from typing import Dict, Any, List
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# --- Performance Guardrails ---
PERFORMANCE_BUDGET_CYCLE_MS = 5
PERFORMANCE_BUDGET_WINDOW_MS = 50


def timeit(budget_ms: float):
    """Decorator to measure function execution time and log if it exceeds the budget."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            if duration_ms > budget_ms:
                print(f"PERFORMANCE WARNING: {func.__name__} took {duration_ms:.2f}ms (budget: {budget_ms}ms)", file=sys.stderr)
            return result
        return wrapper
    return decorator

# --- Lightweight Placeholder TDA Metrics ---

def _compute_placeholder_pcs(window: np.ndarray) -> float:
    """Placeholder for Path Connectivity Score. Measures normalized path length."""
    if len(window) < 2:
        return 1.0
    distances = np.linalg.norm(np.diff(window, axis=0), axis=1)
    path_length = np.sum(distances)
    # Normalize by the number of steps to keep it stable across window sizes
    normalized_length = path_length / len(window)
    # Invert and scale to be a "score" between 0 and 1, assuming healthy paths have a consistent length.
    return np.exp(-normalized_length / 10.0)

def _compute_placeholder_drs(window: np.ndarray) -> float:
    """Placeholder for Dimensionality Reduction Stability. Measures variance concentration."""
    if len(window) < 3:
        return 1.0
    pca = PCA(n_components=2)
    pca.fit(window)
    # A stable system should have a consistent explained variance ratio.
    # This score is high if the first two components explain most of the variance.
    return float(np.sum(pca.explained_variance_ratio_))

def _compute_placeholder_sns(window: np.ndarray) -> float:
    """Placeholder for State Neighborhood Similarity. Measures local density."""
    if len(window) < 2:
        return 1.0
    # Higher score means points are closer together (more stable neighborhood)
    neighbors = NearestNeighbors(n_neighbors=5).fit(window)
    distances, _ = neighbors.kneighbors(window)
    mean_dist = distances.mean()
    return np.exp(-mean_dist)

def _compute_placeholder_hss(window: np.ndarray) -> float:
    """Placeholder for Homological Scar Score. Detects outliers."""
    if len(window) < 10:
        return 0.0
    # High score means more outliers (potential "scars")
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    # Avoid division by zero for flat dimensions
    std[std == 0] = 1
    z_scores = np.abs((window - mean) / std)
    # Count how many points are more than 3 standard deviations from the mean
    outliers = np.any(z_scores > 3, axis=1)
    return float(np.sum(outliers)) / len(window)


# --- P3: First Light / Baseline Computation ---

@timeit(PERFORMANCE_BUDGET_WINDOW_MS)
def compute_first_light_metrics(usla_window: np.ndarray) -> Dict[str, Any]:
    """
    Computes baseline TDA metrics for the 'First Light' phase from a large window.
    
    Args:
        usla_window: A (N, D) numpy array of normalized USLA state vectors.
    
    Returns:
        A dictionary conforming to the `first_light_tda_metrics.json` schema.
    """
    if not isinstance(usla_window, np.ndarray) or usla_window.ndim != 2:
        raise ValueError("usla_window must be a 2D numpy array.")
    
    pca = PCA(n_components=0.95) # Retain 95% of variance
    pca.fit(usla_window)
    intrinsic_dim = pca.n_components_

    # In a real implementation, this would involve complex topological calculations.
    # We provide plausible placeholder values.
    betti_numbers = [
        1, # B0: Number of connected components
        max(0, int(np.log(intrinsic_dim))) if intrinsic_dim > 1 else 0, # B1: Plausible number of loops
        max(0, int(np.log(intrinsic_dim/2))) if intrinsic_dim > 2 else 0 # B2: Plausible number of voids
    ]

    return {
        "snapshot_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usla_sample_size": usla_window.shape[0],
        "baseline_manifold": {
            "intrinsic_dimensionality": intrinsic_dim,
            "betti_numbers": betti_numbers,
            "persistence_diagram": "placeholder_diagram_data_string"
        }
    }


# --- P4: Real vs. Twin Divergence Computation ---

@timeit(PERFORMANCE_BUDGET_CYCLE_MS)
def _compute_tda_trace(window: np.ndarray) -> Dict[str, float]:
    """Computes all placeholder TDA metrics for a single trajectory window."""
    return {
        "sns": _compute_placeholder_sns(window),
        "pcs": _compute_placeholder_pcs(window),
        "drs": _compute_placeholder_drs(window),
        "hss": _compute_placeholder_hss(window)
    }

def _compute_divergence(trace_real: Dict[str, float], trace_twin: Dict[str, float], sns_history_real: List[float], sns_history_twin: List[float]) -> Dict[str, float]:
    """Computes divergence between two TDA traces."""
    # SNS correlation requires a history. Use numpy's corrcoef for efficiency.
    sns_corr = 0.0
    if len(sns_history_real) >= 2 and len(sns_history_twin) >= 2:
        # Ensure lists are of the same length for correlation
        min_len = min(len(sns_history_real), len(sns_history_twin))
        corr_matrix = np.corrcoef(sns_history_real[-min_len:], sns_history_twin[-min_len:])
        if np.isnan(corr_matrix).any():
             sns_corr = 0.0 # Handle case where variance is zero
        else:
            sns_corr = corr_matrix[0, 1]

    return {
        "sns_correlation": sns_corr,
        "hss_abs_diff": abs(trace_real["hss"] - trace_twin["hss"]),
        "pcs_abs_diff": abs(trace_real["pcs"] - trace_twin["pcs"]),
        "drs_abs_diff": abs(trace_real["drs"] - trace_twin["drs"])
    }

def compute_p4_metrics(
    p4_cycle_id: int, 
    window_real: np.ndarray, 
    window_twin: np.ndarray,
    sns_history_real: List[float],
    sns_history_twin: List[float]
    ) -> Dict[str, Any]:
    """
    Computes P4 TDA metrics comparing a real and twin trajectory window.

    Args:
        p4_cycle_id: The current cycle ID.
        window_real: The (N, D) window of USLA states for the real trajectory.
        window_twin: The (N, D) window of USLA states for the twin trajectory.
        sns_history_real: A list of the recent SNS scores for the real trajectory.
        sns_history_twin: A list of the recent SNS scores for the twin trajectory.

    Returns:
        A dictionary conforming to the `p4_tda_metrics.json` schema.
    """
    trace_real = _compute_tda_trace(window_real)
    trace_twin = _compute_tda_trace(window_twin)
    
    # Append current SNS to history for the next cycle's correlation calculation
    sns_history_real.append(trace_real["sns"])
    sns_history_twin.append(trace_twin["sns"])

    divergence = _compute_divergence(trace_real, trace_twin, sns_history_real, sns_history_twin)

    return {
        "p4_cycle_id": p4_cycle_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "real_trajectory": trace_real,
        "twin_trajectory": trace_twin,
        "divergence": divergence
    }
