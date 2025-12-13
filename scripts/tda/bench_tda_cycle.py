import json
import time
import numpy as np
import sys
import os

# Add the project root to the Python path to allow importing from 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.tda.compute_tda_metrics import compute_p4_metrics

# --- Benchmark Parameters ---
HOT_ITERATIONS = 100
WINDOW_SIZE = 100
STATE_DIM = 50
RNG_SEED = 42

def generate_fixed_input(num_points: int, dim: int, seed: int) -> np.ndarray:
    """Generates a deterministic, fixed trajectory for benchmarking."""
    rng = np.random.default_rng(seed)
    return rng.random((num_points, dim))

def run_benchmark():
    """
    Runs a benchmark to measure cold and hot execution times for the P4 TDA computation.
    """
    # --- Data Setup ---
    window_real = generate_fixed_input(WINDOW_SIZE, STATE_DIM, RNG_SEED)
    window_twin = generate_fixed_input(WINDOW_SIZE, STATE_DIM, RNG_SEED + 1)
    
    # --- Cold Start ---
    start_time = time.perf_counter()
    # Histories are empty for the very first run
    compute_p4_metrics(1, window_real, window_twin, [], [])
    end_time = time.perf_counter()
    cold_ms = (end_time - start_time) * 1000

    # --- Hot Runs ---
    hot_timings_ms = []
    # Prime the histories so correlation doesn't start from scratch every time
    sns_history_real = [0.9] * 10
    sns_history_twin = [0.9] * 10
    
    for i in range(HOT_ITERATIONS):
        start_time = time.perf_counter()
        compute_p4_metrics(i + 2, window_real, window_twin, sns_history_real, sns_history_twin)
        end_time = time.perf_counter()
        hot_timings_ms.append((end_time - start_time) * 1000)
    
    # --- Results ---
    hot_p50_ms = np.percentile(hot_timings_ms, 50)
    hot_p95_ms = np.percentile(hot_timings_ms, 95)
    hot_max_ms = max(hot_timings_ms)

    results = {
        "cold_ms": cold_ms,
        "hot_p50_ms": hot_p50_ms,
        "hot_p95_ms": hot_p95_ms,
        "hot_max_ms": hot_max_ms
    }

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()
