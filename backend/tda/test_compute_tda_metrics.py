import unittest
import json
import jsonschema
import numpy as np
import os

from backend.tda.compute_tda_metrics import (
    compute_first_light_metrics,
    compute_p4_metrics
)

# --- Constants and Test Data Generation ---

# Get the directory of the current test file to locate schemas
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_DIR = os.path.join(TEST_DIR, '..', '..') # Assumes schemas are in the project root

# Parameters for synthetic data
WINDOW_SIZE_CYCLE = 100
WINDOW_SIZE_WINDOW = 1000
STATE_DIM = 50
RNG_SEED = 42


def load_schema(name: str):
    """Loads a JSON schema from the project root."""
    path = os.path.join(SCHEMAS_DIR, name)
    with open(path, 'r') as f:
        return json.load(f)

def generate_trajectory(num_points: int, anomaly: bool = False, seed: int = RNG_SEED) -> np.ndarray:
    """Generates a synthetic USLA state trajectory."""
    rng = np.random.default_rng(seed)
    # Start with a slowly changing base signal (sine wave)
    base_signal = np.sin(np.linspace(0, 10 * np.pi, num_points))
    trajectory = np.zeros((num_points, STATE_DIM))
    for i in range(STATE_DIM):
        trajectory[:, i] = base_signal + rng.normal(0, 0.1, num_points)

    if anomaly:
        # Introduce a sharp, transient deviation in a few dimensions
        anomaly_start = num_points // 2
        anomaly_len = 10
        trajectory[anomaly_start:anomaly_start+anomaly_len, :5] += rng.normal(0, 2.0, (anomaly_len, 5))
    
    return trajectory


class TestTDAMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load schemas once for all tests."""
        cls.first_light_schema = load_schema('first_light_tda_metrics.json')
        cls.p4_schema = load_schema('p4_tda_metrics.json')

    def test_first_light_schema_conformance(self):
        """Verify that the First Light output conforms to its JSON schema."""
        usla_window = generate_trajectory(WINDOW_SIZE_WINDOW)
        metrics = compute_first_light_metrics(usla_window)

        # This will raise an exception if validation fails
        jsonschema.validate(instance=metrics, schema=self.first_light_schema)
        self.assertIn("baseline_manifold", metrics)
        self.assertEqual(metrics["usla_sample_size"], WINDOW_SIZE_WINDOW)

    def test_p4_schema_conformance(self):
        """Verify that the P4 output conforms to its JSON schema."""
        window_real = generate_trajectory(WINDOW_SIZE_CYCLE)
        window_twin = generate_trajectory(WINDOW_SIZE_CYCLE) # Identical twin for this test
        metrics = compute_p4_metrics(12345, window_real, window_twin, [0.9, 0.91], [0.9, 0.91])

        jsonschema.validate(instance=metrics, schema=self.p4_schema)
        self.assertIn("divergence", metrics)
        self.assertEqual(metrics["p4_cycle_id"], 12345)

    def test_determinism_with_seed(self):
        """Ensure identical inputs produce identical outputs."""
        # P3 First Light
        usla_window_1 = generate_trajectory(WINDOW_SIZE_WINDOW, seed=RNG_SEED)
        usla_window_2 = generate_trajectory(WINDOW_SIZE_WINDOW, seed=RNG_SEED)
        metrics_1 = compute_first_light_metrics(usla_window_1)
        metrics_2 = compute_first_light_metrics(usla_window_2)
        # Compare relevant fields, ignoring UUID and timestamp
        self.assertEqual(metrics_1["baseline_manifold"], metrics_2["baseline_manifold"])

        # P4
        window_real_1 = generate_trajectory(WINDOW_SIZE_CYCLE, seed=RNG_SEED)
        window_twin_1 = generate_trajectory(WINDOW_SIZE_CYCLE, seed=RNG_SEED+1)
        window_real_2 = generate_trajectory(WINDOW_SIZE_CYCLE, seed=RNG_SEED)
        window_twin_2 = generate_trajectory(WINDOW_SIZE_CYCLE, seed=RNG_SEED+1)
        
        metrics_p4_1 = compute_p4_metrics(1, window_real_1, window_twin_1, [], [])
        metrics_p4_2 = compute_p4_metrics(1, window_real_2, window_twin_2, [], [])
        # Compare all but timestamp
        del metrics_p4_1["timestamp"]
        del metrics_p4_2["timestamp"]
        self.assertDictEqual(metrics_p4_1, metrics_p4_2)

    def test_known_anomaly_increases_divergence(self):
        """
        Verify that a synthetic anomaly in one trajectory significantly increases divergence metrics.
        This is the most critical test, acting as the harness.
        """
        # --- Simulation Setup ---
        sns_history_real_healthy, sns_history_twin_healthy = [], []
        sns_history_real_anomaly, sns_history_twin_anomaly = [], []
        
        # --- Run 1: Healthy vs Healthy ---
        # Generate two identical long trajectories
        traj_healthy_1 = generate_trajectory(200, seed=1)
        traj_healthy_2 = generate_trajectory(200, seed=1)
        
        window_real_h = traj_healthy_1[0:WINDOW_SIZE_CYCLE]
        window_twin_h = traj_healthy_2[0:WINDOW_SIZE_CYCLE]
        
        metrics_healthy = compute_p4_metrics(1, window_real_h, window_twin_h, sns_history_real_healthy, sns_history_twin_healthy)
        
        # --- Run 2: Healthy vs Anomaly ---
        # Generate one healthy and one anomalous trajectory
        traj_healthy = generate_trajectory(200, seed=1)
        traj_anomaly = generate_trajectory(200, anomaly=True, seed=1) # Same seed, but with anomaly injected
        
        # The window captures the anomaly
        anomaly_start_index = (200 // 2) - (WINDOW_SIZE_CYCLE // 2)
        window_real_a = traj_healthy[anomaly_start_index : anomaly_start_index + WINDOW_SIZE_CYCLE]
        window_twin_a = traj_anomaly[anomaly_start_index : anomaly_start_index + WINDOW_SIZE_CYCLE]
        
        metrics_anomaly = compute_p4_metrics(2, window_real_a, window_twin_a, sns_history_real_anomaly, sns_history_twin_anomaly)

        # --- Assertions ---
        # Healthy divergence should be near zero
        self.assertAlmostEqual(metrics_healthy["divergence"]["hss_abs_diff"], 0.0, places=5)
        self.assertAlmostEqual(metrics_healthy["divergence"]["pcs_abs_diff"], 0.0, places=5)

        # Anomaly divergence should be significantly higher
        print(f"Healthy HSS Diff: {metrics_healthy['divergence']['hss_abs_diff']:.4f}")
        print(f"Anomaly HSS Diff: {metrics_anomaly['divergence']['hss_abs_diff']:.4f}")
        self.assertGreater(metrics_anomaly["divergence"]["hss_abs_diff"], metrics_healthy["divergence"]["hss_abs_diff"])
        
        # The HSS (outlier score) should be the most prominent indicator
        self.assertGreater(metrics_anomaly["divergence"]["hss_abs_diff"], 0.01)


if __name__ == '__main__':
    unittest.main()
