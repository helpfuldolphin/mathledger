import unittest
import json
import subprocess
import sys
import os

class TestTDABenchmarkHarness(unittest.TestCase):

    def test_benchmark_script_runs_and_produces_valid_json(self):
        """
        Tests that the benchmark script executes and its output is a JSON object
        with the expected fields. Does not validate timings.
        """
        # Get the path to the benchmark script
        script_path = os.path.join(
            os.path.dirname(__file__), 
            'bench_tda_cycle.py'
        )

        # Run the script as a subprocess to capture its stdout
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True # Raise an exception if the script fails
        )

        # Parse the output
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.fail(f"Benchmark script did not produce valid JSON. Output:\n{result.stdout}")

        # Validate schema/fields
        expected_keys = ["cold_ms", "hot_p50_ms", "hot_p95_ms", "hot_max_ms"]
        self.assertEqual(list(output_data.keys()), expected_keys)

        # Validate types
        for key, value in output_data.items():
            self.assertIsInstance(value, (int, float), f"Value for '{key}' is not a number.")
            self.assertGreaterEqual(value, 0, f"Value for '{key}' should be non-negative.")

if __name__ == '__main__':
    unittest.main()

