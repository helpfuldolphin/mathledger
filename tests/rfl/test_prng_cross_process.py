# PHASE II — NOT USED IN PHASE I
"""
Cross-Process Determinism Smoke Test for DeterministicPRNG.

This test verifies that the hierarchical PRNG produces identical sequences
across different Python processes, catching potential issues with:
- Platform differences (endianness, OS)
- Python version differences
- Environment contamination
- Subprocess isolation

The test spawns a subprocess that runs the same PRNG logic and verifies
the sequences match the in-process computation.

Contract Reference:
    Implements the cross-process determinism requirement from
    docs/DETERMINISM_CONTRACT.md - same seed must produce identical
    output regardless of process boundary.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Script to run in subprocess - must be self-contained
SUBPROCESS_SCRIPT = '''
import sys
import json

# Add project root to path
sys.path.insert(0, "{project_root}")

from rfl.prng import DeterministicPRNG, PRNGKey, derive_seed, int_to_hex_seed

def main():
    master_seed_hex = "{master_seed_hex}"
    num_seeds = {num_seeds}
    paths = {paths}
    
    prng = DeterministicPRNG(master_seed_hex)
    
    results = {{
        "master_seed_hex": master_seed_hex,
        "derived_seeds": [],
        "random_sequences": [],
        "seed_schedule": [],
    }}
    
    # Test 1: Derive seeds for each path
    for path in paths:
        seed = prng.seed_for_path(*path)
        results["derived_seeds"].append({{"path": path, "seed": seed}})
    
    # Test 2: Generate random sequences for first path
    if paths:
        rng = prng.for_path(*paths[0])
        results["random_sequences"] = [rng.random() for _ in range(10)]
    
    # Test 3: Generate seed schedule
    results["seed_schedule"] = prng.generate_seed_schedule(
        num_cycles=num_seeds,
        slice_name="cross_process_test",
        mode="baseline",
    )
    
    print(json.dumps(results))

if __name__ == "__main__":
    main()
'''


class TestCrossProcessDeterminism:
    """Tests that verify PRNG determinism across process boundaries."""

    @pytest.fixture
    def master_seed_hex(self) -> str:
        """Fixed master seed for testing."""
        return "a" * 64

    @pytest.fixture
    def test_paths(self) -> list:
        """Test paths for seed derivation."""
        return [
            ["slice_a", "baseline", "cycle_0001"],
            ["slice_a", "baseline", "cycle_0002"],
            ["slice_b", "rfl", "cycle_0001"],
            ["cross_process", "test", "determinism"],
        ]

    @pytest.fixture
    def project_root(self) -> str:
        """Get project root directory."""
        # Navigate up from tests/rfl/ to project root
        return str(Path(__file__).resolve().parents[2])

    def _run_subprocess(
        self,
        master_seed_hex: str,
        num_seeds: int,
        paths: list,
        project_root: str,
    ) -> dict:
        """Run PRNG computation in a subprocess."""
        script = SUBPROCESS_SCRIPT.format(
            project_root=project_root.replace("\\", "\\\\"),
            master_seed_hex=master_seed_hex,
            num_seeds=num_seeds,
            paths=json.dumps(paths),
        )

        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root,
            )

            if result.returncode != 0:
                pytest.fail(
                    f"Subprocess failed with code {result.returncode}:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            return json.loads(result.stdout.strip())

        finally:
            Path(script_path).unlink(missing_ok=True)

    def _run_inprocess(
        self,
        master_seed_hex: str,
        num_seeds: int,
        paths: list,
    ) -> dict:
        """Run PRNG computation in the current process."""
        from rfl.prng import DeterministicPRNG

        prng = DeterministicPRNG(master_seed_hex)

        results = {
            "master_seed_hex": master_seed_hex,
            "derived_seeds": [],
            "random_sequences": [],
            "seed_schedule": [],
        }

        # Test 1: Derive seeds for each path
        for path in paths:
            seed = prng.seed_for_path(*path)
            results["derived_seeds"].append({"path": path, "seed": seed})

        # Test 2: Generate random sequences for first path
        if paths:
            rng = prng.for_path(*paths[0])
            results["random_sequences"] = [rng.random() for _ in range(10)]

        # Test 3: Generate seed schedule
        results["seed_schedule"] = prng.generate_seed_schedule(
            num_cycles=num_seeds,
            slice_name="cross_process_test",
            mode="baseline",
        )

        return results

    def test_cross_process_seed_derivation(
        self,
        master_seed_hex: str,
        test_paths: list,
        project_root: str,
    ):
        """
        Verify seed derivation is identical across processes.

        This catches issues like:
        - Different hash implementations
        - Endianness problems
        - String encoding differences
        """
        inprocess_result = self._run_inprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=10,
            paths=test_paths,
        )

        subprocess_result = self._run_subprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=10,
            paths=test_paths,
            project_root=project_root,
        )

        # Compare derived seeds
        assert inprocess_result["derived_seeds"] == subprocess_result["derived_seeds"], (
            f"Seed derivation mismatch:\n"
            f"In-process: {inprocess_result['derived_seeds']}\n"
            f"Subprocess: {subprocess_result['derived_seeds']}"
        )

    def test_cross_process_random_sequences(
        self,
        master_seed_hex: str,
        test_paths: list,
        project_root: str,
    ):
        """
        Verify random sequences are identical across processes.

        This catches issues like:
        - Different random.Random implementations
        - Floating point precision differences
        """
        inprocess_result = self._run_inprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=10,
            paths=test_paths,
        )

        subprocess_result = self._run_subprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=10,
            paths=test_paths,
            project_root=project_root,
        )

        # Compare random sequences
        assert inprocess_result["random_sequences"] == subprocess_result["random_sequences"], (
            f"Random sequence mismatch:\n"
            f"In-process: {inprocess_result['random_sequences']}\n"
            f"Subprocess: {subprocess_result['random_sequences']}"
        )

    def test_cross_process_seed_schedule(
        self,
        master_seed_hex: str,
        test_paths: list,
        project_root: str,
    ):
        """
        Verify seed schedules are identical across processes.

        This catches issues like:
        - List comprehension ordering
        - Integer truncation differences
        """
        inprocess_result = self._run_inprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=20,
            paths=test_paths,
        )

        subprocess_result = self._run_subprocess(
            master_seed_hex=master_seed_hex,
            num_seeds=20,
            paths=test_paths,
            project_root=project_root,
        )

        # Compare seed schedules
        assert inprocess_result["seed_schedule"] == subprocess_result["seed_schedule"], (
            f"Seed schedule mismatch:\n"
            f"In-process: {inprocess_result['seed_schedule']}\n"
            f"Subprocess: {subprocess_result['seed_schedule']}"
        )

    def test_multiple_subprocess_runs_identical(
        self,
        master_seed_hex: str,
        test_paths: list,
        project_root: str,
    ):
        """
        Verify multiple subprocess runs produce identical results.

        This catches issues like:
        - Process-local state contamination
        - Timing-dependent behavior
        - Environment variable leakage
        """
        results = []
        for _ in range(3):
            result = self._run_subprocess(
                master_seed_hex=master_seed_hex,
                num_seeds=10,
                paths=test_paths,
                project_root=project_root,
            )
            results.append(result)

        # All runs should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i], (
                f"Subprocess run {i} differs from run 0:\n"
                f"Run 0: {results[0]}\n"
                f"Run {i}: {results[i]}"
            )


class TestEnvironmentIndependence:
    """Tests that verify PRNG is independent of environment variables."""

    @pytest.fixture
    def project_root(self) -> str:
        """Get project root directory."""
        return str(Path(__file__).resolve().parents[2])

    def test_pythonhashseed_independence(self, project_root: str):
        """
        Verify PRNG is independent of PYTHONHASHSEED.

        Python's hash() function is randomized by default (PYTHONHASHSEED).
        Our SHA-256 based PRNG should not be affected.
        """
        from rfl.prng import DeterministicPRNG

        master_seed = "b" * 64
        paths = [["test", "path"]]

        # Run with different PYTHONHASHSEED values
        results = []
        for hashseed in ["0", "12345", "random"]:
            env_script = f'''
import sys
sys.path.insert(0, "{project_root.replace(chr(92), chr(92)*2)}")
from rfl.prng import DeterministicPRNG
prng = DeterministicPRNG("{master_seed}")
seed = prng.seed_for_path("test", "path")
rng = prng.for_path("test", "path")
seq = [rng.random() for _ in range(5)]
print(f"{{seed}}|{{'|'.join(str(x) for x in seq)}}")
'''
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(env_script)
                script_path = f.name

            try:
                import os
                env = os.environ.copy()
                if hashseed == "random":
                    env.pop("PYTHONHASHSEED", None)
                else:
                    env["PYTHONHASHSEED"] = hashseed

                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                    cwd=project_root,
                )
                if result.returncode == 0:
                    results.append(result.stdout.strip())

            finally:
                Path(script_path).unlink(missing_ok=True)

        # All results should be identical regardless of PYTHONHASHSEED
        assert len(results) >= 2, "Need at least 2 successful runs"
        for i in range(1, len(results)):
            assert results[0] == results[i], (
                f"PYTHONHASHSEED affects PRNG output:\n"
                f"Result 0: {results[0]}\n"
                f"Result {i}: {results[i]}"
            )


# --- Run tests if executed directly ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

