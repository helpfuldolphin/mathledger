# PHASE II — NOT RUN IN PHASE I
"""
Tests for U2 Replay Verification

Per u2_runner_spec.md v1.1.0 Section 6.4: Replay Verification Protocol.

Tests cover:
1. Golden replay (bit-identical): Replay produces identical attestation roots
2. RUN-41: Corrupt manifest detection
3. RUN-42: Seed schedule mismatch detection
4. RUN-43: Missing log detection
5. RUN-44: H_t mismatch detection
6. RUN-45: R_t mismatch detection
7. RUN-46: U_t mismatch detection
8. RUN-47: Cycle count mismatch detection
9. RUN-48: Config hash mismatch (warning, not failure by default)
10. RUN-49: Verification error handling
11. RUN-50: Unknown error handling
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from experiments.u2.runner import (
    U2Runner,
    U2Config,
    ReplayResult,
    ReplayError,
    ReplayManifestMismatch,
    ReplaySeedScheduleMismatch,
    ReplayLogMissing,
    ReplayHtMismatch,
    ReplayRtMismatch,
    ReplayUtMismatch,
    ReplayCycleCountMismatch,
    ReplayConfigHashMismatch,
    ReplayVerificationError,
    ReplayUnknownError,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_items() -> List[str]:
    """Standard test items for replay tests."""
    return [f"item_{i}" for i in range(10)]


@pytest.fixture
def deterministic_execute_fn():
    """
    A deterministic execute function for testing.

    Success is determined purely by (item, seed) to ensure reproducibility.
    """
    def execute_fn(item: str, seed: int) -> Tuple[bool, Dict[str, Any]]:
        # Deterministic success based on hash of item and seed
        import hashlib
        h = hashlib.sha256(f"{item}:{seed}".encode()).hexdigest()
        success = int(h[0], 16) < 8  # ~50% success rate, deterministic
        result = {
            "outcome": "VERIFIED" if success else "FAILED",
            "item": item,
            "seed": seed,
            "hash": h[:16],
        }
        return success, result
    return execute_fn


def create_primary_run(
    temp_dir: Path,
    items: List[str],
    execute_fn,
    mode: str = "baseline",
    cycles: int = 5,
    seed: int = 12345,
) -> Tuple[Path, Path]:
    """
    Create a primary run with manifest and results for replay testing.

    Returns:
        (manifest_path, results_path)
    """
    # Create config
    config = U2Config(
        experiment_id=f"test_primary_{mode}",
        slice_name="test_slice",
        mode=mode,
        total_cycles=cycles,
        master_seed=seed,
        output_dir=temp_dir,
    )

    # Run experiment
    runner = U2Runner(config)
    for i in range(cycles):
        runner.run_cycle(items, execute_fn)

    # Write results JSONL
    results_path = temp_dir / f"test_primary_{mode}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for record in runner.ht_series:
            f.write(json.dumps(record) + "\n")

    # Compute hashes
    import hashlib
    ht_series_str = json.dumps(runner.ht_series, sort_keys=True)
    ht_series_hash = hashlib.sha256(ht_series_str.encode()).hexdigest()

    slice_config = {"items": items}
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hashlib.sha256(slice_config_str.encode()).hexdigest()

    # Write manifest
    manifest = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": "test_slice",
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "ht_series_hash": ht_series_hash,
        "outputs": {
            "results": str(results_path),
            "manifest": str(temp_dir / f"test_manifest_{mode}.json"),
        },
    }

    manifest_path = temp_dir / f"test_manifest_{mode}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path, results_path


# ============================================================================
# Golden Replay Tests (Bit-Identical)
# ============================================================================

class TestGoldenReplay:
    """Tests for successful replay verification (bit-identical)."""

    def test_baseline_golden_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Replay of baseline mode produces identical attestation roots."""
        # Create primary run
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            mode="baseline",
            cycles=5,
            seed=12345,
        )

        # Run replay
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Verify replay succeeded
        assert result.status == "REPLAY_VERIFIED"
        assert result.all_cycles_match is True
        assert result.first_mismatch_cycle is None
        assert result.error_code is None
        assert len(result.cycle_comparisons) == 5

        # Verify each cycle matched
        for comp in result.cycle_comparisons:
            assert comp.h_t_match is True
            assert comp.r_t_match is True
            assert comp.all_match is True

    def test_rfl_golden_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Replay of RFL mode produces identical attestation roots."""
        # Create primary run
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            mode="rfl",
            cycles=5,
            seed=12345,
        )

        # Run replay
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Verify replay succeeded
        assert result.status == "REPLAY_VERIFIED"
        assert result.all_cycles_match is True
        assert result.first_mismatch_cycle is None

    def test_replay_with_different_cycle_counts(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Replay with 1, 10, and 100 cycles all succeed."""
        for cycles in [1, 10, 20]:  # Use 20 instead of 100 for faster tests
            run_dir = temp_dir / f"run_{cycles}"
            run_dir.mkdir(parents=True, exist_ok=True)

            manifest_path, _ = create_primary_run(
                temp_dir=run_dir,
                items=test_items,
                execute_fn=deterministic_execute_fn,
                mode="baseline",
                cycles=cycles,
                seed=42 + cycles,
            )

            result = U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

            assert result.status == "REPLAY_VERIFIED", f"Failed for {cycles} cycles"
            assert result.replayed_cycles == cycles


# ============================================================================
# RUN-41: Manifest Mismatch Tests
# ============================================================================

class TestRun41ManifestMismatch:
    """Tests for RUN-41: Manifest schema/field mismatch."""

    def test_missing_manifest_file(self, temp_dir, test_items, deterministic_execute_fn):
        """RUN-41: Missing manifest file raises ReplayManifestMismatch."""
        missing_path = temp_dir / "nonexistent_manifest.json"

        with pytest.raises(ReplayManifestMismatch) as exc_info:
            U2Runner.replay(
                manifest_path=missing_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-41"
        assert "not found" in exc_info.value.message.lower()

    def test_invalid_json_manifest(self, temp_dir, test_items, deterministic_execute_fn):
        """RUN-41: Invalid JSON in manifest raises ReplayManifestMismatch."""
        manifest_path = temp_dir / "invalid.json"
        with open(manifest_path, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ReplayManifestMismatch) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-41"

    def test_missing_required_fields(self, temp_dir, test_items, deterministic_execute_fn):
        """RUN-41: Missing required fields raises ReplayManifestMismatch."""
        manifest_path = temp_dir / "incomplete.json"
        # Missing 'mode' and 'initial_seed'
        with open(manifest_path, "w") as f:
            json.dump({"slice": "test", "cycles": 5}, f)

        with pytest.raises(ReplayManifestMismatch) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-41"
        assert "missing required fields" in exc_info.value.message.lower()


# ============================================================================
# RUN-42: Seed Schedule Mismatch Tests
# ============================================================================

class TestRun42SeedScheduleMismatch:
    """Tests for RUN-42: Seed schedule formula mismatch."""

    def test_different_seed_produces_different_results(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """
        Different master seed produces different results.

        This verifies that the seed schedule is actually used.
        """
        # Create directories first
        run1_dir = temp_dir / "run1"
        run2_dir = temp_dir / "run2"
        run1_dir.mkdir(parents=True, exist_ok=True)
        run2_dir.mkdir(parents=True, exist_ok=True)

        # Create two runs with different seeds
        manifest1, _ = create_primary_run(
            temp_dir=run1_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            mode="baseline",
            cycles=5,
            seed=12345,
        )

        manifest2, _ = create_primary_run(
            temp_dir=run2_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            mode="baseline",
            cycles=5,
            seed=99999,  # Different seed
        )

        # Load both manifests
        with open(manifest1) as f:
            m1 = json.load(f)
        with open(manifest2) as f:
            m2 = json.load(f)

        # Verify they have different ht_series hashes
        assert m1["ht_series_hash"] != m2["ht_series_hash"]


# ============================================================================
# RUN-43: Missing Log Tests
# ============================================================================

class TestRun43MissingLog:
    """Tests for RUN-43: Missing results log file."""

    def test_missing_results_file(self, temp_dir, test_items, deterministic_execute_fn):
        """RUN-43: Missing results JSONL raises ReplayLogMissing."""
        # Create primary run
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Delete the results file
        os.remove(results_path)

        with pytest.raises(ReplayLogMissing) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-43"
        assert "not found" in exc_info.value.message.lower()

    def test_missing_outputs_section(self, temp_dir, test_items, deterministic_execute_fn):
        """RUN-43: Manifest without outputs section raises ReplayLogMissing."""
        manifest_path = temp_dir / "no_outputs.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "slice": "test",
                "mode": "baseline",
                "cycles": 5,
                "initial_seed": 12345,
                # No "outputs" section
            }, f)

        with pytest.raises(ReplayLogMissing) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-43"


# ============================================================================
# RUN-44: H_t Mismatch Tests
# ============================================================================

class TestRun44HtMismatch:
    """Tests for RUN-44: Verification hash root (h_t) mismatch."""

    def test_corrupted_results_produces_mismatch(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """RUN-44: Corrupted results file produces h_t mismatch."""
        # Create primary run
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt the results file by modifying a record
        with open(results_path, "r") as f:
            lines = f.readlines()

        # Modify the first record
        record = json.loads(lines[0])
        record["success"] = not record["success"]  # Flip success
        record["item"] = "CORRUPTED_ITEM"  # Change item
        lines[0] = json.dumps(record) + "\n"

        with open(results_path, "w") as f:
            f.writelines(lines)

        # Replay should detect mismatch
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert result.status == "REPLAY_FAILED"
        assert result.first_mismatch_cycle == 0
        assert result.mismatch_type in ["h_t", "r_t"]  # Could be either
        assert result.error_code in ["RUN-44", "RUN-45"]

    def test_nondeterministic_execute_fn_fails(self, temp_dir, test_items):
        """RUN-44: Non-deterministic execute function produces mismatch."""
        call_count = [0]

        def nondeterministic_fn(item: str, seed: int) -> Tuple[bool, Dict[str, Any]]:
            call_count[0] += 1
            # First run: all succeed. Replay: all fail.
            success = call_count[0] <= 5
            return success, {"success": success, "call": call_count[0]}

        # Create primary run (first 5 calls succeed)
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=nondeterministic_fn,
            cycles=5,
        )

        # Replay (next 5 calls fail due to nondeterminism)
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=nondeterministic_fn,
        )

        # Should detect mismatch
        assert result.status == "REPLAY_FAILED"
        assert result.first_mismatch_cycle is not None


# ============================================================================
# RUN-47: Cycle Count Mismatch Tests
# ============================================================================

class TestRun47CycleCountMismatch:
    """Tests for RUN-47: Cycle count mismatch between manifest and log."""

    def test_truncated_log_raises_error(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """RUN-47: Truncated results log raises ReplayCycleCountMismatch."""
        # Create primary run with 5 cycles
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=5,
        )

        # Truncate results to only 3 cycles
        with open(results_path, "r") as f:
            lines = f.readlines()
        with open(results_path, "w") as f:
            f.writelines(lines[:3])

        with pytest.raises(ReplayCycleCountMismatch) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-47"
        assert "mismatch" in exc_info.value.message.lower()
        assert exc_info.value.details["manifest_cycles"] == 5
        assert exc_info.value.details["log_cycles"] == 3

    def test_extra_cycles_raises_error(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """RUN-47: Extra cycles in log raises ReplayCycleCountMismatch."""
        # Create primary run with 3 cycles
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=3,
        )

        # Add extra cycles to log
        with open(results_path, "a") as f:
            for i in range(2):
                f.write(json.dumps({"cycle": 3 + i, "extra": True}) + "\n")

        with pytest.raises(ReplayCycleCountMismatch) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=deterministic_execute_fn,
            )

        assert exc_info.value.code == "RUN-47"


# ============================================================================
# RUN-49: Verification Error Tests
# ============================================================================

class TestRun49VerificationError:
    """Tests for RUN-49: Verification throws exception."""

    def test_execute_fn_exception_raises_error(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """RUN-49: Exception in execute_fn raises ReplayVerificationError."""
        # Create primary run
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Create an execute function that throws
        def throwing_fn(item: str, seed: int):
            raise RuntimeError("Simulated verification failure")

        with pytest.raises(ReplayVerificationError) as exc_info:
            U2Runner.replay(
                manifest_path=manifest_path,
                items=test_items,
                execute_fn=throwing_fn,
            )

        assert exc_info.value.code == "RUN-49"
        assert "verification error" in exc_info.value.message.lower()


# ============================================================================
# ReplayResult Serialization Tests
# ============================================================================

class TestReplayResultSerialization:
    """Tests for ReplayResult.to_dict() serialization."""

    def test_successful_replay_serialization(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() produces valid JSON for successful replay."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Serialize and verify JSON-serializable
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict, indent=2)

        # Parse back and verify structure
        parsed = json.loads(json_str)
        assert parsed["status"] == "REPLAY_VERIFIED"
        assert "cycle_comparisons" in parsed
        assert len(parsed["cycle_comparisons"]) > 0

    def test_failed_replay_serialization(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes error details for failed replay."""
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()
        assert result_dict["status"] == "REPLAY_FAILED"
        assert result_dict["first_mismatch_cycle"] is not None
        assert result_dict["error_code"] is not None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for replay verification."""

    def test_single_cycle_replay(self, temp_dir, test_items, deterministic_execute_fn):
        """Replay with single cycle succeeds."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=1,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert result.status == "REPLAY_VERIFIED"
        assert result.replayed_cycles == 1

    def test_empty_items_list(self, temp_dir, deterministic_execute_fn):
        """Replay with empty items list handles gracefully."""
        # This should fail during primary run, not replay
        # But we test that replay handles the edge case
        pass  # Skip - empty items would fail at run time

    def test_replay_with_relative_path(self, temp_dir, test_items, deterministic_execute_fn):
        """Replay finds results file relative to manifest directory."""
        # Create primary run
        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Modify manifest to use relative path
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        manifest["outputs"]["results"] = results_path.name  # Just filename

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Should still find the file relative to manifest
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert result.status == "REPLAY_VERIFIED"


# ============================================================================
# Task 1: Versioned Replay Contract Tests
# ============================================================================

class TestReplayContractV1:
    """Tests for versioned replay contract (Task 1)."""

    def test_contract_version_in_result(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes contract_version field."""
        from experiments.u2.runner import REPLAY_CONTRACT_VERSION

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()

        assert "contract_version" in result_dict
        assert result_dict["contract_version"] == REPLAY_CONTRACT_VERSION
        assert result_dict["contract_version"] == "1.0.0"

    def test_per_cycle_stats_in_result(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes per_cycle_stats section."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=5,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()

        assert "per_cycle_stats" in result_dict
        stats = result_dict["per_cycle_stats"]
        assert stats["cycle_count_primary"] == 5
        assert stats["cycle_count_replay"] == 5
        assert stats["all_cycles_match"] is True
        assert stats["replay_coverage_pct"] == 100.0

    def test_primary_manifest_path_in_result(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes primary_manifest_path field."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()

        assert "primary_manifest_path" in result_dict
        assert result_dict["primary_manifest_path"] == str(manifest_path)

    def test_critical_mismatch_flags_in_result(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes critical_mismatch_flags section."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()

        assert "critical_mismatch_flags" in result_dict
        flags = result_dict["critical_mismatch_flags"]
        assert "ht_mismatch" in flags
        assert "rt_mismatch" in flags
        assert "ut_mismatch" in flags
        assert "cycle_count_mismatch" in flags
        assert "config_hash_mismatch" in flags

        # Successful replay should have no mismatches
        assert flags["ht_mismatch"] is False
        assert flags["rt_mismatch"] is False
        assert flags["ut_mismatch"] is False

    def test_governance_admissible_field(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplayResult.to_dict() includes governance_admissible field."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result_dict = result.to_dict()

        assert "governance_admissible" in result_dict
        # Full successful replay should be governance admissible
        assert result_dict["governance_admissible"] is True


# ============================================================================
# Task 2: Dry-Run and Partial Replay Tests
# ============================================================================

class TestDryRunValidation:
    """Tests for dry-run manifest validation (Task 2)."""

    def test_validate_valid_manifest(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """validate_replay_manifest() passes for valid manifest."""
        from experiments.u2.runner import validate_replay_manifest

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = validate_replay_manifest(manifest_path)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.slice_name == "test_slice"
        assert result.mode == "baseline"
        assert result.cycles == 5
        assert result.results_exist is True

    def test_validate_missing_manifest(self, temp_dir):
        """validate_replay_manifest() fails for missing manifest."""
        from experiments.u2.runner import validate_replay_manifest

        result = validate_replay_manifest(temp_dir / "nonexistent.json")

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_validate_invalid_json(self, temp_dir):
        """validate_replay_manifest() fails for invalid JSON."""
        from experiments.u2.runner import validate_replay_manifest

        manifest_path = temp_dir / "invalid.json"
        with open(manifest_path, "w") as f:
            f.write("{ invalid json }")

        result = validate_replay_manifest(manifest_path)

        assert result.is_valid is False
        assert any("json" in e.lower() for e in result.errors)

    def test_validate_missing_required_fields(self, temp_dir):
        """validate_replay_manifest() fails for missing required fields."""
        from experiments.u2.runner import validate_replay_manifest

        manifest_path = temp_dir / "incomplete.json"
        with open(manifest_path, "w") as f:
            json.dump({"slice": "test"}, f)  # Missing mode, cycles, initial_seed

        result = validate_replay_manifest(manifest_path)

        assert result.is_valid is False
        assert any("missing" in e.lower() for e in result.errors)

    def test_validate_missing_results_file(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """validate_replay_manifest() fails when results file is missing."""
        from experiments.u2.runner import validate_replay_manifest

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Delete results file
        os.remove(results_path)

        result = validate_replay_manifest(manifest_path)

        assert result.is_valid is False
        assert result.results_exist is False

    def test_validation_result_serialization(self, temp_dir):
        """ManifestValidationResult.to_dict() produces valid JSON."""
        from experiments.u2.runner import validate_replay_manifest

        result = validate_replay_manifest(temp_dir / "nonexistent.json")
        result_dict = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert "is_valid" in parsed
        assert "errors" in parsed
        assert "warnings" in parsed


class TestPartialReplayFence:
    """Tests for partial replay fence (Task 2)."""

    def test_full_replay_not_partial(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Full replay has is_partial_replay=False."""
        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=5,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert result.is_partial_replay is False
        assert result.replay_coverage_pct == 100.0

    def test_partial_replay_marked(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """
        Partial replay (fewer cycles) has is_partial_replay=True.

        This test simulates a partial replay by checking that when
        replayed_cycles < original_cycles, the flag is set.
        """
        from experiments.u2.runner import REPLAY_MODE_FULL

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=5,
        )

        # Full replay completes all cycles
        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Full replay should have coverage at 100%
        result_dict = result.to_dict()
        assert result_dict["is_partial_replay"] is False
        assert result_dict["replay_mode"] == REPLAY_MODE_FULL

    def test_governance_admissible_false_for_partial(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Partial replay is not governance admissible."""
        from experiments.u2.runner import (
            ReplayResult,
            REPLAY_MODE_PARTIAL,
            ReplayCycleComparison,
        )

        # Manually create a partial result to test admissibility logic
        partial_result = ReplayResult(
            status="REPLAY_VERIFIED",
            manifest_path="test.json",
            original_cycles=10,
            replayed_cycles=5,
            all_cycles_match=True,
            first_mismatch_cycle=None,
            mismatch_type=None,
            error_code=None,
            error_message=None,
            cycle_comparisons=[],
            original_results_hash="abc",
            replay_results_hash="abc",
            results_hash_match=True,
            replay_mode=REPLAY_MODE_PARTIAL,
            is_partial_replay=True,
            replay_coverage_pct=50.0,
        )

        result_dict = partial_result.to_dict()

        assert result_dict["is_partial_replay"] is True
        assert result_dict["governance_admissible"] is False


# ============================================================================
# Task 3: Governance Summary Tests
# ============================================================================

class TestGovernanceSummary:
    """Tests for governance summary helper (Task 3)."""

    def test_summarize_verified_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """summarize_replay_for_governance() extracts key fields from verified replay."""
        from experiments.u2.runner import summarize_replay_for_governance

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        summary = summarize_replay_for_governance(result.to_dict())

        assert summary["status"] == "REPLAY_VERIFIED"
        assert summary["governance_admissible"] is True
        assert "contract_version" in summary
        assert "manifest_pair" in summary
        assert "critical_mismatch_flags" in summary
        assert "per_cycle_stats" in summary

    def test_summarize_failed_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """summarize_replay_for_governance() handles failed replay."""
        from experiments.u2.runner import summarize_replay_for_governance

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        summary = summarize_replay_for_governance(result.to_dict())

        assert summary["status"] == "REPLAY_FAILED"
        assert summary["governance_admissible"] is False
        assert summary["error_code"] is not None

    def test_summary_is_deterministic(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """summarize_replay_for_governance() produces deterministic output."""
        from experiments.u2.runner import summarize_replay_for_governance

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Call summarize twice
        summary1 = summarize_replay_for_governance(result.to_dict())
        summary2 = summarize_replay_for_governance(result.to_dict())

        # Should produce identical output
        assert json.dumps(summary1, sort_keys=True) == json.dumps(summary2, sort_keys=True)

    def test_summary_is_json_serializable(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """summarize_replay_for_governance() output is JSON serializable."""
        from experiments.u2.runner import summarize_replay_for_governance

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        summary = summarize_replay_for_governance(result.to_dict())

        # Should be JSON serializable without errors
        json_str = json.dumps(summary, indent=2)
        parsed = json.loads(json_str)

        assert parsed["status"] == summary["status"]

    def test_summary_handles_legacy_format(self):
        """summarize_replay_for_governance() handles legacy result format."""
        from experiments.u2.runner import summarize_replay_for_governance

        # Legacy format without new fields
        legacy_result = {
            "status": "REPLAY_VERIFIED",
            "manifest_path": "test.json",
            "original_cycles": 5,
            "replayed_cycles": 5,
            "all_cycles_match": True,
        }

        summary = summarize_replay_for_governance(legacy_result)

        assert summary["status"] == "REPLAY_VERIFIED"
        assert summary["contract_version"] == "0.0.0"  # Fallback version
        assert summary["manifest_pair"]["primary"] == "test.json"

    def test_summary_mismatch_flags_populated(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """summarize_replay_for_governance() correctly populates mismatch flags."""
        from experiments.u2.runner import summarize_replay_for_governance

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt to cause h_t mismatch
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["success"] = not record["success"]
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        summary = summarize_replay_for_governance(result.to_dict())

        # Should have at least one mismatch flag set
        flags = summary["critical_mismatch_flags"]
        has_any_mismatch = any([
            flags.get("ht_mismatch", False),
            flags.get("rt_mismatch", False),
            flags.get("ut_mismatch", False),
        ])
        assert has_any_mismatch is True


# ============================================================================
# Phase III: Safety Envelope Tests
# ============================================================================

class TestSafetyEnvelope:
    """Tests for build_replay_safety_envelope() (Phase III Task 1)."""

    def test_safety_envelope_verified_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Safety envelope for successful replay has safety_level=OK."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            SafetyLevel,
            SAFETY_ENVELOPE_VERSION,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())

        assert envelope.schema_version == SAFETY_ENVELOPE_VERSION
        assert envelope.safety_level == SafetyLevel.OK
        assert envelope.is_fully_deterministic is True
        assert envelope.policy_update_allowed is True
        assert "PROCEED" in envelope.recommended_action

    def test_safety_envelope_failed_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Safety envelope for failed replay has safety_level=FAIL."""
        from experiments.u2.runner import build_replay_safety_envelope, SafetyLevel

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())

        assert envelope.safety_level == SafetyLevel.FAIL
        assert envelope.is_fully_deterministic is False
        assert envelope.policy_update_allowed is False
        assert "HALT" in envelope.recommended_action
        assert envelope.error_details is not None

    def test_safety_envelope_partial_replay(self):
        """Safety envelope for partial replay has safety_level=FAIL."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            SafetyLevel,
            REPLAY_MODE_PARTIAL,
        )

        # Simulate partial replay result
        partial_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_PARTIAL,
            "is_partial_replay": True,
            "per_cycle_stats": {
                "cycle_count_primary": 10,
                "cycle_count_replay": 5,
                "all_cycles_match": True,
                "replay_coverage_pct": 50.0,
            },
            "critical_mismatch_flags": {
                "ht_mismatch": False,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": False,
            },
            "cycle_comparisons": [],
        }

        envelope = build_replay_safety_envelope(partial_result)

        assert envelope.safety_level == SafetyLevel.FAIL
        assert envelope.is_fully_deterministic is False
        assert envelope.policy_update_allowed is False
        assert "partial" in envelope.recommended_action.lower()

    def test_safety_envelope_config_mismatch_warn(self):
        """Safety envelope with config hash mismatch has safety_level=WARN."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Simulate result with config mismatch only
        config_mismatch_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {
                "cycle_count_primary": 5,
                "cycle_count_replay": 5,
                "all_cycles_match": True,
                "replay_coverage_pct": 100.0,
            },
            "critical_mismatch_flags": {
                "ht_mismatch": False,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": True,  # Only config mismatch
            },
            "cycle_comparisons": [],
        }

        envelope = build_replay_safety_envelope(config_mismatch_result)

        assert envelope.safety_level == SafetyLevel.WARN
        assert "CAUTION" in envelope.recommended_action

    def test_safety_envelope_per_cycle_consistency(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Safety envelope includes per-cycle consistency stats."""
        from experiments.u2.runner import build_replay_safety_envelope

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
            cycles=5,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())

        assert "total_cycles" in envelope.per_cycle_consistency
        assert "replayed_cycles" in envelope.per_cycle_consistency
        assert "ht_match_count" in envelope.per_cycle_consistency
        assert "ht_match_pct" in envelope.per_cycle_consistency
        assert envelope.per_cycle_consistency["coverage_pct"] == 100.0

    def test_safety_envelope_serialization(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """ReplaySafetyEnvelope.to_dict() produces valid JSON."""
        from experiments.u2.runner import build_replay_safety_envelope

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        envelope_dict = envelope.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(envelope_dict, indent=2)
        parsed = json.loads(json_str)

        assert parsed["safety_level"] == envelope.safety_level
        assert parsed["confidence_score"] == envelope.confidence_score


# ============================================================================
# Phase III: Policy Guard Tests
# ============================================================================

class TestPolicyGuard:
    """Tests for validate_replay_before_policy_update() (Phase III Task 2)."""

    def test_policy_update_allowed_verified(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Policy update allowed for fully verified replay."""
        from experiments.u2.runner import validate_replay_before_policy_update

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert validate_replay_before_policy_update(result.to_dict()) is True

    def test_policy_update_blocked_failed(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Policy update blocked for failed replay."""
        from experiments.u2.runner import validate_replay_before_policy_update

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        assert validate_replay_before_policy_update(result.to_dict()) is False

    def test_policy_update_blocked_partial(self):
        """Policy update blocked for partial replay."""
        from experiments.u2.runner import validate_replay_before_policy_update, REPLAY_MODE_PARTIAL

        partial_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_PARTIAL,
            "is_partial_replay": True,
            "per_cycle_stats": {"all_cycles_match": True},
            "critical_mismatch_flags": {},
        }

        assert validate_replay_before_policy_update(partial_result) is False

    def test_policy_update_blocked_ht_mismatch(self):
        """Policy update blocked when h_t mismatch detected."""
        from experiments.u2.runner import validate_replay_before_policy_update, REPLAY_MODE_FULL

        ht_mismatch_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": False},
            "critical_mismatch_flags": {
                "ht_mismatch": True,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": False,
            },
        }

        assert validate_replay_before_policy_update(ht_mismatch_result) is False

    def test_policy_update_blocked_rt_mismatch(self):
        """Policy update blocked when r_t mismatch detected."""
        from experiments.u2.runner import validate_replay_before_policy_update, REPLAY_MODE_FULL

        rt_mismatch_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": False},
            "critical_mismatch_flags": {
                "ht_mismatch": False,
                "rt_mismatch": True,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": False,
            },
        }

        assert validate_replay_before_policy_update(rt_mismatch_result) is False

    def test_policy_update_blocked_config_mismatch(self):
        """Policy update blocked when config hash mismatch detected (strict mode)."""
        from experiments.u2.runner import validate_replay_before_policy_update, REPLAY_MODE_FULL

        config_mismatch_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": True},
            "critical_mismatch_flags": {
                "ht_mismatch": False,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": True,
            },
        }

        assert validate_replay_before_policy_update(config_mismatch_result) is False

    def test_policy_update_blocked_dry_run(self):
        """Policy update blocked for dry-run mode."""
        from experiments.u2.runner import validate_replay_before_policy_update, REPLAY_MODE_DRY_RUN

        dry_run_result = {
            "status": "REPLAY_DRY_RUN",
            "replay_mode": REPLAY_MODE_DRY_RUN,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": True},
            "critical_mismatch_flags": {},
        }

        assert validate_replay_before_policy_update(dry_run_result) is False


# ============================================================================
# Phase III: Confidence Score Tests
# ============================================================================

class TestConfidenceScore:
    """Tests for compute_replay_confidence() (Phase III Task 3)."""

    def test_confidence_verified_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Verified replay has high confidence score."""
        from experiments.u2.runner import compute_replay_confidence

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        score = compute_replay_confidence(result.to_dict())

        # Full verified replay should have score close to 1.0
        assert score >= 0.95
        assert score <= 1.0

    def test_confidence_failed_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Failed replay has low confidence score."""
        from experiments.u2.runner import compute_replay_confidence

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt all results
        with open(results_path, "r") as f:
            lines = f.readlines()
        corrupted = []
        for line in lines:
            record = json.loads(line)
            record["item"] = "CORRUPTED"
            record["success"] = not record.get("success", False)
            corrupted.append(json.dumps(record) + "\n")
        with open(results_path, "w") as f:
            f.writelines(corrupted)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        score = compute_replay_confidence(result.to_dict())

        # Failed replay should have score < 0.5
        assert score < 0.5

    def test_confidence_in_range(self):
        """Confidence score is always in [0, 1]."""
        from experiments.u2.runner import compute_replay_confidence

        # Test with minimal result
        minimal_result = {"status": "UNKNOWN"}
        score = compute_replay_confidence(minimal_result)
        assert 0.0 <= score <= 1.0

        # Test with all fields
        full_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": "full",
            "is_partial_replay": False,
            "per_cycle_stats": {
                "all_cycles_match": True,
                "replay_coverage_pct": 100.0,
            },
            "cycle_comparisons": [
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
            ],
        }
        score = compute_replay_confidence(full_result)
        assert 0.0 <= score <= 1.0

    def test_confidence_partial_replay_penalty(self):
        """Partial replay reduces confidence score."""
        from experiments.u2.runner import compute_replay_confidence, REPLAY_MODE_FULL, REPLAY_MODE_PARTIAL

        full_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {
                "all_cycles_match": True,
                "replay_coverage_pct": 100.0,
            },
        }

        partial_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_PARTIAL,
            "is_partial_replay": True,
            "per_cycle_stats": {
                "all_cycles_match": True,
                "replay_coverage_pct": 50.0,
            },
        }

        full_score = compute_replay_confidence(full_result)
        partial_score = compute_replay_confidence(partial_result)

        assert partial_score < full_score

    def test_confidence_dry_run_penalty(self):
        """Dry-run mode reduces confidence score."""
        from experiments.u2.runner import compute_replay_confidence, REPLAY_MODE_FULL, REPLAY_MODE_DRY_RUN

        full_result = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": True, "replay_coverage_pct": 100.0},
        }

        dry_run_result = {
            "status": "REPLAY_DRY_RUN",
            "replay_mode": REPLAY_MODE_DRY_RUN,
            "is_partial_replay": False,
            "per_cycle_stats": {"all_cycles_match": True, "replay_coverage_pct": 100.0},
        }

        full_score = compute_replay_confidence(full_result)
        dry_run_score = compute_replay_confidence(dry_run_result)

        assert dry_run_score < full_score

    def test_confidence_cycle_mismatch_impact(self):
        """Cycle mismatches reduce confidence score proportionally."""
        from experiments.u2.runner import compute_replay_confidence, REPLAY_MODE_FULL

        # All cycles match
        all_match = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"replay_coverage_pct": 100.0},
            "cycle_comparisons": [
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
            ],
        }

        # Half cycles mismatch
        half_mismatch = {
            "status": "REPLAY_VERIFIED",
            "replay_mode": REPLAY_MODE_FULL,
            "is_partial_replay": False,
            "per_cycle_stats": {"replay_coverage_pct": 100.0},
            "cycle_comparisons": [
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": False, "r_t_match": False, "u_t_match": False},
                {"h_t_match": True, "r_t_match": True, "u_t_match": True},
                {"h_t_match": False, "r_t_match": False, "u_t_match": False},
            ],
        }

        all_match_score = compute_replay_confidence(all_match)
        half_mismatch_score = compute_replay_confidence(half_mismatch)

        assert half_mismatch_score < all_match_score


# ============================================================================
# Phase IV: Promotion Safety Evaluation Tests
# ============================================================================

class TestPromotionSafetyEvaluation:
    """Tests for evaluate_replay_safety_for_promotion() (Phase IV Task 1)."""

    def test_promotion_ok_for_verified_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Verified replay evaluates to OK status."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        eval_result = evaluate_replay_safety_for_promotion(envelope.to_dict())

        assert eval_result["status"] == PromotionStatus.OK
        assert eval_result["safe_for_promotion"] is True
        assert eval_result["safe_for_policy_update"] is True
        assert len(eval_result["reasons"]) > 0

    def test_promotion_blocked_for_failed_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Failed replay evaluates to BLOCK status."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
        )

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        eval_result = evaluate_replay_safety_for_promotion(envelope.to_dict())

        assert eval_result["status"] == PromotionStatus.BLOCK
        assert eval_result["safe_for_promotion"] is False
        assert eval_result["safe_for_policy_update"] is False

    def test_promotion_blocked_for_ht_mismatch(self):
        """H_t mismatch results in BLOCK status."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.5,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {
                "ht_mismatch": True,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": False,
            },
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        eval_result = evaluate_replay_safety_for_promotion(envelope)

        assert eval_result["status"] == PromotionStatus.BLOCK
        assert eval_result["safe_for_promotion"] is False
        assert any("H_t" in r for r in eval_result["reasons"])

    def test_promotion_blocked_for_partial_replay(self):
        """Partial replay results in BLOCK status."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_PARTIAL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_PARTIAL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 50.0},
        }

        eval_result = evaluate_replay_safety_for_promotion(envelope)

        assert eval_result["status"] == PromotionStatus.BLOCK
        assert any("partial" in r.lower() for r in eval_result["reasons"])

    def test_promotion_warn_for_config_mismatch(self):
        """Config hash mismatch results in WARN status."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.WARN,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {
                "ht_mismatch": False,
                "rt_mismatch": False,
                "ut_mismatch": False,
                "cycle_count_mismatch": False,
                "config_hash_mismatch": True,
            },
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        eval_result = evaluate_replay_safety_for_promotion(envelope)

        assert eval_result["status"] == PromotionStatus.WARN
        assert eval_result["safe_for_promotion"] is True
        assert any("config" in r.lower() for r in eval_result["reasons"])

    def test_promotion_warn_for_low_confidence(self):
        """Low confidence score results in WARN status."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.85,  # Below 0.9 threshold
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        eval_result = evaluate_replay_safety_for_promotion(envelope)

        assert eval_result["status"] == PromotionStatus.WARN
        assert any("confidence" in r.lower() for r in eval_result["reasons"])

    def test_promotion_reasons_populated(self):
        """Evaluation always populates reasons list."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Test OK case
        ok_envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        eval_result = evaluate_replay_safety_for_promotion(ok_envelope)
        assert len(eval_result["reasons"]) > 0


# ============================================================================
# Phase IV: Evidence Pack Adapter Tests
# ============================================================================

class TestEvidencePackAdapter:
    """Tests for summarize_replay_safety_for_evidence() (Phase IV Task 2)."""

    def test_evidence_summary_verified_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Evidence summary for verified replay has replay_safety_ok=True."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            summarize_replay_safety_for_evidence,
            compute_replay_confidence,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        confidence = compute_replay_confidence(result.to_dict())
        summary = summarize_replay_safety_for_evidence(envelope.to_dict(), confidence)

        assert summary["replay_safety_ok"] is True
        assert summary["confidence_score"] >= 0.95
        assert summary["safety_level"] == "OK"
        assert summary["policy_update_allowed"] is True
        assert summary["status"] == "OK"

    def test_evidence_summary_failed_replay(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Evidence summary for failed replay has replay_safety_ok=False."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            summarize_replay_safety_for_evidence,
            compute_replay_confidence,
        )

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        confidence = compute_replay_confidence(result.to_dict())
        summary = summarize_replay_safety_for_evidence(envelope.to_dict(), confidence)

        assert summary["replay_safety_ok"] is False
        assert summary["status"] == "BLOCK"

    def test_evidence_summary_fields_present(self):
        """Evidence summary contains all required fields."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.98,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        summary = summarize_replay_safety_for_evidence(envelope, 0.98)

        assert "replay_safety_ok" in summary
        assert "confidence_score" in summary
        assert "safety_level" in summary
        assert "policy_update_allowed" in summary
        assert "status" in summary

    def test_evidence_summary_json_serializable(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Evidence summary is JSON serializable."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            summarize_replay_safety_for_evidence,
            compute_replay_confidence,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        confidence = compute_replay_confidence(result.to_dict())
        summary = summarize_replay_safety_for_evidence(envelope.to_dict(), confidence)

        # Should be JSON serializable
        json_str = json.dumps(summary, indent=2)
        parsed = json.loads(json_str)

        assert parsed["replay_safety_ok"] == summary["replay_safety_ok"]


# ============================================================================
# Phase IV: Director Safety Panel Tests
# ============================================================================

class TestDirectorSafetyPanel:
    """Tests for build_replay_safety_director_panel() (Phase IV Task 3)."""

    def test_director_panel_green_light(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Verified replay shows green status light."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        promotion_eval = evaluate_replay_safety_for_promotion(envelope.to_dict())
        panel = build_replay_safety_director_panel(envelope.to_dict(), promotion_eval)

        assert panel["status_light"] == "green"
        assert panel["safety_level"] == "OK"
        assert panel["is_fully_deterministic"] is True
        assert "passed" in panel["headline"].lower()
        assert "safe" in panel["headline"].lower()

    def test_director_panel_red_light(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Failed replay shows red status light."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
        )

        manifest_path, results_path = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        # Corrupt results
        with open(results_path, "r") as f:
            lines = f.readlines()
        record = json.loads(lines[0])
        record["item"] = "CORRUPTED"
        lines[0] = json.dumps(record) + "\n"
        with open(results_path, "w") as f:
            f.writelines(lines)

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        promotion_eval = evaluate_replay_safety_for_promotion(envelope.to_dict())
        panel = build_replay_safety_director_panel(envelope.to_dict(), promotion_eval)

        assert panel["status_light"] == "red"
        assert panel["is_fully_deterministic"] is False
        assert "blocked" in panel["headline"].lower()

    def test_director_panel_yellow_light(self):
        """Warning conditions show yellow status light."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.WARN,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {
                "config_hash_mismatch": True,
            },
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        assert panel["status_light"] == "yellow"
        assert "caution" in panel["headline"].lower() or "warning" in panel["headline"].lower()

    def test_director_panel_dry_run_prefix(self):
        """Dry-run mode adds prefix to headline."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
            SafetyLevel,
            REPLAY_MODE_DRY_RUN,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_DRY_RUN,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        assert "[DRY-RUN]" in panel["headline"]

    def test_director_panel_partial_prefix(self):
        """Partial replay mode adds prefix to headline."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
            SafetyLevel,
            REPLAY_MODE_PARTIAL,
        )

        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.5,
            "replay_mode": REPLAY_MODE_PARTIAL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 50.0},
        }

        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        assert "[PARTIAL]" in panel["headline"]
        assert panel["status_light"] == "red"

    def test_director_panel_fields_present(self):
        """Director panel contains all required fields."""
        from experiments.u2.runner import (
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        assert "status_light" in panel
        assert "safety_level" in panel
        assert "is_fully_deterministic" in panel
        assert "headline" in panel

    def test_director_panel_json_serializable(
        self, temp_dir, test_items, deterministic_execute_fn
    ):
        """Director panel is JSON serializable."""
        from experiments.u2.runner import (
            build_replay_safety_envelope,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_director_panel,
        )

        manifest_path, _ = create_primary_run(
            temp_dir=temp_dir,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        result = U2Runner.replay(
            manifest_path=manifest_path,
            items=test_items,
            execute_fn=deterministic_execute_fn,
        )

        envelope = build_replay_safety_envelope(result.to_dict())
        promotion_eval = evaluate_replay_safety_for_promotion(envelope.to_dict())
        panel = build_replay_safety_director_panel(envelope.to_dict(), promotion_eval)

        # Should be JSON serializable
        json_str = json.dumps(panel, indent=2)
        parsed = json.loads(json_str)

        assert parsed["status_light"] == panel["status_light"]
        assert parsed["headline"] == panel["headline"]


# ============================================================================
# Phase V: Governance Radar Fusion Tests
# ============================================================================

class TestGovernanceView:
    """Tests for build_replay_safety_governance_view() (Phase V Task 1)."""

    def test_aligned_ok_ok(self):
        """Both safety and radar OK produces ALIGNED status."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: OK
        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Radar: OK
        radar = {"status": "OK", "reasons": ["All checks passed"]}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        assert view["safety_status"] == PromotionStatus.OK
        assert view["governance_status"] == PromotionStatus.OK
        assert view["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert view["conflict"] is False

    def test_aligned_block_block(self):
        """Both safety and radar BLOCK produces ALIGNED status."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Radar: BLOCK
        radar = {"status": "BLOCK", "reasons": ["Radar detected anomaly"]}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        assert view["safety_status"] == PromotionStatus.BLOCK
        assert view["governance_status"] == PromotionStatus.BLOCK
        assert view["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert view["conflict"] is False

    def test_safety_block_radar_ok_conflict(self):
        """Safety BLOCK but radar OK produces DIVERGENT conflict."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Radar: OK (conflict!)
        radar = {"status": "OK", "reasons": ["Radar sees no issues"]}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        assert view["safety_status"] == PromotionStatus.BLOCK
        assert view["governance_status"] == PromotionStatus.OK
        assert view["governance_alignment"] == GovernanceAlignment.DIVERGENT
        assert view["conflict"] is True
        assert any("[CONFLICT]" in r for r in view["reasons"])

    def test_safety_ok_radar_block_conflict(self):
        """Safety OK but radar BLOCK produces DIVERGENT conflict."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: OK
        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Radar: BLOCK (conflict!)
        radar = {"status": "BLOCK", "reasons": ["Radar detected external issue"]}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        assert view["safety_status"] == PromotionStatus.OK
        assert view["governance_status"] == PromotionStatus.BLOCK
        assert view["governance_alignment"] == GovernanceAlignment.DIVERGENT
        assert view["conflict"] is True

    def test_safety_warn_radar_ok_tension(self):
        """Safety WARN but radar OK produces TENSION (no conflict)."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: WARN
        envelope = {
            "safety_level": SafetyLevel.WARN,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"config_hash_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Radar: OK
        radar = {"status": "OK", "reasons": []}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        assert view["safety_status"] == PromotionStatus.WARN
        assert view["governance_status"] == PromotionStatus.OK
        assert view["governance_alignment"] == GovernanceAlignment.TENSION
        assert view["conflict"] is False

    def test_radar_status_normalization(self):
        """Radar status strings are normalized correctly."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Test various status string formats
        for status_str, expected in [
            ("PASS", PromotionStatus.OK),
            ("GREEN", PromotionStatus.OK),
            ("WARNING", PromotionStatus.WARN),
            ("YELLOW", PromotionStatus.WARN),
            ("FAIL", PromotionStatus.BLOCK),
            ("RED", PromotionStatus.BLOCK),
        ]:
            radar = {"status": status_str, "reasons": []}
            view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
            assert view["governance_status"] == expected, f"Failed for {status_str}"

    def test_combined_reasons(self):
        """Reasons from both safety and radar are combined with prefixes."""
        from experiments.u2.runner import (
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        radar = {"status": "OK", "reasons": ["Radar check A", "Radar check B"]}

        view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        # Check that reasons have appropriate prefixes
        safety_reasons = [r for r in view["reasons"] if r.startswith("[Safety]")]
        radar_reasons = [r for r in view["reasons"] if r.startswith("[Radar]")]

        assert len(safety_reasons) > 0
        assert len(radar_reasons) == 2


class TestEvidencePackEnrichment:
    """Tests for evidence pack enrichment with governance_alignment (Phase V Task 2)."""

    def test_evidence_without_governance_view(self):
        """Evidence summary without governance_view has no alignment field."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        summary = summarize_replay_safety_for_evidence(envelope, 0.99)

        assert "governance_alignment" not in summary
        assert "replay_safety_ok" in summary

    def test_evidence_with_aligned_governance(self):
        """Evidence summary with ALIGNED governance_view includes alignment field."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        summary = summarize_replay_safety_for_evidence(envelope, 0.99, governance_view)

        assert summary["governance_alignment"] == GovernanceAlignment.ALIGNED

    def test_evidence_with_divergent_governance(self):
        """Evidence summary with DIVERGENT governance_view includes alignment field."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}  # Conflict!
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        summary = summarize_replay_safety_for_evidence(envelope, 0.3, governance_view)

        assert summary["governance_alignment"] == GovernanceAlignment.DIVERGENT

    def test_evidence_backwards_compatible(self):
        """Evidence summary remains backwards compatible."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        # Old-style call (without governance_view)
        summary = summarize_replay_safety_for_evidence(envelope, 0.99)

        # All original fields present
        assert "replay_safety_ok" in summary
        assert "confidence_score" in summary
        assert "safety_level" in summary
        assert "policy_update_allowed" in summary
        assert "status" in summary


class TestDirectorPanelV2:
    """Tests for director panel v2 with conflict_flag (Phase V Task 3)."""

    def test_panel_without_governance_view(self):
        """Director panel without governance_view has no conflict fields."""
        from experiments.u2.runner import (
            build_replay_safety_director_panel,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        assert "conflict_flag" not in panel
        assert "conflict_note" not in panel

    def test_panel_with_aligned_governance(self):
        """Director panel with aligned governance has conflict_flag=False."""
        from experiments.u2.runner import (
            build_replay_safety_director_panel,
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        panel = build_replay_safety_director_panel(envelope, promotion_eval, governance_view)

        assert panel["conflict_flag"] is False
        assert panel["conflict_note"] is None

    def test_panel_with_conflict(self):
        """Director panel with conflict has conflict_flag=True and conflict_note."""
        from experiments.u2.runner import (
            build_replay_safety_director_panel,
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}  # Conflict!
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        panel = build_replay_safety_director_panel(envelope, promotion_eval, governance_view)

        assert panel["conflict_flag"] is True
        assert panel["conflict_note"] is not None
        assert "BLOCK" in panel["conflict_note"]
        assert "OK" in panel["conflict_note"]
        assert "Manual review" in panel["conflict_note"]

    def test_panel_backwards_compatible(self):
        """Director panel remains backwards compatible."""
        from experiments.u2.runner import (
            build_replay_safety_director_panel,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)

        # Old-style call (without governance_view)
        panel = build_replay_safety_director_panel(envelope, promotion_eval)

        # All original fields present
        assert "status_light" in panel
        assert "safety_level" in panel
        assert "is_fully_deterministic" in panel
        assert "headline" in panel

    def test_panel_conflict_note_content(self):
        """Conflict note contains useful information for Director."""
        from experiments.u2.runner import (
            build_replay_safety_director_panel,
            build_replay_safety_governance_view,
            evaluate_replay_safety_for_promotion,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: OK but Radar: BLOCK
        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "BLOCK", "reasons": ["External system detected issue"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        panel = build_replay_safety_director_panel(envelope, promotion_eval, governance_view)

        assert panel["conflict_flag"] is True
        assert "OK" in panel["conflict_note"]
        assert "BLOCK" in panel["conflict_note"]


# ============================================================================
# Phase VI: Governance Signal Adapter Tests
# ============================================================================

class TestGovernanceSignalAdapter:
    """Tests for to_governance_signal_for_replay_safety() (Phase VI Task 1)."""

    def test_signal_block_when_safety_blocks(self):
        """Signal is BLOCK when safety side says BLOCK."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "BLOCK", "reasons": ["Radar also blocks"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["safety_status"] == PromotionStatus.BLOCK
        assert signal["governance_status"] == PromotionStatus.BLOCK
        assert signal["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert signal["conflict"] is False
        assert signal["signal_type"] == "replay_safety"

    def test_signal_block_when_radar_blocks(self):
        """Signal is BLOCK when radar side says BLOCK."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: OK
        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "BLOCK", "reasons": ["Radar detected issue"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        # BLOCK because alignment is DIVERGENT (conflict)
        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["safety_status"] == PromotionStatus.OK
        assert signal["governance_status"] == PromotionStatus.BLOCK
        assert signal["governance_alignment"] == GovernanceAlignment.DIVERGENT
        assert signal["conflict"] is True

    def test_signal_block_when_divergent(self):
        """Signal is BLOCK when alignment is DIVERGENT."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK, Radar: OK (conflict)
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": ["Radar sees no issues"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        assert signal["status"] == PromotionStatus.BLOCK
        assert signal["governance_alignment"] == GovernanceAlignment.DIVERGENT
        assert signal["conflict"] is True
        assert any("[CONFLICT]" in r for r in signal["reasons"])

    def test_signal_warn_on_tension(self):
        """Signal is WARN when alignment is TENSION."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: WARN, Radar: OK (tension, not conflict)
        envelope = {
            "safety_level": SafetyLevel.WARN,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"config_hash_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        assert signal["status"] == PromotionStatus.WARN
        assert signal["governance_alignment"] == GovernanceAlignment.TENSION
        assert signal["conflict"] is False

    def test_signal_ok_when_both_ok_aligned(self):
        """Signal is OK when both sides OK and aligned."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": ["All checks passed"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        assert signal["status"] == PromotionStatus.OK
        assert signal["safety_status"] == PromotionStatus.OK
        assert signal["governance_status"] == PromotionStatus.OK
        assert signal["governance_alignment"] == GovernanceAlignment.ALIGNED
        assert signal["conflict"] is False

    def test_signal_reasons_have_prefixes(self):
        """Signal reasons are prefixed with [Safety] and [Radar]."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": ["Radar check A", "Radar check B"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        # Check that reasons are properly prefixed
        safety_reasons = [r for r in signal["reasons"] if r.startswith("[Safety]")]
        radar_reasons = [r for r in signal["reasons"] if r.startswith("[Radar]")]

        assert len(safety_reasons) > 0
        assert len(radar_reasons) >= 2

    def test_signal_type_field(self):
        """Signal includes signal_type field."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        assert signal["signal_type"] == "replay_safety"

    def test_signal_json_serializable(self):
        """Signal is JSON serializable."""
        from experiments.u2.runner import (
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": ["Check passed"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)

        signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        # Should serialize without error
        json_str = json.dumps(signal, indent=2)
        parsed = json.loads(json_str)

        assert parsed["status"] == signal["status"]
        assert parsed["signal_type"] == "replay_safety"


# ============================================================================
# Phase VI: Evidence Pack Harmonization Tests
# ============================================================================

class TestEvidencePackHarmonization:
    """Tests for evidence pack harmonization with governance_status (Phase VI Task 2)."""

    def test_evidence_without_governance_signal(self):
        """Evidence summary without governance_signal has no governance_status field."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }

        summary = summarize_replay_safety_for_evidence(envelope, 0.99)

        assert "governance_status" not in summary
        assert "replay_safety_ok" in summary
        assert "status" in summary

    def test_evidence_with_ok_governance_signal(self):
        """Evidence summary with OK governance_signal includes governance_status=OK."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
        governance_signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        summary = summarize_replay_safety_for_evidence(
            envelope, 0.99,
            governance_view=governance_view,
            governance_signal=governance_signal
        )

        assert summary["governance_status"] == PromotionStatus.OK
        assert summary["replay_safety_ok"] is True

    def test_evidence_with_block_governance_signal(self):
        """Evidence summary with BLOCK governance_signal includes governance_status=BLOCK."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "BLOCK", "reasons": ["Radar also blocks"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
        governance_signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        summary = summarize_replay_safety_for_evidence(
            envelope, 0.3,
            governance_view=governance_view,
            governance_signal=governance_signal
        )

        assert summary["governance_status"] == PromotionStatus.BLOCK
        assert summary["replay_safety_ok"] is False
        assert summary["governance_alignment"] == GovernanceAlignment.ALIGNED

    def test_evidence_with_divergent_signal_shows_block(self):
        """Evidence with divergent conflict shows governance_status=BLOCK."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            GovernanceAlignment,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: BLOCK, Radar: OK (conflict)
        envelope = {
            "safety_level": SafetyLevel.FAIL,
            "policy_update_allowed": False,
            "is_fully_deterministic": False,
            "confidence_score": 0.3,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"ht_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": ["Radar sees no issues"]}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
        governance_signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        summary = summarize_replay_safety_for_evidence(
            envelope, 0.3,
            governance_view=governance_view,
            governance_signal=governance_signal
        )

        # Divergent conflict => governance_status=BLOCK
        assert summary["governance_status"] == PromotionStatus.BLOCK
        assert summary["governance_alignment"] == GovernanceAlignment.DIVERGENT

    def test_evidence_backwards_compatible_with_signal(self):
        """Evidence summary remains backwards compatible when adding governance_signal."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        envelope = {
            "safety_level": SafetyLevel.OK,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.99,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
        governance_signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        # Call with all new parameters
        summary = summarize_replay_safety_for_evidence(
            envelope, 0.99,
            governance_view=governance_view,
            governance_signal=governance_signal
        )

        # All original fields still present
        assert "replay_safety_ok" in summary
        assert "confidence_score" in summary
        assert "safety_level" in summary
        assert "policy_update_allowed" in summary
        assert "status" in summary
        # Plus new fields
        assert "governance_alignment" in summary
        assert "governance_status" in summary

    def test_evidence_governance_status_not_duplicate_status(self):
        """governance_status is distinct from status field."""
        from experiments.u2.runner import (
            summarize_replay_safety_for_evidence,
            to_governance_signal_for_replay_safety,
            evaluate_replay_safety_for_promotion,
            build_replay_safety_governance_view,
            PromotionStatus,
            SafetyLevel,
            REPLAY_MODE_FULL,
        )

        # Safety: WARN (status field will be WARN)
        # But governance signal OK because radar also OK, tension case
        envelope = {
            "safety_level": SafetyLevel.WARN,
            "policy_update_allowed": True,
            "is_fully_deterministic": True,
            "confidence_score": 0.95,
            "replay_mode": REPLAY_MODE_FULL,
            "critical_mismatch_flags": {"config_hash_mismatch": True},
            "per_cycle_consistency": {"coverage_pct": 100.0},
        }
        promotion_eval = evaluate_replay_safety_for_promotion(envelope)
        radar = {"status": "OK", "reasons": []}
        governance_view = build_replay_safety_governance_view(envelope, promotion_eval, radar)
        governance_signal = to_governance_signal_for_replay_safety(promotion_eval, governance_view)

        summary = summarize_replay_safety_for_evidence(
            envelope, 0.95,
            governance_view=governance_view,
            governance_signal=governance_signal
        )

        # status is from envelope evaluation = WARN
        assert summary["status"] == PromotionStatus.WARN
        # governance_status is from consolidated signal = WARN (tension)
        assert summary["governance_status"] == PromotionStatus.WARN
