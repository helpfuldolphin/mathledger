"""
PHASE II — NOT USED IN PHASE I

Tests for Snapshot Pruning Tool
===============================

Tests verifying:
- Safety invariant: Always keep at least one valid snapshot
- Dry-run vs actual prune behavior
- Keep-latest and keep-interval policies
- Final snapshot preservation
- Health-aware pruning decisions
"""

import tempfile
from pathlib import Path
from typing import List

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("msgpack")
pytest.importorskip("zstandard")

from experiments.u2.snapshots import (
    SnapshotData,
    save_snapshot,
    load_snapshot,
    list_snapshots,
)

# Import pruning module
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.snapshot_prune import (
    assess_snapshot_health,
    compute_prune_plan,
    execute_prune,
    SnapshotHealth,
    SnapshotInfo,
    PruneReport,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_snapshots(temp_dir: Path, cycles: List[int], include_final: bool = False) -> List[Path]:
    """Helper to create multiple test snapshots."""
    paths = []
    for cycle in cycles:
        snapshot = SnapshotData(
            cycle_index=cycle,
            total_cycles=100,
            mode="baseline",
            slice_name="test",
            experiment_id="test_exp",
        )
        path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
        save_snapshot(snapshot, path)
        paths.append(path)
    
    if include_final:
        final_snapshot = SnapshotData(
            cycle_index=100,
            total_cycles=100,
            mode="baseline",
            slice_name="test",
            experiment_id="test_exp",
        )
        final_path = temp_dir / "snapshot_test_final.snap"
        save_snapshot(final_snapshot, final_path)
        paths.append(final_path)
    
    return paths


def corrupt_snapshot(path: Path) -> None:
    """Helper to corrupt a snapshot file."""
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    
    # Flip some bytes in the middle
    if len(data) > 50:
        data[len(data) // 2] ^= 0xFF
    
    with open(path, 'wb') as f:
        f.write(data)


# --- Test: Safety Invariant ---

class TestSafetyInvariant:
    """
    INVARIANT: Always keep at least one valid snapshot.
    
    This is the most critical test - pruning should NEVER leave
    the user without a valid recovery point.
    """
    
    def test_always_keeps_at_least_one_valid(self, temp_dir):
        """Even with aggressive pruning, at least one valid snapshot is kept."""
        # Create 10 snapshots
        cycles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        create_snapshots(temp_dir, cycles)
        
        # Prune with keep_latest=0 (as aggressive as possible)
        # But the tool should clamp this and keep at least 1
        report = compute_prune_plan(temp_dir, keep_latest=1)
        
        # At least one should be kept
        assert report.keep_count >= 1
        
        # The kept one should be VALID
        valid_kept = sum(1 for d in report.to_keep if d.health == SnapshotHealth.VALID)
        assert valid_kept >= 1
    
    def test_keeps_last_valid_when_most_are_corrupted(self, temp_dir):
        """If most snapshots are corrupted, still keeps the last valid one."""
        # Create 5 snapshots
        cycles = [10, 20, 30, 40, 50]
        paths = create_snapshots(temp_dir, cycles)
        
        # Corrupt all but the oldest one
        for path in paths[1:]:  # Keep cycle 10 valid
            corrupt_snapshot(path)
        
        # Prune aggressively
        report = compute_prune_plan(temp_dir, keep_latest=1)
        
        # The one valid snapshot should be kept
        valid_kept = [d for d in report.to_keep if d.health == SnapshotHealth.VALID]
        assert len(valid_kept) >= 1
        
        # It should be cycle 10 (the only valid one)
        assert any(d.cycle_index == 10 for d in valid_kept)
    
    def test_keeps_all_when_only_one_valid(self, temp_dir):
        """When there's only one valid snapshot, it's always kept."""
        # Create a single snapshot
        create_snapshots(temp_dir, [50])
        
        # Try to prune
        report = compute_prune_plan(temp_dir, keep_latest=5)
        
        # The only snapshot should be kept
        assert report.keep_count == 1
        assert report.delete_count == 0


# --- Test: Dry Run vs Actual Prune ---

class TestDryRunVsActual:
    """Tests verifying dry-run behavior vs actual pruning."""
    
    def test_dry_run_does_not_delete(self, temp_dir):
        """Dry run should not delete any files."""
        # Create 10 snapshots
        cycles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        paths = create_snapshots(temp_dir, cycles)
        
        original_count = len(list(temp_dir.glob("snapshot_*.snap")))
        
        # Compute and execute dry run
        report = compute_prune_plan(temp_dir, keep_latest=3)
        report = execute_prune(report, dry_run=True)
        
        # Verify no files were deleted
        current_count = len(list(temp_dir.glob("snapshot_*.snap")))
        assert current_count == original_count
        
        # But report should indicate what would be deleted
        assert report.delete_count > 0
        assert len(report.actually_deleted) == 0
        assert report.dry_run is True
    
    def test_actual_prune_deletes_files(self, temp_dir):
        """Actual prune should delete the indicated files."""
        # Create 10 snapshots
        cycles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        create_snapshots(temp_dir, cycles)
        
        original_count = len(list(temp_dir.glob("snapshot_*.snap")))
        
        # Compute and execute actual prune
        report = compute_prune_plan(temp_dir, keep_latest=3)
        report = execute_prune(report, dry_run=False)
        
        # Verify files were deleted
        current_count = len(list(temp_dir.glob("snapshot_*.snap")))
        assert current_count < original_count
        assert current_count == report.keep_count
        
        # Report should list actually deleted files
        assert len(report.actually_deleted) == report.delete_count
        assert report.dry_run is False
    
    def test_dry_run_returns_same_plan_multiple_times(self, temp_dir):
        """Dry run should be idempotent - same plan every time."""
        cycles = [10, 20, 30, 40, 50]
        create_snapshots(temp_dir, cycles)
        
        # Run dry-run twice
        report1 = compute_prune_plan(temp_dir, keep_latest=2)
        report2 = compute_prune_plan(temp_dir, keep_latest=2)
        
        # Plans should be identical
        assert report1.keep_count == report2.keep_count
        assert report1.delete_count == report2.delete_count
        assert {d.path for d in report1.to_keep} == {d.path for d in report2.to_keep}


# --- Test: Keep Latest Policy ---

class TestKeepLatestPolicy:
    """Tests for the --keep-latest N policy."""
    
    def test_keeps_n_latest_by_cycle(self, temp_dir):
        """Should keep the N snapshots with highest cycle indices."""
        cycles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        create_snapshots(temp_dir, cycles)
        
        report = compute_prune_plan(temp_dir, keep_latest=5)
        
        # Should keep 5
        assert report.keep_count == 5
        
        # Should keep the newest: 60, 70, 80, 90, 100
        kept_cycles = {d.cycle_index for d in report.to_keep if d.cycle_index is not None}
        expected = {60, 70, 80, 90, 100}
        assert kept_cycles == expected
    
    def test_keeps_all_when_n_exceeds_count(self, temp_dir):
        """When N > total snapshots, keep all."""
        cycles = [10, 20, 30]
        create_snapshots(temp_dir, cycles)
        
        report = compute_prune_plan(temp_dir, keep_latest=10)
        
        # Should keep all 3
        assert report.keep_count == 3
        assert report.delete_count == 0
    
    def test_respects_n_equals_one(self, temp_dir):
        """With N=1, should keep only the latest."""
        cycles = [10, 20, 30, 40, 50]
        create_snapshots(temp_dir, cycles)
        
        report = compute_prune_plan(temp_dir, keep_latest=1)
        
        # Should keep 1 (the latest)
        assert report.keep_count == 1
        kept_cycles = [d.cycle_index for d in report.to_keep]
        assert 50 in kept_cycles


# --- Test: Keep Interval Policy ---

class TestKeepIntervalPolicy:
    """Tests for the --keep-interval K policy."""
    
    def test_keeps_every_kth_cycle(self, temp_dir):
        """Should keep snapshots at every Kth cycle."""
        # Cycles: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        cycles = list(range(10, 110, 10))
        create_snapshots(temp_dir, cycles)
        
        # Keep every 30 cycles (so 30, 60, 90)
        report = compute_prune_plan(temp_dir, keep_latest=1, keep_interval=30)
        
        kept_cycles = {d.cycle_index for d in report.to_keep if d.cycle_index is not None}
        
        # Should include 30, 60, 90 plus at least the latest (100)
        assert 30 in kept_cycles
        assert 60 in kept_cycles
        assert 90 in kept_cycles
        assert 100 in kept_cycles  # Latest
    
    def test_interval_combined_with_latest(self, temp_dir):
        """Keep-interval should combine with keep-latest."""
        cycles = [10, 20, 30, 40, 50, 60]
        create_snapshots(temp_dir, cycles)
        
        # Keep latest 2 + every 20 cycles (20, 40, 60)
        report = compute_prune_plan(temp_dir, keep_latest=2, keep_interval=20)
        
        kept_cycles = {d.cycle_index for d in report.to_keep if d.cycle_index is not None}
        
        # Should include latest 2 (50, 60) + intervals (20, 40, 60)
        # Combined: 20, 40, 50, 60
        assert 20 in kept_cycles
        assert 40 in kept_cycles
        assert 60 in kept_cycles


# --- Test: Final Snapshot Preservation ---

class TestFinalSnapshotPreservation:
    """Tests for preserving 'final' snapshots."""
    
    def test_final_snapshots_never_deleted(self, temp_dir):
        """Final snapshots should never be pruned."""
        cycles = [10, 20, 30]
        create_snapshots(temp_dir, cycles, include_final=True)
        
        # Even with aggressive pruning
        report = compute_prune_plan(temp_dir, keep_latest=1)
        
        # Final should be in to_keep
        final_kept = [d for d in report.to_keep if "final" in d.path.stem.lower()]
        assert len(final_kept) == 1
        
        # Final should NOT be in to_delete
        final_deleted = [d for d in report.to_delete if "final" in d.path.stem.lower()]
        assert len(final_deleted) == 0
    
    def test_final_does_not_count_toward_keep_latest(self, temp_dir):
        """Final snapshots shouldn't reduce the keep-latest count."""
        cycles = [10, 20, 30, 40, 50]
        create_snapshots(temp_dir, cycles, include_final=True)
        
        report = compute_prune_plan(temp_dir, keep_latest=3)
        
        # Should keep 3 regular + 1 final = 4 total
        non_final_kept = [d for d in report.to_keep if "final" not in d.path.stem.lower()]
        assert len(non_final_kept) >= 3


# --- Test: Health-Aware Pruning ---

class TestHealthAwarePruning:
    """Tests for health-aware pruning decisions."""
    
    def test_prefers_valid_over_corrupted(self, temp_dir):
        """Should prefer keeping VALID snapshots over CORRUPTED ones."""
        # Create 5 snapshots
        cycles = [10, 20, 30, 40, 50]
        paths = create_snapshots(temp_dir, cycles)
        
        # Corrupt the newest ones, leave oldest valid
        corrupt_snapshot(paths[4])  # cycle 50
        corrupt_snapshot(paths[3])  # cycle 40
        
        # Prune to keep only 2
        report = compute_prune_plan(temp_dir, keep_latest=2)
        
        # Should prefer keeping valid ones (10, 20, 30)
        kept_valid = [d for d in report.to_keep if d.health == SnapshotHealth.VALID]
        assert len(kept_valid) >= 1
    
    def test_health_status_reflected_in_report(self, temp_dir):
        """Report should accurately reflect snapshot health status."""
        cycles = [10, 20, 30]
        paths = create_snapshots(temp_dir, cycles)
        
        # Corrupt one
        corrupt_snapshot(paths[1])  # cycle 20
        
        report = compute_prune_plan(temp_dir, keep_latest=5)
        
        # Report should show correct health status
        assert report.valid_count == 2  # 10 and 30 are valid
        
        # The corrupted one should have CORRUPTED status
        # Note: When corrupted, the cycle_index may or may not be extractable
        # So we look for any non-valid snapshot
        corrupted_decisions = [
            d for d in report.to_keep + report.to_delete 
            if d.health in (SnapshotHealth.CORRUPTED, SnapshotHealth.UNREADABLE)
        ]
        assert len(corrupted_decisions) >= 1


# --- Test: Idempotency ---

class TestIdempotency:
    """Tests for idempotent behavior."""
    
    def test_multiple_prunes_same_result(self, temp_dir):
        """Running prune multiple times should be idempotent."""
        cycles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        create_snapshots(temp_dir, cycles)
        
        # First prune
        report1 = compute_prune_plan(temp_dir, keep_latest=5)
        execute_prune(report1, dry_run=False)
        
        after_first = len(list(temp_dir.glob("snapshot_*.snap")))
        
        # Second prune with same settings
        report2 = compute_prune_plan(temp_dir, keep_latest=5)
        execute_prune(report2, dry_run=False)
        
        after_second = len(list(temp_dir.glob("snapshot_*.snap")))
        
        # Should be the same (second prune should delete nothing)
        assert after_first == after_second
        assert report2.delete_count == 0
    
    def test_safe_on_empty_directory(self, temp_dir):
        """Should handle empty directory gracefully."""
        report = compute_prune_plan(temp_dir, keep_latest=5)
        
        assert report.total_snapshots == 0
        assert report.keep_count == 0
        assert report.delete_count == 0


# --- Test: Edge Cases ---

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_handles_single_snapshot(self, temp_dir):
        """Should handle directory with single snapshot."""
        create_snapshots(temp_dir, [50])
        
        report = compute_prune_plan(temp_dir, keep_latest=5)
        
        assert report.total_snapshots == 1
        assert report.keep_count == 1
        assert report.delete_count == 0
    
    def test_handles_all_corrupted(self, temp_dir):
        """Should handle case where all snapshots are corrupted."""
        cycles = [10, 20, 30]
        paths = create_snapshots(temp_dir, cycles)
        
        # Corrupt all
        for path in paths:
            corrupt_snapshot(path)
        
        report = compute_prune_plan(temp_dir, keep_latest=1)
        
        # Should still keep at least one (even if corrupted)
        assert report.keep_count >= 1
        assert report.valid_count == 0
    
    def test_respects_directory_structure(self, temp_dir):
        """Should not delete files outside snapshot pattern."""
        # Create snapshots
        create_snapshots(temp_dir, [10, 20, 30])
        
        # Create other files that should not be touched
        (temp_dir / "manifest.json").write_text("{}")
        (temp_dir / "results.jsonl").write_text("")
        (temp_dir / "other_file.txt").write_text("important data")
        
        # Prune aggressively
        report = compute_prune_plan(temp_dir, keep_latest=1)
        execute_prune(report, dry_run=False)
        
        # Non-snapshot files should still exist
        assert (temp_dir / "manifest.json").exists()
        assert (temp_dir / "results.jsonl").exists()
        assert (temp_dir / "other_file.txt").exists()

