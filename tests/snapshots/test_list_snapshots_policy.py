"""
PHASE II — NOT USED IN PHASE I

Tests for Snapshot Policy Summary Feature
=========================================

Tests verifying:
- Coverage and gap calculations
- Manifest-aware cycle counting
- Handling of unknown/inferred total cycles
"""

import json
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
)

# Import list_snapshots functions
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.list_snapshots import (
    compute_policy_summary,
    list_run_snapshots,
    try_load_manifest,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_snapshots(temp_dir: Path, cycles: List[int], total_cycles: int = 100) -> List[Path]:
    """Helper to create multiple test snapshots."""
    paths = []
    for cycle in cycles:
        snapshot = SnapshotData(
            cycle_index=cycle,
            total_cycles=total_cycles,
            mode="baseline",
            slice_name="test",
            experiment_id="test_exp",
        )
        path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
        save_snapshot(snapshot, path)
        paths.append(path)
    return paths


def create_manifest(temp_dir: Path, total_cycles: int) -> Path:
    """Helper to create a manifest file."""
    manifest = {
        "slice": "test_slice",
        "mode": "baseline",
        "cycles": total_cycles,
        "initial_seed": 42,
    }
    path = temp_dir / "manifest.json"
    with open(path, 'w') as f:
        json.dump(manifest, f)
    return path


# --- Test: Coverage and Gap Math ---

class TestCoverageAndGapMath:
    """Tests for coverage and gap calculations."""
    
    def test_gap_calculation_basic(self, temp_dir):
        """Gap between consecutive snapshots should be computed correctly."""
        # Snapshots at 10, 30, 50 -> gaps of 20 each
        create_snapshots(temp_dir, [10, 30, 50], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        # Should have 2 gaps
        gaps = summary.get("gaps", [])
        assert len(gaps) == 2
        
        # Gap sizes should be 20 each
        gap_sizes = sorted([g["gap_size"] for g in gaps])
        assert gap_sizes == [20, 20]
    
    def test_longest_gap_identified(self, temp_dir):
        """Longest gap should be correctly identified."""
        # Snapshots at 10, 20, 70, 80 -> gaps of 10, 50, 10
        create_snapshots(temp_dir, [10, 20, 70, 80], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        assert summary["longest_gap"] == 50
    
    def test_average_gap_calculation(self, temp_dir):
        """Average gap should be correctly computed."""
        # Snapshots at 10, 20, 50 -> gaps of 10, 30
        create_snapshots(temp_dir, [10, 20, 50], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        # Average of [10, 30] = 20
        assert summary["average_gap"] == 20.0
    
    def test_no_gaps_with_single_snapshot(self, temp_dir):
        """Single snapshot should have no gaps."""
        create_snapshots(temp_dir, [50], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        assert summary["longest_gap"] == 0
        assert summary["average_gap"] == 0.0
        assert len(summary.get("gaps", [])) == 0
    
    def test_coverage_percent_basic(self, temp_dir):
        """Coverage percent should reflect snapshot density."""
        # 10 snapshots over 100 cycles = 10% coverage
        create_snapshots(temp_dir, list(range(10, 110, 10)), total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        assert summary["coverage_percent"] == 10.0


# --- Test: Manifest-Aware Cycle Counting ---

class TestManifestAwareCounting:
    """Tests for using manifest to get accurate total_cycles."""
    
    def test_uses_manifest_total_cycles(self, temp_dir):
        """Should use manifest total_cycles when provided."""
        create_snapshots(temp_dir, [10, 20, 30], total_cycles=50)
        
        snapshots = list_run_snapshots(temp_dir)
        
        # Without manifest - uses snapshot metadata
        summary1 = compute_policy_summary(snapshots, manifest_total_cycles=None)
        
        # With manifest - uses manifest value
        summary2 = compute_policy_summary(snapshots, manifest_total_cycles=200)
        
        assert summary1["total_cycles"] == 50
        assert summary1["total_cycles_source"] == "snapshot_metadata"
        
        assert summary2["total_cycles"] == 200
        assert summary2["total_cycles_source"] == "manifest"
    
    def test_manifest_loading(self, temp_dir):
        """Should be able to load manifest from common locations."""
        create_manifest(temp_dir, total_cycles=150)
        
        manifest = try_load_manifest(temp_dir)
        
        assert manifest is not None
        assert manifest["cycles"] == 150
    
    def test_coverage_label_accuracy(self, temp_dir):
        """Coverage label should indicate accuracy of total_cycles source."""
        create_snapshots(temp_dir, [10, 20, 30], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        
        # With manifest - exact
        summary_manifest = compute_policy_summary(snapshots, manifest_total_cycles=100)
        assert summary_manifest["coverage_label"] == "exact"
        
        # Without manifest - depends on source
        summary_no_manifest = compute_policy_summary(snapshots, manifest_total_cycles=None)
        assert summary_no_manifest["coverage_label"] in ["exact", "approx", "unknown"]


# --- Test: Unknown/Inferred Total Cycles ---

class TestInferredTotalCycles:
    """Tests for inferring total_cycles from snapshots."""
    
    def test_infers_from_snapshot_total_cycles(self, temp_dir):
        """Should infer from snapshot.total_cycles metadata."""
        # Create snapshots with total_cycles metadata
        create_snapshots(temp_dir, [10, 20, 30], total_cycles=75)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=None)
        
        # Should use total_cycles from snapshot metadata
        assert summary["total_cycles"] == 75
        assert summary["total_cycles_source"] == "snapshot_metadata"
    
    def test_infers_from_max_cycle_when_no_metadata(self, temp_dir):
        """Should infer from max cycle_index when total_cycles not in metadata."""
        # Create snapshots with 0 total_cycles
        for cycle in [10, 20, 50]:
            snapshot = SnapshotData(
                cycle_index=cycle,
                total_cycles=0,  # Unknown
                mode="baseline",
                slice_name="test",
            )
            path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
            save_snapshot(snapshot, path)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=None)
        
        # Should infer as max_cycle + 1
        assert summary["total_cycles"] == 51
        assert summary["total_cycles_source"] == "inferred_approx"
    
    def test_labels_inferred_as_approx(self, temp_dir):
        """Inferred total_cycles should be labeled as approximate."""
        # Create snapshots with 0 total_cycles
        for cycle in [10, 20]:
            snapshot = SnapshotData(
                cycle_index=cycle,
                total_cycles=0,
            )
            path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
            save_snapshot(snapshot, path)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=None)
        
        assert summary["coverage_label"] == "approx"
        assert any("approx" in note.lower() for note in summary.get("notes", []))


# --- Test: Edge Cases ---

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_directory(self, temp_dir):
        """Should handle empty directory gracefully."""
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=None)
        
        assert summary["snapshot_count"] == 0
        assert summary["valid_snapshot_count"] == 0
        assert summary["coverage_percent"] == 0.0
    
    def test_checkpoint_cycles_list(self, temp_dir):
        """checkpoint_cycles should list all valid snapshot cycles."""
        cycles = [5, 15, 25, 35, 45]
        create_snapshots(temp_dir, cycles, total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        assert set(summary["checkpoint_cycles"]) == set(cycles)
    
    def test_has_final_snapshot_detection(self, temp_dir):
        """Should detect presence of final snapshot."""
        create_snapshots(temp_dir, [10, 20], total_cycles=100)
        
        # Add final snapshot
        final = SnapshotData(cycle_index=100, total_cycles=100)
        save_snapshot(final, temp_dir / "snapshot_test_final.snap")
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        assert summary["has_final_snapshot"] is True
    
    def test_large_gap_warning(self, temp_dir):
        """Should generate warning for large gaps."""
        # Create snapshots with a 50-cycle gap
        create_snapshots(temp_dir, [10, 60], total_cycles=100)
        
        snapshots = list_run_snapshots(temp_dir)
        summary = compute_policy_summary(snapshots, manifest_total_cycles=100)
        
        # Should have a warning about the large gap
        assert summary["longest_gap"] == 50
        assert any("Warning" in note and "gap" in note.lower() for note in summary.get("notes", []))

