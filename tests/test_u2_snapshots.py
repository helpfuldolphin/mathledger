"""
Tests for U2 Snapshots

Validates snapshot save/load/validation and rotation.
"""

import pytest
from pathlib import Path

from experiments.u2.snapshots import (
    SnapshotData,
    save_snapshot,
    load_snapshot,
    find_latest_snapshot,
    rotate_snapshots,
    SnapshotValidationError,
    SnapshotCorruptionError,
    NoSnapshotFoundError,
)


@pytest.fixture
def sample_snapshot():
    """Fixture for sample snapshot data."""
    return SnapshotData(
        cycle_index=10,
        ht_series=["h1", "h2", "h3"],
        policy_update_count=5,
        success_count={"item1": 3, "item2": 2},
        attempt_count={"item1": 5, "item2": 5},
        weights={"item1": 0.6, "item2": 0.4},
        config={"experiment_id": "test", "mode": "rfl"},
    )


class TestSnapshotData:
    """Tests for SnapshotData."""
    
    def test_snapshot_creation(self, sample_snapshot):
        """Test creating a snapshot."""
        assert sample_snapshot.cycle_index == 10
        assert len(sample_snapshot.ht_series) == 3
        assert sample_snapshot.policy_update_count == 5
    
    def test_snapshot_to_dict(self, sample_snapshot):
        """Test converting snapshot to dictionary."""
        snapshot_dict = sample_snapshot.to_dict()
        
        assert snapshot_dict["cycle_index"] == 10
        assert "ht_series" in snapshot_dict
        assert "config" in snapshot_dict
    
    def test_snapshot_from_dict(self, sample_snapshot):
        """Test creating snapshot from dictionary."""
        snapshot_dict = sample_snapshot.to_dict()
        restored = SnapshotData.from_dict(snapshot_dict)
        
        assert restored.cycle_index == sample_snapshot.cycle_index
        assert restored.ht_series == sample_snapshot.ht_series
        assert restored.config == sample_snapshot.config


class TestSaveLoadSnapshot:
    """Tests for snapshot save and load."""
    
    def test_save_snapshot(self, tmp_path, sample_snapshot):
        """Test saving a snapshot."""
        snapshot_path = tmp_path / "test.snap"
        
        snapshot_hash = save_snapshot(sample_snapshot, snapshot_path)
        
        assert snapshot_path.exists()
        assert len(snapshot_hash) == 64  # SHA256 hex length
    
    def test_load_snapshot(self, tmp_path, sample_snapshot):
        """Test loading a snapshot."""
        snapshot_path = tmp_path / "test.snap"
        save_snapshot(sample_snapshot, snapshot_path)
        
        loaded = load_snapshot(snapshot_path, verify_hash=True)
        
        assert loaded.cycle_index == sample_snapshot.cycle_index
        assert loaded.ht_series == sample_snapshot.ht_series
        assert loaded.config == sample_snapshot.config
    
    def test_save_load_roundtrip(self, tmp_path, sample_snapshot):
        """Test save-load roundtrip."""
        snapshot_path = tmp_path / "roundtrip.snap"
        
        save_snapshot(sample_snapshot, snapshot_path)
        loaded = load_snapshot(snapshot_path)
        
        # Verify all fields match
        assert loaded.cycle_index == sample_snapshot.cycle_index
        assert loaded.ht_series == sample_snapshot.ht_series
        assert loaded.policy_update_count == sample_snapshot.policy_update_count
        assert loaded.success_count == sample_snapshot.success_count
        assert loaded.attempt_count == sample_snapshot.attempt_count
    
    def test_load_nonexistent_snapshot(self, tmp_path):
        """Test loading a nonexistent snapshot."""
        snapshot_path = tmp_path / "nonexistent.snap"
        
        with pytest.raises(NoSnapshotFoundError):
            load_snapshot(snapshot_path)
    
    def test_load_corrupted_json(self, tmp_path):
        """Test loading a corrupted snapshot."""
        snapshot_path = tmp_path / "corrupted.snap"
        snapshot_path.write_text("not valid json{{{")
        
        with pytest.raises(SnapshotCorruptionError):
            load_snapshot(snapshot_path)
    
    def test_hash_verification_failure(self, tmp_path, sample_snapshot):
        """Test hash verification failure."""
        snapshot_path = tmp_path / "test.snap"
        save_snapshot(sample_snapshot, snapshot_path)
        
        # Tamper with the file
        import json
        with open(snapshot_path, "r") as f:
            data = json.load(f)
        data["cycle_index"] = 999  # Change data but keep old hash
        with open(snapshot_path, "w") as f:
            json.dump(data, f)
        
        with pytest.raises(SnapshotValidationError):
            load_snapshot(snapshot_path, verify_hash=True)
    
    def test_load_without_hash_verification(self, tmp_path, sample_snapshot):
        """Test loading without hash verification."""
        snapshot_path = tmp_path / "test.snap"
        save_snapshot(sample_snapshot, snapshot_path)
        
        # Should load successfully even if hash is missing
        loaded = load_snapshot(snapshot_path, verify_hash=False)
        assert loaded.cycle_index == sample_snapshot.cycle_index


class TestFindLatestSnapshot:
    """Tests for finding latest snapshot."""
    
    def test_find_latest_single(self, tmp_path, sample_snapshot):
        """Test finding latest snapshot with single file."""
        snapshot_path = tmp_path / "snap1.snap"
        save_snapshot(sample_snapshot, snapshot_path)
        
        latest = find_latest_snapshot(tmp_path)
        
        assert latest == snapshot_path
    
    def test_find_latest_multiple(self, tmp_path, sample_snapshot):
        """Test finding latest snapshot with multiple files."""
        import time
        
        snap1 = tmp_path / "snap1.snap"
        save_snapshot(sample_snapshot, snap1)
        
        time.sleep(0.01)  # Ensure different mtimes
        
        snap2 = tmp_path / "snap2.snap"
        save_snapshot(sample_snapshot, snap2)
        
        latest = find_latest_snapshot(tmp_path)
        
        assert latest == snap2
    
    def test_find_latest_no_snapshots(self, tmp_path):
        """Test finding latest when no snapshots exist."""
        latest = find_latest_snapshot(tmp_path)
        assert latest is None
    
    def test_find_latest_nonexistent_dir(self, tmp_path):
        """Test finding latest in nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        latest = find_latest_snapshot(nonexistent)
        assert latest is None


class TestRotateSnapshots:
    """Tests for snapshot rotation."""
    
    def test_rotate_no_deletion(self, tmp_path, sample_snapshot):
        """Test rotation when count is under limit."""
        snap1 = tmp_path / "snap1.snap"
        snap2 = tmp_path / "snap2.snap"
        save_snapshot(sample_snapshot, snap1)
        save_snapshot(sample_snapshot, snap2)
        
        deleted = rotate_snapshots(tmp_path, keep_count=5)
        
        assert len(deleted) == 0
        assert snap1.exists()
        assert snap2.exists()
    
    def test_rotate_deletes_oldest(self, tmp_path, sample_snapshot):
        """Test rotation deletes oldest snapshots."""
        import time
        
        snapshots = []
        for i in range(5):
            snap_path = tmp_path / f"snap{i}.snap"
            save_snapshot(sample_snapshot, snap_path)
            snapshots.append(snap_path)
            time.sleep(0.01)  # Ensure different mtimes
        
        deleted = rotate_snapshots(tmp_path, keep_count=3)
        
        assert len(deleted) == 2
        # Oldest two should be deleted
        assert not snapshots[0].exists()
        assert not snapshots[1].exists()
        # Newest three should remain
        assert snapshots[2].exists()
        assert snapshots[3].exists()
        assert snapshots[4].exists()
    
    def test_rotate_disabled(self, tmp_path, sample_snapshot):
        """Test rotation with keep_count=0 (disabled)."""
        snap1 = tmp_path / "snap1.snap"
        save_snapshot(sample_snapshot, snap1)
        
        deleted = rotate_snapshots(tmp_path, keep_count=0)
        
        assert len(deleted) == 0
        assert snap1.exists()
    
    def test_rotate_nonexistent_dir(self, tmp_path):
        """Test rotation in nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        deleted = rotate_snapshots(nonexistent, keep_count=3)
        
        assert len(deleted) == 0


@pytest.mark.unit
class TestSnapshotIntegration:
    """Integration tests for snapshot functionality."""
    
    def test_snapshot_workflow(self, tmp_path):
        """Test complete snapshot workflow."""
        # Create initial snapshot
        snapshot1 = SnapshotData(
            cycle_index=10,
            ht_series=["h1", "h2"],
            policy_update_count=5,
            success_count={"a": 2},
            attempt_count={"a": 3},
            weights={"a": 0.67},
            config={"mode": "rfl"},
        )
        
        path1 = tmp_path / "snap1.snap"
        save_snapshot(snapshot1, path1)
        
        # Create another snapshot
        snapshot2 = SnapshotData(
            cycle_index=20,
            ht_series=["h1", "h2", "h3", "h4"],
            policy_update_count=10,
            success_count={"a": 4, "b": 3},
            attempt_count={"a": 6, "b": 5},
            weights={"a": 0.67, "b": 0.60},
            config={"mode": "rfl"},
        )
        
        path2 = tmp_path / "snap2.snap"
        save_snapshot(snapshot2, path2)
        
        # Find latest
        latest = find_latest_snapshot(tmp_path)
        assert latest == path2
        
        # Load latest
        loaded = load_snapshot(latest)
        assert loaded.cycle_index == 20
        assert len(loaded.ht_series) == 4
