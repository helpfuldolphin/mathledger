"""
PHASE II — NOT USED IN PHASE I

Tests for u2_snapshot_inspect.py CLI tool
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from experiments.u2.snapshots import SnapshotData, save_snapshot


@pytest.fixture
def temp_snapshot_root():
    """Create a temporary snapshot root with synthetic runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create run1 with snapshots
        run1 = root / "run1"
        run1.mkdir()
        snapshots1 = run1 / "snapshots"
        snapshots1.mkdir()
        
        for cycle in [10, 20, 30]:
            snapshot = SnapshotData(
                experiment_id="run1",
                slice_name="test_slice",
                mode="baseline",
                master_seed="test_seed",
                current_cycle=cycle,
                total_cycles=100,
                snapshot_cycle=cycle,
            )
            from experiments.u2.snapshots import create_snapshot_name
            snapshot_filename = create_snapshot_name("run1", cycle) + ".json"
            save_snapshot(snapshot, snapshots1 / snapshot_filename)
        
        # Create run2 with snapshots
        run2 = root / "run2"
        run2.mkdir()
        snapshots2 = run2 / "snapshots"
        snapshots2.mkdir()
        
        for cycle in [10, 20, 30, 40, 50]:
            snapshot = SnapshotData(
                experiment_id="run2",
                slice_name="test_slice",
                mode="baseline",
                master_seed="test_seed",
                current_cycle=cycle,
                total_cycles=100,
                snapshot_cycle=cycle,
            )
            from experiments.u2.snapshots import create_snapshot_name
            snapshot_filename = create_snapshot_name("run2", cycle) + ".json"
            save_snapshot(snapshot, snapshots2 / snapshot_filename)
        
        yield root


class TestU2SnapshotInspect:
    """Tests for u2_snapshot_inspect.py CLI."""
    
    def test_inspect_with_json_output(self, temp_snapshot_root):
        """Should produce valid JSON output."""
        result = subprocess.run(
            [
                "python",
                "scripts/u2_snapshot_inspect.py",
                "--snapshot-root",
                str(temp_snapshot_root),
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        
        assert output["status"] == "OK"
        assert "multi_history" in output
        assert "plan" in output
        assert "orchestrator_summary" in output
        assert "console_tile" in output
        assert output["multi_history"]["run_count"] == 2
    
    def test_inspect_human_readable(self, temp_snapshot_root):
        """Should produce human-readable output."""
        result = subprocess.run(
            [
                "python",
                "scripts/u2_snapshot_inspect.py",
                "--snapshot-root",
                str(temp_snapshot_root),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        
        assert result.returncode == 0
        assert "Snapshot Inspector" in result.stdout
        assert "Runs Analyzed" in result.stdout
        assert "Orchestrator Summary" in result.stdout
        assert "Console Tile" in result.stdout
    
    def test_inspect_empty_root(self):
        """Should handle empty snapshot root gracefully."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            result = subprocess.run(
                [
                    "python",
                    "scripts/u2_snapshot_inspect.py",
                    "--snapshot-root",
                    str(root),
                    "--json",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            
            assert result.returncode == 0
            output = json.loads(result.stdout)
            assert output["status"] == "NO_DATA"
            assert output["runs_analyzed"] == 0
    
    def test_inspect_nonexistent_root(self):
        """Should handle nonexistent snapshot root."""
        nonexistent = Path("/nonexistent/path/that/does/not/exist")
        
        result = subprocess.run(
            [
                "python",
                "scripts/u2_snapshot_inspect.py",
                "--snapshot-root",
                str(nonexistent),
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        
        # Should exit with error code
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["status"] == "ERROR"
        assert "not found" in output["error"].lower() or "not found" in output.get("message", "").lower()

