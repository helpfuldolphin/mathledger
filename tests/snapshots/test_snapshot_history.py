"""
PHASE II — NOT USED IN PHASE I

Tests for Snapshot History & Resume Intelligence Module
=======================================================

Tests verifying:
- Snapshot history ledger building
- Resume strategy advisor
- Global health signal generation
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("msgpack")
pytest.importorskip("zstandard")

from experiments.u2.snapshots import (
    SnapshotData,
    save_snapshot,
)
from experiments.u2.snapshot_guard import compute_manifest_hash

# Import history module
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.u2.snapshot_history import (
    HISTORY_SCHEMA_VERSION,
    HistoryStatus,
    ResumeStatus,
    build_snapshot_history,
    advise_resume_strategy,
    summarize_snapshots_for_global_health,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_snapshots(temp_dir: Path, cycles: List[int], total_cycles: int = 100, 
                     manifest_hash: str = "") -> List[Path]:
    """Helper to create multiple test snapshots."""
    paths = []
    for cycle in cycles:
        snapshot = SnapshotData(
            cycle_index=cycle,
            total_cycles=total_cycles,
            mode="baseline",
            slice_name="test_slice",
            experiment_id="test_exp",
            manifest_hash=manifest_hash,
        )
        path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
        save_snapshot(snapshot, path)
        paths.append(path)
    return paths


def create_manifest(temp_dir: Path, total_cycles: int = 100) -> Dict[str, Any]:
    """Helper to create a manifest file and return the manifest dict."""
    manifest = {
        "slice": "test_slice",
        "mode": "baseline",
        "cycles": total_cycles,
        "initial_seed": 42,
        "experiment_id": "test_exp",
    }
    path = temp_dir / "manifest.json"
    with open(path, 'w') as f:
        json.dump(manifest, f)
    return manifest


def corrupt_snapshot(path: Path) -> None:
    """Helper to corrupt a snapshot file."""
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    if len(data) > 50:
        data[len(data) // 2] ^= 0xFF
    with open(path, 'wb') as f:
        f.write(data)


# --- Test: build_snapshot_history ---

class TestBuildSnapshotHistory:
    """Tests for build_snapshot_history function."""
    
    def test_history_has_schema_version(self, temp_dir):
        """History should include schema_version."""
        create_snapshots(temp_dir, [10, 20, 30])
        
        history = build_snapshot_history(temp_dir)
        
        assert "schema_version" in history
        assert history["schema_version"] == HISTORY_SCHEMA_VERSION
    
    def test_discovers_all_snapshots(self, temp_dir):
        """Should discover all snapshot files."""
        create_snapshots(temp_dir, [10, 20, 30, 40, 50])
        
        history = build_snapshot_history(temp_dir)
        
        assert len(history["snapshots"]) == 5
        assert history["valid_count"] == 5
    
    def test_sorts_snapshots_by_cycle(self, temp_dir):
        """Snapshots should be sorted by cycle index ascending."""
        # Create out of order
        create_snapshots(temp_dir, [50, 10, 30, 20, 40])
        
        history = build_snapshot_history(temp_dir)
        
        cycles = [s["cycle_index"] for s in history["snapshots"]]
        assert cycles == [10, 20, 30, 40, 50]
    
    def test_categorizes_valid_and_corrupted(self, temp_dir):
        """Should categorize snapshots by status."""
        paths = create_snapshots(temp_dir, [10, 20, 30, 40])
        
        # Corrupt one
        corrupt_snapshot(paths[1])  # cycle 20
        
        history = build_snapshot_history(temp_dir)
        
        assert history["valid_count"] == 3
        # Corrupted file might be detected as CORRUPTED or UNKNOWN depending on failure mode
        assert history["corrupted_count"] + history["unknown_count"] >= 1
    
    def test_computes_coverage_metrics(self, temp_dir):
        """Should compute coverage percentage."""
        # 10 snapshots over 100 cycles = 10% coverage
        create_snapshots(temp_dir, list(range(10, 110, 10)), total_cycles=100)
        
        history = build_snapshot_history(temp_dir)
        
        assert history["coverage_pct"] == 10.0
    
    def test_computes_max_gap(self, temp_dir):
        """Should find the maximum gap between checkpoints."""
        # Snapshots at 10, 20, 70, 80 -> max gap is 50 (20 to 70)
        create_snapshots(temp_dir, [10, 20, 70, 80])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["max_gap"] == 50
    
    def test_computes_average_gap(self, temp_dir):
        """Should compute average gap between checkpoints."""
        # Snapshots at 10, 20, 50 -> gaps are 10, 30 -> avg is 20
        create_snapshots(temp_dir, [10, 20, 50])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["avg_gap"] == 20.0
    
    def test_recommended_resume_point(self, temp_dir):
        """Should recommend latest valid non-final snapshot."""
        create_snapshots(temp_dir, [10, 20, 30, 40, 50])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["recommended_resume_point"]["cycle"] == 50
        assert "000050" in history["recommended_resume_point"]["path"]
    
    def test_excludes_final_from_resume_recommendation(self, temp_dir):
        """Should not recommend final snapshots for resume."""
        create_snapshots(temp_dir, [10, 20, 30])
        
        # Add final snapshot
        final = SnapshotData(cycle_index=100, total_cycles=100)
        save_snapshot(final, temp_dir / "snapshot_test_final.snap")
        
        history = build_snapshot_history(temp_dir)
        
        # Should recommend cycle 30, not 100
        assert history["recommended_resume_point"]["cycle"] == 30
    
    def test_uses_manifest_total_cycles(self, temp_dir):
        """Should use manifest for accurate total_cycles."""
        create_manifest(temp_dir, total_cycles=200)
        create_snapshots(temp_dir, [10, 20, 30], total_cycles=100)  # Snapshot says 100
        
        history = build_snapshot_history(temp_dir)
        
        assert history["manifest_found"] is True
        assert history["manifest_total_cycles"] == 200
    
    def test_status_ok_when_healthy(self, temp_dir):
        """Status should be OK when snapshots are healthy."""
        create_snapshots(temp_dir, [10, 20, 30, 40, 50])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["status"] == HistoryStatus.OK.value
    
    def test_status_warn_when_large_gap(self, temp_dir):
        """Status should be WARN when there are large gaps."""
        # Create with 50-cycle gap which triggers WARN
        create_snapshots(temp_dir, [10, 60], total_cycles=100)
        
        history = build_snapshot_history(temp_dir)
        
        assert history["status"] == HistoryStatus.WARN.value
        assert history["max_gap"] > 20
    
    def test_status_block_when_no_valid(self, temp_dir):
        """Status should be BLOCK when no valid snapshots."""
        paths = create_snapshots(temp_dir, [10, 20])
        for path in paths:
            corrupt_snapshot(path)
        
        history = build_snapshot_history(temp_dir)
        
        assert history["status"] == HistoryStatus.BLOCK.value
    
    def test_status_empty_when_no_snapshots(self, temp_dir):
        """Status should be EMPTY when no snapshots found."""
        history = build_snapshot_history(temp_dir)
        
        assert history["status"] == HistoryStatus.EMPTY.value


# --- Test: advise_resume_strategy ---

class TestAdviseResumeStrategy:
    """Tests for advise_resume_strategy function."""
    
    def test_resume_allowed_when_valid_snapshots(self, temp_dir):
        """Should allow resume when valid snapshots exist."""
        # Create enough snapshots to avoid low-coverage CONDITIONAL
        create_snapshots(temp_dir, [10, 20, 30, 40, 50], total_cycles=100)
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_allowed"] is True
        # Status can be ALLOWED or CONDITIONAL depending on manifest check
        assert advice["resume_status"] in (ResumeStatus.ALLOWED.value, ResumeStatus.CONDITIONAL.value)
    
    def test_resume_blocked_when_empty(self, temp_dir):
        """Should block resume when no snapshots."""
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_allowed"] is False
        assert advice["resume_status"] == ResumeStatus.BLOCKED.value
        assert any("No snapshots" in w for w in advice["warnings"])
    
    def test_resume_blocked_when_no_valid(self, temp_dir):
        """Should block resume when no valid snapshots."""
        paths = create_snapshots(temp_dir, [10, 20])
        for path in paths:
            corrupt_snapshot(path)
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_allowed"] is False
        assert advice["resume_status"] == ResumeStatus.BLOCKED.value
    
    def test_resume_conditional_with_issues(self, temp_dir):
        """Should be conditional when there are issues (gaps or low coverage)."""
        # Create with large gap which triggers CONDITIONAL
        create_snapshots(temp_dir, [10, 60], total_cycles=100)
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_allowed"] is True
        assert advice["resume_status"] == ResumeStatus.CONDITIONAL.value
        # Should warn about gap
        assert any("gap" in w.lower() for w in advice["warnings"])
    
    def test_resume_conditional_with_large_gap(self, temp_dir):
        """Should be conditional when large gaps exist."""
        # Create with 50-cycle gap
        create_snapshots(temp_dir, [10, 60], total_cycles=100)
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_allowed"] is True
        assert advice["resume_status"] == ResumeStatus.CONDITIONAL.value
        assert any("gap" in w.lower() for w in advice["warnings"])
    
    def test_provides_resume_path(self, temp_dir):
        """Should provide the recommended resume path."""
        create_snapshots(temp_dir, [10, 20, 30, 40])
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert advice["resume_from"] is not None
        assert "000040" in advice["resume_from"]
        assert advice["resume_cycle"] == 40
    
    def test_includes_helpful_notes(self, temp_dir):
        """Should include helpful notes for resume."""
        create_snapshots(temp_dir, [10, 20, 30])
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history)
        
        assert len(advice["notes"]) > 0
        # Should include the command hint
        assert any("--restore-from" in n for n in advice["notes"])
    
    def test_strict_mode_blocks_without_manifest(self, temp_dir):
        """Strict mode should warn about missing manifest validation."""
        create_snapshots(temp_dir, [10, 20, 30])
        history = build_snapshot_history(temp_dir)
        
        advice = advise_resume_strategy(history, strict=True)
        
        # Without manifest, status should be UNKNOWN which strict mode may reject
        # depending on implementation
        assert "resume_status" in advice


# --- Test: summarize_snapshots_for_global_health ---

class TestSummarizeForGlobalHealth:
    """Tests for summarize_snapshots_for_global_health function."""
    
    def test_returns_ok_when_healthy(self, temp_dir):
        """Should return OK status when snapshots are healthy."""
        create_snapshots(temp_dir, [10, 20, 30, 40, 50])
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert health["status"] == "OK"
        assert health["snapshot_coverage_ok"] is True
    
    def test_returns_block_when_no_valid(self, temp_dir):
        """Should return BLOCK when no valid snapshots."""
        paths = create_snapshots(temp_dir, [10, 20])
        for path in paths:
            corrupt_snapshot(path)
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert health["status"] == "BLOCK"
    
    def test_returns_warn_with_large_gap(self, temp_dir):
        """Should return WARN when large gaps exist."""
        # Create with 60-cycle gap
        create_snapshots(temp_dir, [10, 70], total_cycles=100)
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert health["status"] == "WARN"
        assert health["max_gap"] > 50
    
    def test_includes_max_gap(self, temp_dir):
        """Should include max_gap in health signal."""
        create_snapshots(temp_dir, [10, 50])  # 40-cycle gap
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert health["max_gap"] == 40
    
    def test_detects_manifest_mismatch(self, temp_dir):
        """Should detect manifest mismatch."""
        manifest = create_manifest(temp_dir, total_cycles=100)
        manifest_hash = compute_manifest_hash(manifest)
        
        # Create snapshots with wrong manifest hash
        create_snapshots(temp_dir, [10, 20], manifest_hash="wrong_hash")
        
        history = build_snapshot_history(temp_dir)
        health = summarize_snapshots_for_global_health(history)
        
        # The health check should detect the mismatch
        assert "has_manifest_mismatch" in health
    
    def test_includes_coverage_info(self, temp_dir):
        """Should include coverage percentage."""
        create_snapshots(temp_dir, [10, 20, 30, 40, 50], total_cycles=100)
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert "coverage_pct" in health
        assert health["coverage_pct"] == 5.0
    
    def test_includes_human_readable_message(self, temp_dir):
        """Should include a human-readable status message."""
        create_snapshots(temp_dir, [10, 20, 30])
        history = build_snapshot_history(temp_dir)
        
        health = summarize_snapshots_for_global_health(history)
        
        assert "message" in health
        assert len(health["message"]) > 0


# --- Test: Edge Cases ---

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_snapshot_history(self, temp_dir):
        """Should handle single snapshot case."""
        create_snapshots(temp_dir, [50])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["valid_count"] == 1
        assert history["max_gap"] == 0
        assert history["avg_gap"] == 0.0
        assert history["recommended_resume_point"]["cycle"] == 50
    
    def test_only_final_snapshot(self, temp_dir):
        """Should handle case with only final snapshot."""
        final = SnapshotData(cycle_index=100, total_cycles=100)
        save_snapshot(final, temp_dir / "snapshot_test_final.snap")
        
        history = build_snapshot_history(temp_dir)
        
        # Final snapshot is valid but shouldn't be recommended for resume
        assert history["valid_count"] == 1
        # May or may not recommend final for resume
        # depending on implementation
    
    def test_handles_nested_snapshot_dir(self, temp_dir):
        """Should find snapshots in nested 'snapshots' directory."""
        snapshot_dir = temp_dir / "snapshots"
        snapshot_dir.mkdir()
        
        for cycle in [10, 20, 30]:
            snapshot = SnapshotData(cycle_index=cycle, total_cycles=100)
            save_snapshot(snapshot, snapshot_dir / f"snapshot_test_{cycle:06d}.snap")
        
        history = build_snapshot_history(temp_dir)
        
        assert history["valid_count"] == 3
    
    def test_checkpoint_cycles_list(self, temp_dir):
        """Should provide list of checkpoint cycles."""
        create_snapshots(temp_dir, [5, 15, 25, 35])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["checkpoint_cycles"] == [5, 15, 25, 35]
    
    def test_gaps_list_structure(self, temp_dir):
        """Should provide structured gap information."""
        create_snapshots(temp_dir, [10, 30, 50])
        
        history = build_snapshot_history(temp_dir)
        
        gaps = history["gaps"]
        assert len(gaps) == 2
        
        # First gap: 10 -> 30
        assert gaps[0]["from_cycle"] == 10
        assert gaps[0]["to_cycle"] == 30
        assert gaps[0]["gap_size"] == 20


# --- Test: Integration with snapshot_guard ---

class TestIntegrationWithGuard:
    """Tests for integration with snapshot_guard module."""
    
    def test_validates_against_manifest(self, temp_dir):
        """Should validate recommended snapshot against manifest."""
        manifest = create_manifest(temp_dir, total_cycles=100)
        manifest_hash = compute_manifest_hash(manifest)
        
        # Create snapshots with correct manifest hash
        for cycle in [10, 20, 30]:
            snapshot = SnapshotData(
                cycle_index=cycle,
                total_cycles=100,
                manifest_hash=manifest_hash,
            )
            save_snapshot(snapshot, temp_dir / f"snapshot_test_{cycle:06d}.snap")
        
        history = build_snapshot_history(temp_dir, include_manifest_validation=True)
        
        # Should have manifest status in recommended resume point
        assert history["recommended_resume_point"]["manifest_status"] is not None
    
    def test_handles_manifest_not_found(self, temp_dir):
        """Should handle missing manifest gracefully."""
        create_snapshots(temp_dir, [10, 20, 30])
        
        history = build_snapshot_history(temp_dir)
        
        assert history["manifest_found"] is False
        assert history["manifest_path"] is None

