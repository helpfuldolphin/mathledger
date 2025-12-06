"""
Tests for RFL Coverage Tracker

Validates statement coverage tracking, novelty measurement, and aggregation.
"""

import pytest
import numpy as np
import hashlib
import tempfile
from pathlib import Path

from backend.rfl.coverage import (
    CoverageTracker,
    CoverageMetrics,
    compute_statement_hash
)


class TestCoverageMetrics:
    """Tests for CoverageMetrics dataclass."""

    def test_basic_metrics(self):
        """Test basic coverage metrics creation."""
        metrics = CoverageMetrics(
            total_statements=100,
            distinct_statements=80,
            novel_statements=60,
            coverage_rate=0.80,
            novelty_rate=0.60,
            run_id="run_01"
        )

        assert metrics.total_statements == 100
        assert metrics.distinct_statements == 80
        assert metrics.novel_statements == 60
        assert metrics.coverage_rate == 0.80
        assert metrics.novelty_rate == 0.60
        assert metrics.run_id == "run_01"

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = CoverageMetrics(
            total_statements=100,
            distinct_statements=80,
            novel_statements=60,
            coverage_rate=0.80,
            novelty_rate=0.60,
            run_id="run_01"
        )

        d = metrics.to_dict()

        assert d["total_statements"] == 100
        assert d["coverage_rate"] == 0.80
        assert d["run_id"] == "run_01"


class TestCoverageTracker:
    """Tests for CoverageTracker."""

    def test_empty_tracker(self):
        """Test tracker with no baseline."""
        tracker = CoverageTracker()

        assert len(tracker.baseline_statements) == 0
        assert len(tracker.accumulated_statements) == 0
        assert len(tracker.run_metrics) == 0

    def test_tracker_with_baseline(self):
        """Test tracker initialized with baseline statements."""
        baseline = {
            "hash1",
            "hash2",
            "hash3"
        }

        tracker = CoverageTracker(baseline_statements=baseline)

        assert len(tracker.baseline_statements) == 3
        assert len(tracker.accumulated_statements) == 3

    def test_record_single_run(self):
        """Test recording a single run."""
        tracker = CoverageTracker()

        hashes = ["hash_a", "hash_b", "hash_c", "hash_a"]  # hash_a is duplicate

        metrics = tracker.record_run(
            statement_hashes=hashes,
            run_id="run_01",
            target_space_size=10
        )

        assert metrics.total_statements == 4
        assert metrics.distinct_statements == 3
        assert metrics.novel_statements == 3  # All novel (no baseline)
        assert metrics.coverage_rate == 3 / 10  # 3 distinct / 10 target
        assert metrics.novelty_rate == 3 / 4  # 3 novel / 4 total

    def test_record_multiple_runs(self):
        """Test recording multiple runs with accumulation."""
        baseline = {"hash_0"}
        tracker = CoverageTracker(baseline_statements=baseline)

        # Run 1
        hashes1 = ["hash_1", "hash_2", "hash_0"]
        metrics1 = tracker.record_run(
            hashes1,
            run_id="run_01",
            target_space_size=10
        )

        assert metrics1.distinct_statements == 3
        assert metrics1.novel_statements == 2  # hash_1, hash_2 are novel

        # Run 2 (some overlap with run 1)
        hashes2 = ["hash_2", "hash_3", "hash_4"]
        metrics2 = tracker.record_run(
            hashes2,
            run_id="run_02",
            target_space_size=10
        )

        assert metrics2.distinct_statements == 3
        assert metrics2.novel_statements == 2  # hash_3, hash_4 are novel (hash_2 in baseline now)

        # Check accumulation
        assert len(tracker.accumulated_statements) == 5  # hash_0, 1, 2, 3, 4

    def test_novelty_rate_all_duplicates(self):
        """Test novelty rate when all statements are duplicates."""
        baseline = {"hash_a", "hash_b"}
        tracker = CoverageTracker(baseline_statements=baseline)

        hashes = ["hash_a", "hash_b", "hash_a"]

        metrics = tracker.record_run(hashes, run_id="run_01", target_space_size=10)

        assert metrics.novel_statements == 0
        assert metrics.novelty_rate == 0.0

    def test_coverage_rate_no_target(self):
        """Test coverage rate when no target space size given."""
        tracker = CoverageTracker()

        hashes = ["hash_a", "hash_b"]

        metrics = tracker.record_run(hashes, run_id="run_01", target_space_size=None)

        # Without target, coverage should be 1.0 (covers what it generates)
        assert metrics.coverage_rate == 1.0

    def test_empty_run(self):
        """Test recording an empty run."""
        tracker = CoverageTracker()

        metrics = tracker.record_run([], run_id="run_01", target_space_size=10)

        assert metrics.total_statements == 0
        assert metrics.distinct_statements == 0
        assert metrics.novel_statements == 0
        assert metrics.coverage_rate == 0.0
        assert metrics.novelty_rate == 0.0

    def test_cumulative_coverage(self):
        """Test cumulative coverage calculation."""
        tracker = CoverageTracker()

        # Run 1
        tracker.record_run(
            ["hash_1", "hash_2"],
            run_id="run_01",
            target_space_size=10
        )

        # Run 2
        tracker.record_run(
            ["hash_3", "hash_4", "hash_5"],
            run_id="run_02",
            target_space_size=10
        )

        cumulative = tracker.get_cumulative_coverage()

        # 5 unique hashes / 10 target = 0.5
        assert cumulative == pytest.approx(0.5, abs=0.01)

    def test_aggregate_metrics(self):
        """Test aggregate statistics computation."""
        tracker = CoverageTracker()

        # Simulate 5 runs
        for i in range(5):
            coverage = 0.8 + i * 0.02  # Increasing coverage
            num_statements = 100

            # Generate hashes with target coverage rate
            distinct = int(num_statements * coverage)
            hashes = [f"hash_{i}_{j}" for j in range(distinct)]

            tracker.record_run(
                hashes,
                run_id=f"run_{i+1:02d}",
                target_space_size=num_statements
            )

        agg = tracker.get_aggregate_metrics()

        assert agg["num_runs"] == 5
        assert 0.8 <= agg["coverage_mean"] <= 0.9
        assert agg["coverage_std"] > 0
        assert agg["coverage_min"] <= agg["coverage_max"]
        assert "novelty_mean" in agg
        assert "cumulative_coverage" in agg

    def test_export_results(self):
        """Test exporting results to JSON."""
        tracker = CoverageTracker()

        tracker.record_run(
            ["hash_1", "hash_2"],
            run_id="run_01",
            target_space_size=10
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage.json"
            tracker.export_results(str(output_path))

            assert output_path.exists()

            # Load and validate
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "baseline_count" in data
            assert "accumulated_count" in data
            assert "runs" in data
            assert "aggregate" in data
            assert len(data["runs"]) == 1


class TestComputeStatementHash:
    """Tests for statement hash computation."""

    def test_basic_hash(self):
        """Test basic hash computation."""
        text = "p → p"
        hash_result = compute_statement_hash(text)

        assert len(hash_result) == 64  # SHA-256 produces 64-char hex
        assert hash_result.isalnum()

    def test_hash_determinism(self):
        """Test hash determinism (same input → same output)."""
        text = "p ∧ q"

        hash1 = compute_statement_hash(text)
        hash2 = compute_statement_hash(text)

        assert hash1 == hash2

    def test_hash_uniqueness(self):
        """Test hash uniqueness (different input → different output)."""
        text1 = "p → q"
        text2 = "q → p"

        hash1 = compute_statement_hash(text1)
        hash2 = compute_statement_hash(text2)

        assert hash1 != hash2

    def test_hash_unicode(self):
        """Test hash handles Unicode characters."""
        text = "∀x. P(x) → Q(x)"
        hash_result = compute_statement_hash(text)

        assert len(hash_result) == 64


class TestCoverageTrackerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_run(self):
        """Test tracker with large number of statements."""
        tracker = CoverageTracker()

        # Generate 10,000 hashes
        hashes = [hashlib.sha256(f"stmt_{i}".encode()).hexdigest() for i in range(10000)]

        metrics = tracker.record_run(
            hashes,
            run_id="large_run",
            target_space_size=20000
        )

        assert metrics.total_statements == 10000
        assert metrics.distinct_statements == 10000
        assert metrics.coverage_rate == 0.5

    def test_all_duplicates_within_run(self):
        """Test run where all statements are duplicates within the run."""
        tracker = CoverageTracker()

        hashes = ["hash_same"] * 100

        metrics = tracker.record_run(
            hashes,
            run_id="duplicates",
            target_space_size=10
        )

        assert metrics.total_statements == 100
        assert metrics.distinct_statements == 1
        assert metrics.novel_statements == 1

    def test_tracker_accumulation_correctness(self):
        """Test that accumulation doesn't double-count."""
        tracker = CoverageTracker()

        # Run 1: Add hash_a, hash_b
        tracker.record_run(["hash_a", "hash_b"], run_id="run_01")

        assert len(tracker.accumulated_statements) == 2

        # Run 2: Add hash_a (duplicate), hash_c (novel)
        tracker.record_run(["hash_a", "hash_c"], run_id="run_02")

        # Should only have 3 total (hash_a not double-counted)
        assert len(tracker.accumulated_statements) == 3
        assert "hash_a" in tracker.accumulated_statements
        assert "hash_b" in tracker.accumulated_statements
        assert "hash_c" in tracker.accumulated_statements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
