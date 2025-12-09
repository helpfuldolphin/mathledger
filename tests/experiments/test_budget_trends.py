"""
Tests for Budget Trend Analyzer (Agent A5 - Advanced Observability)

PHASE II — NOT USED IN PHASE I

Tests verify:
    1. Trend classification logic (IMPROVING/STABLE/DEGRADING/UNKNOWN)
    2. Multi-run analysis with synthetic inputs
    3. Output formatting (Markdown, JSON)
    4. Read-only invariant
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.budget_trends import (
    TrendDirection,
    HEALTH_ORDER,
    health_to_score,
    classify_trend,
    RunHealth,
    SliceTrend,
    TrendReport,
    analyze_trends,
    format_markdown,
    format_json,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def improving_sequence() -> List[str]:
    """Status sequence showing improvement."""
    return ["STARVED", "STARVED", "TIGHT", "TIGHT", "SAFE", "SAFE"]


@pytest.fixture
def degrading_sequence() -> List[str]:
    """Status sequence showing degradation."""
    return ["SAFE", "SAFE", "TIGHT", "TIGHT", "STARVED", "STARVED"]


@pytest.fixture
def stable_sequence() -> List[str]:
    """Status sequence showing stability."""
    return ["TIGHT", "TIGHT", "TIGHT", "TIGHT", "TIGHT", "TIGHT"]


@pytest.fixture
def mixed_sequence() -> List[str]:
    """Status sequence with fluctuation but overall stable."""
    # Same average in both halves: [TIGHT, SAFE, TIGHT] avg = [TIGHT, SAFE, TIGHT] avg
    return ["TIGHT", "SAFE", "TIGHT", "TIGHT", "SAFE", "TIGHT"]


@pytest.fixture
def sample_health_json(tmp_path: Path) -> Path:
    """Create a sample health JSON file."""
    data = {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "health_report": [
            {
                "slice": "slice_uplift_goal",
                "mode": "rfl",
                "health": {
                    "status": "TIGHT",
                    "metrics": {
                        "budget_exhausted_pct": 2.5,
                        "timeout_abstentions_avg": 0.3,
                        "max_candidates_hit_pct": 85.0,
                    },
                },
            },
            {
                "slice": "slice_uplift_sparse",
                "mode": "rfl",
                "health": {
                    "status": "SAFE",
                    "metrics": {
                        "budget_exhausted_pct": 0.5,
                        "timeout_abstentions_avg": 0.05,
                        "max_candidates_hit_pct": 95.0,
                    },
                },
            },
        ],
    }
    path = tmp_path / "run1.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def multiple_health_jsons(tmp_path: Path) -> List[Path]:
    """Create multiple health JSON files showing a trend."""
    paths = []
    
    # Run 1: STARVED
    run1 = {
        "health_report": [
            {
                "slice": "slice_test",
                "health": {"status": "STARVED", "metrics": {}},
            },
        ],
    }
    path1 = tmp_path / "run1.json"
    path1.write_text(json.dumps(run1))
    paths.append(path1)
    
    # Run 2: TIGHT
    run2 = {
        "health_report": [
            {
                "slice": "slice_test",
                "health": {"status": "TIGHT", "metrics": {}},
            },
        ],
    }
    path2 = tmp_path / "run2.json"
    path2.write_text(json.dumps(run2))
    paths.append(path2)
    
    # Run 3: SAFE
    run3 = {
        "health_report": [
            {
                "slice": "slice_test",
                "health": {"status": "SAFE", "metrics": {}},
            },
        ],
    }
    path3 = tmp_path / "run3.json"
    path3.write_text(json.dumps(run3))
    paths.append(path3)
    
    return paths


# =============================================================================
# Test: Health Score Conversion
# =============================================================================


class TestHealthScore:
    """Tests for health_to_score() function."""

    def test_safe_score(self):
        """SAFE should have lowest score."""
        assert health_to_score("SAFE") == 0

    def test_tight_score(self):
        """TIGHT should be between SAFE and STARVED."""
        assert health_to_score("TIGHT") == 1

    def test_starved_score(self):
        """STARVED should be higher than TIGHT."""
        assert health_to_score("STARVED") == 2

    def test_invalid_score(self):
        """INVALID should have highest score."""
        assert health_to_score("INVALID") == 3

    def test_unknown_status_score(self):
        """Unknown status should default to INVALID score."""
        assert health_to_score("UNKNOWN") == 3
        assert health_to_score("GARBAGE") == 3

    def test_score_ordering(self):
        """Verify score ordering matches HEALTH_ORDER."""
        for i, status in enumerate(HEALTH_ORDER):
            assert health_to_score(status) == i


# =============================================================================
# Test: Trend Classification
# =============================================================================


class TestTrendClassification:
    """Tests for classify_trend() function."""

    def test_improving_trend(self, improving_sequence: List[str]):
        """Test IMPROVING classification."""
        trend = classify_trend(improving_sequence)
        assert trend == TrendDirection.IMPROVING

    def test_degrading_trend(self, degrading_sequence: List[str]):
        """Test DEGRADING classification."""
        trend = classify_trend(degrading_sequence)
        assert trend == TrendDirection.DEGRADING

    def test_stable_trend(self, stable_sequence: List[str]):
        """Test STABLE classification."""
        trend = classify_trend(stable_sequence)
        assert trend == TrendDirection.STABLE

    def test_mixed_trend_is_stable(self, mixed_sequence: List[str]):
        """Test mixed sequence is classified as STABLE."""
        trend = classify_trend(mixed_sequence)
        # Mixed oscillation should be STABLE
        assert trend == TrendDirection.STABLE

    def test_single_status_unknown(self):
        """Test single status returns UNKNOWN."""
        trend = classify_trend(["SAFE"])
        assert trend == TrendDirection.UNKNOWN

    def test_empty_sequence_unknown(self):
        """Test empty sequence returns UNKNOWN."""
        trend = classify_trend([])
        assert trend == TrendDirection.UNKNOWN

    def test_two_status_improving(self):
        """Test two-status sequence: STARVED → SAFE."""
        trend = classify_trend(["STARVED", "SAFE"])
        assert trend == TrendDirection.IMPROVING

    def test_two_status_degrading(self):
        """Test two-status sequence: SAFE → STARVED."""
        trend = classify_trend(["SAFE", "STARVED"])
        assert trend == TrendDirection.DEGRADING


# =============================================================================
# Test: Multi-Run Analysis
# =============================================================================


class TestMultiRunAnalysis:
    """Tests for analyze_trends() function."""

    def test_analyze_improving_trend(self, multiple_health_jsons: List[Path]):
        """Test analysis detects improving trend."""
        report = analyze_trends(multiple_health_jsons)
        
        assert len(report.slices) == 1
        assert report.slices[0].slice_name == "slice_test"
        assert report.slices[0].trend == TrendDirection.IMPROVING

    def test_analyze_summary_counts(self, multiple_health_jsons: List[Path]):
        """Test summary counts are correct."""
        report = analyze_trends(multiple_health_jsons)
        
        assert report.summary["total_slices"] == 1
        assert report.summary["improving"] == 1
        assert report.summary["stable"] == 0
        assert report.summary["degrading"] == 0

    def test_analyze_status_sequence(self, multiple_health_jsons: List[Path]):
        """Test status sequence is captured."""
        report = analyze_trends(multiple_health_jsons)
        
        assert report.slices[0].status_sequence == ["STARVED", "TIGHT", "SAFE"]

    def test_analyze_preserves_input_order(self, sample_health_json: Path, tmp_path: Path):
        """Test inputs are recorded in order."""
        # Create second file
        data2 = {
            "health_report": [
                {"slice": "slice_uplift_goal", "health": {"status": "SAFE", "metrics": {}}},
            ],
        }
        path2 = tmp_path / "run2.json"
        path2.write_text(json.dumps(data2))
        
        report = analyze_trends([sample_health_json, path2])
        
        assert str(sample_health_json) in report.inputs
        assert str(path2) in report.inputs


# =============================================================================
# Test: Output Formatting
# =============================================================================


class TestOutputFormatting:
    """Tests for output formatting functions."""

    def test_format_markdown_valid(self, multiple_health_jsons: List[Path]):
        """Test Markdown output is valid."""
        report = analyze_trends(multiple_health_jsons)
        md = format_markdown(report)
        
        assert "##" in md  # Header
        assert "|" in md  # Table
        assert "IMPROVING" in md or "📈" in md

    def test_format_markdown_contains_emoji(self, multiple_health_jsons: List[Path]):
        """Test Markdown contains trend emoji."""
        report = analyze_trends(multiple_health_jsons)
        md = format_markdown(report)
        
        # Should contain at least one trend emoji
        assert "📈" in md or "➡️" in md or "📉" in md

    def test_format_json_valid(self, multiple_health_jsons: List[Path]):
        """Test JSON output is valid."""
        report = analyze_trends(multiple_health_jsons)
        json_str = format_json(report)
        
        # Should be parseable
        data = json.loads(json_str)
        assert "slices" in data
        assert "summary" in data

    def test_format_json_structure(self, multiple_health_jsons: List[Path]):
        """Test JSON has expected structure."""
        report = analyze_trends(multiple_health_jsons)
        json_str = format_json(report)
        data = json.loads(json_str)
        
        assert data["phase"] == "PHASE II — NOT USED IN PHASE I"
        assert len(data["slices"]) == 1
        assert data["slices"][0]["trend"] == "IMPROVING"


# =============================================================================
# Test: SliceTrend.to_dict()
# =============================================================================


class TestSliceTrendToDict:
    """Tests for SliceTrend.to_dict() method."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all expected fields."""
        trend = SliceTrend(
            slice_name="test_slice",
            runs=[
                RunHealth(run_id="r1", path="p1", slice_name="test_slice", status="SAFE"),
                RunHealth(run_id="r2", path="p2", slice_name="test_slice", status="TIGHT"),
            ],
            trend=TrendDirection.DEGRADING,
            status_sequence=["SAFE", "TIGHT"],
        )
        d = trend.to_dict()
        
        assert d["slice_name"] == "test_slice"
        assert d["trend"] == "DEGRADING"
        assert d["status_sequence"] == ["SAFE", "TIGHT"]
        assert d["num_runs"] == 2
        assert d["first_status"] == "SAFE"
        assert d["last_status"] == "TIGHT"


# =============================================================================
# Test: Read-Only Invariant
# =============================================================================


class TestReadOnlyInvariant:
    """Tests verifying trend analyzer is read-only."""

    def test_analyze_does_not_modify_files(self, multiple_health_jsons: List[Path]):
        """Test analyze_trends does not modify input files."""
        # Read original contents
        original_contents = [p.read_text() for p in multiple_health_jsons]
        
        # Run analysis
        _ = analyze_trends(multiple_health_jsons)
        
        # Verify files unchanged
        for path, original in zip(multiple_health_jsons, original_contents):
            assert path.read_text() == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

