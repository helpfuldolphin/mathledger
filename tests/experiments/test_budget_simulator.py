"""
Tests for Budget What-If Simulator (Agent A5 - Advanced Observability)

PHASE II — NOT USED IN PHASE I

Tests verify:
    1. Heuristic model produces deterministic outputs
    2. Scaling formulas are correct
    3. Health re-classification works
    4. Clear "heuristic only" warnings
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.budget_simulator import (
    estimate_exhausted_pct,
    estimate_timeout_avg,
    SimulationScenario,
    SimulationResult,
    simulate_scenario,
    load_baseline,
    format_markdown,
    format_json,
    MIN_EXHAUSTED_PCT,
    MIN_TIMEOUT_AVG,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def baseline_metrics() -> Dict[str, float]:
    """Baseline metrics for simulation."""
    return {
        "budget_exhausted_pct": 4.0,  # Near TIGHT/STARVED boundary
        "timeout_abstentions_avg": 0.5,  # TIGHT
        "max_candidates_hit_pct": 85.0,
    }


@pytest.fixture
def sample_health_json(tmp_path: Path) -> Path:
    """Create a sample health JSON file for simulation."""
    data = {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "health_report": [
            {
                "slice": "slice_uplift_goal",
                "mode": "rfl",
                "health": {
                    "status": "TIGHT",
                    "metrics": {
                        "budget_exhausted_pct": 3.0,
                        "timeout_abstentions_avg": 0.4,
                        "max_candidates_hit_pct": 90.0,
                    },
                },
            },
        ],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    return path


# =============================================================================
# Test: Heuristic Model - Exhausted Percentage
# =============================================================================


class TestExhaustedEstimate:
    """Tests for estimate_exhausted_pct() function."""

    def test_double_budget_halves_exhausted(self):
        """Doubling budget should roughly halve exhausted percentage."""
        baseline = 10.0
        estimated = estimate_exhausted_pct(baseline, 2.0)
        assert estimated == pytest.approx(5.0, rel=0.01)

    def test_halve_budget_doubles_exhausted(self):
        """Halving budget should roughly double exhausted percentage."""
        baseline = 5.0
        estimated = estimate_exhausted_pct(baseline, 0.5)
        assert estimated == pytest.approx(10.0, rel=0.01)

    def test_no_change_same_value(self):
        """Scale of 1.0 should return same value."""
        baseline = 3.5
        estimated = estimate_exhausted_pct(baseline, 1.0)
        assert estimated == pytest.approx(baseline, rel=0.01)

    def test_minimum_clamping(self):
        """Very low estimates should be clamped to minimum."""
        baseline = 0.1
        estimated = estimate_exhausted_pct(baseline, 100.0)  # 100x budget
        assert estimated >= MIN_EXHAUSTED_PCT

    def test_maximum_clamping(self):
        """Very high estimates should be clamped to maximum."""
        baseline = 50.0
        estimated = estimate_exhausted_pct(baseline, 0.01)  # 1% budget
        assert estimated <= 100.0

    def test_deterministic(self):
        """Same inputs should always produce same outputs."""
        baseline = 4.5
        scale = 1.5
        
        result1 = estimate_exhausted_pct(baseline, scale)
        result2 = estimate_exhausted_pct(baseline, scale)
        
        assert result1 == result2

    def test_invalid_scale_raises(self):
        """Zero or negative scale should raise ValueError."""
        with pytest.raises(ValueError):
            estimate_exhausted_pct(5.0, 0.0)
        with pytest.raises(ValueError):
            estimate_exhausted_pct(5.0, -1.0)


# =============================================================================
# Test: Heuristic Model - Timeout Average
# =============================================================================


class TestTimeoutEstimate:
    """Tests for estimate_timeout_avg() function."""

    def test_double_timeout_halves_abstentions(self):
        """Doubling timeout should roughly halve abstentions."""
        baseline = 1.0
        estimated = estimate_timeout_avg(baseline, 2.0)
        assert estimated == pytest.approx(0.5, rel=0.01)

    def test_halve_timeout_doubles_abstentions(self):
        """Halving timeout should roughly double abstentions."""
        baseline = 0.5
        estimated = estimate_timeout_avg(baseline, 0.5)
        assert estimated == pytest.approx(1.0, rel=0.01)

    def test_no_change_same_value(self):
        """Scale of 1.0 should return same value."""
        baseline = 0.3
        estimated = estimate_timeout_avg(baseline, 1.0)
        assert estimated == pytest.approx(baseline, rel=0.01)

    def test_minimum_clamping(self):
        """Very low estimates should be clamped to minimum."""
        baseline = 0.01
        estimated = estimate_timeout_avg(baseline, 100.0)
        assert estimated >= MIN_TIMEOUT_AVG

    def test_deterministic(self):
        """Same inputs should always produce same outputs."""
        baseline = 0.75
        scale = 1.25
        
        result1 = estimate_timeout_avg(baseline, scale)
        result2 = estimate_timeout_avg(baseline, scale)
        
        assert result1 == result2

    def test_invalid_scale_raises(self):
        """Zero or negative scale should raise ValueError."""
        with pytest.raises(ValueError):
            estimate_timeout_avg(0.5, 0.0)
        with pytest.raises(ValueError):
            estimate_timeout_avg(0.5, -1.0)


# =============================================================================
# Test: Simulation Scenario
# =============================================================================


class TestSimulationScenario:
    """Tests for SimulationScenario dataclass."""

    def test_describe_no_changes(self):
        """Test description when no changes."""
        scenario = SimulationScenario(budget_scale=1.0, timeout_scale=1.0)
        assert scenario.describe() == "no changes"

    def test_describe_budget_increase(self):
        """Test description for budget increase."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=1.0)
        desc = scenario.describe()
        assert "budget" in desc
        assert "increased" in desc

    def test_describe_budget_decrease(self):
        """Test description for budget decrease."""
        scenario = SimulationScenario(budget_scale=0.5, timeout_scale=1.0)
        desc = scenario.describe()
        assert "budget" in desc
        assert "decreased" in desc

    def test_describe_combined(self):
        """Test description for combined changes."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=0.5)
        desc = scenario.describe()
        assert "budget" in desc
        assert "timeout" in desc


# =============================================================================
# Test: Full Simulation
# =============================================================================


class TestSimulateScenario:
    """Tests for simulate_scenario() function."""

    def test_simulation_improves_health(self, baseline_metrics: Dict[str, float]):
        """Test that doubling budget can improve health."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=2.0)
        
        result = simulate_scenario(
            slice_name="test_slice",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        # Simulated metrics should be better
        assert result.simulated_metrics["budget_exhausted_pct"] < baseline_metrics["budget_exhausted_pct"]
        assert result.simulated_metrics["timeout_abstentions_avg"] < baseline_metrics["timeout_abstentions_avg"]

    def test_simulation_degrades_health(self, baseline_metrics: Dict[str, float]):
        """Test that halving budget can degrade health."""
        scenario = SimulationScenario(budget_scale=0.5, timeout_scale=0.5)
        
        result = simulate_scenario(
            slice_name="test_slice",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        # Simulated metrics should be worse
        assert result.simulated_metrics["budget_exhausted_pct"] > baseline_metrics["budget_exhausted_pct"]
        assert result.simulated_metrics["timeout_abstentions_avg"] > baseline_metrics["timeout_abstentions_avg"]

    def test_simulation_max_candidates_unchanged(self, baseline_metrics: Dict[str, float]):
        """Test that max_candidates_hit is not affected by budget scaling."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=2.0)
        
        result = simulate_scenario(
            slice_name="test_slice",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        assert result.simulated_metrics["max_candidates_hit_pct"] == baseline_metrics["max_candidates_hit_pct"]

    def test_simulation_status_changes(self, baseline_metrics: Dict[str, float]):
        """Test that status can change with significant budget increase."""
        # Use metrics that are close to STARVED
        starved_metrics = {
            "budget_exhausted_pct": 6.0,  # > 5% = STARVED
            "timeout_abstentions_avg": 1.5,  # > 1.0 = STARVED
            "max_candidates_hit_pct": 85.0,
        }
        
        # Triple the budget
        scenario = SimulationScenario(budget_scale=3.0, timeout_scale=3.0)
        
        result = simulate_scenario(
            slice_name="test_slice",
            baseline_metrics=starved_metrics,
            baseline_status="STARVED",
            scenario=scenario,
        )
        
        # Status should improve
        assert result.simulated_status != "STARVED"

    def test_simulation_deterministic(self, baseline_metrics: Dict[str, float]):
        """Test simulation is deterministic."""
        scenario = SimulationScenario(budget_scale=1.5, timeout_scale=1.5)
        
        result1 = simulate_scenario(
            slice_name="test",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        result2 = simulate_scenario(
            slice_name="test",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        assert result1.simulated_metrics == result2.simulated_metrics
        assert result1.simulated_status == result2.simulated_status


# =============================================================================
# Test: Load Baseline
# =============================================================================


class TestLoadBaseline:
    """Tests for load_baseline() function."""

    def test_load_health_json_format(self, sample_health_json: Path):
        """Test loading health JSON format."""
        data = load_baseline(sample_health_json)
        
        assert len(data) == 1
        assert data[0]["slice_name"] == "slice_uplift_goal"
        assert data[0]["status"] == "TIGHT"
        assert "budget_exhausted_pct" in data[0]["metrics"]


# =============================================================================
# Test: Output Formatting
# =============================================================================


class TestOutputFormatting:
    """Tests for output formatting functions."""

    def test_format_markdown_contains_warning(self, baseline_metrics: Dict[str, float]):
        """Test Markdown output contains heuristic warning."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=1.0)
        result = simulate_scenario(
            slice_name="test",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        md = format_markdown([result], scenario)
        
        assert "HEURISTIC" in md or "heuristic" in md

    def test_format_json_contains_warning(self, baseline_metrics: Dict[str, float]):
        """Test JSON output contains heuristic warning."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=1.0)
        result = simulate_scenario(
            slice_name="test",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        json_str = format_json([result])
        data = json.loads(json_str)
        
        assert "warning" in data
        assert "HEURISTIC" in data["warning"]

    def test_format_json_valid(self, baseline_metrics: Dict[str, float]):
        """Test JSON output is valid JSON."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=1.0)
        result = simulate_scenario(
            slice_name="test",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        
        json_str = format_json([result])
        data = json.loads(json_str)  # Should not raise
        
        assert "results" in data


# =============================================================================
# Test: SimulationResult.to_dict()
# =============================================================================


class TestSimulationResultToDict:
    """Tests for SimulationResult.to_dict() method."""

    def test_to_dict_contains_all_fields(self, baseline_metrics: Dict[str, float]):
        """Test to_dict includes all expected fields."""
        scenario = SimulationScenario(budget_scale=2.0, timeout_scale=1.5)
        result = simulate_scenario(
            slice_name="test_slice",
            baseline_metrics=baseline_metrics,
            baseline_status="TIGHT",
            scenario=scenario,
        )
        d = result.to_dict()
        
        assert d["slice_name"] == "test_slice"
        assert "scenario" in d
        assert "baseline" in d
        assert "simulated" in d
        assert "status_change" in d

    def test_to_dict_status_change_flag(self, baseline_metrics: Dict[str, float]):
        """Test status_change flag is correct."""
        # Scenario that changes status
        starved_metrics = {
            "budget_exhausted_pct": 10.0,
            "timeout_abstentions_avg": 2.0,
            "max_candidates_hit_pct": 85.0,
        }
        scenario = SimulationScenario(budget_scale=4.0, timeout_scale=4.0)
        result = simulate_scenario(
            slice_name="test",
            baseline_metrics=starved_metrics,
            baseline_status="STARVED",
            scenario=scenario,
        )
        d = result.to_dict()
        
        # Status should change from STARVED to something better
        assert d["status_change"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

