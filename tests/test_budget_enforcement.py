"""
Test suite for Phase II Budget Enforcement.

Tests verify that:
    1. Budget exhaustion (cycle_budget_s) triggers budget_exhausted flag
    2. Max candidates limit triggers max_candidates_hit flag
    3. No-budget runs behave identically to pre-budget baseline
    4. Budget configuration loads correctly from YAML
    5. VerifierBudget validation works correctly

These tests ensure budget enforcement is a pure runtime safety mechanism
that does NOT change the scientific meaning of success/failure/abstain.

PHASE II ONLY — These tests target Phase II uplift slices.
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.verification.budget_loader import (
    VerifierBudget,
    load_budget_for_slice,
    load_default_budget,
    is_phase2_slice,
    DEFAULT_CONFIG_PATH,
)
from derivation.pipeline import (
    DerivationPipeline,
    PipelineStats,
    DerivationSummary,
    run_slice_for_test,
    StatementRecord,
)
from derivation.bounds import SliceBounds
from derivation.verification import StatementVerifier


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_bounds() -> SliceBounds:
    """Create minimal bounds for fast test execution."""
    return SliceBounds(
        max_atoms=2,
        max_formula_depth=2,
        max_mp_depth=1,
        max_breadth=10,
        max_total=20,
        max_axiom_instances=5,
        max_formula_pool=10,
        lean_timeout_s=0.001,
    )


@pytest.fixture
def minimal_budget() -> VerifierBudget:
    """Create a minimal budget for testing."""
    return VerifierBudget(
        cycle_budget_s=10.0,
        taut_timeout_s=1.0,
        max_candidates_per_cycle=50,
    )


@pytest.fixture
def tiny_cycle_budget() -> VerifierBudget:
    """Create a budget with very short cycle time to trigger exhaustion."""
    return VerifierBudget(
        cycle_budget_s=0.001,  # 1ms - will definitely be exceeded
        taut_timeout_s=1.0,
        max_candidates_per_cycle=1000,
    )


@pytest.fixture
def single_candidate_budget() -> VerifierBudget:
    """Create a budget that allows only one candidate."""
    return VerifierBudget(
        cycle_budget_s=300.0,
        taut_timeout_s=1.0,
        max_candidates_per_cycle=1,
    )


@pytest.fixture
def temp_budget_config(tmp_path: Path) -> Path:
    """Create a temporary budget config file."""
    config_content = """
version: 1

defaults:
  cycle_budget_s: 100.0
  taut_timeout_s: 2.0
  max_candidates_per_cycle: 50

slices:
  test_slice:
    cycle_budget_s: 50.0
    taut_timeout_s: 1.0
    max_candidates_per_cycle: 25
  partial_slice:
    cycle_budget_s: 30.0
"""
    config_path = tmp_path / "test_budget.yaml"
    config_path.write_text(config_content)
    return config_path


# =============================================================================
# VerifierBudget Tests
# =============================================================================


class TestVerifierBudget:
    """Tests for VerifierBudget dataclass."""

    def test_budget_creation_valid(self):
        """Test creating a valid budget."""
        budget = VerifierBudget(
            cycle_budget_s=60.0,
            taut_timeout_s=5.0,
            max_candidates_per_cycle=100,
        )
        assert budget.cycle_budget_s == 60.0
        assert budget.taut_timeout_s == 5.0
        assert budget.max_candidates_per_cycle == 100

    def test_budget_max_candidates_alias(self):
        """Test that max_candidates is an alias for max_candidates_per_cycle."""
        budget = VerifierBudget(
            cycle_budget_s=60.0,
            taut_timeout_s=5.0,
            max_candidates_per_cycle=100,
        )
        assert budget.max_candidates == 100
        assert budget.max_candidates == budget.max_candidates_per_cycle

    def test_budget_validation_negative_cycle(self):
        """Test that negative cycle_budget_s raises ValueError."""
        with pytest.raises(ValueError, match="cycle_budget_s must be positive"):
            VerifierBudget(
                cycle_budget_s=-1.0,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=10,
            )

    def test_budget_validation_zero_cycle(self):
        """Test that zero cycle_budget_s raises ValueError."""
        with pytest.raises(ValueError, match="cycle_budget_s must be positive"):
            VerifierBudget(
                cycle_budget_s=0.0,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=10,
            )

    def test_budget_validation_negative_timeout(self):
        """Test that negative taut_timeout_s raises ValueError."""
        with pytest.raises(ValueError, match="taut_timeout_s must be positive"):
            VerifierBudget(
                cycle_budget_s=10.0,
                taut_timeout_s=-1.0,
                max_candidates_per_cycle=10,
            )

    def test_budget_validation_negative_candidates(self):
        """Test that negative max_candidates_per_cycle raises ValueError."""
        with pytest.raises(ValueError, match="max_candidates_per_cycle must be positive"):
            VerifierBudget(
                cycle_budget_s=10.0,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=-1,
            )

    def test_budget_validation_zero_candidates(self):
        """Test that zero max_candidates_per_cycle raises ValueError."""
        with pytest.raises(ValueError, match="max_candidates_per_cycle must be positive"):
            VerifierBudget(
                cycle_budget_s=10.0,
                taut_timeout_s=1.0,
                max_candidates_per_cycle=0,
            )

    def test_budget_immutable(self):
        """Test that budget is frozen (immutable)."""
        budget = VerifierBudget(
            cycle_budget_s=60.0,
            taut_timeout_s=5.0,
            max_candidates_per_cycle=100,
        )
        with pytest.raises(AttributeError):
            budget.cycle_budget_s = 120.0  # type: ignore


# =============================================================================
# Budget Loader Tests
# =============================================================================


class TestBudgetLoader:
    """Tests for budget configuration loading."""

    def test_load_budget_for_slice_with_override(self, temp_budget_config: Path):
        """Test loading budget for a slice with overrides."""
        budget = load_budget_for_slice("test_slice", str(temp_budget_config))
        assert budget.cycle_budget_s == 50.0  # Overridden
        assert budget.taut_timeout_s == 1.0  # Overridden
        assert budget.max_candidates_per_cycle == 25  # Overridden

    def test_load_budget_for_slice_partial_override(self, temp_budget_config: Path):
        """Test loading budget for a slice with partial overrides."""
        budget = load_budget_for_slice("partial_slice", str(temp_budget_config))
        assert budget.cycle_budget_s == 30.0  # Overridden
        assert budget.taut_timeout_s == 2.0  # Default
        assert budget.max_candidates_per_cycle == 50  # Default

    def test_load_budget_for_unknown_slice_raises(self, temp_budget_config: Path):
        """Test loading budget for unknown slice raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            load_budget_for_slice("unknown_slice", str(temp_budget_config))

    def test_load_budget_file_not_found(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_budget_for_slice("test", "/nonexistent/path.yaml")

    def test_load_default_budget(self, temp_budget_config: Path):
        """Test loading default budget."""
        budget = load_default_budget(str(temp_budget_config))
        assert budget.cycle_budget_s == 100.0
        assert budget.taut_timeout_s == 2.0
        assert budget.max_candidates_per_cycle == 50

    def test_is_phase2_slice(self):
        """Test Phase II slice detection."""
        assert is_phase2_slice("slice_uplift_goal") is True
        assert is_phase2_slice("slice_uplift_sparse") is True
        assert is_phase2_slice("slice_uplift_tree") is True
        assert is_phase2_slice("slice_medium") is False
        assert is_phase2_slice("atoms4-depth4") is False


# =============================================================================
# Pipeline Stats Tests
# =============================================================================


class TestPipelineStats:
    """Tests for PipelineStats budget fields."""

    def test_stats_budget_fields_default(self):
        """Test that budget fields have correct defaults."""
        stats = PipelineStats()
        assert stats.budget_exhausted is False
        assert stats.max_candidates_hit is False
        assert stats.timeout_abstentions == 0
        assert stats.budget_remaining_s == -1.0

    def test_stats_budget_fields_mutable(self):
        """Test that budget fields can be set."""
        stats = PipelineStats()
        stats.budget_exhausted = True
        stats.max_candidates_hit = True
        stats.timeout_abstentions = 5
        stats.budget_remaining_s = 10.5
        
        assert stats.budget_exhausted is True
        assert stats.max_candidates_hit is True
        assert stats.timeout_abstentions == 5
        assert stats.budget_remaining_s == 10.5


# =============================================================================
# Budget Enforcement Tests
# =============================================================================


class TestBudgetExhaustion:
    """Tests for cycle_budget_s enforcement."""

    def test_budget_exhaustion_triggers_flag(
        self, 
        minimal_bounds: SliceBounds,
        tiny_cycle_budget: VerifierBudget,
    ):
        """
        Test that exceeding cycle_budget_s sets budget_exhausted=True.
        
        Uses a 1ms budget which will definitely be exceeded.
        """
        verifier = StatementVerifier(minimal_bounds, None)
        pipeline = DerivationPipeline(
            minimal_bounds, 
            verifier,
            budget=tiny_cycle_budget,
        )
        
        # Create some seed statements to process
        seeds = [
            StatementRecord(
                normalized="p",
                hash="abc123",
                pretty="p",
                rule="seed",
                is_axiom=True,
                mp_depth=0,
                parents=(),
                verification_method="seed",
            ),
        ]
        
        # Add a small delay to ensure budget is exceeded
        time.sleep(0.002)
        
        outcome = pipeline.run_step(seeds, budget=tiny_cycle_budget)
        
        # Budget should be exhausted
        assert outcome.stats.budget_exhausted is True
        assert outcome.stats.budget_remaining_s == 0.0

    def test_budget_exhaustion_in_run_slice(self, tiny_cycle_budget: VerifierBudget):
        """Test budget exhaustion through run_slice_for_test entry point."""
        from curriculum.gates import (
            CurriculumSlice,
            SliceGates,
            CoverageGateSpec,
            AbstentionGateSpec,
            VelocityGateSpec,
            CapsGateSpec,
        )
        
        # Create a minimal slice config
        gates = SliceGates(
            coverage=CoverageGateSpec(ci_lower_min=0.01, sample_min=1, require_attestation=False),
            abstention=AbstentionGateSpec(max_rate_pct=100.0, max_mass=1000),
            velocity=VelocityGateSpec(min_pph=0.01, stability_cv_max=1.0, window_minutes=1),
            caps=CapsGateSpec(min_attempt_mass=1, min_runtime_minutes=0.001, backlog_max=1.0),
        )
        slice_cfg = CurriculumSlice(
            name="budget_test_slice",
            params={"atoms": 2, "depth_max": 2, "breadth_max": 5, "total_max": 10},
            gates=gates,
        )
        
        # Small delay to ensure budget exhaustion
        time.sleep(0.002)
        
        result = run_slice_for_test(
            slice_cfg,
            budget=tiny_cycle_budget,
            emit_log=False,
        )
        
        assert result.stats.budget_exhausted is True
        assert result.summary.budget_exhausted is True


class TestMaxCandidatesEnforcement:
    """Tests for max_candidates enforcement."""

    def test_max_candidates_stops_early(
        self,
        minimal_bounds: SliceBounds,
        single_candidate_budget: VerifierBudget,
    ):
        """
        Test that max_candidates=1 stops after one candidate.
        """
        verifier = StatementVerifier(minimal_bounds, None)
        pipeline = DerivationPipeline(
            minimal_bounds,
            verifier,
            budget=single_candidate_budget,
        )
        
        outcome = pipeline.run_step([], budget=single_candidate_budget)
        
        # Should hit max_candidates limit
        # Note: may be 0 or 1 depending on whether any candidates were found
        assert outcome.stats.candidates_considered <= 1
        if outcome.stats.candidates_considered >= 1:
            assert outcome.stats.max_candidates_hit is True

    def test_max_candidates_in_run_slice(self):
        """Test max_candidates through run_slice_for_test entry point."""
        from curriculum.gates import (
            CurriculumSlice,
            SliceGates,
            CoverageGateSpec,
            AbstentionGateSpec,
            VelocityGateSpec,
            CapsGateSpec,
        )
        
        gates = SliceGates(
            coverage=CoverageGateSpec(ci_lower_min=0.01, sample_min=1, require_attestation=False),
            abstention=AbstentionGateSpec(max_rate_pct=100.0, max_mass=1000),
            velocity=VelocityGateSpec(min_pph=0.01, stability_cv_max=1.0, window_minutes=1),
            caps=CapsGateSpec(min_attempt_mass=1, min_runtime_minutes=0.001, backlog_max=1.0),
        )
        slice_cfg = CurriculumSlice(
            name="max_candidates_test_slice",
            params={"atoms": 2, "depth_max": 2, "breadth_max": 100, "total_max": 100},
            gates=gates,
        )
        
        # Budget with small max_candidates
        small_budget = VerifierBudget(
            cycle_budget_s=300.0,
            taut_timeout_s=1.0,
            max_candidates_per_cycle=3,
        )
        
        result = run_slice_for_test(
            slice_cfg,
            budget=small_budget,
            emit_log=False,
        )
        
        # Should not exceed max_candidates
        assert result.stats.candidates_considered <= 3
        # If we processed 3 candidates, max_candidates_hit should be True
        if result.stats.candidates_considered >= 3:
            assert result.stats.max_candidates_hit is True


class TestNoBudgetBaseline:
    """Tests that no-budget runs match pre-budget behavior."""

    def test_no_budget_no_change(self, minimal_bounds: SliceBounds):
        """
        Test that budget=None produces baseline behavior.
        
        Budget fields should remain at defaults.
        """
        verifier = StatementVerifier(minimal_bounds, None)
        pipeline = DerivationPipeline(minimal_bounds, verifier, budget=None)
        
        outcome = pipeline.run_step([])
        
        # Budget fields should be at defaults
        assert outcome.stats.budget_exhausted is False
        assert outcome.stats.max_candidates_hit is False
        assert outcome.stats.timeout_abstentions == 0
        # budget_remaining_s should be -1 (no budget)
        assert outcome.stats.budget_remaining_s == -1.0

    def test_no_budget_in_run_slice(self):
        """Test that run_slice_for_test without budget matches baseline."""
        from curriculum.gates import (
            CurriculumSlice,
            SliceGates,
            CoverageGateSpec,
            AbstentionGateSpec,
            VelocityGateSpec,
            CapsGateSpec,
        )
        
        gates = SliceGates(
            coverage=CoverageGateSpec(ci_lower_min=0.01, sample_min=1, require_attestation=False),
            abstention=AbstentionGateSpec(max_rate_pct=100.0, max_mass=1000),
            velocity=VelocityGateSpec(min_pph=0.01, stability_cv_max=1.0, window_minutes=1),
            caps=CapsGateSpec(min_attempt_mass=1, min_runtime_minutes=0.001, backlog_max=1.0),
        )
        slice_cfg = CurriculumSlice(
            name="no_budget_test_slice",
            params={"atoms": 2, "depth_max": 2, "breadth_max": 10, "total_max": 20},
            gates=gates,
        )
        
        # Run without budget
        result = run_slice_for_test(
            slice_cfg,
            budget=None,  # Explicit None
            emit_log=False,
        )
        
        # Budget fields should indicate no budget enforcement
        assert result.stats.budget_exhausted is False
        assert result.stats.max_candidates_hit is False
        assert result.summary.budget_exhausted is False
        assert result.summary.max_candidates_hit is False


class TestDerivationSummaryBudget:
    """Tests for budget fields in DerivationSummary."""

    def test_summary_to_dict_includes_budget(self):
        """Test that to_dict() includes budget section."""
        summary = DerivationSummary(
            slice_name="test",
            n_candidates=10,
            n_verified=5,
            n_abstain=3,
            abstained_statements=(),
            budget_exhausted=True,
            max_candidates_hit=False,
            timeout_abstentions=2,
            budget_remaining_s=0.0,
        )
        
        d = summary.to_dict()
        
        assert "budget" in d
        assert d["budget"]["exhausted"] is True
        assert d["budget"]["max_candidates_hit"] is False
        assert d["budget"]["timeout_abstentions"] == 2
        assert d["budget"]["remaining_s"] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestBudgetIntegration:
    """Integration tests for budget enforcement with real slices."""

    def test_load_production_budget_config(self):
        """Test loading the production budget config file."""
        config_path = Path("config/verifier_budget_phase2.yaml")
        if not config_path.exists():
            pytest.skip("Production budget config not found")
        
        budget = load_default_budget(str(config_path))
        assert budget.cycle_budget_s > 0
        assert budget.taut_timeout_s > 0
        assert budget.max_candidates_per_cycle > 0

    def test_load_budget_for_uplift_slices(self):
        """Test loading budget for Phase II uplift slices."""
        config_path = Path("config/verifier_budget_phase2.yaml")
        if not config_path.exists():
            pytest.skip("Production budget config not found")
        
        # Test all defined Phase II slices
        for slice_name in [
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        ]:
            budget = load_budget_for_slice(slice_name, str(config_path))
            assert budget.cycle_budget_s > 0
            assert budget.taut_timeout_s > 0
            assert budget.max_candidates_per_cycle > 0


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
