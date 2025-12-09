"""
Phase II Verifier Budget Loader Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for backend/verification/budget_loader.py

PHASE II — NOT USED IN PHASE I

Test coverage:
    1. Valid slice loading → merged VerifierBudget object
    2. Missing file → FileNotFoundError
    3. Missing defaults → KeyError
    4. Missing slice → KeyError
    5. Invalid types → ValueError
    6. Default budget loading
    7. Phase II slice detection
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from backend.verification.budget_loader import (
    VerifierBudget,
    load_budget_for_slice,
    load_default_budget,
    is_phase2_slice,
    DEFAULT_CONFIG_PATH,
)


class TestVerifierBudget:
    """Tests for VerifierBudget dataclass."""
    
    def test_valid_budget_creation(self):
        """Valid budget parameters should create immutable object."""
        budget = VerifierBudget(
            cycle_budget_s=5.0,
            taut_timeout_s=0.10,
            max_candidates_per_cycle=40,
        )
        assert budget.cycle_budget_s == 5.0
        assert budget.taut_timeout_s == 0.10
        assert budget.max_candidates_per_cycle == 40
        # Test alias
        assert budget.max_candidates == 40
    
    def test_budget_is_immutable(self):
        """VerifierBudget should be frozen (immutable)."""
        budget = VerifierBudget(
            cycle_budget_s=5.0,
            taut_timeout_s=0.10,
            max_candidates_per_cycle=40,
        )
        with pytest.raises(AttributeError):
            budget.cycle_budget_s = 10.0
    
    def test_zero_cycle_budget_valid(self):
        """Zero cycle_budget_s should be valid (means immediately expired)."""
        budget = VerifierBudget(
            cycle_budget_s=0.0,
            taut_timeout_s=0.10,
            max_candidates_per_cycle=40,
        )
        assert budget.cycle_budget_s == 0.0
    
    def test_zero_taut_timeout_valid(self):
        """Zero taut_timeout_s should be valid (means no timeout)."""
        budget = VerifierBudget(
            cycle_budget_s=5.0,
            taut_timeout_s=0.0,
            max_candidates_per_cycle=40,
        )
        assert budget.taut_timeout_s == 0.0
    
    def test_negative_cycle_budget_raises(self):
        """Negative cycle_budget_s should raise ValueError."""
        with pytest.raises(ValueError, match="cycle_budget_s must be >= 0.0"):
            VerifierBudget(
                cycle_budget_s=-1.0,
                taut_timeout_s=0.10,
                max_candidates_per_cycle=40,
            )
    
    def test_negative_taut_timeout_raises(self):
        """Negative taut_timeout_s should raise ValueError."""
        with pytest.raises(ValueError, match="taut_timeout_s must be >= 0.0"):
            VerifierBudget(
                cycle_budget_s=5.0,
                taut_timeout_s=-0.1,
                max_candidates_per_cycle=40,
            )
    
    def test_zero_max_candidates_raises(self):
        """Zero max_candidates_per_cycle should raise ValueError."""
        with pytest.raises(ValueError, match="max_candidates_per_cycle must be >= 1"):
            VerifierBudget(
                cycle_budget_s=5.0,
                taut_timeout_s=0.10,
                max_candidates_per_cycle=0,
            )
    
    def test_negative_max_candidates_raises(self):
        """Negative max_candidates_per_cycle should raise ValueError."""
        with pytest.raises(ValueError, match="max_candidates_per_cycle must be >= 1"):
            VerifierBudget(
                cycle_budget_s=5.0,
                taut_timeout_s=0.10,
                max_candidates_per_cycle=-5,
            )
    
    def test_wrong_type_cycle_budget_raises(self):
        """Non-numeric cycle_budget_s should raise TypeError."""
        with pytest.raises(TypeError, match="cycle_budget_s must be a number"):
            VerifierBudget(
                cycle_budget_s="not_a_number",
                taut_timeout_s=0.10,
                max_candidates_per_cycle=40,
            )
    
    def test_wrong_type_taut_timeout_raises(self):
        """Non-numeric taut_timeout_s should raise TypeError."""
        with pytest.raises(TypeError, match="taut_timeout_s must be a number"):
            VerifierBudget(
                cycle_budget_s=5.0,
                taut_timeout_s="invalid",
                max_candidates_per_cycle=40,
            )
    
    def test_wrong_type_max_candidates_raises(self):
        """Non-integer max_candidates_per_cycle should raise TypeError."""
        with pytest.raises(TypeError, match="max_candidates_per_cycle must be an integer"):
            VerifierBudget(
                cycle_budget_s=5.0,
                taut_timeout_s=0.10,
                max_candidates_per_cycle=40.5,
            )
    
    def test_int_coerced_to_float_for_time_fields(self):
        """Integer values for time fields should work."""
        budget = VerifierBudget(
            cycle_budget_s=5,  # int, not float
            taut_timeout_s=1,  # int, not float
            max_candidates_per_cycle=40,
        )
        assert budget.cycle_budget_s == 5
        assert budget.taut_timeout_s == 1


class TestLoadBudgetForSlice:
    """Tests for load_budget_for_slice function."""
    
    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid budget config file."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": 5.0,
                "taut_timeout_s": 0.10,
                "max_candidates_per_cycle": 40,
            },
            "slices": {
                "slice_uplift_goal": {
                    "cycle_budget_s": 5.0,
                    "taut_timeout_s": 0.10,
                    "max_candidates_per_cycle": 40,
                },
                "slice_uplift_sparse": {
                    "cycle_budget_s": 6.0,
                    "taut_timeout_s": 0.12,
                    "max_candidates_per_cycle": 40,
                },
                "slice_uplift_tree": {
                    "cycle_budget_s": 4.0,
                    "taut_timeout_s": 0.10,
                    "max_candidates_per_cycle": 30,
                },
            },
        }
        config_path = tmp_path / "verifier_budget_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    def test_load_valid_slice(self, valid_config_file: Path):
        """Valid slice should return merged VerifierBudget."""
        budget = load_budget_for_slice("slice_uplift_goal", valid_config_file)
        
        assert budget.cycle_budget_s == 5.0
        assert budget.taut_timeout_s == 0.10
        assert budget.max_candidates_per_cycle == 40
    
    def test_load_slice_with_overrides(self, valid_config_file: Path):
        """Slice-specific overrides should take precedence."""
        budget = load_budget_for_slice("slice_uplift_sparse", valid_config_file)
        
        # These are overridden from defaults
        assert budget.cycle_budget_s == 6.0
        assert budget.taut_timeout_s == 0.12
        # This inherits from defaults
        assert budget.max_candidates_per_cycle == 40
    
    def test_load_slice_with_all_overrides(self, valid_config_file: Path):
        """Slice with all overrides should use all slice values."""
        budget = load_budget_for_slice("slice_uplift_tree", valid_config_file)
        
        assert budget.cycle_budget_s == 4.0
        assert budget.taut_timeout_s == 0.10
        assert budget.max_candidates_per_cycle == 30
    
    def test_missing_file_raises(self, tmp_path: Path):
        """Missing config file should raise FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError, match="Verifier budget config not found"):
            load_budget_for_slice("slice_uplift_goal", missing_path)
    
    def test_missing_defaults_raises(self, tmp_path: Path):
        """Missing defaults block should raise KeyError."""
        config = {
            "version": 1,
            "slices": {
                "slice_uplift_goal": {
                    "cycle_budget_s": 5.0,
                    "taut_timeout_s": 0.10,
                    "max_candidates_per_cycle": 40,
                },
            },
        }
        config_path = tmp_path / "no_defaults.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with pytest.raises(KeyError, match="missing 'defaults' block"):
            load_budget_for_slice("slice_uplift_goal", config_path)
    
    def test_missing_slices_block_raises(self, tmp_path: Path):
        """Missing slices block should raise KeyError."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": 5.0,
                "taut_timeout_s": 0.10,
                "max_candidates_per_cycle": 40,
            },
        }
        config_path = tmp_path / "no_slices.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with pytest.raises(KeyError, match="missing 'slices' block"):
            load_budget_for_slice("slice_uplift_goal", config_path)
    
    def test_missing_slice_raises(self, valid_config_file: Path):
        """Requesting non-existent slice should raise KeyError."""
        with pytest.raises(KeyError, match="not found in verifier budget config"):
            load_budget_for_slice("slice_that_does_not_exist", valid_config_file)
    
    def test_missing_required_param_raises(self, tmp_path: Path):
        """Missing required parameter in defaults should raise KeyError."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": 5.0,
                "taut_timeout_s": 0.10,
                # Missing max_candidates_per_cycle
            },
            "slices": {
                "slice_uplift_goal": {},
            },
        }
        config_path = tmp_path / "missing_param.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with pytest.raises(KeyError, match="max_candidates_per_cycle"):
            load_budget_for_slice("slice_uplift_goal", config_path)
    
    def test_invalid_type_raises(self, tmp_path: Path):
        """Invalid type for budget parameter should raise ValueError."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": "not_a_number",  # Invalid type
                "taut_timeout_s": 0.10,
                "max_candidates_per_cycle": 40,
            },
            "slices": {
                "slice_uplift_goal": {},
            },
        }
        config_path = tmp_path / "invalid_type.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with pytest.raises(ValueError, match="must be number"):
            load_budget_for_slice("slice_uplift_goal", config_path)


class TestLoadDefaultBudget:
    """Tests for load_default_budget function."""
    
    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid budget config file."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": 5.0,
                "taut_timeout_s": 0.10,
                "max_candidates_per_cycle": 40,
            },
            "slices": {},
        }
        config_path = tmp_path / "verifier_budget_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    def test_load_default_budget(self, valid_config_file: Path):
        """Default budget should return values from defaults block."""
        budget = load_default_budget(valid_config_file)
        
        assert budget.cycle_budget_s == 5.0
        assert budget.taut_timeout_s == 0.10
        assert budget.max_candidates_per_cycle == 40


class TestIsPhase2Slice:
    """Tests for is_phase2_slice function."""
    
    def test_phase2_slices(self):
        """Phase II slices should return True."""
        assert is_phase2_slice("slice_uplift_goal") is True
        assert is_phase2_slice("slice_uplift_sparse") is True
        assert is_phase2_slice("slice_uplift_tree") is True
        assert is_phase2_slice("slice_uplift_dependency") is True
        assert is_phase2_slice("slice_uplift_anything") is True
    
    def test_non_phase2_slices(self):
        """Non-Phase II slices should return False."""
        assert is_phase2_slice("first-organism-slice") is False
        assert is_phase2_slice("slice_medium") is False
        assert is_phase2_slice("core") is False
        assert is_phase2_slice("warmup") is False
        assert is_phase2_slice("") is False


class TestDeterminism:
    """Tests for deterministic behavior."""
    
    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid budget config file."""
        config = {
            "version": 1,
            "defaults": {
                "cycle_budget_s": 5.0,
                "taut_timeout_s": 0.10,
                "max_candidates_per_cycle": 40,
            },
            "slices": {
                "slice_uplift_goal": {
                    "cycle_budget_s": 5.0,
                    "taut_timeout_s": 0.10,
                    "max_candidates_per_cycle": 40,
                },
            },
        }
        config_path = tmp_path / "verifier_budget_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    def test_same_file_same_result(self, valid_config_file: Path):
        """Same file should produce same result."""
        budget1 = load_budget_for_slice("slice_uplift_goal", valid_config_file)
        budget2 = load_budget_for_slice("slice_uplift_goal", valid_config_file)
        
        assert budget1.cycle_budget_s == budget2.cycle_budget_s
        assert budget1.taut_timeout_s == budget2.taut_timeout_s
        assert budget1.max_candidates_per_cycle == budget2.max_candidates_per_cycle
    
    def test_no_global_state(self, valid_config_file: Path):
        """Loading should not modify global state."""
        # Load twice, should not have side effects
        load_budget_for_slice("slice_uplift_goal", valid_config_file)
        load_budget_for_slice("slice_uplift_goal", valid_config_file)
        # If this passes without exception, no global state issues


class TestRealConfig:
    """Tests against the actual config file (if present)."""
    
    @pytest.mark.skipif(
        not Path(DEFAULT_CONFIG_PATH).exists(),
        reason="Real config file not present"
    )
    def test_real_config_loads(self):
        """Real config file should load without errors."""
        budget = load_budget_for_slice("slice_uplift_goal")
        
        assert budget.cycle_budget_s > 0
        assert budget.taut_timeout_s > 0
        assert budget.max_candidates_per_cycle > 0
    
    @pytest.mark.skipif(
        not Path(DEFAULT_CONFIG_PATH).exists(),
        reason="Real config file not present"
    )
    def test_all_slices_load(self):
        """All documented slices should load from real config."""
        slices = [
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        ]
        
        for slice_name in slices:
            budget = load_budget_for_slice(slice_name)
            assert budget.cycle_budget_s > 0, f"Invalid cycle_budget_s for {slice_name}"
            assert budget.taut_timeout_s > 0, f"Invalid taut_timeout_s for {slice_name}"
            assert budget.max_candidates_per_cycle > 0, f"Invalid max_candidates for {slice_name}"

