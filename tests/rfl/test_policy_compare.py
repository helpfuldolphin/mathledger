"""
Tests for RFL Policy Comparison Tool (TASK 2).

Validates:
- Distance computations (L1, L2)
- Sign flip detection
- Top-K delta selection
- Handling of mismatched feature sets
- JSONL vs JSON loading
- CLI functionality
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile

from rfl.policy import PolicyState
from rfl.policy.compare import (
    compare_policy_states,
    load_policy_from_json,
    ComparisonResult,
)


class TestComparisonBasics:
    """Test basic comparison operations."""
    
    def test_identical_policies(self):
        """Test comparing identical policies."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        
        assert result.l2_distance == pytest.approx(0.0)
        assert result.l1_distance == pytest.approx(0.0)
        assert result.num_sign_flips == 0
    
    def test_simple_difference(self):
        """Test comparing policies with simple differences."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 2.0, "feat_b": 3.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        
        # L2: sqrt((2-1)^2 + (3-2)^2) = sqrt(2)
        expected_l2 = np.sqrt(2.0)
        assert result.l2_distance == pytest.approx(expected_l2)
        
        # L1: |2-1| + |3-2| = 2
        assert result.l1_distance == pytest.approx(2.0)
        
        # No sign flips (both positive in both states)
        assert result.num_sign_flips == 0
    
    def test_l2_distance_computation(self):
        """Test L2 distance is computed correctly."""
        state_a = PolicyState(
            weights={"feat_a": 3.0, "feat_b": 4.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 0.0, "feat_b": 0.0},
            step=1,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        
        # L2: sqrt(3^2 + 4^2) = 5
        assert result.l2_distance == pytest.approx(5.0)
    
    def test_l1_distance_computation(self):
        """Test L1 distance is computed correctly."""
        state_a = PolicyState(
            weights={"feat_a": 3.0, "feat_b": -2.0, "feat_c": 1.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 1.0, "feat_c": 1.0},
            step=1,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        
        # L1: |1-3| + |1-(-2)| + |1-1| = 2 + 3 + 0 = 5
        assert result.l1_distance == pytest.approx(5.0)


class TestSignFlips:
    """Test sign flip detection."""
    
    def test_no_sign_flips(self):
        """Test no sign flips when all weights maintain sign."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": -2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 2.0, "feat_b": -1.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        assert result.num_sign_flips == 0
    
    def test_single_sign_flip(self):
        """Test single sign flip detection."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": -2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": -1.0, "feat_b": -2.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        assert result.num_sign_flips == 1
    
    def test_multiple_sign_flips(self):
        """Test multiple sign flips."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": -2.0, "feat_c": 3.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": -1.0, "feat_b": 2.0, "feat_c": 3.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        assert result.num_sign_flips == 2
    
    def test_zero_to_nonzero_not_sign_flip(self):
        """Test that 0 -> nonzero is not counted as sign flip."""
        state_a = PolicyState(
            weights={"feat_a": 0.0, "feat_b": 0.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 1.0, "feat_b": -1.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b)
        # 0 is not considered a "sign" for flip counting
        assert result.num_sign_flips == 0


class TestTopKDeltas:
    """Test top-K delta selection."""
    
    def test_top_k_selection(self):
        """Test top-K features by absolute delta are selected correctly."""
        state_a = PolicyState(
            weights={
                "feat_a": 1.0,
                "feat_b": 2.0,
                "feat_c": 3.0,
                "feat_d": 4.0,
            },
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={
                "feat_a": 1.5,   # delta = 0.5
                "feat_b": 4.0,   # delta = 2.0
                "feat_c": 3.1,   # delta = 0.1
                "feat_d": 0.0,   # delta = 4.0
            },
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b, top_k=3)
        
        # Should get feat_d (4.0), feat_b (2.0), feat_a (0.5)
        assert len(result.top_k_deltas) == 3
        
        # Check order (largest absolute delta first)
        assert result.top_k_deltas[0][0] == "feat_d"
        assert abs(result.top_k_deltas[0][3]) == pytest.approx(4.0)
        
        assert result.top_k_deltas[1][0] == "feat_b"
        assert abs(result.top_k_deltas[1][3]) == pytest.approx(2.0)
        
        assert result.top_k_deltas[2][0] == "feat_a"
        assert abs(result.top_k_deltas[2][3]) == pytest.approx(0.5)
    
    def test_top_k_with_negative_deltas(self):
        """Test top-K works with negative deltas."""
        state_a = PolicyState(
            weights={"feat_a": 5.0, "feat_b": 5.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 0.0, "feat_b": 3.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b, top_k=2)
        
        # feat_a has delta of -5.0, feat_b has delta of -2.0
        # Should be ordered by absolute value: feat_a, then feat_b
        assert result.top_k_deltas[0][0] == "feat_a"
        assert result.top_k_deltas[0][3] == pytest.approx(-5.0)


class TestMismatchedFeatureSets:
    """Test handling of mismatched feature sets."""
    
    def test_error_on_mismatch(self):
        """Test error is raised when feature sets don't match."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 1.0, "feat_c": 3.0},  # feat_c instead of feat_b
            step=2,
            seed=42
        )
        
        with pytest.raises(ValueError, match="Feature sets don't match"):
            compare_policy_states(state_a, state_b, handle_missing="error")
    
    def test_union_handling(self):
        """Test union mode treats missing features as 0.0."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 1.0, "feat_c": 3.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b, handle_missing="union")
        
        # Should compute over union: {feat_a, feat_b, feat_c}
        # feat_b is 2.0 in A, 0.0 in B (delta = 2.0)
        # feat_c is 0.0 in A, 3.0 in B (delta = 3.0)
        # L2: sqrt(0^2 + 2^2 + 3^2) = sqrt(13)
        assert result.l2_distance == pytest.approx(np.sqrt(13))
        
        # Check feature set diff is recorded
        assert "feat_b" in result.feature_set_diff["only_in_a"]
        assert "feat_c" in result.feature_set_diff["only_in_b"]
    
    def test_intersection_handling(self):
        """Test intersection mode uses only common features."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 2.0, "feat_c": 3.0},
            step=2,
            seed=42
        )
        
        result = compare_policy_states(state_a, state_b, handle_missing="intersection")
        
        # Should only compare feat_a
        # L2: |2 - 1| = 1
        assert result.l2_distance == pytest.approx(1.0)
        assert result.l1_distance == pytest.approx(1.0)


class TestSerialization:
    """Test loading from files and serialization."""
    
    def test_load_from_json(self):
        """Test loading policy from JSON file."""
        state = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0},
            step=5,
            total_reward=10.0,
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "policy.json"
            with open(filepath, 'w') as f:
                json.dump(state.to_dict(), f)
            
            loaded = load_policy_from_json(filepath)
            
            assert loaded.weights == state.weights
            assert loaded.step == state.step
    
    def test_load_from_jsonl_last_line(self):
        """Test loading from JSONL file (default: last line)."""
        state1 = PolicyState(weights={"feat_a": 1.0}, step=1, seed=42)
        state2 = PolicyState(weights={"feat_a": 2.0}, step=2, seed=42)
        state3 = PolicyState(weights={"feat_a": 3.0}, step=3, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "policy.jsonl"
            with open(filepath, 'w') as f:
                f.write(json.dumps(state1.to_dict()) + '\n')
                f.write(json.dumps(state2.to_dict()) + '\n')
                f.write(json.dumps(state3.to_dict()) + '\n')
            
            loaded = load_policy_from_json(filepath)
            
            # Should load last line (state3)
            assert loaded.step == 3
            assert loaded.weights["feat_a"] == 3.0
    
    def test_load_from_jsonl_specific_index(self):
        """Test loading specific line from JSONL file."""
        state1 = PolicyState(weights={"feat_a": 1.0}, step=1, seed=42)
        state2 = PolicyState(weights={"feat_a": 2.0}, step=2, seed=42)
        state3 = PolicyState(weights={"feat_a": 3.0}, step=3, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "policy.jsonl"
            with open(filepath, 'w') as f:
                f.write(json.dumps(state1.to_dict()) + '\n')
                f.write(json.dumps(state2.to_dict()) + '\n')
                f.write(json.dumps(state3.to_dict()) + '\n')
            
            loaded = load_policy_from_json(filepath, index=1)
            
            # Should load second line (index 1 = state2)
            assert loaded.step == 2
            assert loaded.weights["feat_a"] == 2.0
    
    def test_comparison_result_to_dict(self):
        """Test ComparisonResult can be serialized."""
        state_a = PolicyState(weights={"feat_a": 1.0}, step=1, seed=42)
        state_b = PolicyState(weights={"feat_a": 2.0}, step=2, seed=42)
        
        result = compare_policy_states(state_a, state_b)
        result_dict = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0
        
        # Check required fields
        assert "l2_distance" in result_dict
        assert "l1_distance" in result_dict
        assert "num_sign_flips" in result_dict
        assert "top_k_deltas" in result_dict


class TestDeterminism:
    """Test deterministic behavior of comparison."""
    
    def test_comparison_is_deterministic(self):
        """Test same inputs produce same outputs."""
        state_a = PolicyState(
            weights={"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0},
            step=1,
            seed=42
        )
        state_b = PolicyState(
            weights={"feat_a": 2.0, "feat_b": 1.0, "feat_c": 4.0},
            step=2,
            seed=42
        )
        
        result1 = compare_policy_states(state_a, state_b, top_k=5)
        result2 = compare_policy_states(state_a, state_b, top_k=5)
        
        assert result1.l2_distance == result2.l2_distance
        assert result1.l1_distance == result2.l1_distance
        assert result1.num_sign_flips == result2.num_sign_flips
        assert result1.top_k_deltas == result2.top_k_deltas


class TestCLIFunctionality:
    """Test CLI can be invoked (integration-style test)."""
    
    def test_cli_module_exists(self):
        """Test __main__.py module exists and can be imported."""
        from rfl.policy import __main__
        assert hasattr(__main__, 'main')
