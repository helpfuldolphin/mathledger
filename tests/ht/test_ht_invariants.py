"""
Tests for the HT-Series Invariant Checker

This module contains tests for the functions and classes in
backend.ht.ht_invariant_checker, ensuring they correctly implement the
invariants from HT_INVARIANT_SPEC_v1.md.
"""

import pytest
from backend.ht.ht_invariant_checker import (
    baseline_shuffle,
    rfl_order,
    compute_cycle_seed,
)

# Mark all tests in this file as 'ht_spec'
pytestmark = pytest.mark.ht_spec

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mdap_seed() -> int:
    """Provides the default MDAP seed from the spec."""
    return 0x4D444150

@pytest.fixture
def experiment_id() -> str:
    """Provides a sample experiment ID."""
    return "uplift_u2_goal_001"

@pytest.fixture
def candidates() -> list:
    """Provides a sample list of candidates for shuffling/ordering."""
    return [
        "formula_A",
        "formula_B",
        "formula_C",
        "formula_D",
        "formula_E",
        "formula_F",
    ]

# ==============================================================================
# Tests for INV-DETERMINISM-*
# ==============================================================================

def test_baseline_shuffle_is_deterministic(candidates, experiment_id, mdap_seed):
    """
    Tests INV-DETERMINISM-1: The baseline policy random shuffle MUST use ONLY
    the cycle seed as an entropy source.
    """
    cycle_index = 42

    # Shuffle the candidates twice with the same parameters
    shuffled_1 = baseline_shuffle(candidates, cycle_index, experiment_id, mdap_seed)
    shuffled_2 = baseline_shuffle(candidates, cycle_index, experiment_id, mdap_seed)

    # The results must be identical
    assert shuffled_1 == shuffled_2, "Shuffle with same seed must be deterministic"
    # Ensure the list was actually shuffled and not just returned in order
    assert shuffled_1 != candidates, "List should have been shuffled"

def test_baseline_shuffle_differs_with_seed(candidates, experiment_id, mdap_seed):
    """
    Tests INV-DETERMINISM-1: Verifies that different cycle seeds produce
    different shuffles.
    """
    cycle_index_1 = 42
    cycle_index_2 = 43

    # Shuffle with two different cycle indexes
    shuffled_1 = baseline_shuffle(candidates, cycle_index_1, experiment_id, mdap_seed)
    shuffled_2 = baseline_shuffle(candidates, cycle_index_2, experiment_id, mdap_seed)

    # The results must be different
    assert shuffled_1 != shuffled_2, "Shuffles with different seeds must differ"

def test_rfl_order_is_deterministic(candidates, experiment_id, mdap_seed):
    """
    Tests INV-DETERMINISM-2: RFL ordering is deterministic for the same inputs.
    """
    cycle_index = 101
    policy_weights = {"a": 0.1, "b": -0.5} # Dummy weights

    order_1 = rfl_order(candidates, policy_weights, cycle_index, experiment_id, mdap_seed)
    order_2 = rfl_order(candidates, policy_weights, cycle_index, experiment_id, mdap_seed)

    assert order_1 == order_2, "RFL ordering should be deterministic for the same inputs"

def test_rfl_order_differs_with_seed_for_tiebreak(experiment_id, mdap_seed):
    """
    Tests INV-DETERMINISM-2: The seed affects tie-breaking in RFL ordering.
    
    This test uses candidates that will have the same mock score.
    """
    # These candidates will have the same mock score in the stubbed function
    tie_candidates = ["A", "B", "C"] 
    policy_weights = {}
    
    # Ordering with two different cycle indexes
    order_1 = rfl_order(tie_candidates, policy_weights, 1, experiment_id, mdap_seed)
    order_2 = rfl_order(tie_candidates, policy_weights, 2, experiment_id, mdap_seed)

    # The tie-breaking should be different, resulting in a different order
    assert order_1 != order_2, "Different seeds should lead to different tie-breaking"

def test_cycle_seed_implementation(experiment_id, mdap_seed):
    """
    Tests the cycle seed generation against a known value from the spec's logic.
    """
    # Let's compute an expected value.
    # cycle_seed(c, e) = SHA256(
    #     DOMAIN_CYCLE_SEED ||
    #     BE32(MDAP_SEED) ||
    #     BE32(c) ||
    #     experiment_id_bytes(e)
    # )
    import hashlib
    import struct
    
    DOMAIN_CYCLE_SEED = b"MathLedger:CycleSeed:v2:"
    cycle_index = 1
    
    payload = (
        DOMAIN_CYCLE_SEED +
        struct.pack(">I", mdap_seed) +
        struct.pack(">I", cycle_index) +
        experiment_id.encode("utf-8")
    )
    expected_seed = hashlib.sha256(payload).digest()

    computed_seed = compute_cycle_seed(cycle_index, experiment_id, mdap_seed)

    assert computed_seed == expected_seed
