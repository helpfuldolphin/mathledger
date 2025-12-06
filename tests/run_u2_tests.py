#!/usr/bin/env python3.11
"""
Simple test runner for U2 determinism tests (no pytest required).
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfl.prng import DeterministicPRNG, int_to_hex_seed
from experiments.u2 import (
    FrontierManager,
    BaselinePolicy,
    RFLPolicy,
)


def test_prng_same_seed_same_sequence():
    """Same seed produces same random sequence."""
    print("TEST: PRNG same seed same sequence...", end=" ")
    
    seed = "0x1234abcd"
    
    prng1 = DeterministicPRNG(seed)
    prng2 = DeterministicPRNG(seed)
    
    seq1 = [prng1.random() for _ in range(100)]
    seq2 = [prng2.random() for _ in range(100)]
    
    assert seq1 == seq2, "Same seed must produce same sequence"
    print("✓ PASS")


def test_prng_hierarchical_isolation():
    """Child PRNGs are isolated from each other."""
    print("TEST: PRNG hierarchical isolation...", end=" ")
    
    master = DeterministicPRNG("0xmaster")
    
    child1 = master.for_path("child1")
    child2 = master.for_path("child2")
    
    seq1 = [child1.random() for _ in range(10)]
    seq2 = [child2.random() for _ in range(10)]
    
    assert seq1 != seq2, "Different paths must produce different sequences"
    print("✓ PASS")


def test_prng_hierarchical_determinism():
    """Same path produces same child PRNG."""
    print("TEST: PRNG hierarchical determinism...", end=" ")
    
    master1 = DeterministicPRNG("0xmaster")
    master2 = DeterministicPRNG("0xmaster")
    
    child1 = master1.for_path("slice", "arithmetic")
    child2 = master2.for_path("slice", "arithmetic")
    
    seq1 = [child1.random() for _ in range(10)]
    seq2 = [child2.random() for _ in range(10)]
    
    assert seq1 == seq2, "Same path must produce same child PRNG"
    print("✓ PASS")


def test_prng_state_serialization():
    """PRNG state can be saved and restored."""
    print("TEST: PRNG state serialization...", end=" ")
    
    prng = DeterministicPRNG("0xtest")
    
    # Generate some values
    [prng.random() for _ in range(50)]
    
    # Save state
    state = prng.get_state()
    
    # Generate more values
    seq1 = [prng.random() for _ in range(10)]
    
    # Restore state
    prng.set_state(state)
    
    # Generate again
    seq2 = [prng.random() for _ in range(10)]
    
    assert seq1 == seq2, "Restored PRNG must produce same sequence"
    print("✓ PASS")


def test_prng_integer_seed_conversion():
    """Integer seeds are converted deterministically."""
    print("TEST: PRNG integer seed conversion...", end=" ")
    
    seed_int = 12345
    seed_hex = int_to_hex_seed(seed_int)
    
    prng1 = DeterministicPRNG(seed_int)
    prng2 = DeterministicPRNG(seed_hex)
    
    seq1 = [prng1.random() for _ in range(10)]
    seq2 = [prng2.random() for _ in range(10)]
    
    assert seq1 == seq2, "Integer and hex seeds must be equivalent"
    print("✓ PASS")


def test_frontier_push_pop_determinism():
    """Same operations produce same frontier state."""
    print("TEST: Frontier push/pop determinism...", end=" ")
    
    prng1 = DeterministicPRNG("0xfrontier")
    prng2 = DeterministicPRNG("0xfrontier")
    
    frontier1 = FrontierManager(max_beam_width=10, prng=prng1)
    frontier2 = FrontierManager(max_beam_width=10, prng=prng2)
    
    # Push same candidates
    for i in range(20):
        frontier1.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
        frontier2.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
    
    # Pop all candidates
    seq1 = []
    while not frontier1.is_empty():
        c = frontier1.pop()
        if c:
            seq1.append(c.item)
    
    seq2 = []
    while not frontier2.is_empty():
        c = frontier2.pop()
        if c:
            seq2.append(c.item)
    
    assert seq1 == seq2, "Same operations must produce same pop sequence"
    print("✓ PASS")


def test_frontier_state_serialization():
    """Frontier state can be saved and restored."""
    print("TEST: Frontier state serialization...", end=" ")
    
    prng = DeterministicPRNG("0xfrontier")
    frontier = FrontierManager(max_beam_width=10, prng=prng)
    
    # Push candidates
    for i in range(15):
        frontier.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
    
    # Pop some
    [frontier.pop() for _ in range(5)]
    
    # Save state
    state = frontier.get_state()
    
    # Pop more
    seq1 = []
    for _ in range(3):
        if not frontier.is_empty():
            c = frontier.pop()
            if c:
                seq1.append(c.item)
    
    # Restore state
    frontier.set_state(state)
    
    # Pop again
    seq2 = []
    for _ in range(3):
        if not frontier.is_empty():
            c = frontier.pop()
            if c:
                seq2.append(c.item)
    
    assert seq1 == seq2, "Restored frontier must produce same sequence"
    print("✓ PASS")


def test_baseline_policy_determinism():
    """Baseline policy produces deterministic rankings."""
    print("TEST: Baseline policy determinism...", end=" ")
    
    prng1 = DeterministicPRNG("0xpolicy")
    prng2 = DeterministicPRNG("0xpolicy")
    
    policy1 = BaselinePolicy(prng1)
    policy2 = BaselinePolicy(prng2)
    
    candidates = [f"item_{i}" for i in range(20)]
    
    ranked1 = policy1.rank(candidates)
    ranked2 = policy2.rank(candidates)
    
    items1 = [item for item, _ in ranked1]
    items2 = [item for item, _ in ranked2]
    
    assert items1 == items2, "Same seed must produce same ranking"
    print("✓ PASS")


def test_rfl_policy_determinism():
    """RFL policy produces deterministic rankings."""
    print("TEST: RFL policy determinism...", end=" ")
    
    prng1 = DeterministicPRNG("0xpolicy")
    prng2 = DeterministicPRNG("0xpolicy")
    
    feedback = {
        "item_0": {"success_rate": 0.8},
        "item_5": {"success_rate": 0.6},
        "item_10": {"success_rate": 0.9},
    }
    
    policy1 = RFLPolicy(prng1, feedback)
    policy2 = RFLPolicy(prng2, feedback)
    
    candidates = [{"item": f"item_{i}", "depth": i % 3} for i in range(20)]
    
    ranked1 = policy1.rank(candidates)
    ranked2 = policy2.rank(candidates)
    
    items1 = [str(item) for item, _ in ranked1]
    items2 = [str(item) for item, _ in ranked2]
    
    assert items1 == items2, "Same seed must produce same RFL ranking"
    print("✓ PASS")


def main():
    """Run all tests."""
    print("=" * 60)
    print("U2 PLANNER DETERMINISM TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_prng_same_seed_same_sequence,
        test_prng_hierarchical_isolation,
        test_prng_hierarchical_determinism,
        test_prng_state_serialization,
        test_prng_integer_seed_conversion,
        test_frontier_push_pop_determinism,
        test_frontier_state_serialization,
        test_baseline_policy_determinism,
        test_rfl_policy_determinism,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
