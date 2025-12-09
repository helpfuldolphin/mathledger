# tests/phase2/metrics_adversarial/__init__.py
"""
Phase II Adversarial Metric Fault Injection Suite

This package contains adversarial tests for the metrics substrate:
- Randomized field removal (fuzzing)
- Type mismatch injection (string/float/int swaps)
- Extreme value injection (IEEE 754 limits)
- Malformed parameter handling
- High-volume synthetic batches (10^5 cycles)
- Mutation detection with shadow implementations
- Equivalence oracle verification

All tests are hermetic, deterministic, and make NO modifications
to production metric code.

NO METRIC INTERPRETATION: These tests verify fault handling only.
"""

