"""
Deterministic PRNG for MathLedger U2 Planner

INVARIANTS:
- All randomness is seeded and reproducible
- Hierarchical seeding ensures slice-level isolation
- PRNG state is serializable for snapshot/replay
- Cross-platform determinism (OS/machine independent)
"""

from .deterministic_prng import (
    DeterministicPRNG,
    int_to_hex_seed,
    hex_to_int_seed,
    canonicalize_seed,
)

# Governance exports
from .governance import (
    DriftStatus,
    build_prng_drift_radar,
    build_prng_governance_tile,
    attach_prng_tile_to_evidence,
    attach_prng_governance_tile,
    build_prng_drift_ledger,
    attach_prng_drift_ledger_to_evidence,
)

# Calibration integration exports
from .calibration_integration import (
    align_prng_drift_to_windows,
    compute_per_window_prng_volatility_deltas,
    compute_prng_confounded_windows,
    build_prng_window_audit_table,
    build_prng_calibration_annex,
    attach_prng_calibration_annex_to_evidence,
)

__all__ = [
    "DeterministicPRNG",
    "int_to_hex_seed",
    "hex_to_int_seed",
    "canonicalize_seed",
    # Governance
    "DriftStatus",
    "build_prng_drift_radar",
    "build_prng_governance_tile",
    "attach_prng_tile_to_evidence",
    "attach_prng_governance_tile",
    "build_prng_drift_ledger",
    "attach_prng_drift_ledger_to_evidence",
    # Calibration integration
    "align_prng_drift_to_windows",
    "compute_per_window_prng_volatility_deltas",
    "compute_prng_confounded_windows",
    "build_prng_window_audit_table",
    "build_prng_calibration_annex",
    "attach_prng_calibration_annex_to_evidence",
]
