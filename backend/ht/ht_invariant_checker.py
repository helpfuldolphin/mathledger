"""
HT-Series Invariant Checker for MathLedger Phase II

This module implements the verification logic for the invariants defined in
HT_INVARIANT_SPEC_v1.md.
"""

import hashlib
import json
import struct
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ==============================================================================
# Domain Prefixes (from §18 of the spec)
# ==============================================================================

DOMAIN_SLICE_CONFIG = b"MathLedger:SliceConfig:v2:"
DOMAIN_CYCLE_SEED = b"MathLedger:CycleSeed:v2:"
DOMAIN_SEED_SCHEDULE = b"MathLedger:SeedSchedule:v2:"
DOMAIN_HT_CYCLE = b"MathLedger:HtCycle:v2:"
DOMAIN_HT_CHAIN = b"MathLedger:HtChain:v2:"
DOMAIN_PREREG_BIND = b"MathLedger:PreregBinding:v2:"
DOMAIN_SUCCESS_METRIC = b"MathLedger:SuccessMetric:v2:"
DOMAIN_MDAP_ATTEST = b"MathLedger:MDAPAttestation:v2:"
DOMAIN_HT_MDAP_BIND = b"MathLedger:HtMdapBinding:v2:"

# ==============================================================================
# Helper Functions
# ==============================================================================

def canonical_json(obj: Dict[str, Any]) -> bytes:
    """Computes the canonical JSON representation of an object."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')

def compute_cycle_seed(cycle_index: int, experiment_id: str, mdap_seed: int) -> bytes:
    """Computes the deterministic seed for a given cycle."""
    payload = (
        DOMAIN_CYCLE_SEED +
        struct.pack(">I", mdap_seed) +
        struct.pack(">I", cycle_index) +
        experiment_id.encode("utf-8")
    )
    return hashlib.sha256(payload).digest()

def cycle_seed_int(cycle_seed_bytes: bytes) -> int:
    """Returns the first 8 bytes of the cycle seed as a 64-bit integer."""
    return int.from_bytes(cycle_seed_bytes[:8], "big")

# ==============================================================================
# Invariant Implementation (as per spec)
# ==============================================================================

class HtInvariantChecker:
    """
    A class to verify Phase II invariants based on the specification document.
    """

    def __init__(self, manifest: Dict, prereg: Dict, ht_series: Dict):
        self.manifest = manifest
        self.prereg = prereg
        self.ht_series = ht_series
        self.failures = []

    def run_all_checks(self) -> List[str]:
        """Runs all invariant checks and returns a list of failures."""
        self.failures = []

        # Run checks for each invariant group
        self._check_prereg_invariants()
        self._check_success_metric_invariants()
        self._check_mdap_attest_invariants()
        self._check_determinism_invariants()
        self._check_ht_crypto_invariants()

        return self.failures

    def _add_failure(self, inv_id: str, message: str):
        self.failures.append(f"[{inv_id}] FAILED: {message}")

    # --- INV-PREREG-* ---
    def _check_prereg_invariants(self):
        """Verifies preregistration binding invariants."""
        # Placeholder for INV-PREREG-* implementations
        pass

    # --- INV-SUCCESS-* ---
    def _check_success_metric_invariants(self):
        """Verifies success metric binding invariants."""
        # Placeholder for INV-SUCCESS-* implementations
        pass

    # --- INV-MDAP-ATTEST-* ---
    def _check_mdap_attest_invariants(self):
        """Verifies MDAP seed schedule attestation."""
        # Placeholder for INV-MDAP-ATTEST-* implementations
        pass

    # --- INV-DETERMINISM-* ---
    def _check_determinism_invariants(self):
        """Verifies deterministic behavior of policies."""
        # This is harder to check post-hoc and is more of an implementation mandate.
        # We can check for the presence of the functions.
        # Actual verification would require re-running with the same inputs.
        pass

    # --- INV-HT-CRYPTO-* ---
    def _check_ht_crypto_invariants(self):
        """Verifies the cryptographic guarantees of the Ht chain."""
        # Placeholder for INV-HT-CRYPTO-* implementations
        pass


# ==============================================================================
# Standalone Functions from Spec (for use in other modules)
# ==============================================================================

def baseline_shuffle(candidates: list, cycle_index: int, experiment_id: str, mdap_seed: int) -> list:
    """
    Shuffles candidates deterministically based on the cycle seed.
    Conforms to INV-DETERMINISM-1.
    """
    seed_bytes = compute_cycle_seed(cycle_index, experiment_id, mdap_seed)
    seed_as_int = cycle_seed_int(seed_bytes)
    
    rng = random.Random(seed_as_int)
    shuffled_candidates = candidates.copy()
    rng.shuffle(shuffled_candidates)
    
    return shuffled_candidates

def rfl_order(candidates: list, policy_weights: dict, cycle_index: int, experiment_id: str, mdap_seed: int) -> list:
    """
    Orders candidates by RFL policy score, using the cycle seed for tie-breaking.
    Conforms to INV-DETERMINISM-2.
    """
    # This is a stub; actual implementation requires feature vectors and a scoring function.
    def compute_policy_score(candidate, weights):
        # In a real implementation, this would compute a score based on features.
        # For this stub, we'll use a hash as a mock score.
        return int.from_bytes(hashlib.sha256(str(candidate).encode()).digest()[:4], 'little')

    seed_bytes = compute_cycle_seed(cycle_index, experiment_id, mdap_seed)
    seed_as_int = cycle_seed_int(seed_bytes)
    rng = random.Random(seed_as_int)

    def score_with_tiebreak(candidate):
        base_score = compute_policy_score(candidate, policy_weights)
        tiebreak = rng.random() * 1e-12
        return (base_score, tiebreak)

    return sorted(candidates, key=score_with_tiebreak, reverse=True)
