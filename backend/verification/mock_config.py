"""
Mock Oracle Configuration for Phase II Testing.

This module defines configuration dataclasses and profile constants for the
deterministic mock verification oracle. Used ONLY in tests and synthetic
experiments — never in production or Evidence Pack v1/v2.

ABSOLUTE SAFEGUARD: No uplift claims may reference mock oracle results.

================================================================================
PROFILE CONTRACTS — BEHAVIORAL GUARANTEES
================================================================================

The mock oracle provides CONTRACTUAL guarantees about bucket distributions.
These contracts are binding and verified by CI tests. Any change to these
values constitutes a breaking change requiring version bump.

CONTRACT VERSION: 1.0.0

PROFILE: default (Balanced Test Profile)
────────────────────────────────────────
  verified:  60%  (hash % 100 in [0, 60))
  failed:    15%  (hash % 100 in [60, 75))
  abstain:   10%  (hash % 100 in [75, 85))
  timeout:    8%  (hash % 100 in [85, 93))
  error:      4%  (hash % 100 in [93, 97))
  crash:      3%  (hash % 100 in [97, 100))
  TOTAL:    100%

PROFILE: goal_hit (Rare Success / Targeted Search)
──────────────────────────────────────────────────
  verified:  15%  — rare hits force policy learning
  failed:    35%
  abstain:   35%  — high abstention zone
  timeout:   10%
  error:      3%
  crash:      2%
  TOTAL:    100%

PROFILE: sparse (Wide Proof Space / Low Density)
────────────────────────────────────────────────
  verified:  25%  — sparse verification
  failed:    30%
  abstain:   30%  — wide abstention zone
  timeout:   10%
  error:      3%
  crash:      2%
  TOTAL:    100%

PROFILE: tree (Chain-Depth Patterns)
────────────────────────────────────
  verified:  45%  — medium for chain-building
  failed:    20%
  abstain:   20%
  timeout:   10%
  error:      3%
  crash:      2%
  TOTAL:    100%

PROFILE: dependency (Multi-Goal Coordination)
─────────────────────────────────────────────
  verified:  35%
  failed:    25%
  abstain:   25%
  timeout:   10%
  error:      3%
  crash:      2%
  TOTAL:    100%

NEGATIVE CONTROL CONTRACT:
──────────────────────────
When negative_control=True:
  - result.verified MUST be False (always)
  - result.abstained MUST be True (always)
  - result.timed_out MUST be False (always)
  - result.crashed MUST be False (always)
  - result.reason MUST be "negative_control"
  - result.bucket MUST be "negative_control"
  - Stats tracking MUST be suppressed
  - No exceptions raised (including crash bucket)

DETERMINISM CONTRACT:
─────────────────────
  - Same input + same config → identical output
  - Hash function: SHA-256(formula.encode("utf-8"))
  - Bucket selection: int(hash_hex, 16) % 100
  - No randomness outside hash-based branching
  - No external dependencies (network, files, time)

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ============================================================================
# CONTRACT VERSION
# ============================================================================
# Increment MAJOR for breaking changes to profile distributions.
# Increment MINOR for new profiles or non-breaking additions.
# Increment PATCH for documentation/test-only changes.

MOCK_ORACLE_CONTRACT_VERSION: str = "1.0.0"


# ============================================================================
# PROFILE CONTRACT DEFINITIONS (Authoritative Source)
# ============================================================================
# These are the CONTRACTUAL percentages. Tests verify SLICE_PROFILES matches.

PROFILE_CONTRACTS: Dict[str, Dict[str, float]] = {
    "default": {
        "verified": 60.0,
        "failed": 15.0,
        "abstain": 10.0,
        "timeout": 8.0,
        "error": 4.0,
        "crash": 3.0,
    },
    "goal_hit": {
        "verified": 15.0,
        "failed": 35.0,
        "abstain": 35.0,
        "timeout": 10.0,
        "error": 3.0,
        "crash": 2.0,
    },
    "sparse": {
        "verified": 25.0,
        "failed": 30.0,
        "abstain": 30.0,
        "timeout": 10.0,
        "error": 3.0,
        "crash": 2.0,
    },
    "tree": {
        "verified": 45.0,
        "failed": 20.0,
        "abstain": 20.0,
        "timeout": 10.0,
        "error": 3.0,
        "crash": 2.0,
    },
    "dependency": {
        "verified": 35.0,
        "failed": 25.0,
        "abstain": 25.0,
        "timeout": 10.0,
        "error": 3.0,
        "crash": 2.0,
    },
}


# ============================================================================
# NEGATIVE CONTROL CONTRACT
# ============================================================================

@dataclass(frozen=True)
class NegativeControlContract:
    """
    Contractual guarantees for negative control mode.
    
    These are INVARIANTS that must hold for ALL inputs when negative_control=True.
    """
    
    verified: bool = False          # MUST always be False
    abstained: bool = True          # MUST always be True
    timed_out: bool = False         # MUST always be False
    crashed: bool = False           # MUST always be False
    reason: str = "negative_control"  # MUST always be this string
    bucket: str = "negative_control"  # MUST always be this string
    stats_suppressed: bool = True   # Stats MUST NOT be updated


NEGATIVE_CONTROL_CONTRACT = NegativeControlContract()


@dataclass(frozen=True)
class MockOracleConfig:
    """
    Configuration for MockVerifiableOracle behavior.
    
    Attributes:
        slice_profile: Profile name determining bucket boundaries.
            One of: "goal_hit", "sparse", "tree", "dependency", "default"
        timeout_ms: Base timeout latency in milliseconds for timeout bucket.
        enable_crashes: If True, crash bucket raises MockOracleCrashError.
        latency_jitter_pct: Percentage jitter for latency (0.10 = ±10%).
        seed: Master seed for deterministic behavior (affects latency jitter).
        negative_control: If True, all evaluations return verified=False with
            reason="negative_control". Used for baseline/control experiments.
        target_hashes: Optional set of target formula hashes for goal_hit profile.
        required_goals: Optional set of required goal hashes for dependency profile.
        chain_depth_map: Optional mapping of formula_hash → chain depth for tree profile.
    """
    
    slice_profile: str = "default"
    timeout_ms: int = 50
    enable_crashes: bool = False
    latency_jitter_pct: float = 0.10
    seed: int = 0
    negative_control: bool = False
    target_hashes: FrozenSet[str] = field(default_factory=frozenset)
    required_goals: FrozenSet[str] = field(default_factory=frozenset)
    chain_depth_map: Optional[Dict[str, int]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_profiles = {"goal_hit", "sparse", "tree", "dependency", "default"}
        if self.slice_profile not in valid_profiles:
            raise ValueError(
                f"Invalid slice_profile '{self.slice_profile}'. "
                f"Must be one of: {valid_profiles}"
            )
        if self.timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if not (0.0 <= self.latency_jitter_pct <= 1.0):
            raise ValueError("latency_jitter_pct must be in [0.0, 1.0]")


@dataclass(frozen=True)
class MockVerificationResult:
    """
    Result from mock oracle verification.
    
    Exactly one of verified, abstained, timed_out, crashed should be True,
    OR all are False (meaning explicit failure/rejection).
    
    Attributes:
        verified: True if formula was "verified" by mock oracle.
        abstained: True if oracle abstained (inconclusive).
        timed_out: True if oracle simulated a timeout.
        crashed: True if oracle simulated a crash.
        reason: Human-readable reason string (e.g., "mock-verified", "mock-timeout").
        latency_ms: Simulated latency in milliseconds.
        bucket: The bucket name this result fell into.
        hash_int: The integer hash used for bucketing (for debugging).
    """
    
    verified: bool
    abstained: bool
    timed_out: bool
    crashed: bool
    reason: str
    latency_ms: int
    bucket: str = ""
    hash_int: int = 0
    
    def __post_init__(self) -> None:
        """Validate that result state is consistent."""
        flags = [self.verified, self.abstained, self.timed_out, self.crashed]
        true_count = sum(flags)
        # At most one flag should be True (or all False for explicit failure)
        if true_count > 1:
            raise ValueError(
                f"At most one of verified/abstained/timed_out/crashed can be True. "
                f"Got: verified={self.verified}, abstained={self.abstained}, "
                f"timed_out={self.timed_out}, crashed={self.crashed}"
            )


# ============================================================================
# SLICE PROFILES — Bucket boundary percentages for each profile
# ============================================================================
#
# Each profile maps to bucket boundaries using modular arithmetic on hash % 100.
# Format: {"verified": upper_bound, "failed": upper_bound, "abstain": upper_bound,
#          "timeout": upper_bound, "crash": 100}
#
# Bucket selection: h % 100 falls into first bucket where h % 100 < upper_bound
#
# Example for "default":
#   h % 100 in [0, 60)   → verified
#   h % 100 in [60, 75)  → failed
#   h % 100 in [75, 85)  → abstain
#   h % 100 in [85, 93)  → timeout
#   h % 100 in [93, 97)  → error (treated as failed with error reason)
#   h % 100 in [97, 100) → crash

SLICE_PROFILES: Dict[str, Dict[str, int]] = {
    # default: Balanced test profile
    "default": {
        "verified": 60,    # 0-59   (60%)
        "failed": 75,      # 60-74  (15%)
        "abstain": 85,     # 75-84  (10%)
        "timeout": 93,     # 85-92  (8%)
        "error": 97,       # 93-96  (4%)
        "crash": 100,      # 97-99  (3%)
    },
    
    # goal_hit: Rare successes, simulate targeted search
    # Low verification rate forces policy to learn target selection
    "goal_hit": {
        "verified": 15,    # 0-14   (15%) — rare hits
        "failed": 50,      # 15-49  (35%)
        "abstain": 85,     # 50-84  (35%) — high abstention
        "timeout": 95,     # 85-94  (10%)
        "error": 98,       # 95-97  (3%)
        "crash": 100,      # 98-99  (2%) — minimal crashes
    },
    
    # sparse: Wide proof space, low density
    # Many candidates, few provable — tests navigation of sparse reward landscape
    "sparse": {
        "verified": 25,    # 0-24   (25%) — sparse verification
        "failed": 55,      # 25-54  (30%)
        "abstain": 85,     # 55-84  (30%) — wide abstention zone
        "timeout": 95,     # 85-94  (10%)
        "error": 98,       # 95-97  (3%)
        "crash": 100,      # 98-99  (2%)
    },
    
    # tree: Chain-depth patterns
    # Medium verification rate for building proof chains
    "tree": {
        "verified": 45,    # 0-44   (45%) — medium for chain-building
        "failed": 65,      # 45-64  (20%)
        "abstain": 85,     # 65-84  (20%)
        "timeout": 95,     # 85-94  (10%)
        "error": 98,       # 95-97  (3%)
        "crash": 100,      # 98-99  (2%) — no crashes in chain building
    },
    
    # dependency: Multi-goal coordination
    # Moderate verification, tests coordinated goal achievement
    "dependency": {
        "verified": 35,    # 0-34   (35%)
        "failed": 60,      # 35-59  (25%)
        "abstain": 85,     # 60-84  (25%)
        "timeout": 95,     # 85-94  (10%)
        "error": 98,       # 95-97  (3%)
        "crash": 100,      # 98-99  (2%)
    },
}


# Base latency values (ms) for each bucket type
BUCKET_BASE_LATENCY: Dict[str, int] = {
    "verified": 5,
    "failed": 10,
    "abstain": 25,
    "timeout": 50,  # Will use config.timeout_ms instead
    "error": 2,
    "crash": 0,
}


@dataclass(frozen=True)
class ProfileCoverageMap:
    """
    Static coverage percentages for a profile.
    
    Computed from profile boundaries, NOT from runtime stats.
    Useful for test planning and negative control validation.
    
    Attributes:
        profile_name: Name of the profile.
        verified_pct: Percentage of formulas expected to verify.
        failed_pct: Percentage expected to fail explicitly.
        abstain_pct: Percentage expected to abstain.
        timeout_pct: Percentage expected to timeout.
        error_pct: Percentage expected to error.
        crash_pct: Percentage expected to crash.
    """
    
    profile_name: str
    verified_pct: float
    failed_pct: float
    abstain_pct: float
    timeout_pct: float
    error_pct: float
    crash_pct: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "verified": self.verified_pct,
            "failed": self.failed_pct,
            "abstain": self.abstain_pct,
            "timeout": self.timeout_pct,
            "error": self.error_pct,
            "crash": self.crash_pct,
        }
    
    @classmethod
    def from_profile(cls, profile_name: str) -> "ProfileCoverageMap":
        """
        Compute coverage map from profile boundaries.
        
        Args:
            profile_name: Name of the profile in SLICE_PROFILES.
            
        Returns:
            ProfileCoverageMap with computed percentages.
            
        Raises:
            ValueError: If profile_name is not valid.
        """
        if profile_name not in SLICE_PROFILES:
            raise ValueError(f"Invalid profile '{profile_name}'")
        
        boundaries = SLICE_PROFILES[profile_name]
        
        return cls(
            profile_name=profile_name,
            verified_pct=float(boundaries["verified"]),
            failed_pct=float(boundaries["failed"] - boundaries["verified"]),
            abstain_pct=float(boundaries["abstain"] - boundaries["failed"]),
            timeout_pct=float(boundaries["timeout"] - boundaries["abstain"]),
            error_pct=float(boundaries["error"] - boundaries["timeout"]),
            crash_pct=float(100 - boundaries["error"]),
        )


def compute_profile_coverage(profile_name: str) -> Dict[str, float]:
    """
    Compute static coverage percentages for a profile.
    
    This is a convenience function returning a simple dict.
    For a full object, use ProfileCoverageMap.from_profile().
    
    Args:
        profile_name: Name of the profile.
        
    Returns:
        Dictionary with bucket → percentage mappings.
    """
    return ProfileCoverageMap.from_profile(profile_name).to_dict()


def verify_profile_contracts(epsilon: float = 0.001) -> Tuple[bool, list]:
    """
    Verify that SLICE_PROFILES matches PROFILE_CONTRACTS.
    
    This is a contract enforcement function used by CI to detect
    unintentional changes to profile distributions.
    
    Args:
        epsilon: Tolerance for floating-point comparison.
        
    Returns:
        Tuple of (all_valid, list_of_errors).
    """
    errors = []
    
    for profile_name, contract in PROFILE_CONTRACTS.items():
        if profile_name not in SLICE_PROFILES:
            errors.append(f"Profile '{profile_name}' missing from SLICE_PROFILES")
            continue
        
        coverage = compute_profile_coverage(profile_name)
        
        for bucket, expected_pct in contract.items():
            actual_pct = coverage.get(bucket, 0.0)
            if abs(actual_pct - expected_pct) > epsilon:
                errors.append(
                    f"Profile '{profile_name}' bucket '{bucket}': "
                    f"expected {expected_pct}%, got {actual_pct}%"
                )
    
    # Check for extra profiles in SLICE_PROFILES
    for profile_name in SLICE_PROFILES:
        if profile_name not in PROFILE_CONTRACTS:
            errors.append(f"Profile '{profile_name}' in SLICE_PROFILES but not in PROFILE_CONTRACTS")
    
    return len(errors) == 0, errors


def verify_negative_control_result(result: "MockVerificationResult") -> Tuple[bool, list]:
    """
    Verify that a result conforms to the negative control contract.
    
    Args:
        result: MockVerificationResult to verify.
        
    Returns:
        Tuple of (is_valid, list_of_violations).
    """
    violations = []
    contract = NEGATIVE_CONTROL_CONTRACT
    
    if result.verified != contract.verified:
        violations.append(f"verified: expected {contract.verified}, got {result.verified}")
    if result.abstained != contract.abstained:
        violations.append(f"abstained: expected {contract.abstained}, got {result.abstained}")
    if result.timed_out != contract.timed_out:
        violations.append(f"timed_out: expected {contract.timed_out}, got {result.timed_out}")
    if result.crashed != contract.crashed:
        violations.append(f"crashed: expected {contract.crashed}, got {result.crashed}")
    if result.reason != contract.reason:
        violations.append(f"reason: expected '{contract.reason}', got '{result.reason}'")
    if result.bucket != contract.bucket:
        violations.append(f"bucket: expected '{contract.bucket}', got '{result.bucket}'")
    
    return len(violations) == 0, violations


# ============================================================================
# SCENARIO DEFINITION LAYER
# ============================================================================
# Scenarios provide a declarative layer on top of profiles for expressing
# test scenarios. Each scenario has a name, profile, tags, and description.

@dataclass(frozen=True)
class Scenario:
    """
    A declarative test scenario built on top of a profile.
    
    Scenarios make it easy to express and discover test cases by combining
    a profile with semantic tags and descriptions.
    
    Attributes:
        name: Unique scenario identifier (e.g., "default_sanity").
        profile: The underlying profile to use (must be in SLICE_PROFILES).
        tags: Set of tags for filtering (e.g., {"sanity", "default"}).
        description: Human-readable description of the scenario's purpose.
        negative_control: If True, run in negative control mode.
        samples_default: Default number of samples for this scenario.
    """
    
    name: str
    profile: str
    tags: FrozenSet[str]
    description: str
    negative_control: bool = False
    samples_default: int = 100
    
    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        if not self.name:
            raise ValueError("Scenario name cannot be empty")
        if self.profile not in SLICE_PROFILES:
            raise ValueError(
                f"Invalid profile '{self.profile}' for scenario '{self.name}'. "
                f"Must be one of: {set(SLICE_PROFILES.keys())}"
            )
        if self.samples_default < 1:
            raise ValueError("samples_default must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "profile": self.profile,
            "tags": sorted(self.tags),
            "description": self.description,
            "negative_control": self.negative_control,
            "samples_default": self.samples_default,
        }


# ============================================================================
# SCENARIO REGISTRY
# ============================================================================
# Pre-defined scenarios for common test patterns. The registry is deterministic
# and can be filtered by tags.

SCENARIOS: Dict[str, Scenario] = {
    # Default profile scenarios
    "default_sanity": Scenario(
        name="default_sanity",
        profile="default",
        tags=frozenset({"sanity", "default", "quick"}),
        description="Basic sanity check with balanced default profile",
        samples_default=50,
    ),
    "default_stress": Scenario(
        name="default_stress",
        profile="default",
        tags=frozenset({"stress", "default", "slow"}),
        description="Stress test with default profile (high sample count)",
        samples_default=1000,
    ),
    
    # Goal-hit scenarios
    "goal_hit_stress": Scenario(
        name="goal_hit_stress",
        profile="goal_hit",
        tags=frozenset({"stress", "goal_hit", "rare_success"}),
        description="Stress test for rare-success/targeted-search behavior",
        samples_default=500,
    ),
    "goal_hit_sanity": Scenario(
        name="goal_hit_sanity",
        profile="goal_hit",
        tags=frozenset({"sanity", "goal_hit", "quick"}),
        description="Quick sanity check for goal-hit profile",
        samples_default=50,
    ),
    
    # Sparse scenarios
    "sparse_exploration": Scenario(
        name="sparse_exploration",
        profile="sparse",
        tags=frozenset({"exploration", "sparse", "wide_space"}),
        description="Exploration of wide proof space with low density",
        samples_default=200,
    ),
    
    # Tree scenarios
    "tree_chain_building": Scenario(
        name="tree_chain_building",
        profile="tree",
        tags=frozenset({"chain", "tree", "depth"}),
        description="Chain-depth pattern testing for proof trees",
        samples_default=200,
    ),
    
    # Dependency scenarios
    "dependency_coordination": Scenario(
        name="dependency_coordination",
        profile="dependency",
        tags=frozenset({"coordination", "dependency", "multi_goal"}),
        description="Multi-goal coordination testing",
        samples_default=200,
    ),
    
    # Negative control scenarios
    "nc_baseline": Scenario(
        name="nc_baseline",
        profile="default",
        tags=frozenset({"negative_control", "baseline", "control"}),
        description="Negative control baseline (all results abstain)",
        negative_control=True,
        samples_default=100,
    ),
    "nc_stress": Scenario(
        name="nc_stress",
        profile="default",
        tags=frozenset({"negative_control", "stress", "control"}),
        description="Negative control stress test (high sample count)",
        negative_control=True,
        samples_default=500,
    ),
}


def list_scenarios(filter_tags: Optional[Set[str]] = None) -> List[Scenario]:
    """
    List scenarios, optionally filtered by tags.
    
    Args:
        filter_tags: If provided, only return scenarios that have ALL these tags.
                     If None, return all scenarios.
    
    Returns:
        List of matching Scenario objects, sorted by name.
    
    Examples:
        >>> list_scenarios()  # All scenarios
        >>> list_scenarios({"sanity"})  # Only sanity scenarios
        >>> list_scenarios({"stress", "default"})  # Stress + default
    """
    scenarios = list(SCENARIOS.values())
    
    if filter_tags:
        filter_set = frozenset(filter_tags)
        scenarios = [s for s in scenarios if filter_set.issubset(s.tags)]
    
    return sorted(scenarios, key=lambda s: s.name)


def get_scenario(name: str) -> Scenario:
    """
    Get a scenario by name.
    
    Args:
        name: The scenario name.
        
    Returns:
        The Scenario object.
        
    Raises:
        KeyError: If scenario not found.
    """
    if name not in SCENARIOS:
        available = sorted(SCENARIOS.keys())
        raise KeyError(
            f"Scenario '{name}' not found. Available: {available}"
        )
    return SCENARIOS[name]


# ============================================================================
# CONTRACT EXPORT FOR EVIDENCE PACKS
# ============================================================================

def export_mock_oracle_contract() -> Dict[str, Any]:
    """
    Export a compact contract snapshot suitable for Evidence Packs.
    
    This function produces a deterministic, JSON-serializable representation
    of the mock oracle's behavioral contract. It can be used to:
    - Embed contract metadata in Evidence Packs
    - Detect contract drift between versions
    - Document expected mock behavior
    
    Returns:
        Dictionary containing:
        - contract_version: Current contract version string
        - profiles: All profile distributions
        - negative_control: NC contract semantics
        - scenarios: All registered scenarios
        - determinism: Determinism guarantees
    """
    return {
        "contract_version": MOCK_ORACLE_CONTRACT_VERSION,
        "profiles": {
            name: {
                "boundaries": SLICE_PROFILES[name],
                "distribution": PROFILE_CONTRACTS[name],
            }
            for name in sorted(PROFILE_CONTRACTS.keys())
        },
        "negative_control": {
            "verified": NEGATIVE_CONTROL_CONTRACT.verified,
            "abstained": NEGATIVE_CONTROL_CONTRACT.abstained,
            "timed_out": NEGATIVE_CONTROL_CONTRACT.timed_out,
            "crashed": NEGATIVE_CONTROL_CONTRACT.crashed,
            "reason": NEGATIVE_CONTROL_CONTRACT.reason,
            "bucket": NEGATIVE_CONTROL_CONTRACT.bucket,
            "stats_suppressed": NEGATIVE_CONTROL_CONTRACT.stats_suppressed,
        },
        "scenarios": {
            name: scenario.to_dict()
            for name, scenario in sorted(SCENARIOS.items())
        },
        "determinism": {
            "hash_algorithm": "SHA-256",
            "bucket_selection": "int(hash_hex, 16) % 100",
            "guarantees": [
                "Same input + same config → identical output",
                "No randomness outside hash-based branching",
                "No external dependencies (network, files, time)",
            ],
        },
    }


# ============================================================================
# SCENARIO ANALYTICS & DRIFT DETECTION
# ============================================================================
# Schema version for analytics summaries. Bump when schema changes.
ANALYTICS_SCHEMA_VERSION = "1.0.0"

# Default drift tolerance (percentage points)
DEFAULT_DRIFT_TOLERANCE = 5.0

# Drift tolerance scaling factor based on sample count
# Tolerance = max(base_tolerance, scale_factor / sqrt(samples))
DRIFT_TOLERANCE_SCALE_FACTOR = 50.0


def summarize_scenario_results(
    name: str,
    outcomes: Sequence["MockVerificationResult"],
    drift_tolerance: float = DEFAULT_DRIFT_TOLERANCE,
) -> Dict[str, Any]:
    """
    Compute an analytics summary for scenario results.
    
    This function analyzes a sequence of mock oracle results and produces
    a structured summary comparing empirical distributions to expected
    contract distributions.
    
    Args:
        name: Scenario name (must exist in SCENARIOS).
        outcomes: Sequence of MockVerificationResult objects.
        drift_tolerance: Maximum allowed deviation (in percentage points)
                         before flagging contract violation.
    
    Returns:
        Dictionary containing:
        - schema_version: Analytics schema version
        - scenario_name: Name of the scenario
        - sample_count: Number of outcomes analyzed
        - bucket_counts: Raw counts per bucket
        - empirical_distribution: Observed percentages
        - expected_distribution: Contract-defined percentages
        - deltas: Difference (empirical - expected) per bucket
        - max_delta: Maximum absolute deviation
        - drift_tolerance: Tolerance used for contract check
        - contract_respected: True if all deltas within tolerance
    
    Raises:
        KeyError: If scenario name not found.
        ValueError: If outcomes sequence is empty.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Scenario '{name}' not found in SCENARIOS registry")
    
    if len(outcomes) == 0:
        raise ValueError("Cannot summarize empty outcomes sequence")
    
    scenario = SCENARIOS[name]
    sample_count = len(outcomes)
    
    # Count buckets
    bucket_counts: Dict[str, int] = {
        "verified": 0,
        "failed": 0,
        "abstain": 0,
        "timeout": 0,
        "error": 0,
        "crash": 0,
        "negative_control": 0,
    }
    
    for outcome in outcomes:
        bucket = outcome.bucket
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
    
    # Compute empirical distribution
    empirical: Dict[str, float] = {
        bucket: round(count / sample_count * 100, 4)
        for bucket, count in bucket_counts.items()
    }
    
    # Get expected distribution
    if scenario.negative_control:
        expected: Dict[str, float] = {
            "verified": 0.0,
            "failed": 0.0,
            "abstain": 0.0,
            "timeout": 0.0,
            "error": 0.0,
            "crash": 0.0,
            "negative_control": 100.0,
        }
    else:
        profile_dist = PROFILE_CONTRACTS.get(scenario.profile, {})
        expected = {
            "verified": profile_dist.get("verified", 0.0),
            "failed": profile_dist.get("failed", 0.0),
            "abstain": profile_dist.get("abstain", 0.0),
            "timeout": profile_dist.get("timeout", 0.0),
            "error": profile_dist.get("error", 0.0),
            "crash": profile_dist.get("crash", 0.0),
            "negative_control": 0.0,
        }
    
    # Compute deltas
    deltas: Dict[str, float] = {
        bucket: round(empirical[bucket] - expected[bucket], 4)
        for bucket in empirical
    }
    
    # Find max absolute delta
    max_delta = max(abs(d) for d in deltas.values())
    
    # Adjust tolerance based on sample size (statistical wiggle room)
    # With small samples, expect higher variance
    adjusted_tolerance = max(
        drift_tolerance,
        DRIFT_TOLERANCE_SCALE_FACTOR / (sample_count ** 0.5)
    )
    
    # Check contract
    contract_respected = max_delta <= adjusted_tolerance
    
    return {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "scenario_name": name,
        "sample_count": sample_count,
        "bucket_counts": bucket_counts,
        "empirical_distribution": empirical,
        "expected_distribution": expected,
        "deltas": deltas,
        "max_delta": round(max_delta, 4),
        "drift_tolerance": round(adjusted_tolerance, 4),
        "contract_respected": contract_respected,
    }


# Drift status constants
DRIFT_STATUS_IN_CONTRACT = "IN_CONTRACT"
DRIFT_STATUS_DRIFTED = "DRIFTED"
DRIFT_STATUS_BROKEN = "BROKEN"


def detect_scenario_drift(
    contract: Dict[str, Any],
    summary: Dict[str, Any],
    warning_threshold: float = 3.0,
    broken_threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Detect drift between observed behavior and contract expectations.
    
    This function compares an analytics summary against the contract
    and produces a drift report with actionable recommendations.
    
    Status levels:
    - IN_CONTRACT: All buckets within tolerance; no action needed.
    - DRIFTED: Some buckets exceed warning_threshold; investigate.
    - BROKEN: Some buckets exceed broken_threshold; critical failure.
    
    Args:
        contract: Contract dict from export_mock_oracle_contract().
        summary: Summary dict from summarize_scenario_results().
        warning_threshold: Delta threshold (%) for DRIFTED status.
        broken_threshold: Delta threshold (%) for BROKEN status.
    
    Returns:
        Dictionary containing:
        - status: "IN_CONTRACT" | "DRIFTED" | "BROKEN"
        - contract_version: Version of contract checked against
        - scenario_name: Name of scenario analyzed
        - sample_count: Number of samples in summary
        - drift_signals: List of buckets that exceeded thresholds
        - max_drift: Maximum observed drift
        - thresholds: Dict of thresholds used
        - recommended_action: Human-readable guidance
    """
    scenario_name = summary.get("scenario_name", "unknown")
    deltas = summary.get("deltas", {})
    sample_count = summary.get("sample_count", 0)
    
    # Collect drift signals
    drift_signals: List[Dict[str, Any]] = []
    max_drift = 0.0
    
    for bucket, delta in deltas.items():
        abs_delta = abs(delta)
        max_drift = max(max_drift, abs_delta)
        
        if abs_delta > warning_threshold:
            signal = {
                "bucket": bucket,
                "delta": round(delta, 4),
                "abs_delta": round(abs_delta, 4),
                "expected": summary["expected_distribution"].get(bucket, 0.0),
                "observed": summary["empirical_distribution"].get(bucket, 0.0),
                "severity": "BROKEN" if abs_delta > broken_threshold else "WARNING",
            }
            drift_signals.append(signal)
    
    # Determine overall status
    has_broken = any(s["severity"] == "BROKEN" for s in drift_signals)
    has_warning = len(drift_signals) > 0
    
    if has_broken:
        status = DRIFT_STATUS_BROKEN
    elif has_warning:
        status = DRIFT_STATUS_DRIFTED
    else:
        status = DRIFT_STATUS_IN_CONTRACT
    
    # Generate recommendation
    if status == DRIFT_STATUS_IN_CONTRACT:
        recommended_action = "No action required. Mock oracle behavior is within contract bounds."
    elif status == DRIFT_STATUS_DRIFTED:
        bucket_list = ", ".join(s["bucket"] for s in drift_signals)
        recommended_action = (
            f"Minor drift detected in buckets: {bucket_list}. "
            f"Consider increasing sample count (current: {sample_count}) or "
            f"investigating mock oracle hash distribution."
        )
    else:  # BROKEN
        bucket_list = ", ".join(
            s["bucket"] for s in drift_signals if s["severity"] == "BROKEN"
        )
        recommended_action = (
            f"CRITICAL: Significant drift in buckets: {bucket_list}. "
            f"Mock oracle contract may be violated. Actions: "
            f"1) Verify SLICE_PROFILES unchanged. "
            f"2) Check hash function implementation. "
            f"3) Review recent mock_config.py changes. "
            f"4) If intentional, bump contract version."
        )
    
    return {
        "status": status,
        "contract_version": contract.get("contract_version", "unknown"),
        "scenario_name": scenario_name,
        "sample_count": sample_count,
        "drift_signals": drift_signals,
        "max_drift": round(max_drift, 4),
        "thresholds": {
            "warning": warning_threshold,
            "broken": broken_threshold,
        },
        "recommended_action": recommended_action,
    }


# ============================================================================
# ADVERSARIAL COVERAGE RADAR & GOVERNANCE HOOKS
# ============================================================================
# These functions transform scenario analytics into governance-oriented views
# for MAAS (Metrics as a Service) and global health monitoring.

# All verification outcome buckets (metrics)
VERIFICATION_METRICS = [
    "verified",
    "failed",
    "abstain",
    "timeout",
    "error",
    "crash",
    "negative_control",
]

# Governance status constants
GOVERNANCE_STATUS_OK = "OK"
GOVERNANCE_STATUS_ATTENTION = "ATTENTION"
GOVERNANCE_STATUS_BLOCK = "BLOCK"

# Director panel status lights
STATUS_LIGHT_GREEN = "GREEN"
STATUS_LIGHT_YELLOW = "YELLOW"
STATUS_LIGHT_RED = "RED"

# Minimum scenarios per metric to be considered "well exercised"
MIN_SCENARIOS_FOR_WELL_EXERCISED = 3


def build_metric_scenario_coverage_view(
    scenario_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a metric-centric view of scenario coverage.
    
    This function analyzes scenario summaries to determine which verification
    outcome metrics (buckets) are well-tested and which are under-tested.
    It identifies scenarios that test each metric and tracks drift status
    per metric.
    
    Args:
        scenario_summaries: Sequence of scenario summary dicts. Each should
                           have a "drift_report" key with "status" and
                           "drift_signals" fields.
    
    Returns:
        Dictionary containing:
        - metrics: Dict mapping metric name to:
          - scenarios_tested: List of scenario names that test this metric
          - drift_free_count: Number of scenarios with IN_CONTRACT status
          - drifted_scenarios: List of scenarios with DRIFTED status
          - broken_scenarios: List of scenarios with BROKEN status
        - metrics_well_exercised: List of metrics tested by >= MIN_SCENARIOS
        - metrics_under_tested: List of metrics tested by < MIN_SCENARIOS
        - total_scenarios: Total number of scenarios analyzed
    """
    # Initialize metric tracking
    metric_data: Dict[str, Dict[str, Any]] = {}
    for metric in VERIFICATION_METRICS:
        metric_data[metric] = {
            "scenarios_tested": [],
            "drift_free_count": 0,
            "drifted_scenarios": [],
            "broken_scenarios": [],
        }
    
    # Process each scenario summary
    for summary in scenario_summaries:
        drift_report = summary.get("drift_report", {})
        scenario_name = drift_report.get("scenario_name", summary.get("scenario_name", "unknown"))
        status = drift_report.get("status", "UNKNOWN")
        drift_signals = drift_report.get("drift_signals", [])
        
        # Determine which metrics this scenario tests
        # A scenario tests a metric if it has that bucket in its distribution
        # or if that metric appears in drift signals
        tested_metrics = set()
        
        # Check empirical distribution (scenario always tests all metrics)
        empirical = summary.get("empirical_distribution", {})
        for metric in VERIFICATION_METRICS:
            if metric in empirical:
                tested_metrics.add(metric)
        
        # Also check drift signals (metrics that drifted)
        for signal in drift_signals:
            bucket = signal.get("bucket")
            if bucket in VERIFICATION_METRICS:
                tested_metrics.add(bucket)
        
        # Update metric tracking
        for metric in tested_metrics:
            if scenario_name not in metric_data[metric]["scenarios_tested"]:
                metric_data[metric]["scenarios_tested"].append(scenario_name)
            
            if status == DRIFT_STATUS_IN_CONTRACT:
                metric_data[metric]["drift_free_count"] += 1
            elif status == DRIFT_STATUS_DRIFTED:
                if scenario_name not in metric_data[metric]["drifted_scenarios"]:
                    metric_data[metric]["drifted_scenarios"].append(scenario_name)
            elif status == DRIFT_STATUS_BROKEN:
                if scenario_name not in metric_data[metric]["broken_scenarios"]:
                    metric_data[metric]["broken_scenarios"].append(scenario_name)
    
    # Sort scenario lists for determinism
    for metric in metric_data:
        metric_data[metric]["scenarios_tested"].sort()
        metric_data[metric]["drifted_scenarios"].sort()
        metric_data[metric]["broken_scenarios"].sort()
    
    # Identify well-exercised and under-tested metrics
    metrics_well_exercised = [
        metric
        for metric, data in metric_data.items()
        if len(data["scenarios_tested"]) >= MIN_SCENARIOS_FOR_WELL_EXERCISED
    ]
    metrics_under_tested = [
        metric
        for metric, data in metric_data.items()
        if len(data["scenarios_tested"]) < MIN_SCENARIOS_FOR_WELL_EXERCISED
    ]
    
    return {
        "metrics": metric_data,
        "metrics_well_exercised": sorted(metrics_well_exercised),
        "metrics_under_tested": sorted(metrics_under_tested),
        "total_scenarios": len(scenario_summaries),
    }


def summarize_mock_oracle_drift_for_governance(
    coverage_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a governance-oriented summary of mock oracle drift.
    
    This function analyzes the coverage view to produce a high-level summary
    suitable for governance and risk review. It identifies critical issues
    and provides actionable status indicators.
    
    Args:
        coverage_view: Coverage view from build_metric_scenario_coverage_view().
    
    Returns:
        Dictionary containing:
        - has_broken_scenarios: True if any scenario has BROKEN status
        - metrics_impacted_by_drift: List of metrics with drifted/broken scenarios
        - status: "OK" | "ATTENTION" | "BLOCK"
        - broken_scenario_count: Number of scenarios with BROKEN status
        - drifted_scenario_count: Number of scenarios with DRIFTED status
        - total_scenarios: Total scenarios analyzed
    """
    metrics = coverage_view.get("metrics", {})
    
    # Collect all broken and drifted scenarios
    all_broken_scenarios = set()
    all_drifted_scenarios = set()
    metrics_impacted = []
    
    for metric, data in metrics.items():
        broken = data.get("broken_scenarios", [])
        drifted = data.get("drifted_scenarios", [])
        
        if broken:
            all_broken_scenarios.update(broken)
            metrics_impacted.append(metric)
        elif drifted:
            all_drifted_scenarios.update(drifted)
            if metric not in metrics_impacted:
                metrics_impacted.append(metric)
    
    broken_count = len(all_broken_scenarios)
    drifted_count = len(all_drifted_scenarios)
    has_broken = broken_count > 0
    
    # Determine governance status
    if has_broken:
        status = GOVERNANCE_STATUS_BLOCK
    elif drifted_count > 0:
        status = GOVERNANCE_STATUS_ATTENTION
    else:
        status = GOVERNANCE_STATUS_OK
    
    return {
        "has_broken_scenarios": has_broken,
        "metrics_impacted_by_drift": sorted(metrics_impacted),
        "status": status,
        "broken_scenario_count": broken_count,
        "drifted_scenario_count": drifted_count,
        "total_scenarios": coverage_view.get("total_scenarios", 0),
    }


def build_mock_oracle_director_panel(
    coverage_view: Dict[str, Any],
    governance_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director-level panel for global health monitoring.
    
    This function creates a high-level executive summary suitable for
    global health dashboards and director-level reporting. It provides
    a status light, key metrics, and a neutral headline.
    
    Args:
        coverage_view: Coverage view from build_metric_scenario_coverage_view().
        governance_summary: Governance summary from summarize_mock_oracle_drift_for_governance().
    
    Returns:
        Dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - scenario_count: Total number of scenarios
        - broken_scenario_count: Number of broken scenarios
        - impacted_metrics: List of metrics impacted by drift
        - headline: Neutral summary text
        - metrics_well_exercised_count: Number of well-exercised metrics
        - metrics_under_tested_count: Number of under-tested metrics
    """
    status = governance_summary.get("status", GOVERNANCE_STATUS_OK)
    broken_count = governance_summary.get("broken_scenario_count", 0)
    drifted_count = governance_summary.get("drifted_scenario_count", 0)
    impacted_metrics = governance_summary.get("metrics_impacted_by_drift", [])
    scenario_count = governance_summary.get("total_scenarios", 0)
    
    # Determine status light
    if status == GOVERNANCE_STATUS_BLOCK:
        status_light = STATUS_LIGHT_RED
    elif status == GOVERNANCE_STATUS_ATTENTION:
        status_light = STATUS_LIGHT_YELLOW
    else:
        status_light = STATUS_LIGHT_GREEN
    
    # Count well-exercised and under-tested metrics
    well_exercised = coverage_view.get("metrics_well_exercised", [])
    under_tested = coverage_view.get("metrics_under_tested", [])
    
    # Generate headline
    if broken_count > 0:
        headline = (
            f"Mock oracle adversarial coverage: {broken_count} scenario(s) "
            f"exhibit critical drift. {len(impacted_metrics)} metric(s) impacted. "
            f"Contract integrity review recommended."
        )
    elif drifted_count > 0:
        headline = (
            f"Mock oracle adversarial coverage: {drifted_count} scenario(s) "
            f"show minor drift. {len(impacted_metrics)} metric(s) require attention. "
            f"Monitoring recommended."
        )
    elif len(under_tested) > 0:
        headline = (
            f"Mock oracle adversarial coverage: {scenario_count} scenarios active. "
            f"{len(under_tested)} metric(s) under-tested. Coverage expansion recommended."
        )
    else:
        headline = (
            f"Mock oracle adversarial coverage: {scenario_count} scenarios active. "
            f"All {len(well_exercised)} metrics well-exercised. Coverage robust."
        )
    
    return {
        "status_light": status_light,
        "scenario_count": scenario_count,
        "broken_scenario_count": broken_count,
        "impacted_metrics": impacted_metrics,
        "headline": headline,
        "metrics_well_exercised_count": len(well_exercised),
        "metrics_under_tested_count": len(under_tested),
    }


# ============================================================================
# SCENARIO FLEET CONSOLE & REGRESSION WATCHDOG
# ============================================================================

# Fleet status constants
FLEET_STATUS_OK = "OK"
FLEET_STATUS_ATTENTION = "ATTENTION"
FLEET_STATUS_GAP = "GAP"

# Critical metrics that must be well-exercised
CRITICAL_METRICS = ["verified", "failed", "abstain"]

# Threshold for considering a metric "underspecified"
# A metric is underspecified if it has fewer scenarios than this threshold
MIN_SCENARIOS_FOR_SPECIFIED = 2


def build_scenario_fleet_summary(
    coverage_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a fleet-level summary of scenario coverage.
    
    This function analyzes the coverage view to identify gaps in adversarial
    coverage. It flags metrics that are underspecified (too few scenarios)
    and determines overall fleet status.
    
    Args:
        coverage_view: Coverage view from build_metric_scenario_coverage_view().
    
    Returns:
        Dictionary containing:
        - total_scenarios: Total number of scenarios in the fleet
        - metrics_well_exercised: List of metrics with >= MIN_SCENARIOS scenarios
        - metrics_underspecified: List of metrics with < MIN_SCENARIOS scenarios
        - fleet_status: "OK" | "ATTENTION" | "GAP"
        - summary_text: Neutral summary text describing fleet health
    """
    metrics = coverage_view.get("metrics", {})
    total_scenarios = coverage_view.get("total_scenarios", 0)
    well_exercised = coverage_view.get("metrics_well_exercised", [])
    under_tested = coverage_view.get("metrics_under_tested", [])
    
    # Identify underspecified metrics (critical metrics that are under-tested)
    metrics_underspecified = []
    for metric in VERIFICATION_METRICS:
        metric_data = metrics.get(metric, {})
        scenarios_tested = len(metric_data.get("scenarios_tested", []))
        broken_count = len(metric_data.get("broken_scenarios", []))
        
        # A metric is underspecified if:
        # 1. It has fewer scenarios than MIN_SCENARIOS_FOR_SPECIFIED, OR
        # 2. It's a critical metric and has many broken scenarios relative to total
        is_underspecified = (
            scenarios_tested < MIN_SCENARIOS_FOR_SPECIFIED
            or (metric in CRITICAL_METRICS and broken_count > 0 and broken_count >= scenarios_tested / 2)
        )
        
        if is_underspecified:
            metrics_underspecified.append(metric)
    
    # Determine fleet status
    # GAP: Critical metrics are underspecified or have many broken scenarios
    has_critical_gap = any(
        m in CRITICAL_METRICS for m in metrics_underspecified
    )
    
    # Check for high broken scenario ratio in critical metrics
    critical_metrics_broken = False
    for metric in CRITICAL_METRICS:
        metric_data = metrics.get(metric, {})
        scenarios_tested = len(metric_data.get("scenarios_tested", []))
        broken_count = len(metric_data.get("broken_scenarios", []))
        
        if scenarios_tested > 0 and broken_count > 0:
            broken_ratio = broken_count / scenarios_tested
            if broken_ratio > 0.5:  # More than 50% broken
                critical_metrics_broken = True
                break
    
    if has_critical_gap or critical_metrics_broken:
        fleet_status = FLEET_STATUS_GAP
    elif len(metrics_underspecified) > 0:
        fleet_status = FLEET_STATUS_ATTENTION
    else:
        fleet_status = FLEET_STATUS_OK
    
    # Generate summary text
    if fleet_status == FLEET_STATUS_GAP:
        gap_metrics = [m for m in metrics_underspecified if m in CRITICAL_METRICS]
        summary_text = (
            f"Scenario fleet: {total_scenarios} scenarios active. "
            f"GAP detected in critical metrics: {', '.join(sorted(gap_metrics))}. "
            f"Coverage expansion required."
        )
    elif fleet_status == FLEET_STATUS_ATTENTION:
        summary_text = (
            f"Scenario fleet: {total_scenarios} scenarios active. "
            f"{len(metrics_underspecified)} metric(s) underspecified. "
            f"Coverage monitoring recommended."
        )
    else:
        summary_text = (
            f"Scenario fleet: {total_scenarios} scenarios active. "
            f"All {len(well_exercised)} metrics well-exercised. "
            f"Fleet coverage adequate."
        )
    
    return {
        "total_scenarios": total_scenarios,
        "metrics_well_exercised": sorted(well_exercised),
        "metrics_underspecified": sorted(metrics_underspecified),
        "fleet_status": fleet_status,
        "summary_text": summary_text,
    }


def detect_mock_oracle_regression(
    previous_view: Dict[str, Any],
    current_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect regression in mock oracle coverage between two time points.
    
    This function compares a previous coverage view with a current one to
    identify regressions. A regression is detected when:
    - The number of broken scenarios increases for any metric
    - The number of drifted scenarios increases significantly for any metric
    - A previously well-exercised metric becomes under-tested
    
    Args:
        previous_view: Coverage view from an earlier time point.
        current_view: Coverage view from the current time point.
    
    Returns:
        Dictionary containing:
        - regression_detected: True if any regression is detected
        - affected_metrics: List of metrics that regressed
        - notes: List of human-readable notes describing regressions
    """
    previous_metrics = previous_view.get("metrics", {})
    current_metrics = current_view.get("metrics", {})
    
    regression_detected = False
    affected_metrics = []
    notes = []
    
    # Compare each metric
    for metric in VERIFICATION_METRICS:
        prev_data = previous_metrics.get(metric, {})
        curr_data = current_metrics.get(metric, {})
        
        prev_broken = len(prev_data.get("broken_scenarios", []))
        curr_broken = len(curr_data.get("broken_scenarios", []))
        
        prev_drifted = len(prev_data.get("drifted_scenarios", []))
        curr_drifted = len(curr_data.get("drifted_scenarios", []))
        
        prev_scenarios = len(prev_data.get("scenarios_tested", []))
        curr_scenarios = len(curr_data.get("scenarios_tested", []))
        
        # Check for increased broken scenarios
        if curr_broken > prev_broken:
            regression_detected = True
            if metric not in affected_metrics:
                affected_metrics.append(metric)
            
            increase = curr_broken - prev_broken
            notes.append(
                f"Metric '{metric}': broken scenarios increased from {prev_broken} to {curr_broken} (+{increase})"
            )
        
        # Check for significant increase in drifted scenarios
        # (more than 2x increase AND absolute increase > 2, or absolute increase > 3)
        if curr_drifted > prev_drifted:
            increase = curr_drifted - prev_drifted
            significant_increase = (
                increase > 3 or (prev_drifted > 0 and curr_drifted >= prev_drifted * 2 and increase > 2)
            )
            
            if significant_increase:
                regression_detected = True
                if metric not in affected_metrics:
                    affected_metrics.append(metric)
                
                notes.append(
                    f"Metric '{metric}': drifted scenarios increased from {prev_drifted} to {curr_drifted} (+{increase})"
                )
        
        # Check for metric becoming under-tested
        prev_well_exercised = prev_scenarios >= MIN_SCENARIOS_FOR_WELL_EXERCISED
        curr_well_exercised = curr_scenarios >= MIN_SCENARIOS_FOR_WELL_EXERCISED
        
        if prev_well_exercised and not curr_well_exercised:
            regression_detected = True
            if metric not in affected_metrics:
                affected_metrics.append(metric)
            
            notes.append(
                f"Metric '{metric}': became under-tested (scenarios decreased from {prev_scenarios} to {curr_scenarios})"
            )
    
    return {
        "regression_detected": regression_detected,
        "affected_metrics": sorted(affected_metrics),
        "notes": notes,
    }


# ============================================================================
# ORACLE FLEET STABILITY RADAR & CONSOLE INTEGRATION
# ============================================================================

# Stability band constants
STABILITY_BAND_STABLE = "STABLE"
STABILITY_BAND_DRIFTING = "DRIFTING"
STABILITY_BAND_UNSTABLE = "UNSTABLE"

# CI exit codes
CI_EXIT_OK = 0
CI_EXIT_WARN = 1
CI_EXIT_BLOCK = 2

# Stability radar schema version
STABILITY_RADAR_SCHEMA_VERSION = "1.0.0"


def build_oracle_fleet_stability_radar(
    fleet_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a stability radar from a time series of fleet summaries.
    
    This function analyzes trends in fleet summaries over time to determine
    stability patterns. It tracks broken rate trends, underspecification trends,
    and critical metric failures.
    
    Args:
        fleet_summaries: Sequence of fleet summary dicts from build_scenario_fleet_summary(),
                        ordered chronologically (oldest first).
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - stability_band: "STABLE" | "DRIFTING" | "UNSTABLE"
        - broken_rate_trend: "IMPROVING" | "STABLE" | "DETERIORATING"
        - underspecification_trend: "IMPROVING" | "STABLE" | "DETERIORATING"
        - critical_metric_failures: List of critical metrics with failures
        - summary: Neutral text describing stability
    """
    if len(fleet_summaries) == 0:
        return {
            "schema_version": STABILITY_RADAR_SCHEMA_VERSION,
            "stability_band": STABILITY_BAND_STABLE,
            "broken_rate_trend": "STABLE",
            "underspecification_trend": "STABLE",
            "critical_metric_failures": [],
            "summary": "Fleet stability radar: No data available.",
        }
    
    if len(fleet_summaries) == 1:
        # Single data point - assume stable
        summary = fleet_summaries[0]
        return {
            "schema_version": STABILITY_RADAR_SCHEMA_VERSION,
            "stability_band": STABILITY_BAND_STABLE,
            "broken_rate_trend": "STABLE",
            "underspecification_trend": "STABLE",
            "critical_metric_failures": [],
            "summary": f"Fleet stability radar: Single data point. Fleet status: {summary.get('fleet_status', 'UNKNOWN')}.",
        }
    
    # Extract trends
    fleet_statuses = [s.get("fleet_status", FLEET_STATUS_OK) for s in fleet_summaries]
    total_scenarios = [s.get("total_scenarios", 0) for s in fleet_summaries]
    underspecified_counts = [len(s.get("metrics_underspecified", [])) for s in fleet_summaries]
    
    # Calculate broken rate trend (simplified: track GAP status frequency)
    gap_count = sum(1 for status in fleet_statuses if status == FLEET_STATUS_GAP)
    gap_ratio = gap_count / len(fleet_statuses)
    
    # Determine broken rate trend
    recent_gaps = sum(1 for status in fleet_statuses[-3:] if status == FLEET_STATUS_GAP)
    early_gaps = sum(1 for status in fleet_statuses[:max(1, len(fleet_statuses) - 3)] if status == FLEET_STATUS_GAP)
    
    if recent_gaps > early_gaps:
        broken_rate_trend = "DETERIORATING"
    elif recent_gaps < early_gaps:
        broken_rate_trend = "IMPROVING"
    else:
        broken_rate_trend = "STABLE"
    
    # Determine underspecification trend
    if len(underspecified_counts) >= 2:
        recent_avg = sum(underspecified_counts[-3:]) / min(3, len(underspecified_counts))
        early_avg = sum(underspecified_counts[:max(1, len(underspecified_counts) - 3)]) / max(1, len(underspecified_counts) - 3)
        
        if recent_avg > early_avg * 1.2:  # 20% increase
            underspecification_trend = "DETERIORATING"
        elif recent_avg < early_avg * 0.8:  # 20% decrease
            underspecification_trend = "IMPROVING"
        else:
            underspecification_trend = "STABLE"
    else:
        underspecification_trend = "STABLE"
    
    # Identify critical metric failures
    critical_metric_failures = []
    latest_summary = fleet_summaries[-1]
    latest_underspecified = latest_summary.get("metrics_underspecified", [])
    
    for metric in CRITICAL_METRICS:
        if metric in latest_underspecified:
            critical_metric_failures.append(metric)
    
    # Also check if latest status is GAP
    if latest_summary.get("fleet_status") == FLEET_STATUS_GAP:
        # All critical metrics are considered at risk
        for metric in CRITICAL_METRICS:
            if metric not in critical_metric_failures:
                critical_metric_failures.append(metric)
    
    # Determine stability band
    has_critical_failures = len(critical_metric_failures) > 0
    has_deteriorating_trends = (
        broken_rate_trend == "DETERIORATING" or
        underspecification_trend == "DETERIORATING"
    )
    high_gap_ratio = gap_ratio > 0.5  # More than 50% of time in GAP
    
    if has_critical_failures and (high_gap_ratio or has_deteriorating_trends):
        stability_band = STABILITY_BAND_UNSTABLE
    elif has_critical_failures or has_deteriorating_trends or gap_ratio > 0.3:
        stability_band = STABILITY_BAND_DRIFTING
    else:
        stability_band = STABILITY_BAND_STABLE
    
    # Generate summary text
    if stability_band == STABILITY_BAND_UNSTABLE:
        summary = (
            f"Fleet stability radar: UNSTABLE. "
            f"{len(critical_metric_failures)} critical metric(s) failing. "
            f"Broken rate: {broken_rate_trend.lower()}. "
            f"Immediate intervention required."
        )
    elif stability_band == STABILITY_BAND_DRIFTING:
        summary = (
            f"Fleet stability radar: DRIFTING. "
            f"{len(critical_metric_failures)} critical metric(s) at risk. "
            f"Trends: broken rate {broken_rate_trend.lower()}, "
            f"underspecification {underspecification_trend.lower()}. "
            f"Monitoring recommended."
        )
    else:
        summary = (
            f"Fleet stability radar: STABLE. "
            f"All critical metrics healthy. "
            f"Trends: broken rate {broken_rate_trend.lower()}, "
            f"underspecification {underspecification_trend.lower()}. "
            f"Fleet coverage adequate."
        )
    
    return {
        "schema_version": STABILITY_RADAR_SCHEMA_VERSION,
        "stability_band": stability_band,
        "broken_rate_trend": broken_rate_trend,
        "underspecification_trend": underspecification_trend,
        "critical_metric_failures": sorted(critical_metric_failures),
        "summary": summary,
    }


def build_fleet_console_tile(
    stability_radar: Dict[str, Any],
    fleet_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a console tile for dashboard display.
    
    This function creates a simplified, dashboard-friendly view of fleet health
    combining stability radar data with current fleet summary.
    
    Args:
        stability_radar: Stability radar from build_oracle_fleet_stability_radar().
        fleet_summary: Current fleet summary from build_scenario_fleet_summary().
    
    Returns:
        Dictionary containing:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - high_risk_metrics: List of metrics at high risk
        - fleet_status: Current fleet status
        - headline: Brief headline text
    """
    stability_band = stability_radar.get("stability_band", STABILITY_BAND_STABLE)
    critical_failures = stability_radar.get("critical_metric_failures", [])
    fleet_status = fleet_summary.get("fleet_status", FLEET_STATUS_OK)
    
    # Determine status light
    if stability_band == STABILITY_BAND_UNSTABLE or fleet_status == FLEET_STATUS_BLOCK:
        status_light = STATUS_LIGHT_RED
    elif stability_band == STABILITY_BAND_DRIFTING or fleet_status == FLEET_STATUS_ATTENTION:
        status_light = STATUS_LIGHT_YELLOW
    else:
        status_light = STATUS_LIGHT_GREEN
    
    # Identify high-risk metrics
    high_risk_metrics = []
    
    # Add critical metric failures
    high_risk_metrics.extend(critical_failures)
    
    # Add underspecified critical metrics
    underspecified = fleet_summary.get("metrics_underspecified", [])
    for metric in CRITICAL_METRICS:
        if metric in underspecified and metric not in high_risk_metrics:
            high_risk_metrics.append(metric)
    
    # Generate headline
    if len(high_risk_metrics) > 0:
        metrics_str = ", ".join(sorted(high_risk_metrics))
        headline = f"Fleet: {stability_band.lower()} — {len(high_risk_metrics)} metric(s) at risk ({metrics_str})"
    elif fleet_status == FLEET_STATUS_GAP:
        headline = f"Fleet: {fleet_status.lower()} — coverage expansion required"
    else:
        headline = f"Fleet: {fleet_status.lower()} — {fleet_summary.get('total_scenarios', 0)} scenarios active"
    
    return {
        "status_light": status_light,
        "high_risk_metrics": sorted(high_risk_metrics),
        "fleet_status": fleet_status,
        "headline": headline,
    }


def evaluate_fleet_for_ci(
    stability_radar: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """
    Evaluate fleet stability for CI integration.
    
    This function provides CI-friendly exit codes and evaluation results
    based on fleet stability radar data.
    
    Args:
        stability_radar: Stability radar from build_oracle_fleet_stability_radar().
    
    Returns:
        Tuple of (exit_code, evaluation_dict):
        - exit_code: 0 (OK), 1 (WARN), or 2 (BLOCK)
        - evaluation_dict: Contains evaluation details
    """
    stability_band = stability_radar.get("stability_band", STABILITY_BAND_STABLE)
    critical_failures = stability_radar.get("critical_metric_failures", [])
    broken_trend = stability_radar.get("broken_rate_trend", "STABLE")
    underspec_trend = stability_radar.get("underspecification_trend", "STABLE")
    
    # Determine exit code
    # BLOCK (2): Critical metrics failing OR unstable band
    if len(critical_failures) > 0 or stability_band == STABILITY_BAND_UNSTABLE:
        exit_code = CI_EXIT_BLOCK
        evaluation_status = "BLOCK"
        reason = (
            f"Critical metric failures: {', '.join(sorted(critical_failures))}"
            if critical_failures
            else "Fleet stability: UNSTABLE"
        )
    # WARN (1): Drifting OR deteriorating trends
    elif stability_band == STABILITY_BAND_DRIFTING or broken_trend == "DETERIORATING" or underspec_trend == "DETERIORATING":
        exit_code = CI_EXIT_WARN
        evaluation_status = "WARN"
        reasons = []
        if stability_band == STABILITY_BAND_DRIFTING:
            reasons.append("Fleet stability: DRIFTING")
        if broken_trend == "DETERIORATING":
            reasons.append("Broken rate trend: DETERIORATING")
        if underspec_trend == "DETERIORATING":
            reasons.append("Underspecification trend: DETERIORATING")
        reason = "; ".join(reasons)
    # OK (0): Stable
    else:
        exit_code = CI_EXIT_OK
        evaluation_status = "OK"
        reason = "Fleet stability: STABLE"
    
    evaluation = {
        "exit_code": exit_code,
        "status": evaluation_status,
        "stability_band": stability_band,
        "critical_metric_failures": sorted(critical_failures),
        "broken_rate_trend": broken_trend,
        "underspecification_trend": underspec_trend,
        "reason": reason,
    }
    
    return exit_code, evaluation


__all__ = [
    # Contract version
    "MOCK_ORACLE_CONTRACT_VERSION",
    # Contract definitions
    "PROFILE_CONTRACTS",
    "NEGATIVE_CONTROL_CONTRACT",
    "NegativeControlContract",
    # Contract verification
    "verify_profile_contracts",
    "verify_negative_control_result",
    # Contract export
    "export_mock_oracle_contract",
    # Scenario layer
    "Scenario",
    "SCENARIOS",
    "list_scenarios",
    "get_scenario",
    # Scenario analytics & drift detection
    "ANALYTICS_SCHEMA_VERSION",
    "DEFAULT_DRIFT_TOLERANCE",
    "DRIFT_TOLERANCE_SCALE_FACTOR",
    "DRIFT_STATUS_IN_CONTRACT",
    "DRIFT_STATUS_DRIFTED",
    "DRIFT_STATUS_BROKEN",
    "summarize_scenario_results",
    "detect_scenario_drift",
    # Adversarial coverage radar & governance hooks
    "VERIFICATION_METRICS",
    "MIN_SCENARIOS_FOR_WELL_EXERCISED",
    "GOVERNANCE_STATUS_OK",
    "GOVERNANCE_STATUS_ATTENTION",
    "GOVERNANCE_STATUS_BLOCK",
    "STATUS_LIGHT_GREEN",
    "STATUS_LIGHT_YELLOW",
    "STATUS_LIGHT_RED",
    "build_metric_scenario_coverage_view",
    "summarize_mock_oracle_drift_for_governance",
    "build_mock_oracle_director_panel",
    # Scenario fleet console & regression watchdog
    "FLEET_STATUS_OK",
    "FLEET_STATUS_ATTENTION",
    "FLEET_STATUS_GAP",
    "CRITICAL_METRICS",
    "MIN_SCENARIOS_FOR_SPECIFIED",
    "build_scenario_fleet_summary",
    "detect_mock_oracle_regression",
    # Config types
    "MockOracleConfig",
    "MockVerificationResult", 
    "SLICE_PROFILES",
    "BUCKET_BASE_LATENCY",
    "ProfileCoverageMap",
    "compute_profile_coverage",
]

