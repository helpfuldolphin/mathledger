"""
Mock Oracle Expectations Library for Phase II Testing.

This module provides pre-computed formulas that map to specific verification
outcomes. Test harnesses can use these to assert behaviors without
reverse-engineering the hash function.

ABSOLUTE SAFEGUARD: For tests only — never in production.

Usage:
    from backend.verification.mock_expectations import MockOracleExpectations
    
    # Get a formula guaranteed to verify under default profile
    formula = MockOracleExpectations.get_verified_formula("default")
    assert oracle.verify(formula).verified == True
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from backend.verification.mock_config import SLICE_PROFILES


def _hash_formula(normalized: str) -> int:
    """Compute deterministic integer hash from normalized formula."""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _bucket_for_hash(h: int, profile_name: str) -> str:
    """Determine bucket from hash using profile boundaries."""
    profile = SLICE_PROFILES[profile_name]
    mod = h % 100
    
    if mod < profile["verified"]:
        return "verified"
    elif mod < profile["failed"]:
        return "failed"
    elif mod < profile["abstain"]:
        return "abstain"
    elif mod < profile["timeout"]:
        return "timeout"
    elif mod < profile["error"]:
        return "error"
    else:
        return "crash"


def _find_formula_for_bucket(bucket: str, profile: str, start_index: int = 0) -> Tuple[str, int]:
    """
    Find a formula that maps to a specific bucket under a profile.
    
    Iterates through candidate formulas until finding one that hashes
    to the target bucket.
    
    Args:
        bucket: Target bucket name.
        profile: Profile name.
        start_index: Starting index for search.
        
    Returns:
        Tuple of (formula, hash_mod_100)
    """
    # Generate candidate formulas deterministically
    atoms = ["p", "q", "r", "s", "t"]
    ops = ["->", "/\\", "\\/"]
    
    index = start_index
    max_iterations = 10000
    
    for _ in range(max_iterations):
        # Generate formula based on index
        a1 = atoms[index % len(atoms)]
        a2 = atoms[(index // len(atoms)) % len(atoms)]
        op = ops[(index // (len(atoms) ** 2)) % len(ops)]
        
        # Various formula patterns
        pattern = index // (len(atoms) ** 2 * len(ops)) % 5
        if pattern == 0:
            formula = f"{a1} {op} {a2}"
        elif pattern == 1:
            formula = f"({a1} {op} {a2}) {op} {a1}"
        elif pattern == 2:
            formula = f"~{a1} {op} {a2}"
        elif pattern == 3:
            formula = f"({a1} {op} ~{a2}) {op} ({a2} {op} {a1})"
        else:
            formula = f"~(~{a1} {op} ~{a2})"
        
        # Add index suffix for uniqueness
        formula = f"{formula}_{index}"
        
        h = _hash_formula(formula)
        if _bucket_for_hash(h, profile) == bucket:
            return formula, h % 100
        
        index += 1
    
    raise RuntimeError(f"Could not find formula for bucket '{bucket}' in profile '{profile}'")


# Pre-computed formula tables for each profile and bucket
# These are computed once and cached for consistent test behavior
_FORMULA_CACHE: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}


def _ensure_cache(profile: str, bucket: str, count: int = 10) -> None:
    """Ensure cache has enough formulas for the profile/bucket combination."""
    if profile not in _FORMULA_CACHE:
        _FORMULA_CACHE[profile] = {}
    
    if bucket not in _FORMULA_CACHE[profile]:
        _FORMULA_CACHE[profile][bucket] = []
    
    cache = _FORMULA_CACHE[profile][bucket]
    
    # Find more formulas if needed
    start_index = len(cache) * 100  # Start from different region
    while len(cache) < count:
        formula, mod = _find_formula_for_bucket(bucket, profile, start_index)
        cache.append((formula, mod))
        start_index += 1000  # Jump ahead to find different formulas


class MockOracleExpectations:
    """
    Library of expected behaviors for test harnesses.
    
    Provides pre-computed formulas that produce specific outcomes.
    All formulas are guaranteed to hash to their designated bucket
    under the specified profile.
    
    Thread-safe: Formula cache is built lazily but deterministically.
    """
    
    @staticmethod
    def get_verified_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to verify under the given profile.
        
        Args:
            profile: Profile name.
            index: Index into the formula table (0-9 for variety).
            
        Returns:
            Formula string that will produce verified=True.
        """
        _ensure_cache(profile, "verified", max(10, index + 1))
        return _FORMULA_CACHE[profile]["verified"][index][0]
    
    @staticmethod
    def get_failed_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to fail verification.
        
        Args:
            profile: Profile name.
            index: Index into the formula table.
            
        Returns:
            Formula string that will produce verified=False, abstained=False.
        """
        _ensure_cache(profile, "failed", max(10, index + 1))
        return _FORMULA_CACHE[profile]["failed"][index][0]
    
    @staticmethod
    def get_abstain_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to produce abstention.
        
        Args:
            profile: Profile name.
            index: Index into the formula table.
            
        Returns:
            Formula string that will produce abstained=True.
        """
        _ensure_cache(profile, "abstain", max(10, index + 1))
        return _FORMULA_CACHE[profile]["abstain"][index][0]
    
    @staticmethod
    def get_timeout_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to timeout.
        
        Args:
            profile: Profile name.
            index: Index into the formula table.
            
        Returns:
            Formula string that will produce timed_out=True.
        """
        _ensure_cache(profile, "timeout", max(10, index + 1))
        return _FORMULA_CACHE[profile]["timeout"][index][0]
    
    @staticmethod
    def get_error_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to produce an error.
        
        Args:
            profile: Profile name.
            index: Index into the formula table.
            
        Returns:
            Formula string that will produce reason="mock-error".
        """
        _ensure_cache(profile, "error", max(10, index + 1))
        return _FORMULA_CACHE[profile]["error"][index][0]
    
    @staticmethod
    def get_crash_formula(profile: str = "default", index: int = 0) -> str:
        """
        Get a formula guaranteed to crash (if crashes enabled).
        
        Args:
            profile: Profile name.
            index: Index into the formula table.
            
        Returns:
            Formula string that will produce crashed=True or raise MockOracleCrashError.
        """
        _ensure_cache(profile, "crash", max(10, index + 1))
        return _FORMULA_CACHE[profile]["crash"][index][0]
    
    @staticmethod
    def get_goal_hit_set(count: int = 5) -> List[str]:
        """
        Get formulas that simulate rare goal hits for goal_hit profile.
        
        Args:
            count: Number of formulas to return.
            
        Returns:
            List of formulas that verify under goal_hit profile.
        """
        _ensure_cache("goal_hit", "verified", count)
        return [f for f, _ in _FORMULA_CACHE["goal_hit"]["verified"][:count]]
    
    @staticmethod
    def get_chain_formulas(depth: int = 3) -> List[str]:
        """
        Get formulas that simulate a chain-depth pattern for tree profile.
        
        Returns formulas that verify under tree profile, simulating
        a proof chain of the specified depth.
        
        Args:
            depth: Number of formulas in the chain.
            
        Returns:
            List of formulas for chain building.
        """
        _ensure_cache("tree", "verified", depth)
        return [f for f, _ in _FORMULA_CACHE["tree"]["verified"][:depth]]
    
    @staticmethod
    def get_dependency_goals(count: int = 3) -> List[str]:
        """
        Get formulas that simulate multi-goal dependencies.
        
        Returns formulas that verify under dependency profile.
        
        Args:
            count: Number of goal formulas.
            
        Returns:
            List of goal formulas for dependency testing.
        """
        _ensure_cache("dependency", "verified", count)
        return [f for f, _ in _FORMULA_CACHE["dependency"]["verified"][:count]]
    
    @staticmethod
    def get_formula_hash_mod(formula: str) -> int:
        """
        Get the hash mod 100 for a formula.
        
        Useful for debugging which bucket a formula falls into.
        
        Args:
            formula: Formula string.
            
        Returns:
            hash % 100 value.
        """
        return _hash_formula(formula) % 100
    
    @staticmethod
    def get_all_profiles() -> List[str]:
        """
        Get list of all available profiles.
        
        Returns:
            List of profile names.
        """
        return list(SLICE_PROFILES.keys())
    
    @staticmethod
    def get_profile_boundaries(profile: str) -> Dict[str, int]:
        """
        Get bucket boundaries for a profile.
        
        Args:
            profile: Profile name.
            
        Returns:
            Dictionary of bucket name → upper boundary.
        """
        if profile not in SLICE_PROFILES:
            raise ValueError(f"Invalid profile '{profile}'")
        return SLICE_PROFILES[profile].copy()


__all__ = ["MockOracleExpectations"]

