"""
Deterministic Mock Verification Oracle for Phase II Testing.

This module provides a fully deterministic mock oracle that simulates
verification outcomes without any external calls. Used ONLY in tests
and synthetic experiments — never in production or Evidence Pack v1/v2.

ABSOLUTE SAFEGUARDS:
- Mock oracle NEVER touches production runtime
- NEVER influences real verification (StatementVerifier, LeanFallback)
- Purely for tests, simulations, and CI stability
- No uplift claims may reference mock oracle results

Usage:
    from backend.verification.mock_oracle import MockVerifiableOracle
    from backend.verification.mock_config import MockOracleConfig
    
    config = MockOracleConfig(slice_profile="goal_hit", seed=42)
    oracle = MockVerifiableOracle(config)
    result = oracle.verify("p -> p")
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from backend.verification.mock_config import (
    BUCKET_BASE_LATENCY,
    SLICE_PROFILES,
    MockOracleConfig,
    MockVerificationResult,
    ProfileCoverageMap,
)
from backend.verification.mock_exceptions import MockOracleCrashError


class MockVerifiableOracle:
    """
    Deterministic mock oracle for controlled test environments.
    
    PROHIBITION: This class MUST NOT be imported or used in production code.
    It is strictly for tests, simulations, and CI stability.
    
    The oracle uses a deterministic hash-to-behavior mapping:
    1. SHA-256 hash of normalized formula → integer
    2. Integer mod 100 → bucket selection based on profile
    3. Bucket → verification outcome (verified/failed/abstain/timeout/crash)
    4. Deterministic latency based on bucket and hash
    
    Same input formula + same config → identical result across runs/platforms.
    
    Attributes:
        config: MockOracleConfig controlling behavior.
        _profile: Current slice profile bucket boundaries.
        _stats: Internal statistics counters.
    """
    
    def __init__(self, config: Optional[MockOracleConfig] = None) -> None:
        """
        Initialize mock oracle with configuration.
        
        Args:
            config: MockOracleConfig instance. Defaults to default profile.
        """
        self.config = config or MockOracleConfig()
        self._profile = SLICE_PROFILES[self.config.slice_profile]
        self._stats = {
            "verified": 0,
            "failed": 0,
            "abstain": 0,
            "timeout": 0,
            "error": 0,
            "crash": 0,
            "total": 0,
        }
    
    def _hash_formula(self, normalized: str) -> int:
        """
        Compute deterministic integer hash from normalized formula.
        
        Args:
            normalized: Canonicalized formula string.
            
        Returns:
            Integer hash derived from SHA-256.
        """
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return int(digest, 16)
    
    def _bucket_for_hash(self, h: int) -> str:
        """
        Determine bucket from hash using profile boundaries.
        
        Uses h % 100 and compares against cumulative profile boundaries.
        
        Args:
            h: Integer hash from _hash_formula.
            
        Returns:
            Bucket name: "verified", "failed", "abstain", "timeout", "error", "crash"
        """
        mod = h % 100
        
        if mod < self._profile["verified"]:
            return "verified"
        elif mod < self._profile["failed"]:
            return "failed"
        elif mod < self._profile["abstain"]:
            return "abstain"
        elif mod < self._profile["timeout"]:
            return "timeout"
        elif mod < self._profile["error"]:
            return "error"
        else:
            return "crash"
    
    def _compute_latency(self, bucket: str, h: int) -> int:
        """
        Compute deterministic latency for a verification result.
        
        Latency = base_latency + deterministic_jitter
        Jitter is derived from hash to ensure reproducibility.
        
        Args:
            bucket: The bucket name.
            h: Integer hash for jitter calculation.
            
        Returns:
            Latency in milliseconds.
        """
        if bucket == "timeout":
            base = self.config.timeout_ms
        else:
            base = BUCKET_BASE_LATENCY.get(bucket, 5)
        
        # Deterministic jitter: use different bits of hash for jitter
        jitter_range = int(base * self.config.latency_jitter_pct)
        if jitter_range > 0:
            # Use bits 8-24 of hash for jitter (avoiding low bits used for bucket)
            jitter = (h >> 8) % (jitter_range * 2 + 1) - jitter_range
        else:
            jitter = 0
        
        return max(0, base + jitter)
    
    def verify(self, normalized: str) -> MockVerificationResult:
        """
        Perform deterministic mock verification.
        
        Args:
            normalized: Canonicalized formula string.
            
        Returns:
            MockVerificationResult with deterministic outcome.
            
        Raises:
            MockOracleCrashError: If bucket is "crash" and enable_crashes=True.
        
        Note:
            If config.negative_control=True, all evaluations return verified=False
            with reason="negative_control". Stats are NOT updated in this mode.
        """
        h = self._hash_formula(normalized)
        
        # Negative control mode: override all evaluations
        if self.config.negative_control:
            # Deterministic latency even in negative control mode
            latency = self._compute_latency("abstain", h)
            # Stats tracking is suppressed in negative control mode
            return MockVerificationResult(
                verified=False,
                abstained=True,
                timed_out=False,
                crashed=False,
                reason="negative_control",
                latency_ms=latency,
                bucket="negative_control",
                hash_int=h,
            )
        
        bucket = self._bucket_for_hash(h)
        latency = self._compute_latency(bucket, h)
        
        # Update stats (only in normal mode, not negative control)
        self._stats[bucket] += 1
        self._stats["total"] += 1
        
        # Handle crash bucket
        if bucket == "crash":
            if self.config.enable_crashes:
                raise MockOracleCrashError(
                    formula=normalized,
                    hash_int=h,
                    reason="mock-crash-bucket",
                )
            # If crashes disabled, return result with crashed=True
            return MockVerificationResult(
                verified=False,
                abstained=False,
                timed_out=False,
                crashed=True,
                reason="mock-crash-disabled",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
        
        # Build result based on bucket
        if bucket == "verified":
            return MockVerificationResult(
                verified=True,
                abstained=False,
                timed_out=False,
                crashed=False,
                reason="mock-verified",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
        elif bucket == "failed":
            return MockVerificationResult(
                verified=False,
                abstained=False,
                timed_out=False,
                crashed=False,
                reason="mock-failed",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
        elif bucket == "abstain":
            return MockVerificationResult(
                verified=False,
                abstained=True,
                timed_out=False,
                crashed=False,
                reason="mock-abstain",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
        elif bucket == "timeout":
            return MockVerificationResult(
                verified=False,
                abstained=False,
                timed_out=True,
                crashed=False,
                reason="mock-timeout",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
        else:  # error bucket
            return MockVerificationResult(
                verified=False,
                abstained=False,
                timed_out=False,
                crashed=False,
                reason="mock-error",
                latency_ms=latency,
                bucket=bucket,
                hash_int=h,
            )
    
    def verify_batch(self, formulas: List[str]) -> List[MockVerificationResult]:
        """
        Verify multiple formulas with consistent behavior.
        
        Args:
            formulas: List of normalized formula strings.
            
        Returns:
            List of MockVerificationResult, one per formula.
            
        Raises:
            MockOracleCrashError: If any formula crashes and enable_crashes=True.
        """
        return [self.verify(f) for f in formulas]
    
    def get_expected_outcome(self, normalized: str) -> str:
        """
        Preview what outcome a formula would produce without affecting stats.
        
        Useful for test assertions without side effects.
        
        Args:
            normalized: Canonicalized formula string.
            
        Returns:
            Bucket name: "verified", "failed", "abstain", "timeout", "error", "crash"
        """
        h = self._hash_formula(normalized)
        return self._bucket_for_hash(h)
    
    def set_profile(self, profile: str) -> None:
        """
        Switch slice profile.
        
        Args:
            profile: Profile name ("goal_hit", "sparse", "tree", "dependency", "default")
            
        Raises:
            ValueError: If profile is not valid.
        """
        if profile not in SLICE_PROFILES:
            raise ValueError(f"Invalid profile '{profile}'. Must be one of: {set(SLICE_PROFILES.keys())}")
        self._profile = SLICE_PROFILES[profile]
        # Note: This creates a new config since MockOracleConfig is frozen
        self.config = MockOracleConfig(
            slice_profile=profile,
            timeout_ms=self.config.timeout_ms,
            enable_crashes=self.config.enable_crashes,
            latency_jitter_pct=self.config.latency_jitter_pct,
            seed=self.config.seed,
            negative_control=self.config.negative_control,
            target_hashes=self.config.target_hashes,
            required_goals=self.config.required_goals,
            chain_depth_map=self.config.chain_depth_map,
        )
    
    def profile_coverage(self) -> ProfileCoverageMap:
        """
        Get static coverage map for current profile.
        
        Returns coverage percentages computed from profile boundaries,
        NOT from runtime statistics. Useful for test planning and
        validating expected distributions.
        
        Returns:
            ProfileCoverageMap with static percentage allocations.
            
        Note:
            In negative_control mode, this still returns the profile's
            theoretical coverage (though all actual results will be
            negative_control abstentions).
        """
        return ProfileCoverageMap.from_profile(self.config.slice_profile)
    
    def reset_stats(self) -> None:
        """Reset internal statistics counters."""
        for key in self._stats:
            self._stats[key] = 0
    
    @property
    def stats(self) -> dict:
        """
        Get verification statistics for this session.
        
        Returns:
            Dictionary with counts: verified, failed, abstain, timeout, error, crash, total
        """
        return self._stats.copy()
    
    def get_bucket_distribution(self) -> dict:
        """
        Get the percentage distribution of buckets for current profile.
        
        Returns:
            Dictionary mapping bucket names to their percentage ranges.
        """
        profile = self._profile
        return {
            "verified": f"0-{profile['verified']-1} ({profile['verified']}%)",
            "failed": f"{profile['verified']}-{profile['failed']-1} ({profile['failed'] - profile['verified']}%)",
            "abstain": f"{profile['failed']}-{profile['abstain']-1} ({profile['abstain'] - profile['failed']}%)",
            "timeout": f"{profile['abstain']}-{profile['timeout']-1} ({profile['timeout'] - profile['abstain']}%)",
            "error": f"{profile['timeout']}-{profile['error']-1} ({profile['error'] - profile['timeout']}%)",
            "crash": f"{profile['error']}-99 ({100 - profile['error']}%)",
        }
    
    @property
    def is_negative_control(self) -> bool:
        """
        Check if oracle is in negative control mode.
        
        Returns:
            True if all evaluations return negative_control abstention.
        """
        return self.config.negative_control


__all__ = ["MockVerifiableOracle"]

