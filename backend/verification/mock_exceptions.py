"""
Mock Oracle Exceptions for Phase II Testing.

Custom exceptions raised by the mock verification oracle to simulate
failure modes in controlled test environments.

ABSOLUTE SAFEGUARD: These exceptions are for tests only — never in production.
"""

from __future__ import annotations


class MockOracleError(Exception):
    """Base exception for mock oracle errors."""
    
    pass


class MockOracleCrashError(MockOracleError):
    """
    Simulated crash from mock oracle.
    
    Raised when:
    - Formula hash falls into the "crash" bucket
    - MockOracleConfig.enable_crashes is True
    
    This allows tests to verify that callers properly handle
    unexpected verification failures.
    
    Attributes:
        formula: The formula that triggered the crash.
        hash_int: The integer hash that determined the crash bucket.
        reason: Description of why the crash was triggered.
    """
    
    def __init__(
        self,
        formula: str,
        hash_int: int,
        reason: str = "mock-crash-simulated",
    ) -> None:
        self.formula = formula
        self.hash_int = hash_int
        self.reason = reason
        super().__init__(
            f"MockOracleCrashError: {reason} "
            f"(formula_hash={hash_int}, formula={formula[:50]}...)"
        )


class MockOracleTimeoutError(MockOracleError):
    """
    Simulated timeout from mock oracle.
    
    Raised when:
    - Formula hash falls into the "timeout" bucket
    - Caller explicitly requests timeout-as-exception mode
    
    Note: By default, timeouts return MockVerificationResult with timed_out=True
    rather than raising. This exception is for callers who want exception semantics.
    
    Attributes:
        formula: The formula that triggered the timeout.
        timeout_ms: The simulated timeout duration.
    """
    
    def __init__(
        self,
        formula: str,
        timeout_ms: int,
        reason: str = "mock-timeout-simulated",
    ) -> None:
        self.formula = formula
        self.timeout_ms = timeout_ms
        self.reason = reason
        super().__init__(
            f"MockOracleTimeoutError: {reason} "
            f"(timeout_ms={timeout_ms}, formula={formula[:50]}...)"
        )


class MockOracleConfigError(MockOracleError):
    """
    Configuration error for mock oracle.
    
    Raised when MockOracleConfig is invalid or inconsistent.
    """
    
    pass


__all__ = [
    "MockOracleError",
    "MockOracleCrashError",
    "MockOracleTimeoutError",
    "MockOracleConfigError",
]

