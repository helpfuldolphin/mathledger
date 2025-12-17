# REAL-READY
"""
Error Mapper for Lean Verification

Maps Lean verification outcomes to stable error codes.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: REAL-READY
"""

from __future__ import annotations


def map_lean_outcome_to_error_code(
    returncode: int,
    stderr: str,
    duration_ms: float,
    timeout_ms: float,
) -> str:
    """
    Map Lean verification outcome to stable error code.
    
    Args:
        returncode: Process return code
        stderr: Standard error output
        duration_ms: Verification duration in milliseconds
        timeout_ms: Timeout threshold in milliseconds
    
    Returns:
        Error code string (one of 11 stable codes)
    """
    
    stderr_lower = stderr.lower()
    
    # 1. Success
    if returncode == 0 and "error" not in stderr_lower:
        return "verified"
    
    # 2. Timeout (check duration or signal)
    if duration_ms >= timeout_ms * 0.95:  # Within 5% of timeout
        return "verifier_timeout"
    if returncode == 137 or returncode == -15:  # SIGKILL or SIGTERM
        return "verifier_timeout"
    
    # 3. Memory limit exceeded
    if "out of memory" in stderr_lower or "memory limit" in stderr_lower:
        return "memory_limit_exceeded"
    if returncode == 137 and duration_ms < timeout_ms * 0.5:
        # Killed early, likely OOM
        return "memory_limit_exceeded"
    
    # 4. Proof invalid (type errors, tactic failures)
    proof_invalid_patterns = [
        "error: type mismatch",
        "error: failed to synthesize",
        "error: tactic failed",
        "error: unsolved goals",
        "error: unknown identifier",
        "error: application type mismatch",
        "error: invalid field notation",
    ]
    
    for pattern in proof_invalid_patterns:
        if pattern in stderr_lower:
            return "proof_invalid"
    
    # 5. Proof incomplete
    if "error: unsolved goals" in stderr_lower:
        return "proof_incomplete"
    
    # 6. Internal errors
    internal_error_patterns = [
        "internal error",
        "panic",
        "assertion failed",
        "segmentation fault",
        "stack overflow",
    ]
    
    for pattern in internal_error_patterns:
        if pattern in stderr_lower:
            return "verifier_internal_error"
    
    # 7. Resource constraints (other than memory)
    if "resource limit" in stderr_lower or "quota exceeded" in stderr_lower:
        return "budget_exhausted"
    
    # 8. If returncode != 0 and no specific pattern matched, assume proof invalid
    if returncode != 0:
        return "proof_invalid"
    
    # 9. Default to unknown
    return "unknown"
