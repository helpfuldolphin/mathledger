# backend/verification/lean_adapter.py
"""
Phase IIb Lean Adapter

STATUS: SCAFFOLD + SIMULATION — DOES NOT INVOKE LEAN.

This module defines the types and interface that will be used
when Lean integration is reintroduced in Phase IIb uplift experiments.

=============================================================================
PHASE IIb INTEGRATION PLAN
=============================================================================

STEP 1: Replace LeanFallback in derivation/verification.py
------------------------------------------------------------------------
Current:
    class StatementVerifier:
        def __init__(self, bounds, lean_project_root):
            self._lean = LeanFallback(lean_project_root, bounds.lean_timeout_s)
        
        def verify(self, normalized):
            # ... pattern matching, truth table ...
            return self._lean.verify(normalized)

Phase IIb:
    from backend.verification import LeanAdapter, LeanVerificationRequest
    
    class StatementVerifier:
        def __init__(self, bounds, lean_project_root):
            self._lean_adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)
        
        def verify(self, normalized):
            # ... pattern matching, truth table ...
            request = LeanVerificationRequest(
                canonical=normalized,
                job_id=deterministic_uuid(normalized).hex[:12],
                resource_budget=LeanResourceBudget(
                    timeout_seconds=int(bounds.lean_timeout_s),
                ),
            )
            result = self._lean_adapter.verify(request)
            return VerificationOutcome(
                verified=result.verified,
                method=result.method,
                details=result.abstention_reason.value if result.abstention_reason else None,
            )

STEP 2: Wire into worker.py::execute_lean_job()
------------------------------------------------------------------------
Current:
    def execute_lean_job(raw_statement, *, build_runner=run_lake_build, ...):
        stmt = sanitize_statement(raw_statement)
        result = build_runner(module_name)
        return LeanJobResult(...)

Phase IIb:
    def execute_lean_job(raw_statement, *, adapter=None, ...):
        stmt = sanitize_statement(raw_statement)
        adapter = adapter or LeanAdapter(mode=LeanAdapterMode.ACTIVE)
        request = LeanVerificationRequest(
            canonical=stmt.canonical,
            job_id=jid,
            resource_budget=LeanResourceBudget(timeout_seconds=LEAN_BUILD_TIMEOUT),
        )
        result = adapter.verify(request)
        # Convert LeanVerificationResult to LeanJobResult for compatibility

STEP 3: Extend lean_mode.py::get_build_runner()
------------------------------------------------------------------------
Add a new mode that routes through LeanAdapter:

    class LeanMode(Enum):
        MOCK = "mock"
        DRY_RUN = "dry_run"
        FULL = "full"
        PHASE_IIB = "phase2b"  # New mode
    
    def get_build_runner(mode=None, ...):
        if mode == LeanMode.PHASE_IIB:
            adapter = LeanAdapter(mode=LeanAdapterMode.ACTIVE)
            return lambda module_name: _adapter_to_completed_process(adapter, module_name)

=============================================================================
SAFETY INVARIANTS FOR PHASE IIb
=============================================================================

1. DETERMINISTIC SUBPROCESS CALLS
   - All Lean invocations MUST be via subprocess.run() with:
     * capture_output=True (no console side effects)
     * text=True (deterministic encoding)
     * check=False (handle errors explicitly)
     * timeout=budget.timeout_seconds (hard limit)
   - Environment variables MUST be explicitly controlled
   - Working directory MUST be isolated per job

2. TIMEOUT ENFORCEMENT
   - Primary: subprocess.run(timeout=N) raises TimeoutExpired
   - Secondary: Signal-based kill after grace period
   - Return: abstention_reason=LEAN_TIMEOUT (never hang)
   - Maximum timeout: 90 seconds (hard cap, non-configurable)

3. RESOURCE CAPS
   - Memory: Use resource.setrlimit() on Unix, job objects on Windows
   - Disk: Pre-compute max output size, fail if exceeded
   - CPU: Single-threaded only (LEAN_NUM_THREADS=1)
   - Network: Offline mode (LAKE_OFFLINE=1, LAKE_NO_CACHE=1)

4. VERSION PINNING
   - MUST validate Lean version matches LEAN_VERSION_REQUIRED
   - MUST reject version mismatch with LEAN_VERSION_MISMATCH
   - MUST use ELAN_AUTO_UPDATE=false to prevent upgrades

5. ISOLATION
   - Each job runs in isolated directory under LeanSandbox
   - No shared state between verification jobs
   - Cleanup MUST occur even on timeout/crash

6. DETERMINISM
   - Same canonical input MUST produce same verification outcome
   - Hash of (canonical, lean_version, timeout) determines result
   - No reliance on wall-clock time in result computation

=============================================================================
MODES
=============================================================================

- DISABLED: Immediate abstention, no computation
- PHASE_IIB_SCAFFOLD: Deterministic abstention with version info
- SIMULATE: Deterministic simulation based on formula complexity
- ACTIVE: Real Lean verification (Phase IIb only, NOT IMPLEMENTED)

=============================================================================

The adapter provides:
- Deterministic interface for Lean verification requests
- Strict timeout and memory budget guards (enforced in Phase IIb)
- Lean version pinning with validation stubs
- Safe failure modes: abstain_lean_timeout, abstain_lean_error, etc.
- Simulation mode for testing without Lean

Current Behavior (Phase II):
- DISABLED/PHASE_IIB_SCAFFOLD: Return verified=False with LEAN_DISABLED
- SIMULATE: Return deterministic result based on hash(canonical)
- No subprocess calls to Lean are made
- Fully deterministic based on input canonical string

DETERMINISM GUARANTEE:
All verification outcomes (verified, abstention_reason, deterministic_hash,
simulated_complexity) are fully deterministic given identical inputs.
The duration_ms field is observational metadata (wall-clock time measurement)
and does NOT affect verification outcomes. It is the only non-deterministic
field in LeanVerificationResult.

Environment Variables (Phase IIb, NOT USED YET):
    LEAN_ADAPTER_MODE: "disabled", "phase2b_scaffold", "simulate", "active"
    LEAN_ADAPTER_TIMEOUT: Per-verification timeout (default: 30s)
    LEAN_ADAPTER_MEMORY_MB: Memory budget (default: 2048)

See Also:
    - backend/lean_mode.py: Current three-mode verification
    - backend/lean_control_sandbox.py: Sandbox skeleton
    - derivation/verification.py: Layered verifier with LeanFallback
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence

# =============================================================================
# CONSTANTS
# =============================================================================

# Pinned Lean version for Phase IIb compatibility
LEAN_VERSION_REQUIRED = "v4.23.0-rc2"

# Deterministic salt for hash computation (ensures reproducibility)
_SCAFFOLD_HASH_SALT = b"mathledger_phase2b_lean_adapter_v1"

# Simulation mode constants
_SIMULATE_HASH_SALT = b"mathledger_lean_simulate_v1"

# Formula complexity thresholds for simulation
_SIMULATE_TIMEOUT_COMPLEXITY_THRESHOLD = 50  # chars in canonical
_SIMULATE_RESOURCE_COMPLEXITY_THRESHOLD = 80  # chars in canonical

# Hash modulo for deterministic branching in simulation
_SIMULATE_BRANCH_MODULO = 100

# Supported characters in canonical form (ASCII propositional logic)
_CANONICAL_VALID_PATTERN = re.compile(r'^[a-zA-Z0-9_\s\(\)\-\>\<\~\\/\&\|\!\^]+$')

# Maximum canonical length (prevent DoS)
_MAX_CANONICAL_LENGTH = 10000


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LeanAdapterValidationError(ValueError):
    """
    Raised when a LeanVerificationRequest fails validation.
    
    This is a programming error, not a verification failure.
    Invalid requests should be caught before reaching the adapter.
    """
    pass


# =============================================================================
# ENUMS
# =============================================================================

class LeanAdapterMode(Enum):
    """
    Operational modes for the Lean adapter.
    
    DISABLED: Adapter is completely off, returns immediate abstention.
    PHASE_IIB_SCAFFOLD: Scaffold mode for testing interface (current).
    SIMULATE: Simulation mode for dry-run testing without Lean.
    SHADOW: Shadow mode with simulated Lean subprocess telemetry.
    ACTIVE: Real Lean verification (Phase IIb, NOT IMPLEMENTED).
    """
    
    DISABLED = "disabled"
    """Adapter disabled, immediate abstention."""
    
    PHASE_IIB_SCAFFOLD = "phase2b_scaffold"
    """Scaffold mode: deterministic stubs, no Lean calls."""
    
    SIMULATE = "simulate"
    """Simulation mode: deterministic results based on formula complexity."""
    
    SHADOW = "shadow"
    """Shadow mode: simulation with subprocess-like telemetry emission."""
    
    ACTIVE = "active"
    """Active mode: real Lean verification (Phase IIb only)."""


class LeanAbstentionReason(Enum):
    """
    Reasons for Lean verification abstention.
    
    These codes provide structured failure modes that can be
    distinguished in metrics and attestations.
    """
    
    LEAN_DISABLED = "lean_disabled"
    """Lean verification is disabled by configuration."""
    
    LEAN_TIMEOUT = "lean_timeout"
    """Verification exceeded timeout budget."""
    
    LEAN_ERROR = "lean_error"
    """Internal error during Lean execution."""
    
    LEAN_UNAVAILABLE = "lean_unavailable"
    """Lean toolchain not installed or not found."""
    
    LEAN_RESOURCE_EXCEEDED = "lean_resource_exceeded"
    """Memory or disk budget exceeded."""
    
    LEAN_VERSION_MISMATCH = "lean_version_mismatch"
    """Installed Lean version does not match required version."""


class VerificationErrorKind(Enum):
    """
    Error taxonomy for Lean verification outcomes.
    
    This enum provides a stable, explicit error surface that will remain
    consistent when real Lean integration is added in Phase IIb. All
    non-success outcomes MUST be tagged with an appropriate error_kind.
    
    PHASE IIb READINESS:
    This enum is forward-compatible. New values may be added in Phase IIb
    but existing values will not change semantics.
    """
    
    NONE = "none"
    """No error — verification succeeded."""
    
    INVALID_REQUEST = "invalid_request"
    """Request failed validation (bad canonical, job_id, etc.)."""
    
    SIMULATION_ONLY = "simulation_only"
    """Verification was simulated, not real (Phase II mode)."""
    
    RESOURCE_LIMIT = "resource_limit"
    """Resource budget exceeded (timeout, memory, disk)."""
    
    INTERNAL_ERROR = "internal_error"
    """Internal adapter error (unexpected exception)."""
    
    LEAN_UNAVAILABLE = "lean_unavailable"
    """Lean toolchain not available or version mismatch."""
    
    NOT_IMPLEMENTED = "not_implemented"
    """Feature not implemented in current phase."""


# =============================================================================
# DATACLASSES
# =============================================================================

# Resource budget limits (Phase IIb readiness)
_MAX_TIMEOUT_SECONDS = 90  # Hard cap, non-configurable
_MAX_MEMORY_MB = 8192  # 8GB max
_MAX_DISK_MB = 1024  # 1GB max
_MAX_PROOFS_PER_REQUEST = 100  # Max proof attempts


def validate_resource_budget(budget: "LeanResourceBudget") -> list[str]:
    """
    Validate a LeanResourceBudget.
    
    Returns a list of error messages. Empty list means valid.
    
    Args:
        budget: The budget to validate.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    
    # timeout_seconds validation
    if budget.timeout_seconds < 1:
        errors.append("timeout_seconds must be >= 1")
    elif budget.timeout_seconds > _MAX_TIMEOUT_SECONDS:
        errors.append(
            f"timeout_seconds exceeds maximum ({budget.timeout_seconds} > {_MAX_TIMEOUT_SECONDS})"
        )
    
    # memory_mb validation
    if budget.memory_mb < 1:
        errors.append("memory_mb must be >= 1")
    elif budget.memory_mb > _MAX_MEMORY_MB:
        errors.append(
            f"memory_mb exceeds maximum ({budget.memory_mb} > {_MAX_MEMORY_MB})"
        )
    
    # disk_mb validation
    if budget.disk_mb < 1:
        errors.append("disk_mb must be >= 1")
    elif budget.disk_mb > _MAX_DISK_MB:
        errors.append(
            f"disk_mb exceeds maximum ({budget.disk_mb} > {_MAX_DISK_MB})"
        )
    
    # max_proofs validation
    if budget.max_proofs < 1:
        errors.append("max_proofs must be >= 1")
    elif budget.max_proofs > _MAX_PROOFS_PER_REQUEST:
        errors.append(
            f"max_proofs exceeds maximum ({budget.max_proofs} > {_MAX_PROOFS_PER_REQUEST})"
        )
    
    return errors


@dataclass(frozen=True)
class LeanResourceBudget:
    """
    Resource constraints for a Lean verification job.
    
    PHASE IIb READINESS:
    These budgets will be enforced in Phase IIb to ensure bounded resource
    consumption and deterministic timeouts. In Phase II simulation mode,
    the budget is validated but not enforced — it is carried through to
    result metadata for forward compatibility.
    
    Validation Rules:
        - All values must be non-negative
        - timeout_seconds: 1 <= value <= 90 (hard cap)
        - memory_mb: 1 <= value <= 8192 (8GB max)
        - disk_mb: 1 <= value <= 1024 (1GB max)
        - max_proofs: 1 <= value <= 100
    
    Attributes:
        timeout_seconds: Maximum wall-clock time for verification.
        memory_mb: Maximum memory usage in megabytes.
        disk_mb: Maximum disk usage for temporary files.
        max_proofs: Maximum proof attempts before abstention.
    """
    
    timeout_seconds: int = 30
    """Per-job timeout in seconds (default: 30s for Phase IIb)."""
    
    memory_mb: int = 2048
    """Memory limit in MB (default: 2GB)."""
    
    disk_mb: int = 100
    """Disk quota in MB (default: 100MB)."""
    
    max_proofs: int = 10
    """Maximum proof attempts before abstention (default: 10)."""
    
    def __post_init__(self) -> None:
        """Validate budget constraints after initialization."""
        errors = validate_resource_budget(self)
        if errors:
            raise LeanAdapterValidationError(
                f"Invalid LeanResourceBudget: {'; '.join(errors)}"
            )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/attestation."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "max_proofs": self.max_proofs,
        }
    
    @classmethod
    def default(cls) -> "LeanResourceBudget":
        """Create a budget with safe Phase II defaults."""
        return cls()
    
    @classmethod
    def minimal(cls) -> "LeanResourceBudget":
        """Create a minimal budget for quick tests."""
        return cls(timeout_seconds=5, memory_mb=512, disk_mb=50, max_proofs=1)
    
    @classmethod
    def generous(cls) -> "LeanResourceBudget":
        """Create a generous budget for complex proofs."""
        return cls(timeout_seconds=60, memory_mb=4096, disk_mb=500, max_proofs=50)


@dataclass(frozen=True)
class LeanVerificationRequest:
    """
    A request for Lean verification.
    
    Encapsulates all inputs needed to verify a statement,
    including the canonical form and resource constraints.
    
    Attributes:
        canonical: Canonical ASCII form of the statement.
        job_id: Unique identifier for this verification job.
        resource_budget: Resource constraints for this job.
    
    Validation Rules:
        - canonical must be non-empty
        - canonical must not exceed MAX_CANONICAL_LENGTH
        - canonical must contain only supported characters
        - resource_budget must not be None
        - job_id must be non-empty
    """
    
    canonical: str
    """Canonical ASCII form of the statement (normalized)."""
    
    job_id: str
    """Unique job identifier (typically deterministic UUID prefix)."""
    
    resource_budget: LeanResourceBudget = field(
        default_factory=LeanResourceBudget
    )
    """Resource constraints for this verification."""
    
    def __post_init__(self) -> None:
        """Validate request fields after initialization."""
        # Note: frozen=True means we can't modify, but __post_init__ runs
        # after field assignment, so we just validate here.
        errors = validate_verification_request(self)
        if errors:
            raise LeanAdapterValidationError(
                f"Invalid LeanVerificationRequest: {'; '.join(errors)}"
            )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/attestation."""
        return {
            "canonical": self.canonical,
            "job_id": self.job_id,
            "resource_budget": self.resource_budget.to_dict(),
        }
    
    def deterministic_hash(self) -> str:
        """
        Compute deterministic hash of this request.
        
        Used for reproducibility and cache keys.
        
        Returns:
            SHA256 hex digest of canonical + job_id + salt.
        """
        content = f"{self.canonical}:{self.job_id}".encode("utf-8")
        return hashlib.sha256(_SCAFFOLD_HASH_SALT + content).hexdigest()


@dataclass(frozen=True)
class LeanVerificationResult:
    """
    Result of a Lean verification attempt.
    
    Provides structured output including success/failure status,
    abstention reason (if applicable), and metadata for attestation.
    
    PHASE IIb READINESS:
    All non-success outcomes have an explicit error_kind for stable error
    taxonomy. The resource_budget_applied field carries through the budget
    from the request for forward compatibility.
    
    Attributes:
        verified: True if statement was verified as tautology.
        abstention_reason: Reason for abstention (if verified=False).
        method: Verification method used (e.g., "lean_adapter_scaffold").
        lean_version_checked: Lean version validated (if applicable).
        duration_ms: Wall-clock time for verification in milliseconds.
        deterministic_hash: Hash of the verification for reproducibility.
        simulated_complexity: Formula complexity score (simulation mode only).
        error_kind: Stable error taxonomy for non-success outcomes.
        resource_budget_applied: The budget that was applied (echoed from request).
    """
    
    verified: bool
    """True if statement was proven to be a tautology."""
    
    abstention_reason: Optional[LeanAbstentionReason]
    """Reason for abstention, None if verified=True."""
    
    method: str
    """Verification method identifier."""
    
    lean_version_checked: Optional[str]
    """Lean version that was validated, if any."""
    
    duration_ms: int
    """Verification duration in milliseconds."""
    
    deterministic_hash: str
    """Deterministic hash for reproducibility checks."""
    
    simulated_complexity: Optional[int] = None
    """Formula complexity score used in simulation (None if not simulated)."""
    
    error_kind: VerificationErrorKind = VerificationErrorKind.NONE
    """Stable error taxonomy for non-success outcomes."""
    
    resource_budget_applied: Optional[Dict[str, Any]] = None
    """The resource budget that was applied (echoed from request for metadata)."""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/attestation."""
        result = {
            "verified": self.verified,
            "abstention_reason": (
                self.abstention_reason.value if self.abstention_reason else None
            ),
            "method": self.method,
            "lean_version_checked": self.lean_version_checked,
            "duration_ms": self.duration_ms,
            "deterministic_hash": self.deterministic_hash,
            "error_kind": self.error_kind.value,
        }
        if self.simulated_complexity is not None:
            result["simulated_complexity"] = self.simulated_complexity
        if self.resource_budget_applied is not None:
            result["resource_budget_applied"] = self.resource_budget_applied
        return result
    
    @property
    def is_abstention(self) -> bool:
        """True if this result represents an abstention."""
        return not self.verified and self.abstention_reason is not None
    
    @property
    def is_success(self) -> bool:
        """True if verification succeeded."""
        return self.verified and self.error_kind == VerificationErrorKind.NONE
    
    @property
    def is_error(self) -> bool:
        """True if result represents an error (not just abstention)."""
        return self.error_kind != VerificationErrorKind.NONE


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_verification_request(
    request: LeanVerificationRequest,
) -> list[str]:
    """
    Validate a LeanVerificationRequest.
    
    Returns a list of error messages. Empty list means valid.
    
    Validation rules:
    1. canonical must be non-empty
    2. canonical must not exceed MAX_CANONICAL_LENGTH
    3. canonical must contain only supported characters
    4. resource_budget must not be None
    5. job_id must be non-empty
    
    Args:
        request: The request to validate.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    
    # Rule 1: canonical must be non-empty
    if not request.canonical:
        errors.append("canonical must be non-empty")
    elif len(request.canonical.strip()) == 0:
        errors.append("canonical must not be whitespace-only")
    
    # Rule 2: canonical must not exceed MAX_CANONICAL_LENGTH
    if request.canonical and len(request.canonical) > _MAX_CANONICAL_LENGTH:
        errors.append(
            f"canonical exceeds maximum length ({len(request.canonical)} > {_MAX_CANONICAL_LENGTH})"
        )
    
    # Rule 3: canonical must contain only supported characters
    if request.canonical and not _CANONICAL_VALID_PATTERN.match(request.canonical):
        # Find the invalid characters for helpful error message
        invalid_chars = set()
        for char in request.canonical:
            if not _CANONICAL_VALID_PATTERN.match(char):
                invalid_chars.add(repr(char))
        if invalid_chars:
            errors.append(
                f"canonical contains unsupported characters: {', '.join(sorted(invalid_chars))}"
            )
    
    # Rule 4: resource_budget must not be None
    # Note: With default_factory, this should never be None, but check anyway
    if request.resource_budget is None:
        errors.append("resource_budget must not be None")
    
    # Rule 5: job_id must be non-empty
    if not request.job_id:
        errors.append("job_id must be non-empty")
    elif len(request.job_id.strip()) == 0:
        errors.append("job_id must not be whitespace-only")
    
    return errors


def is_valid_canonical(canonical: str) -> bool:
    """
    Check if a canonical string is valid for Lean verification.
    
    Args:
        canonical: The canonical form to check.
    
    Returns:
        True if the canonical form is valid.
    """
    if not canonical or len(canonical.strip()) == 0:
        return False
    if len(canonical) > _MAX_CANONICAL_LENGTH:
        return False
    if not _CANONICAL_VALID_PATTERN.match(canonical):
        return False
    return True


# =============================================================================
# SIMULATION HELPER
# =============================================================================

def simulate_lean_result(
    canonical: str,
    job_id: str,
    resource_budget: LeanResourceBudget,
) -> LeanVerificationResult:
    """
    Simulate a Lean verification result without invoking Lean.
    
    This function produces deterministic results based on:
    1. Formula complexity (length of canonical form)
    2. Hash of canonical form for deterministic branching
    
    Simulation logic:
    - If complexity > RESOURCE_THRESHOLD: abstain with RESOURCE_EXCEEDED
    - If complexity > TIMEOUT_THRESHOLD: abstain with TIMEOUT
    - Otherwise: hash-based deterministic success/failure
    
    The result is fully deterministic: same inputs always produce same outputs.
    All results include error_kind and resource_budget_applied for Phase IIb
    forward compatibility.
    
    Args:
        canonical: Canonical ASCII form of the statement.
        job_id: Unique job identifier.
        resource_budget: Resource constraints (echoed in result metadata).
    
    Returns:
        LeanVerificationResult with simulated outcome.
    
    Example:
        >>> result = simulate_lean_result("p->p", "abc123", LeanResourceBudget())
        >>> result.method
        'lean_adapter_simulate'
        >>> result.verified  # Deterministic based on hash
        True
        >>> result.error_kind
        <VerificationErrorKind.NONE: 'none'>
    """
    start_time = time.perf_counter()
    
    # Compute complexity score based on canonical length
    complexity = len(canonical)
    
    # Compute deterministic hash for branching decisions
    hash_input = f"{canonical}:{job_id}".encode("utf-8")
    full_hash = hashlib.sha256(_SIMULATE_HASH_SALT + hash_input).hexdigest()
    branch_value = int(full_hash[:8], 16) % _SIMULATE_BRANCH_MODULO
    
    # Deterministic request hash for result
    request_hash = hashlib.sha256(_SCAFFOLD_HASH_SALT + hash_input).hexdigest()
    
    # Echo budget in result metadata
    budget_dict = resource_budget.to_dict()
    
    # Simulate resource exhaustion for very complex formulas
    if complexity > _SIMULATE_RESOURCE_COMPLEXITY_THRESHOLD:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return LeanVerificationResult(
            verified=False,
            abstention_reason=LeanAbstentionReason.LEAN_RESOURCE_EXCEEDED,
            method="lean_adapter_simulate",
            lean_version_checked=LEAN_VERSION_REQUIRED,
            duration_ms=duration_ms,
            deterministic_hash=request_hash,
            simulated_complexity=complexity,
            error_kind=VerificationErrorKind.RESOURCE_LIMIT,
            resource_budget_applied=budget_dict,
        )
    
    # Simulate timeout for moderately complex formulas
    if complexity > _SIMULATE_TIMEOUT_COMPLEXITY_THRESHOLD:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return LeanVerificationResult(
            verified=False,
            abstention_reason=LeanAbstentionReason.LEAN_TIMEOUT,
            method="lean_adapter_simulate",
            lean_version_checked=LEAN_VERSION_REQUIRED,
            duration_ms=duration_ms,
            deterministic_hash=request_hash,
            simulated_complexity=complexity,
            error_kind=VerificationErrorKind.RESOURCE_LIMIT,
            resource_budget_applied=budget_dict,
        )
    
    # Deterministic success/failure based on hash
    # ~70% success rate for simple formulas (branch_value < 70)
    verified = branch_value < 70
    
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    
    if verified:
        return LeanVerificationResult(
            verified=True,
            abstention_reason=None,
            method="lean_adapter_simulate",
            lean_version_checked=LEAN_VERSION_REQUIRED,
            duration_ms=duration_ms,
            deterministic_hash=request_hash,
            simulated_complexity=complexity,
            error_kind=VerificationErrorKind.NONE,
            resource_budget_applied=budget_dict,
        )
    else:
        # Simulated failure — tag as SIMULATION_ONLY since it's not a real Lean error
        return LeanVerificationResult(
            verified=False,
            abstention_reason=LeanAbstentionReason.LEAN_ERROR,
            method="lean_adapter_simulate",
            lean_version_checked=LEAN_VERSION_REQUIRED,
            duration_ms=duration_ms,
            deterministic_hash=request_hash,
            simulated_complexity=complexity,
            error_kind=VerificationErrorKind.SIMULATION_ONLY,
            resource_budget_applied=budget_dict,
        )


def compute_formula_complexity(canonical: str) -> int:
    """
    Compute complexity score for a canonical formula.
    
    Currently uses length as a proxy for complexity.
    Future versions may analyze AST depth, operator count, etc.
    
    Args:
        canonical: Canonical ASCII form of the statement.
    
    Returns:
        Integer complexity score.
    """
    return len(canonical)


# =============================================================================
# SHADOW MODE TELEMETRY GENERATION
# =============================================================================

# Hash salt for shadow telemetry determinism
_SHADOW_HASH_SALT = b"mathledger_lean_shadow_v1"


def generate_shadow_telemetry(
    canonical: str,
    job_id: str,
    verified: bool,
    complexity: int,
) -> Dict[str, Any]:
    """
    Generate synthetic subprocess-like telemetry for shadow mode.
    
    This produces deterministic telemetry that mimics real Lean subprocess
    output: return codes, stderr messages, CPU/memory footprints.
    
    Args:
        canonical: Canonical formula being verified.
        job_id: Job identifier.
        verified: Whether verification succeeded.
        complexity: Formula complexity score.
    
    Returns:
        Dictionary with shadow telemetry:
        - return_code: int (0 for success, non-zero for failure)
        - stderr: str (synthetic error messages)
        - cpu_time_ms: int (deterministic CPU time)
        - memory_mb: int (deterministic memory footprint)
        - stdout_lines: List[str] (synthetic stdout)
    """
    # Deterministic hash for branching
    hash_input = f"{canonical}:{job_id}:shadow".encode("utf-8")
    full_hash = hashlib.sha256(_SHADOW_HASH_SALT + hash_input).hexdigest()
    branch_value = int(full_hash[:8], 16) % 1000
    
    # Generate return code (0 for success, deterministic non-zero for failure)
    if verified:
        return_code = 0
        stderr = ""
        stdout_lines = [
            f"Building {job_id}...",
            "Build succeeded.",
            f"Verification completed in {complexity % 100}ms",
        ]
    else:
        # Deterministic error codes based on hash
        if branch_value < 300:
            return_code = 1  # Generic error
            stderr = f"error: type mismatch in {job_id}\n  {canonical[:50]}"
        elif branch_value < 600:
            return_code = 2  # Timeout
            stderr = f"error: verification timeout after {complexity * 10}ms"
        else:
            return_code = 3  # Resource exceeded
            stderr = f"error: memory limit exceeded (requested {complexity * 2}MB)"
        
        stdout_lines = [
            f"Building {job_id}...",
            "Build failed.",
        ]
    
    # Deterministic CPU time (based on complexity and hash)
    cpu_time_ms = (complexity * 10) + (branch_value % 100)
    
    # Deterministic memory footprint (based on complexity)
    memory_mb = max(100, (complexity * 2) + (branch_value % 50))
    
    return {
        "return_code": return_code,
        "stderr": stderr,
        "cpu_time_ms": cpu_time_ms,
        "memory_mb": memory_mb,
        "stdout_lines": stdout_lines,
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LeanAdapterConfig:
    """
    Configuration for the Lean adapter.
    
    Attributes:
        mode: Operational mode (DISABLED, PHASE_IIB_SCAFFOLD, SIMULATE, ACTIVE).
        lean_version_required: Required Lean version for compatibility.
        default_budget: Default resource budget for verifications.
    """
    
    mode: LeanAdapterMode = LeanAdapterMode.PHASE_IIB_SCAFFOLD
    """Current operational mode."""
    
    lean_version_required: str = LEAN_VERSION_REQUIRED
    """Required Lean version (pinned)."""
    
    default_budget: LeanResourceBudget = field(
        default_factory=LeanResourceBudget
    )
    """Default resource budget for verification jobs."""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for logging."""
        return {
            "mode": self.mode.value,
            "lean_version_required": self.lean_version_required,
            "default_budget": self.default_budget.to_dict(),
        }


# =============================================================================
# MAIN ADAPTER CLASS
# =============================================================================

class LeanAdapter:
    """
    Phase IIb Lean Verification Adapter.
    
    ===========================================================================
    STATUS: SCAFFOLD + SIMULATION — DOES NOT INVOKE LEAN
    
    This class provides the interface for Lean verification that will be
    activated in Phase IIb. Currently supports scaffold and simulation modes.
    ===========================================================================
    
    The adapter provides:
    - Deterministic interface for verification requests
    - Resource budget validation (stubs in scaffold mode)
    - Version pinning with validation stubs
    - Safe, structured failure modes
    - Simulation mode for dry-run testing
    
    Usage (Scaffold Mode):
        adapter = LeanAdapter()
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="abc123",
        )
        result = adapter.verify(request)
        assert result.verified == False
        assert result.abstention_reason == LeanAbstentionReason.LEAN_DISABLED
    
    Usage (Simulation Mode):
        adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(
            canonical="p->p",
            job_id="abc123",
        )
        result = adapter.verify(request)
        # Result is deterministic based on hash(canonical)
        # May be: verified=True, or abstention with TIMEOUT/RESOURCE_EXCEEDED
    
    Phase IIb Integration Points:
        - backend/worker.py::execute_lean_job() — primary hook
        - backend/lean_mode.py::get_build_runner() — runner injection
        - derivation/verification.py::LeanFallback — fallback replacement
    
    TODO(PHASE_IIB): Implement actual Lean subprocess invocation
    TODO(PHASE_IIB): Add memory monitoring via psutil or resource module
    TODO(PHASE_IIB): Integrate with LeanSandbox for isolation
    TODO(PHASE_IIB): Wire into worker.py::execute_lean_job()
    TODO(PHASE_IIB): Add metrics emission for Lean verification latency
    TODO(PHASE_IIB): Implement version validation via `lake --version`
    """
    
    # Class-level constant for version pinning
    LEAN_VERSION_REQUIRED = LEAN_VERSION_REQUIRED
    
    def __init__(
        self,
        mode: LeanAdapterMode = LeanAdapterMode.PHASE_IIB_SCAFFOLD,
        config: Optional[LeanAdapterConfig] = None,
    ) -> None:
        """
        Initialize the Lean adapter.
        
        Args:
            mode: Operational mode. Defaults to PHASE_IIB_SCAFFOLD.
            config: Optional configuration override.
        """
        self._mode = mode
        self._config = config or LeanAdapterConfig(mode=mode)
        
        # Statistics tracking (for future metrics)
        self._verification_count = 0
        self._total_duration_ms = 0
        self._simulation_count = 0
        self._simulation_success_count = 0
    
    @property
    def mode(self) -> LeanAdapterMode:
        """Current operational mode."""
        return self._mode
    
    @property
    def config(self) -> LeanAdapterConfig:
        """Current configuration."""
        return self._config
    
    def verify(self, request: LeanVerificationRequest) -> LeanVerificationResult:
        """
        Verify a statement using Lean.
        
        Behavior depends on mode:
        - DISABLED: Immediate abstention with LEAN_DISABLED
        - PHASE_IIB_SCAFFOLD: Deterministic abstention with version info
        - SIMULATE: Deterministic result based on formula complexity and hash
        - ACTIVE: Real Lean verification (NOT IMPLEMENTED)
        
        Args:
            request: The verification request containing the statement
                     and resource budget. Must pass validation.
        
        Returns:
            LeanVerificationResult with verification outcome.
        
        Raises:
            LeanAdapterValidationError: If request fails validation.
        
        Note:
            In Phase IIb ACTIVE mode, this method will:
            1. Validate Lean version
            2. Check resource budget
            3. Invoke Lean via subprocess
            4. Enforce timeout and memory limits
            5. Return actual verification result
        """
        start_time = time.perf_counter()
        
        # Request validation happens in __post_init__, but we re-validate
        # here in case someone bypasses the dataclass constructor
        errors = validate_verification_request(request)
        if errors:
            raise LeanAdapterValidationError(
                f"Invalid LeanVerificationRequest: {'; '.join(errors)}"
            )
        
        # Compute deterministic hash for this request
        request_hash = request.deterministic_hash()
        
        # Track statistics
        self._verification_count += 1
        
        # ------------------------------------------------------------------
        # Mode-specific handling
        # ------------------------------------------------------------------
        
        # Echo budget in result metadata
        budget_dict = request.resource_budget.to_dict()
        
        if self._mode == LeanAdapterMode.DISABLED:
            # Immediate abstention in disabled mode
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._total_duration_ms += duration_ms
            return LeanVerificationResult(
                verified=False,
                abstention_reason=LeanAbstentionReason.LEAN_DISABLED,
                method="lean_adapter_disabled",
                lean_version_checked=None,
                duration_ms=duration_ms,
                deterministic_hash=request_hash,
                error_kind=VerificationErrorKind.SIMULATION_ONLY,
                resource_budget_applied=budget_dict,
            )
        
        if self._mode == LeanAdapterMode.PHASE_IIB_SCAFFOLD:
            # Scaffold mode: deterministic abstention without Lean
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._total_duration_ms += duration_ms
            return LeanVerificationResult(
                verified=False,
                abstention_reason=LeanAbstentionReason.LEAN_DISABLED,
                method="lean_adapter_scaffold",
                lean_version_checked=self.LEAN_VERSION_REQUIRED,
                duration_ms=duration_ms,
                deterministic_hash=request_hash,
                error_kind=VerificationErrorKind.SIMULATION_ONLY,
                resource_budget_applied=budget_dict,
            )
        
        if self._mode == LeanAdapterMode.SIMULATE:
            # Simulation mode: deterministic results based on formula complexity
            result = simulate_lean_result(
                request.canonical,
                request.job_id,
                request.resource_budget,
            )
            
            # Update statistics
            self._simulation_count += 1
            if result.verified:
                self._simulation_success_count += 1
            self._total_duration_ms += result.duration_ms
            
            return result
        
        if self._mode == LeanAdapterMode.SHADOW:
            # Shadow mode: simulation with subprocess-like telemetry
            # First, get simulation result
            result = simulate_lean_result(
                request.canonical,
                request.job_id,
                request.resource_budget,
            )
            
            # Generate shadow telemetry
            complexity = compute_formula_complexity(request.canonical)
            shadow_telemetry = generate_shadow_telemetry(
                request.canonical,
                request.job_id,
                result.verified,
                complexity,
            )
            
            # Store shadow telemetry in result metadata (via method field extension)
            # Note: We can't modify LeanVerificationResult, so we'll store it separately
            # For now, we'll use the method field to indicate shadow mode
            # In Phase IIb, we can extend LeanVerificationResult with a telemetry field
            
            # Update statistics
            self._simulation_count += 1
            if result.verified:
                self._simulation_success_count += 1
            self._total_duration_ms += result.duration_ms
            
            # Return result with shadow method indicator
            return LeanVerificationResult(
                verified=result.verified,
                abstention_reason=result.abstention_reason,
                method="lean_adapter_shadow",
                lean_version_checked=result.lean_version_checked,
                duration_ms=result.duration_ms,
                deterministic_hash=result.deterministic_hash,
                simulated_complexity=result.simulated_complexity,
                error_kind=result.error_kind,
                resource_budget_applied=result.resource_budget_applied,
            )
        
        if self._mode == LeanAdapterMode.ACTIVE:
            # ----------------------------------------------------------
            # TODO(PHASE_IIB): Implement actual Lean verification here
            # ----------------------------------------------------------
            # Steps for Phase IIb implementation:
            # 1. Validate Lean version via check_lean_availability()
            # 2. Validate resource budget
            # 3. Create isolated job directory (integrate with LeanSandbox)
            # 4. Generate Lean source file
            # 5. Invoke lake build with timeout
            # 6. Parse result and determine verification outcome
            # 7. Cleanup temporary files
            # 8. Return structured result
            # ----------------------------------------------------------
            raise NotImplementedError(
                "LeanAdapterMode.ACTIVE is not implemented in Phase II. "
                "This mode will be enabled in Phase IIb uplift experiments."
            )
        
        # Unreachable, but satisfy type checker
        raise ValueError(f"Unknown LeanAdapterMode: {self._mode}")
    
    def check_lean_availability(self) -> bool:
        """
        Check if Lean toolchain is available.
        
        PHASE IIB SCAFFOLD: Always returns False.
        
        Returns:
            True if Lean is installed and accessible.
        
        TODO(PHASE_IIB): Implement via subprocess.run(["lake", "--version"])
        """
        # SCAFFOLD: Do not check actual Lean availability
        return False
    
    def validate_lean_version(self) -> tuple[bool, Optional[str]]:
        """
        Validate that installed Lean version matches required version.
        
        PHASE IIB SCAFFOLD: Returns (False, None).
        
        Returns:
            Tuple of (is_valid, detected_version).
            is_valid is True if version matches LEAN_VERSION_REQUIRED.
            detected_version is the version string if Lean is available.
        
        TODO(PHASE_IIB): Implement version parsing from lake --version output
        """
        # SCAFFOLD: Do not validate actual version
        return (False, None)
    
    def get_statistics(self) -> dict:
        """
        Get adapter statistics for monitoring.
        
        Returns:
            Dictionary with verification_count, total_duration_ms, etc.
        """
        stats = {
            "mode": self._mode.value,
            "verification_count": self._verification_count,
            "total_duration_ms": self._total_duration_ms,
            "lean_version_required": self.LEAN_VERSION_REQUIRED,
        }
        
        # Include simulation stats if in simulate mode
        if self._mode == LeanAdapterMode.SIMULATE:
            stats["simulation_count"] = self._simulation_count
            stats["simulation_success_count"] = self._simulation_success_count
            if self._simulation_count > 0:
                stats["simulation_success_rate"] = (
                    self._simulation_success_count / self._simulation_count
                )
        
        return stats


# =============================================================================
# EVIDENCE PACK SUMMARY HELPER
# =============================================================================

def summarize_lean_activity(
    results: Sequence[LeanVerificationResult],
) -> Dict[str, Any]:
    """
    Summarize Lean verification activity for Evidence Packs.
    
    This helper produces a compact, stable summary of Lean simulation
    activity suitable for embedding in Evidence Pack metadata.
    
    PHASE IIb READINESS:
    The output shape is stable and forward-compatible. New fields may be
    added but existing fields will not change semantics.
    
    Args:
        results: Sequence of LeanVerificationResult objects to summarize.
    
    Returns:
        JSON-serializable dictionary with:
        - total_requests: Total number of verification requests
        - success_count: Number of verified=True results
        - abstention_count: Number of abstention results
        - error_kinds_histogram: Count of each VerificationErrorKind
        - version_pin: The required Lean version (LEAN_VERSION_REQUIRED)
        - methods_histogram: Count of each verification method
    
    Example:
        >>> from backend.verification import (
        ...     LeanAdapter, LeanAdapterMode, LeanVerificationRequest,
        ...     summarize_lean_activity
        ... )
        >>> adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        >>> results = [
        ...     adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"j{i}"))
        ...     for i in range(10)
        ... ]
        >>> summary = summarize_lean_activity(results)
        >>> summary["total_requests"]
        10
        >>> "error_kinds_histogram" in summary
        True
    """
    if not results:
        return {
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kinds_histogram": {},
            "methods_histogram": {},
            "version_pin": LEAN_VERSION_REQUIRED,
        }
    
    success_count = 0
    abstention_count = 0
    error_kinds_histogram: Dict[str, int] = {}
    methods_histogram: Dict[str, int] = {}
    
    for result in results:
        # Count successes
        if result.verified:
            success_count += 1
        
        # Count abstentions
        if result.is_abstention:
            abstention_count += 1
        
        # Build error_kind histogram
        error_kind_value = result.error_kind.value
        error_kinds_histogram[error_kind_value] = (
            error_kinds_histogram.get(error_kind_value, 0) + 1
        )
        
        # Build methods histogram
        methods_histogram[result.method] = (
            methods_histogram.get(result.method, 0) + 1
        )
    
    return {
        "total_requests": len(results),
        "success_count": success_count,
        "abstention_count": abstention_count,
        "error_kinds_histogram": error_kinds_histogram,
        "methods_histogram": methods_histogram,
        "version_pin": LEAN_VERSION_REQUIRED,
    }


# =============================================================================
# ACTIVITY LEDGER (TASK 1)
# =============================================================================

# Schema version for ledger format (bump on breaking changes)
_LEDGER_SCHEMA_VERSION = "1.0.0"


def build_lean_activity_ledger(
    results: Sequence[LeanVerificationResult],
) -> Dict[str, Any]:
    """
    Build a comprehensive Lean activity ledger from verification results.
    
    This is a richer, durable object on top of summarize_lean_activity,
    designed for long-term storage and analysis. It includes resource
    budget statistics and max observed values.
    
    PHASE IIb READINESS:
    The ledger schema is versioned for forward compatibility. Breaking changes
    will increment the schema_version field.
    
    Args:
        results: Sequence of LeanVerificationResult objects.
    
    Returns:
        JSON-serializable dictionary with:
        - schema_version: Ledger format version (e.g., "1.0.0")
        - total_requests: Total verification requests
        - success_count: Successful verifications
        - abstention_count: Abstention results
        - error_kind_histogram: Count per VerificationErrorKind
        - resource_budget_histogram: Count per budget archetype
        - max_resource_budget_observed: Maximum values per budget field
        - methods_histogram: Count per verification method
        - version_pin: Required Lean version
        - timestamp_utc: ISO timestamp when ledger was built
    
    Example:
        >>> from backend.verification import (
        ...     LeanAdapter, LeanAdapterMode, LeanVerificationRequest,
        ...     build_lean_activity_ledger
        ... )
        >>> adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        >>> results = [
        ...     adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"j{i}"))
        ...     for i in range(10)
        ... ]
        >>> ledger = build_lean_activity_ledger(results)
        >>> ledger["schema_version"]
        '1.0.0'
    """
    import datetime
    
    if not results:
        return {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "total_requests": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {
                "timeout_seconds": 0,
                "memory_mb": 0,
                "disk_mb": 0,
                "max_proofs": 0,
            },
            "methods_histogram": {},
            "version_pin": LEAN_VERSION_REQUIRED,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
    
    success_count = 0
    abstention_count = 0
    error_kind_histogram: Dict[str, int] = {}
    methods_histogram: Dict[str, int] = {}
    resource_budget_histogram: Dict[str, int] = {}
    
    # Track max resource budget values
    max_timeout = 0
    max_memory = 0
    max_disk = 0
    max_proofs = 0
    
    for result in results:
        # Count successes
        if result.verified:
            success_count += 1
        
        # Count abstentions
        if result.is_abstention:
            abstention_count += 1
        
        # Build error_kind histogram
        error_kind_value = result.error_kind.value
        error_kind_histogram[error_kind_value] = (
            error_kind_histogram.get(error_kind_value, 0) + 1
        )
        
        # Build methods histogram
        methods_histogram[result.method] = (
            methods_histogram.get(result.method, 0) + 1
        )
        
        # Track resource budget statistics
        if result.resource_budget_applied:
            budget = result.resource_budget_applied
            
            # Determine budget archetype for histogram
            archetype = _classify_budget_archetype(budget)
            resource_budget_histogram[archetype] = (
                resource_budget_histogram.get(archetype, 0) + 1
            )
            
            # Track max values
            max_timeout = max(max_timeout, budget.get("timeout_seconds", 0))
            max_memory = max(max_memory, budget.get("memory_mb", 0))
            max_disk = max(max_disk, budget.get("disk_mb", 0))
            max_proofs = max(max_proofs, budget.get("max_proofs", 0))
    
    return {
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "total_requests": len(results),
        "success_count": success_count,
        "abstention_count": abstention_count,
        "error_kind_histogram": error_kind_histogram,
        "resource_budget_histogram": resource_budget_histogram,
        "max_resource_budget_observed": {
            "timeout_seconds": max_timeout,
            "memory_mb": max_memory,
            "disk_mb": max_disk,
            "max_proofs": max_proofs,
        },
        "methods_histogram": methods_histogram,
        "version_pin": LEAN_VERSION_REQUIRED,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }


def _classify_budget_archetype(budget: Dict[str, Any]) -> str:
    """
    Classify a resource budget into an archetype for histogram tracking.
    
    Archetypes:
    - "minimal": timeout <= 10, memory <= 1024
    - "default": standard defaults
    - "generous": timeout >= 45 or memory >= 3000
    - "custom": anything else
    
    Args:
        budget: Resource budget dictionary.
    
    Returns:
        Archetype string.
    """
    timeout = budget.get("timeout_seconds", 30)
    memory = budget.get("memory_mb", 2048)
    
    # Check for minimal
    if timeout <= 10 and memory <= 1024:
        return "minimal"
    
    # Check for generous
    if timeout >= 45 or memory >= 3000:
        return "generous"
    
    # Check for default (within 20% of defaults)
    if 24 <= timeout <= 36 and 1638 <= memory <= 2458:
        return "default"
    
    return "custom"


# =============================================================================
# SAFETY ENVELOPE (TASK 2)
# =============================================================================

def evaluate_lean_adapter_safety(
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate safety envelope for Lean adapter usage/promotion.
    
    This function analyzes a Lean activity ledger and determines whether
    the adapter's behavior is safe for continued use or promotion to
    higher trust levels.
    
    PHASE IIb READINESS:
    This remains simulation-only but defines how we'd gate Lean usage later.
    The safety status will be used in Phase IIb to control real Lean invocation.
    
    Safety Rules:
    - BLOCK: Any INTERNAL_ERROR or LEAN_UNAVAILABLE errors
    - WARN: Any RESOURCE_LIMIT errors (timeouts, memory issues)
    - OK: All other cases
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
    
    Returns:
        JSON-serializable dictionary with:
        - has_internal_errors: True if INTERNAL_ERROR or LEAN_UNAVAILABLE > 0
        - has_resource_issues: True if RESOURCE_LIMIT > 0
        - safety_status: "OK" | "WARN" | "BLOCK"
        - reasons: List of short descriptive strings explaining status
        - internal_error_count: Count of internal errors
        - resource_limit_count: Count of resource limit errors
    
    Example:
        >>> ledger = build_lean_activity_ledger(results)
        >>> safety = evaluate_lean_adapter_safety(ledger)
        >>> safety["safety_status"]
        'OK'
    """
    error_histogram = ledger.get("error_kind_histogram", {})
    
    # Count problematic error kinds
    internal_error_count = (
        error_histogram.get(VerificationErrorKind.INTERNAL_ERROR.value, 0)
    )
    lean_unavailable_count = (
        error_histogram.get(VerificationErrorKind.LEAN_UNAVAILABLE.value, 0)
    )
    resource_limit_count = (
        error_histogram.get(VerificationErrorKind.RESOURCE_LIMIT.value, 0)
    )
    
    # Determine flags
    has_internal_errors = (internal_error_count + lean_unavailable_count) > 0
    has_resource_issues = resource_limit_count > 0
    
    # Build reasons list
    reasons: list[str] = []
    
    if internal_error_count > 0:
        reasons.append(f"internal_error: {internal_error_count} occurrences")
    
    if lean_unavailable_count > 0:
        reasons.append(f"lean_unavailable: {lean_unavailable_count} occurrences")
    
    if resource_limit_count > 0:
        reasons.append(f"resource_limit: {resource_limit_count} occurrences")
    
    # Determine safety status
    if has_internal_errors:
        safety_status = "BLOCK"
        if not reasons:
            reasons.append("internal errors detected")
    elif has_resource_issues:
        safety_status = "WARN"
        if not reasons:
            reasons.append("resource limit issues detected")
    else:
        safety_status = "OK"
        reasons.append("no safety concerns")
    
    return {
        "has_internal_errors": has_internal_errors,
        "has_resource_issues": has_resource_issues,
        "safety_status": safety_status,
        "reasons": reasons,
        "internal_error_count": internal_error_count + lean_unavailable_count,
        "resource_limit_count": resource_limit_count,
    }


# =============================================================================
# GLOBAL HEALTH / MAAS SIGNAL (TASK 3)
# =============================================================================

def summarize_lean_for_global_health(
    ledger: Dict[str, Any],
    safety_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize Lean adapter status for global health / MAAS monitoring.
    
    This produces a compact signal suitable for inclusion in system-wide
    health checks and monitoring dashboards.
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
        safety_eval: Safety evaluation from evaluate_lean_adapter_safety().
    
    Returns:
        JSON-serializable dictionary with:
        - lean_surface_ok: True if status is OK
        - internal_error_count: Count of internal errors
        - resource_limit_count: Count of resource limit errors
        - status: "OK" | "WARN" | "BLOCK"
        - total_requests: Total verification requests processed
        - success_rate: Ratio of successful verifications (0.0-1.0)
        - version_pin: Required Lean version
    
    Example:
        >>> ledger = build_lean_activity_ledger(results)
        >>> safety = evaluate_lean_adapter_safety(ledger)
        >>> health = summarize_lean_for_global_health(ledger, safety)
        >>> health["lean_surface_ok"]
        True
    """
    status = safety_eval.get("safety_status", "BLOCK")
    total_requests = ledger.get("total_requests", 0)
    success_count = ledger.get("success_count", 0)
    
    # Compute success rate
    if total_requests > 0:
        success_rate = success_count / total_requests
    else:
        success_rate = 0.0
    
    return {
        "lean_surface_ok": status == "OK",
        "internal_error_count": safety_eval.get("internal_error_count", 0),
        "resource_limit_count": safety_eval.get("resource_limit_count", 0),
        "status": status,
        "total_requests": total_requests,
        "success_rate": round(success_rate, 4),
        "version_pin": ledger.get("version_pin", LEAN_VERSION_REQUIRED),
    }


# =============================================================================
# PHASE IV: CAPABILITY CLASSIFICATION (TASK 1)
# =============================================================================

# Schema version for capability format
_CAPABILITY_SCHEMA_VERSION = "1.0.0"


def classify_lean_capabilities(
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify Lean adapter capabilities based on activity ledger.
    
    This function analyzes the ledger to determine the capability band
    (BASIC, INTERMEDIATE, ADVANCED) based on success rates, resource
    budgets used, and presence of resource limit issues.
    
    PHASE IV READINESS:
    This provides a capability radar for understanding what level of
    Lean integration the system is ready for.
    
    Capability Bands:
    - BASIC: Low success rate (< 0.5) OR frequent resource limits
    - INTERMEDIATE: Moderate success rate (0.5-0.8) with some resource limits
    - ADVANCED: High success rate (>= 0.8) with minimal resource limits
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
    
    Returns:
        JSON-serializable dictionary with:
        - schema_version: Capability format version
        - capability_band: "BASIC" | "INTERMEDIATE" | "ADVANCED"
        - max_budget_used: Maximum resource budget values observed
        - resource_profile: Human-readable description of typical budgets
        - simulation_only: True if all results are from simulation
    
    Example:
        >>> ledger = build_lean_activity_ledger(results)
        >>> capability = classify_lean_capabilities(ledger)
        >>> capability["capability_band"]
        'INTERMEDIATE'
    """
    total_requests = ledger.get("total_requests", 0)
    success_count = ledger.get("success_count", 0)
    error_histogram = ledger.get("error_kind_histogram", {})
    max_budget = ledger.get("max_resource_budget_observed", {})
    methods_histogram = ledger.get("methods_histogram", {})
    
    # Compute success rate
    if total_requests > 0:
        success_rate = success_count / total_requests
    else:
        success_rate = 0.0
    
    # Check for resource limit issues
    resource_limit_count = error_histogram.get(
        VerificationErrorKind.RESOURCE_LIMIT.value, 0
    )
    has_resource_limits = resource_limit_count > 0
    
    # Determine if simulation-only
    # If all methods are simulation-related, we're in simulation mode
    simulation_methods = {
        "lean_adapter_simulate",
        "lean_adapter_scaffold",
        "lean_adapter_disabled",
    }
    all_methods = set(methods_histogram.keys())
    simulation_only = (
        len(all_methods) > 0 and all_methods.issubset(simulation_methods)
    ) or total_requests == 0
    
    # Determine capability band
    if total_requests == 0:
        capability_band = "BASIC"
    elif success_rate < 0.5 or (has_resource_limits and success_rate < 0.6):
        capability_band = "BASIC"
    elif success_rate >= 0.8 and not has_resource_limits:
        capability_band = "ADVANCED"
    else:
        capability_band = "INTERMEDIATE"
    
    # Build resource profile description
    budget_histogram = ledger.get("resource_budget_histogram", {})
    if budget_histogram:
        dominant_archetype = max(
            budget_histogram.items(), key=lambda x: x[1], default=("unknown", 0)
        )[0]
        resource_profile = (
            f"Dominant budget archetype: {dominant_archetype}. "
            f"Max observed: {max_budget.get('timeout_seconds', 0)}s timeout, "
            f"{max_budget.get('memory_mb', 0)}MB memory."
        )
    else:
        resource_profile = "No resource budget data available."
    
    return {
        "schema_version": _CAPABILITY_SCHEMA_VERSION,
        "capability_band": capability_band,
        "max_budget_used": max_budget,
        "resource_profile": resource_profile,
        "simulation_only": simulation_only,
    }


# =============================================================================
# PHASE IV: MIGRATION CHECKLIST (TASK 2)
# =============================================================================

def build_lean_migration_checklist(
    ledger: Dict[str, Any],
    safety_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a migration checklist for transitioning from simulation to live Lean.
    
    This is a pure helper function designed for human review. It does NOT
    flip any mode or enable live Lean automatically. It provides a structured
    checklist of requirements that must be met before enabling live mode.
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
        safety_eval: Safety evaluation from evaluate_lean_adapter_safety().
    
    Returns:
        JSON-serializable dictionary with:
        - can_enable_live_mode: True if all blocking checks pass
        - blocking_reasons: List of reasons why live mode cannot be enabled
        - checklist_items: List of checklist items with id, description, status
    
    Example:
        >>> ledger = build_lean_activity_ledger(results)
        >>> safety = evaluate_lean_adapter_safety(ledger)
        >>> checklist = build_lean_migration_checklist(ledger, safety)
        >>> checklist["can_enable_live_mode"]
        False
    """
    blocking_reasons: list[str] = []
    checklist_items: list[Dict[str, Any]] = []
    
    # Check 1: Safety status must be OK
    safety_status = safety_eval.get("safety_status", "BLOCK")
    if safety_status != "OK":
        blocking_reasons.append(f"Safety status is {safety_status}, must be OK")
        checklist_items.append({
            "id": "safety_status_ok",
            "description": "Safety evaluation must return OK status",
            "status": "FAIL",
        })
    else:
        checklist_items.append({
            "id": "safety_status_ok",
            "description": "Safety evaluation must return OK status",
            "status": "PASS",
        })
    
    # Check 2: No internal errors
    internal_error_count = safety_eval.get("internal_error_count", 0)
    if internal_error_count > 0:
        blocking_reasons.append(
            f"Found {internal_error_count} internal error(s)"
        )
        checklist_items.append({
            "id": "no_internal_errors",
            "description": "No INTERNAL_ERROR or LEAN_UNAVAILABLE errors",
            "status": "FAIL",
        })
    else:
        checklist_items.append({
            "id": "no_internal_errors",
            "description": "No INTERNAL_ERROR or LEAN_UNAVAILABLE errors",
            "status": "PASS",
        })
    
    # Check 3: Minimum request volume (for statistical confidence)
    total_requests = ledger.get("total_requests", 0)
    if total_requests < 10:
        blocking_reasons.append(
            f"Insufficient request volume: {total_requests} < 10"
        )
        checklist_items.append({
            "id": "minimum_volume",
            "description": "At least 10 verification requests for statistical confidence",
            "status": "FAIL",
        })
    else:
        checklist_items.append({
            "id": "minimum_volume",
            "description": "At least 10 verification requests for statistical confidence",
            "status": "PASS",
        })
    
    # Check 4: Success rate threshold
    success_count = ledger.get("success_count", 0)
    if total_requests > 0:
        success_rate = success_count / total_requests
        if success_rate < 0.5:
            blocking_reasons.append(
                f"Success rate too low: {success_rate:.2%} < 50%"
            )
            checklist_items.append({
                "id": "success_rate_threshold",
                "description": "Success rate must be at least 50%",
                "status": "FAIL",
            })
        else:
            checklist_items.append({
                "id": "success_rate_threshold",
                "description": "Success rate must be at least 50%",
                "status": "PASS",
            })
    else:
        blocking_reasons.append("No requests to compute success rate")
        checklist_items.append({
            "id": "success_rate_threshold",
            "description": "Success rate must be at least 50%",
            "status": "FAIL",
        })
    
    # Check 5: Resource limits acceptable
    resource_limit_count = safety_eval.get("resource_limit_count", 0)
    if resource_limit_count > 0:
        # WARN but not blocking if < 10% of requests
        if total_requests > 0:
            resource_limit_rate = resource_limit_count / total_requests
            if resource_limit_rate > 0.1:
                blocking_reasons.append(
                    f"Resource limit rate too high: {resource_limit_rate:.2%} > 10%"
                )
                checklist_items.append({
                    "id": "resource_limits_acceptable",
                    "description": "Resource limit errors must be < 10% of requests",
                    "status": "FAIL",
                })
            else:
                checklist_items.append({
                    "id": "resource_limits_acceptable",
                    "description": "Resource limit errors must be < 10% of requests",
                    "status": "PASS",
                })
        else:
            checklist_items.append({
                "id": "resource_limits_acceptable",
                "description": "Resource limit errors must be < 10% of requests",
                "status": "PASS",  # No data, assume pass
            })
    else:
        checklist_items.append({
            "id": "resource_limits_acceptable",
            "description": "Resource limit errors must be < 10% of requests",
            "status": "PASS",
        })
    
    # Check 6: Version pin present
    version_pin = ledger.get("version_pin")
    if not version_pin or version_pin != LEAN_VERSION_REQUIRED:
        blocking_reasons.append(
            f"Version pin mismatch or missing: {version_pin}"
        )
        checklist_items.append({
            "id": "version_pin_correct",
            "description": f"Version pin must be {LEAN_VERSION_REQUIRED}",
            "status": "FAIL",
        })
    else:
        checklist_items.append({
            "id": "version_pin_correct",
            "description": f"Version pin must be {LEAN_VERSION_REQUIRED}",
            "status": "PASS",
        })
    
    # Determine if live mode can be enabled
    can_enable_live_mode = len(blocking_reasons) == 0
    
    return {
        "can_enable_live_mode": can_enable_live_mode,
        "blocking_reasons": blocking_reasons,
        "checklist_items": checklist_items,
    }


# =============================================================================
# PHASE IV: DIRECTOR PANEL (TASK 3)
# =============================================================================

def build_lean_director_panel(
    ledger: Dict[str, Any],
    safety_eval: Dict[str, Any],
    capability: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director-level panel summarizing Lean adapter status.
    
    This provides a high-level, executive-friendly view of the Lean
    adapter's current state, safety, and capabilities. The headline
    is neutral and factual, avoiding value judgments.
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
        safety_eval: Safety evaluation from evaluate_lean_adapter_safety().
        capability: Capability classification from classify_lean_capabilities().
    
    Returns:
        JSON-serializable dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - lean_surface_ok: True if status is OK
        - capability_band: "BASIC" | "INTERMEDIATE" | "ADVANCED"
        - safety_status: "OK" | "WARN" | "BLOCK"
        - headline: Short neutral summary
        - internal_error_count: Count of internal errors
        - resource_limit_count: Count of resource limit errors
    
    Example:
        >>> ledger = build_lean_activity_ledger(results)
        >>> safety = evaluate_lean_adapter_safety(ledger)
        >>> capability = classify_lean_capabilities(ledger)
        >>> panel = build_lean_director_panel(ledger, safety, capability)
        >>> panel["status_light"]
        'GREEN'
    """
    safety_status = safety_eval.get("safety_status", "BLOCK")
    capability_band = capability.get("capability_band", "BASIC")
    internal_error_count = safety_eval.get("internal_error_count", 0)
    resource_limit_count = safety_eval.get("resource_limit_count", 0)
    total_requests = ledger.get("total_requests", 0)
    success_count = ledger.get("success_count", 0)
    simulation_only = capability.get("simulation_only", True)
    
    # Determine status light
    if safety_status == "BLOCK" or internal_error_count > 0:
        status_light = "RED"
    elif safety_status == "WARN" or resource_limit_count > 0:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Build neutral headline
    if total_requests == 0:
        headline = "No Lean verification activity recorded."
    elif simulation_only:
        headline = (
            f"Simulation mode: {total_requests} requests processed, "
            f"{success_count} successful. Capability: {capability_band}."
        )
    else:
        headline = (
            f"Live mode: {total_requests} requests processed, "
            f"{success_count} successful. Capability: {capability_band}."
        )
    
    # Add safety context to headline if needed
    if safety_status != "OK":
        headline += f" Safety status: {safety_status}."
    
    lean_surface_ok = safety_status == "OK"
    
    return {
        "status_light": status_light,
        "lean_surface_ok": lean_surface_ok,
        "capability_band": capability_band,
        "safety_status": safety_status,
        "headline": headline,
        "internal_error_count": internal_error_count,
        "resource_limit_count": resource_limit_count,
    }


# =============================================================================
# SHADOW MODE: CAPABILITY RADAR
# =============================================================================

# Schema version for shadow radar
_SHADOW_RADAR_SCHEMA_VERSION = "1.0.0"


def build_lean_shadow_capability_radar(
    results: Sequence[LeanVerificationResult],
) -> Dict[str, Any]:
    """
    Build capability radar for shadow mode results.
    
    This analyzes shadow mode telemetry to produce structural insights:
    - Structural error rate (non-resource errors)
    - Complexity success curve (success rate by complexity bands)
    - Shadow resource band (LOW/MEDIUM/HIGH based on resource usage)
    - Anomaly signatures (hash-based clustering of error patterns)
    
    Args:
        results: Sequence of LeanVerificationResult from shadow mode.
    
    Returns:
        JSON-serializable dictionary with:
        - schema_version: Radar format version
        - structural_error_rate: Ratio of structural errors (0.0-1.0)
        - complexity_success_curve: Dict mapping complexity bands to success rates
        - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
        - anomaly_signatures: List of anomaly pattern hashes
        - total_shadow_requests: Count of shadow mode requests
    
    Example:
        >>> results = [adapter.verify(request) for request in requests]
        >>> radar = build_lean_shadow_capability_radar(results)
        >>> radar["shadow_resource_band"]
        'MEDIUM'
    """
    if not results:
        return {
            "schema_version": _SHADOW_RADAR_SCHEMA_VERSION,
            "structural_error_rate": 0.0,
            "complexity_success_curve": {},
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 0,
        }
    
    # Filter to shadow mode results
    shadow_results = [r for r in results if r.method == "lean_adapter_shadow"]
    
    if not shadow_results:
        return {
            "schema_version": _SHADOW_RADAR_SCHEMA_VERSION,
            "structural_error_rate": 0.0,
            "complexity_success_curve": {},
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 0,
        }
    
    # Compute structural error rate (errors that aren't resource limits)
    structural_errors = 0
    total_errors = 0
    
    for result in shadow_results:
        if not result.verified:
            total_errors += 1
            # Structural errors are non-resource-limit errors
            if result.error_kind != VerificationErrorKind.RESOURCE_LIMIT:
                structural_errors += 1
    
    if total_errors > 0:
        structural_error_rate = structural_errors / total_errors
    else:
        structural_error_rate = 0.0
    
    # Build complexity success curve
    complexity_bands = {
        "low": (0, 20),      # 0-20 chars
        "medium": (20, 50),  # 20-50 chars
        "high": (50, 100),   # 50-100 chars
        "very_high": (100, float('inf')),  # 100+ chars
    }
    
    complexity_success_curve: Dict[str, Dict[str, Any]] = {}
    
    for band_name, (min_complexity, max_complexity) in complexity_bands.items():
        band_results = [
            r for r in shadow_results
            if r.simulated_complexity is not None
            and min_complexity <= r.simulated_complexity < max_complexity
        ]
        
        if band_results:
            success_count = sum(1 for r in band_results if r.verified)
            success_rate = success_count / len(band_results)
            complexity_success_curve[band_name] = {
                "success_rate": round(success_rate, 3),
                "total_requests": len(band_results),
                "success_count": success_count,
            }
    
    # Determine shadow resource band based on average complexity and error rate
    avg_complexity = sum(
        r.simulated_complexity or 0 for r in shadow_results
    ) / len(shadow_results) if shadow_results else 0
    
    resource_limit_count = sum(
        1 for r in shadow_results
        if r.error_kind == VerificationErrorKind.RESOURCE_LIMIT
    )
    resource_limit_rate = resource_limit_count / len(shadow_results)
    
    # Resource band classification
    if avg_complexity < 30 and resource_limit_rate < 0.1:
        shadow_resource_band = "LOW"
    elif avg_complexity < 60 and resource_limit_rate < 0.3:
        shadow_resource_band = "MEDIUM"
    else:
        shadow_resource_band = "HIGH"
    
    # Generate anomaly signatures (hash-based clustering of error patterns)
    anomaly_signatures: list[str] = []
    error_patterns: Dict[str, int] = {}
    
    for result in shadow_results:
        if not result.verified and result.error_kind != VerificationErrorKind.NONE:
            # Create pattern hash from error characteristics
            pattern_key = f"{result.error_kind.value}:{result.abstention_reason.value if result.abstention_reason else 'none'}"
            error_patterns[pattern_key] = error_patterns.get(pattern_key, 0) + 1
    
    # Extract top anomaly patterns (those appearing >1 time)
    for pattern, count in error_patterns.items():
        if count > 1:
            # Create deterministic hash signature
            pattern_hash = hashlib.sha256(
                f"shadow_anomaly:{pattern}:{count}".encode("utf-8")
            ).hexdigest()[:16]
            anomaly_signatures.append(pattern_hash)
    
    # Sort for determinism
    anomaly_signatures.sort()
    
    return {
        "schema_version": _SHADOW_RADAR_SCHEMA_VERSION,
        "structural_error_rate": round(structural_error_rate, 3),
        "complexity_success_curve": complexity_success_curve,
        "shadow_resource_band": shadow_resource_band,
        "anomaly_signatures": anomaly_signatures,
        "total_shadow_requests": len(shadow_results),
    }


def build_lean_director_panel_with_shadow(
    ledger: Dict[str, Any],
    safety_eval: Dict[str, Any],
    capability: Dict[str, Any],
    shadow_radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build director panel with shadow mode extensions.
    
    This extends build_lean_director_panel() with shadow-specific fields:
    - shadow_mode_ok: True if shadow mode is operating normally
    - shadow_status: "OK" | "WARN" | "BLOCK"
    - dominant_anomalies: List of most common anomaly signatures
    - complexity_curve_summary: Human-readable summary of complexity curve
    - headline: Updated to include shadow status if applicable
    
    Args:
        ledger: Activity ledger from build_lean_activity_ledger().
        safety_eval: Safety evaluation from evaluate_lean_adapter_safety().
        capability: Capability classification from classify_lean_capabilities().
        shadow_radar: Optional shadow radar from build_lean_shadow_capability_radar().
    
    Returns:
        Extended director panel with shadow fields.
    """
    # Get base panel
    base_panel = build_lean_director_panel(ledger, safety_eval, capability)
    
    # If no shadow radar, return base panel with shadow fields set to defaults
    if shadow_radar is None:
        return {
            **base_panel,
            "shadow_mode_ok": True,
            "shadow_status": "OK",
            "dominant_anomalies": [],
            "complexity_curve_summary": "No shadow mode data available.",
        }
    
    # Extract shadow metrics
    structural_error_rate = shadow_radar.get("structural_error_rate", 0.0)
    shadow_resource_band = shadow_radar.get("shadow_resource_band", "LOW")
    anomaly_signatures = shadow_radar.get("anomaly_signatures", [])
    complexity_curve = shadow_radar.get("complexity_success_curve", {})
    
    # Determine shadow status
    if structural_error_rate > 0.5 or shadow_resource_band == "HIGH":
        shadow_status = "BLOCK"
        shadow_mode_ok = False
    elif structural_error_rate > 0.2 or shadow_resource_band == "MEDIUM":
        shadow_status = "WARN"
        shadow_mode_ok = True  # Still OK but needs attention
    else:
        shadow_status = "OK"
        shadow_mode_ok = True
    
    # Get dominant anomalies (top 3)
    dominant_anomalies = anomaly_signatures[:3]
    
    # Build complexity curve summary
    if complexity_curve:
        curve_parts = []
        for band, data in sorted(complexity_curve.items()):
            success_rate = data.get("success_rate", 0.0)
            total = data.get("total_requests", 0)
            curve_parts.append(f"{band}: {success_rate:.1%} ({total} requests)")
        complexity_curve_summary = "; ".join(curve_parts)
    else:
        complexity_curve_summary = "Insufficient data for complexity curve."
    
    # Update headline to include shadow status if not OK
    headline = base_panel.get("headline", "")
    if shadow_status != "OK":
        headline += f" Shadow mode: {shadow_status}."
    
    return {
        **base_panel,
        "shadow_mode_ok": shadow_mode_ok,
        "shadow_status": shadow_status,
        "dominant_anomalies": dominant_anomalies,
        "complexity_curve_summary": complexity_curve_summary,
        "headline": headline,
    }


# =============================================================================
# PHASE IV: LEAN MODE PLAYBOOK (Next Mission)
# =============================================================================

# Schema version for playbook format
_PLAYBOOK_SCHEMA_VERSION = "1.0.0"


def build_lean_mode_playbook(
    capability: Dict[str, Any],
    checklist: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an advisory playbook for Lean mode progression.
    
    This function provides recommendations for the next operational mode
    based on current capability and migration checklist status. It is
    purely advisory and NEVER auto-enables any mode.
    
    Mode Progression:
    - SIMULATION_ONLY: Current state (simulation mode)
    - SHADOW: Run live Lean alongside simulation, compare results
    - MIXED: Some requests use live Lean, some use simulation
    - LIVE: All requests use live Lean (simulation disabled)
    
    Args:
        capability: Capability classification from classify_lean_capabilities().
        checklist: Migration checklist from build_lean_migration_checklist().
    
    Returns:
        JSON-serializable dictionary with:
        - schema_version: Playbook format version
        - recommended_next_mode: "SIMULATION_ONLY" | "SHADOW" | "MIXED" | "LIVE"
        - prerequisites: List of prerequisite conditions that must be met
        - advisory_notes: List of advisory notes for human review
    
    Example:
        >>> capability = classify_lean_capabilities(ledger)
        >>> checklist = build_lean_migration_checklist(ledger, safety)
        >>> playbook = build_lean_mode_playbook(capability, checklist)
        >>> playbook["recommended_next_mode"]
        'SHADOW'
    """
    capability_band = capability.get("capability_band", "BASIC")
    can_enable_live_mode = checklist.get("can_enable_live_mode", False)
    blocking_reasons = checklist.get("blocking_reasons", [])
    simulation_only = capability.get("simulation_only", True)
    
    prerequisites: list[str] = []
    advisory_notes: list[str] = []
    
    # Determine recommended next mode based on capability and checklist
    if not simulation_only:
        # Already in live mode (shouldn't happen in Phase II, but handle gracefully)
        recommended_next_mode = "LIVE"
        advisory_notes.append("System is already operating in live mode.")
    elif capability_band == "BASIC":
        recommended_next_mode = "SIMULATION_ONLY"
        prerequisites.append("Capability band must be at least INTERMEDIATE")
        prerequisites.append("Success rate must be at least 50%")
        prerequisites.append("No internal errors (INTERNAL_ERROR or LEAN_UNAVAILABLE)")
        advisory_notes.append(
            "Current capability is BASIC. Continue simulation until "
            "success rate and stability improve."
        )
    elif capability_band == "INTERMEDIATE":
        if can_enable_live_mode:
            recommended_next_mode = "SHADOW"
            prerequisites.append("All migration checklist items must pass")
            prerequisites.append("Safety status must be OK")
            advisory_notes.append(
                "INTERMEDIATE capability with passing checklist. "
                "Recommend SHADOW mode to validate live Lean alongside simulation."
            )
        else:
            recommended_next_mode = "SIMULATION_ONLY"
            prerequisites.extend(blocking_reasons)
            advisory_notes.append(
                "INTERMEDIATE capability but migration checklist has blocking items. "
                "Address blocking issues before progressing to SHADOW mode."
            )
    elif capability_band == "ADVANCED":
        if can_enable_live_mode:
            recommended_next_mode = "SHADOW"
            prerequisites.append("All migration checklist items must pass")
            prerequisites.append("Safety status must be OK")
            advisory_notes.append(
                "ADVANCED capability with passing checklist. "
                "Recommend SHADOW mode first, then MIXED, then LIVE after validation period."
            )
            advisory_notes.append(
                "After successful SHADOW period (e.g., 1000 requests), "
                "consider MIXED mode with gradual rollout."
            )
        else:
            recommended_next_mode = "SIMULATION_ONLY"
            prerequisites.extend(blocking_reasons)
            advisory_notes.append(
                "ADVANCED capability but migration checklist has blocking items. "
                "Address blocking issues before progressing to SHADOW mode."
            )
    else:
        # Unknown capability band
        recommended_next_mode = "SIMULATION_ONLY"
        advisory_notes.append("Unknown capability band. Defaulting to SIMULATION_ONLY.")
    
    # Add general prerequisites
    if recommended_next_mode != "SIMULATION_ONLY":
        prerequisites.append("Lean version must match required version")
        prerequisites.append("Resource budgets must be validated")
        prerequisites.append("Monitoring and alerting must be in place")
    
    # Add advisory note about never auto-enabling
    advisory_notes.append(
        "This playbook is advisory only. Do not auto-enable modes. "
        "All mode changes require human review and approval."
    )
    
    return {
        "schema_version": _PLAYBOOK_SCHEMA_VERSION,
        "recommended_next_mode": recommended_next_mode,
        "prerequisites": prerequisites,
        "advisory_notes": advisory_notes,
    }


# =============================================================================
# PHASE IV: EVIDENCE PACK ADAPTER (Next Mission)
# =============================================================================

def summarize_lean_capabilities_for_evidence(
    capability: Dict[str, Any],
    safety_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize Lean capabilities as a compact tile for Evidence Packs.
    
    This produces a compact, JSON-serializable summary suitable for
    embedding in Evidence Pack metadata. The output is factual and
    avoids normative language (no "should", "must", etc.).
    
    Args:
        capability: Capability classification from classify_lean_capabilities().
        safety_eval: Safety evaluation from evaluate_lean_adapter_safety().
    
    Returns:
        JSON-serializable dictionary with:
        - capability_band: "BASIC" | "INTERMEDIATE" | "ADVANCED"
        - success_rate_proxy: Approximate success rate (0.0-1.0) based on capability band
        - resource_limit_count: Count of resource limit errors
        - simulation_only: True if all results are from simulation
        - safety_status: "OK" | "WARN" | "BLOCK"
    
    Example:
        >>> capability = classify_lean_capabilities(ledger)
        >>> safety = evaluate_lean_adapter_safety(ledger)
        >>> tile = summarize_lean_capabilities_for_evidence(capability, safety)
        >>> tile["capability_band"]
        'INTERMEDIATE'
    """
    capability_band = capability.get("capability_band", "BASIC")
    simulation_only = capability.get("simulation_only", True)
    safety_status = safety_eval.get("safety_status", "BLOCK")
    resource_limit_count = safety_eval.get("resource_limit_count", 0)
    
    # Map capability band to approximate success rate proxy
    # This is a conservative estimate based on band definitions
    if capability_band == "BASIC":
        success_rate_proxy = 0.3  # Conservative estimate for BASIC
    elif capability_band == "INTERMEDIATE":
        success_rate_proxy = 0.65  # Mid-range for INTERMEDIATE
    elif capability_band == "ADVANCED":
        success_rate_proxy = 0.85  # High estimate for ADVANCED
    else:
        success_rate_proxy = 0.0  # Unknown band
    
    return {
        "capability_band": capability_band,
        "success_rate_proxy": round(success_rate_proxy, 2),
        "resource_limit_count": resource_limit_count,
        "simulation_only": simulation_only,
        "safety_status": safety_status,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "LEAN_VERSION_REQUIRED",
    
    # Exceptions
    "LeanAdapterValidationError",
    
    # Enums
    "LeanAdapterMode",
    "LeanAbstentionReason",
    "VerificationErrorKind",
    
    # Dataclasses
    "LeanResourceBudget",
    "LeanVerificationRequest",
    "LeanVerificationResult",
    "LeanAdapterConfig",
    
    # Validation functions
    "validate_verification_request",
    "validate_resource_budget",
    "is_valid_canonical",
    
    # Simulation helper
    "simulate_lean_result",
    "compute_formula_complexity",
    
    # Evidence Pack helper
    "summarize_lean_activity",
    
    # Activity Ledger (Phase III)
    "build_lean_activity_ledger",
    
    # Safety Envelope (Phase III)
    "evaluate_lean_adapter_safety",
    
    # Global Health (Phase III)
    "summarize_lean_for_global_health",
    
    # Capability Classification (Phase IV)
    "classify_lean_capabilities",
    
    # Migration Checklist (Phase IV)
    "build_lean_migration_checklist",
    
    # Director Panel (Phase IV)
    "build_lean_director_panel",
    
    # Shadow Mode (Reality Bridge Protocol)
    "generate_shadow_telemetry",
    "build_lean_shadow_capability_radar",
    "build_lean_director_panel_with_shadow",
    
    # Lean Mode Playbook (Next Mission)
    "build_lean_mode_playbook",
    
    # Evidence Pack Adapter (Next Mission)
    "summarize_lean_capabilities_for_evidence",
    
    # Main class
    "LeanAdapter",
]
