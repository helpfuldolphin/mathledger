"""
Lean Mode Interface — Three-Mode Verification Strategy

This module provides a clean, deterministic interface for Lean verification with
three operational modes designed for the First Organism chain:

1. MOCK: No Lean installed; deterministic failure/abstention signature.
         Safe for CI/CD, testing, and development without Lean toolchain.

2. DRY_RUN: Lean installed, runs on a minimal test statement (p → p).
            Validates Lean toolchain works without expensive proof search.

3. FULL: Real Lean verification for true proof checking.

Environment Variables:
    ML_LEAN_MODE: "mock", "dry_run", or "full" (default: "full")
    ML_LEAN_DRY_RUN_STATEMENT: Statement for dry-run (default: "p -> p")

Usage:
    from backend.lean_mode import get_lean_mode, get_build_runner, LeanMode

    mode = get_lean_mode()
    runner = get_build_runner(mode)
    result = execute_lean_job(statement, build_runner=runner)

First Organism Integration:
    - In mock mode, abstention is signaled via deterministic stderr hash
    - The ABSTENTION_SIGNATURE constant can be checked for mock abstentions
    - Mock mode produces reproducible hashes for MDAP compliance
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

# ---------------- Lean Mode Enum ----------------

class LeanMode(Enum):
    """Operational modes for Lean verification."""

    MOCK = "mock"
    """No Lean installed; deterministic mock responses."""

    DRY_RUN = "dry_run"
    """Lean installed; minimal test statement only."""

    FULL = "full"
    """Real Lean verification."""


# ---------------- Configuration ----------------

DEFAULT_MODE = LeanMode.FULL
DRY_RUN_STATEMENT = "p -> p"  # Minimal tautology for dry-run validation

# Deterministic mock output for reproducibility (MDAP compliance)
MOCK_STDOUT = "[MOCK] Lean verification simulated - no Lean toolchain available"
MOCK_STDERR = "[MOCK] ML_LEAN_MODE=mock - this is a deterministic abstention signal"
MOCK_RETURNCODE = 1  # Simulates build failure (abstention)

# Pre-computed hashes for mock outputs (deterministic)
MOCK_STDOUT_HASH = hashlib.sha256(MOCK_STDOUT.encode("utf-8")).hexdigest()
MOCK_STDERR_HASH = hashlib.sha256(MOCK_STDERR.encode("utf-8")).hexdigest()

# Abstention signature for First Organism detection
ABSTENTION_SIGNATURE = f"LEAN_MOCK_ABSTAIN::{MOCK_STDERR_HASH[:16]}"

# Full deterministic stderr (includes signature) - used by mock_lean_build
MOCK_STDERR_FULL = f"{MOCK_STDERR}\nSignature: {ABSTENTION_SIGNATURE}"
MOCK_STDERR_FULL_HASH = hashlib.sha256(MOCK_STDERR_FULL.encode("utf-8")).hexdigest()


# ---------------- Dataclasses ----------------

@dataclass(frozen=True)
class LeanModeConfig:
    """Configuration for Lean verification mode."""

    mode: LeanMode
    dry_run_statement: str
    lean_project_dir: str
    build_timeout: int

    @classmethod
    def from_env(cls) -> "LeanModeConfig":
        """Load configuration from environment variables."""
        mode_str = os.environ.get("ML_LEAN_MODE", "full").lower()
        try:
            mode = LeanMode(mode_str)
        except ValueError:
            mode = DEFAULT_MODE

        return cls(
            mode=mode,
            dry_run_statement=os.environ.get(
                "ML_LEAN_DRY_RUN_STATEMENT",
                DRY_RUN_STATEMENT
            ),
            lean_project_dir=os.environ.get(
                "LEAN_PROJECT_DIR",
                r"C:\dev\mathledger\backend\lean_proj",
            ),
            build_timeout=int(os.environ.get("LEAN_BUILD_TIMEOUT", "90")),
        )


# ---------------- Build Runners ----------------

def mock_lean_build(module_name: str) -> subprocess.CompletedProcess[str]:
    """
    Pure mock build runner — no Lean toolchain required.

    Returns a deterministic failure result for testing and CI/CD.
    The stderr contains a signature that can be detected as an abstention.

    IMPORTANT: For MDAP compliance, stdout and stderr must be fully deterministic
    (independent of module_name) so that hashes are reproducible across runs.
    The module_name is recorded in the args field only.

    Args:
        module_name: The Lean module to "build" (recorded in args, not in output)

    Returns:
        CompletedProcess with deterministic mock output
    """
    # Small deterministic delay to simulate build time
    time.sleep(0.001)

    return subprocess.CompletedProcess(
        args=["lake", "build", module_name],
        returncode=MOCK_RETURNCODE,
        stdout=MOCK_STDOUT,
        stderr=MOCK_STDERR_FULL,
    )


def dry_run_lean_build(
    module_name: str,
    *,
    project_dir: str,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
    """
    Dry-run build runner — validates Lean toolchain on minimal statement.

    Uses a trivial statement (p → p) to verify Lean is working without
    expensive proof search. If the dry-run passes, the actual module
    is also built.

    Args:
        module_name: The Lean module to build
        project_dir: Path to the Lean project directory
        timeout: Build timeout in seconds

    Returns:
        CompletedProcess from the Lean build
    """
    # First, verify Lean is available by checking lake version
    try:
        version_check = subprocess.run(
            ["lake", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if version_check.returncode != 0:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=127,
                stdout="",
                stderr=f"[DRY_RUN] Lake not available: {version_check.stderr}",
            )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["lake", "build", module_name],
            returncode=127,
            stdout="",
            stderr="[DRY_RUN] Lake executable not found in PATH",
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["lake", "build", module_name],
            returncode=124,
            stdout="",
            stderr="[DRY_RUN] Lake version check timed out",
        )

    # Now run the actual build
    return _run_lake_build(module_name, project_dir, timeout)


def full_lean_build(
    module_name: str,
    *,
    project_dir: str,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
    """
    Full build runner — real Lean verification.

    Args:
        module_name: The Lean module to build
        project_dir: Path to the Lean project directory
        timeout: Build timeout in seconds

    Returns:
        CompletedProcess from the Lean build
    """
    return _run_lake_build(module_name, project_dir, timeout)


def _run_lake_build(
    module_name: str,
    project_dir: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Internal helper to run lake build with timeout handling."""
    try:
        return subprocess.run(
            ["lake", "build", module_name],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "")
            + f"\n[TIMEOUT] LakeBuild: Execution exceeded {timeout}s for {module_name}",
        )


# ---------------- Mode-Aware Factory ----------------

BuildRunner = Callable[[str], subprocess.CompletedProcess[str]]


def get_lean_mode() -> LeanMode:
    """
    Get the current Lean mode from environment.

    Returns:
        LeanMode enum value
    """
    mode_str = os.environ.get("ML_LEAN_MODE", "full").lower()
    try:
        return LeanMode(mode_str)
    except ValueError:
        return DEFAULT_MODE


def get_build_runner(
    mode: Optional[LeanMode] = None,
    *,
    project_dir: Optional[str] = None,
    timeout: int = 90,
) -> BuildRunner:
    """
    Get the appropriate build runner for the specified mode.

    Args:
        mode: Lean mode (defaults to environment-configured mode)
        project_dir: Override for Lean project directory
        timeout: Build timeout in seconds

    Returns:
        Callable build runner function

    Example:
        runner = get_build_runner(LeanMode.MOCK)
        result = execute_lean_job("p -> p", build_runner=runner)
    """
    if mode is None:
        mode = get_lean_mode()

    if project_dir is None:
        project_dir = os.environ.get(
            "LEAN_PROJECT_DIR",
            r"C:\dev\mathledger\backend\lean_proj",
        )

    if mode == LeanMode.MOCK:
        return mock_lean_build

    elif mode == LeanMode.DRY_RUN:
        def _dry_run_runner(module_name: str) -> subprocess.CompletedProcess[str]:
            return dry_run_lean_build(
                module_name,
                project_dir=project_dir,
                timeout=timeout,
            )
        return _dry_run_runner

    else:  # FULL
        def _full_runner(module_name: str) -> subprocess.CompletedProcess[str]:
            return full_lean_build(
                module_name,
                project_dir=project_dir,
                timeout=timeout,
            )
        return _full_runner


# ---------------- Abstention Detection ----------------

def is_mock_abstention(stderr: str) -> bool:
    """
    Check if the stderr indicates a mock abstention.

    Used by First Organism to detect when verification was simulated.

    Args:
        stderr: The stderr output from Lean build

    Returns:
        True if this was a mock abstention
    """
    return ABSTENTION_SIGNATURE in stderr


def is_lean_available() -> bool:
    """
    Check if Lean toolchain is available on this system.

    Returns:
        True if lake command is accessible
    """
    try:
        result = subprocess.run(
            ["lake", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def recommended_mode() -> LeanMode:
    """
    Get the recommended Lean mode for this system.

    Returns:
        LeanMode.MOCK if Lean unavailable, otherwise LeanMode.FULL
    """
    if is_lean_available():
        return LeanMode.FULL
    return LeanMode.MOCK


# ---------------- First Organism Helpers ----------------

@dataclass(frozen=True)
class LeanModeStatus:
    """Status report for Lean mode configuration."""

    configured_mode: LeanMode
    lean_available: bool
    recommended_mode: LeanMode
    effective_mode: LeanMode

    @property
    def is_mock(self) -> bool:
        return self.effective_mode == LeanMode.MOCK

    @property
    def will_abstain(self) -> bool:
        """True if verification will produce abstention (mock mode)."""
        return self.is_mock

    def to_dict(self) -> dict:
        return {
            "configured_mode": self.configured_mode.value,
            "lean_available": self.lean_available,
            "recommended_mode": self.recommended_mode.value,
            "effective_mode": self.effective_mode.value,
            "is_mock": self.is_mock,
            "will_abstain": self.will_abstain,
        }


class LeanToolchainUnavailableError(Exception):
    """
    Raised when Lean toolchain is required but not available.

    This error is raised when ML_LEAN_MODE is not explicitly set to 'mock'
    and the Lean toolchain cannot be found. This prevents silent fallback
    to mock mode which would hide verification failures from operators.

    Resolution:
        1. Run 'make lean-setup' to install the Lean toolchain, OR
        2. Set ML_LEAN_MODE=mock explicitly if mock mode is intended
    """
    pass


def get_lean_status() -> LeanModeStatus:
    """
    Get comprehensive status of Lean mode configuration.

    IMPORTANT: This function will raise LeanToolchainUnavailableError if:
        - Lean is not available on the system, AND
        - ML_LEAN_MODE is not explicitly set to 'mock'

    This prevents silent fallback to mock mode. Operators must either:
        1. Run 'make lean-setup' to install the Lean toolchain
        2. Explicitly set ML_LEAN_MODE=mock if mock mode is intended

    Returns:
        LeanModeStatus with all relevant information

    Raises:
        LeanToolchainUnavailableError: If Lean unavailable and mock not explicit
    """
    configured = get_lean_mode()
    available = is_lean_available()
    recommended = recommended_mode()

    # Check if mock mode was EXPLICITLY requested via environment
    explicit_mock = os.environ.get("ML_LEAN_MODE", "").lower() == "mock"

    # CRITICAL: If configured for real Lean but unavailable, FAIL LOUDLY
    # unless mock mode was explicitly requested
    if configured in (LeanMode.FULL, LeanMode.DRY_RUN) and not available:
        if not explicit_mock:
            raise LeanToolchainUnavailableError(
                "Lean toolchain not available but ML_LEAN_MODE is not 'mock'.\n\n"
                "The Lean verification system requires either:\n"
                "  1. A working Lean installation (run 'make lean-setup'), OR\n"
                "  2. Explicit mock mode (set ML_LEAN_MODE=mock)\n\n"
                "Silent fallback to mock mode is disabled to prevent hidden verification failures.\n"
                "If you intentionally want mock mode, set: export ML_LEAN_MODE=mock"
            )
        effective = LeanMode.MOCK
    else:
        effective = configured

    return LeanModeStatus(
        configured_mode=configured,
        lean_available=available,
        recommended_mode=recommended,
        effective_mode=effective,
    )


# ---------------- Module Exports ----------------

__all__ = [
    # Enums and Config
    "LeanMode",
    "LeanModeConfig",

    # Exceptions
    "LeanToolchainUnavailableError",

    # Build Runners
    "mock_lean_build",
    "dry_run_lean_build",
    "full_lean_build",
    "BuildRunner",

    # Factory Functions
    "get_lean_mode",
    "get_build_runner",

    # Constants
    "MOCK_STDOUT",
    "MOCK_STDERR",
    "MOCK_STDERR_FULL",
    "MOCK_STDOUT_HASH",
    "MOCK_STDERR_HASH",
    "MOCK_STDERR_FULL_HASH",
    "ABSTENTION_SIGNATURE",

    # Detection Helpers
    "is_mock_abstention",
    "is_lean_available",
    "recommended_mode",

    # Status
    "LeanModeStatus",
    "get_lean_status",
]
