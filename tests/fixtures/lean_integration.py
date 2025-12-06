"""
Real Lean Integration Fixtures.

Provides fixtures for testing with actual Lean toolchain when available.
These fixtures automatically skip tests if Lean is not installed.

Usage:
    @pytest.mark.lean_required
    def test_real_lean_verification(lean_runner):
        result = lean_runner("p -> p")
        assert result.build_result.returncode == 0
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional

import pytest

from backend.lean_mode import (
    LeanMode,
    get_lean_status,
    get_build_runner,
    is_lean_available,
    full_lean_build,
    dry_run_lean_build,
)
from backend.worker import (
    LeanJobResult,
    execute_lean_job,
    ensure_namespace_dirs,
)


@dataclass(frozen=True)
class LeanEnvironment:
    """Describes the Lean environment for integration tests."""

    available: bool
    version: Optional[str]
    project_dir: str
    mode: LeanMode

    @classmethod
    def detect(cls) -> "LeanEnvironment":
        """Detect the current Lean environment."""
        available = is_lean_available()
        version = None

        if available:
            try:
                result = subprocess.run(
                    ["lake", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
            except Exception:
                pass

        project_dir = os.environ.get(
            "LEAN_PROJECT_DIR",
            str(Path(__file__).parent.parent.parent / "backend" / "lean_proj"),
        )

        status = get_lean_status()

        return cls(
            available=available,
            version=version,
            project_dir=project_dir,
            mode=status.effective_mode,
        )


def pytest_configure(config):
    """Register the lean_required marker."""
    config.addinivalue_line(
        "markers",
        "lean_required: mark test as requiring real Lean toolchain",
    )


@pytest.fixture(scope="session")
def lean_environment() -> LeanEnvironment:
    """Session-scoped fixture providing Lean environment info."""
    return LeanEnvironment.detect()


@pytest.fixture(scope="session")
def lean_available(lean_environment: LeanEnvironment) -> bool:
    """Session-scoped fixture indicating if Lean is available."""
    return lean_environment.available


@pytest.fixture(scope="function")
def require_lean(lean_environment: LeanEnvironment):
    """Skip test if Lean is not available."""
    if not lean_environment.available:
        pytest.skip("Lean toolchain not available")


@pytest.fixture(scope="function")
def lean_runner(
    lean_environment: LeanEnvironment,
    tmp_path: Path,
) -> Generator[Callable[[str], LeanJobResult], None, None]:
    """
    Fixture providing a Lean job runner for integration tests.

    Automatically skips if Lean is not available.

    Usage:
        def test_real_proof(lean_runner):
            result = lean_runner("p -> p")
            assert result.build_result.returncode == 0
    """
    if not lean_environment.available:
        pytest.skip("Lean toolchain not available")

    runner = get_build_runner(
        mode=LeanMode.FULL,
        project_dir=lean_environment.project_dir,
        timeout=90,
    )

    def run_lean_job(statement: str) -> LeanJobResult:
        return execute_lean_job(
            statement,
            jobs_dir=str(tmp_path),
            build_runner=runner,
            cleanup=True,
        )

    yield run_lean_job


@pytest.fixture(scope="function")
def lean_dry_run_runner(
    lean_environment: LeanEnvironment,
    tmp_path: Path,
) -> Generator[Callable[[str], LeanJobResult], None, None]:
    """
    Fixture providing a dry-run Lean job runner.

    Validates Lean toolchain before running the actual proof.
    """
    if not lean_environment.available:
        pytest.skip("Lean toolchain not available")

    runner = get_build_runner(
        mode=LeanMode.DRY_RUN,
        project_dir=lean_environment.project_dir,
        timeout=90,
    )

    def run_lean_job(statement: str) -> LeanJobResult:
        return execute_lean_job(
            statement,
            jobs_dir=str(tmp_path),
            build_runner=runner,
            cleanup=True,
        )

    yield run_lean_job


@pytest.fixture(scope="session")
def lean_project_initialized(lean_environment: LeanEnvironment) -> bool:
    """
    Ensure the Lean project directory is properly initialized.

    Returns True if initialization succeeded.
    """
    if not lean_environment.available:
        return False

    try:
        ensure_namespace_dirs()
        return True
    except Exception:
        return False


# Test statements for Lean integration
KNOWN_TAUTOLOGIES = [
    "p -> p",
    "p -> q -> p",
    "p \\/ ~p",
    "(p -> q) -> (~q -> ~p)",
]

KNOWN_NON_TAUTOLOGIES = [
    "p",
    "p /\\ ~p",
    "p -> q",
]

COMPLEX_STATEMENTS = [
    "((p -> q) -> p) -> p",  # Peirce's law
    "(p -> q) -> (q -> r) -> (p -> r)",  # Transitivity
]


@pytest.fixture
def known_tautology() -> str:
    """Return a known tautology for testing."""
    return KNOWN_TAUTOLOGIES[0]


@pytest.fixture
def known_non_tautology() -> str:
    """Return a known non-tautology for testing."""
    return KNOWN_NON_TAUTOLOGIES[0]


@pytest.fixture
def complex_statement() -> str:
    """Return a complex statement for testing."""
    return COMPLEX_STATEMENTS[0]

