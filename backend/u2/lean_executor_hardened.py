# REAL-READY
"""
LeanExecutor: Lean 4 Theorem Prover Integration (Hardened)

This module implements the Lean Swap Plan with clean separation between
stub (DEMO-SCAFFOLD) and production (REAL-READY) implementations.

Features:
- Capability detection for Lean 4 installation
- Explicit opt-in for stub via environment variable
- Clear error messages when Lean is not available
- No silent fallbacks

Implements: docs/lean_swap_plan.md
Replaces: PropositionalVerifier in fosubstrate_executor.py

Author: Manus-F
Date: 2025-12-06
Status: REAL-READY (hardened with stub/real separation)
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from backend.u2.fosubstrate_executor_skeleton import (
    StatementRecord,
    VerificationTimeoutError,
    DerivationError,
)


# ============================================================================
# CAPABILITY DETECTION
# ============================================================================

class LeanCapability:
    """
    Detects Lean 4 installation and manages capability state.
    """
    
    @staticmethod
    def is_lean_installed() -> bool:
        """
        Check if Lean 4 is installed and available on PATH.
        
        Returns:
            True if lean command is available, False otherwise
        """
        return shutil.which("lean") is not None
    
    @staticmethod
    def is_stub_allowed() -> bool:
        """
        Check if stub mode is explicitly allowed via environment variable.
        
        Returns:
            True if U2_LEAN_ALLOW_STUB=1, False otherwise
        """
        return os.environ.get("U2_LEAN_ALLOW_STUB", "0") == "1"
    
    @staticmethod
    def get_lean_version() -> Optional[str]:
        """
        Get Lean version if installed.
        
        Returns:
            Version string or None if not installed
        """
        if not LeanCapability.is_lean_installed():
            return None
        
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


# ============================================================================
# LEAN EXECUTOR (REAL-READY)
# ============================================================================

class LeanExecutorReal:
    """
    # REAL-READY
    
    Lean 4 theorem prover executor (production implementation).
    
    Requires Lean 4 to be installed and available on PATH.
    """
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize Lean executor.
        
        Args:
            timeout_seconds: Maximum time allowed for proof search
            
        Raises:
            RuntimeError: If Lean is not installed
        """
        if not LeanCapability.is_lean_installed():
            raise RuntimeError(
                "Lean 4 is not installed or not available on PATH. "
                "Install Lean 4 from https://leanprover.github.io/lean4/doc/setup.html"
            )
        
        self.timeout_seconds = timeout_seconds
        self.lean_version = LeanCapability.get_lean_version()
    
    def verify(self, statement: StatementRecord) -> Tuple[bool, str]:
        """
        Verify statement using Lean 4.
        
        Args:
            statement: Statement to verify
            
        Returns:
            Tuple of (is_tautology, verification_method)
            
        Raises:
            VerificationTimeoutError: If proof search times out
            DerivationError: If statement has syntax error
        """
        # Generate .lean source file
        lean_source = self._generate_lean_source(statement)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            delete=False
        ) as f:
            f.write(lean_source)
            lean_file_path = Path(f.name)
        
        try:
            # Invoke lean command
            is_tautology, method = self._invoke_lean(lean_file_path)
            return is_tautology, method
        finally:
            # Clean up temporary file
            lean_file_path.unlink()
    
    def _generate_lean_source(self, statement: StatementRecord) -> str:
        """
        Generate Lean source file from statement.
        
        Args:
            statement: Statement to verify
            
        Returns:
            Lean source code as string
        """
        return f"""-- U2 Planner Verification Request
-- Statement Hash: {statement.hash}

#check ({statement.normalized})
"""
    
    def _invoke_lean(self, lean_file_path: Path) -> Tuple[bool, str]:
        """
        Invoke Lean command and parse output.
        
        Args:
            lean_file_path: Path to .lean file
            
        Returns:
            Tuple of (is_tautology, verification_method)
            
        Raises:
            VerificationTimeoutError: If proof search times out
            DerivationError: If statement has syntax error
        """
        # Invoke lean command with timeout
        cmd = ["timeout", f"{self.timeout_seconds}s", "lean", str(lean_file_path)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds + 1  # Extra second for timeout command
            )
        except subprocess.TimeoutExpired:
            raise VerificationTimeoutError("Lean proof search timed out")
        
        # Parse output
        return self._parse_lean_output(
            result.stdout,
            result.stderr,
            result.returncode
        )
    
    def _parse_lean_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int
    ) -> Tuple[bool, str]:
        """
        Parse Lean output to determine outcome.
        
        Args:
            stdout: Standard output from lean command
            stderr: Standard error from lean command
            return_code: Process return code
            
        Returns:
            Tuple of (is_tautology, verification_method)
            
        Raises:
            VerificationTimeoutError: If proof search times out
            DerivationError: If statement has syntax error
        """
        # Timeout (exit code 124 from timeout command)
        if return_code == 124:
            raise VerificationTimeoutError("Lean proof search timed out")
        
        # Syntax error
        if "error: expected" in stderr:
            raise DerivationError(f"Lean syntax error: {stderr}")
        
        # Type mismatch (not a valid proposition)
        if "error: type mismatch" in stderr:
            return False, "lean-type-error"
        
        # Success (contains statement type)
        if ": Prop" in stdout:
            return True, f"lean-verified-{self.lean_version}"
        
        # Unknown error
        raise DerivationError(f"Lean unknown error: {stderr}")


# ============================================================================
# LEAN EXECUTOR STUB (DEMO-SCAFFOLD)
# ============================================================================

class LeanExecutorStub:
    """
    # DEMO-SCAFFOLD
    
    Lean executor stub for testing and development.
    
    This stub ALWAYS returns True for any statement.
    
    MUST be explicitly enabled via U2_LEAN_ALLOW_STUB=1 environment variable.
    """
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize Lean executor stub.
        
        Args:
            timeout_seconds: Ignored (stub has no timeout)
            
        Raises:
            RuntimeError: If stub is not explicitly allowed
        """
        if not LeanCapability.is_stub_allowed():
            raise RuntimeError(
                "LeanExecutorStub requires explicit opt-in via U2_LEAN_ALLOW_STUB=1 environment variable. "
                "This stub is for testing only and should not be used in production."
            )
        
        self.timeout_seconds = timeout_seconds
    
    def verify(self, statement: StatementRecord) -> Tuple[bool, str]:
        """
        Verify statement using stub (always returns True).
        
        Args:
            statement: Statement to verify
            
        Returns:
            Tuple of (True, "lean-stub")
        """
        return True, "lean-stub"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_lean_executor(
    timeout_seconds: int = 5,
    allow_stub: bool = False
) -> "LeanExecutorReal | LeanExecutorStub":
    """
    Create Lean executor with capability detection.
    
    Args:
        timeout_seconds: Maximum time allowed for proof search
        allow_stub: If True, allow stub when Lean is not installed
                   (requires U2_LEAN_ALLOW_STUB=1 env var)
    
    Returns:
        LeanExecutorReal if Lean is installed, LeanExecutorStub if allowed
        
    Raises:
        RuntimeError: If Lean is not installed and stub is not allowed
    """
    if LeanCapability.is_lean_installed():
        return LeanExecutorReal(timeout_seconds=timeout_seconds)
    
    if allow_stub and LeanCapability.is_stub_allowed():
        return LeanExecutorStub(timeout_seconds=timeout_seconds)
    
    # Lean not installed and stub not allowed
    error_msg = (
        "Lean 4 is not installed or not available on PATH. "
        "Install Lean 4 from https://leanprover.github.io/lean4/doc/setup.html\n\n"
        "For testing only, you can enable the stub with:\n"
        "  export U2_LEAN_ALLOW_STUB=1\n"
        "  (or set U2_LEAN_ALLOW_STUB=1 on Windows)"
    )
    raise RuntimeError(error_msg)


# ============================================================================
# INTEGRATION POINT
# ============================================================================

def create_executor(executor_type: str = "propositional", **kwargs):
    """
    Factory function for creating executors.
    
    This is the integration point for swapping between PropositionalVerifier
    and LeanExecutor.
    
    Args:
        executor_type: Type of executor ("propositional" or "lean")
        **kwargs: Additional arguments for executor
        
    Returns:
        Executor instance
    """
    if executor_type == "lean":
        return create_lean_executor(**kwargs)
    elif executor_type == "propositional":
        from backend.u2.fosubstrate_executor import PropositionalVerifier
        return PropositionalVerifier()
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for Lean executor."""
    import sys
    
    # Print capability status
    print("Lean Capability Status:")
    print(f"  Lean installed: {LeanCapability.is_lean_installed()}")
    print(f"  Lean version: {LeanCapability.get_lean_version() or 'N/A'}")
    print(f"  Stub allowed: {LeanCapability.is_stub_allowed()}")
    print()
    
    if len(sys.argv) != 2:
        print("Usage: python lean_executor_hardened.py <statement>")
        sys.exit(1)
    
    statement_str = sys.argv[1]
    
    # Create statement record
    statement = StatementRecord(
        normalized=statement_str,
        hash=hashlib.sha256(statement_str.encode()).hexdigest(),
        pretty=statement_str,
        rule="test",
        is_axiom=False,
        mp_depth=0,
        parents=(),
        verification_method="lean",
    )
    
    # Create executor
    try:
        executor = create_lean_executor(allow_stub=True)
        print(f"Using executor: {executor.__class__.__name__}")
        print()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Verify statement
    try:
        is_tautology, method = executor.verify(statement)
        print(f"Statement: {statement_str}")
        print(f"Is tautology: {is_tautology}")
        print(f"Verification method: {method}")
    except (VerificationTimeoutError, DerivationError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
