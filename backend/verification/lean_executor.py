# REAL-READY
"""
Lean Executor

Provides Lean command construction and version detection.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: REAL-READY
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def construct_lean_command(
    module_path: Path,
    timeout_s: float = 60.0,
    use_lake: bool = False,
    trace_tactics: bool = False,
) -> List[str]:
    """
    Construct Lean verification command.
    
    Args:
        module_path: Path to Lean module file
        timeout_s: Timeout in seconds
        use_lake: Use Lake build system
        trace_tactics: Enable tactic tracing
    
    Returns:
        List of command arguments
    """
    
    if use_lake:
        # Use Lake for project builds
        cmd = ["lake", "env", "lean"]
    else:
        # Use standalone Lean
        cmd = ["lean"]
    
    # Add timeout (convert to milliseconds for Lean 4)
    timeout_ms = int(timeout_s * 1000)
    cmd.extend(["--timeout", str(timeout_ms)])
    
    # Add tactic tracing if requested
    if trace_tactics:
        cmd.extend(["--trace", "tactic"])
    
    # Add module path
    cmd.append(str(module_path))
    
    return cmd


def get_lean_version() -> str:
    """
    Get Lean version string.
    
    Returns:
        Lean version or "unknown" if not available
    """
    
    try:
        result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        
        if result.returncode == 0:
            # Parse version from output
            # Example: "Lean (version 4.0.0, commit abcd1234, Release)"
            version_line = result.stdout.strip().split("\n")[0]
            return version_line
        else:
            return "unknown"
    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "unknown"
