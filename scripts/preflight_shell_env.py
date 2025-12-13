#!/usr/bin/env python3
"""
Cross-Shell Environment Preflight Check

Detects common cross-shell issues (Git Bash / PowerShell) and prints
advisory messages with fix commands. Does NOT block execution.

Usage:
    python scripts/preflight_shell_env.py

Or import and call check_bashrc_bom() programmatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


# =============================================================================
# BOM Detection
# =============================================================================

# UTF-16 LE BOM bytes
UTF16_LE_BOM = b"\xff\xfe"

# UTF-8 BOM bytes
UTF8_BOM = b"\xef\xbb\xbf"


def detect_bom(file_bytes: bytes) -> Optional[str]:
    """
    Detect BOM type from file bytes.

    Args:
        file_bytes: Raw bytes from beginning of file (at least 3 bytes recommended)

    Returns:
        "UTF-16-LE" if UTF-16 LE BOM detected
        "UTF-8" if UTF-8 BOM detected
        None if no BOM detected
    """
    if len(file_bytes) >= 2 and file_bytes[:2] == UTF16_LE_BOM:
        return "UTF-16-LE"
    if len(file_bytes) >= 3 and file_bytes[:3] == UTF8_BOM:
        return "UTF-8"
    return None


def check_bashrc_bom() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if ~/.bashrc has a BOM that causes Git Bash errors.

    Returns:
        (has_bom, bom_type, advisory_message)
        - has_bom: True if problematic BOM detected
        - bom_type: "UTF-16-LE" or "UTF-8" if detected, None otherwise
        - advisory_message: Human-readable fix instruction if BOM found
    """
    # Locate .bashrc
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        return False, None, None

    bashrc_path = Path(home) / ".bashrc"
    if not bashrc_path.exists():
        return False, None, None

    # Read first few bytes
    try:
        with open(bashrc_path, "rb") as f:
            header = f.read(4)
    except (OSError, IOError):
        return False, None, None

    bom_type = detect_bom(header)
    if bom_type is None:
        return False, None, None

    # BOM detected - generate advisory
    advisory = (
        f".bashrc has {bom_type} BOM (causes Git Bash error). "
        f"Fix: powershell -ExecutionPolicy Bypass -File scripts/fix_bashrc_encoding.ps1"
    )
    return True, bom_type, advisory


def print_preflight_advisory() -> None:
    """
    Print preflight advisory if issues detected.
    Does NOT block execution - advisory only.
    """
    has_bom, bom_type, advisory = check_bashrc_bom()
    if has_bom and advisory:
        print(f"[ADVISORY] {advisory}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> int:
    """Run preflight checks and print advisories."""
    print("=== Cross-Shell Environment Preflight ===")

    has_bom, bom_type, advisory = check_bashrc_bom()
    if has_bom:
        print(f"[ADVISORY] {advisory}")
    else:
        print("[OK] .bashrc encoding: no BOM detected")

    print()
    print("Preflight complete (advisory only, does not block)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
