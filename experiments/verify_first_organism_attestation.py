#!/usr/bin/env python3
"""
First Organism Attestation Verifier

Verifies the integrity of the First Organism attestation artifact by:
1. Loading and validating the JSON structure
2. Ensuring all required fields (R_t, U_t, H_t) are present and valid
3. Recomputing H_t = SHA256(R_t || U_t) and verifying it matches stored H_t

This implements the deterministic recompute hook for SPARK verification.

Usage:
    python experiments/verify_first_organism_attestation.py [attestation_file]
    
    Default attestation file: artifacts/first_organism/attestation.json

Exit codes:
    0: All verifications passed
    1: Verification failed (with detailed error message)
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from basis.attestation.dual import composite_root


# Attestation Schema Documentation
# =================================
# The First Organism attestation.json must contain:
#
# Required fields:
#   - R_t: Reasoning merkle root (64-char hex string)
#   - U_t: UI merkle root (64-char hex string)  
#   - H_t: Composite attestation root (64-char hex string)
#
# H_t must equal SHA256(R_t || U_t) where || denotes concatenation
# in ASCII encoding.
#
# Optional fields (for audit trail):
#   - statement_hash: Hash of the statement that triggered the run
#   - mdap_seed: Deterministic seed for reproducibility
#   - run_id: Unique identifier for this run
#   - run_timestamp_iso: ISO 8601 timestamp
#   - version: Attestation format version
#   - environment_mode: Execution mode (standalone, integrated, etc.)
#   - chain_status: Blockchain integration status
#   - components: Dictionary mapping component names to implementations


def is_valid_hex_digest(s: str, length: int = 64) -> bool:
    """Check if string is a valid hex digest of specified length."""
    if len(s) != length:
        return False
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def validate_attestation_structure(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate attestation JSON structure.
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    required_fields = ["R_t", "U_t", "H_t"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        value = data[field]
        if not isinstance(value, str):
            return False, f"Field {field} must be a string, got {type(value).__name__}"
        
        if not is_valid_hex_digest(value, length=64):
            return False, (
                f"Field {field} must be a 64-character hex string, "
                f"got {len(value)} characters: {value[:16]}..."
            )
    
    return True, ""


def recompute_composite_root(r_t: str, u_t: str) -> str:
    """
    Recompute H_t from R_t and U_t using canonical function.
    
    This uses the same algorithm as basis.attestation.dual.composite_root:
        H_t = SHA256(R_t || U_t) where || is ASCII concatenation
    """
    return composite_root(r_t, u_t)


def verify_attestation_integrity(attestation_file: Path) -> tuple[bool, str]:
    """
    Verify attestation integrity by recomputing H_t.
    
    Returns:
        (is_valid, error_message)
    """
    if not attestation_file.exists():
        return False, f"Attestation file not found: {attestation_file}"
    
    try:
        with open(attestation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in attestation file: {e}"
    except Exception as e:
        return False, f"Error reading attestation file: {e}"
    
    # Validate structure
    is_valid, error = validate_attestation_structure(data)
    if not is_valid:
        return False, error
    
    # Extract roots
    r_t = data["R_t"]
    u_t = data["U_t"]
    stored_h_t = data["H_t"]
    
    # Recompute H_t
    try:
        recomputed_h_t = recompute_composite_root(r_t, u_t)
    except Exception as e:
        return False, f"Error recomputing composite root: {e}"
    
    # Verify match
    if recomputed_h_t != stored_h_t:
        return False, (
            f"H_t mismatch:\n"
            f"  Stored:    {stored_h_t}\n"
            f"  Recomputed: {recomputed_h_t}\n"
            f"  R_t:       {r_t}\n"
            f"  U_t:       {u_t}"
        )
    
    return True, ""


def main():
    """Main entry point."""
    # Default attestation file path
    default_attestation = Path(__file__).parent.parent / "artifacts" / "first_organism" / "attestation.json"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        attestation_file = Path(sys.argv[1])
    else:
        attestation_file = default_attestation
    
    # Verify integrity
    is_valid, error = verify_attestation_integrity(attestation_file)
    
    if not is_valid:
        print(f"❌ Attestation verification FAILED", file=sys.stderr)
        print(f"   {error}", file=sys.stderr)
        sys.exit(1)
    
    # Load and display summary
    with open(attestation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("✅ Attestation verification PASSED")
    print(f"   File: {attestation_file}")
    print(f"   R_t: {data['R_t']}")
    print(f"   U_t: {data['U_t']}")
    print(f"   H_t: {data['H_t']}")
    
    # Verify recomputation
    recomputed = recompute_composite_root(data['R_t'], data['U_t'])
    print(f"   Recomputed H_t: {recomputed}")
    print(f"   ✓ H_t matches recomputed value")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

