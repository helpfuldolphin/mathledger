#!/usr/bin/env python3
"""Test script to validate dual-attestation composite root calculation."""

import json
import hashlib
import sys
import os

sys.path.insert(0, '.')

def test_composite_calculation():
    """Test the composite root calculation logic."""
    
    ui_root = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
    reasoning_root = "fedcba0987654321098765432109876543210fedcba0987654321098765432"
    
    composite_input = ui_root + reasoning_root
    composite_root = hashlib.sha256(composite_input.encode('utf-8')).hexdigest()
    
    print("Dual-Attestation Composite Root Test")
    print("=" * 50)
    print(f"UI Root:        {ui_root}")
    print(f"Reasoning Root: {reasoning_root}")
    print(f"Composite Root: {composite_root}")
    print(f"Input Length:   {len(composite_input)} chars")
    print(f"Valid SHA256:   {len(composite_root) == 64}")
    
    composite_root_2 = hashlib.sha256(composite_input.encode('utf-8')).hexdigest()
    print(f"Deterministic:  {composite_root == composite_root_2}")
    
    return composite_root

if __name__ == "__main__":
    test_composite_calculation()
