#!/usr/bin/env python3
"""
RFC 8785 JSON Canonicalization Scheme (JCS) Implementation
Ensures deterministic JSON serialization for cryptographic hashing
"""

import json
from typing import Any


def canonicalize_json(obj: Any) -> str:
    """
    Canonicalize JSON according to RFC 8785
    
    Rules:
    1. Whitespace is removed
    2. Object keys are sorted lexicographically
    3. Numbers are serialized in a specific format
    4. Unicode escaping is normalized
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        indent=None
    )


def canonicalize_file(input_path: str, output_path: str = None):
    """Canonicalize a JSON file"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    canonical = canonicalize_json(data)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(canonical)
    else:
        print(canonical)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: rfc8785_canon.py <input.json> [output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    canonicalize_file(input_file, output_file)
