#!/usr/bin/env python3
"""
Attestation Snapshot Generator

Generates a deterministic, versioned snapshot of dual-attestation seals.

Data Sources:
  - artifacts/governance/attestation_history.jsonl (JSONL file with attestation records)

Output:
  - JSON snapshot compliant with schemas/attestation_snapshot.schema.json
  - Printed to stdout in RFC 8785 canonical form

Exit Codes:
  0 - Success
  1 - Data source missing or corrupted
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


def generate_attestation_snapshot(repo_root: Path) -> Dict[str, Any]:
    """Generate the attestation snapshot."""
    
    # Path to the attestation history file
    history_file = repo_root / 'artifacts' / 'governance' / 'attestation_history.jsonl'
    
    if not history_file.exists():
        print(f"ERROR: Attestation history file not found: {history_file}", file=sys.stderr)
        sys.exit(1)
    
    # Read attestation records
    attestations = []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    record = json.loads(line)
                    
                    # Extract required fields
                    attestation = {
                        'id': record.get('id', ''),
                        'H_t': record.get('H_t', ''),
                        'R_t': record.get('R_t', ''),
                        'U_t': record.get('U_t', '')
                    }
                    
                    attestations.append(attestation)
                
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Failed to read attestation history: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build snapshot
    snapshot = {
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'attestations': attestations
    }
    
    return snapshot


def canonicalize_json(obj: Any) -> str:
    """
    Serialize an object to RFC 8785 canonical JSON.
    """
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def main():
    repo_root = Path.cwd()
    
    # Generate snapshot
    snapshot = generate_attestation_snapshot(repo_root)
    
    # Canonicalize and print
    canonical_json = canonicalize_json(snapshot)
    print(canonical_json)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
