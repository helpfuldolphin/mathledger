#!/usr/bin/env python3
"""
Metrics v1 linter CLI for enforcing v1 contract before exporter runs.

Validates metrics files to ensure they follow the v1 schema and detects mixed schema
feeds with actionable guidance.

Exit codes:
  0: OK (no violations)
  2: Mixed schema detected
  3: Bad fields (missing required fields or invalid values)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def lint_v1(path: str) -> Dict[str, Any]:
    """
    Lint a metrics file for v1 contract compliance.

    Args:
        path: Path to the metrics file (JSONL format)

    Returns:
        Dict with counts and violations:
        {
            'v1_count': int,
            'legacy_count': int,
            'violations': List[str],
            'is_mixed_schema': bool
        }
    """
    violations = []
    v1_count = 0
    legacy_count = 0

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    violations.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Check if this is a v1 record
                is_v1 = _is_v1_record(record)
                if is_v1:
                    v1_count += 1
                    # Validate v1 fields
                    v1_violations = _validate_v1_fields(record, line_num)
                    violations.extend(v1_violations)
                else:
                    legacy_count += 1
                    # Check for legacy-specific issues
                    legacy_violations = _validate_legacy_record(record, line_num)
                    violations.extend(legacy_violations)

    except FileNotFoundError:
        violations.append(f"File not found: {path}")
    except Exception as e:
        violations.append(f"Error reading file: {e}")

    is_mixed_schema = v1_count > 0 and legacy_count > 0

    return {
        'v1_count': v1_count,
        'legacy_count': legacy_count,
        'violations': violations,
        'is_mixed_schema': is_mixed_schema
    }


def _is_v1_record(record: Dict[str, Any]) -> bool:
    """Check if a record follows v1 schema (has all required v1 fields)."""
    required_v1_fields = {
        'system', 'mode', 'method', 'seed', 'inserted_proofs',
        'wall_minutes', 'block_no', 'merkle'
    }
    return required_v1_fields.issubset(record.keys())


def _validate_v1_fields(record: Dict[str, Any], line_num: int) -> List[str]:
    """Validate v1 record fields according to the contract."""
    violations = []

    # Required fields validation
    required_fields = {
        'system': str,
        'mode': str,
        'method': str,
        'seed': str,
        'inserted_proofs': int,
        'wall_minutes': (int, float),
        'block_no': int,
        'merkle': str
    }

    for field, expected_type in required_fields.items():
        if field not in record:
            violations.append(f"Line {line_num}: Missing required field '{field}'")
            continue

        value = record[field]
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = " or ".join(t.__name__ for t in expected_type)
            else:
                type_names = expected_type.__name__
            violations.append(f"Line {line_num}: Field '{field}' must be {type_names}, got {type(value).__name__}")
            continue

    # Specific field validations
    if 'system' in record and record['system'] != 'fol':
        violations.append(f"Line {line_num}: Field 'system' must be 'fol', got '{record['system']}'")

    if 'mode' in record and record['mode'] not in {'baseline', 'guided'}:
        violations.append(f"Line {line_num}: Field 'mode' must be 'baseline' or 'guided', got '{record['mode']}'")

    if 'method' in record:
        method = record['method']
        if method != 'fol-baseline' and not method.startswith('fol-guided@policy='):
            violations.append(f"Line {line_num}: Field 'method' must be 'fol-baseline' or 'fol-guided@policy=<shortsha8>', got '{method}'")
        elif method.startswith('fol-guided@policy='):
            policy = method.split('@policy=')[1]
            if not re.match(r'^[a-f0-9]{8}$', policy):
                violations.append(f"Line {line_num}: Policy must be 8-character hex string, got '{policy}'")

    if 'inserted_proofs' in record and isinstance(record['inserted_proofs'], (int, float)) and record['inserted_proofs'] <= 0:
        violations.append(f"Line {line_num}: Field 'inserted_proofs' must be > 0, got {record['inserted_proofs']}")

    if 'wall_minutes' in record and isinstance(record['wall_minutes'], (int, float)) and record['wall_minutes'] <= 0:
        violations.append(f"Line {line_num}: Field 'wall_minutes' must be > 0, got {record['wall_minutes']}")

    if 'merkle' in record:
        merkle = record['merkle']
        if not re.match(r'^[a-f0-9]{64}$', merkle):
            violations.append(f"Line {line_num}: Field 'merkle' must be 64-character hex string, got '{merkle}'")

    return violations


def _validate_legacy_record(record: Dict[str, Any], line_num: int) -> List[str]:
    """Validate legacy record and identify missing v1 fields."""
    violations = []

    # Check for missing v1 fields in legacy records
    missing_v1_fields = []
    v1_fields = {'system', 'mode', 'method', 'seed', 'inserted_proofs', 'wall_minutes', 'block_no', 'merkle'}

    for field in v1_fields:
        if field not in record:
            missing_v1_fields.append(field)

    if missing_v1_fields:
        violations.append(f"Line {line_num}: Legacy record missing v1 fields: {', '.join(missing_v1_fields)}")

    return violations


def main() -> int:
    """Main entry point for the metrics linter CLI."""
    if len(sys.argv) != 2:
        print("Usage: python tools/metrics_lint_v1.py <path_to_metrics_file>", file=sys.stderr)
        return 1

    path = sys.argv[1]
    result = lint_v1(path)

    # Print violations
    if result['violations']:
        for violation in result['violations']:
            print(violation, file=sys.stderr)

    # Check for mixed schema
    if result['is_mixed_schema']:
        print(f"mixed schema detected: {result['v1_count']} v1 records, {result['legacy_count']} legacy records. Use artifacts/wpv5/run_metrics_v1.jsonl", file=sys.stderr)
        return 2

    # Check for field violations
    if result['violations']:
        return 3

    # Success
    print(f"OK: {result['v1_count']} v1 records validated")
    return 0


if __name__ == '__main__':
    sys.exit(main())
