# MathLedger Metrics V1 Linter Message Specification

## Overview
This document specifies the error and warning messages produced by the MathLedger Metrics V1 Linter. All messages are ASCII-only and designed for stable CI/CD integration.

## Message Format
All messages follow the pattern: `CATEGORY: Description`

## Error Messages

### File/Input Errors
- `EMPTY_FILE: Input file is empty`
- `FILE_NOT_FOUND: Input file does not exist`
- `INVALID_INPUT: Expected dict, JSON string, or file path`
- `JSON_PARSE_ERROR: Invalid JSON - {details}`

### Structure Errors
- `MISSING_FIELD: Required field '{field_name}' is missing`
- `MISSING_FIELD: statements.{field_name} is missing`
- `MISSING_FIELD: proofs.{field_name} is missing`
- `MISSING_FIELD: blocks.{field_name} is missing`

### Type Errors
- `TYPE_ERROR: statements.{field_name} must be integer, got {actual_type}`
- `TYPE_ERROR: proofs.{field_name} must be numeric, got {actual_type}`
- `TYPE_ERROR: blocks.{field_name} must be integer, got {actual_type}`
- `WALL_MINUTES_TYPE_ERROR: {path} must be numeric, got {actual_type}`
- `SEED_TYPE_ERROR: {path} must be string or integer, got {actual_type}`

### Range Errors
- `RANGE_ERROR: statements.total cannot be negative`
- `RANGE_ERROR: proofs.success_rate must be between 0 and 100`
- `WALL_MINUTES_RANGE_ERROR: {path} cannot be negative`

### Hash Validation Errors
- `MERKLE_LENGTH_ERROR: {path} hash length is {actual_length}, expected 64`

### Unexpected Errors
- `UNEXPECTED_ERROR: {error_details}`

## Warning Messages

### Legacy Format Warnings
- `LEGACY_FORMAT: Detected legacy format indicator`
- `LEGACY_FIELD: Found legacy field '{field_name}'`

### Range Warnings
- `RANGE_WARNING: statements.total is very large`

## Message Stability
- All messages are ASCII-only (no Unicode characters)
- Message format is consistent and stable across runs
- Error codes are prefixed for easy filtering
- Path information is included for nested validation errors

## Usage in CI/CD
Messages can be filtered by category:
- `grep "ERROR:"` - Show only errors
- `grep "WARNING:"` - Show only warnings
- `grep "MISSING_FIELD"` - Show only missing field errors
- `grep "TYPE_ERROR"` - Show only type errors

## Examples

### Valid Metrics
```json
{
  "statements": {"total": 1000, "axioms": 100, "derived": 900, "max_depth": 10},
  "proofs": {"by_status": {"success": 800}, "by_prover": {"lean": 1000}, "recent_hour": 50, "success_rate": 80.0},
  "derivation_rules": {"modus_ponens": 500},
  "blocks": {"total": 25, "latest_id": 24},
  "lemmas": {"total": 150, "top": []},
  "throughput": {"proofs_per_min": 2.5, "statements_per_min": 5.0},
  "frontier": {"depth_max": 15, "queue_backlog": 5},
  "failures_by_class": {"LEAN_TIMEOUT": 10, "LEAN_COMPILE": 5, "LEAN_RUNTIME": 3, "DERIVATION_ERROR": 2, "OTHER": 1},
  "queue": {"length": 12}
}
```

### Common Error Cases
1. **Empty file**: `EMPTY_FILE: Input file is empty`
2. **Missing field**: `MISSING_FIELD: Required field 'statements' is missing`
3. **Type error**: `TYPE_ERROR: statements.total must be integer, got str`
4. **Range error**: `RANGE_ERROR: proofs.success_rate must be between 0 and 100`
5. **Hash length**: `MERKLE_LENGTH_ERROR: root_hash hash length is 10, expected 64`
6. **Wall minutes**: `WALL_MINUTES_TYPE_ERROR: wall_minutes must be numeric, got bool`
7. **Seed type**: `SEED_TYPE_ERROR: seed must be string or integer, got dict`
