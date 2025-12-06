# Metrics V1 Linter

A lightweight, network-free CLI tool for validating metrics files against the v1 schema contract and detecting mixed schema feeds.

## Usage

```bash
python tools/metrics_lint_v1.py <path_to_metrics_file>
```

The linter expects a JSONL (JSON Lines) file where each line contains a JSON object representing a metrics record.

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | OK | No violations found, file is valid |
| 2 | Mixed Schema | File contains both v1 and legacy records |
| 3 | Bad Fields | File has validation errors (missing fields, wrong types, invalid values) |

## V1 Schema Requirements

A valid v1 record must contain all of the following fields:

### Required Fields

- **system**: Must be `"fol"`
- **mode**: Must be `"baseline"` or `"guided"`
- **method**: Must be `"fol-baseline"` or `"fol-guided@policy=<8-char-hex>"`
- **seed**: String value
- **inserted_proofs**: Integer > 0
- **wall_minutes**: Number (int or float) > 0
- **block_no**: Integer
- **merkle**: 64-character hexadecimal string

### Field Validation Rules

- **system**: Must exactly equal `"fol"`
- **mode**: Must be either `"baseline"` or `"guided"`
- **method**:
  - For baseline: must be `"fol-baseline"`
  - For guided: must be `"fol-guided@policy=<8-char-hex>"` where policy is 8-character hex string
- **inserted_proofs**: Must be a positive integer
- **wall_minutes**: Must be a positive number (integer or float)
- **merkle**: Must be exactly 64 hexadecimal characters (0-9, a-f)

## Remediation Map

### Common Issues and Solutions

#### Exit Code 2: Mixed Schema Detected

**Error Message:**
```
mixed schema detected: X v1 records, Y legacy records. Use artifacts/wpv5/run_metrics_v1.jsonl
```

**Solution:**
- Separate v1 and legacy records into different files
- Use `artifacts/wpv5/run_metrics_v1.jsonl` for v1 records only
- Ensure all records in a single file follow the same schema

#### Exit Code 3: Bad Fields

**Missing Required Fields:**
```
Line N: Missing required field 'field_name'
```

**Solution:**
- Add the missing field to the record
- Ensure all v1 records contain all required fields

**Wrong Field Types:**
```
Line N: Field 'field_name' must be int, got str
```

**Solution:**
- Convert the field value to the correct type
- Check that numeric fields contain numbers, not strings

**Invalid Field Values:**
```
Line N: Field 'system' must be 'fol', got 'invalid'
Line N: Field 'mode' must be 'baseline' or 'guided', got 'invalid'
Line N: Field 'inserted_proofs' must be > 0, got -1
```

**Solution:**
- Correct the field value according to validation rules
- Ensure numeric fields are positive
- Use valid enum values for constrained fields

**Invalid Format:**
```
Line N: Policy must be 8-character hex string, got 'invalid'
Line N: Field 'merkle' must be 64-character hex string, got 'invalid'
```

**Solution:**
- Use proper hexadecimal format (0-9, a-f)
- Ensure correct length for hex strings
- For guided method, use 8-character hex for policy

## Examples

### Valid V1 Record

```json
{
  "system": "fol",
  "mode": "baseline",
  "method": "fol-baseline",
  "seed": "abc12345",
  "inserted_proofs": 10,
  "wall_minutes": 5.5,
  "block_no": 123,
  "merkle": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
}
```

### Valid Guided V1 Record

```json
{
  "system": "fol",
  "mode": "guided",
  "method": "fol-guided@policy=12345678",
  "seed": "def67890",
  "inserted_proofs": 15,
  "wall_minutes": 7.2,
  "block_no": 456,
  "merkle": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567"
}
```

### Legacy Record (Missing V1 Fields)

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "duration": 300,
  "proofs": 5
}
```

## Testing

Run the test suite to verify the linter works correctly:

```bash
# Run all tests
python -m unittest tests.qa.test_metrics_lint_v1 -v

# Run tests quietly
python -m unittest tests.qa.test_metrics_lint_v1 -q
```

## ASCII-Only Compliance

All output from the linter is guaranteed to be ASCII-only:
- No Unicode characters in error messages
- No BOM (Byte Order Mark) in output
- No smart quotes or special characters
- Compatible with all text processing tools

## Network-Free Design

The linter operates entirely offline:
- No external API calls
- No network dependencies
- Uses only local file system access
- Suitable for CI/CD environments with restricted network access
