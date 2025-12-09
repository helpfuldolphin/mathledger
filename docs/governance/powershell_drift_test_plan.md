# PowerShell Drift Tests Completion Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **DRAFT**

## 1. Overview

This document provides a comprehensive plan to implement the **PowerShell Drift Tests**. The initial repository analysis (`drift_table_summary.md`) revealed that while Python and JavaScript interop tests were complete, the PowerShell tests were marked as `ABSTAIN`. This represents a significant gap in our cross-language drift detection capabilities.

This plan specifies the test catalog, API schemas, execution harness, and drift classification mapping required to close this gap. The goal is to create a robust CI gate that detects subtle type coercion and serialization differences between Python (our backend language) and PowerShell (a primary client for scripting and automation).

## 2. Test Catalog

The test suite will focus on data types that are notoriously problematic in cross-language JSON serialization.

| Test ID | Data Type | Test Description | Expected Python Value | Expected PowerShell Value |
|---|---|---|---|---|
| `PS-DRIFT-01` | **Integer** | Verify that a standard integer is correctly serialized and deserialized. | `123` | `123` |
| `PS-DRIFT-02` | **Large Integer** | Verify that a 64-bit integer (long) is handled correctly without precision loss. | `9007199254740991` | `9007199254740991` |
| `PS-DRIFT-03` | **Float** | Verify that a standard float is correctly serialized and deserialized. | `123.456` | `123.456` |
| `PS-DRIFT-04` | **Boolean (True)** | Verify that a boolean `true` is handled correctly. | `true` | `$true` |
| `PS-DRIFT-05` | **Boolean (False)** | Verify that a boolean `false` is handled correctly. | `false` | `$false` |
| `PS-DRIFT-06` | **Null/None** | Verify that a `null` value is correctly mapped to `$null`. | `null` | `$null` |
| `PS-DRIFT-07` | **String** | Verify that a standard string is handled correctly. | `"hello world"` | `"hello world"` |
| `PS-DRIFT-08` | **Empty String** | Verify that an empty string is handled correctly. | `""` | `""` |
| `PS-DRIFT-09` | **Date/Time** | Verify that an ISO 8601 timestamp is correctly serialized and deserialized as a string. | `"2025-12-06T12:00:00Z"` | `"2025-12-06T12:00:00Z"` |
| `PS-DRIFT-10` | **Array of Primitives** | Verify that an array of mixed primitives is handled correctly. | `[1, "two", 3.0, true]` | `@(1, "two", 3.0, $true)` |
| `PS-DRIFT-11` | **Nested Object** | Verify that a nested JSON object is handled correctly. | `{"a": {"b": 1}}` | `@{a=@{b=1}}` |

## 3. API Schemas

A lightweight Python test server will expose a single API endpoint (`/echo`) that accepts a JSON object and returns it. This allows us to test the round-trip serialization and deserialization.

### Request Schema (`/echo` - POST)

```json
{
  "type": "object",
  "properties": {
    "test_id": { "type": "string" },
    "payload": { "type": "object" }
  },
  "required": ["test_id", "payload"]
}
```

### Response Schema (`/echo` - POST)

```json
{
  "type": "object",
  "properties": {
    "test_id": { "type": "string" },
    "payload": { "type": "object" },
    "server_language": { "type": "string", "enum": ["python"] }
  }
}
```

## 4. Execution Harness

The test harness consists of two main components:

1.  **Python Test Server** (`scripts/testing/ps_drift_server.py`): A simple Flask or FastAPI server that implements the `/echo` endpoint.
2.  **PowerShell Test Script** (`scripts/testing/run_ps_drift_tests.ps1`): The main test runner.

### Execution Flow

1.  The CI workflow starts the Python test server in the background.
2.  The CI workflow then executes the `run_ps_drift_tests.ps1` script.
3.  The PowerShell script iterates through the test catalog.
4.  For each test:
    a. It constructs a PowerShell object (e.g., `@{ test_id = "PS-DRIFT-01"; payload = @{ value = 123 } }`).
    b. It converts this object to a JSON string using `ConvertTo-Json`.
    c. It sends this JSON to the Python server's `/echo` endpoint using `Invoke-RestMethod`.
    d. It receives the JSON response from the server.
    e. It compares the returned `payload` with the original payload, checking for both type and value equality.
5.  The script generates a JSON report of all test results (`ps_drift_report.json`).
6.  If any test fails, the script exits with a non-zero status code.

## 5. Drift Classification Mapping

Test failures will be mapped to drift severities to determine the CI action.

| Failure Type | Example | Drift Severity | CI Action |
|---|---|---|---|
| **Type Mismatch** | Python returns `123` (int), PowerShell receives it as a string `"123"`. | **CRITICAL** | Block Merge |
| **Value Mismatch** | Python returns `9007199254740991`, PowerShell receives `9007199254740992` (precision loss). | **CRITICAL** | Block Merge |
| **Null/None Mismatch** | Python returns `null`, PowerShell receives an empty string `""`. | **CRITICAL** | Block Merge |
| **Array Order Mismatch** | Python returns `[1, 2]`, PowerShell receives `[2, 1]`. | **WARNING** | Allow with Review |
| **Missing Key** | Python returns `{"a": 1, "b": 2}`, PowerShell receives `{"a": 1}`. | **CRITICAL** | Block Merge |

## 6. CI Integration

A new workflow, `gate-powershell-drift.yml`, will be created to run these tests.

```yaml
# .github/workflows/gate-powershell-drift.yml
name: 'Gate: PowerShell Drift'

on:
  pull_request:
    paths:
      - 'backend/**.py' # Run on any backend changes
      - 'scripts/testing/run_ps_drift_tests.ps1'

jobs:
  test-powershell-interop:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Install Python dependencies
        run: pip3 install flask

      - name: Start Test Server
        run: |
          python3 scripts/testing/ps_drift_server.py &
          sleep 3 # Give server time to start

      - name: Run PowerShell Drift Tests
        shell: pwsh
        run: |
          ./scripts/testing/run_ps_drift_tests.ps1

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: powershell-drift-report
          path: ps_drift_report.json
          retention-days: 7
```

This plan provides a clear path to completing the PowerShell drift tests, closing a critical gap in our governance and ensuring the stability of the MathLedger ecosystem across all supported client languages.
