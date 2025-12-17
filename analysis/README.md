# README: Analysis Tools

This directory contains **EXPERIMENTAL** and **NON-CANONICAL** tools for data analysis.

## Tool: `replay_summary_generator.py`

### Purpose: Statistical Summary of Replay Data

This script aggregates individual component replay metrics and produces a single JSON file containing purely descriptive statistics (mean, median, min, max, std dev).

**CRITICAL: THIS IS NOT A GOVERNANCE TOOL.**
*   It contains **NO THRESHOLDS** and **NO GATING LOGIC**.
*   Its output is **ADVISORY ONLY** and must not be used to block CI/CD workflows or make automated promotion decisions.
*   It lives outside the formal `system_law` and governance process, serving only to collect data for human analysis and to inform future policy discussions.

### Usage

To run the `replay_summary_generator.py` tool:

```bash
python analysis/replay_summary_generator.py --input-dir <path-to-component-jsons> --output-file <path-to-output-summary.json>
```

### Example Output (`replay_run_summary.json` snippet)

```json
{
  "schema_version": "1.0",
  "mode": "ANALYSIS",
  "scope_note": "NOT_GOVERNANCE_NOT_GATING",
  "run_id": "local-run-1702390800",
  "timestamp_utc": "2023-12-12T10:20:00.123456Z",
  "summary_statistics": {
    "component_count": 2,
    "determinism_rate": {
      "mean": 99.5,
      "median": 99.5,
      "min": 99.0,
      "max": 100.0,
      "std_dev": 0.7071
    },
    "drift_metric": {
      "mean": 0.05,
      "median": 0.05,
      "min": 0.02,
      "max": 0.08,
      "std_dev": 0.0424
    }
  },
  "components": [
    {
      "name": "component-a",
      "determinism_rate": 100.0,
      "drift_metric": 0.02
    },
    {
      "name": "component-b",
      "determinism_rate": 99.0,
      "drift_metric": 0.08
    }
  ]
}
```

---
**This tool and its documentation MUST NOT be referenced by any file under `docs/system_law/`.**
