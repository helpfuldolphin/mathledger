# Documentation Governance Layer

## Overview

The Documentation Governance Layer provides automated risk signals, evidence pack integration, and uplift-safe narrative management for MathLedger documentation.

## Core Functions

### `build_docs_governance_snapshot()`

Aggregates lint and validation reports into a unified governance snapshot.

**Inputs:**
- `snippet_report`: Output from `snippet_check.py` with code snippet validation
- `phase_marker_report`: Output from `phase_marker_lint.py` with phase marker validation  
- `toc_index`: Output from `generate_evidence_pack_toc.py` with TOC metadata

**Output:**
```json
{
  "schema_version": "1.0.0",
  "doc_count": 45,
  "docs_with_invalid_snippets": ["docs/U2_PORT_PLAN.md"],
  "docs_missing_phase_markers": ["docs/VSD_PHASE_2.md"],
  "docs_with_uplift_mentions_without_disclaimer": ["docs/U2_PORT_PLAN.md"],
  "evidence_docs_covered": 42
}
```

All fields are deterministic and JSON-safe.

### `evaluate_uplift_safety()`

Pure function to assess documentation uplift safety based on governance snapshot.

**Input:**
- `snapshot`: Output from `build_docs_governance_snapshot()`

**Output:**
```json
{
  "uplift_safe": false,
  "issues": [
    "docs/U2_PORT_PLAN.md mentions uplift but lacks disclaimer"
  ],
  "status": "BLOCK"
}
```

**Status Values:**
- `OK`: No issues found
- `WARN`: Non-blocking issues (invalid snippets, missing markers)
- `BLOCK`: Blocking issues (uplift mentions without disclaimer)

**Important:** This function only interprets documentation structure and disclaimers, NOT experimental results.

### `build_docs_section_for_evidence_pack()`

Formats governance data for Evidence Pack v2 integration.

**Inputs:**
- `snapshot`: Output from `build_docs_governance_snapshot()`
- `uplift_safety`: Output from `evaluate_uplift_safety()`

**Output:**
```json
{
  "docs_governance_ok": false,
  "docs_with_phase_issues": ["docs/VSD_PHASE_2.md"],
  "docs_with_snippet_issues": ["docs/U2_PORT_PLAN.md"],
  "uplift_safe": false
}
```

## CLI Usage

### Basic Usage

```bash
python3 docs/docs_governance.py \
  --snippet-report path/to/snippet_report.json \
  --phase-marker-report path/to/phase_marker_report.json \
  --toc-index path/to/toc_index.json \
  --output docs_governance_report.json
```

### Exit Codes

- `0`: Status OK (no issues)
- `1`: Status WARN (non-blocking issues)
- `2`: Status BLOCK (blocking issues)

### Example Output

```
Docs governance report written to docs_governance_report.json
Status: BLOCK
Issues found: 4
  - docs/PHASE2_RFL_UPLIFT_PLAN.md contains invalid code snippets
  - docs/U2_PORT_PLAN.md contains invalid code snippets
  - docs/U2_PORT_PLAN.md mentions uplift but lacks disclaimer
  - docs/VSD_PHASE_2.md missing required phase markers
```

## Integration with Evidence Packs

The governance layer output can be embedded directly into Evidence Pack v2:

```python
from docs.docs_governance import (
    build_docs_governance_snapshot,
    evaluate_uplift_safety,
    build_docs_section_for_evidence_pack
)

# Build snapshot from lint reports
snapshot = build_docs_governance_snapshot(
    snippet_report, phase_marker_report, toc_index
)

# Evaluate uplift safety
uplift_safety = evaluate_uplift_safety(snapshot)

# Get evidence pack section
evidence_section = build_docs_section_for_evidence_pack(
    snapshot, uplift_safety
)

# Add to evidence pack
evidence_pack["docs_governance"] = evidence_section
```

## Sober Truth Guardrails

The governance layer enforces key invariants:

1. **No Normative Vocabulary**: Issue descriptions use neutral, descriptive language
2. **Structure Only**: Evaluates documentation structure and disclaimers, NOT experimental results
3. **Uplift Safety**: Blocks documentation that mentions uplift without proper disclaimers
4. **Deterministic Output**: All functions produce deterministic, JSON-safe output

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_docs_governance.py -v
```

Test coverage includes:
- Snapshot building from lint outputs
- Uplift safety evaluation logic
- Evidence pack section generation
- Deterministic JSON output
- Normative vocabulary absence
- Full pipeline integration

## Input Report Schemas

### Snippet Report Schema

```json
{
  "invalid_files": ["docs/file1.md", "docs/file2.md"],
  "total_files_checked": 45
}
```

### Phase Marker Report Schema

```json
{
  "docs_missing_markers": ["docs/file1.md"],
  "docs_with_uplift_mentions_without_disclaimer": ["docs/file2.md"],
  "total_docs_checked": 50
}
```

### TOC Index Schema

```json
{
  "evidence_docs": ["docs/file1.md", "docs/file2.md"],
  "toc_version": "1.0.0"
}
```

## Design Principles

1. **Pure Functions**: All core functions are pure with no side effects
2. **Deterministic**: Output is always identical given identical input
3. **JSON-Safe**: All data structures serialize cleanly to JSON
4. **Composable**: Functions can be used independently or in pipeline
5. **Neutral Language**: No value judgments in issue descriptions
6. **Uplift Safety First**: Prioritizes detection of unsafe uplift claims

## Future Extensions

Potential enhancements:
- Integration with CI/CD pipelines
- Automated remediation suggestions
- Historical tracking of governance metrics
- Cross-reference validation
- Phase boundary consistency checks
