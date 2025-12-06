# Phase II U2 Evidence Feed Integration

## Overview

This document describes how the U2 experiment orchestrator's evidence feed integrates with D3's `build_evidence_pack()` for governance and statistical analysis.

**CRITICAL**: This is PHASE II work. All artifacts are labeled "PHASE II — NOT USED IN PHASE I". No uplift claims are made until all gates (G1-G5) pass with documented 95% CI.

## Architecture

```
┌─────────────────────────────────────────┐
│  run_uplift_u2.py orchestrate           │
│                                         │
│  1. Calibration (optional)              │
│  2. Baseline run                        │
│  3. RFL run                             │
│  4. build_u2_run_summary()              │
│  5. summarize_u2_run_for_evidence()     │
└──────────────┬──────────────────────────┘
               │
               │ Outputs:
               │ - run_summary.json
               │ - evidence_summary.json
               │
               v
┌─────────────────────────────────────────┐
│  D3 Governance / build_evidence_pack()  │
│                                         │
│  Consumes evidence_summary.json to:     │
│  - Verify artifact completeness         │
│  - Check calibration status             │
│  - Determine bootstrap readiness        │
│  - Gate statistical analysis            │
└─────────────────────────────────────────┘
```

## U2 Run Summary Contract

The `build_u2_run_summary()` function produces a structured summary with:

- **schema_version**: "1.0.0"
- **slice_name**: Experiment slice name
- **mode**: "baseline" or "rfl"
- **calibration_used**: Boolean flag
- **cycles_requested**: Number of cycles requested
- **cycles_completed**: Number of cycles completed
- **paths**: Dict with paths to:
  - `baseline_jsonl`: Baseline experiment results
  - `rfl_jsonl`: RFL experiment results
  - `calibration_summary_json`: Calibration summary (if used)
  - `manifest_json`: Experiment manifest
- **determinism_verified**: Boolean flag (future: determinism checks)
- **label**: "PHASE II — NOT USED IN PHASE I"

### Example

```json
{
  "schema_version": "1.0.0",
  "slice_name": "slice_uplift_goal",
  "mode": "baseline",
  "calibration_used": true,
  "cycles_requested": 50,
  "cycles_completed": 50,
  "paths": {
    "baseline_jsonl": "/path/to/baseline.jsonl",
    "rfl_jsonl": "/path/to/rfl.jsonl",
    "calibration_summary_json": "/path/to/calibration.json",
    "manifest_json": "/path/to/manifest.json"
  },
  "determinism_verified": false,
  "label": "PHASE II — NOT USED IN PHASE I"
}
```

## Evidence Feed Contract

The `summarize_u2_run_for_evidence()` function produces a neutral summary for D3:

- **schema_version**: "1.0.0"
- **has_all_required_artifacts**: Boolean (baseline, rfl, manifest present)
- **calibration_ok**: Boolean (calibration present if required)
- **ready_for_bootstrap**: Boolean (all artifacts present and valid)
- **notes**: Neutral string describing run state
- **label**: "PHASE II — Evidence feed, no uplift claims"

### Example

```json
{
  "schema_version": "1.0.0",
  "has_all_required_artifacts": true,
  "calibration_ok": true,
  "ready_for_bootstrap": true,
  "notes": "All required artifacts present; ready for statistical analysis",
  "label": "PHASE II — Evidence feed, no uplift claims"
}
```

## D3 Integration

### Consuming Evidence Summary in build_evidence_pack()

```python
def build_evidence_pack(run_dir: Path) -> Dict[str, Any]:
    """
    Build evidence pack from U2 orchestrated run.
    
    This function should:
    1. Load evidence_summary.json from run_dir
    2. Verify ready_for_bootstrap is True
    3. Load baseline and RFL results
    4. Perform statistical analysis (bootstrap, CI)
    5. Check gates G1-G5
    6. Return evidence pack (no uplift claims until all gates pass)
    """
    evidence_summary_path = run_dir / "evidence_summary.json"
    
    if not evidence_summary_path.exists():
        raise FileNotFoundError("evidence_summary.json not found")
    
    with open(evidence_summary_path) as f:
        evidence = json.load(f)
    
    # Gate check: Verify readiness
    if not evidence["ready_for_bootstrap"]:
        notes = evidence["notes"]
        raise ValueError(f"Run not ready for bootstrap: {notes}")
    
    # Gate check: Verify calibration if required
    if not evidence["calibration_ok"]:
        raise ValueError("Calibration required but not present")
    
    # Gate check: Verify all artifacts present
    if not evidence["has_all_required_artifacts"]:
        raise ValueError("Missing required artifacts")
    
    # Load run summary for artifact paths
    run_summary_path = run_dir / "run_summary.json"
    with open(run_summary_path) as f:
        run_summary = json.load(f)
    
    # Load baseline and RFL results
    baseline_path = Path(run_summary["paths"]["baseline_jsonl"])
    rfl_path = Path(run_summary["paths"]["rfl_jsonl"])
    
    baseline_data = load_jsonl(baseline_path)
    rfl_data = load_jsonl(rfl_path)
    
    # Perform statistical analysis
    # (bootstrap, compute CI, check gates G1-G5)
    # ...
    
    # Build evidence pack (no uplift claims until all gates pass)
    evidence_pack = {
        "schema_version": "1.0.0",
        "run_summary": run_summary,
        "evidence_summary": evidence,
        "statistical_analysis": {
            # Results from bootstrap, CI, gates
        },
        "label": "PHASE II — Evidence pack, no uplift claims until all gates pass"
    }
    
    return evidence_pack
```

## Orchestrated Run Mode

### Usage

```bash
# Run orchestrated experiment with calibration
uv run python experiments/run_uplift_u2.py orchestrate \
  --slice slice_uplift_goal \
  --cycles 50 \
  --require-calibration \
  --out-dir artifacts/uplift_runs/run_001
```

### Output Structure

```
artifacts/uplift_runs/run_001/
├── calibration/
│   └── calibration_summary.json
├── baseline/
│   ├── uplift_u2_slice_uplift_goal_baseline.jsonl
│   └── uplift_u2_manifest_slice_uplift_goal_baseline.json
├── rfl/
│   ├── uplift_u2_slice_uplift_goal_rfl.jsonl
│   └── uplift_u2_manifest_slice_uplift_goal_rfl.json
├── run_summary.json
└── evidence_summary.json
```

## Guardrails

1. **No uplift claims**: All functions produce neutral summaries. No claims of uplift, improvement, or significance.
2. **Phase II labeling**: All outputs labeled "PHASE II — NOT USED IN PHASE I".
3. **Gate enforcement**: Evidence summary explicitly checks artifact completeness and calibration.
4. **Determinism**: All runs use deterministic seeds for reproducibility.
5. **Verification hooks**: `determinism_verified` flag for future determinism checks.

## Future Enhancements

1. **Determinism verification**: Add explicit determinism checks before marking `determinism_verified: true`
2. **Gate integration**: Add explicit G1-G5 gate checks to evidence summary
3. **Metric computation**: Add preliminary metric values (goal_hit, sparse_density, etc.) to run summary
4. **Bootstrap integration**: Direct integration with bootstrap statistical analysis
5. **Preregistration verification**: Verify run matches preregistration file

## Testing

See `tests/test_u2_orchestration.py` for comprehensive tests covering:

- Run summary contract
- Evidence feed contract
- Artifact completeness checks
- Calibration requirement checks
- No uplift claims verification
- Phase II labeling

All tests pass and verify the contract requirements.
