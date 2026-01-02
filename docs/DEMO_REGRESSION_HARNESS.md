# UVIL v0 Demo Regression Harness

This document describes how to run the demo regression harness and capture fixture outputs.

---

## Quick Start

### 1. Start the backend

```bash
uv run python demo/app.py
```

This starts the UVIL API at `http://localhost:8000`.

### 2. Run the regression harness

In a separate terminal:

```bash
uv run python tools/run_demo_cases.py
```

This runs all 5 demo cases and writes outputs to `fixtures/`.

### 3. View results

Fixtures are saved to:
```
fixtures/
  mv_only/
    input.json    # Case definition
    output.json   # API responses + attestation
  mixed_mv_adv/
    ...
  pa_only/
    ...
  adv_only/
    ...
  underdetermined_navier_stokes/
    ...
```

---

## Available Cases

| Case | Description | Expected Behavior |
|------|-------------|-------------------|
| `mv_only` | Pure MV claim | 1 authority-bearing claim, enters R_t |
| `mixed_mv_adv` | MV + ADV claims | MV enters R_t, ADV excluded |
| `pa_only` | User attestation (PA) | 1 authority-bearing claim |
| `adv_only` | All ADV claims | 0 authority-bearing, R_t empty, ABSTAINED |
| `underdetermined_navier_stokes` | Open problem | All ADV, system abstains |

---

## Running Specific Cases

```bash
# List available cases
uv run python tools/run_demo_cases.py --list

# Run specific case(s)
uv run python tools/run_demo_cases.py --case mv_only
uv run python tools/run_demo_cases.py --case adv_only --case mixed_mv_adv

# Use different API URL
uv run python tools/run_demo_cases.py --url http://localhost:9000
```

---

## What Gets Captured

For each case, `output.json` contains:

- **exploration**: Draft proposal (proposal_id is exploration-only)
- **authority**: Committed partition snapshot
- **verification**: Verification response with outcome
- **attestation**: Final U_t, R_t, H_t values
- **analysis**: Claim counts, ADV exclusion check

---

## Invariant Checks

The harness verifies:

1. **ADV exclusion**: ADV claims never enter R_t
2. **Authority count**: Matches expected count per case
3. **Determinism**: Same inputs produce same outputs (compare across runs)

If an invariant is violated, the summary shows `[!!]`.

---

## Using Fixtures for Regression

Once you have baseline fixtures:

1. Make changes to the demo/partitioner/API
2. Re-run the harness
3. Diff the output files to detect changes

```bash
# Compare outputs after a change
diff fixtures/mv_only/output.json fixtures/mv_only/output.json.bak
```

Changes to U_t, R_t, or H_t indicate behavioral drift and should be investigated.
