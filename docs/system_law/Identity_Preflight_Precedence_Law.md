# Identity Preflight Precedence Law

**Status:** PHASE X P5 PRE-FLIGHT
**Version:** 1.0.0

## Source Precedence Order

When multiple identity preflight sources exist, the following precedence
applies (highest to lowest):

1. **Dedicated artifact file** (`p5_identity_preflight.json`)
2. **CLI-provided report** (`--p5-identity-report`)
3. **Legacy separate file** (`identity_preflight.json`)
4. **Embedded in run_config** (`run_config.json` → `identity_preflight`)

## Detection Priority for Evidence Pack

```
detect_identity_preflight(run_dir):
  1. p5_identity_preflight.json  → return if exists
  2. identity_preflight.json     → return if exists
  3. run_config.json embedded    → return if exists
  4. None
```

## CLI Authority Override

When `--p5-identity-report` is explicitly provided, it takes precedence
over evidence pack detection for status generation. Rationale:

- CLI report represents operator's explicit intent
- Enables re-analysis with updated configs without regenerating pack
- Supports audit replay with specific report versions

If CLI report differs from evidence pack, a mismatch warning is emitted
but CLI remains authoritative. This is advisory only—no gating occurs.

## SHADOW MODE CONTRACT

All precedence logic is observational. Regardless of which source wins:
- No control flow is modified
- No governance decisions are gated
- Warnings are informational only
