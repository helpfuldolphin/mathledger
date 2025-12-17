# What-If Telemetry Field Mapping

**Version**: 1.0.0
**Status**: FROZEN
**Last Updated**: 2025-01-01

## Overview

This document defines the canonical field mapping for What-If telemetry parsing. The mapping allows telemetry from various sources (P4, P5, First Light) to be normalized to What-If Engine input format.

## FROZEN CONTRACT

This mapping table is **FROZEN**. Changes require governance review.

- **Unknown fields are IGNORED** (not errors)
- **Aliases are resolved in order** (first match wins)
- **Missing fields use defaults** (safe defaults)

## Field Mapping Table

| Canonical Field | Accepted Aliases | Default | Type | Description |
|-----------------|------------------|---------|------|-------------|
| `cycle` | `cycle`, `cycle_num`, `step`, `iteration` | (required index) | `int` | Cycle number |
| `timestamp` | `timestamp`, `ts`, `time`, `created_at` | (current time) | `str` | ISO 8601 timestamp |
| `invariant_violations` | `invariant_violations`, `violations`, `invariant_errors`, `failed_invariants` | `[]` | `List[str]` | G2: List of invariant violation IDs |
| `in_omega` | `in_omega`, `in_safe_region`, `is_safe`, `omega_safe` | `True` | `bool` | G3: Whether state is in safe region Ω |
| `omega_exit_streak` | `omega_exit_streak`, `safe_region_exit_streak`, `outside_omega_cycles`, `omega_exit_cycles` | `0` | `int` | G3: Consecutive cycles outside Ω |
| `rho` | `rho`, `rsi`, `stability_index`, `stability` | `1.0` | `float` | G4: Relative Stability Index (0.0-1.0) |
| `rho_collapse_streak` | `rho_collapse_streak`, `rsi_streak`, `stability_collapse_streak`, `rho_low_streak` | `0` | `int` | G4: Consecutive cycles with ρ below threshold |

## Alias Resolution

Aliases are resolved in the order listed above. For example, if telemetry contains both `rho` and `rsi`, the value of `rho` is used (first match).

```python
# Resolution order for rho field:
# 1. rho           (canonical)
# 2. rsi           (alias)
# 3. stability_index (alias)
# 4. stability     (alias)
```

## Unknown Field Handling

Unknown fields are **silently ignored**. This ensures forward compatibility when telemetry sources add new fields.

```python
# Example: Unknown fields ignored
telemetry = {
    "cycle": 1,
    "rho": 0.85,
    "unknown_field": "value",  # Ignored, no error
    "future_metric": 42,       # Ignored, no error
}
```

## Type Coercion

The parser performs safe type coercion:

| Source Type | Target Type | Coercion |
|-------------|-------------|----------|
| `str` "true"/"false" | `bool` | Case-insensitive boolean parsing |
| `str` comma-separated | `List[str]` | Split on comma, trim whitespace |
| `int`/`float` | `float` | Standard numeric conversion |
| `None` | (default) | Use field default value |

## Examples

### Standard Telemetry (P5)

```json
{
    "cycle": 42,
    "timestamp": "2025-01-01T12:00:00Z",
    "invariant_violations": [],
    "in_omega": true,
    "omega_exit_streak": 0,
    "rho": 0.85,
    "rho_collapse_streak": 0
}
```

### Alternative Field Names (Legacy)

```json
{
    "step": 42,
    "ts": "2025-01-01T12:00:00Z",
    "violations": ["CDI-010"],
    "is_safe": false,
    "outside_omega_cycles": 15,
    "rsi": 0.35,
    "rsi_streak": 8
}
```

### Minimal Telemetry

```json
{
    "cycle": 1,
    "rho": 0.9
}
```

All other fields use defaults.

## Gate Mapping

Fields are mapped to governance gates as follows:

| Gate | Canonical Fields |
|------|------------------|
| G2 (Invariant) | `invariant_violations` |
| G3 (Safe Region) | `in_omega`, `omega_exit_streak` |
| G4 (Soft/RSI) | `rho`, `rho_collapse_streak` |

## Implementation Reference

The field mapping is implemented in:
- `scripts/generate_what_if_report.py` - `FIELD_MAP` constant
- `backend/governance/what_if_engine.py` - `WhatIfCycleInput.from_telemetry()`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-01-01 | Initial frozen mapping |
