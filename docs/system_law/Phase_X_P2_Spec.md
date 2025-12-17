# Phase X P2 Specification: USLA Observability Integration

**Status**: Implemented
**Phase**: X (SHADOW MODE ONLY)
**Version**: 1.0.0
**Date**: 2025-12-09

---

## 1. Purpose

Phase X P2 integrates the USLA (Unified System Law Abstraction) health tile into the global health surface for observability purposes. This is a **SHADOW MODE ONLY** implementation.

### SHADOW MODE CONTRACT

The following invariants are strictly maintained:

1. **No Governance Modification**: The USLA tile NEVER influences any governance decisions
2. **Purely Observational**: All USLA outputs are for logging and monitoring only
3. **No Control Flow Dependency**: No system behavior depends on USLA tile contents
4. **No Abort Logic**: No automatic rollbacks or aborts are enabled
5. **Reversible**: The integration can be disabled via `USLA_SHADOW_ENABLED=false`

---

## 2. Architecture

### 2.1 Components

```
┌─────────────────────────────────────────────────────────────┐
│                    GlobalHealthSurface                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   dynamics   │  │     usla     │  │   (others)   │      │
│  │    tile      │  │    tile      │  │              │      │
│  │              │  │  (SHADOW)    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                           │                                  │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │ USLAHealthTileProducer │                     │
│              │    (pure, read-only)   │                     │
│              └────────────────────────┘                     │
│                           │                                  │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │    USLAIntegration     │                     │
│              │     (SHADOW mode)      │                     │
│              └────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. `USLAIntegration` captures telemetry during runner cycles (SHADOW mode)
2. `USLAHealthTileProducer` reads state and produces JSON-serializable dict
3. `GlobalHealthSurface.build_global_health_surface()` attaches tile under `usla` key
4. Dashboard/console displays tile for observability

**Critical**: No control flow or decision logic depends on the USLA tile.

---

## 3. JSON Schema

### 3.1 USLA Health Tile Schema

```json
{
  "schema_version": "1.0.0",
  "tile_type": "usla_health",
  "timestamp": "2025-12-09T12:00:00.000000+00:00",
  "mode": "SHADOW",
  "cycle": 42,
  "state_summary": {
    "H": 0.75,
    "rho": 0.85,
    "tau": 0.21,
    "beta": 0.05,
    "J": 2.5,
    "C": "CONVERGING",
    "Gamma": 0.88
  },
  "hard_mode_status": "OK",
  "safe_region": {
    "within_omega": true
  },
  "active_cdis": [],
  "invariant_violations": [],
  "delta": 0,
  "divergence_summary": {
    "governance_aligned": true,
    "consecutive_divergence": 0,
    "max_severity": "NONE"
  },
  "headline": "Topology stable; monitoring active",
  "alerts": []
}
```

### 3.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version (currently "1.0.0") |
| `tile_type` | string | Always "usla_health" |
| `timestamp` | string | ISO 8601 timestamp |
| `mode` | string | Always "SHADOW" (Phase X) |
| `cycle` | int | Current cycle number |
| `state_summary.H` | float | Homological Stability Score [0, 1] |
| `state_summary.rho` | float | Rolling Stability Index [0, 1] |
| `state_summary.tau` | float | Effective threshold |
| `state_summary.beta` | float | Block rate |
| `state_summary.J` | float | Jacobian sensitivity |
| `state_summary.C` | string | Convergence class |
| `state_summary.Gamma` | float | TGRS score [0, 1] |
| `hard_mode_status` | string | "OK" or "FAIL" |
| `safe_region.within_omega` | bool | Whether state is in safe region Ω |
| `active_cdis` | array | List of active CDI codes |
| `invariant_violations` | array | List of violated invariants |
| `delta` | int | CDI defect count |
| `divergence_summary` | object | Divergence monitoring summary |
| `headline` | string | Human-readable status |
| `alerts` | array | Recent divergence alerts |

### 3.3 Headline Values

| Headline | Condition |
|----------|-----------|
| "Topology stable; monitoring active" | Nominal state |
| "Topology degraded; active CDIs detected" | Active CDIs present |
| "Topology degraded; invariant violations detected" | Invariants violated |
| "HARD mode inactive; system outside safe region" | HARD mode failed |
| "Governance divergence detected; monitoring continues" | Sim/real divergence |
| "CRITICAL: Multiple topology anomalies detected" | 3+ issues |

---

## 4. Integration API

### 4.1 Setting Up USLA Producer

```python
from backend.health import set_usla_producer, clear_usla_producer
from backend.topology.usla_health_tile import USLAHealthTileProducer
from backend.topology.usla_integration import USLAIntegration, RunnerType

# Create integration (SHADOW mode)
integration = USLAIntegration.create_for_runner(
    runner_type=RunnerType.RFL,
    runner_id="my_runner",
    enabled=True,  # SHADOW mode
)

# Create and install producer
producer = USLAHealthTileProducer()
set_usla_producer(producer, integration)

# Later: clear when done
clear_usla_producer()
```

### 4.2 Building Global Health Surface

```python
from backend.health import build_global_health_surface

# Build payload (USLA tile auto-attached if SHADOW mode enabled)
payload = build_global_health_surface()

# Access USLA tile (may be None if SHADOW mode disabled)
usla_tile = payload.get("usla")
```

### 4.3 Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `USLA_SHADOW_ENABLED` | `true`, `false` | Enable/disable USLA shadow mode |

---

## 5. CI Requirements

### 5.1 Test File

`tests/ci/test_usla_health_tile_serializes.py`

### 5.2 Test Cases

1. **`test_usla_tile_serializes_without_error`**: Primary gate test
   - Creates mock integration (enabled=True)
   - Produces tile
   - Verifies `isinstance(tile, dict)`
   - Verifies `json.dumps(tile)` succeeds

2. **`test_usla_tile_has_required_fields`**: Schema validation
3. **`test_usla_tile_state_summary_structure`**: Structure validation
4. **`test_disabled_integration_returns_none`**: Disabled behavior

### 5.3 Workflow

`.github/workflows/usla-shadow-gate.yml`

```yaml
- name: Verify USLA Health Tile
  if: env.USLA_SHADOW_ENABLED == 'true'
  run: uv run pytest tests/ci/test_usla_health_tile_serializes.py -v
```

---

## 6. Safety Guarantees

### 6.1 What This Integration Does

- Produces observability data for dashboards
- Logs simulator state for offline analysis
- Enables divergence monitoring between real and simulated governance

### 6.2 What This Integration Does NOT Do

- Modify any governance decisions
- Block or allow any cycles based on simulator output
- Trigger aborts or rollbacks
- Influence any other tiles or health classifications
- Change any runner behavior

### 6.3 Failure Modes

If the USLA tile producer fails:
1. The exception is caught silently
2. The `usla` key is omitted from the payload
3. All other tiles continue to function normally
4. No governance decisions are affected

---

## 7. Files Modified/Created

### 7.1 Modified

- `backend/health/global_surface.py`: Added USLA tile attachment
- `backend/health/__init__.py`: Exported new functions

### 7.2 Created

- `tests/ci/__init__.py`: CI test package
- `tests/ci/test_usla_health_tile_serializes.py`: Serialization tests
- `.github/workflows/usla-shadow-gate.yml`: CI workflow
- `docs/system_law/Phase_X_P2_Spec.md`: This document

---

## 8. Future Phases

### Phase X P3 (Not Yet Authorized)

- First-Light experiment harness implementation
- 1000-cycle shadow validation
- Red-flag abort condition monitoring (logging only)

### Phase XI (Not Yet Authorized)

- ACTIVE mode consideration
- Governance integration (requires separate authorization)

---

## 9. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-09 | Initial Phase X P2 implementation |
