# Replay Governance Phase X Binding

**Status:** DESIGN SPECIFICATION
**Classification:** INTERNAL - Architect-Level
**Date:** 2025-12-10
**Dependencies:** Phase_X_P3_Spec.md, Phase_X_P4_Spec.md, Phase_X_Prelaunch_Review.md

---

## 1. Overview

This document specifies how the Replay Governance Layer integrates with Phase X (P3/P4) safety doctrine. It defines:

1. How replay signals feed P3/P4 safety validation
2. Replay-to-GovernanceSignal mapping rules
3. Replay requirements for whitepaper evidence
4. Integration points for global console and CLAUDE I governance fusion

---

## 2. Replay Signals and P3/P4 Safety Doctrine

### 2.1 P3 Synthetic Validation (Wind Tunnel)

In P3 (synthetic first-light experiments), replay signals validate the **internal consistency** of the governance model:

| Replay Signal | P3 Role | Validation Purpose |
|---------------|---------|-------------------|
| `determinism_score` | Verify synthetic traces reproduce identically | Internal consistency |
| `hash_match_rate` | Ensure H_t hash threading is intact | Provenance chain integrity |
| `violation_count` | Count synthetic anomalies | Anomaly detection coverage |
| `coverage_pct` | Measure replay coverage of synthetic space | Test completeness |

**P3 Contract:**
- Replay verification runs in **SHADOW mode** (observational only)
- No control flow depends on replay signals
- Violations are logged but never enforced
- Evidence is collected for whitepaper package

### 2.2 P4 Shadow-Mode Validation (Flight Test)

In P4 (real-runner shadow coupling), replay signals validate **model-reality correspondence**:

| Replay Signal | P4 Role | Validation Purpose |
|---------------|---------|-------------------|
| `twin_divergence_detected` | Twin-vs-reality mismatch | Model fidelity |
| `h_t_drift_detected` | H_t hash drift from expected | Provenance integrity |
| `governance_alignment` | Safety-radar alignment status | Conflict detection |
| `safe_for_promotion` | Promotion gate signal | Phase transition safety |

**P4 Contract:**
- Shadow-mode invariant: No control paths from P4 to real runner
- Replay signals are **read-only observations**
- Divergence logs are append-only, tamper-evident
- Conflict detection triggers manual review, not automatic abort

---

## 3. Replay-to-GovernanceSignal Mapping Rules

### 3.1 Signal Collapse Algorithm

The `to_governance_signal_for_replay_safety()` function collapses Safety and Radar evaluations into a unified governance signal:

```
INPUTS:
  safety_eval: {status, reasons, safe_for_policy_update, safe_for_promotion}
  radar_eval:  {status, reasons, governance_alignment, conflict}

OUTPUT:
  governance_signal: {status, governance_status, reasons, governance_alignment, conflict}

COLLAPSE RULES:
  1. IF safety_status == BLOCK OR radar_status == BLOCK:
       status = BLOCK
  2. ELSE IF governance_alignment == DIVERGENT:
       status = BLOCK
  3. ELSE IF conflict == TRUE:
       status = BLOCK
  4. ELSE IF safety_status == WARN OR radar_status == WARN:
       status = WARN
  5. ELSE IF governance_alignment == TENSION:
       status = WARN
  6. ELSE:
       status = OK

REASON PREFIXING:
  - Safety reasons prefixed with "[Safety]"
  - Radar reasons prefixed with "[Radar]"
  - Conflict reasons prefixed with "[CONFLICT]"
  - Already-prefixed reasons are not double-prefixed
```

### 3.2 Signal Field Mapping

| Source Field | GovernanceSignal Field | Type |
|--------------|----------------------|------|
| `safety_eval.status` | `safety_status` | enum(ok,warn,block) |
| `radar_eval.status` | `governance_status` | enum(ok,warn,block) |
| Computed | `status` | enum(ok,warn,block) |
| `radar_eval.governance_alignment` | `governance_alignment` | enum(aligned,tension,divergent) |
| `radar_eval.conflict` | `conflict` | boolean |
| Merged | `reasons` | string[] |
| `safety_eval.safe_for_policy_update` | `safe_for_policy_update` | boolean |
| `safety_eval.safe_for_promotion` | `safe_for_promotion` | boolean |

### 3.3 Invariant: Signal Type Identifier

All replay governance signals MUST include:
```json
{
  "signal_type": "replay_safety"
}
```

This identifier enables CLAUDE I governance fusion to route signals correctly.

---

## 4. Replay Requirements for Whitepaper Evidence

### 4.1 Required Artifacts

| Artifact | Schema | Producer | Whitepaper Section |
|----------|--------|----------|-------------------|
| `replay_governance_radar.json` | `replay_governance_radar.schema.json` | `build_replay_safety_governance_view()` | Appendix: Governance Evidence |
| `replay_promotion_eval.json` | `replay_promotion_eval.schema.json` | `evaluate_replay_safety_for_promotion()` | Section: Promotion Gates |
| `replay_director_panel.json` | `replay_director_panel.schema.json` | `build_replay_safety_director_panel()` | Executive Summary |
| `replay_global_console_tile.json` | `replay_global_console_tile.schema.json` | Global console builder | Appendix: Console Artifacts |

### 4.2 Evidence Requirements

| Requirement | Field | Threshold | Rationale |
|-------------|-------|-----------|-----------|
| Determinism | `determinism_score` | >= 95 | Replay must be deterministic |
| Hash Integrity | `hash_match_rate` | == 1.0 | All hashes must match |
| No Critical Violations | `violation_count` (critical) | == 0 | No critical violations |
| Coverage | `coverage_pct` | >= 90% | High test coverage |
| Alignment | `governance_alignment` | != divergent | No unresolved conflicts |

### 4.3 Evidence Package Structure

```
evidence/
  replay/
    replay_governance_radar.json       # Radar view
    replay_promotion_eval.json         # Promotion evaluation
    replay_director_panel.json         # Director panel
    replay_global_console_tile.json    # Console tile
    replay_evidence_hash.txt           # SHA-256 of evidence bundle
```

---

## 5. Global Console Tile Registration

### 5.1 Tile Registration Pattern

The replay governance tile follows the established pattern in `backend/health/global_surface.py`:

```python
# SHADOW MODE CONTRACT (replay_governance tile):
# - The replay_governance tile does NOT influence any other tiles
# - No control flow depends on the replay_governance tile
# - The replay_governance tile is purely for observability
```

### 5.2 Tile Dependencies

| Dependency | Required | Description |
|------------|----------|-------------|
| `replay_safety_envelope` | Yes | From `build_replay_safety_envelope()` |
| `replay_promotion_eval` | Yes | From `evaluate_replay_safety_for_promotion()` |
| `replay_governance_view` | Yes | From `build_replay_safety_governance_view()` |
| `replay_director_panel` | Optional | From `build_replay_safety_director_panel()` |

### 5.3 Attachment Function Signature

```python
def attach_replay_governance_tile(
    payload: MutableMapping[str, Any],
    envelope: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    governance_view: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach replay governance tile to global health surface (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The replay_governance tile does NOT influence any other tiles
    - No control flow depends on the replay_governance tile contents
    """
```

---

## 6. CLAUDE I Governance Signal Fusion

### 6.1 Signal Registry Entry

The replay governance signal is registered in the CLAUDE I governance signal registry:

| Signal Type | Producer | Consumer | Fusion Rule |
|-------------|----------|----------|-------------|
| `replay_safety` | Replay Governance Layer | CLAUDE I Arbiter | Conservative (BLOCK propagates) |

### 6.2 Fusion Rules

When CLAUDE I fuses multiple governance signals:

1. **BLOCK Propagation:** If `replay_safety.status == BLOCK`, overall governance status is BLOCK
2. **WARN Aggregation:** If `replay_safety.status == WARN`, contributes to WARN count
3. **Conflict Escalation:** If `replay_safety.conflict == true`, triggers manual review flag
4. **Alignment Weighting:** `divergent` alignment increases overall divergence score

### 6.3 Evidence Pack Integration

The governance signal includes fields for evidence pack integration:

```json
{
  "signal_type": "replay_safety",
  "status": "ok",
  "governance_status": "ok",
  "governance_alignment": "aligned",
  "safe_for_policy_update": true,
  "safe_for_promotion": true,
  "evidence_hash": "<sha256>",
  "whitepaper_ready": true
}
```

---

## 7. Schema Definitions

### 7.1 Schema Locations

| Schema | Path |
|--------|------|
| Governance Radar | `docs/system_law/schemas/replay/replay_governance_radar.schema.json` |
| Promotion Eval | `docs/system_law/schemas/replay/replay_promotion_eval.schema.json` |
| Director Panel | `docs/system_law/schemas/replay/replay_director_panel.schema.json` |
| Global Console Tile | `docs/system_law/schemas/replay/replay_global_console_tile.schema.json` |

### 7.2 Schema Version Policy

- All schemas use JSON Schema Draft-07
- Version field: `schema_version: "1.0.0"`
- Breaking changes require major version bump
- Evidence collected under version N is valid for whitepaper evidence

---

## 8. Implementation TODO Anchors

### 8.1 Global Console Tile Registration

```python
# TODO(PHASE-X-REPLAY): Register replay_governance tile in build_global_health_surface()
# Location: backend/health/global_surface.py
# Depends on: attach_replay_governance_tile() implementation
# Blocked by: replay_governance_adapter.py creation
```

### 8.2 CLAUDE I Governance Signal Fusion

```python
# TODO(PHASE-X-REPLAY): Add replay_safety signal to CLAUDE I governance registry
# Location: backend/analytics/governance_verifier.py (or designated registry)
# Depends on: GovernanceSignalRegistry implementation
# Fusion rule: Conservative BLOCK propagation
```

### 8.3 Evidence Pack Builder

```python
# TODO(PHASE-X-REPLAY): Add replay evidence to whitepaper evidence builder
# Location: scripts/build_whitepaper_evidence.py (to be created)
# Artifacts: radar, promotion_eval, director_panel, console_tile
# Hash: SHA-256 of bundle
```

---

## 9. Validation Checklist

Before Phase X execution, verify:

| Checkpoint | Status | Sign-Off |
|------------|--------|----------|
| All 4 schemas pass JSON Schema Draft-07 validation | PENDING | ________ |
| `to_governance_signal_for_replay_safety()` tests pass | PENDING | ________ |
| SHADOW mode contract enforced in all tile builders | PENDING | ________ |
| Evidence hash computed correctly | PENDING | ________ |
| No control flow depends on replay signals | PENDING | ________ |

---

## 10. References

- `Phase_X_P3_Spec.md` - P3 Synthetic First-Light specification
- `Phase_X_P4_Spec.md` - P4 Shadow-Mode specification
- `Phase_X_Prelaunch_Review.md` - Pre-launch review doctrine
- `experiments/u2/replay_safety.py` - Replay safety implementation
- `tests/test_replay_safety_governance_signal.py` - Test suite

---

*Authored by: CLAUDE A - Replay Governance Integration*
*Date: 2025-12-10*
*Classification: INTERNAL - Phase X Integration Prep*
