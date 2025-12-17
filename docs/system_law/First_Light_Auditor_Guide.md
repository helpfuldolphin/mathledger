# First Light Auditor Guide — Status + Alignment Interpretation

**Document Version:** 1.1.0
**Status:** Reference
**Phase:** X (SHADOW MODE)
**Date:** 2025-12-11

---

## 1. Purpose

This guide explains how an external auditor should review the two primary First Light output artifacts:

1. **`first_light_status.json`** — Operational status summary
2. **`first_light_alignment.json`** — Governance signal fusion analysis

Together, these files provide a complete picture of First Light experiment health and governance posture.

---

## 2. Document Relationship

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FIRST LIGHT EVIDENCE PACK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────┐            │
│  │  first_light_status.json │    │ first_light_alignment.json│           │
│  │  ─────────────────────── │    │ ─────────────────────────│           │
│  │  • Harness OK/FAIL       │    │ • Signal fusion result    │           │
│  │  • Artifact presence     │    │ • Escalation level        │           │
│  │  • Metrics snapshot      │◄──►│ • Conflict detection      │           │
│  │  • Shadow mode status    │    │ • Per-signal breakdown    │           │
│  │  • Evidence pack intact  │    │ • Governance headline     │           │
│  └─────────────────────────┘    └─────────────────────────┘            │
│              │                              │                            │
│              │      AUDITOR WORKFLOW        │                            │
│              └──────────────┬───────────────┘                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                  │
│                    │  AUDIT VERDICT  │                                  │
│                    │  PASS / REVIEW  │                                  │
│                    └─────────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Status File (`first_light_status.json`)

### 3.1 Schema Overview

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-12-11T12:00:00.000000+00:00",
  "mode": "SHADOW",
  "pipeline": "local",
  "shadow_mode_ok": true,
  "determinism_ok": null,
  "p3_harness_ok": true,
  "p4_harness_ok": true,
  "evidence_pack_ok": true,
  "last_run_id": {
    "p3": "fl_20251211_120000_abc123",
    "p4": "p4_20251211_120100_def456"
  },
  "metrics_snapshot": {
    "p3_success_rate": 0.85,
    "p3_omega_occupancy": 0.92,
    "p3_mean_rsi": 0.87,
    "p4_success_rate": 0.83,
    "p4_divergence_rate": 0.15,
    "p4_twin_accuracy": 0.78
  },
  "warnings": [],
  "errors": null,
  "artifacts": {
    "evidence_pack": "results/first_light/evidence_pack_first_light",
    "manifest_sha256": "abc123..."
  }
}
```

### 3.2 Key Fields for Auditors

| Field | Type | Interpretation |
|-------|------|----------------|
| `shadow_mode_ok` | bool | **PRIMARY GATE**: Must be `true` for valid SHADOW experiment |
| `p3_harness_ok` | bool | P3 synthetic experiment completed successfully |
| `p4_harness_ok` | bool | P4 shadow coupling completed successfully |
| `evidence_pack_ok` | bool | Evidence pack integrity verified |
| `mode` | string | Must be `"SHADOW"` — any other value is invalid |
| `warnings` | array | Operational concerns for review |
| `errors` | array/null | Critical issues — if present, audit fails |

### 3.3 Status Decision Matrix

| shadow_mode_ok | evidence_pack_ok | errors | Audit Status |
|----------------|------------------|--------|--------------|
| true | true | null | ✅ PASS |
| true | true | present | ⚠️ REVIEW REQUIRED |
| true | false | any | ❌ FAIL — Pack integrity issue |
| false | any | any | ❌ FAIL — Shadow mode violation |

---

## 4. Alignment File (`first_light_alignment.json`)

### 4.1 Schema Overview

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-12-11T12:00:00.000000+00:00",
  "cycle": 0,
  "mode": "shadow",
  "signals": {
    "topology": {...},
    "replay": {...},
    "metrics": {...},
    "budget": {...},
    "structure": {...},
    "telemetry": {...},
    "identity": {...},
    "narrative": {...}
  },
  "fusion_result": {
    "decision": "ALLOW",
    "is_hard": false,
    "primary_reason": "All signals nominal",
    "block_score": 0.0,
    "allow_score": 45.0,
    "determining_signal": null
  },
  "escalation": {
    "level": 0,
    "level_name": "L0_NOMINAL",
    "trigger_reason": "All systems nominal"
  },
  "conflict_detections": [],
  "recommendations": [...],
  "headline": "All signals nominal; governance fusion ALLOW",
  "first_light": {
    "p3_run_id": "fl_20251211_120000_abc123",
    "p4_run_id": "p4_20251211_120100_def456",
    "evidence_pack": "results/first_light/evidence_pack_first_light",
    "generated_at": "2025-12-11T12:05:00.000000+00:00"
  }
}
```

### 4.2 Key Fields for Auditors

| Field | Type | Interpretation |
|-------|------|----------------|
| `fusion_result.decision` | string | `"ALLOW"` or `"REJECT"` — unified governance decision |
| `fusion_result.is_hard` | bool | If `true`, decision is non-overridable |
| `escalation.level_name` | string | L0-L5 escalation level |
| `conflict_detections` | array | Cross-signal consistency violations |
| `headline` | string | Human-readable summary |

### 4.3 Escalation Level Interpretation

| Level | Name | Meaning | Auditor Action |
|-------|------|---------|----------------|
| L0 | NOMINAL | All systems healthy | ✅ No action |
| L1 | WARNING | Minor concerns detected | Review warnings |
| L2 | DEGRADED | Multiple soft issues | Review metrics |
| L3 | CRITICAL | Hard failure detected | ⚠️ Investigate |
| L4 | CONFLICT | Cross-signal inconsistency | ⚠️ Investigate signals |
| L5 | EMERGENCY | Critical failure | ❌ Escalate immediately |

### 4.4 Signal Health Summary

The `signal_summary` field provides per-signal status:

```json
"signal_summary": {
  "topology": {"status": "healthy", "recommendations": 1},
  "replay": {"status": "healthy", "recommendations": 1},
  "metrics": {"status": "healthy", "recommendations": 1},
  "budget": {"status": "healthy", "recommendations": 1},
  "structure": {"status": "healthy", "recommendations": 1},
  "telemetry": {"status": "healthy", "recommendations": 1},
  "identity": {"status": "healthy", "recommendations": 1},
  "narrative": {"status": "healthy", "recommendations": 1}
}
```

Status values:
- `healthy` — Signal nominal
- `degraded` — Warnings present
- `unhealthy` — Rejection recommendation
- `missing` — Signal not provided

---

## 5. Joint Interpretation Workflow

### 5.1 Auditor Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FIRST LIGHT AUDIT CHECKLIST                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: STATUS FILE CHECK                                               │
│  □ shadow_mode_ok = true                                                 │
│  □ evidence_pack_ok = true                                               │
│  □ errors = null (or empty)                                              │
│  □ mode = "SHADOW"                                                       │
│                                                                          │
│  STEP 2: ALIGNMENT FILE CHECK                                            │
│  □ escalation.level_name in [L0_NOMINAL, L1_WARNING]                    │
│  □ fusion_result.decision = "ALLOW" (if shadow validation)              │
│  □ conflict_detections = [] (no cross-signal conflicts)                 │
│  □ All signal statuses are "healthy" or "degraded"                      │
│                                                                          │
│  STEP 3: CROSS-FILE CONSISTENCY                                          │
│  □ Run IDs match between status and alignment files                     │
│  □ Metrics snapshot aligns with signal values                           │
│  □ Timestamps are within acceptable range                               │
│                                                                          │
│  STEP 4: SHADOW MODE COMPLIANCE                                          │
│  □ No governance modifications occurred                                  │
│  □ All divergence was logged only                                        │
│  □ No abort enforcement                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Decision Tree

```
START
  │
  ├─ Is shadow_mode_ok = true?
  │  ├─ NO → FAIL: Shadow mode violation
  │  └─ YES → Continue
  │
  ├─ Is evidence_pack_ok = true?
  │  ├─ NO → FAIL: Pack integrity issue
  │  └─ YES → Continue
  │
  ├─ Are there errors in status file?
  │  ├─ YES → REVIEW: Examine error details
  │  └─ NO → Continue
  │
  ├─ Is escalation level L3+ (CRITICAL/CONFLICT/EMERGENCY)?
  │  ├─ YES → REVIEW: Investigate escalation trigger
  │  └─ NO → Continue
  │
  ├─ Are there conflict detections?
  │  ├─ YES → REVIEW: Examine CSC-* rules violated
  │  └─ NO → Continue
  │
  ├─ Is fusion decision REJECT with is_hard = true?
  │  ├─ YES → REVIEW: Examine determining signal
  │  └─ NO → Continue
  │
  └─ PASS: First Light experiment validated
```

---

## 6. Common Scenarios

### 6.1 Nominal Pass

**Status:**
```json
{
  "shadow_mode_ok": true,
  "p3_harness_ok": true,
  "p4_harness_ok": true,
  "evidence_pack_ok": true,
  "errors": null
}
```

**Alignment:**
```json
{
  "fusion_result": {"decision": "ALLOW", "is_hard": false},
  "escalation": {"level_name": "L0_NOMINAL"},
  "conflict_detections": [],
  "headline": "All signals nominal; governance fusion ALLOW"
}
```

**Verdict:** ✅ PASS

---

### 6.2 Warning Review

**Status:**
```json
{
  "shadow_mode_ok": true,
  "warnings": ["p3_omega_occupancy (0.88) below 90% threshold"]
}
```

**Alignment:**
```json
{
  "fusion_result": {"decision": "ALLOW"},
  "escalation": {"level_name": "L1_WARNING"},
  "headline": "Governance nominal with warnings"
}
```

**Verdict:** ⚠️ REVIEW — Omega occupancy slightly below threshold, investigate stability metrics.

---

### 6.3 Conflict Detection

**Status:**
```json
{
  "shadow_mode_ok": true,
  "evidence_pack_ok": true
}
```

**Alignment:**
```json
{
  "escalation": {"level_name": "L4_CONFLICT"},
  "conflict_detections": [
    {
      "rule_id": "CSC-003",
      "description": "Lean verifier healthy but replay verification failed",
      "signals_involved": ["telemetry", "replay"]
    }
  ]
}
```

**Verdict:** ⚠️ REVIEW — Cross-signal conflict detected. Investigate why telemetry reports Lean healthy while replay reports failure. This may indicate:
- Mock telemetry in P4 shadow mode
- Actual verification discrepancy
- Data extraction error

---

### 6.4 Critical Failure

**Status:**
```json
{
  "shadow_mode_ok": true,
  "errors": ["P4 mode is ACTIVE, expected SHADOW"]
}
```

**Alignment:**
```json
{
  "escalation": {"level_name": "L5_EMERGENCY"},
  "fusion_result": {"decision": "REJECT", "is_hard": true}
}
```

**Verdict:** ❌ FAIL — Shadow mode contract violated. P4 ran in ACTIVE mode, which is unauthorized for Phase X.

---

## 7. Reviewing the Release Attitude Annex

The **Release Attitude Annex** is a cross-pillar summary that answers: "What attitude is the whole system in?" regarding release readiness. This annex appears in the evidence pack under `evidence["governance"]["envelope_v4"]["first_light_release_attitude"]`.

### 7.1 Schema Overview

```json
{
  "schema_version": "1.0.0",
  "global_band": "GREEN",
  "system_alignment": "ALIGNED",
  "release_ready": true,
  "status_light": "🟢"
}
```

### 7.2 Key Fields

| Field | Type | Interpretation |
|-------|------|----------------|
| `global_band` | string | The envelope's global health band: `"GREEN"` (nominal), `"YELLOW"` (elevated risk), or `"RED"` (critical) |
| `system_alignment` | string | Cross-pillar alignment status: `"ALIGNED"` (≥80% components GREEN), `"PARTIAL"` (50-79% GREEN), or `"MISALIGNED"` (<50% GREEN) |
| `release_ready` | bool | Boolean indicator from director mega-panel: `true` if all critical checks pass, `false` otherwise |
| `status_light` | string | High-level visual indicator: `"🟢"` (GREEN), `"🟡"` (YELLOW), or `"🔴"` (RED) |

### 7.3 Interpretation Guidance

**Important:** The Release Attitude Annex is a **summary badge**, not an enforcement gate. It provides observational information about system health posture across all governance pillars (E1-E5). Decisions regarding release remain with human governance processes and are not automated based on this annex alone.

### 7.4 Common Scenarios

#### Scenario 1: Evidence Supports Release (SHADOW MODE)

```json
{
  "global_band": "GREEN",
  "system_alignment": "ALIGNED",
  "release_ready": true,
  "status_light": "🟢"
}
```

**Interpretation:** Evidence supports release in SHADOW MODE. All health pillars report nominal status, system components are aligned (≥80% GREEN), and the director mega-panel indicates release readiness. This suggests the system is in a stable state suitable for shadow-mode deployment and observation.

**Auditor Action:** ✅ Review supporting evidence in the full envelope v4 and director mega-panel for detailed component breakdown.

---

#### Scenario 2: System Not Release-Ready

```json
{
  "global_band": "RED",
  "system_alignment": "MISALIGNED",
  "release_ready": false,
  "status_light": "🔴"
}
```

**Interpretation:** Evidence suggests system is not release-ready even in shadow mode; further work required. Critical health issues detected across multiple pillars, system components are misaligned (<50% GREEN), and the director mega-panel indicates release is not ready. This signals that fundamental health checks have failed and the system requires remediation before shadow-mode deployment.

**Auditor Action:** ⚠️ Investigate component summaries in the director mega-panel to identify specific health drivers. Review coherence analysis for cross-pillar mismatches. Examine cross-signal hotspots for systemic issues.

### 7.5 Release Attitude Strip (P5 Calibration Experiments)

For P5 calibration workflows, a **Release Attitude Strip** aggregates release attitude annexes across multiple calibration experiments (CAL-EXP-1, CAL-EXP-2, etc.) into a single visual summary strip. This provides auditors with a quick cross-experiment posture assessment.

#### 7.5.1 Location

The Release Attitude Strip appears in the evidence pack under:
```
evidence["governance"]["release_attitude_strip"]
```

#### 7.5.2 Schema

```json
{
  "schema_version": "1.0.0",
  "experiments": [
    {
      "cal_id": "CAL-EXP-1",
      "global_band": "GREEN",
      "system_alignment": "ALIGNED",
      "release_ready": true,
      "status_light": "🟢"
    },
    {
      "cal_id": "CAL-EXP-2",
      "global_band": "YELLOW",
      "system_alignment": "PARTIAL",
      "release_ready": false,
      "status_light": "🟡"
    }
  ],
  "summary": {
    "total_count": 2,
    "release_ready_count": 1,
    "release_ready_ratio": 0.5
  }
}
```

#### 7.5.3 Interpretation

- **experiments**: List of per-experiment release attitude summaries, sorted by `cal_id` for deterministic ordering
- **summary**: Aggregated counts showing how many experiments had `release_ready=true`

**Important:** The Release Attitude Strip is a **visual shorthand for auditors**, not an enforcement gate. It provides a quick overview of release posture across calibration experiments. Decisions remain with human governance processes and are not automated based on the strip alone.

#### 7.5.4 Per-Experiment Annex Files

Individual calibration experiment annexes are exported as separate JSON files:
```
calibration/release_attitude_annex_<cal_id>.json
```

Each file contains a single experiment's release attitude annex with the `cal_id` included. These files can be reviewed individually or aggregated into the strip for cross-experiment comparison.

### 7.6 Integration with Other Artifacts

The Release Attitude Annex synthesizes information from:
- **Global Health Envelope v4**: Provides `global_band` and `system_alignment` calculations
- **Director Mega-Panel**: Provides `release_ready` boolean and `status_light` emoji indicator
- **Coherence Analysis**: Informs system alignment assessment through mismatch detection

For detailed component-level analysis, auditors should refer to the full `envelope_v4`, `coherence_analysis`, and `director_mega_panel` structures also present in `evidence["governance"]["envelope_v4"]`.

---

## 8. Metrics Cross-Reference

The following metrics should align between status and alignment files:

| Status Metric | Alignment Signal Field | Expected Relationship |
|---------------|------------------------|----------------------|
| `p3_success_rate` | `signals.topology.H` | Should be close (±0.05) |
| `p3_omega_occupancy` | `signals.topology.within_omega` | `>0.90` → `true` |
| `p3_mean_rsi` | `signals.topology.rho` | Should match exactly |
| `p4_divergence_rate` | `signals.replay.replay_divergence` | Should match exactly |
| `p4_twin_accuracy` | `signals.replay.replay_verified` | `>0.70` → `true` |

If metrics diverge significantly, investigate:
1. Different data sources (P3 vs P4)
2. Signal extraction logic
3. Timestamp mismatch

---

## 9. Shadow Mode Compliance Verification

### 9.1 Required Invariants

The following must be true for valid SHADOW mode:

1. **No Governance Modification**
   - `status.mode = "SHADOW"`
   - `alignment.mode = "shadow"`
   - Evidence pack `shadow_mode_compliance.no_governance_modification = true`

2. **Logged-Only Divergence**
   - All red flags have `action = "LOGGED_ONLY"`
   - Evidence pack `shadow_mode_compliance.all_divergence_logged_only = true`

3. **No Abort Enforcement**
   - Red flag `hypothetical_abort` may be `true`, but no actual abort occurred
   - Evidence pack `shadow_mode_compliance.no_abort_enforcement = true`

### 9.2 Compliance Check

```python
def verify_shadow_compliance(status: dict, alignment: dict) -> bool:
    """Verify SHADOW mode compliance across both files."""
    # Status checks
    if status.get("mode") != "SHADOW":
        return False
    if not status.get("shadow_mode_ok"):
        return False

    # Alignment checks
    if alignment.get("mode") != "shadow":
        return False

    # Fusion must not have enforced any decisions
    # (In SHADOW mode, fusion is purely observational)

    return True
```

---

## 10. Artifact Traceability

Both files should reference the same evidence pack and run IDs:

```
first_light_status.json:
  └── last_run_id.p3 ─────────────────┐
  └── last_run_id.p4 ─────────────────┤
  └── artifacts.evidence_pack ────────┤
                                      │
first_light_alignment.json:           │
  └── first_light.p3_run_id ──────────┴── MUST MATCH
  └── first_light.p4_run_id ──────────┴── MUST MATCH
  └── first_light.evidence_pack ──────┴── MUST MATCH
```

---

## 11. Summary

For a successful First Light audit:

| Check | Status File | Alignment File | Required |
|-------|-------------|----------------|----------|
| Shadow mode | `shadow_mode_ok = true` | `mode = "shadow"` | ✅ |
| Pack integrity | `evidence_pack_ok = true` | — | ✅ |
| No errors | `errors = null` | — | ✅ |
| Escalation | — | `level ≤ L2` | ⚠️ |
| No conflicts | — | `conflict_detections = []` | ⚠️ |
| Fusion decision | — | `decision = "ALLOW"` | ⚠️ |

- ✅ = Primary gate (must pass)
- ⚠️ = Review gate (investigate if not met)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.0 | 2025-12-11 | Added Release Attitude Strip (P5 calibration) documentation |
| 1.1.0 | 2025-12-11 | Added Release Attitude Annex interpretation section |
| 1.0.0 | 2025-12-11 | Initial auditor guide |
