# PILOT PHASE INDEX

**Status**: COMPLETE / FROZEN (SHADOW MODE)
**Initiated**: 2025-12-13
**Authority**: STRATCOM
**Mode**: SHADOW (observational only)
**Canonical ask-shaped artifact**: `docs/evidence/pilot/PILOT_EVIDENCE_PACKET_v1.0.pdf`

---

## 1. Scope Definition

### 1.1 What Pilot IS

The Pilot Phase is an **operational dry-run** of the MathLedger system under controlled conditions. It serves to:

- Exercise code paths end-to-end
- Identify integration gaps
- Establish operational baselines
- Build narrative infrastructure for future phases

### 1.2 What Pilot IS NOT

| NOT | Explanation |
|-----|-------------|
| A validation of learning | No claims about system learning or adaptation |
| A scientific experiment | No hypothesis testing, no statistical conclusions |
| Authorization for external claims | Nothing produced here supports marketing, papers, or announcements |
| A gating mechanism | Pilot outcomes do not gate subsequent phases |
| An accuracy benchmark | No claims about correctness, precision, or recall |

---

## 2. Operational Constraints

### 2.1 SHADOW MODE (Binding)

All Pilot Phase operations execute under **SHADOW MODE**:

- **Observational only** — no enforcement actions
- **No state mutation** — system state unchanged by observations
- **No gating decisions** — outputs are advisory, never blocking
- **Logged but not acted upon** — all signals recorded, none enforced

### 2.2 Forbidden Actions

| Action | Status |
|--------|--------|
| Enforcement based on pilot outputs | FORBIDDEN |
| Claims of validation or verification | FORBIDDEN |
| Use of pilot data in external communications | FORBIDDEN |
| Modification of governance rules based on pilot | FORBIDDEN |
| Exit from SHADOW MODE | FORBIDDEN |

### 2.3 Single-Path Smoke Proof

Copy/paste proof that pilot ingestion is SHADOW-only and non-interfering.

See: [PILOT_SMOKE_PROOF_SINGLE_PATH.md](PILOT_SMOKE_PROOF_SINGLE_PATH.md)

---

## 3. Participating Agents

| Agent | Role | Boundary |
|-------|------|----------|
| CLAUDE A | Replay signal freeze (v1.3.0) | STANDING DOWN |
| CLAUDE C | P5 Diagnostic Harness | STANDING DOWN |
| CLAUDE L | Metric Integrity | STANDING DOWN |
| CLAUDE N | NVR Run-Dir Shape | STANDING DOWN |
| CLAUDE O | TDA Adapter + Cross-Shell Preflight | STANDING DOWN |
| ALL (A-O) | CAL-EXP-2 Phase Closure | STANDING DOWN |

New agent participation requires explicit STRATCOM directive.

---

## 4. Frozen Boundaries

### 4.1 Code Paths

The following are frozen for Pilot Phase:

- `scripts/generate_first_light_status.py` — Status generator
- `backend/health/tda_windowed_patterns_adapter.py` — TDA GGFL adapter
- `scripts/preflight_shell_env.py` — Cross-shell advisory (no blocking)
- `scripts/verify_cal_exp_2_run.py` — CAL-EXP-2 verifier

### 4.2 Documentation

The following documents are frozen:

- `docs/system_law/calibration/CAL_EXP_2_INDEX.md`
- `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
- `docs/system_law/TDA_PhaseX_Binding.md`

### 4.3 Behavioral Contracts

| Contract | Status |
|----------|--------|
| Preflight emits at most 1 line | FROZEN |
| Preflight never blocks execution | FROZEN |
| Preflight never writes files | FROZEN |
| GGFL adapters use DRIVER_ prefix | FROZEN |
| Extraction sources normalize to 3 values | FROZEN |

---

## 5. Success Criteria (Operational)

Pilot Phase success is **operational**, not scientific:

### 5.1 Success Means

| Criterion | Definition |
|-----------|------------|
| Code executes | No crashes, no unhandled exceptions |
| Outputs are well-formed | JSON valid, schema compliant |
| Logs are written | Telemetry captured as specified |
| No enforcement triggered | SHADOW MODE maintained throughout |
| Artifacts are auditable | Clear provenance chain |

### 5.2 Success Does NOT Mean

| NOT Success | Explanation |
|-------------|-------------|
| System "works" | No correctness claims |
| Learning occurred | No adaptation claims |
| Accuracy improved | No performance claims |
| Production ready | No deployment claims |

---

## 6. Explicit Non-Claims

This section exists to prevent scope creep and claim inflation.

### 6.1 The Pilot Phase Does NOT:

1. **Validate learning** — No system learning has been demonstrated or claimed
2. **Authorize external claims** — Nothing here supports announcements, papers, or marketing
3. **Prove correctness** — Execution does not imply correctness
4. **Establish baselines** — Observations are not baselines for future comparison
5. **Gate future phases** — Pilot outcomes do not determine phase transitions

### 6.2 Language Prohibited in Pilot Artifacts

| Prohibited | Reason |
|------------|--------|
| "verified" | Implies validation |
| "validated" | Implies correctness proof |
| "proven" | Implies mathematical certainty |
| "accurate" | Implies measurement against ground truth |
| "learned" | Implies adaptation occurred |
| "improved" | Implies comparison to baseline |
| "production" | Implies deployment readiness |

---

## 7. Pilot External Ingest (SHADOW)

TDA Windowed Patterns external signal ingestion surface.

### 7.1 Artifacts

| Type | Path |
|------|------|
| Adapter | `backend/health/tda_windowed_patterns_adapter.py` |
| Tests | `tests/health/test_tda_windowed_patterns_adapter.py` |
| Documentation | `docs/system_law/TDA_PhaseX_Binding.md` (Section 14) |

### 7.2 Test Command

```bash
uv run python -m pytest tests/health/test_tda_windowed_patterns_adapter.py -v
# Expected: 55 passed
```

### 7.3 Frozen Enums

- `extraction_source`: MANIFEST, EVIDENCE_JSON, MISSING
- `reason_code`: DRIVER_WINDOWED_DETECTED_PATTERN, DRIVER_SINGLE_SHOT_DETECTED_PATTERN
- `signal_type`: SIG-TDAW

### 7.4 Non-Interference Guarantee

The external ingest surface operates under non-interference:

- Adapter reads manifest and evidence data without modification
- Extraction failures return None or empty structures (no exceptions propagated)
- Warning output is capped (max 1 line per warning type)
- No enforcement actions triggered by any signal value
- No governance state modified by adapter execution
- All outputs are logged for observability, none acted upon

---

## 8. Transition Out of Pilot

Exit from Pilot Phase requires:

1. Explicit STRATCOM directive
2. New experiment ID (CAL-EXP-3+)
3. Documented scope change
4. Updated constraint set

Until then, Pilot Phase remains ACTIVE under SHADOW MODE.

---

## 9. Document Control

| Field | Value |
|-------|-------|
| Owner | STRATCOM |
| Classification | Narrative Infrastructure |
| Review Cycle | Per STRATCOM directive only |
| Amendment Authority | STRATCOM only |

---

*This document is narrative infrastructure. It defines operational boundaries, not scientific claims. Precision over optimism.*
