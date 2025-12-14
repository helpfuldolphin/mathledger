# PILOT CONTRACT POSTURE

**Status:** BINDING
**Effective:** 2025-12-13
**Authority:** CLAUDE P (Contract Owner)

---

## 1. Mode Declaration

Pilot Phase operates under **SHADOW MODE ONLY**:
- Observational — no state mutation
- Advisory — no enforcement actions
- Non-gating — outputs never block execution

## 2. Enforcement Prohibition

| Action | Status |
|--------|--------|
| Enforcement based on pilot outputs | **FORBIDDEN** |
| Gating decisions from pilot signals | **FORBIDDEN** |
| Exit from SHADOW MODE | **FORBIDDEN** |

## 3. Schema Isolation

Pilot schemas (`pilot_toolchain_manifest.schema.json`) are **pilot-scoped**:
- Do NOT modify CAL-EXP schemas
- Do NOT extend v0.1 contract schemas
- Do NOT introduce new required artifacts to v0.1 surfaces

## 4. Frozen Surfaces

| Surface | Path | Status |
|---------|------|--------|
| RUN_SHADOW_AUDIT v0.1 Contract | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | **FROZEN** |
| CAL-EXP-1 Harness | `scripts/first_light_cal_exp1_*.py` | **FROZEN** |
| CAL-EXP-2 Harness | `scripts/first_light_cal_exp2_convergence.py` | **FROZEN** |
| P5 CAL-EXP Harness | `scripts/run_p5_cal_exp1.py` | **FROZEN** |
| Pilot Ingest Adapter | `external_ingest/adapter_enums.py` | **FROZEN** |

---

*Pilot does not modify frozen surfaces. Violations require STRATCOM override.*
