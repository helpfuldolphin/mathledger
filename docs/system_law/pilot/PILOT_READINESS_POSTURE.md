# Pilot Readiness Posture

**Status:** BINDING
**Effective:** 2025-12-18
**Authority:** STRATCOM
**Mode:** SHADOW-OBSERVE (invariant)

---

## 1. Containment Verdict

**VERDICT: PASS**

| Criterion | Status | Enforcement |
|-----------|--------|-------------|
| SHADOW-OBSERVE only | PASS | 89 tests verify, CI gate active |
| No learning influence | PASS | Adapter is read-only, no metric creation |
| No gating/control | PASS | Contract forbids, CI enforces |
| No deployment implication | PASS | Level 2/3 authorization BLOCKED |
| No new APIs | PASS | Uses existing evidence pack interfaces |
| No bespoke integrations | PASS | Wraps to existing manifest schema |

---

## 2. Pilot Posture

MathLedger presents a sealed, governance-complete substrate. The system is open to controlled pilot evaluations under SHADOW-OBSERVE constraints.

**Mode:** All pilot operations are observational only. Outputs are logged but not acted upon. No enforcement, gating, or control actions are permitted.

**Scope:** Internal execution authorized (Level 1). Public pilot and external claims remain BLOCKED (Levels 2-3).

**Review:** All outputs are subject to second-maintainer review. Provenance chain required for all artifacts.

**Principle:** Pilot evaluation produces artifacts, not authority.

**Freeze Notice:** The Pilot Evaluation Checklist and associated artifacts are frozen for the duration of second-maintainer review and pilot outreach.

---

## 3. Pilot Scope Freeze

### Frozen Surfaces

| Surface | Path | Version |
|---------|------|---------|
| External Ingest Adapter | `backend/health/pilot_external_ingest_adapter.py` | v1.0.0 |
| Pilot Toolchain Hook | `scripts/pilot_toolchain_hook.py` | v1.0.0 |
| Ingest Enums | `PilotIngestSource`, `PilotIngestResult` | FROZEN |
| Pilot CI Gate | `.github/workflows/pilot-phase-gate.yml` | FROZEN |
| Non-Interference Tests | `tests/*/test_*pilot*.py` | FROZEN |

### Out of Scope

| Item | Reason |
|------|--------|
| New external data formats | No schema extension permitted |
| API endpoints for pilot ingestion | No new APIs |
| Pilot-to-governance feedback loops | SHADOW mode prohibits |
| Enforcement based on pilot signals | Forbidden |
| Modification of CAL-EXP-2 frozen paths | Scope fence violation |

### Protected During Second-Maintainer Review

1. Authorization hierarchy (Levels 0-3)
2. SHADOW mode invariant
3. Frozen enum values
4. Non-interference test assertions
5. Explicit non-claims

---

## 4. Explicit Non-Actions

### Code That Must Not Be Written

| Prohibited | Reason |
|------------|--------|
| New pilot APIs or endpoints | No new APIs permitted |
| Learning feedback loops | SHADOW mode prohibits influence |
| Pilot-to-governance signal propagation | Isolation requirement |
| Enforcement logic based on pilot outputs | Forbidden by contract |
| External data schema extensions | Schema freeze |

### Features That Must Not Be Added

| Prohibited | Reason |
|------------|--------|
| Accuracy measurement | Not authorized |
| Performance benchmarking | Not a validation phase |
| Baseline establishment | Non-claim |
| A/B testing infrastructure | Beyond scope |
| Production deployment hooks | Level 3 blocked |

---

## 5. Safe External Statement

> "Here is a sealed, governance-complete substrate. We are open to controlled pilot evaluations under SHADOW-OBSERVE. All outputs are observational only, logged but not acted upon, and subject to second-maintainer review."

---

## 6. Verification

```bash
# Containment test suite
uv run pytest tests/health/test_pilot_external_ingest_non_interference.py \
    tests/integration/test_external_pilot_signal_isolation.py \
    tests/policy/test_pilot_text_neutrality.py -v
# Expected: 89 passed
```

---

## 7. References

| Document | Path |
|----------|------|
| Pilot Index | `docs/system_law/pilot/PILOT_INDEX.md` |
| Authorization Hierarchy | `docs/system_law/pilot/PILOT_AUTHORIZATION.md` |
| Contract Posture | `docs/system_law/pilot/PILOT_CONTRACT_POSTURE.md` |
| Non-Interference Attestation | `docs/system_law/pilot/PILOT_NON_INTERFERENCE_ATTESTATION.md` |
| Non-Claims | `docs/system_law/pilot/PILOT_NON_CLAIMS.md` |

---

*This document defines operational posture, not scientific claims. SHADOW-OBSERVE mode is invariant.*
