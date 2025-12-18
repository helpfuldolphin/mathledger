# Pilot Non-Claims

**Status:** BINDING
**Effective:** 2025-12-18
**Authority:** STRATCOM
**Purpose:** Explicit boundary on what pilot execution does and does not demonstrate

---

## 1. What Pilot Demonstrates

Pilot execution demonstrates **operational properties only**:

| Property | Meaning |
|----------|---------|
| Code executes | No crashes, no unhandled exceptions |
| Outputs are well-formed | JSON valid, schema compliant |
| Logs are written | Telemetry captured as specified |
| SHADOW mode maintained | No enforcement triggered |
| Artifacts are auditable | Clear provenance chain |

---

## 2. What Pilot Does NOT Demonstrate

| Non-Claim | Explanation |
|-----------|-------------|
| System correctness | Execution does not imply correctness |
| Learning or adaptation | No system learning has been demonstrated or claimed |
| Accuracy or precision | No measurement against ground truth |
| Performance improvement | No comparison to baseline |
| Production readiness | No deployment authorization |
| External claims authorization | Nothing supports announcements, papers, or marketing |
| Baseline establishment | Observations are not baselines for future comparison |
| Gating authority | Pilot outcomes do not determine phase transitions |

---

## 3. Forbidden Language

The following terms are **prohibited** in pilot artifacts:

| Term | Status | Reason |
|------|--------|--------|
| "verified" | FORBIDDEN | Implies validation |
| "validated" | FORBIDDEN | Implies correctness proof |
| "proven" | FORBIDDEN | Implies mathematical certainty |
| "accurate" | FORBIDDEN | Implies measurement against ground truth |
| "learned" | FORBIDDEN | Implies adaptation occurred |
| "improved" | FORBIDDEN | Implies comparison to baseline |
| "production" | FORBIDDEN | Implies deployment readiness |
| "deployment" | FORBIDDEN | Implies operational use |
| "blocking" | FORBIDDEN | Implies enforcement |
| "control" | FORBIDDEN | Implies authority |

### Permitted Alternatives

| Instead of | Use |
|------------|-----|
| "verified" | "executed" |
| "validated" | "observed" |
| "proven" | "recorded" |
| "production" | "pilot" |
| "deployment" | "execution" |
| "blocking" | "advisory" |
| "control" | "observation" |

---

## 4. Authorization Boundaries

| Level | Name | Status | Implication |
|-------|------|--------|-------------|
| 0 | Internal Calibration | COMPLETE | Prerequisites satisfied |
| 1 | Pilot Readiness | AUTHORIZED | Internal execution permitted |
| 2 | Public Pilot | BLOCKED | No public-facing pilot |
| 3 | External Claims | BLOCKED | No announcements, papers, marketing |

**Gap between Level 1 and Level 3 is significant.** Pilot execution does not authorize external claims.

---

## 5. Constitutional Rule

**Any pilot behavior violating these constraints voids experimental validity.**

Specifically:
- Use of forbidden language invalidates associated artifacts
- Enforcement actions based on pilot outputs are contract violations
- Claims exceeding authorization level are governance breaches

---

## 6. Enforcement

| Mechanism | Path |
|-----------|------|
| CI Gate | `.github/workflows/pilot-phase-gate.yml` |
| Text Neutrality Tests | `tests/policy/test_pilot_text_neutrality.py` |
| Non-Interference Tests | `tests/health/test_pilot_external_ingest_non_interference.py` |
| Isolation Tests | `tests/integration/test_external_pilot_signal_isolation.py` |

---

## 7. Summary

```
PILOT EXECUTION:
- Demonstrates: operational execution
- Does NOT demonstrate: correctness, learning, accuracy, readiness

PILOT CLAIMS:
- Authorized: "code ran", "outputs produced", "logs written"
- NOT authorized: "system works", "learning occurred", "ready for production"

PILOT MODE:
- SHADOW-OBSERVE only
- Logged, not acted upon
- Observational, not authoritative
```

---

*Pilot evaluation produces artifacts, not authority. This document exists to prevent scope creep and claim inflation. Precision over optimism.*
