# Pilot Intake Checklist

> **DEPRECATED (2025-12-21):** This document covers internal intake decisions.
> For external audit procedures, see:
> - [`docs/pilot/PILOT_NON_CLAIMS.md`](../../../docs/pilot/PILOT_NON_CLAIMS.md) - Authoritative non-claims
> - [`docs/pilot/PILOT_EVALUATION_CHECKLIST.md`](../../../docs/pilot/PILOT_EVALUATION_CHECKLIST.md) - External reviewer checklist
> - [`docs/pilot/AUDIT_WALKTHROUGH.md`](../../../docs/pilot/AUDIT_WALKTHROUGH.md) - Complete audit procedure

**Status:** BINDING (internal use only)
**Effective:** 2025-12-18
**Authority:** STRATCOM
**Purpose:** Pre-approval checklist before accepting any pilot engagement (internal)

---

## 1. Red Flags (Automatic Decline)

If the prospective partner requests ANY of the following, **decline immediately**:

| Red Flag | Reason |
|----------|--------|
| Blocking or gating capability | SHADOW mode prohibits enforcement |
| Production deployment | Level 3 authorization BLOCKED |
| Accuracy or correctness validation | Non-claim; not what pilots demonstrate |
| Learning or adaptation measurement | Non-claim; not demonstrated |
| Performance benchmarks with guarantees | Non-claim; no baselines established |
| Custom API endpoints | No new APIs permitted |
| Schema modifications | Frozen surfaces |
| Integration into their production systems | Beyond SHADOW-OBSERVE scope |
| Public announcements based on pilot | Level 3 authorization BLOCKED |
| Removal of SHADOW mode constraints | Invariant; cannot be lifted |

---

## 2. Required Acknowledgements

Partner must explicitly acknowledge (in writing) before pilot begins:

- [ ] Pilot operates under **SHADOW-OBSERVE only** (observational, non-authoritative)
- [ ] Pilot does **NOT** validate correctness, accuracy, or learning
- [ ] Pilot does **NOT** authorize production deployment
- [ ] Pilot does **NOT** authorize external claims or announcements
- [ ] Artifacts produced are **observational records**, not authority
- [ ] All outputs are subject to **second-maintainer review**
- [ ] Partner will **not** represent pilot results as validation or proof

---

## 3. Scope Alignment Confirmation

Before proceeding, confirm alignment on:

| Item | Aligned? |
|------|----------|
| Partner understands SHADOW-OBSERVE semantics | [ ] Yes |
| Partner's evaluation goals match pilot scope | [ ] Yes |
| Partner is not expecting blocking/enforcement | [ ] Yes |
| Partner is not expecting accuracy metrics | [ ] Yes |
| Partner accepts artifact-only deliverables | [ ] Yes |
| Partner understands no production authorization | [ ] Yes |
| Timeline expectations are reasonable | [ ] Yes |

---

## 4. Permitted Pilot Activities

| Activity | Status |
|----------|--------|
| Execute code end-to-end | PERMITTED |
| Generate evidence packs | PERMITTED |
| Review provenance chains | PERMITTED |
| Examine telemetry logs | PERMITTED |
| Audit schema compliance | PERMITTED |
| Observe SHADOW mode behavior | PERMITTED |

---

## 5. Prohibited Pilot Activities

| Activity | Status |
|----------|--------|
| Modify frozen surfaces | PROHIBITED |
| Add new APIs | PROHIBITED |
| Enable enforcement | PROHIBITED |
| Lift SHADOW mode | PROHIBITED |
| Make external claims | PROHIBITED |
| Establish baselines | PROHIBITED |

---

## 6. Approval Gate

**Pilot may proceed only if:**

1. No red flags present
2. All acknowledgements received in writing
3. Scope alignment confirmed
4. Internal review completed

**Approver:** STRATCOM or designated authority

---

## 7. Post-Intake Actions

After approval:

- [ ] Document partner acknowledgements
- [ ] Confirm artifact delivery format
- [ ] Schedule pilot execution window
- [ ] Assign internal review point-of-contact
- [ ] Prepare non-claims briefing for partner

---

*Pilot evaluation produces artifacts, not authority. When in doubt, decline.*
