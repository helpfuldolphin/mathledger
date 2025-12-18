# If You Only Read Three Files

**Audience**: External verifiers, pilot partners, second maintainers
**Reading time**: 15–20 minutes total

---

If you only read three documents to understand MathLedger's governance posture, read these:

---

## 1. SECOND_MAINTAINER_BRIEF_v1.2.md

**Location**: `docs/SECOND_MAINTAINER_BRIEF_v1.2.md`

**What it tells you**: What we're asking you to verify, how long it takes, and what a successful verification looks like.

This is your entry point. It includes a concrete checklist, explicit scope boundaries, and instructions for producing a Verification Note.

---

## 2. SHADOW_MODE_CONTRACT.md

**Location**: `docs/system_law/SHADOW_MODE_CONTRACT.md`

**What it tells you**: The canonical definition of SHADOW MODE semantics — specifically, what "SHADOW-OBSERVE" means (non-blocking, evidence-only) versus "SHADOW-GATED" (registered blocking gates).

If you encounter "SHADOW MODE" anywhere in the codebase, this document is authoritative.

---

## 3. DETERMINISM_CLAIMS.md

**Location**: `docs/DETERMINISM_CLAIMS.md`

**What it tells you**: Exactly what MathLedger claims to be deterministic, under what conditions, and — critically — what is NOT claimed.

This document scopes our claims precisely. If a claim isn't listed here, we don't make it.

---

## Optional: Audit Plane v0

If you want to understand the evidentiary audit layer:

**Location**: `docs/system_law/audit/AUDIT_PLANE_V0_SPEC.md`

**Key point**: Audit Plane v0 produces evidence only. It has no authority, no blocking capability, and does not participate in `H_t = SHA256(R_t || U_t)`.

---

## Questions?

If something is unclear or appears contradictory, that's valuable feedback. Note it in your Verification Note — we prefer honest skepticism over polite confusion.

---

*This document is external-safe and may be shared with prospective verifiers or partners.*
