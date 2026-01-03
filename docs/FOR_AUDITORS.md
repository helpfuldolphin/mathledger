# For Auditors: 5-Minute Verification Checklist

This page provides a quick verification path for external auditors evaluating MathLedger's governance claims.

## 5-Minute Verification Checklist

Complete these steps to verify the system's core claims:

### Step 1: Open the Hosted Demo (30 seconds)

1. Navigate to [/demo/](/demo/)
2. Confirm the demo loads without errors
3. Verify the version banner shows `{{CURRENT_TAG}}`

### Step 2: Run the Boundary Demo (90 seconds)

1. In the demo, click **"Boundary Demo: Same Claim, Different Authority"**
2. Observe the 4-step sequence:
   - Step 1: Claim "2+2=4" as **MV** (Mechanically Verified) → VERIFIED
   - Step 2: Same claim "2+2=4" as **PA** (Professional Authority) → ABSTAINED
   - Step 3: Same claim "2+2=4" as **ADV** (Advisory) → ABSTAINED
   - Step 4: Claim "2+2=5" as **MV** → REFUTED
3. **Key insight:** Same mathematical truth, different outcomes based on trust class declaration

### Step 3: Download Evidence Pack (30 seconds)

1. After running the boundary demo, click **"Download Evidence Pack"**
2. Save the JSON file locally
3. Note: This file contains cryptographic commitments to all demo operations

### Step 4: Verify with Auditor Tool (60 seconds)

1. Open the [Evidence Pack Verifier](/{{CURRENT_VERSION}}/evidence-pack/verify/)
2. Upload the evidence pack JSON you downloaded
3. Click **"Verify"**
4. Confirm status shows **PASS**
5. Observe the hash verification: U_t, R_t, and H_t all match

### Step 5: Tamper Test (60 seconds)

1. Open the downloaded evidence pack in a text editor
2. Change any character in the `reasoning_artifacts` section
3. Re-upload to the verifier
4. Click **"Verify"**
5. Confirm status shows **FAIL** with hash mismatch

**If all 5 steps complete as described, the core verification loop is functioning correctly.**

---

## Ready-to-Verify Examples (No Demo Required)

If the demo is unavailable or you want to verify without running it, use these pre-generated examples:

### Download Examples

The file [`/{{CURRENT_VERSION}}/evidence-pack/examples.json`](/{{CURRENT_VERSION}}/evidence-pack/examples.json) contains:

| Example | Expected Result | Purpose |
|---------|-----------------|---------|
| `valid_boundary_demo` | PASS | Shows a correctly-formed evidence pack from a boundary demo |
| `tampered_ht_mismatch` | FAIL (h_t mismatch) | Demonstrates detection of tampered H_t field |
| `tampered_rt_mismatch` | FAIL (r_t mismatch) | Demonstrates detection of tampered reasoning artifacts |

### Verification Steps

1. Download [`examples.json`](/{{CURRENT_VERSION}}/evidence-pack/examples.json)
2. Copy the content of `examples.valid_boundary_demo.pack` to a new file
3. Open the [Evidence Pack Verifier](/{{CURRENT_VERSION}}/evidence-pack/verify/)
4. Upload the pack JSON
5. Click **Verify** and observe **PASS**
6. Repeat with `tampered_ht_mismatch.pack` and observe **FAIL**

**Or use the verifier's built-in self-test:** Click "Run self-test vectors" on the verifier page to see all examples tested automatically.

### Why This Matters

These examples prove:
- The verification algorithm is deterministic (same input always produces same hash)
- Tamper detection works (any modification is detected)
- No demo required (auditors can verify offline using the same code)

---

## What This Version Claims

This version ({{CURRENT_VERSION}}) demonstrates:

- **Deterministic canonicalization** of reasoning artifacts (RFC 8785-style)
- **Cryptographic binding** via H_t = SHA256(R_t || U_t)
- **Trust class separation** where ADV claims are excluded from R_t
- **Replay verification** using the same code paths as live execution
- **Self-auditing capability** via the evidence pack verifier

See [Invariants Status](../invariants/) for the complete Tier A/B/C breakdown.

---

## What This Version Does NOT Claim

**Explicit non-claims:**

1. **No formal verification** — FV (Formally Verified) claims always return ABSTAINED. There is no Lean/Z3/Coq integration.

2. **No multi-model consensus** — Only one partitioner template is active. The "multi-model arena" is not operational.

3. **No learning loop** — RFL (Reinforcement from Ledger) is not active. The system does not learn from past decisions.

4. **No production hardening** — This is a demonstration, not a production system. Do not use for real financial or legal decisions.

5. **No guaranteed uptime** — The hosted demo may be unavailable. Local execution is always possible.

---

## Key Documentation Links

| Document | Description |
|----------|-------------|
| [Scope Lock](../scope-lock/) | What this version does and does not demonstrate |
| [Invariants](../invariants/) | Tier A/B/C enforcement status with explicit counts |
| [Field Manual](../field-manual/) | Obligation ledger driving version promotions |
| [Hostile Demo Rehearsal](../hostile-rehearsal/) | Pre-prepared answers to skeptical questions |
| [How the Demo Explains Itself](../explanation/) | UI self-explanation integration points |

---

## Field Manual (Why It Exists)

The [Field Manual](../field-manual/) (fm.tex/pdf) is an internal constraint artifact, not marketing documentation.

**It is not rewritten yet; it is used to surface obligations and gaps.**

How auditors should use it:
- **Cross-reference claims** — If a feature is claimed, check FM for caveats
- **Look for "TODO" and "OBLIGATION"** — Explicit acknowledgments of gaps
- **Compare versions** — FM changes show what was addressed between releases
- **Trust gaps over features** — The gaps we document are more honest than features we claim

Download: [fm.pdf](../field-manual/fm.pdf) | [fm.tex](../field-manual/fm.tex)

---

## Local Verification (Alternative Path)

If the hosted demo is unavailable, verify locally:

```bash
git clone https://github.com/helpfuldolphin/mathledger
cd mathledger
git checkout {{CURRENT_TAG}}
uv run python demo/app.py
# Open http://localhost:8000
```

The local demo produces identical evidence packs that can be verified with the same auditor tool.

---

## External Audits

Independent audits of this project are listed below.

| Date | Auditor Role | Version Audited | Report |
|------|--------------|-----------------|--------|
| 2026-01-03 | External safety lead (no prior context) | v0.2.1 archive, v0.2.0 demo | [Cold-Start Audit Report](../external_audits/manus_site_audit_2026-01-03/) |
| 2026-01-03 | Hostile link integrity auditor | v0.2.1 site | [Link Integrity Audit](../external_audits/manus_link_integrity_audit_2026-01-03/) |
| 2026-01-03 | Hostile acquisition auditor | v0.2.2 site | [Hostile Audit v0.2.2](../external_audits/manus_hostile_audit_v0.2.2_2026-01-03/) |

**Disclaimer:** These are independent audits. Findings may have been addressed in later versions. Audits do not constitute endorsements.

---

## Questions?

For technical questions about this verification process, see the [Hostile Demo Rehearsal](../hostile-rehearsal/) document which addresses common skeptical questions with prepared answers.
