# Pilot Evaluation Checklist

**Status:** BINDING
**Effective:** 2025-12-18
**Mode:** SHADOW-OBSERVE (invariant)
**Purpose:** Define what pilot evaluation produces and what "pass" means

---

## 1. Artifacts Produced

Pilot evaluation produces the following artifacts:

| Artifact | Description | Required |
|----------|-------------|----------|
| **Signed Manifest** | `manifest.json` with SHA-256 hashes, toolchain fingerprint, timestamps | YES |
| **Evidence Pack** | `evidence_pack/` directory with telemetry, logs, schema-compliant JSON | YES |
| **Provenance Chain** | Hash chain linking artifacts to execution context | YES |
| **Execution Log** | Timestamped record of pilot run (SHADOW mode throughout) | YES |
| **Audit Plane v0 Sample** | `A_t` attestation record (additive, non-authoritative) | OPTIONAL |

---

## 2. Signed Manifest

The manifest is the root artifact. It contains:

```json
{
  "schema_version": "1.0.0",
  "timestamp": "<ISO8601>",
  "mode": "SHADOW",
  "toolchain_fingerprint": "<sha256>",
  "uv_lock_hash": "<sha256>",
  "artifacts": {
    "evidence_pack": "<sha256>",
    "execution_log": "<sha256>"
  }
}
```

**Verification:** `sha256sum manifest.json` matches recorded hash.

---

## 3. Sample Evidence Pack Structure

```
evidence_pack/
├── manifest.json          # Root manifest (signed)
├── telemetry/
│   └── run_telemetry.json # Execution telemetry
├── logs/
│   └── execution.log      # Timestamped execution log
├── governance/
│   └── shadow_mode.json   # SHADOW mode attestation
└── external/              # (if external ingest used)
    └── ingest_manifest.json
```

All files are JSON, schema-compliant, and hash-verified.

---

## 4. Audit Plane v0 Sample (Optional)

If included, the `A_t` record contains:

```json
{
  "audit_plane_version": "0.1.0",
  "timestamp": "<ISO8601>",
  "mode": "SHADOW",
  "status": "OBSERVED",
  "note": "Additive, non-authoritative observation"
}
```

**Constraint:** Audit Plane v0 is additive only. It does not modify governance state or establish authority.

---

## 5. What "PASS" Means

A pilot evaluation **PASSES** if:

| Criterion | Verification |
|-----------|--------------|
| Code executes | No crashes, no unhandled exceptions |
| Artifacts present | Manifest, evidence pack, logs exist |
| Signatures verify | SHA-256 hashes match recorded values |
| Schema compliant | All JSON validates against declared schemas |
| SHADOW mode maintained | `mode: "SHADOW"` in all artifacts |
| Provenance intact | Hash chain is unbroken |

**"PASS" = operational execution completed, artifacts are well-formed and verifiable.**

---

## 6. What "PASS" Does NOT Mean

A pilot evaluation **PASS** does **NOT** indicate:

| Non-Claim | Explanation |
|-----------|-------------|
| Correctness | Execution does not imply the system is correct |
| Safety guarantee | No safety properties are validated or claimed |
| Accuracy | No measurement against ground truth |
| Learning | No adaptation or learning is demonstrated |
| Production readiness | No deployment authorization granted |
| External claims | Nothing authorizes announcements or papers |
| Baseline establishment | Observations are not comparison baselines |
| Gating authority | Pilot outcomes do not gate future phases |

**"PASS" = artifacts produced. "PASS" ≠ system validated.**

---

## 7. Verification Commands

```bash
# Verify manifest hash
sha256sum evidence_pack/manifest.json

# Verify SHADOW mode in all artifacts
grep -r '"mode"' evidence_pack/ | grep -v SHADOW && echo "FAIL" || echo "PASS"

# Verify schema compliance (if jq available)
jq empty evidence_pack/manifest.json && echo "Schema OK" || echo "Schema FAIL"
```

---

## 8. Evaluation Summary Template

After pilot completion, record:

```
PILOT EVALUATION SUMMARY

Date:           <ISO8601>
Mode:           SHADOW-OBSERVE
Duration:       <minutes>

ARTIFACTS:
- Manifest:     [PRESENT/MISSING]  Hash: <sha256>
- Evidence:     [PRESENT/MISSING]  Hash: <sha256>
- Logs:         [PRESENT/MISSING]  Hash: <sha256>
- Audit Plane:  [PRESENT/MISSING/SKIPPED]

VERIFICATION:
- Signatures:   [PASS/FAIL]
- Schema:       [PASS/FAIL]
- SHADOW mode:  [PASS/FAIL]

RESULT:         [PASS/FAIL]

NOTE: PASS indicates operational execution and artifact integrity.
      PASS does NOT indicate correctness, safety, or deployment readiness.
```

---

*Pilot evaluation produces artifacts, not authority.*
