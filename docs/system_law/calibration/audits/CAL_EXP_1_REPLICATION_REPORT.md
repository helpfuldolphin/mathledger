# CAL-EXP-1 Replication Report

**Attestation Type:** REPLICATION
**Attestation Date:** 2025-12-13
**Replication Verdict:** MATCH

---

## Executive Summary

CAL-EXP-1 was successfully replicated from a fresh checkout at commit `55bd258a8af327af5d12332b2544c91600dafa99`. All six hypotheses (H1–H6) passed under independent execution. The normalized Merkle root matches across three independent runs:

1. Original working tree run 1
2. Original working tree run 2
3. Fresh checkout replication (C:\dev\cal_exp_1_replication)

---

## Hypothesis Results

| Hypothesis | Perturbation | Expected Behavior | Observed Behavior | Result |
|------------|--------------|-------------------|-------------------|--------|
| H1 | P1 (ordering) | determinism_rate=1.0 (content unchanged) | determinism_rate=1.0 (hashes match) | **PASS** |
| H2 | P2 (content) | determinism_rate < 1.0 (content divergence) | determinism_rate=0.0 | **PASS** |
| H3 | P3 (height) | MonotoneViolation(type=height) produced | violations=1, types=['height'] | **PASS** |
| H4 | P4 (timestamp) | MonotoneViolation(type=timestamp) produced | violations=1, types=['timestamp'] | **PASS** |
| H5 | P5 (TDA threshold) | TDA red-flag logged with action=LOGGED_ONLY | red_flags=147, logged_only=147, control_flow_modified=False | **PASS** |
| H6 | P6 (tier skew) | tier_skew alarm produced (p_value < alpha) | alarm_count=1 | **PASS** |

---

## Cryptographic Evidence

### Hashes

| Artifact | Hash |
|----------|------|
| Toolchain | `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc` |
| Harness | `98834dc727c4508c7a067392ce7eb0e5bf0bee76b76121e2e304a74796e37b98` |
| Commit | `55bd258a8af327af5d12332b2544c91600dafa99` |

### Artifact Hashes

| File | SHA-256 |
|------|---------|
| h1_ordering_perturbation.jsonl | `7d4f87012d82248b45a88ac585732dbaa3609964653d370169756f8b5ecbdfe2` |
| h2_content_perturbation.jsonl | `3fdb110f549b9c446a3d7a9f150b10fbee1ac37965edbb89fff6c8342d9815fd` |
| h3_height_rejection.jsonl | `1be08636037acb96e989f7e47b85558f15c2c9eede0d7dbf89ff909ad4116864` |
| h4_timestamp_rejection.jsonl | `c288535686a7eaf19dd2e0316259ee9e5b4be92a0ea790e85a032b57d4b9c5e8` |
| h6_tier_skew.jsonl | `7e8e4333fcd89d63ff0882d9b2396d40d730e03bd4222e940f079e060a5ccf95` |

### Merkle Roots

| Type | Value |
|------|-------|
| Raw (includes timestamps) | `ea9820e02cfbccc2f9243ce14c3558cd8845e7c9e800487ff4ecf74c792e71c6` |
| Normalized (timestamps stripped) | `cb6278e65cf13388ee0062bf2f97b43bad1c085280d1f86461b2fea5be696c20` |

---

## Time-Variant Keys

The following keys are excluded from determinism comparison per the experiment design (Appendix: Determinism Exclusions):

- `timestamp` — Execution timestamp (ISO 8601)
- `created_at` — Creation timestamp
- `updated_at` — Update timestamp
- `run_timestamp` — Run-specific timestamp

Raw Merkle roots differ across runs due to these timestamp fields. Normalized Merkle roots (with time-variant keys stripped) are identical, confirming semantic reproducibility.

---

## Replication Protocol

1. Created fresh directory: `C:\dev\cal_exp_1_replication`
2. Cloned repository at commit `55bd258a8af327af5d12332b2544c91600dafa99`
3. Ran `uv sync` to install dependencies
4. Confirmed `shadow_mode=True` in harness configuration
5. Executed harness with PRNG seed 42
6. Compared normalized Merkle root against original runs

---

## Pre-Run Checks (Replication)

| Check | Result | Evidence |
|-------|--------|----------|
| Fixtures exist | PASS | C:\dev\cal_exp_1_replication\tests\fixtures |
| Toolchain hash computed | PASS | d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc |
| shadow_mode=True | PASS | FirstLightConfig.shadow_mode=True |
| Baseline determinism | PASS | hash1=24677a854f2e780f, hash2=24677a854f2e780f |

---

## Notes

- Normalized Merkle root verified identical across: (1) original working tree run 1, (2) original working tree run 2, (3) fresh checkout replication
- Raw roots differ due to timestamp variance as documented
- All required modules are now tracked in git (TDA, ht, synthetic, rfl) per commit `55bd258`
- Experiment complies with FORB-03 (shadow_mode enforcement)

---

## Attestation Reference

Machine-readable attestation: `CAL_EXP_1_REPLICATION_ATTESTATION.json`
