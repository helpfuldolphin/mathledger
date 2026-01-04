# Operator Proof — v0.2.9 Gate 3 Fix Applied

**Source:** Claude B (operator verification)  
**Date:** 2026-01-04  
**Target:** v0.2.9  

## Summary

### Root Cause
v0.2.9 verifier used `SHA256(can(uvil_events))` rather than Merkle + domain separation used by v0.2.7+.

### Fix Applied
Copied v0.2.7 verifier (Merkle + domain separation) to v0.2.9 and updated version metadata.

### External Verification Proof (operator checks)

**Check 1: examples.json u_t**
- `u_t`: `0d1b61da395bb759b4558e1329e9ea561450e66d66421f88b540f7e828c0cd2d`
- Expected canonical value: matches v0.2.7

**Check 2: verifier includes Merkle + domain separation**
- `DOMAIN_UI_LEAF` present
- `merkleRoot()` present
- `computeUt()` uses `shaD(can(e), DOMAIN_UI_LEAF)` and `merkleRoot(lh)`

### Deployment Summary
- `/v0.2.9/evidence-pack/examples.json` deployed
- `/v0.2.9/evidence-pack/verify/` deployed
- `/versions/status.json` shows v0.2.9 current

### Smoke Checklist (expected after propagation)
Self-test should show:
- `SELF-TEST PASSED (3 vectors)`
- valid_boundary_demo: Expected PASS, Actual PASS, Test PASS
- tampered_ht_mismatch: Expected FAIL, Actual FAIL, Test PASS
- tampered_rt_mismatch: Expected FAIL, Actual FAIL, Test PASS
