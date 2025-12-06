# AllBlue Gate Acceptance Tests

Acceptance criteria for dynamic lane discovery and epoch seal generation.

## Test 1: Dynamic Lane Addition

**Objective:** Verify that adding a new lane to `config/allblue_lanes.json` is automatically discovered by the gate.

**Steps:**
1. Add new lane to `config/allblue_lanes.json`:
   ```json
   {
     "id": "security_scan",
     "name": "Security Vulnerability Scan",
     "type": "ci_workflow",
     "required": false,
     "workflow": "security.yml",
     "jobs": ["trivy-scan"]
   }
   ```

2. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

3. Verify output includes new lane:
   ```
   Verifying lane: Security Vulnerability Scan (security_scan)...
   ```

**Expected Result:** Gate discovers and attempts to verify the new lane.

**Pass Criteria:** New lane appears in discovery output and fleet state JSON.

---

## Test 2: ABSTAIN on Missing Artifacts

**Objective:** Verify that gate ABSTAINS when required artifacts are missing.

**Steps:**
1. Remove required artifact:
   ```bash
   rm ci_verification/triple_hash_summary.json
   ```

2. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

3. Check exit code and output:
   ```bash
   echo $?  # Should be 1 (ABSTAIN)
   ```

**Expected Result:**
```
[ABSTAIN] AllBlue Gate: N required lane(s) missing artifacts
[INFO] Epoch Seal <hash>
```

**Pass Criteria:**
- Exit code: 1
- Status: ABSTAIN
- Message clearly indicates missing artifacts
- Epoch seal still computed and logged

---

## Test 3: Epoch Seal Stability Across Replay

**Objective:** Verify that epoch seal is deterministic and stable when replaying the same run.

**Steps:**
1. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

2. Record epoch hash from output:
   ```
   [PASS] Epoch Seal <hash1>
   ```

3. Run generator again without changing any inputs:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

4. Record second epoch hash:
   ```
   [PASS] Epoch Seal <hash2>
   ```

5. Compare hashes:
   ```bash
   # hash1 should equal hash2
   ```

**Expected Result:** Epoch seal is identical across multiple runs with same inputs.

**Pass Criteria:**
- hash1 == hash2
- RFC 8785 canonicalization ensures deterministic ordering
- No timestamp drift in epoch manifest (verification_signature is deterministic)

**Note:** Hygiene state hash may differ if git state changes (commit, branch, clean/dirty status).

---

## Test 4: RFC 8785 Canonicalization

**Objective:** Verify that fleet state JSON follows RFC 8785 canonicalization rules.

**Steps:**
1. Generate fleet state:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

2. Inspect canonicalized output:
   ```bash
   cat artifacts/allblue/fleet_state.json
   ```

3. Verify RFC 8785 compliance:
   - Keys sorted lexicographically
   - No whitespace between tokens
   - Single-line output
   - Deterministic number representation

**Expected Result:** Fleet state JSON is a single line with sorted keys and no whitespace.

**Pass Criteria:**
- `fleet_state.json` is single-line
- Keys appear in lexicographic order
- No spaces after `:` or `,`
- Human-readable version exists at `fleet_state_readable.json`

---

## Test 5: Optional Lane Handling

**Objective:** Verify that optional lanes do not block gate when missing.

**Steps:**
1. Ensure hygiene lane is marked as optional in config:
   ```json
   {
     "id": "hygiene",
     "required": false
   }
   ```

2. Run epoch seal generator (hygiene workflow may not exist):
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

3. Check status:
   ```bash
   echo $?  # Should be 0 if all required lanes pass
   ```

**Expected Result:**
```
Lane Summary:
  Required PASS:    5
  Required ABSTAIN: 0
  Optional PASS:    0
  Optional ABSTAIN: 1
```

**Pass Criteria:**
- Gate does not ABSTAIN due to optional lane failure
- Optional lane status recorded in fleet state
- Overall status determined only by required lanes

---

## Test 6: Fallback to Canonical 6 Lanes

**Objective:** Verify that gate falls back to canonical 6 lanes when config is missing.

**Steps:**
1. Rename config file:
   ```bash
   mv config/allblue_lanes.json config/allblue_lanes.json.bak
   ```

2. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

3. Check warning message:
   ```
   WARNING: Lane config not found at config/allblue_lanes.json, using canonical 6 lanes
   ```

**Expected Result:** Gate uses hardcoded canonical 6-lane configuration.

**Pass Criteria:**
- Warning message appears
- 6 lanes discovered (triple_hash, dual_attestation, unit_tests, uplift_omega, uplift_evaluation, hygiene)
- Gate continues execution with fallback config

---

## Test 7: Pass-Line Format Compliance

**Objective:** Verify that pass-line output matches required format.

**Steps:**
1. Run epoch seal generator with all lanes passing:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

2. Check output format:
   ```
   [PASS] ALL BLUE: sha256=<manifest_hash>
   [PASS] Epoch Seal <64-hex>
   ```

**Expected Result:** Output matches exact format specification.

**Pass Criteria:**
- First line: `[PASS] ALL BLUE: sha256=<hash>`
- Second line: `[PASS] Epoch Seal <hash>`
- Hash is 64 hexadecimal characters
- No extra whitespace or formatting

---

## Test 8: Hygiene State Hash Computation

**Objective:** Verify that hygiene state hash reflects repository state.

**Steps:**
1. Run on clean branch:
   ```bash
   git status  # Should be clean
   python3 scripts/generate_allblue_epoch_seal.py
   ```

2. Record hygiene hash:
   ```
   Hygiene hash: <hash1>
   ```

3. Make uncommitted change:
   ```bash
   echo "test" >> README.md
   python3 scripts/generate_allblue_epoch_seal.py
   ```

4. Record second hygiene hash:
   ```
   Hygiene hash: <hash2>
   ```

**Expected Result:** Hygiene hash changes when repository state changes.

**Pass Criteria:**
- hash1 != hash2 (dirty state detected)
- Hygiene hash includes: commit hash, branch name, clean/dirty status
- Format: `sha256("commit:<hash>|branch:<name>|clean:<bool>")`

---

## Test 9: Multi-Lane Verification

**Objective:** Verify that all 6 canonical lanes are verified correctly.

**Steps:**
1. Ensure all required artifacts exist:
   - `ci_verification/triple_hash_summary.json`
   - CI workflows: dual-attestation.yml, ci.yml, uplift-evaluation.yml

2. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

3. Check lane summary:
   ```
   Lane Summary:
     Required PASS:    5
     Required ABSTAIN: 0
     Optional PASS:    1
     Optional ABSTAIN: 0
   ```

**Expected Result:** All required lanes pass, optional lanes may pass or abstain.

**Pass Criteria:**
- 5 required lanes: triple_hash, dual_attestation, unit_tests, uplift_omega, uplift_evaluation
- 1 optional lane: hygiene
- Overall status: PASS
- Exit code: 0

---

## Test 10: Epoch Manifest Components

**Objective:** Verify that epoch manifest includes all required components.

**Steps:**
1. Run epoch seal generator:
   ```bash
   python3 scripts/generate_allblue_epoch_seal.py
   ```

2. Inspect fleet state:
   ```bash
   cat artifacts/allblue/fleet_state_readable.json | jq '.epoch_seal'
   ```

**Expected Result:**
```json
{
  "algorithm": "sha256",
  "canonicalization": "RFC8785",
  "components": ["lanes", "H_t", "hygiene_state_hash", "verification_signature"],
  "hash": "<64-hex>"
}
```

**Pass Criteria:**
- All 4 components present: lanes, H_t, hygiene_state_hash, verification_signature
- Algorithm: sha256
- Canonicalization: RFC8785
- Hash is 64 hexadecimal characters

---

## Acceptance Criteria Summary

**MUST PASS:**
1. Dynamic lane addition (Test 1)
2. ABSTAIN on missing artifacts (Test 2)
3. Epoch seal stability (Test 3)
4. RFC 8785 canonicalization (Test 4)
5. Pass-line format compliance (Test 7)

**SHOULD PASS:**
6. Optional lane handling (Test 5)
7. Fallback to canonical 6 (Test 6)
8. Hygiene state hash (Test 8)
9. Multi-lane verification (Test 9)
10. Epoch manifest components (Test 10)

**Definition of Done:**
- All MUST PASS tests pass
- At least 4/5 SHOULD PASS tests pass
- Documentation complete
- PR created with fork-safe pattern
