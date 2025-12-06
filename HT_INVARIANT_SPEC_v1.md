# HT Invariant Specification v1

> **STATUS: PHASE II — NOT YET IMPLEMENTED**
>
> This specification describes a **proposed future design** for dual-attestation hashing.
> It is **not used** in Evidence Pack v1 or the current First Organism implementation.
>
> **Current Phase I ground truth:**
> - `backend/attestation/dual_root.py` — actual attestation logic
> - `fo_baseline/attestation.json` and `fo_rfl/attestation.json` — sealed artifacts
>
> **RFL and Hₜ are independent systems:**
> - RFL logs (abstention rates, 1000-cycle Dyno Charts) do **not** affect Hₜ computation
> - Hₜ derives solely from First Organism dual-root attestation (`attestation.json`)
> - No Phase I RFL run triggers or validates the Hₜ invariant tests in this spec
>
> Do not cite this spec as evidence of "formally verified Hₜ" or "proven invariants."
> This document exists for future implementation guidance only.

---

## Dual-Attestation Hash (Hₜ) — Formal Definition

This document specifies the invariants for a **proposed** dual-attestation hash system, which would cryptographically bind proof attestations (Rₜ) with UI event attestations (Uₜ) into a single verifiable commitment (Hₜ).

---

## 1. Definitions

### 1.1 Proof Root (Rₜ)

```
Rₜ = Merkle(proofs)
```

Where:
- `proofs` is an ordered list of proof attestations from a derivation run
- Each proof leaf is computed as: `SHA256(DOMAIN_PROOF || canonical_json(proof))`
- `DOMAIN_PROOF = b"MathLedger:Proof:v1:"`

### 1.2 UI Event Root (Uₜ)

```
Uₜ = Merkle(ui_events)
```

Where:
- `ui_events` is an ordered list of UI interaction events
- Each event leaf is computed as: `SHA256(DOMAIN_UI || canonical_json(event))`
- `DOMAIN_UI = b"MathLedger:UIEvent:v1:"`

### 1.3 Dual-Attestation Hash (Hₜ)

```
Hₜ = SHA256(DOMAIN_HT || Rₜ || Uₜ)
```

Where:
- `DOMAIN_HT = b"MathLedger:DualAttest:v1:"`
- `Rₜ` is the 32-byte proof Merkle root
- `Uₜ` is the 32-byte UI event Merkle root
- `||` denotes byte concatenation

---

## 2. Required Invariants

### 2.1 Deterministic Canonicalization

**INV-CANON-1**: All JSON objects MUST be canonicalized before hashing.

Canonical JSON requirements:
- Keys sorted lexicographically (Unicode code point order)
- No whitespace between tokens
- No trailing commas
- UTF-8 encoding
- Numbers without unnecessary leading zeros or trailing decimal zeros
- Strings escaped per RFC 8259

```python
def canonical_json(obj: dict) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')
```

### 2.2 Domain-Separated Hashing

**INV-DOMAIN-1**: All hash computations MUST use domain separation prefixes.

| Context | Domain Prefix |
|---------|---------------|
| Proof leaf | `b"MathLedger:Proof:v1:"` |
| UI event leaf | `b"MathLedger:UIEvent:v1:"` |
| Dual attestation | `b"MathLedger:DualAttest:v1:"` |
| Merkle internal node | `b"MathLedger:MerkleNode:v1:"` |

**Rationale**: Domain separation prevents cross-protocol attacks where a valid hash in one context could be reinterpreted in another.

### 2.3 Sorted Leaf Order

**INV-ORDER-1**: Merkle tree leaves MUST be sorted before tree construction.

Sort order:
- Primary: lexicographic sort by the hex-encoded leaf hash
- This ensures identical leaf sets always produce identical roots

```python
def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return SHA256(DOMAIN_MERKLE_NODE + b"empty")
    sorted_leaves = sorted(leaves)  # Lexicographic byte order
    return build_merkle_tree(sorted_leaves)
```

### 2.4 Stable Casing

**INV-CASE-1**: All hexadecimal representations MUST use lowercase.

```python
def to_hex(data: bytes) -> str:
    return data.hex()  # Always lowercase in Python

# CORRECT: "a1b2c3d4..."
# WRONG:   "A1B2C3D4..."
```

### 2.5 No Whitespace in Canonical JSON

**INV-WS-1**: Canonical JSON MUST contain no extraneous whitespace.

```python
# CORRECT
'{"a":1,"b":2}'

# WRONG
'{"a": 1, "b": 2}'
'{ "a" : 1 , "b" : 2 }'
```

### 2.6 Verifiable Recomputation

**INV-VERIFY-1**: Given the same inputs, Hₜ MUST be independently recomputable.

Required stored data for verification:
1. Complete list of proof objects (or their canonical JSON)
2. Complete list of UI event objects (or their canonical JSON)
3. The domain prefixes used (versioned)

```python
def verify_ht(proofs: list[dict], ui_events: list[dict], claimed_ht: bytes) -> bool:
    recomputed_rt = compute_rt(proofs)
    recomputed_ut = compute_ut(ui_events)
    recomputed_ht = compute_ht(recomputed_rt, recomputed_ut)
    return recomputed_ht == claimed_ht
```

---

## 3. Merkle Tree Construction

### 3.1 Algorithm

```
function build_merkle_tree(leaves: bytes[]) -> bytes:
    if len(leaves) == 0:
        return SHA256(DOMAIN_MERKLE_NODE || "empty")
    if len(leaves) == 1:
        return leaves[0]

    # Pad to power of 2 by duplicating last leaf
    while not is_power_of_2(len(leaves)):
        leaves.append(leaves[-1])

    # Build tree bottom-up
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left, right = leaves[i], leaves[i+1]
            # Ensure consistent ordering within node
            if left > right:
                left, right = right, left
            parent = SHA256(DOMAIN_MERKLE_NODE || left || right)
            next_level.append(parent)
        leaves = next_level

    return leaves[0]
```

### 3.2 Empty Tree Handling

**INV-EMPTY-1**: Empty input lists produce a well-defined "empty" root.

```python
EMPTY_RT = SHA256(b"MathLedger:MerkleNode:v1:empty")
EMPTY_UT = SHA256(b"MathLedger:MerkleNode:v1:empty")
```

---

## 4. Proof Object Schema

```json
{
  "statement_hash": "<64-char lowercase hex>",
  "statement_text": "<normalized formula>",
  "method": "modus_ponens|axiom_instance|truth_table",
  "parent_hashes": ["<hex>", ...],
  "prover": "lean4|truth_table",
  "timestamp_ms": 1234567890123,
  "run_id": "<uuid>"
}
```

Required fields for hashing: `statement_hash`, `method`, `parent_hashes`, `timestamp_ms`

---

## 5. UI Event Object Schema

```json
{
  "event_type": "view|query|export",
  "target_hash": "<64-char lowercase hex>",
  "timestamp_ms": 1234567890123,
  "session_id": "<uuid>",
  "user_agent_hash": "<32-char lowercase hex>"
}
```

Required fields for hashing: `event_type`, `target_hash`, `timestamp_ms`

---

## 6. Implementation Reference

### 6.1 Computing Rₜ

```python
DOMAIN_PROOF = b"MathLedger:Proof:v1:"
DOMAIN_MERKLE_NODE = b"MathLedger:MerkleNode:v1:"

def compute_rt(proofs: list[dict]) -> bytes:
    leaves = []
    for proof in proofs:
        canonical = canonical_json(proof)
        leaf = hashlib.sha256(DOMAIN_PROOF + canonical).digest()
        leaves.append(leaf)
    return merkle_root(leaves)
```

### 6.2 Computing Uₜ

```python
DOMAIN_UI = b"MathLedger:UIEvent:v1:"

def compute_ut(ui_events: list[dict]) -> bytes:
    leaves = []
    for event in ui_events:
        canonical = canonical_json(event)
        leaf = hashlib.sha256(DOMAIN_UI + canonical).digest()
        leaves.append(leaf)
    return merkle_root(leaves)
```

### 6.3 Computing Hₜ

```python
DOMAIN_HT = b"MathLedger:DualAttest:v1:"

def compute_ht(rt: bytes, ut: bytes) -> bytes:
    return hashlib.sha256(DOMAIN_HT + rt + ut).digest()
```

---

## 7. Security Considerations

### 7.1 Collision Resistance

The dual-attestation hash inherits SHA-256's collision resistance. An attacker cannot find two different (Rₜ, Uₜ) pairs that produce the same Hₜ.

### 7.2 Domain Separation

Domain prefixes prevent:
- Proof leaves being mistaken for UI event leaves
- Internal Merkle nodes being mistaken for leaves
- Cross-version attacks when schemas evolve

### 7.3 Timestamp Binding

Timestamps in both proofs and UI events provide temporal ordering and prevent replay attacks.

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2025-11-30 | Initial specification (Phase II draft) |
| v1.1 | 2025-12-05 | Phase II Extensions: manifest composition invariants (§12), seed schedule rules (§13), Hₜ series hashing (§14) |
| v1.2 | 2025-12-05 | Phase II Extensions: Manifest-Preregistration binding (§12.4), slice-specific success metric binding (§12.5), MDAP schedule attestation (§13.6), Hₜ series cryptographic guarantees (§14.6) |

---

## 9. Compliance Checklist

> **Note:** This checklist is for future implementation. None of these items are
> verified or enforced in the current Phase I codebase.

- [ ] All JSON canonicalized with sorted keys, no whitespace
- [ ] All hex strings lowercase
- [ ] Domain separation applied at all hash points
- [ ] Merkle leaves sorted before tree construction
- [ ] Empty trees handled with domain-separated "empty" marker
- [ ] Hₜ recomputable from stored proofs and UI events

---

## 10. Phase I vs Phase II Relationship

| Aspect | Phase I (Current) | Phase II (This Spec) |
|--------|-------------------|----------------------|
| Attestation file | `attestation.json` | Not yet implemented |
| Hash scheme | `dual_root.py` Merkle root | Hₜ = SHA256(domain \|\| Rₜ \|\| Uₜ) |
| UI events | Not collected | Uₜ Merkle root |
| Domain separation | Partial | Full (4 domains) |
| Formal verification | None | Planned |

**Do not conflate these two systems in any claims or documentation.**

---

## 11. RFL Independence

RFL (Reflective Feedback Loop) and Hₜ are **completely independent systems**:

| System | Purpose | Data Source | Phase I Status |
|--------|---------|-------------|----------------|
| RFL | Abstention rate tracking, Dyno Charts | `rfl_logs/`, 1000-cycle metrics | Implemented, measured |
| Hₜ | Cryptographic attestation binding | `attestation.json` (dual-root) | Spec only (Phase II) |

**Key clarifications:**
- RFL logs do **not** feed into Hₜ computation
- RFL abstention metrics are observational; Hₜ is cryptographic commitment
- The 1000-cycle Dyno Chart (baseline vs RFL) is RFL evidence, not Hₜ evidence
- No Phase I RFL run triggers, validates, or depends on the Hₜ invariant tests

This separation is intentional: RFL measures *behavior*, Hₜ attests *provenance*.

---

---

# PHASE II EXTENSIONS — NOT RUN IN PHASE I

> **STATUS: PHASE II SPECIFICATION — NOT YET IMPLEMENTED**
>
> All content below this marker is Phase II design only. None of these invariants
> are active, tested, or enforced in Phase I artifacts. Do not cite this section
> as evidence of implemented functionality.

---

## 12. Phase II Manifest Composition Invariants

### 12.1 Manifest Structure Requirements

**INV-MANIFEST-1**: Every Phase II experiment MUST produce a manifest conforming to the schema below.

```json
{
  "meta": {
    "version": "2.0.0",
    "type": "mathledger_experiment_manifest",
    "phase": "II"
  },
  "experiment": {
    "id": "<experiment_id>",
    "type": "uplift_u2",
    "family": "<slice_uplift_goal|slice_uplift_sparse|slice_uplift_tree|slice_uplift_dependency>",
    "timestamp_utc": "<ISO8601>",
    "status": "<preregistered|running|complete|failed>"
  },
  "seed_schedule": { /* see Section 13 */ },
  "ht_series": { /* see Section 14 */ },
  "provenance": { /* unchanged from v1 */ },
  "configuration": { /* slice-specific params */ },
  "artifacts": { /* logs, figures */ },
  "signature": "<base64_encoded_governance_signature>"
}
```

**INV-MANIFEST-2**: Manifest `meta.phase` MUST equal `"II"` for all Phase II experiments.

**INV-MANIFEST-3**: Manifest `experiment.family` MUST match exactly one of:
- `"slice_uplift_goal"`
- `"slice_uplift_sparse"`
- `"slice_uplift_tree"`
- `"slice_uplift_dependency"`

**INV-MANIFEST-4**: Phase II manifests MUST NOT reference Phase I log files.

**INV-MANIFEST-5**: The manifest MUST be signed by the Phase II Governance key. The signature is computed over the SHA-256 hash of the canonical JSON representation of the manifest, with the `signature` field removed during canonicalization.

The following paths are **forbidden** in `artifacts.logs[].path`:
- `fo_rfl*.jsonl`
- `fo_baseline*.jsonl`
- Any path containing `phase_i` or `first_organism` unless explicitly marked `reference_only: true`

```python
def validate_no_phase1_references(manifest: dict) -> bool:
    forbidden_patterns = [
        r"fo_rfl.*\.jsonl",
        r"fo_baseline.*\.jsonl",
        r"phase_i",
        r"first_organism"
    ]
    for log in manifest.get("artifacts", {}).get("logs", []):
        path = log.get("path", "")
        if log.get("reference_only", False):
            continue
        for pattern in forbidden_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return False
    return True
```

### 12.2 Paired Run Binding

**INV-PAIRED-1**: For any uplift experiment, baseline and RFL runs MUST share identical `seed_schedule.schedule_hash`.

**INV-PAIRED-2**: Paired runs MUST be recorded in the same manifest OR cross-referenced via `paired_manifest_hash`.

```json
{
  "paired_runs": {
    "baseline": {
      "manifest_path": "results/uplift_u2_goal_baseline_manifest.json",
      "manifest_hash": "<sha256>"
    },
    "rfl": {
      "manifest_path": "results/uplift_u2_goal_rfl_manifest.json",
      "manifest_hash": "<sha256>"
    },
    "binding_proof": "<sha256(baseline_hash || rfl_hash)>"
  }
}
```

### 12.3 Slice Configuration Binding

**INV-SLICE-1**: Manifest MUST include the slice configuration hash.

```python
def slice_config_hash(slice_config: dict) -> bytes:
    canonical = canonical_json(slice_config)
    return hashlib.sha256(b"MathLedger:SliceConfig:v2:" + canonical).digest()
```

**INV-SLICE-2**: Slice configuration MUST be immutable after preregistration.

If `preregistration.status == "EXECUTED"`, the `slice_config_hash` recorded at preregistration MUST match the runtime slice config.

### 12.4 Manifest-Preregistration Binding

**INV-PREREG-1**: Every Phase II manifest MUST cryptographically bind to its preregistration document.

```json
{
  "preregistration": {
    "source_file": "experiments/prereg/PREREG_UPLIFT_U2.yaml",
    "source_hash": "<sha256 of PREREG_UPLIFT_U2.yaml at execution start>",
    "experiment_template": "<slice_uplift_goal|slice_uplift_sparse|...>",
    "status": "<NOT_YET_EXECUTED|EXECUTING|EXECUTED|FAILED>",
    "binding_hash": "<sha256(source_hash || experiment_id || timestamp_utc)>"
  }
}
```

**INV-PREREG-2**: The `preregistration.source_hash` MUST be computed before execution begins and MUST NOT change during execution.

```python
DOMAIN_PREREG_BIND = b"MathLedger:PreregBinding:v2:"

def compute_prereg_binding(
    source_hash: bytes,
    experiment_id: str,
    timestamp_utc: str
) -> bytes:
    """Compute the binding hash that locks manifest to preregistration."""
    payload = (
        DOMAIN_PREREG_BIND +
        source_hash +
        experiment_id.encode("utf-8") +
        timestamp_utc.encode("utf-8")
    )
    return hashlib.sha256(payload).digest()

def verify_prereg_binding(manifest: dict, prereg_path: Path) -> bool:
    """Verify manifest is bound to the claimed preregistration."""
    # Recompute source hash
    with open(prereg_path, "rb") as f:
        actual_source_hash = hashlib.sha256(f.read()).digest()

    claimed_source_hash = bytes.fromhex(
        manifest["preregistration"]["source_hash"]
    )

    if actual_source_hash != claimed_source_hash:
        return False  # Preregistration was modified

    # Verify binding hash
    expected_binding = compute_prereg_binding(
        claimed_source_hash,
        manifest["experiment"]["id"],
        manifest["experiment"]["timestamp_utc"]
    )
    claimed_binding = bytes.fromhex(
        manifest["preregistration"]["binding_hash"]
    )

    return expected_binding == claimed_binding
```

**INV-PREREG-3**: Once `preregistration.status` transitions to `"EXECUTING"`, no changes to the preregistration file are permitted until the experiment completes or fails.

**INV-PREREG-4**: The manifest MUST record which slice template from `PREREG_UPLIFT_U2.yaml` was used.

### 12.5 Slice-Specific Success Metric Binding

**INV-SUCCESS-1**: Each slice family MUST define exactly one success metric kind.

| Slice Family | Success Metric Kind | Definition |
|--------------|---------------------|------------|
| `slice_uplift_goal` | `goal_hit` | Cycle succeeds if ≥1 target hash was derived |
| `slice_uplift_sparse` | `density` | Cycle succeeds if (verified / candidates) ≥ threshold |
| `slice_uplift_tree` | `chain_length` | Cycle succeeds if a proof chain of length ≥ L was built |
| `slice_uplift_dependency` | `multi_goal` | Cycle succeeds if all required goals were hit |

**INV-SUCCESS-2**: The success metric parameters MUST be recorded in the manifest and MUST match preregistration.

```json
{
  "success_metric": {
    "kind": "goal_hit",
    "parameters": {
      "target_hashes": [
        "a1b2c3d4...",
        "e5f6g7h8..."
      ],
      "min_goal_hits": 1,
      "min_total_verified": 3
    },
    "parameters_hash": "<sha256 of canonical JSON parameters>"
  }
}
```

**INV-SUCCESS-3**: The `success_metric.parameters_hash` MUST be computed via domain-separated hashing.

```python
DOMAIN_SUCCESS_METRIC = b"MathLedger:SuccessMetric:v2:"

def success_metric_hash(metric_kind: str, parameters: dict) -> bytes:
    """Hash the success metric parameters for binding."""
    payload = {
        "kind": metric_kind,
        "parameters": parameters
    }
    canonical = canonical_json(payload)
    return hashlib.sha256(DOMAIN_SUCCESS_METRIC + canonical).digest()
```

**INV-SUCCESS-4**: Success evaluation MUST be deterministic and verifiable.

For each success metric kind, the evaluation function signature is:

```python
def evaluate_success(
    cycle_log: dict,
    parameters: dict
) -> tuple[bool, dict]:
    """
    Evaluate whether a cycle succeeded.

    Returns:
        (success: bool, evidence: dict)

    The evidence dict MUST contain sufficient data to independently
    verify the success determination.
    """
    ...
```

**INV-SUCCESS-5**: Each cycle log entry MUST include the success evaluation result and evidence.

```json
{
  "cycle": 42,
  "rt": "<merkle root hex>",
  "success": true,
  "success_evidence": {
    "kind": "goal_hit",
    "target_hits": ["a1b2c3d4..."],
    "verified_count": 5,
    "evaluation_hash": "<sha256 of deterministic re-evaluation>"
  }
}
```

---

## 13. Seed Schedule Invariants

### 13.1 MDAP Seed Foundation

**DEF-MDAP**: The MDAP (MathLedger Deterministic Attestation Protocol) seed is the root of all determinism.

```
MDAP_SEED = 0x4D444150  # ASCII "MDAP" as 32-bit integer = 1296318800
```

**INV-SEED-1**: All Phase II experiments MUST derive cycle seeds from MDAP_SEED.

### 13.2 Cycle Seed Derivation

**DEF-CYCLE-SEED**: For cycle `c` in experiment `e`:

```
cycle_seed(c, e) = SHA256(
    DOMAIN_CYCLE_SEED ||
    BE32(MDAP_SEED) ||
    BE32(c) ||
    experiment_id_bytes(e)
)
```

Where:
- `DOMAIN_CYCLE_SEED = b"MathLedger:CycleSeed:v2:"`
- `BE32(x)` = 32-bit big-endian encoding
- `experiment_id_bytes(e)` = UTF-8 encoding of experiment ID

**INV-SEED-2**: Cycle seeds MUST be deterministic: same (MDAP_SEED, c, e) → same seed.

**INV-SEED-3**: Cycle seeds MUST be independent: knowing seed(c) reveals nothing about seed(c+1) without MDAP_SEED.

**INV-SEED-4**: All sources of randomness used within a cycle `c` (e.g., for baseline policy candidate shuffling) MUST be derived from a PRNG (Pseudo-Random Number Generator) initialized **only** with `cycle_seed(c, e)`. No other source of entropy is permitted.

```python
import hashlib
import struct

DOMAIN_CYCLE_SEED = b"MathLedger:CycleSeed:v2:"
MDAP_SEED = 0x4D444150

def cycle_seed(cycle_index: int, experiment_id: str) -> bytes:
    payload = (
        DOMAIN_CYCLE_SEED +
        struct.pack(">I", MDAP_SEED) +
        struct.pack(">I", cycle_index) +
        experiment_id.encode("utf-8")
    )
    return hashlib.sha256(payload).digest()

def cycle_seed_int(cycle_index: int, experiment_id: str) -> int:
    """Return seed as 64-bit integer for RNG initialization."""
    seed_bytes = cycle_seed(cycle_index, experiment_id)
    return int.from_bytes(seed_bytes[:8], "big")
```

### 13.3 Schedule Hash

**DEF-SCHEDULE-HASH**: The schedule hash commits to the entire seed sequence.

```
schedule_hash(n, e) = SHA256(
    DOMAIN_SCHEDULE ||
    BE32(n) ||
    cycle_seed(0, e) ||
    cycle_seed(1, e) ||
    ... ||
    cycle_seed(n-1, e)
)
```

Where:
- `DOMAIN_SCHEDULE = b"MathLedger:SeedSchedule:v2:"`
- `n` = total number of cycles

**INV-SCHEDULE-1**: The schedule hash MUST be recorded in the manifest before execution begins.

**INV-SCHEDULE-2**: Post-execution, the manifest MUST include:
- `seed_schedule.schedule_hash` (pre-computed)
- `seed_schedule.verification_hash` (recomputed post-run)
- `seed_schedule.match` (boolean: did they match?)

```json
{
  "seed_schedule": {
    "mdap_seed": "0x4D444150",
    "cycles": 500,
    "experiment_id": "uplift_u2_goal_001",
    "schedule_hash": "<sha256 pre-computed>",
    "verification_hash": "<sha256 post-computed>",
    "match": true,
    "first_cycle_seed": "<hex of cycle_seed(0, e)>",
    "last_cycle_seed": "<hex of cycle_seed(n-1, e)>"
  }
}
```

### 13.4 Schedule Verification

**INV-SCHEDULE-3**: Schedule verification MUST be independently recomputable.

```python
def verify_schedule(manifest: dict) -> bool:
    n = manifest["seed_schedule"]["cycles"]
    e = manifest["seed_schedule"]["experiment_id"]
    mdap = int(manifest["seed_schedule"]["mdap_seed"], 16)

    # Recompute schedule hash
    hasher = hashlib.sha256()
    hasher.update(b"MathLedger:SeedSchedule:v2:")
    hasher.update(struct.pack(">I", n))

    for c in range(n):
        seed = cycle_seed(c, e)
        hasher.update(seed)

    recomputed = hasher.hexdigest()
    claimed = manifest["seed_schedule"]["schedule_hash"]

    return recomputed == claimed
```

### 13.5 Baseline vs RFL Seed Parity

**INV-SEED-PARITY-1**: For paired baseline/RFL runs, both MUST use identical cycle seeds.

**INV-SEED-PARITY-2**: The difference MUST be only in candidate ordering, not seed selection.

```
For all c in [0, n):
    cycle_seed(c, "uplift_u2_goal_001_baseline") = cycle_seed(c, "uplift_u2_goal_001")
    cycle_seed(c, "uplift_u2_goal_001_rfl")      = cycle_seed(c, "uplift_u2_goal_001")
```

Implementation note: Use the base experiment ID (without `_baseline`/`_rfl` suffix) for seed computation.

### 13.6 MDAP Schedule Attestation

**INV-MDAP-ATTEST-1**: The complete seed schedule MUST be attested in a dedicated attestation record before execution.

```json
{
  "mdap_attestation": {
    "version": "v2",
    "domain": "MathLedger:MDAPAttestation:v2:",
    "mdap_seed": "0x4D444150",
    "experiment_id": "uplift_u2_goal_001",
    "total_cycles": 500,
    "schedule_hash": "<sha256>",
    "attestation_timestamp_utc": "<ISO8601>",
    "attestation_hash": "<sha256 of this record>"
  }
}
```

**INV-MDAP-ATTEST-2**: The attestation hash MUST be computed as:

```python
DOMAIN_MDAP_ATTEST = b"MathLedger:MDAPAttestation:v2:"

def compute_mdap_attestation(
    mdap_seed: int,
    experiment_id: str,
    total_cycles: int,
    schedule_hash: bytes,
    timestamp_utc: str
) -> bytes:
    """
    Compute the MDAP attestation hash.

    This hash commits to the entire deterministic schedule before
    any cycles execute.
    """
    payload = (
        DOMAIN_MDAP_ATTEST +
        struct.pack(">I", mdap_seed) +
        experiment_id.encode("utf-8") +
        struct.pack(">I", total_cycles) +
        schedule_hash +
        timestamp_utc.encode("utf-8")
    )
    return hashlib.sha256(payload).digest()
```

**INV-MDAP-ATTEST-3**: The MDAP attestation MUST be recorded to the manifest BEFORE the first cycle begins.

**INV-MDAP-ATTEST-4**: Post-execution, the manifest MUST include a verification section:

```json
{
  "mdap_verification": {
    "pre_execution_attestation_hash": "<hash from mdap_attestation>",
    "post_execution_recomputed_hash": "<recomputed after all cycles>",
    "match": true,
    "cycles_executed": 500,
    "first_cycle_timestamp_utc": "<ISO8601>",
    "last_cycle_timestamp_utc": "<ISO8601>"
  }
}
```

**INV-MDAP-ATTEST-5**: If `mdap_verification.match` is `false`, the experiment MUST be marked as `"FAILED"` with reason `"MDAP_SCHEDULE_MISMATCH"`.

### 13.7 Seed Schedule Determinism Enforcement

**INV-DETERMINISM-1**: The baseline policy random shuffle MUST use ONLY the cycle seed as entropy source.

```python
import random

def baseline_shuffle(candidates: list, cycle_index: int, experiment_id: str) -> list:
    """
    Shuffle candidates using deterministic cycle seed.

    CRITICAL: No other entropy source is permitted.
    """
    seed_int = cycle_seed_int(cycle_index, experiment_id)
    rng = random.Random(seed_int)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    return shuffled
```

**INV-DETERMINISM-2**: The RFL policy candidate ordering MUST be deterministic given:
1. The policy weights at cycle start
2. The candidate feature vectors
3. The cycle seed (for tie-breaking only)

```python
def rfl_order(
    candidates: list,
    policy_weights: dict,
    cycle_index: int,
    experiment_id: str
) -> list:
    """
    Order candidates by RFL policy score.

    Uses cycle seed ONLY for tie-breaking when scores are equal.
    """
    seed_int = cycle_seed_int(cycle_index, experiment_id)
    rng = random.Random(seed_int)

    def score_with_tiebreak(candidate):
        base_score = compute_policy_score(candidate, policy_weights)
        # Add tiny random tiebreaker (< 1e-10 to avoid affecting ordering)
        tiebreak = rng.random() * 1e-12
        return (base_score, tiebreak)

    return sorted(candidates, key=score_with_tiebreak, reverse=True)
```

**INV-DETERMINISM-3**: Any violation of determinism (same inputs → different outputs) MUST fail the experiment with reason `"DETERMINISM_VIOLATION"`.

---

## 14. Hₜ Series Hashing Invariants

### 14.1 Per-Cycle Hₜ

**DEF-HT-CYCLE**: Each cycle `c` produces an attestation hash Hₜ(c).

```
Hₜ(c) = SHA256(
    DOMAIN_HT_CYCLE ||
    BE32(c) ||
    Rₜ(c) ||
    cycle_seed(c, e)
)
```

Where:
- `DOMAIN_HT_CYCLE = b"MathLedger:HtCycle:v2:"`
- `Rₜ(c)` = proof Merkle root for cycle `c` (32 bytes)
- Phase II omits Uₜ in per-cycle Hₜ (no UI events during batch experiments)

**INV-HT-CYCLE-1**: Hₜ(c) MUST be computed and recorded for every cycle.

**INV-HT-CYCLE-2**: Hₜ(c) MUST include the cycle seed to bind attestation to schedule.

### 14.2 Hₜ Series Chain

**DEF-HT-CHAIN**: The Hₜ series forms a hash chain.

```
Hₜ_chain(0) = Hₜ(0)
Hₜ_chain(c) = SHA256(DOMAIN_HT_CHAIN || Hₜ_chain(c-1) || Hₜ(c))  for c > 0
```

Where:
- `DOMAIN_HT_CHAIN = b"MathLedger:HtChain:v2:"`

**INV-HT-CHAIN-1**: The chain is append-only; earlier entries cannot be modified without invalidating all subsequent entries.

**INV-HT-CHAIN-2**: The final chain value Hₜ_chain(n-1) MUST be recorded in the manifest.

```python
DOMAIN_HT_CYCLE = b"MathLedger:HtCycle:v2:"
DOMAIN_HT_CHAIN = b"MathLedger:HtChain:v2:"

def compute_ht_cycle(cycle: int, rt: bytes, cycle_seed: bytes) -> bytes:
    payload = (
        DOMAIN_HT_CYCLE +
        struct.pack(">I", cycle) +
        rt +
        cycle_seed
    )
    return hashlib.sha256(payload).digest()

def compute_ht_chain(ht_series: list[bytes]) -> bytes:
    """Compute cumulative chain hash from series of per-cycle Hₜ values."""
    if not ht_series:
        return hashlib.sha256(b"MathLedger:HtChain:v2:empty").digest()

    chain = ht_series[0]
    for ht in ht_series[1:]:
        chain = hashlib.sha256(DOMAIN_HT_CHAIN + chain + ht).digest()

    return chain
```

### 14.3 Hₜ Series Manifest Entry

**INV-HT-MANIFEST-1**: Manifest MUST include Hₜ series summary.

```json
{
  "ht_series": {
    "version": "v2",
    "domain_cycle": "MathLedger:HtCycle:v2:",
    "domain_chain": "MathLedger:HtChain:v2:",
    "total_cycles": 500,
    "ht_first": "<hex of Hₜ(0)>",
    "ht_last": "<hex of Hₜ(n-1)>",
    "ht_chain_final": "<hex of Hₜ_chain(n-1)>",
    "checkpoints": [
      {"cycle": 0, "ht": "<hex>", "chain": "<hex>"},
      {"cycle": 100, "ht": "<hex>", "chain": "<hex>"},
      {"cycle": 200, "ht": "<hex>", "chain": "<hex>"},
      {"cycle": 300, "ht": "<hex>", "chain": "<hex>"},
      {"cycle": 400, "ht": "<hex>", "chain": "<hex>"},
      {"cycle": 499, "ht": "<hex>", "chain": "<hex>"}
    ]
  }
}
```

**INV-HT-MANIFEST-2**: Checkpoints MUST be recorded at intervals ≤ 100 cycles.

**INV-HT-MANIFEST-3**: First and last cycle MUST always be checkpointed.

### 14.4 Series Verification Protocol

**INV-HT-VERIFY-1**: Given the log file and manifest, the Hₜ series MUST be independently recomputable.

```python
def verify_ht_series(log_path: Path, manifest: dict) -> bool:
    """Verify Hₜ series from log matches manifest claims."""
    experiment_id = manifest["experiment"]["id"]

    # Read per-cycle Rₜ values from log
    rt_series = []
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            rt_series.append(bytes.fromhex(record["rt"]))

    # Recompute Hₜ series
    ht_series = []
    for c, rt in enumerate(rt_series):
        seed = cycle_seed(c, experiment_id)
        ht = compute_ht_cycle(c, rt, seed)
        ht_series.append(ht)

    # Verify chain
    chain_final = compute_ht_chain(ht_series)

    claimed_chain = bytes.fromhex(manifest["ht_series"]["ht_chain_final"])
    return chain_final == claimed_chain
```

### 14.5 Delta-Hₜ Observables

**DEF-DELTA-HT**: The ΔHₜ between successive cycles.

```
ΔHₜ(c) = Hₜ(c) ⊕ Hₜ(c-1)  for c > 0
ΔHₜ(0) = Hₜ(0) ⊕ ZERO_HASH
```

Where `⊕` is bitwise XOR and `ZERO_HASH` is 32 zero bytes.

**INV-DELTA-HT-1**: ΔHₜ is observational only; it does not affect attestation validity.

**INV-DELTA-HT-2**: ΔHₜ MUST NOT be used in pass/fail decisions for Hₜ invariant tests.

```python
def compute_delta_ht(ht_prev: bytes, ht_curr: bytes) -> bytes:
    """XOR of successive Hₜ values for observational metrics."""
    return bytes(a ^ b for a, b in zip(ht_prev, ht_curr))

def delta_magnitude(delta: bytes) -> int:
    """Count of differing bits (Hamming weight of delta)."""
    return sum(bin(b).count('1') for b in delta)
```

### 14.6 Hₜ Series Cryptographic Guarantees

**INV-HT-CRYPTO-1**: The Hₜ series provides the following cryptographic guarantees:

| Guarantee | Definition | Enforcement |
|-----------|------------|-------------|
| **Immutability** | Once Hₜ(c) is recorded, it cannot be changed without detection | Hash chain invalidation |
| **Ordering** | Hₜ(c) must follow Hₜ(c-1) in the chain | Chain structure |
| **Binding** | Hₜ(c) is bound to both Rₜ(c) and cycle_seed(c) | Domain-separated hash |
| **Non-repudiation** | Manifest signer cannot deny producing Hₜ_chain | Signature over manifest |

**INV-HT-CRYPTO-2**: Tampering with any single Hₜ(c) MUST be detectable via chain verification.

```python
def detect_tampering(
    log_path: Path,
    manifest: dict,
    tampered_cycle: int
) -> bool:
    """
    Demonstrate that tampering with cycle c is detectable.

    If Hₜ(c) is modified, Hₜ_chain(c) through Hₜ_chain(n-1) will
    all be invalid.
    """
    experiment_id = manifest["experiment"]["id"]
    checkpoints = {
        cp["cycle"]: bytes.fromhex(cp["chain"])
        for cp in manifest["ht_series"]["checkpoints"]
    }

    # Recompute from logs
    rt_series = []
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            rt_series.append(bytes.fromhex(record["rt"]))

    ht_series = []
    for c, rt in enumerate(rt_series):
        seed = cycle_seed(c, experiment_id)
        ht = compute_ht_cycle(c, rt, seed)
        ht_series.append(ht)

    chain = ht_series[0]
    for c in range(1, len(ht_series)):
        chain = hashlib.sha256(DOMAIN_HT_CHAIN + chain + ht_series[c]).digest()

        # Check against checkpoints
        if c in checkpoints:
            if chain != checkpoints[c]:
                return True  # Tampering detected

    return False  # No tampering (or checkpoints match tampered data - unlikely)
```

**INV-HT-CRYPTO-3**: The Hₜ chain MUST be bound to the experiment's MDAP attestation.

```
Hₜ_chain_final || mdap_attestation_hash
```

This binding ensures that the Hₜ series is inseparable from its deterministic schedule.

```python
DOMAIN_HT_MDAP_BIND = b"MathLedger:HtMdapBinding:v2:"

def compute_ht_mdap_binding(
    ht_chain_final: bytes,
    mdap_attestation_hash: bytes
) -> bytes:
    """
    Bind the Hₜ chain to its MDAP attestation.

    This proves the Hₜ series was produced under the attested schedule.
    """
    return hashlib.sha256(
        DOMAIN_HT_MDAP_BIND +
        ht_chain_final +
        mdap_attestation_hash
    ).digest()
```

**INV-HT-CRYPTO-4**: The manifest MUST include the Hₜ-MDAP binding hash.

```json
{
  "ht_series": {
    "ht_chain_final": "<hex>",
    "ht_mdap_binding": "<hex of binding hash>"
  }
}
```

### 14.7 Hₜ Series Failure Modes

**INV-HT-FAIL-1**: The following conditions MUST cause experiment failure:

| Failure Mode | Detection Method | Manifest Status |
|--------------|------------------|-----------------|
| Rₜ(c) missing | Cycle log incomplete | `FAILED:INCOMPLETE_RT` |
| Hₜ(c) recomputation mismatch | Chain verification | `FAILED:HT_MISMATCH` |
| Checkpoint mismatch | Checkpoint verification | `FAILED:CHECKPOINT_MISMATCH` |
| MDAP binding mismatch | Binding verification | `FAILED:MDAP_BINDING_MISMATCH` |

**INV-HT-FAIL-2**: On failure, the manifest MUST record:

```json
{
  "experiment": {
    "status": "failed"
  },
  "failure": {
    "reason": "HT_MISMATCH",
    "cycle": 142,
    "expected_ht": "<hex>",
    "actual_ht": "<hex>",
    "detection_timestamp_utc": "<ISO8601>"
  }
}
```

---

## 15. Hₜ-Adjacent RFL Monitoring (Design Only)

> **STATUS: DESIGN PROPOSAL — NOT IMPLEMENTED**
>
> This section describes how future uplift experiments *could* interface with Hₜ
> monitoring without violating the independence between provenance (Hₜ) and behavior (RFL).
> Nothing in this section is implemented or affects Phase I claims.

### 15.1 Design Principles

1. **Hₜ definition is immutable**: `Hₜ = SHA256(DOMAIN_HT || Rₜ || Uₜ)` — no RFL data enters this computation
2. **Hₜ invariant tests ignore RFL**: Pass/fail semantics are solely about cryptographic properties
3. **RFL monitoring is post-hoc observation**: Any correlation analysis happens *after* Hₜ is sealed
4. **No test flips based on RFL**: RFL metadata cannot change an Hₜ invariant test from pass to fail

### 15.2 Proposed Observables (Future Work)

If uplift experiments produce meaningful data, the following could be tracked *alongside* (not *inside*) Hₜ:

| Observable | Definition | Purpose |
|------------|------------|---------|
| ΔHₜ | `Hₜ(epoch_n) ⊕ Hₜ(epoch_n-1)` (XOR or diff) | Measure ledger churn between epochs |
| ΔHₜ magnitude | Count of differing bits in successive Hₜ | Proxy for "how much changed" |
| Abstention correlation | Pearson(ΔHₜ_magnitude, abstention_rate) | Post-hoc analysis only |
| Epoch metadata | `{epoch_id, run_type, slice_config}` | Context for later analysis |

### 15.3 Interface Boundary

```
┌─────────────────────────────────────────────────────────────────┐
│                         Hₜ INVARIANT ZONE                       │
│  (cryptographic, deterministic, RFL-blind)                      │
│                                                                 │
│  Inputs:  proofs[] → Rₜ                                         │
│           ui_events[] → Uₜ                                      │
│  Output:  Hₜ = SHA256(DOMAIN_HT || Rₜ || Uₜ)                    │
│                                                                 │
│  Tests:   Recomputation, mutation detection, domain separation  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Hₜ value (sealed, immutable)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING ENVELOPE (optional)               │
│  (observational, post-hoc, no invariant authority)              │
│                                                                 │
│  Inputs:  Hₜ (from above), RFL logs (separate source)           │
│  Outputs: ΔHₜ time series, correlation reports                  │
│                                                                 │
│  NOT:     Pass/fail decisions, invariant violations             │
└─────────────────────────────────────────────────────────────────┘
```

### 15.4 What This Enables (Hypothetical)

If a future uplift experiment shows:
- Abstention rate drops from 12% → 4% over 1000 cycles
- ΔHₜ magnitude stabilizes (ledger churn decreases)

We could *observe* correlation without *claiming* causation. The Hₜ invariant tests would still pass/fail based solely on cryptographic properties. Any "uplift causes Hₜ stability" claim would require separate evidence outside the invariant framework.

### 15.5 What This Does NOT Enable

- Hₜ tests that read RFL logs
- Hₜ pass/fail based on abstention thresholds
- Embedding RFL metrics into Rₜ or Uₜ computation
- Treating ΔHₜ correlation as proof of anything

### 15.6 Implementation Prerequisites (Not Started)

Before any Hₜ-adjacent monitoring could be implemented:

1. [ ] Hₜ computation must be production-wired (currently Phase II spec only)
2. [ ] Epoch boundaries must be well-defined in the ledger
3. [ ] RFL logs must include epoch markers for alignment
4. [ ] A separate `monitoring/` module (not `attestation/`) must own this logic
5. [ ] Explicit spec review before any monitoring affects CI

---

## 16. Phase II Compliance Checklist

> **Note:** This checklist is for Phase II implementation verification. All items
> are blocked until Phase II implementation begins.

### 16.1 Manifest Composition

- [ ] Manifest `meta.version` = "2.0.0"
- [ ] Manifest `meta.phase` = "II"
- [ ] Manifest `experiment.family` matches one of the four uplift slices
- [ ] No Phase I log file references in `artifacts.logs`
- [ ] Paired runs share identical `seed_schedule.schedule_hash`
- [ ] `slice_config_hash` matches preregistration
- [ ] Governance signature present and valid
- [ ] Preregistration binding hash verifiable

### 16.2 Preregistration Binding

- [ ] `preregistration.source_hash` matches PREREG_UPLIFT_U2.yaml
- [ ] `preregistration.binding_hash` computed correctly
- [ ] Slice template from PREREG recorded in manifest
- [ ] Status transitions: NOT_YET_EXECUTED → EXECUTING → EXECUTED

### 16.3 Success Metric Binding

- [ ] `success_metric.kind` matches slice family
- [ ] `success_metric.parameters` matches preregistration
- [ ] `success_metric.parameters_hash` computed with domain separation
- [ ] Each cycle log includes `success` and `success_evidence`
- [ ] Success evaluation is deterministically reproducible

### 16.4 Seed Schedule

- [ ] MDAP_SEED = 0x4D444150 used as root
- [ ] Cycle seeds derived via domain-separated SHA-256
- [ ] Schedule hash pre-computed before execution
- [ ] Schedule verification hash recomputed post-run
- [ ] `seed_schedule.match` = true
- [ ] MDAP attestation recorded before first cycle
- [ ] MDAP verification section present post-run
- [ ] Baseline shuffle uses only cycle seed (no other entropy)
- [ ] RFL ordering deterministic given (weights, features, seed)

### 16.5 Hₜ Series

- [ ] Per-cycle Hₜ computed with cycle seed binding
- [ ] Hₜ chain computed incrementally
- [ ] `ht_series.ht_chain_final` recorded in manifest
- [ ] Checkpoints at ≤ 100 cycle intervals
- [ ] First and last cycle checkpointed
- [ ] Series independently verifiable from log + manifest
- [ ] Hₜ-MDAP binding hash present in manifest
- [ ] Tampering detection via checkpoint verification

### 16.6 Governance

- [ ] Preregistration completed before run (`PREREG_UPLIFT_U2.yaml`)
- [ ] No reinterpretation of Phase I logs
- [ ] All artifacts labeled "PHASE II — NOT RUN IN PHASE I"
- [ ] Statistical summary includes Δp and 95% CIs
- [ ] RFL uses verifiable feedback only (no human preferences)
- [ ] Determinism enforced (except baseline random policy)
- [ ] Failure modes properly recorded if triggered

---

## 17. Invariant Summary Table

| ID | Invariant | Section | Phase |
|----|-----------|---------|-------|
| INV-CANON-1 | All JSON canonicalized | §2.1 | I |
| INV-DOMAIN-1 | Domain separation required | §2.2 | I |
| INV-ORDER-1 | Merkle leaves sorted | §2.3 | I |
| INV-CASE-1 | Lowercase hex | §2.4 | I |
| INV-WS-1 | No whitespace in canonical JSON | §2.5 | I |
| INV-VERIFY-1 | Hₜ independently recomputable | §2.6 | I |
| INV-EMPTY-1 | Empty tree has defined root | §3.2 | I |
| INV-MANIFEST-1 | Manifest schema compliance | §12.1 | II |
| INV-MANIFEST-2 | Phase marker required | §12.1 | II |
| INV-MANIFEST-3 | Valid slice family | §12.1 | II |
| INV-MANIFEST-4 | No Phase I references | §12.1 | II |
| INV-MANIFEST-5 | Governance signature required | §12.1 | II |
| INV-PAIRED-1 | Same schedule hash | §12.2 | II |
| INV-PAIRED-2 | Cross-referenced manifests | §12.2 | II |
| INV-SLICE-1 | Slice config hash included | §12.3 | II |
| INV-SLICE-2 | Config immutable after prereg | §12.3 | II |
| INV-SEED-1 | MDAP seed derivation | §13.1 | II |
| INV-SEED-2 | Deterministic cycle seeds | §13.2 | II |
| INV-SEED-3 | Independent cycle seeds | §13.2 | II |
| INV-SEED-4 | RNG uses only cycle seed | §13.2 | II |
| INV-SCHEDULE-1 | Pre-computed schedule hash | §13.3 | II |
| INV-SCHEDULE-2 | Post-run verification hash | §13.3 | II |
| INV-SCHEDULE-3 | Schedule recomputable | §13.4 | II |
| INV-SEED-PARITY-1 | Paired runs same seeds | §13.5 | II |
| INV-SEED-PARITY-2 | Only ordering differs | §13.5 | II |
| INV-MDAP-ATTEST-1 | MDAP attestation before execution | §13.6 | II |
| INV-MDAP-ATTEST-2 | MDAP attestation hash formula | §13.6 | II |
| INV-MDAP-ATTEST-3 | MDAP attestation timing | §13.6 | II |
| INV-MDAP-ATTEST-4 | MDAP verification section | §13.6 | II |
| INV-MDAP-ATTEST-5 | MDAP mismatch causes failure | §13.6 | II |
| INV-DETERMINISM-1 | Baseline uses only cycle seed | §13.7 | II |
| INV-DETERMINISM-2 | RFL ordering deterministic | §13.7 | II |
| INV-DETERMINISM-3 | Determinism violation fails | §13.7 | II |
| INV-HT-CYCLE-1 | Per-cycle Hₜ recorded | §14.1 | II |
| INV-HT-CYCLE-2 | Cycle seed in Hₜ | §14.1 | II |
| INV-HT-CHAIN-1 | Append-only chain | §14.2 | II |
| INV-HT-CHAIN-2 | Final chain in manifest | §14.2 | II |
| INV-HT-MANIFEST-1 | Hₜ series summary | §14.3 | II |
| INV-HT-MANIFEST-2 | Checkpoint intervals | §14.3 | II |
| INV-HT-MANIFEST-3 | First/last checkpointed | §14.3 | II |
| INV-HT-VERIFY-1 | Series independently verifiable | §14.4 | II |
| INV-DELTA-HT-1 | ΔHₜ observational only | §14.5 | II |
| INV-DELTA-HT-2 | ΔHₜ not in pass/fail | §14.5 | II |
| INV-HT-CRYPTO-1 | Cryptographic guarantees | §14.6 | II |
| INV-HT-CRYPTO-2 | Tampering detectable | §14.6 | II |
| INV-HT-CRYPTO-3 | Hₜ chain bound to MDAP | §14.6 | II |
| INV-HT-CRYPTO-4 | Hₜ-MDAP binding in manifest | §14.6 | II |
| INV-HT-FAIL-1 | Failure mode detection | §14.7 | II |
| INV-HT-FAIL-2 | Failure recording | §14.7 | II |
| INV-PREREG-1 | Manifest binds to preregistration | §12.4 | II |
| INV-PREREG-2 | Prereg hash immutable | §12.4 | II |
| INV-PREREG-3 | No prereg changes during execution | §12.4 | II |
| INV-PREREG-4 | Slice template recorded | §12.4 | II |
| INV-SUCCESS-1 | One success metric per slice | §12.5 | II |
| INV-SUCCESS-2 | Success params match prereg | §12.5 | II |
| INV-SUCCESS-3 | Success params hash | §12.5 | II |
| INV-SUCCESS-4 | Success eval deterministic | §12.5 | II |
| INV-SUCCESS-5 | Cycle logs include success | §12.5 | II |

---

## 18. Domain Prefix Registry

All domain separation prefixes used in Phase I and Phase II:

| Domain | Prefix | Phase | Purpose |
|--------|--------|-------|---------|
| Proof Leaf | `b"MathLedger:Proof:v1:"` | I | Proof object hashing |
| UI Event Leaf | `b"MathLedger:UIEvent:v1:"` | I | UI event hashing |
| Dual Attestation | `b"MathLedger:DualAttest:v1:"` | I | Final Hₜ computation |
| Merkle Node | `b"MathLedger:MerkleNode:v1:"` | I | Internal tree nodes |
| Slice Config | `b"MathLedger:SliceConfig:v2:"` | II | Slice configuration binding |
| Cycle Seed | `b"MathLedger:CycleSeed:v2:"` | II | Per-cycle seed derivation |
| Schedule | `b"MathLedger:SeedSchedule:v2:"` | II | Schedule hash computation |
| Hₜ Cycle | `b"MathLedger:HtCycle:v2:"` | II | Per-cycle attestation |
| Hₜ Chain | `b"MathLedger:HtChain:v2:"` | II | Chain hash computation |
| Prereg Binding | `b"MathLedger:PreregBinding:v2:"` | II | Manifest-preregistration binding |
| Success Metric | `b"MathLedger:SuccessMetric:v2:"` | II | Success metric parameter hashing |
| MDAP Attestation | `b"MathLedger:MDAPAttestation:v2:"` | II | Schedule attestation before execution |
| Hₜ-MDAP Binding | `b"MathLedger:HtMdapBinding:v2:"` | II | Binding Hₜ chain to MDAP attestation |

**Rationale for v2 prefixes**: Phase II domains use `:v2:` to distinguish them from Phase I domains and prevent any accidental collision or misuse between phases.
