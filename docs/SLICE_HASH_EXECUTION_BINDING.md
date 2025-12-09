# Slice Hash Execution Binding Specification

> **STATUS: PHASE II — NOT USED FOR FO/PHASE I RUNS**
>
> This specification defines how slice hashes bind into execution manifests,
> preregistration documents, and reconciliation protocols. It is the bridge
> between the Slice Hash Ledger and the U2 Governance Pipeline.

**Version:** 1.0.0
**Author:** Claude E — Hash Law (Phase II Slices)
**Date:** 2025-12-06
**Bound to:** `docs/SLICE_HASH_LEDGER.md`, `docs/U2_GOVERNANCE_PIPELINE.md`

---

## 1. Purpose

This document specifies:

1. **Execution Binding Schema** — How slice hashes are embedded in prereg and manifests
2. **Reconciliation Link** — Cross-reference with governance pipeline hashes
3. **Slice Author Protocol** — Safe creation and migration of slices

The binding ensures that **once a slice is preregistered, its hash identity is frozen** and any drift is detectable at governance verification time.

---

## 2. Execution Binding Schema

### 2.1 The `slice_hash_binding` Object

The `slice_hash_binding` object is the canonical structure for embedding slice hash provenance into execution artifacts. It **must** appear in both preregistration and manifest documents.

```yaml
slice_hash_binding:
  # Identity fields (required)
  slice_name: "<slice identifier>"          # e.g., "slice_uplift_goal"
  slice_config_hash: "<64-char hex>"        # SHA256 of canonical slice config
  ledger_entry_id: "<unique entry ref>"     # Reference to ledger position

  # Temporal fields (required)
  frozen_at: "<ISO 8601 timestamp>"         # When binding became immutable
  frozen_by: "<agent identifier>"           # Who created the binding

  # Provenance fields (required)
  config_source: "<file path>"              # e.g., "config/curriculum_uplift_phase2.yaml"
  config_version: "<version string>"        # e.g., "2.1.0"

  # Formula pool digest (required)
  formula_pool_hash: "<64-char hex>"        # SHA256 of serialized formula_pool_entries
  formula_count: <integer>                  # Number of formulas in pool
  target_count: <integer>                   # Number of target formulas

  # Verification metadata (optional but recommended)
  verification:
    tool: "scripts/verify_slice_hashes.py"
    verified_at: "<ISO 8601 timestamp>"
    result: "PASS"
    checked_entries: <integer>
    failed_entries: 0
```

### 2.2 Field Definitions

| Field | Type | Description | Immutability |
|-------|------|-------------|--------------|
| `slice_name` | string | Unique slice identifier within system | Immutable after binding |
| `slice_config_hash` | hex64 | SHA256 of canonical slice YAML block | Immutable after binding |
| `ledger_entry_id` | string | Cross-reference to ledger position | Immutable after binding |
| `frozen_at` | ISO 8601 | Timestamp when binding was created | Immutable after binding |
| `frozen_by` | string | Agent/user who created binding | Immutable after binding |
| `config_source` | path | Source file for slice definition | Immutable after binding |
| `config_version` | string | Version of config file | Immutable after binding |
| `formula_pool_hash` | hex64 | Digest of all formula entries | Immutable after binding |
| `formula_count` | integer | Total formulas in pool | Informational |
| `target_count` | integer | Number of target formulas | Informational |

### 2.3 Computing `slice_config_hash`

The `slice_config_hash` is computed over the **canonical serialization** of the slice configuration block:

```python
import hashlib
import json

def compute_slice_config_hash(slice_config: dict) -> str:
    """
    Compute the canonical hash of a slice configuration.

    The slice config is serialized using RFC 8785 (JCS) rules:
    - Keys sorted lexicographically
    - No insignificant whitespace
    - Unicode normalized
    """
    # Extract only the hashable fields (exclude runtime metadata)
    hashable_fields = {
        "name": slice_config.get("name"),
        "params": slice_config.get("params", {}),
        "success_metric": slice_config.get("success_metric", {}),
        "formula_pool_entries": slice_config.get("formula_pool_entries", []),
        "gates": slice_config.get("gates", {}),
    }

    # Canonical JSON serialization
    canonical = json.dumps(hashable_fields, sort_keys=True, separators=(',', ':'))

    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
```

### 2.4 Computing `formula_pool_hash`

The `formula_pool_hash` provides a compact digest of all formula-hash pairs:

```python
def compute_formula_pool_hash(formula_pool_entries: list) -> str:
    """
    Compute a digest of the formula pool for quick integrity checks.

    Uses the concatenation of all (normalized_formula, hash) pairs,
    sorted by hash for determinism.
    """
    pairs = []
    for entry in formula_pool_entries:
        formula = entry.get("formula", "")
        h = entry.get("hash", "")
        pairs.append(f"{formula}:{h}")

    # Sort by hash for determinism
    pairs.sort(key=lambda x: x.split(':')[-1])

    payload = "\n".join(pairs).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
```

---

## 3. Embedding in Preregistration

### 3.1 PREREG_UPLIFT_U2.yaml Structure

The preregistration document **must** include the `slice_hash_binding` block:

```yaml
# PREREG_UPLIFT_U2.yaml
preregistration:
  experiment_id: "uplift_u2_goal_001"
  slice_name: "slice_uplift_goal"

  # === SLICE HASH BINDING (REQUIRED) ===
  slice_hash_binding:
    slice_name: "slice_uplift_goal"
    slice_config_hash: "a1b2c3d4e5f6..."    # Computed at prereg time
    ledger_entry_id: "LEDGER-2025-12-06-001"
    frozen_at: "2025-12-06T10:00:00Z"
    frozen_by: "Claude E"
    config_source: "config/curriculum_uplift_phase2.yaml"
    config_version: "2.1.0"
    formula_pool_hash: "f9e8d7c6b5a4..."
    formula_count: 14
    target_count: 2
    verification:
      tool: "scripts/verify_slice_hashes.py"
      verified_at: "2025-12-06T09:55:00Z"
      result: "PASS"
      checked_entries: 14
      failed_entries: 0

  # ... rest of preregistration ...
```

### 3.2 Binding Immutability Rule

> **INVARIANT: BINDING-FREEZE**
>
> Once a `slice_hash_binding` is committed to a preregistration document,
> it **cannot be modified** without creating a new preregistration.
>
> Any experiment referencing the preregistration inherits this frozen binding.

---

## 4. Embedding in Experiment Manifest

### 4.1 manifest.json Structure

The experiment manifest **must** include the `slice_hash_binding` and validate it against the preregistration:

```json
{
  "experiment_id": "uplift_u2_goal_001",
  "run_index": 1,

  "slice_hash_binding": {
    "slice_name": "slice_uplift_goal",
    "slice_config_hash": "a1b2c3d4e5f6...",
    "ledger_entry_id": "LEDGER-2025-12-06-001",
    "frozen_at": "2025-12-06T10:00:00Z",
    "frozen_by": "Claude E",
    "config_source": "config/curriculum_uplift_phase2.yaml",
    "config_version": "2.1.0",
    "formula_pool_hash": "f9e8d7c6b5a4...",
    "formula_count": 14,
    "target_count": 2
  },

  "hash_verification": {
    "preregistration_hash": "<SHA256 of prereg block>",
    "slice_config_hash_matches_prereg": true,
    "formula_pool_hash_matches_prereg": true,
    "ledger_verification_passed": true
  },

  "provenance": { ... },
  "configuration": { ... },
  "results": { ... }
}
```

### 4.2 Manifest Binding Validation

At manifest generation time, the following checks **must** pass:

1. `slice_hash_binding.slice_config_hash` matches prereg value
2. `slice_hash_binding.formula_pool_hash` matches prereg value
3. `slice_hash_binding.ledger_entry_id` references a valid ledger entry
4. Current `formula_pool_entries` hashes match ledger expectations

---

## 5. Reconciliation Link

### 5.1 Cross-Reference with U2 Governance Pipeline

The U2 Governance Pipeline (see `docs/U2_GOVERNANCE_PIPELINE.md`) performs hash verification at Gate G2. This section defines how the `slice_hash_binding` integrates with that pipeline.

#### Hash Agreement Points

| Binding Field | Governance Check | Agreement Requirement |
|---------------|------------------|----------------------|
| `slice_config_hash` | G2: `slice_config_hash` in manifest | Must be identical |
| `formula_pool_hash` | EV1: Manifest integrity | Implied by config hash |
| `ledger_entry_id` | G2: Ledger cross-reference | Must exist in ledger |
| `frozen_at` | G1: Prereg timestamp | Must precede run start |

#### Reconciliation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RECONCILIATION CHECKPOINTS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   PREREG     │     │   MANIFEST   │     │   LEDGER     │         │
│  │              │     │              │     │              │         │
│  │ slice_hash   │────▶│ slice_hash   │◀────│ entry_hash   │         │
│  │ _binding     │     │ _binding     │     │              │         │
│  │              │     │              │     │              │         │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │         │
│  │ │config_   │═╪═════╪═│config_   │═╪═════╪═│computed_ │ │         │
│  │ │hash      │ │     │ │hash      │ │     │ │hash      │ │         │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │         │
│  │              │     │              │     │              │         │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │         │
│  │ │pool_hash │═╪═════╪═│pool_hash │═╪═════╪═│pool_hash │ │         │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │         │
│  └──────────────┘     └──────────────┘     └──────────────┘         │
│                                                                      │
│         ▲                    ▲                    ▲                  │
│         │                    │                    │                  │
│    G1: Prereg           G2: Manifest         Ledger Audit           │
│    Verification         Integrity            (verify_slice_         │
│                                               hashes.py)             │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Error Codes for Hash Discrepancies

When reconciliation fails, the following error codes identify the failure mode:

#### Slice Hash Drift Errors (SHD-*)

These errors indicate the ledger itself has drifted (normalization change, encoding change, etc.):

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| `SHD-001` | `LEDGER_FORMULA_DRIFT` | A formula's computed hash differs from ledger | Run `verify_slice_hashes.py`; revert normalization change or re-attest ledger |
| `SHD-002` | `LEDGER_POOL_DRIFT` | `formula_pool_hash` differs from ledger | Re-compute pool hash; identify added/removed entries |
| `SHD-003` | `LEDGER_ENTRY_MISSING` | `ledger_entry_id` not found in ledger | Ledger truncation or ID corruption |
| `SHD-004` | `NORMALIZATION_CHANGE` | `normalize()` output differs across versions | Critical: must migrate all hashes |

#### Manifest Hash Mismatch Errors (MHM-*)

These errors indicate the manifest disagrees with preregistration or ledger:

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| `MHM-001` | `CONFIG_HASH_MISMATCH` | Manifest `slice_config_hash` ≠ prereg value | Slice config modified after prereg; invalid experiment |
| `MHM-002` | `POOL_HASH_MISMATCH` | Manifest `formula_pool_hash` ≠ prereg value | Formula pool modified after prereg; invalid experiment |
| `MHM-003` | `LEDGER_ID_MISMATCH` | Manifest `ledger_entry_id` ≠ prereg value | Ledger reference changed; invalid experiment |
| `MHM-004` | `FROZEN_AT_VIOLATION` | `frozen_at` timestamp is after run start | Binding created after experiment began; invalid |

#### Governance Pipeline Integration

Map these error codes to GOV-* codes from `U2_GOVERNANCE_PIPELINE.md`:

| Error Code | GOV Code | Pipeline Stage | Action |
|------------|----------|----------------|--------|
| `SHD-001` | `GOV-23` | G2: verify_manifest_integrity | `FAIL` — Ledger drift detected |
| `SHD-002` | `GOV-23` | G2: verify_manifest_integrity | `FAIL` — Pool drift detected |
| `MHM-001` | `GOV-23` | G2: verify_manifest_integrity | `FAIL` — Config hash mismatch |
| `MHM-002` | `GOV-23` | G2: verify_manifest_integrity | `FAIL` — Pool hash mismatch |
| `MHM-003` | `GOV-23` | G2: verify_manifest_integrity | `FAIL` — Ledger ID mismatch |
| `MHM-004` | `GOV-22` | G2: verify_manifest_integrity | `FAIL` — Temporal violation |

---

## 6. Slice Author Protocol

### 6.1 Creating a New Slice

To create a new slice without violating ledger invariants:

#### Step 1: Design Formula Pool

1. Define all formulas using canonical notation
2. Compute hashes using `hash_statement()` from the canonical pipeline
3. Assign roles: `axiom`, `intermediate`, `target`, `decoy`
4. Document chain dependencies if applicable

#### Step 2: Verify Hashes

```bash
# Create slice in curriculum YAML
# Then verify all hashes

uv run python scripts/verify_slice_hashes.py config/curriculum_uplift_phase2.yaml
# Expected: RESULT: ALL HASHES VERIFIED OK
```

#### Step 3: Register in Ledger

The slice is implicitly registered in the ledger when added to the curriculum YAML. The ledger entry is defined by:

- `ledger_entry_id`: `SLICE-<slice_name>-<version>-<date>`
- `frozen_at`: Timestamp of verification
- `frozen_by`: Author identifier

#### Step 4: Create Preregistration Binding

Before running experiments:

1. Compute `slice_config_hash` using canonical serialization
2. Compute `formula_pool_hash` using pool digest function
3. Create `slice_hash_binding` block in preregistration
4. Commit preregistration to version control

#### Step 5: Validate Binding

```bash
# Verify the binding is consistent
uv run python scripts/verify_prereg.py experiments/prereg/PREREG_UPLIFT_U2.yaml
```

### 6.2 Modifying an Existing Slice

Modifications require careful handling to preserve lineage:

#### Case A: Adding New Formulas (Non-Breaking)

1. Add new entries to `formula_pool_entries`
2. Compute and add hashes for new formulas
3. Run `verify_slice_hashes.py` to confirm integrity
4. **Do not modify existing entries**
5. Create new preregistration with updated binding

#### Case B: Removing Formulas (Soft Deprecation)

1. Mark entries as `deprecated: true` (do not delete)
2. Run `verify_slice_hashes.py` to confirm integrity
3. Create new preregistration with updated binding
4. **Never reference deprecated formulas in new experiments**

#### Case C: Correcting a Hash Error (Breaking)

If a hash was incorrectly recorded:

1. **Document the error** in a migration record
2. Compute the correct hash
3. Update the entry with correct hash
4. Run `verify_slice_hashes.py` to confirm fix
5. **Invalidate all preregistrations** referencing the old hash
6. Create new preregistration with corrected binding

### 6.3 Migrating Slices After Ledger Drift

If ledger drift is **legitimate** (e.g., a normalization fix that improves correctness):

#### Migration Protocol

1. **Document the Change**
   ```yaml
   migration_record:
     id: "MIGRATION-2025-12-06-001"
     type: "normalization_fix"
     reason: "Right-association rule correction for edge case"
     affected_slices:
       - slice_uplift_goal
       - slice_uplift_tree
     affected_formulas: 3
     old_normalization_commit: "<git hash>"
     new_normalization_commit: "<git hash>"
   ```

2. **Recompute All Affected Hashes**
   ```bash
   uv run python scripts/verify_slice_hashes.py config/curriculum_uplift_phase2.yaml --report out/pre_migration.json
   # Note all mismatches
   ```

3. **Update Ledger Entries**
   - For each affected formula, update the hash in the YAML
   - Add `migrated_from` field referencing old hash

4. **Verify Migration**
   ```bash
   uv run python scripts/verify_slice_hashes.py config/curriculum_uplift_phase2.yaml
   # Expected: RESULT: ALL HASHES VERIFIED OK
   ```

5. **Invalidate Pre-Migration Artifacts**
   - Mark all pre-migration preregistrations as `INVALID_POST_MIGRATION`
   - Mark all pre-migration manifests as `INVALID_POST_MIGRATION`

6. **Create New Bindings**
   - New preregistrations with new `slice_hash_binding`
   - New `ledger_entry_id` with migration suffix: `SLICE-<name>-v2-<date>`

#### Migration Record Schema

```yaml
migration_record:
  id: "<unique migration id>"
  type: "<normalization_fix|encoding_fix|domain_tag_fix>"
  reason: "<human-readable explanation>"
  timestamp: "<ISO 8601>"
  author: "<agent identifier>"

  affected_slices:
    - "<slice_name>"

  hash_changes:
    - formula: "<formula text>"
      old_hash: "<64-char hex>"
      new_hash: "<64-char hex>"
      old_normalized: "<old canonical form>"
      new_normalized: "<new canonical form>"

  invalidated_artifacts:
    preregistrations:
      - "<prereg file path>"
    manifests:
      - "<manifest file path>"

  verification:
    pre_migration_report: "<path to pre-migration report>"
    post_migration_report: "<path to post-migration report>"
    post_migration_result: "PASS"
```

---

## 7. Cross-Reference Summary

### 7.1 Documents That Must Agree

| Document | Contains | Must Match |
|----------|----------|------------|
| `SLICE_HASH_LEDGER.md` | Formula-hash bindings | Ground truth |
| `curriculum_uplift_phase2.yaml` | `formula_pool_entries` | Ledger hashes |
| `PREREG_UPLIFT_U2.yaml` | `slice_hash_binding` | Config file hashes |
| `manifest.json` | `slice_hash_binding` | Prereg binding |
| `governance_receipt.json` | `verified_hashes.slice_config` | Manifest binding |

### 7.2 Verification Chain

```
Ledger (ground truth)
    ↓ verify_slice_hashes.py
Config YAML (formula_pool_entries)
    ↓ compute_slice_config_hash()
Preregistration (slice_hash_binding.slice_config_hash)
    ↓ copy at prereg time
Manifest (slice_hash_binding.slice_config_hash)
    ↓ verify_manifest_integrity.py
Governance Receipt (verified_hashes.slice_config)
```

---

## 8. Appendix: Quick Reference

### 8.1 Hash Computation Functions

| Hash | Input | Function | Location |
|------|-------|----------|----------|
| Formula hash | Single formula | `hash_statement(formula)` | `substrate/crypto/hashing.py` |
| Slice config hash | Slice YAML block | `compute_slice_config_hash(slice)` | Section 2.3 |
| Formula pool hash | All pool entries | `compute_formula_pool_hash(entries)` | Section 2.4 |
| Preregistration hash | Prereg YAML | SHA256 of canonical YAML | `verify_prereg.py` |

### 8.2 Error Code Summary

| Category | Code Range | Description |
|----------|------------|-------------|
| `SHD-*` | 001-099 | Slice Hash Drift (ledger-level) |
| `MHM-*` | 001-099 | Manifest Hash Mismatch (binding-level) |
| `GOV-*` | 1-30 | Governance Pipeline (pipeline-level) |

### 8.3 Binding Lifecycle

```
[Design] → [Compute Hashes] → [Verify] → [Freeze in Prereg] → [Copy to Manifest] → [Governance]
                ↑                                                        ↓
                └──────────── No modifications allowed ──────────────────┘
```

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-06 | Initial specification |

---

## 10. References

- `docs/SLICE_HASH_LEDGER.md` — Ledger format and invariants
- `docs/U2_GOVERNANCE_PIPELINE.md` — Governance verification pipeline
- `docs/RFL_EVIDENCE_MANIFEST.md` — Manifest schema and provenance
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` — Preregistration template
- `scripts/verify_slice_hashes.py` — Ledger verification tool

---

*— Claude E, Hash Law (Phase II Slices)*
