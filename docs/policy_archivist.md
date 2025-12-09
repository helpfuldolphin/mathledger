# Codex-Policy-Archivist - Policy Schema & Long-Horizon Storage

**Mission.** Protect the Reflexive Formal Learning (RFL) policy substrate so that policies remain reproducible, hashable, and auditable across years of upgrades. This note defines canonical serialization, migration, archival, delta recording, and configurable learning surfaces for every policy artifact that flows through `backend/axiom_engine/policy.py`, `artifacts/policy/`, and the RFL ledger described in `docs/RFL_LAW.md`.

---

## 1. Canonical Policy Serialization

### 1.1 Artifact Layout

Every policy checkpoint lives under `artifacts/policy/<policy_hash>/` with three required files:

```
policy.manifest.json   # Canonical metadata, hash anchor, references
policy.weights.bin     # Serialized model or scorer parameters
policy.surface.yaml    # Learning surface configuration snapshot (see Section 5)
```

`artifacts/policy/policy.json` remains as the head pointer that names the active hash.

### 1.2 Canonical Manifest Schema

`policy.manifest.json` uses a deterministic JSON representation:

```jsonc
{
  "schema_version": "policy_manifest@v2",
  "policy": {
    "hash": "<64-char hex>",          // SHA-256 of policy.weights.bin
    "version": "2025.12.06-r4",       // SemVer or YYYY.MM.DD tag
    "model_type": "reranker|actor",   // Ties back to backend policy loader
    "serialization": "pickle@3.11",   // Tooling that produced weights
    "byte_size": 123456,              // Bytes in policy.weights.bin
    "created_at": "2025-12-06T13:45:00Z"
  },
  "training_context": {
    "dataset": "phase2_uplift_mixture_v5",
    "curriculum": "config/curriculum_uplift_phase2.yaml",
    "seed": 3735928559,
    "code_commit": "<git sha>",
    "runner": "scripts/train_policy.py"
  },
  "compatibility": {
    "required_features": [
      "atoms", "depth", "lean_score"
    ],
    "supports_symbolic_descent": true,
    "min_runner_version": "2025.11.20",
    "max_runner_version": null
  }
}
```

**Canonicalization rules**

1. Keys are sorted lexicographically at every level before hashing.
2. Numbers are rendered with full precision (no trailing zeros).
3. `policy.weights.bin` hash is computed via `sha256(file_bytes)` and stored in `policy.hash`.
4. `policy.manifest.json` stores an independent `sha256` (hash the canonical JSON bytes) to detect metadata tampering.
5. `backend/axiom_engine/policy.py:get_policy_hash` should read the manifest when available instead of stringifying the object; fallback logic remains for legacy policies.

### 1.3 Policy Hash Propagation

* `backend/axiom_engine/derive.py` already writes `active_policy_hash` via `_set_policy_hash_in_db`. Extend this to read the canonical manifest so that `policy_hash[:16]` in logs matches `policy.manifest.json`.
* Runtime helpers (`backend/axiom_engine/policy.py:load_policy`, `scripts/policy_inference.py`) now read manifest metadata automatically so every caller inherits canonical hashes.
* `docs/methods/EVIDENCE_GRAPH.md` chain 3 references `artifacts/policy/policy.json`; update those docs to point to the manifest once this schema ships.

---

## 2. Version Migration Plans

Policies evolve when feature sets or architectures change. Track migrations via `scripts/policy_migrate.py` and a `policy_migrations/` directory. The canonical loop is:

1. `uv run python scripts/policy_migrate.py --source <legacy>` to emit `policy.manifest.json + policy.weights.bin`.
2. `uv run python scripts/policy_archive.py --policy-dir artifacts/policy/<hash>` to capture rollback-safe state.
3. `uv run python scripts/policy_backfill_runtime_hashes.py --policy-root artifacts/policy --dry-run` to inspect, then re-run with `--apply` (and optional `--db-url`) to update the `policy.json` head pointer and runtime metadata.

```
policy_migrations/
  v1_to_v2.py
  v2_to_v3.py
  MIGRATIONS.md
```

Each migration script implements:

```python
def up(manifest: dict, weights_path: Path) -> tuple[dict, Path]:
    """Returns (new_manifest, new_weights_path)."""

def down(manifest: dict, weights_path: Path) -> tuple[dict, Path]:
    """Rollback used when a migration fails verification."""
```

_Process_

1. **Freeze Source:** Copy `<hash>/` to `<hash>.migration_work/`.
2. **Run Migration:** `uv run python scripts/policy_migrate.py --source artifacts/policy/<hash>/`.
3. **Verify:** 
   - Recompute `sha256(policy.weights.bin)` and ensure manifest matches.
   - Run `uv run pytest tests/policy --hash <new_hash>` (tests tagged `policy_migration`).
4. **Promote:** `git mv` the migrated directory to `artifacts/policy/<new_hash>/`, update `artifacts/policy/policy.json`.
5. **Record:** Append entry to `policy_migrations/MIGRATIONS.md` (`from`, `to`, `hash_before`, `hash_after`, operator, timestamp).

_Version Compatibility Table_

| Manifest `schema_version` | Runner compatibility | Notes |
| --- | --- | --- |
| `policy_manifest@v1` | `<=2025.09` | Legacy single-file metadata (current `policy.json`). |
| `policy_manifest@v2` | `>=2025.12` | Canonical manifest, learning surface snapshot, delta ledger integration. |
| `policy_manifest@v3` | `>=2026.02` | Planned addition of encrypted gradient cache (not implemented yet). |

---

## 3. Rollback-Safe Archives

Long-horizon governance requires immutable, rollback-safe records:

*Archive layout*

```
archive/policy/
  2025-09-14_a1b2c3.../
    policy.manifest.json
    policy.weights.bin
    policy.surface.yaml
    ledger_excerpt.jsonl
    verification.log
```

*Rules*

1. Archives are append-only: use `git add` + `git commit`, never rewrite. Run `uv run python scripts/policy_archive.py --policy-dir artifacts/policy/<hash>` to capture verification logs.
2. `ledger_excerpt.jsonl` captures `RunLedgerEntry` rows where `policy_hash` equals the archived hash (see `docs/RFL_LAW.md`).
3. `verification.log` records commands (`sha256sum`, `uv run pytest -m policy_archive`, etc.) with timestamps.
4. Rollback protocol: 
   - Restore directory from archive.
   - Re-run verification log commands to prove determinism.
   - Update `artifacts/policy/policy.json` to point at restored hash.
   - Emit `rollback_notice.md` summarizing reason, signed by ops owner.

---

## 4. Policy Deltas & Symbolic Update (oplus)

The symbolic update operator `oplus` formalizes incremental policy changes per `docs/RFL_LAW.md`:

```
theta_{t+1} = theta_t oplus delta_t
delta_t = eta_t * grad_sym(H_t, metrics_t)
```

*Logging format:* `artifacts/policy/<hash>/delta_log.jsonl`

```json
{
  "cycle": 173,
  "slice": "goal",
  "mode": "rfl",
  "policy_hash_before": "a1b2...",
  "policy_hash_after": "b3c4...",
  "theta_norm_before": 1.24,
  "theta_norm_after": 1.11,
  "delta": {
    "type": "symbolic_descent",
    "payload": [0.04, -0.01, ...],
    "l2_norm": 0.09
  },
  "gradient_norm": 0.13,
  "learning_rate": 0.05,
  "attestation_root": "Ht_...",
  "timestamp": "2025-12-06T13:45:07Z"
}
```

*Guarantees*

- Each delta references the attested `H_t` root (`RFL_LAW` Section 2.1.4).
- `policy_hash_after` must match the hash of the serialized weights produced after applying the delta; on failure, revert to the previous checkpoint and mark the ledger entry as `policy_update_applied=False`.
- Consumers (e.g., `analysis/conjecture_engine.py`) can reconstruct long-horizon trajectories by replaying `oplus` operations.

*APIs*

Expose a helper in `backend/axiom_engine/policy.py`:

```python
def apply_symbolic_delta(weights: np.ndarray, delta: np.ndarray, learning_rate: float) -> np.ndarray:
    """Returns updated weights plus metadata required for oplus logging."""
```

This ensures mocking and deterministic replay.

---

## 5. Configurable Policy Learning Surfaces

Learning surfaces describe the feature topology and reward shaping budget that the policy optimizes. Capture them in `policy.surface.yaml` so every checkpoint records its training intent.

```yaml
surface_version: policy_surface@v1
feature_space:
  - name: atoms
    transform: log1p
    bounds: [1, 64]
  - name: depth
    transform: identity
    bounds: [1, 8]
  - name: lean_score
    transform: minmax
reward_signals:
  abstention_mass: {weight: -0.8, window: 32}
  throughput: {weight: 0.3, window: 16}
symbolic_descent:
  enabled: true
  operator: concat(policy_gradient, abstention_gradient)
curvature_controls:
  max_gradient_norm: 0.5
  damping_factor: 0.2
deployment_constraints:
  safe_modes:
    - name: deterministic_eval
      description: "Disable stochastic tie-breakers."
      toggle_env: POLICY_SAFE_MODE=1
    - name: shadow
      description: "Score only; do not reorder candidates."
      toggle_env: POLICY_SHADOW_MODE=1
```

*Usage*

- Training scripts read `policy.surface.yaml` to instantiate feature transforms and reward weights.
- `run_uplift_u2.py` can reference the surface to decide whether a slice is allowed to run with the given policy (e.g., enforce `safe_modes` for integration tests).
- When surfaces change (e.g., new features), bump `surface_version` and log a `policy_delta` with `delta.type = "surface_change"`.

---

## 6. Operational Checklist

1. **Serialize** using canonical manifest rules; verify `sha256`.
2. **Store** each version under `artifacts/policy/<hash>/` plus entry in `artifacts/policy/policy.json`.
3. **Archive** the directory under `archive/policy/<timestamp_hash>/`.
4. **Record** all updates via `delta_log.jsonl` (oplus operator) pointing to attested `H_t`.
5. **Migrate** via scripted tools when schema versions change; document in `policy_migrations/MIGRATIONS.md`.
6. **Expose** learning surface toggles so experiments and governance reviews can trace policy intent.
7. **Lint & attest** with `scripts/policy_drift_linter.py --lint` (and the commit guard described below) so manifests, surfaces, and delta logs remain contract-safe.

By following this framework, RFL policies become durable, fully hash-traceable assets that future experiments can reuse, audit, or roll back without ambiguity. This satisfies the Codex-Policy-Archivist charter: preserve cognition across long horizons with deterministic, well-documented artifacts.

---

## 7. Tooling Reference

- `uv run python scripts/policy_migrate.py --source <path>` migrates legacy exports into canonical manifest form.
- `uv run python scripts/policy_archive.py --policy-dir artifacts/policy/<hash>` creates rollback-safe archives with verification logs and checksum captures.
- `uv run python scripts/policy_backfill_runtime_hashes.py --policy-root artifacts/policy --apply` enforces the manifest identity at runtime (rewrites `policy.json`, updates DB if `--db-url` supplied). Use `--dry-run` to audit before applying.
- `policy_migrations/MIGRATIONS.md` records every migration (source hash, destination hash, operator, commands, rationale).
- Runtime helpers (`backend/axiom_engine/policy.py`, `scripts/policy_inference.py`) consume the manifest directly, so compute pipelines inherit canonical hashes without bespoke glue.
- `scripts/policy_drift_linter.py` + `.githooks/policy_drift_guard.sh` enforce ledger-backed drift detection across commits.

## 8. Policy Drift Linter & Commit Guard

- Run `scripts/policy_drift_linter.py --lint` locally (or via the hook) to verify manifest/weights alignment, surface signatures, and dual-attested `delta_log.jsonl` updates.
- Record snapshots with `scripts/policy_drift_linter.py --snapshot --notes "<context>"` to append to `artifacts/policy/policy_hash_ledger.jsonl`. Each entry captures hash, manifest, learning-rate spread, and lint status.
- Enable `.githooks/policy_drift_guard.sh` (`git config core.hooksPath .githooks`) to automatically lint, snapshot, and block commits whenever the ledger reports `BLOCK` drift. Approved changes should come with attested deltas and descriptive notes so downstream reviewers can trace the dual-attestation cycle.
- Integrate `scripts/policy_drift_lint.py` in pre-commit or CI to reject unsafe config edits (learning-rate spikes, clipping removal, disabled abstention gates, promotion threshold changes). See snippet below.

### Sample Hook

```bash
#!/bin/bash
# Policy Config Drift Guard
set -euo pipefail
OLD="${1:-config/policy_baseline.yaml}"
NEW="${2:-config/policy_candidate.yaml}"
echo "[policy-drift-lint] Comparing $OLD -> $NEW"
uv run python scripts/policy_drift_lint.py --old "$OLD" --new "$NEW" --text
```

`policy_drift_lint.py` exits non-zero when `status != OK`, so wiring this script into a pre-commit or CI step automatically blocks the pipeline until a design review approves the drift.
