# Determinism Contract — First Organism Path

**Version**: 1.0
**Status**: Enforced

## 1. The Prime Directive

> "Nothing moves without a content hash."

The First Organism path (`UI Event → Curriculum Gate → Derivation → Lean → LedgerIngestor → DualRoot → RFL Runner`) must be **bit-perfect deterministic**. 

Any two executions with the same **Initial Seed** and **Input Content** must produce:
1. Identical **Merle Roots** ($R_t$, $U_t$, $H_t$).
2. Identical **Timestamps** in all artifacts.
3. Identical **UUIDs/IDs** for all entities.
4. Identical **Byte Sequences** for all JSON/log outputs.

## 2. Forbidden Primitives

The following sources of entropy are **strictly forbidden** in the critical path:

| Forbidden Primitive | Replacement | Reason |
|---------------------|-------------|--------|
| `datetime.now()` | `backend.repro.determinism.deterministic_timestamp(seed)` | Wall-clock time leaks machine state. |
| `datetime.utcnow()` | `backend.repro.determinism.deterministic_timestamp(seed)` | Wall-clock time leaks machine state. |
| `time.time()` | `backend.repro.determinism.deterministic_unix_timestamp(seed)` | Varies by execution speed. |
| `uuid.uuid4()` | `backend.repro.determinism.deterministic_uuid(content)` | Randomness breaks ID stability. |
| `random.*` | `backend.repro.determinism.SeededRNG(seed)` | Unseeded global randomness is chaotic. |
| `os.urandom` | **Forbidden** | No deterministic equivalent. |
| `dict` iteration | `sorted(d.items())` | Python <3.7 order is undefined; safety first. |

## 3. Component Contracts

### 3.1 UI Event
- **Timestamp**: Derived from `SHA256(payload)`.
- **Event ID**: Derived from `SHA256(payload)`.
- **Must not** use client-side wall clock.

### 3.2 Curriculum Gate
- **Evaluation Time**: Passed explicitly or derived from `deterministic_timestamp(0)` for audit logs.
- **Verdict**: Must be a pure function of `Metrics + Config`.

### 3.3 Derivation (Axiom Engine)
- **Candidate Generation**: Must iterate axioms/rules in lexicographical order.
- **Proof Search**: Must use `sorted()` on candidate lists before processing.
- **Timestamps**: All `created_at` fields in DB/Logs must use `deterministic_timestamp(seed)`.

### 3.4 Lean Verification
- **Timeouts**: Must be deterministic (step counts, not wall clock) if possible. If wall clock is needed for safety, the *result* (abstention) must be handled deterministically (e.g. "abstain" status is recorded, but retry logic must be deterministic).
- **Output**: Standardized stdout/stderr normalization.

### 3.5 Ledger & Dual Root
- **Sealing Time**: `sealed_at` must be derived from `H_t` (the composite root itself).
  - Formula: `timestamp = deterministic_timestamp(SHA256(H_t))`
- **Block ID**: Derived from `H_t`, not auto-increment integer (unless purely sequential and reset-proof).

### 3.6 RFL Runner
- **Policy Update**: Must depend solely on `Abstention Metrics`, never on `duration` or `throughput` unless those are normalized/mocked.
- **Experiment IDs**: Derived from `SHA256(Seed || Slice || Index)`.

## 4. Enforcement

### 4.1 Static Analysis
A grep-based sentinel ensures no `datetime.now` or `uuid.uuid4` calls exist in:
- `backend/repro/`
- `backend/axiom_engine/derive_core.py`
- `rfl/`
- `attestation/`

### 4.2 Runtime Verification
The `FirstOrganismResult` must be hashed. The integration test runs the full loop twice:
```python
run1 = run_first_organism(seed=42)
run2 = run_first_organism(seed=42)
assert run1.run_hash == run2.run_hash
```

## 5. Exceptions

The following are permitted **only** for operational metrics (not business logic):
- `time.perf_counter()`: Allowed for logging latency *metrics* pushed to Redis, provided these metrics **never** influence control flow (e.g. gating).
- `datetime.now()`: Allowed in `scripts/` for human-facing CLI logs, provided the log content is not part of the hash chain.

