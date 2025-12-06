# MDAP Determinism Checklist v2

**Auditor**: Gemini C
**Status**: Active

## 1. The Golden Rules
- [ ] **S + I ⇒ A**: Given Seed `S` and Input `I`, Artifact `A` is unique (byte-for-byte).
- [ ] **No Wall Clocks**: `datetime.now()` is strictly forbidden in business logic.
- [ ] **No Race Conditions**: Logic must not depend on DB insertion order or Redis fetch order.
- [ ] **Canonical JSON**: RFC 8785 used for all hashes and serialized outputs.

## 2. Codebase Hardening
- [ ] **Harness**: `backend/repro/first_organism_harness.py` is the source of truth.
- [ ] **Derivation**: `derive_worker.py` uses `deterministic_timestamp_from_content(merkle_root)`.
- [ ] **RFL**: `RFLExperiment` accepts and uses `seed` for all time operations.
- [ ] **Ledger**: `sealed_at` is derived from `H_t`.

## 3. Infrastructure
- [ ] **Database**: `DEFAULT NOW()` removed from `statements`, `proofs`, `blocks`.
- [ ] **Redis**: Workers handle jobs atomically; no dependency on queue order for validity.
- [ ] **CI**: `scripts/determinism_gate.py` runs on every PR.

## 4. Manual Verification
```bash
# 1. Run the Gate
python scripts/determinism_gate.py

# 2. Verify Bitwise Identity
uv run pytest tests/integration/test_rfl_runner_determinism.py
```

## 5. Emergency Breaks
If nondeterminism is detected:
1. **Stop the line**.
2. Run `python scripts/determinism_gate.py` to identify the leak.
3. Grep for `time.time` or `uuid.uuid4`.
4. Check for unseeded `random` calls.
5. Verify `metrics` collection isn't leaking wall-clock duration into hashable artifacts.

