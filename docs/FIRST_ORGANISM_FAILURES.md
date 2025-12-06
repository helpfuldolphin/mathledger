# Failure-Mode Cartographer — First Organism Atlas

**Operation: FIRST ORGANISM**
**Status:** ACTIVE
**Cartographer:** Vibe + Intent Layer

This document enumerates the known failure modes of the First Organism dynamical system (UI Event → Curriculum Gate → Derivation → Lean → LedgerIngestor → DualRoot → RFL Runner).

> "Elegant paranoia." Assume nothing, test everything.

## Failure Atlas

| Layer | Failure Mode | How to Reproduce | Expected Behavior | Status |
|-------|--------------|------------------|-------------------|--------|
| **UI/Event** | `Malformed Payload` | POST JSON with circular ref or non-serializable types to `capture_ui_event`. | `TypeError` / `JSONEncodeError` raised. | ✅ Tested |
| **UI/Event** | `DB/Store Failure` | Simulate `ui_event_store.add` raising exception (e.g. connection lost). | `RuntimeError` raised, event not captured. | ✅ Tested |
| **Curriculum** | `Missing Metrics` | Evaluate gate with empty metrics dict. | Gate returns `passed=False` with reason "missing metrics". | ✅ Tested |
| **Curriculum** | `Threshold Failure` | Provide metrics below strict thresholds (e.g. coverage < 0.99). | Gate returns `passed=False` with specific threshold violation message. | ✅ Tested |
| **Derivation** | `Bounds Misconfig` | Initialize `SliceBounds` with negative values. | `ValueError` raised on init. | ✅ Tested |
| **Derivation** | `Normalization Fail` | Inject failure in `StatementVerifier` normalization logic. | Pipeline raises exception or logs error (depending on context). | ✅ Tested |
| **Ledger** | `Schema Mismatch` | Call `LedgerIngestor.ingest` with missing required args (e.g. `theory_name=None`). | DB Exception or TypeError raised. | ✅ Tested |
| **Attestation** | `Merkle Mismatch` | Manually compute `H_t` from `R_t` and `U_t`, then verify against a corrupted `H_t`. | `verify_composite_integrity` returns `False`. | ✅ Tested |
| **Attestation** | `Null Roots` | Pass `None` to `compute_composite_root`. | `AttributeError` (cannot encode None). | ✅ Tested |
| **RFL Runner** | `Missing Config` | Init `RFLRunner` with `config=None`. | `TypeError` raised. | ✅ Tested |

## Running the Chaos Harness

To execute the failure mode suite:

```bash
pytest -v -m "first_organism_failure_mode" tests/integration/test_first_organism_failure_modes.py
```

## Philosophy

Each component in the chain must fail **loudly** and **atomically**.
- **Loudly:** No silent swallowing of errors. Logs must indicate exactly why the chain broke.
- **Atomically:** Partial writes are forbidden. A block is either fully sealed with valid dual roots, or not written at all.

## Future Coverage Expansion

- **Lean Interface:** Simulate `lean` binary missing or timeout (requires mocking `subprocess`).
- **Network Partitions:** Simulate Redis unavailability during RFL bootstrap.
- **Concurrency:** Race conditions on LedgerIngestor sequence allocation.
