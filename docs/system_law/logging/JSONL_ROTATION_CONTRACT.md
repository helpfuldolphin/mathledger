# JSONL Rotation Contract (Phase Y)

This document freezes the behavior that `JsonlWriter` and any future `JsonlRotator`
implementation must preserve. Phase Y may implement rotation mechanics, but MUST
NOT change the observable JSONL output contract.

## Scope

- `backend/logging/jsonl_writer.py` (`JsonlWriter`, `JsonlRotator`)
- Contract harness: `tests/logging/test_jsonl_rotator_contract.py`

`JsonlRotator` is currently a **Phase Y stub** (raises `NotImplementedError`) and
exists only to lock the interface shape so Phase Y can drop in an implementation
without changing callers.

## Frozen Invariants

1. **No reordering**
   - Record order on disk MUST match the order of `write(record)` calls.
   - Rotation MUST be transparent: concatenating rotated segments MUST equal a
     single non-rotating stream of the same writes.

2. **No mutation**
   - A rotator MUST NOT modify the caller-provided record dict (including nested
     structures). Treat input records as immutable.

3. **UTF-8 + literal Unicode**
   - Files MUST be written as UTF-8 text.
   - Default behavior MUST preserve literal Unicode characters in output (i.e.,
     `ensure_ascii=False` unless explicitly overridden via `json_kwargs`).

4. **JSON separators (default)**
   - Default JSON rendering MUST use compact separators: `(",", ":")`.
   - Each record MUST be exactly one JSON object line followed by a single
     newline.

5. **Flush-per-write**
   - Every successful `write(record)` MUST flush before returning.

Any change to these invariants requires an explicit compatibility review and
updates to the contract harness.

