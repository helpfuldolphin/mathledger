# Basis Invariants

- **Pure functions only** – no ambient reads or writes. All state flows through explicit parameters and return values.
- **Deterministic ordering** – collections emitted by the basis (statements, tiers) are deterministic (`tuple` or sorted lists).
- **ASCII canonicalisation** – input strings pass through `normalize` before entering cryptographic code paths.
- **Domain-separated hashing** – every SHA-256 invocation specifies a non-empty domain tag except where explicitly shared.
- **Explicit validation** – all composite hashes validate hex length/format.
- **Stable JSON** – `block_json` and `ladder_to_json` use canonical separators and key sorting to avoid drift.

