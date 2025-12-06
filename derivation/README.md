# Derivation

Enumerators, derivation engines, and inference rules live here. Modules should:

- Construct candidate statements from the substrate
- Apply inference schedules and search heuristics deterministically
- Emit traceable artifacts for downstream normalization

Avoid mixing normalization or ledger responsibilities; keep this layer purely generative.

