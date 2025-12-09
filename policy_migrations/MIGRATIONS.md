# Policy Migration Log

Track every policy manifest/version migration here. Each entry documents source + destination hashes, operator, date, and verification artifacts.

```
- date: 2025-12-06T18:00:00Z
  operator: codex-l
  from_schema: policy_manifest@v1
  to_schema: policy_manifest@v2
  source_hash: <old hash>
  destination_hash: <new hash>
  commands:
    - uv run python scripts/policy_migrate.py --source <path> --output-root artifacts/policy
    - uv run python scripts/policy_archive.py --policy-dir artifacts/policy/<hash>
  notes: >
    Describe why migration occurred (new features, bug fix, etc.).
```
