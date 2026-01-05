# Version Number Doctrine

This document establishes the versioning scheme used in MathLedger.

## Version Format

MathLedger uses semantic-like versioning: `v{major}.{minor}.{patch}`

- **Major (v0, v1, ...)**: Epoch changes. v0 indicates pre-production/pilot phase.
- **Minor (v0.2, v0.3, ...)**: Capability additions or significant changes.
- **Patch (v0.2.11, v0.2.12, ...)**: Bug fixes, documentation, or infrastructure fixes.

## Version Lifecycle

Each version passes through these states:

1. **current**: The active, recommended version
2. **superseded-by-vX.Y.Z**: Replaced by a newer version
3. **deprecated** (future): Scheduled for removal

## What Triggers a New Version

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| New capability | Minor | v0.2 → v0.3 |
| Bug fix | Patch | v0.2.10 → v0.2.11 |
| Documentation only | Patch | v0.2.11 → v0.2.12 |
| Infrastructure fix | Patch | v0.2.9 → v0.2.10 |
| Security fix | Patch (minimum) | v0.2.x → v0.2.y |

## Version Immutability

Once a version is released and tagged:

1. **No changes to that version's artifacts** - The site directory is frozen
2. **Manifest files are immutable** - sha256 checksums seal the version
3. **Status may change** - A version can become superseded, but its content doesn't change

## Historical Artifacts

Superseded versions remain fully navigable at their original URLs:
- `/v0.2.10/` still works after v0.2.11 is released
- Manifests show "status: superseded-by-v0.2.11" but content is unchanged
- `/versions/` is the canonical source for current/superseded status

## Tag Naming Convention

Tags follow the pattern: `v{version}-{description}`

Examples:
- `v0.2.10-demo-reliability`
- `v0.2.11-verifier-parity`
- `v0.2.12-versioning-doctrine`

The description briefly indicates the primary change in that version.

## releases.json Authority

The file `releases/releases.json` is the single source of truth for:
- Current version
- Version metadata (commit, tag, date)
- Superseded status
- Invariant counts

The build script reads ONLY this file to generate the site.

## Why v0.x.y (Not v1.0)?

MathLedger is in pilot phase. The v0 prefix signals:
- System is under active development
- Interfaces may change
- Not production-ready for general use

Version 1.0 will indicate production readiness.

---
*Document created: v0.2.12 | Last updated: 2026-01-05*
