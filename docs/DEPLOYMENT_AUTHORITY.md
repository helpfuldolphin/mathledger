# Deployment Authority

This document establishes the canonical deployment infrastructure for MathLedger.

## Canonical Infrastructure

| Component | Canonical Value | Notes |
|-----------|-----------------|-------|
| Cloudflare Pages project | `mathledger-ai` | Single authoritative Pages project |
| Production domain | `mathledger.ai` | Custom domain bound to mathledger-ai |
| Demo hosting | Fly.io | App: `mathledger-demo` |
| Production alias | `production.mathledger-ai.pages.dev` | Direct deployment verification |

## Authority Rules

1. **All deployments target `mathledger-ai`** - No other Pages projects are authoritative.
2. **`mathledger.ai` is the only canonical domain** - All public references use this domain.
3. **CDN propagation lag does not change authority** - Verify via production alias if needed.
4. **Deploy-by-tag doctrine applies** - See `docs/DEPLOY_BY_TAG_DOCTRINE.md`.

## Non-Authoritative Infrastructure

Any Cloudflare Pages project other than `mathledger-ai` is non-authoritative and should be deleted.

Historical note: A redundant project `mathledger` (serving `mathledger.pages.dev`) existed and was deleted on 2026-01-07. It contained no unique artifacts.

## Verification Commands

```bash
# Verify static site deployment
curl -s https://production.mathledger-ai.pages.dev/versions/status.json | jq .current_version

# Verify demo deployment
curl -s https://mathledger.ai/demo/health | jq '{version, tag}'

# List Pages projects (should show only mathledger-ai)
wrangler pages project list
```

## Git Branch Authority

| Item | Value |
|------|-------|
| Canonical git branch | `master` |
| Cloudflare production branch label | `acquisition-main` (legacy) |

**Note:** The Cloudflare Pages "production branch" setting is currently labeled `acquisition-main`. This is cosmetic only and does not affect deployed content or authority. The git branch `acquisition-main` no longer exists; all work is on `master`.

If the label needs to match, a manual Cloudflare dashboard update is required. The wrangler CLI cannot change the production branch setting.

## Deployment Workflow

1. Build static site: `uv run python scripts/build_static_site.py`
2. Deploy to Pages: `wrangler pages deploy site --project-name=mathledger-ai --branch=acquisition-main`
3. Deploy demo to Fly.io: `fly deploy`

---
*Document created: 2026-01-07 | Authority cleanup for v0.2.14*
*Updated: 2026-01-07 | Git branch unification complete*
