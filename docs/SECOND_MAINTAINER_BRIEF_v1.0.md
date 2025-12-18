# Second Maintainer Brief v1.0

**Date**: 2025-12-17
**From**: MathLedger Project
**To**: Prospective Independent Verifier

---

## Purpose

MathLedger is a verifiable ledger of mathematical truths with a governance-first architecture. We seek an independent second maintainer to verify our claims before we make them publicly. This is not a recruitment pitch—it's a request for critical review.

## What I'm Asking You To Do

Spend 10–20 hours running our verification tooling and producing a short (1–2 page) **Verification Note** documenting what you observed. The goal is independent reproduction, not endorsement.

Specifically:
- Clone the repository and run our test suites
- Attempt the verification checklist below
- Document what passed, what failed, and what claims you consider supported vs. unsupported
- Flag any stop-ship issues or items requiring clarification

You are welcome to be hostile and critical. We prefer honest skepticism over polite agreement.

## What I'm Not Asking

- Feature development or code contributions
- Marketing help or public endorsement
- Advice on fundraising, valuation, or business strategy
- Review of anything outside the verification scope below

## Time & Compensation

**Estimated time**: 10–20 hours for a strong candidate familiar with Python, formal methods, or cryptographic tooling.

**Compensation options** (your choice):
1. **Paid arrangement**: We can discuss an hourly or fixed rate
2. **Attribution only**: Acknowledgment in project documentation and release notes

## What You Get

- Early access to a governance-first codebase with explicit SHADOW MODE contracts
- Your Verification Note (if favorable) may be cited in our documentation
- If you find serious issues, you help prevent overclaims—which we value

## Verification Checklist

Run these commands and record results. Lean installation is optional for most checks.

```bash
# 1. Clone and install
git clone https://github.com/helpfuldolphin/mathledger.git
cd mathledger
uv sync

# 2. Generate a test signing keypair
uv run python scripts/generate_signing_keypair.py --output-dir tmp_keys --name test

# 3. Run manifest signing tests (19 tests)
uv run pytest tests/evidence/test_manifest_signing*.py -v

# 4. Run mock determinism check (no Lean required)
uv run python scripts/verify_first_light_determinism.py --mode mock

# 5. (Optional) Run real Lean verification for a single proof
# Requires Lean 4 installation
make verify-lean-single PROOF=backend/lean_proj/ML/Taut.lean

# 6. Confirm governance documents exist
ls docs/system_law/SHADOW_MODE_CONTRACT.md
ls docs/system_law/SHADOW_GRADUATION_POLICY.md
ls docs/system_law/PHASE_2_GOVERNANCE_CLOSURE.md

# 7. Cleanup
rm -rf tmp_keys
```

For a faster path (~10 minutes), see `docs/EVALUATOR_10_MINUTE_CHECK.md`.

## Deliverable: Verification Note

Produce a 1–2 page document addressing:

1. **What you ran**: Commands executed, environment details
2. **What passed/failed**: Test results, any errors encountered
3. **Claims supported**: Which project claims you consider adequately demonstrated
4. **Claims unsupported**: Which claims lack sufficient evidence or require qualification
5. **Stop-ship issues**: Anything that should block public release
6. **Clarification needed**: Items that are ambiguous or under-documented

Format: Markdown or PDF. Plain language preferred over formal proof notation.

## Confidentiality & Sharing

Please do not publicize your findings until we've had a chance to discuss them. You're welcome to discuss the project privately with colleagues or advisors (e.g., if you're a PhD student, your advisor is fine). No NDA required—this is a good-faith request.

## Contact

If you're open to this, reply with:

1. Your availability window (e.g., "2 weeks starting Jan 15")
2. Preferred contact method (email, Signal, etc.)
3. Whether you prefer paid arrangement or attribution only

---

*This document is version-controlled at `docs/SECOND_MAINTAINER_BRIEF_v1.0.md` and may be shared with prospective verifiers.*
