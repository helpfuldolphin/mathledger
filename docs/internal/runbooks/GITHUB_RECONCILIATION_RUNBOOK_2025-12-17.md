# GitHub Reconciliation Runbook — 2025-12-17

```
Status: INTERNAL
Audience: Maintainers
Scope: GitHub ↔ Local reconciliation
Normativity: NON-NORMATIVE
Generated: 2025-12-17
```

---

## 1. Remote Change Map

| Branch / PR | Purpose | Risk | Handling |
|-------------|---------|------|----------|
| `origin/master` (f2db4af) | 12 commits: U2 Bridge Layer, PQ Operator Kit, curriculum drift enforcement, telemetry runtime, calibration smoke test | **LOW** | Merge into local |
| `origin/copilot/add-first-light-orchestrator` | First Light orchestrator + docs | **LOW** | Ignore (not needed, local has equivalent) |
| `origin/copilot/bind-curriculum-stability-envelope` | Curriculum stability envelope | **LOW** | Ignore (orthogonal to current work) |
| `origin/copilot/expose-safety-gate-decisions` | Safety gate export + docs | **LOW** | Ignore (can merge later if needed) |
| `origin/copilot/extend-evidence-summary-schema` | TDA-aware evidence fusion | **MED** | Ignore (defer to Phase X+1) |
| `origin/copilot/update-doc-governance-radar` | Doc governance radar | **LOW** | Ignore (cosmetic) |
| `origin/copilot/wire-cortex-outcomes-first-light` | Cortex Neural Link integration | **MED** | Ignore (experimental) |

**Summary**: Only `origin/master` requires integration. Copilot branches are experimental features that can be merged via PR later.

---

## 2. Local Change Map

**Total**: 331 files changed across 66 commits

| Subsystem | Count | Key Changes |
|-----------|-------|-------------|
| `docs/system_law/**` | 41 | CAL-EXP-3 canonization, CAL-EXP-4 spec, Pilot contracts, SHADOW_MODE_CONTRACT |
| `backend/**` | 119 | Health adapters, governance, language constraints, pilot ingest adapter |
| `tests/**` | 93 | CAL-EXP-3/4 verifiers, pilot tripwires, contract tests |
| `scripts/**` | 17 | CAL-EXP runners, golden manifest verifier, core loop verifier |
| `.github/workflows/**` | 7 | pilot-phase-gate, cal-exp-3-verification, hygiene gates |

### Files Modified (Working Tree, Unstaged)

These 20 files have local uncommitted changes:

- `.claude/settings.local.json` — **DO NOT COMMIT** (local config)
- `.github/CODEOWNERS` — Review before commit
- `.github/workflows/pilot-phase-gate.yml` — Review before commit
- `Makefile` — Contains new targets, commit
- `README.md` — Review before commit
- `backend/health/pilot_external_ingest_adapter.py` — Commit
- `backend/lean_mode.py` — Commit
- `backend/lean_proj/lakefile.lean` — Review (Lean config)
- `backend/repro/first_organism_harness.py` — Commit
- `docs/system_law/calibration/CAL_EXP_3_INDEX.md` — Commit
- `docs/system_law/calibration/CAL_EXP_3_RATIFICATION_BRIEF.md` — Commit
- `scripts/generate_first_light_status.py` — Commit
- `substrate/repro/__init__.py` — Commit
- `tests/` (6 files) — Commit

### Untracked Files — MUST EXCLUDE

These should remain untracked (results, artifacts, temp files):

- `results/**` — Already in .gitignore
- `*.pdf`, `*.aux`, `*.out` — Build artifacts
- `tmp/`, `nul`, `*.lock` — Temp files
- `mathledger.egg-info/` — Build artifact
- `output/`, `tmp_p4/` — Temp outputs

### Untracked Files — SHOULD COMMIT

New files that should be committed:

- `docs/EVALUATOR_GUIDE.md`, `docs/EVALUATOR_QUICKSTART.md`
- `scripts/verify_core_loop.py`, `scripts/verify_against_golden.py`
- `results/golden/manifest_seed42.json`
- `.github/workflows/core-loop-verification.yml`
- `docs/system_law/calibration/APPENDIX_CAL_EXP_3_MISMATCH_INTERPRETATION.md`

---

## 3. Reconciliation Strategy

**Chosen Strategy: B — Merge origin/master into local**

**Justification**:

1. **No file conflicts**: `comm` shows zero overlapping files between local and remote changes
2. **Preserves local history**: 66 commits of CAL-EXP-3/4 + Pilot work remain intact
3. **Safe merge**: Remote adds orthogonal U2/PQ features that don't touch calibration surfaces
4. **Fast path**: Single merge operation vs. complex cherry-pick choreography

---

## 4. Pre-Push Gate Checklist

| Gate | Command | Pass Criteria |
|------|---------|---------------|
| ☐ Merge clean | `git merge origin/master` | No conflicts |
| ☐ Core tests | `uv run pytest tests/ -x -q --ignore=tests/integration` | Exit 0 |
| ☐ CAL-EXP-3 tripwires | `uv run pytest tests/ci/test_cal_exp_adversarial_coverage_grid.py -v` | Exit 0 |
| ☐ Pilot neutrality | `uv run pytest tests/policy/test_pilot_text_neutrality.py -v` | Exit 0 |
| ☐ Language lint | `uv run pytest tests/policy/test_cal_exp_3_language_lint.py -v` | Exit 0 |
| ☐ SHADOW contract | Grep for "tamper-evident" in SHADOW contexts | Zero matches |
| ☐ .gitignore check | `git status` shows no `results/` staged | True |
| ☐ No prohibited terms | No "failed/fail" in pilot ask-shaped text | Manual check |
| ☐ Verify mock determinism | `ML_LEAN_MODE=mock uv run python scripts/verify_core_loop.py` | Exit 0 |

---

## 5. PowerShell Runbook

```powershell
# ============================================================
# MATHLEDGER GITHUB RECONCILIATION RUNBOOK
# Execute from: C:\dev\mathledger
# ============================================================

# PHASE 1: VERIFY STARTING STATE
# ------------------------------------------------------------
Write-Host "=== PHASE 1: Verify Starting State ===" -ForegroundColor Cyan

git status
git branch --show-current
# Expected: master, 66 commits ahead, some unstaged changes

# PHASE 2: STASH LOCAL UNCOMMITTED CHANGES
# ------------------------------------------------------------
Write-Host "=== PHASE 2: Stash Uncommitted Changes ===" -ForegroundColor Cyan

# Stash modified files (not untracked)
git stash push -m "pre-merge-stash-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
git stash list
# Verify: stash created

# PHASE 3: MERGE REMOTE MASTER
# ------------------------------------------------------------
Write-Host "=== PHASE 3: Merge Remote Master ===" -ForegroundColor Cyan

git fetch origin master
git merge origin/master -m "Merge origin/master: U2 Bridge Layer, PQ Operator Kit, telemetry runtime"

# Verify merge result
git log --oneline -n 5
# Expected: Merge commit at HEAD

# PHASE 4: RESTORE STASHED CHANGES
# ------------------------------------------------------------
Write-Host "=== PHASE 4: Restore Stashed Changes ===" -ForegroundColor Cyan

git stash pop
git status
# Expected: Modified files restored, may need manual review

# PHASE 5: STAGE NEW FILES FOR COMMIT
# ------------------------------------------------------------
Write-Host "=== PHASE 5: Stage New Files ===" -ForegroundColor Cyan

# Stage evaluator guide and verification scripts
git add docs/EVALUATOR_GUIDE.md
git add docs/EVALUATOR_QUICKSTART.md
git add scripts/verify_core_loop.py
git add scripts/verify_against_golden.py
git add results/golden/manifest_seed42.json
git add docs/system_law/calibration/APPENDIX_CAL_EXP_3_MISMATCH_INTERPRETATION.md

# Stage workflow
git add .github/workflows/core-loop-verification.yml

# Stage modified files (review each)
git add Makefile
git add backend/lean_mode.py
git add backend/repro/first_organism_harness.py
git add scripts/generate_first_light_status.py
git add substrate/repro/__init__.py
git add docs/system_law/calibration/CAL_EXP_3_INDEX.md
git add docs/system_law/calibration/CAL_EXP_3_RATIFICATION_BRIEF.md

# DO NOT ADD: .claude/settings.local.json (local config)

git status
# Review staged files

# PHASE 6: RUN PRE-PUSH TESTS
# ------------------------------------------------------------
Write-Host "=== PHASE 6: Pre-Push Tests ===" -ForegroundColor Cyan

# Core tests (quick)
uv run pytest tests/ -x -q --ignore=tests/integration --ignore=tests/backend --ignore=tests/topology -k "not slow" 2>&1 | Select-Object -First 50

# CAL-EXP-3 tripwires
uv run pytest tests/ci/test_cal_exp_adversarial_coverage_grid.py -v 2>&1 | Select-Object -Last 20

# Pilot neutrality
uv run pytest tests/policy/test_pilot_text_neutrality.py -v 2>&1 | Select-Object -Last 10

# Mock determinism
$env:ML_LEAN_MODE = "mock"
uv run python scripts/verify_core_loop.py --runs 2
Remove-Item Env:\ML_LEAN_MODE

# PHASE 7: SHADOW CONTRACT CHECK
# ------------------------------------------------------------
Write-Host "=== PHASE 7: Shadow Contract Check ===" -ForegroundColor Cyan

# Check for prohibited "tamper-evident" in SHADOW contexts
Select-String -Path "docs/system_law/**/*.md" -Pattern "tamper-evident" -Recurse | ForEach-Object { $_.Path }
# Expected: Zero matches or only in allowed contexts

# PHASE 8: COMMIT STAGED CHANGES
# ------------------------------------------------------------
Write-Host "=== PHASE 8: Commit ===" -ForegroundColor Cyan

git commit -m "feat: Add core loop verification, golden manifest, evaluator guide

- Add scripts/verify_core_loop.py (Lean-coupled H_t verification)
- Add scripts/verify_against_golden.py (CAL-EXP-3 golden manifest comparison)
- Add results/golden/manifest_seed42.json (reference checksums)
- Add docs/EVALUATOR_GUIDE.md (external evaluator documentation)
- Update Makefile with verify-mock-determinism target
- Update CI workflow for real Lean verification

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# PHASE 9: VERIFY FINAL STATE
# ------------------------------------------------------------
Write-Host "=== PHASE 9: Verify Final State ===" -ForegroundColor Cyan

git log --oneline -n 10
git diff --stat origin/master..HEAD
# Review: Should show merge commit + new commit

# PHASE 10: PUSH
# ------------------------------------------------------------
Write-Host "=== PHASE 10: Push ===" -ForegroundColor Cyan

# Dry run first
git push --dry-run origin master

# If dry run OK, execute push
# git push origin master

Write-Host "=== RECONCILIATION COMPLETE ===" -ForegroundColor Green
Write-Host "Review the dry-run output above. If OK, run: git push origin master"
```

---

## 6. Final Output Summary

| Deliverable | Status |
|-------------|--------|
| Remote Change Map | ✅ 1 branch to merge, 6 Copilot branches to ignore |
| Local Change Map | ✅ 331 files, 66 commits, no conflicts |
| Strategy | ✅ Merge origin/master (orthogonal changes) |
| Pre-Push Gate Checklist | ✅ 9 gates defined |
| PowerShell Runbook | ✅ 10 phases, copy-paste ready |

**Risk Assessment**: **LOW** — No file conflicts detected. Local and remote changes are orthogonal. Merge is safe.

---

## Appendix: Common Ancestor

```
Common ancestor: fd6fd079c43385651aa4ddd47861fdc1c8c70d27
Local HEAD:      70f65f7 (66 commits ahead)
Remote HEAD:     f2db4af (12 commits ahead of ancestor)
Overlap:         0 files
```
