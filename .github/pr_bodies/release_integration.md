## Integration Summary (Claude D)
Included branches:
- qa/claudeA-2025-09-27 — exporter --input/--dry-run + linter-first
- qa/codexA-2025-09-27 — QA tests + hygiene fixes
- perf/devinA-modus-ponens-opt-20250920 — O(n²)→O(n) MP optimization

Gates:
- Unit (NO_NETWORK): ✅ green locally
- Hygiene (pre-commit minimal): ✅
- Coverage: (snapshot here)

Perf note:
- /metrics profiled; (re-applied CTE if regression >20%; include before/after p99)

Follow-ups:
- [ ] Update derive tests to DerivationEngine
- [ ] Align exporter QA fixture (valid V1 record)
- [ ] (If regression) perf re-apply PR

Rollback:
- Revert merge-sha if regression; feature branches preserved.
