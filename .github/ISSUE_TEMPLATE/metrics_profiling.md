# /metrics Endpoint Performance Profiling — Post-Merge Validation

## Context

During integration of PR #10, Claude's router-based `backend/orchestrator/app.py` rewrite (commit d7f0cc9 from `qa/claudeA-2025-09-27`) replaced Devin's optimized version (commit dad208c) which included batch CTE queries for the `/metrics` endpoint.

**Conflict Resolution Decision**: Accepted Claude's version for better architecture (router-based, schema-introspecting), **deferred** Devin's optimization re-application pending performance validation.

---

## Task

Profile `/metrics` endpoint latency to determine if Devin's optimization needs re-application.

---

## Prerequisites

1. **Deploy `integrate/ledger-v0.1`** (commit fa647ad) to staging environment
2. **Ensure representative data volume**:
   - ≥1000 statements
   - ≥500 proofs
   - Multiple blocks (≥10)

---

## Profiling Steps

### 1. Baseline Measurement (Optional — if Devin's branch still exists)

```bash
# Deploy Devin's optimized version (commit dad208c or earlier with optimization)
# to staging-baseline environment

# Run load test
ab -n 1000 -c 10 http://staging-baseline:8000/metrics > devin_baseline.txt

# Extract p99 latency
grep "99%" devin_baseline.txt
# Example output: "99%    127ms"
```

**If Devin's branch unavailable**: Use documented baseline from `EFFICIENCY_REPORT.md` (if present).

### 2. Current Measurement (Claude's version)

```bash
# Deploy integrate/ledger-v0.1 to staging

# Run load test
ab -n 1000 -c 10 http://staging:8000/metrics > claude_current.txt

# Extract p99 latency
grep "99%" claude_current.txt
# Example output: "99%    156ms"
```

### 3. Regression Analysis

```python
baseline = 127  # ms (from Devin's optimized version or EFFICIENCY_REPORT.md)
current = 156   # ms (from Claude's current version)
regression_pct = ((current - baseline) / baseline) * 100
# Example: 22.8% regression
```

---

## Decision Tree

### If regression < 20%

**Action**: Accept current performance (close this issue)

**Rationale**: Claude's router architecture benefits (maintainability, schema tolerance) outweigh minor performance cost.

### If regression ≥ 20%

**Action**: Re-apply Devin's CTE optimization inside Claude's router architecture.

**Implementation**:

1. **Extract Devin's CTE logic** (from commit `c3907c3` or `EFFICIENCY_REPORT.md`):

```sql
WITH statement_counts AS (
  SELECT
    COUNT(*) FILTER (WHERE is_axiom) as axioms,
    COUNT(*) FILTER (WHERE NOT is_axiom) as derived,
    COUNT(*) as total
  FROM statements
),
proof_stats AS (
  SELECT
    COUNT(*) FILTER (WHERE success) as success_count,
    COUNT(*) as total_proofs
  FROM proofs
),
block_info AS (
  SELECT MAX(block_number) as height
  FROM blocks
)
SELECT
  sc.axioms, sc.derived, sc.total,
  ps.success_count, ps.total_proofs,
  CASE WHEN ps.total_proofs > 0
       THEN ps.success_count::float / ps.total_proofs
       ELSE 0 END as success_rate,
  bi.height
FROM statement_counts sc, proof_stats ps, block_info bi;
```

2. **Integrate into `backend/orchestrator/app.py`**:

Find the `@ui_router.get("/metrics")` or equivalent handler and replace sequential queries with single CTE:

```python
@ui_router.get("/metrics")
def ui_metrics():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Single CTE query (replaces 10+ sequential queries)
            cur.execute("""
                WITH statement_counts AS (...),
                     proof_stats AS (...),
                     block_info AS (...)
                SELECT * FROM statement_counts, proof_stats, block_info
            """)
            row = cur.fetchone()

            return {
                "statements": {"axioms": row[0], "derived": row[1], "total": row[2]},
                "proofs": {"success": row[3], "total": row[4], "rate": row[5]},
                "block_height": row[6],
                # ... other fields
            }
```

3. **Re-run profiling**:

```bash
ab -n 1000 -c 10 http://staging:8000/metrics > optimized.txt
grep "99%" optimized.txt
# Target: p99 ≤ baseline (127ms)
```

4. **Create follow-up PR**:

```bash
git checkout -b perf/reapply-metrics-cte-$(date +%Y%m%d)
# (edit backend/orchestrator/app.py)
git commit -m "perf: re-apply /metrics CTE batch optimization

- Single CTE query replaces 10+ sequential queries
- Restores performance lost in router-based rewrite
- Profiling: Before p99=156ms, After p99=124ms (20.5% improvement)
- See issue #<THIS_ISSUE> for profiling details
"
git push -u origin perf/reapply-metrics-cte-$(date +%Y%m%d)
gh pr create --base main --title "perf: re-apply /metrics CTE optimization" \
  --body "Fixes #<THIS_ISSUE>"
```

---

## Success Criteria

- [ ] `/metrics` endpoint profiled on staging (p99 latency documented)
- [ ] Regression percentage calculated
- [ ] **If < 20%**: Issue closed (performance acceptable)
- [ ] **If ≥ 20%**: Follow-up PR created & merged with CTE optimization
- [ ] Post-optimization profiling confirms p99 ≤ baseline

---

## Owner

**Primary**: @devin (performance optimization expert)
**Secondary**: @claude (endpoint owner)
**ETA**: 48h post-merge of PR #10

---

## References

- **PR #10**: Integration of claudeA + codexA + devinA
- **Devin's optimization commit**: `c3907c3` or `dad208c`
- **Claude's rewrite commit**: `d7f0cc9` (qa/claudeA-2025-09-27)
- **EFFICIENCY_REPORT.md**: Baseline performance documentation (if exists)
- **Integration decision**: `.github/pr_bodies/release_integration_final.md`

---

## Notes

- **No blocking**: This profiling is **post-merge validation**, not a merge gate
- **Architecture preserved**: CTE optimization will be **added to** Claude's router architecture (not replace it)
- **Test coverage**: New optimization must not break existing tests (123 tests must remain green)
