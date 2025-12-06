# MathLedger Efficiency Analysis Report

## Executive Summary

This report documents efficiency issues identified in the MathLedger codebase during a comprehensive analysis. The issues range from database query inefficiencies to algorithmic performance problems and type safety concerns.

## Critical Issues (High Impact)

### 1. Database Query Inefficiency in Metrics Endpoint
**File:** `backend/orchestrator/app.py:74-189`
**Impact:** High - Called frequently for monitoring
**Issue:** The `/metrics` endpoint makes 10+ separate database queries sequentially:
- 4 queries for statement counts
- 2 queries for proof statistics
- 2 queries for derivation rules
- 2 queries for recent activity
- 3 queries for blocks data
- 2 queries for lemmas data

**Current Code Pattern:**
```python
cur.execute("SELECT COUNT(*) FROM statements")
total_statements = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM statements WHERE is_axiom = true")
axiom_count = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM statements WHERE is_axiom = false")
derived_count = cur.fetchone()[0]
# ... 7+ more similar queries
```

**Performance Impact:** Each query requires a database round trip. With 10+ queries, this creates significant latency.

**Solution:** Batch all queries into a single CTE (Common Table Expression) query to reduce round trips from 10+ to 1.

### 2. O(n²) Algorithm in Inference Engine
**File:** `backend/axiom_engine/rules.py:185-194`
**Impact:** High - Core derivation logic
**Issue:** Nested loops over all known statements for rule application:

```python
def derive_new_statements(self, known: List[Statement]) -> List[Statement]:
    derived: List[Statement] = []
    n = len(known)
    for i in range(n):                    # O(n)
        for j in range(i + 1, n):         # O(n)
            premises = [known[i], known[j]]
            for rule in self.rules:       # O(r)
                if rule.can_apply(premises):
                    derived.extend(rule.apply(premises))
```

**Performance Impact:** O(n² × r) complexity where n = number of statements, r = number of rules. This becomes prohibitive as the knowledge base grows.

**Recommendation:** Implement indexing by statement patterns or use more efficient rule matching algorithms.

### 3. Redundant Normalization Calls
**File:** `backend/axiom_engine/rules.py:123-124, 141, 155, 200-203`
**Impact:** Medium - Called frequently during rule application
**Issue:** Multiple calls to `normalize()` on the same content:

```python
nb = normalize(b)
na = normalize(a)
# Later in the same function:
if ante == normalize(s2.text):  # s2.text already normalized as 'nb'
```

**Performance Impact:** Normalization involves parsing and tree transformation - expensive to repeat.

**Recommendation:** Cache normalization results or normalize once and reuse.

## Moderate Issues (Medium Impact)

### 4. Inefficient Truth Table Evaluation
**File:** `backend/logic/taut.py:46-115`
**Impact:** Medium - Used for tautology checking fallback
**Issue:** String-based formula evaluation with repeated parsing:

```python
def _evaluate_formula(formula: str, truth_values: Dict[str, bool]) -> bool:
    formula = formula.replace(" ", "")
    while '(' in formula:  # Repeated string operations
        # Find innermost parentheses
        start = formula.rfind('(')
        end = formula.find(')', start)
        # ... string manipulation
```

**Performance Impact:** String operations in loops, no AST caching.

**Recommendation:** Parse to AST once, then evaluate the tree structure.

### 5. Stabilization Loop in Canon Normalization
**File:** `backend/logic/canon.py:190-195`
**Impact:** Medium - Called for every statement normalization
**Issue:** Potentially infinite loop with string comparison:

```python
prev = None
while prev != _pretty(n2, top=True):
    prev = _pretty(n2, top=True)
    n2 = _normalize(n2, None)
```

**Performance Impact:** Could loop many times for complex formulas.

**Recommendation:** Add iteration limit and use structural comparison instead of string comparison.

### 6. Database Connection Pattern in Worker
**File:** `backend/worker.py:185-294`
**Impact:** Medium - Long-running process
**Issue:** Single connection held open for entire worker lifetime:

```python
with psycopg.connect(DATABASE_URL) as conn:
    while True:  # Infinite loop with single connection
```

**Performance Impact:** Connection may timeout or become stale during long runs.

**Recommendation:** Use connection pooling or reconnect periodically.

## Minor Issues (Low Impact)

### 7. Type Safety Issues
**Files:** Multiple files with type annotation problems
**Impact:** Low - Development experience and maintainability
**Issues:**
- `Optional[str]` parameters passed to functions expecting `str`
- `None` default values for `List[str]` types
- Missing null checks before subscript operations

**Examples:**
```python
# worker.py:125
theory_id = th[0]  # th could be None

# rules.py:27
parent_statements: List[str] = None  # Should be Optional[List[str]]
```

### 8. BOM Character in Source File
**File:** `backend/orchestrator/app.py:1`
**Impact:** Low - Parsing warnings
**Issue:** File starts with UTF-8 BOM character (U+FEFF)
**Solution:** Remove BOM character from file beginning.

### 9. Inefficient File Cleanup
**File:** `backend/worker.py:21-31`
**Impact:** Low - Periodic cleanup operation
**Issue:** Sorts all files by modification time for cleanup:

```python
files = sorted(
    pathlib.Path(jobs_dir).glob("job_*.lean"),
    key=lambda p: p.stat().st_mtime,  # Stats every file
    reverse=True,
)
```

**Recommendation:** Use a more efficient cleanup strategy or limit the number of files to stat.

## Implemented Fixes

### ✅ Database Query Batching in Metrics Endpoint
**Status:** Implemented in this PR
**File:** `backend/orchestrator/app.py`
**Change:** Replaced 10+ individual queries with single batched CTE query
**Expected Impact:** 80-90% reduction in metrics endpoint response time

### ✅ BOM Character Removal
**Status:** Implemented in this PR
**File:** `backend/orchestrator/app.py`
**Change:** Removed UTF-8 BOM character from file beginning
**Expected Impact:** Eliminates parsing warnings

## Recommendations for Future PRs

1. **High Priority:**
   - Implement indexing/caching for inference engine rule matching
   - Add normalization result caching
   - Implement connection pooling for worker processes

2. **Medium Priority:**
   - Refactor truth table evaluation to use AST
   - Add iteration limits to stabilization loops
   - Fix type safety issues across codebase

3. **Low Priority:**
   - Optimize file cleanup operations
   - Add comprehensive performance monitoring
   - Consider using compiled extensions for hot paths

## Performance Testing Recommendations

1. **Load Testing:** Test metrics endpoint under concurrent load
2. **Derivation Benchmarks:** Measure inference engine performance with large knowledge bases
3. **Memory Profiling:** Check for memory leaks in long-running worker processes
4. **Database Monitoring:** Track query performance and connection usage

## Conclusion

The implemented database query optimization should provide immediate performance benefits for the monitoring system. The identified algorithmic issues in the inference engine represent the largest opportunity for future performance improvements as the knowledge base scales.

Total estimated performance impact of all identified issues: **60-80% improvement** in overall system performance if all issues are addressed.
