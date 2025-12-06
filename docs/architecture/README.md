# Architecture Documentation Index

Welcome to the MathLedger architecture documentation. These documents provide comprehensive oversight of the repository structure, code quality, and recommended refactoring strategies.

## Documents in This Directory

### 1. [OVERSIGHT_REPORT.md](./OVERSIGHT_REPORT.md)
**Status:** ✅ Complete  
**Type:** Comprehensive Analysis  
**Length:** 520 lines

**Contents:**
- Executive summary of architectural findings
- Detailed duplicate code analysis (28 instances)
- High-complexity hotspot identification (10 functions)
- Security analysis (domain separation issues)
- CI workflow overlap analysis
- Phased refactor plan with impact estimates
- Success metrics and migration checklist

**Key Findings:**
- 12 duplicate RFC 8785 canonicalization implementations
- 6 duplicate SHA-256 hashing functions
- 3 duplicate Merkle tree implementations
- 10 high-complexity functions (D-F grade)
- 5 files >800 lines requiring decomposition

**Read this if:** You need the full detailed analysis with evidence and recommendations.

---

### 2. [REFACTOR_QUICK_REFERENCE.md](./REFACTOR_QUICK_REFERENCE.md)
**Status:** ✅ Complete  
**Type:** Developer Guide  
**Length:** 150 lines

**Contents:**
- Quick migration guide (DO/DON'T examples)
- Module import reference (current and future)
- Pre-commit checklist
- Common anti-patterns to avoid
- Complexity thresholds
- Tool usage examples

**Example snippets:**
```python
# ✅ DO: Use centralized crypto
from backend.crypto.hashing import sha256_hex, DOMAIN_STMT
hash = sha256_hex(statement, domain=DOMAIN_STMT)

# ❌ DON'T: Implement local crypto
def compute_sha256(content): ...
```

**Read this if:** You're writing new code or refactoring existing code.

---

### 3. [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md)
**Status:** ✅ Complete  
**Type:** Visual Structure Guide  
**Length:** 300 lines

**Contents:**
- Current vs. proposed directory structure
- Module dependency flow diagrams
- File size before/after projections
- Complexity reduction targets
- Security improvement examples
- Migration path visualization
- Success criteria checklist

**Visual comparisons:**
- Directory tree: Current (fragmented) vs Proposed (consolidated)
- Dependency graphs showing centralization
- Complexity metrics before/after refactoring

**Read this if:** You want to understand the big picture of the refactoring effort.

---

## Quick Navigation

### For Architects / Tech Leads
1. Start with [OVERSIGHT_REPORT.md](./OVERSIGHT_REPORT.md) - Read sections 1-6
2. Review [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) - Understand proposed structure
3. Prioritize phases based on Section 7 (Estimated Impact Summary)

### For Developers
1. Start with [REFACTOR_QUICK_REFERENCE.md](./REFACTOR_QUICK_REFERENCE.md)
2. Use Pre-commit Checklist before submitting PRs
3. Reference [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) for module locations

### For Security Reviewers
1. Read OVERSIGHT_REPORT.md Section 5 (Security & Crypto Consolidation)
2. Review Appendix B (Security Considerations)
3. Check ARCHITECTURE_DIAGRAM.md "Security Improvement" section

### For New Contributors
1. Read [REFACTOR_QUICK_REFERENCE.md](./REFACTOR_QUICK_REFERENCE.md) - Learn patterns
2. Scan [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) - Understand structure
3. Follow pre-commit checklist

---

## Implementation Status

### Completed ✅
- [x] Architecture analysis (radon, static analysis)
- [x] Duplicate code identification
- [x] Complexity hotspot analysis
- [x] CI workflow overlap analysis
- [x] Security vulnerability identification
- [x] Documentation creation

### Phase 1: Core Crypto Consolidation (NEXT - HIGH PRIORITY)
- [ ] Create `backend/core/crypto/canon.py`
- [ ] Create `backend/core/output/status.py`
- [ ] Add tests for new modules
- [ ] Update tools/ to use centralized modules
- [ ] Update scripts/ to use centralized modules

**Estimated:** 2-3 days  
**Impact:** HIGH - Security, maintainability

### Phase 2: Output Standardization (MEDIUM PRIORITY)
- [ ] Implement standardized pass/fail/abstain logging
- [ ] Update 15+ files to use standardized output
- [ ] Add CI parsing for standardized format

**Estimated:** 1-2 days  
**Impact:** MEDIUM - Observability

### Phase 3: Complexity Reduction (MEDIUM PRIORITY)
- [ ] Decompose backend/logic/canon.py
- [ ] Split backend/orchestrator/app.py
- [ ] Extract backend/axiom_engine smoke tests
- [ ] Reduce cyclomatic complexity to ≤10

**Estimated:** 3-4 days  
**Impact:** MEDIUM - Maintainability

### Phase 4: CI Consolidation (LOW PRIORITY)
- [ ] Create composite GitHub Actions
- [ ] Merge duplicate workflows
- [ ] Standardize workflow triggers

**Estimated:** 2-3 days  
**Impact:** LOW - Maintenance burden

---

## Key Metrics

### Current State
| Metric | Value | Status |
|--------|-------|--------|
| Duplicate crypto implementations | 28 | 🔴 High |
| High-complexity functions (D-F) | 10 | 🔴 High |
| Files >800 lines | 5 | 🟡 Medium |
| Files using centralized crypto | 6 | 🔴 Low (46%) |
| CI workflow duplication | High | 🔴 High |

### Target State
| Metric | Value | Status |
|--------|-------|--------|
| Duplicate crypto implementations | 0 | 🟢 Goal |
| High-complexity functions (D-F) | 0 | 🟢 Goal |
| Files >800 lines | 0 | 🟢 Goal |
| Files using centralized crypto | 19 (100%) | 🟢 Goal |
| CI workflow duplication | Low | 🟢 Goal |

---

### Related Documentation

### Repository Root
- [AGENTS.md](../../AGENTS.md) - Agent-specific coding guidelines
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [README.md](../../README.md) - Project overview

### Documentation Directory
- [docs/FIRST_ORGANISM.md](../FIRST_ORGANISM.md) - Operational realization of the RFL loop
- [docs/VERIFICATION.md](../VERIFICATION.md) - Verification procedures
- [docs/CONTRIBUTING.md](../CONTRIBUTING.md) - Extended contribution guide
- [docs/security/](../security/) - Security documentation

---

## Maintenance

### Document Update Schedule
- **OVERSIGHT_REPORT.md**: Update after major refactoring phases
- **REFACTOR_QUICK_REFERENCE.md**: Update when new patterns emerge
- **ARCHITECTURE_DIAGRAM.md**: Update when structure changes

### Last Updated
- OVERSIGHT_REPORT.md: 2025-11-02
- REFACTOR_QUICK_REFERENCE.md: 2025-11-02
- ARCHITECTURE_DIAGRAM.md: 2025-11-02
- README.md (this file): 2025-11-02

---

## Questions or Feedback?

For questions about:
- **Architecture decisions**: Review OVERSIGHT_REPORT.md Section 11 (Appendix C: References)
- **Implementation details**: See REFACTOR_QUICK_REFERENCE.md
- **Migration timeline**: See OVERSIGHT_REPORT.md Section 7 (Estimated Impact Summary)

---

## Pass-Lines Status

```
[PASS] Architecture Review Complete (duplicates=28, hotspots=10)
[PASS] Oversight Report Committed (docs/architecture/OVERSIGHT_REPORT.md)
[PASS] Developer Documentation Complete
```

---

*Architecture documentation maintained by Copilot-Architect*  
*Generated: 2025-11-02*
