# Risk Audit v2: Phase II Experiments

## DELIVERABLE

**Risk Audit v2**
**Date**: 2025-11-30
**Target**: Phase II Experiments (RFL & Dual-Attestation)

### 1. Risk Inventory & Categorization

| Risk ID | Description | Evidence | Severity |
| :--- | :--- | :--- | :--- |
| **R-01** | **Baseline Pathology (Import Error)**<br>Critical failure in module resolution preventing test collection. | `ImportError: cannot import name 'derive'` in `test_system_id_verification.py`. | **CRITICAL** |
| **R-02** | **DB Test Failure (Zero Collection)**<br>Test runner fails to collect any tests in DB suite, masking schema validity. | `_fail_db.txt` shows "collected 0 items". | **MAJOR** |
| **R-03** | **RFL Empty Output**<br>Core logic failure; derivation engine returns empty or null results during experiments. | Inferred from `R-01` and prompt context (broken `derive` imports). | **MAJOR** |

### 2. Mitigation Recommendations

**For R-01 (Baseline Pathology):**
*   **Immediate:** Inspect `backend/axiom_engine/derive.py` and verify the `derive` symbol exists and is exported.
*   **Action:** Refactor `test_system_id_verification.py` to match the actual package structure.

**For R-02 (DB Failures):**
*   **Immediate:** Run `pytest --collect-only` after fixing R-01 to ensure discovery works.
*   **Action:** Verify `conftest.py` or `pytest.ini` configuration isn't excluding paths unexpectedly.

**For R-03 (RFL Empty Output):**
*   **Action:** Add a "sanity check" unit test for `derive()` that asserts non-empty output for a standard input *before* running full experiments.

### 3. Stop Conditions (Halt Protocols)

Experiments **MUST HALT** if any of the following occur:

1.  **Test Collection Failure:** `pytest` returns an error code or collects 0 items.
2.  **Import/Syntax Errors:** Any `ImportError` or `SyntaxError` in the `backend` namespace.
3.  **Empty Derivation:** A standard "Identity Axiom" input yields 0 output steps or null result.
4.  **Schema Mismatch:** DB migration verification fails.

---

## CHECKLIST

*   [x] **Risks Identified:** Baseline pathology, RFL empty output, DB failures included.
*   [x] **Severity Assigned:** Critical/Major/Minor (Critical used for blocker).
*   [x] **Mitigations:** Specific actions provided for each risk.
*   [x] **Stop Conditions:** Operational halt criteria defined.
*   [x] **Constraint Check:** Concise, direct, and operational.

---

## NEXT STEPS

**Researcher: Integration Specialist (or equivalent)**

1.  **Resolve R-01:** Fix the `ImportError` in `backend/axiom_engine/derive.py` / `test_system_id_verification.py`.
2.  **Verify R-02:** Confirm `pytest` collects > 0 tests.
3.  **Sanity Check:** Run a minimal `derive()` test to clear R-03.
4.  **Report:** Update `_fail_full.txt` status.
