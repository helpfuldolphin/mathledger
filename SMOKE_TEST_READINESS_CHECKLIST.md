# Smoke-Test Readiness Checklist

This checklist verifies that the Governance History Dashboard service is ready for a smoke test or informal developer validation.

### Objective:
Ensure the API skeleton is functional, correctly configured, and verifiable before proceeding to more complex integration testing or frontend development.

---

### Phase 1: Local Environment Setup

- [x] **Dependencies Installed:** Core dependencies from `requirements.txt` are installed.
- [x] **Dev Dependencies Installed:** Optional dependencies from `requirements-dev.txt` are installed.
- [x] **Configuration Understood:** No explicit configuration files are needed; the SQLite DB path is hardcoded (`governance_history.db`) for simplicity.

### Phase 2: Core Functionality Verification

- [x] **Service Starts:** The FastAPI application can be started locally without errors using `uvicorn backend.dashboard.api:app`.
- [x] **Database Initializes:** On first run, the `governance_history.db` file is created.
- [x] **Schema Correct:** The `governance_events` table is created, and `PRAGMA table_info('governance_events')` confirms its structure.
- [x] **Indexes Correct:** The `idx_governance_events_layer_timestamp` and `idx_governance_events_timestamp` indexes are created on the `governance_events` table, as verified by `test_api.py`.

### Phase 3: API Endpoint Validation

- [x] **Automated Core Tests Pass:** The `pytest backend/dashboard/test_api.py` suite passes, confirming API contract adherence for filtering and edge cases.
- [x] **Automated Smoke Test (Empty State):** The `test_smoke.py` suite verifies that the API returns an empty `data` array for a fresh, empty database.
- [x] **Automated Smoke Test (Populated State):** The `test_smoke.py` suite verifies that after programmatically populating the database, the API returns the correct, non-empty data.
- [x] **Automated Smoke Test (Filtering):** The core `test_api.py` suite already verifies API filtering logic (e.g., by layer).

### Phase 4: Supporting Components



- [x] **Retention Worker Stub:** The retention worker `retention.py` can be executed (`python -m backend.dashboard.retention`) and logs its no-op actions.

- [ ] **Load-Test Stub:** The `locust` load test can be started (`locust -f backend/dashboard/load_test.py`) and can connect to the running API server.



---



### Governance Perspective on Automation







Automating the smoke tests, rather than relying on manual `curl` checks, is a deliberate choice from a governance standpoint. It provides:







1.  **Repeatability and Consistency:** Automated tests execute the exact same validation steps every single time, removing the risk of human error or shortcuts. This ensures that every pre-deployment check is consistent and trustworthy.



2.  **Traceability and Audit Trail:** The output of an automated test run serves as a formal, archivable artifact. It provides a timestamped, auditable record that the system passed its required health checks before a change was promoted, which is crucial for compliance and incident analysis.



3.  **CI/CD Gatekeeping:** Automated tests can be integrated directly into a CI/CD pipeline as a mandatory quality gate. This programmatically enforces governance by preventing code that fails its basic smoke test from ever reaching a staging or production environment. Manual checks offer no such reliable enforcement mechanism.







From a pure auditability standpoint, manual `curl` checks produce ephemeral evidence that is difficult to verify, often relying on screenshots or developer assertions. In contrast, an automated test integrated into a version control system provides a non-repudiable audit trail. An auditor can link a specific test result (Pass/Fail) directly to a code commit, a timestamp, the exact test code that was executed, and the logged output of that execution. This creates a chain of evidence that is structured, verifiable, and can be archived automatically, satisfying formal governance and compliance requirements that manual checks can never meet.







[x] **READY FOR SMOKE TEST**
This checklist represents the minimum reproducible evidence required for Phase X readiness.




