# Phase IX Pull Request Decomposition Plan

This document breaks down the Phase IX operationalization plan into a sequence of four distinct, verifiable Pull Requests. Each PR is designed to be a self-contained unit of work with clear acceptance criteria and a straightforward rollback procedure.

---

### **PR #1: Attestor Service Containerization & Health Endpoint**

**Scope:** Package the existing `phase_ix_attestation.py` script into a long-running, containerized service and expose a basic health check endpoint. This PR transitions the attestor from a manual tool to a deployable service.

*   **Files Touched:**
    *   `phase_ix_attestation.py`: Modify to run in a continuous loop. Add a lightweight web server (e.g., FastAPI) to serve a `/health` endpoint.
    *   `docker-compose.yml`: Add a new service definition for `attestor` that builds from the new Dockerfile.
    *   `Dockerfile.attestor` (New File): A new Dockerfile to define the container image for the attestor service.
    *   `pyproject.toml`: Add dependencies for the web server (e.g., `fastapi`, `uvicorn`).

*   **Acceptance Tests:**
    - [ ] The command `docker-compose up attestor` successfully builds and starts the service without errors.
    - [ ] An HTTP GET request to the service's `/health` endpoint returns a `200 OK` response with the JSON body `{"status": "healthy"}`.
    - [ ] Service logs (`docker-compose logs attestor`) show the attestation check being performed in a repeating loop.

*   **Tracking:**
| Owner | ETA | Status |
|---|---|---|
| TBD | TBD | TODO |

*   **Rollback Plan:**
    1.  Revert the commit.
    2.  Comment out the `attestor` service definition in `docker-compose.yml` and redeploy the stack. The system returns to its previous state where attestation is performed via the manual script.

*   **Risk Register:**
    *   **Failure Mode:** The `attestor` container fails to start due to misconfiguration or dependency conflicts. The running service could crash repeatedly or fail to connect to the database.
    *   **Detection Signal:** `docker-compose up` command fails. Container enters a crash-loop state (visible via `docker ps`). Health endpoint (`/health`) is unreachable or non-200 status. High rate of errors in container logs.
    *   **Rollback Trigger:** Health endpoint is down for > 5 minutes post-deployment. Container restarts more than 5 times in 15 minutes. Any critical failure during the canary deployment phase.

---

### **PR #2: Implement Asynchronous Logging Queue**

**Scope:** Address the observed non-determinism by refactoring the core simulation's logging mechanism to be asynchronous, decoupling I/O from the simulation loop.

*   **Files Touched:**
    *   `backend/simulation_core.py` (Assumed path): Modify the logging calls to place log messages onto a thread-safe in-memory queue instead of writing directly to a file.
    *   `backend/logging_worker.py` (New File): Create a new worker that runs in a background thread. It will consume messages from the queue and perform the file I/O.
    *   `start_api_server.py` (Assumed path): Modify the application's startup sequence to initialize and start the `LoggingWorker` thread.

*   **Acceptance Tests:**
    - [ ] All existing integration and unit tests must pass without modification.
    - [ ] The content and format of log files generated during test runs must be identical to the files generated before this change. A `diff` of the log files should show no changes.
    - [ ] Application performance under load should be equivalent or slightly improved due to the non-blocking nature of the new logging.

*   **Tracking:**
| Owner | ETA | Status |
|---|---|---|
| TBD | TBD | TODO |

*   **Rollback Plan:**
    1.  Revert the commit. This change is self-contained within the application's codebase, making a code revert a clean rollback strategy.
    2.  Redeploy the application.

*   **Risk Register:**
    *   **Failure Mode:** The background logging worker crashes silently, leading to log loss. The in-memory queue grows unbounded due to a slow or stuck worker, causing an Out-Of-Memory (OOM) error.
    *   **Detection Signal:** Gaps appear in application logs during integration tests. Application memory usage climbs steadily under load (queue leak). Log entries are visibly garbled or incomplete.
    *   **Rollback Trigger:** Any evidence of log loss or corruption during staging tests. Unexplained memory growth is observed during load testing.

---

### **PR #3: Metrics Export & Prometheus Integration**

**Scope:** Implement the metrics and SLAs defined in the operationalization plan. Expose these metrics from the `attestor` service for consumption by a Prometheus monitoring server.

*   **Files Touched:**
    *   `phase_ix_attestation.py`: Add logic to calculate `attestation_success_rate`, `attestation_latency_p99_ms`, and other defined metrics. Integrate a Prometheus client library to expose these on a `/metrics` endpoint.
    *   `pyproject.toml`: Add the `prometheus-client` library as a dependency.
    *   `infra/prometheus/prometheus.yml` (Assumed path): Add a new scrape configuration to target the `attestor` service's `/metrics` endpoint.
    *   `docker-compose.yml`: Ensure the `attestor` service's metrics port is exposed correctly for the Prometheus container.

*   **Acceptance Tests:**
    - [ ] An HTTP GET request to the `attestor` service's `/metrics` endpoint returns a `200 OK` response in the Prometheus exposition format.
    - [ ] The response body contains the required metrics (e.g., `attestation_success_rate`).
    - [ ] In the Prometheus UI, the `attestor` target appears as "UP".
    - [ ] Metrics from the service can be successfully queried using the Prometheus query browser.

*   **Tracking:**
| Owner | ETA | Status |
|---|---|---|
| TBD | TBD | TODO |

*   **Rollback Plan:**
    1.  Revert the commit containing the code and configuration changes.
    2.  Redeploy the `attestor` and Prometheus services. Prometheus will cease scraping the endpoint, and the `attestor` will function as it did after PR #1.

*   **Risk Register:**
    *   **Failure Mode:** The `/metrics` endpoint is unavailable or serves malformed data. Metric calculation is buggy, producing inaccurate values. Performance of the attestation loop is degraded.
    *   **Detection Signal:** Prometheus target for `attestor` shows as "DOWN". `promtool check metrics` fails on the endpoint's output. `attestation_latency_p99_ms` metric shows a significant increase. Dashboards show flatlined or incorrect data.
    *   **Rollback Trigger:** Prometheus target is "DOWN" for > 5 minutes. Key metrics are clearly inaccurate. P99 latency SLA is breached immediately following rollout.

---

### **PR #4: Concurrency Stress Test Harness**

**Scope:** Create a new, dedicated integration test designed to validate the fix implemented in PR #2. This test will attempt to reproduce the original race condition by generating high-concurrency load on the state transition and logging systems.

*   **Files Touched:**
    *   `test_concurrency_stress.py` (New File): A new test file containing the stress test. It will use Python's `threading` or `concurrent.futures` module to spawn numerous threads that simultaneously trigger state transitions.
    *   The test will verify the integrity of the resulting log file, ensuring no data corruption or missed log entries occurred.
    *   `Makefile` or `.github/workflows/ci.yml`: Add the new stress test to the main CI/CD test suite to ensure it runs on every future commit.

*   **Acceptance Tests:**
    - [ ] The test suite (`pytest test_concurrency_stress.py`) must pass reliably when run against the codebase containing the async logging fix (PR #2).
    - [ ] (Verification) When the test is run against the codebase *before* PR #2, it should fail consistently, proving that it effectively catches the bug it was designed to prevent.
    - [ ] The test is successfully integrated into the CI pipeline and does not cause pipeline instability.

*   **Tracking:**
| Owner | ETA | Status |
|---|---|---|
| TBD | TBD | TODO |

*   **Rollback Plan:**
    1.  This is a test-only change and has no impact on production systems.
    2.  If the test proves to be flaky, it can be disabled in the CI configuration file (`Makefile` or CI workflow YAML) as a first step.
    3.  The commit can be safely reverted without affecting any deployed application code.

*   **Risk Register:**
    *   **Failure Mode:** The test is "flaky"—it fails intermittently even with the fix in place, destabilizing the CI pipeline. The test is not stressful enough and provides a false sense of security.
    *   **Detection Signal:** CI pipeline becomes unstable after the merge. Manual runs of the test show inconsistent pass/fail results. A known-bad commit (pre-PR#2) passes the test.
    *   **Rollback Trigger:** CI stability is degraded. The test is proven to be non-deterministic and should be reverted or disabled immediately to unblock the pipeline.

---

### **PR Dependency Graph**

The four PRs can be executed in two parallel tracks.

```ascii
          [ Start ]
              |
      +-------+-------+
      |               |
      v               v
  [ PR #1 ]         [ PR #2 ]
(Attestor)        (Async Log)
      |               |
      v               v
  [ PR #3 ]         [ PR #4 ]
 (Metrics)         (Stress Test)
      |               |
      +-------+-------+
              |
              v
           [ End ]
```

*   **Track A (Services):** `PR #1` must be merged before `PR #3`.
*   **Track B (Core Stability):** `PR #2` must be merged before `PR #4` can be expected to pass.

---

### **Smoke-Test Readiness Checklist**

This checklist must be completed after all PRs are merged and before the full test suite is run against the integrated system.

- [ ] **Attestor Service:** The `attestor` service is running and the `/health` endpoint returns `200 OK`.
- [ ] **Logging Integrity:** Manually inspect the application logs; confirm they are well-formed and no gaps are present.
- [ ] **Prometheus Target:** The `attestor` scrape target is "UP" in the Prometheus UI.
- [ ] **Metric Values:** Basic metrics (e.g., success rate) are present in Prometheus and have plausible values.
- [ ] **CI Pipeline:** The main CI pipeline is green, including the new `test_concurrency_stress.py` test.
