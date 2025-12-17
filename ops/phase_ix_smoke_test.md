# Phase IX Smoke-Test Readiness Checklist

**Note on Scope:** This checklist verifies **operational readiness** only. It confirms that the monitoring and attestation services are deployed, running, and communicating as expected. Passing this smoke test is a prerequisite for functional and integration testing, but it makes no assertion about the correctness of the underlying governance model or the validity of the attestations themselves.

**Operational readiness does not imply governance validity.** For authoritative governance truth sources, please refer to `docs/system_law/`.

---

### Pre-Flight Checks

This checklist must be completed after all Phase IX PRs are merged and before the full test suite is run against the integrated system.

- [ ] **Attestor Service:** The `attestor` service is running and its `/health` endpoint returns `200 OK`.
- [ ] **Logging Integrity:** Manually inspect the main application logs; confirm they are well-formed and no time gaps are present.
- [ ] **Prometheus Target:** The `attestor` scrape target is "UP" in the Prometheus UI.
- [ ] **Metric Values:** Basic metrics from the attestor service (e.g., `attestation_success_rate`) are present in Prometheus and have plausible, non-zero values.
- [ ] **CI Pipeline:** The main CI pipeline is green, including the new `test_concurrency_stress.py` test.
- [ ] **Progress Artifact:** The `Phase IX Progress Report` CI job has run successfully and the `phase-ix-progress-report` artifact is available for download, showing all checklist items as DONE.
