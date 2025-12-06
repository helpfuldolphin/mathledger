# RFL Experiment Risks & Pathology Checklist

**Auditor:** GEMINI-N
**Context:** Risk mitigation for empirical RFL evaluation within the hermetic PL regime.

## 1. Failure Modes Checklist
*Risks that may lead to false positives or misinterpretation of convergence.*

- [ ] **Slice Difficulty Drift:** Apparent loss descent caused merely by the curriculum shifting to easier examples, not actual capability gain.
- [ ] **Nondeterminism Pollution:** Metrics fluctuating due to environment stochasticity rather than policy changes.
- [ ] **Small-N Illusions:** Strong conclusions drawn from insufficient sample sizes (e.g., < 30 runs or narrow prompt sets).
- [ ] **Seed Overfitting:** Performance gains that vanish when the random seed is changed.
- [ ] **Proxy Gaming:** The model optimizes the reward model/metric without improving the underlying task (Goodhart's Law).
- [ ] **Data Leakage:** Evaluation data inadvertently present in the feedback/training loop.
- [ ] **Survivorship Bias:** Aggregating metrics only from successful runs or active agents, ignoring crashes or stalls.
- [ ] **Horizon Effects:** Improvements that look good in the short term but degrade long-term stability or coherence.
- [ ] **Feedback Loop Echo chambers:** The model reinforcing its own biases if self-generated data is used without sufficient external grounding.
- [ ] **Wide Slice Difficulty Drift:** Are we changing slices mid-experiment? A curriculum shift from harder to easier slices can create false convergence signals that look like capability gain but are actually just easier problems.
- [ ] **Trivial Attractor:** Is abstention flat because the slice is too easy? If the Wide Slice (`slice_medium`) is insufficiently challenging, abstention rates may remain constant at zero, providing no signal for RFL metabolism to act upon.
- [ ] **RFL Overfitting to Slice-Specific Quirks:** The RFL policy may learn patterns that are specific to the Wide Slice's particular statement distribution, depth constraints, or verification characteristics, rather than generalizable proof-generation improvements.

## 2. Sanity Checks
*Procedures to validate that observed signals are real.*

- [ ] **Random Policy Baseline:** Run a completely random or heuristic policy. If the complex model doesn't significantly beat this, the signal is noise.
- [ ] **Log Shuffling:** Shuffle the temporal order of training logs and re-evaluate. If "learning curves" persist, the metric is flawed.
- [ ] **Reverse-Time Baselines:** Analyze the trajectory backwards. Does "unlearning" look distinct from learning?
- [ ] **No-Op Training:** Run the training loop with learning rate set to zero. Ensure metrics remain flat (validates the measurement pipeline).
- [ ] **Seed Robustness:** Re-run the best configuration with at least 3 different random seeds.
- [ ] **Ablation Tests:** Remove the core novel component. Performance *must* drop; otherwise, the component is redundant.
- [ ] **Dyno Chart Sample Size:** Do not overinterpret small differences over <200 cycles. Early-cycle variance is high; wait for statistical stability before claiming structural uplift.
- [ ] **Multi-Seed Validation:** Always compare multiple seeds/runs before claiming structural uplift. A single Dyno Chart trajectory may reflect seed-specific luck rather than genuine policy improvement.

## 3. Guardrails for Analysis Agents (D, E, K, H)
*Protocol for agents interpreting experimental data.*

-   **Quantify Uncertainty:** Never report a point estimate (mean) without variance (std dev) or confidence intervals.
-   **Report the Null:** Explicitly state when results are statistically indistinguishable from the baseline.
-   **Full Distribution Reporting:** Do not cherry-pick "best runs." Analyze and report the entire distribution of outcomes, including failures.
-   **Data Integrity First:** Verify log file completeness and format consistency before beginning analysis.
-   **Skepticism of Perfection:** Treat "perfect" or monotonic convergence with extreme suspicion; investigate for bugs in the logging or metric calculation.
-   **Contextualize Improvements:** Absolute numbers mean less than relative gain over a strong baseline. Always frame results relatively.
-   **Check for Regression:** Explicitly look for areas where performance *degraded*, not just where it improved.

## 4. Data Integrity Checks for Dyno Chart Analysis
*Before trusting a Dyno Chart, verify that the underlying JSONL logs are valid and consistent.*

### Pre-Analysis Verification

Before interpreting any Dyno Chart visualization or claiming uplift from cycle-to-cycle trends, perform these integrity checks:

- [ ] **Monotone Cycle Index:** Verify that `cycle` (or `cycle_index`) fields are strictly monotonically increasing with no gaps or reversals. Each cycle should be `cycle[i+1] = cycle[i] + 1` for sequential runs.
- [ ] **No Missing Cycles:** Check that the cycle sequence is continuous. If cycles 0-999 are expected, ensure all 1000 entries exist. Missing cycles indicate crashes, restarts, or logging failures that invalidate temporal analysis.
- [ ] **No Duplicate Cycles:** Ensure each cycle index appears exactly once. Duplicate cycles suggest re-runs, log corruption, or merge errors that will skew aggregation.
- [ ] **Stable Configuration:** Verify that the experiment configuration (slice name, system, mode, derive_steps, max_breadth, etc.) remains constant across all cycles. Configuration drift mid-experiment invalidates comparative analysis.
- [ ] **Canonical JSON Format:** Validate that all JSONL entries are well-formed, use canonical key ordering (`sort_keys=True`), and contain required fields (`cycle`, `status`, `abstention`, `mode`, `slice_name`).
- [ ] **Root Consistency:** For experiments with Merkle roots (`h_t`, `r_t`, `u_t`), verify that root values are valid SHA-256 hashes (64 hex characters) and that they change appropriately with cycle content (not frozen or corrupted).

### Automated Validation Script

Consider implementing a pre-analysis validator that:
1. Parses the JSONL file line-by-line
2. Extracts and validates cycle indices
3. Checks for monotonicity, completeness, and uniqueness
4. Verifies configuration stability
5. Reports any integrity violations before analysis proceeds

**Failure Mode:** If any integrity check fails, **do not proceed to Dyno Chart interpretation**. Fix the data collection or re-run the experiment.

## 5. First Organism Specific Risks
*Infrastructure and concurrency risks that can silently corrupt FO test execution or ledger state.*

### First Organism Specific Risks

- **DB migrations incomplete → FO test silently skipped:** If database migrations are not fully applied before running First Organism tests, the test may skip gracefully (as designed for hermetic operation) but produce no useful output. This creates a false sense of test execution when in fact the critical path was never exercised. Always verify migration status before FO runs.

- **DB/Redis partitions during FO run → partial ledger writes / weird H_t:** If Postgres or Redis becomes unavailable mid-execution (network partition, container restart, resource exhaustion), the FO test may complete partially written ledger entries or generate inconsistent Merkle roots (`H_t`, `R_t`, `U_t`). The dual-attestation seal may appear valid but reflect corrupted state, leading to downstream RFL metabolism operating on invalid data.

- **Concurrent FO runs against same DB:** Running multiple First Organism tests in parallel against the same database instance can cause race conditions in ledger writes, block creation, or attestation sealing. This may result in non-deterministic `H_t` values, duplicate cycle indices, or conflicting block headers that break the temporal integrity assumptions required for RFL analysis.

## 6. Future Chaos Testing TODO
*Chaos harness tests to stress-test First Organism resilience once SPARK is boringly green.*

### TODO: TestFirstOrganismChaos

When SPARK tests are consistently passing and the FO path is stable, implement a comprehensive chaos test suite `TestFirstOrganismChaos` that validates FO behavior under infrastructure failures:

- **Postgres mid-run termination:** Kill the Postgres container/process during an active FO test run. Verify that:
  - The test fails gracefully with clear error reporting (no silent skips)
  - Partial ledger writes are rolled back (transaction integrity)
  - No corrupted `H_t` values are emitted
  - The test can be re-run cleanly after Postgres recovery

- **Redis drop during attestation:** Terminate Redis connectivity during the dual-attestation sealing phase. Verify that:
  - Attestation state is not lost
  - The test either retries or fails with clear diagnostics
  - No partial attestation artifacts are written to disk
  - Recovery path exists for resuming attestation after Redis restoration

- **Parallel FO execution:** Run two FO tests simultaneously against the same database. Verify that:
  - Cycle indices remain unique and monotonic
  - Block headers do not conflict
  - `H_t` values are deterministic per test run (not corrupted by interleaving)
  - Database locks prevent ledger corruption
  - Both tests complete with valid, independent attestation seals

**Implementation Note:** These chaos tests should be gated behind a `CHAOS_HARNESS=true` environment variable and marked with `@pytest.mark.chaos` to keep them separate from standard SPARK runs. They are intended for periodic validation, not continuous integration.
