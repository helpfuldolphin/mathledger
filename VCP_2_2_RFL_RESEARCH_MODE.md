# VCP 2.2 – RFL Research Mode: Rules of Engagement

**Effective Date:** 2025-11-27  
**Applicability:** All Specialized Agents (Devin A-Z, Claude, Cursor, Manus, et al.)  
**Objective:** Prevent drift, architectural churn, and unbounded scope creep during the RFL experiment campaign.

---

## 1. Rules of Engagement

### 🚫 Prohibited Actions
*   **No Core Refactors:** Do not modify `backend/axiom_engine/`, `backend/logic/`, or `ledger/` core consensus logic without explicit authorization.
*   **No Silent Patches:** Do not apply "quick fixes" to RFL logic to make a test pass.
*   **No New Dependencies:** Do not add library dependencies (Python/Node) without a formal request.

### ✅ Authorized Actions
*   **Read-Only Analysis:** Full access to read all codebase files.
*   **Artifact Generation:** Freely create reports, logs, and data in `reports/`, `logs/`, `artifacts/`, and `ops/microtasks/`.
*   **Non-Invasive Instrumentation:** Temporary logging hooks are permitted if they are strictly scoped and removed/reverted after the task.

### 📝 Proposing Core Changes
If a core change is strictly necessary:
1.  **Do NOT** edit the code directly.
2.  **Create a Design Note (DN):** Write a markdown file in `docs/proposals/` (e.g., `DN_001_Fix_Derivation_Bug.md`).
3.  **Content:** Explain the *Why*, *What*, and *Risk*.
4.  **Wait:** Await approval from Manus D or the Lead Architect before implementation.

---

## 2. Experiment Campaign Dependency Map

The fleet operates in a dependency cascade. Do not start downstream tasks until upstream prerequisites are met.

### Tier 1: Foundation & Environment (Prerequisite for ALL)
*   **Agents:** Devin G (CI/Net), Devin B (Perf/Determinism), Devin F (Security).
*   **Deliverables:** Stable CI environment, deterministic baseline, security clearance.
*   **Signal:** `NO_NETWORK` active, `sanity.ps1` PASS.

### Tier 2: Execution & Generation (Depends on Tier 1)
*   **Agents:** Claude A (Generalist), Devin A (Pipeline Opt), FO Runner Agents.
*   **Deliverables:** Proof generation traces, optimization metrics, raw experiment data.
*   **Signal:** `artifacts/wpv5/` populated, `run_metrics_v1.jsonl` updated.

### Tier 3: Verification & Auditing (Depends on Tier 2)
*   **Agents:** Devin C (QA), Cursor C (Guardrails), Integrity Sentinels.
*   **Deliverables:** Pass/Fail verdicts, drift reports, regression alerts, **Abstention Analysis**.
*   **Signal:** Audit reports in `reports/`, validated Merkle roots.

### Tier 4: Synthesis & Coordination (Depends on Tier 3)
*   **Agents:** Manus D (Coordination), Devin I (Docs).
*   **Deliverables:** Final status reports, **Figure Catalog**, release candidates.
*   **Signal:** `MANUS_D_FINAL_REPORT.md`, `SPRINT_STATUS.md` updated.

### 🔗 Workflow Specifics
The following critical path edges are locked for this sprint:
*   **FO Runner (Tier 2) $\to$ Abstention Analysis (Tier 3):** Raw derivation traces must be complete before analyzing abstention rates. No speculative analysis allowed.
*   **Capability Metrics (Tier 2) $\to$ Figure Catalog (Tier 4):** All performance and capability data must be finalized and signed off by Tier 3 Audit before being visualized in the Figure Catalog.

---

## 3. Status Reporting Format

Upon task completion, every agent must append a structured summary to their assigned log or the central `progress.md`.

**Template:**

```markdown
## [AGENT_ID] Task Completion: {TASK_ID}
**Status:** {SUCCESS | FAILURE | BLOCKED}
**Timestamp:** {YYYY-MM-DD HH:MM UTC}

### 📂 Files Touched
- `path/to/modified_file.py`
- `path/to/new_artifact.json`

### 📦 Artifacts Produced
- **Type:** {Report | Patch | Metric | Log}
- **Location:** `artifacts/{category}/{filename}`

### ⚠️ Anomalies / Questions
- {Brief description of any unexpected behavior or blocking questions}

### 📝 Summary
{One concise sentence describing what was actually achieved.}
```