# VERIFIER_RUNTIME_BEHAVIOR_SPEC.md

## 1. Introduction

This document specifies the behavior of the Verifier Runtime, a critical component of the Gemini F system responsible for enforcing worker budget and behavior contracts. It introduces a new runtime model, "job-as-cycle," and defines failure semantics, abstention handling, and interaction with the global worker queue.

## 2. Runtime Model: Job-as-Cycle

The "job-as-cycle" model provides a discrete unit of accounting for worker execution.

*   **Definition of a Cycle:** A cycle is a fixed-size unit of computational effort. It is an abstraction over raw CPU time and memory usage, calibrated against a benchmark set of operations.
*   **Job-to-Cycle Mapping:** Each job submitted to the worker queue is assigned a budget in cycles. This budget is determined by the job's type, complexity, and historical performance data.
*   **Cycle Lifecycle:**
    1.  **Allocation:** The Verifier allocates the cycle budget to a worker upon job assignment.
    2.  **Execution & Decrement:** The worker executes the job, and the Verifier decrements the cycle budget as resources are consumed.
    3.  **Termination:** The job terminates when it completes successfully, fails, or exhausts its cycle budget.

## 3. Failure Semantics for Timeout Enforcement

Timeout enforcement is the primary mechanism for budget control.

*   **Timeout Mechanism:** A job is considered "timed out" when its cycle budget reaches zero. The Verifier actively monitors the cycle consumption of each worker.
*   **Timeout Action:** Upon timeout, the Verifier will:
    1.  Send a non-ignorable termination signal (e.g., `SIGKILL`) to the worker process.
    2.  Log the timeout event, including the job ID, worker ID, and the state of the worker at the time of termination.
    3.  Report the job as failed with a "timeout" reason to the global worker queue.
*   **Error Handling:** The system will not attempt to gracefully shut down the worker, as this could be exploited. State recovery will be handled by a separate reconciliation process.

## 4. Abstention Taxonomy for Candidate Overflow

"Candidate overflow" occurs when a worker is presented with more potential work items (candidates) than it can process within its operational constraints.

*   **Taxonomy of Abstention:**
    *   **Resource Exhaustion:** The worker predicts that processing a candidate would exceed its cycle budget or other resource limits.
    *   **Input Malformation:** The candidate data is malformed or does not conform to the expected schema.
    *   **Capacity Saturation:** The worker's internal buffers or queues are full.
    *   **Capability Mismatch:** The worker does not possess the necessary capabilities (e.g., specific libraries or models) to process the candidate.
*   **Handling Abstention:** The worker must report the abstention to the Verifier with a specific reason from the taxonomy. The Verifier will then requeue the candidate and may adjust its scheduling strategy based on the abstention reason.

## 5. Interaction with the Global Worker Queue

*   **Job Fetching:** The Verifier fetches jobs from the global worker queue via a secure, authenticated endpoint.
*   **Status Reporting:** The Verifier reports the final status of each job (e.g., `SUCCESS`, `FAILURE`, `ABSTENTION`) to the queue. For `FAILURE` and `ABSTENTION`, a detailed reason is provided.
*   **Backpressure:** If the Verifier's internal capacity is reached, it will temporarily stop fetching new jobs from the queue to prevent overload.
