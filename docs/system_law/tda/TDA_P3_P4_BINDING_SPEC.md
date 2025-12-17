# Binding TDA Metrics into P3 & P4 Validation

This document specifies how Topological Data Analysis (TDA) metrics are bound into the P3 (verification) and P4 (validation) loops to enhance safety and stability monitoring.

## P3 Validation: TDA Metric Computation and Logging

TDA metrics provide a continuous, coordinate-free assessment of the system's state-space topology during simulated trajectory execution.

### Metric Computation Cadence

Metrics are computed at two cadences:

*   **Per-Cycle:** Fast-computation metrics that provide instantaneous health checks.
    *   **SNS (State Neighborhood Similarity):** Measures the consistency of local neighborhood topology against a reference manifold. A high score indicates local stability.
    *   **HSS (Homological Scar Score):** Detects the formation of transient, unstable topological features (scars) in the state space.
*   **Per-Window:** Metrics requiring a larger sample of states to compute, offering a view of stability over an epoch.
    *   **PCS (Path Connectivity Score):** Assesses the connectivity of the trajectory path. A drop indicates potential state-space fragmentation or attractor dissolution.
    *   **DRS (Dimensionality Reduction Stability):** Measures the stability of the intrinsic dimensionality of the active state-space manifold. High variance suggests model confusion or regime change.

### P3 Artifact Logging

TDA metrics will be logged to the primary `p3_results.jsonl` artifact. Each line corresponding to a cycle will be augmented with a `tda_metrics` object.

**JSONL Structure Example:**

```json
{
  "cycle_id": 12345,
  "timestamp": "2025-12-10T14:00:00Z",
  "usla_state": { ... },
  "tda_metrics": {
    "sns": 0.998,
    "hss": 0.012
  },
  // ... other cycle data
}
```

For windowed metrics, a summary entry is logged at the end of each window.

```json
{
  "window_id": "window_789",
  "start_cycle_id": 12000,
  "end_cycle_id": 12999,
  "tda_summary": {
    "pcs": 0.95,
    "drs": 0.05,
    "hss_mean": 0.015
  },
  // ... other window summary data
}
```

## P4 Validation: Real vs. Twin Trajectory Analysis

In P4, TDA is a critical tool for detecting divergence between the real-world system (real) and its digital counterpart (twin).

### Trajectory Analysis Protocol

1.  **Concurrent Computation:** TDA metrics (SNS, PCS, DRS, HSS) are computed for both the real and twin trajectories in real-time.
2.  **Metric Correlation:** A rolling correlation window is used to compare the time series of each metric between the real and twin systems.
3.  **Divergence Scoring:** A divergence score is calculated based on the decorrelation and absolute difference between the metric pairs.

### TDA Red Flags and Severity Mapping

A TDA red flag is triggered when the divergence score exceeds predefined thresholds. These flags are mapped to severity levels to guide automated and manual review.

| Metric Divergence                                        | Threshold | Severity | Implication                                           |
| -------------------------------------------------------- | --------- | -------- | ----------------------------------------------------- |
| **SNS** decorrelation > 0.3                              | > 3 cycles| `LOW`    | Minor local instability; potential model drift.       |
| **HSS** spike in real, not twin (or vice versa)          | > 2 sigma | `MEDIUM` | Unpredicted transient anomaly in one system.          |
| **PCS** sustained divergence > 15%                       | > 1 window| `HIGH`   | Significant state-space pathing mismatch.             |
| **DRS** twin manifold dimensionality collapses           | < 10%     | `CRITICAL`| Twin has lost track of the real system's complexity. |
