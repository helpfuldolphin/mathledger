# Risk-Informed Strategy in MathLedger: From Metrics to Governance

Our system, MathLedger, employs a "risk-informed" strategy, which means we do not simply hope for good behavior from our models. Instead, we use a set of quantified metrics to allocate trust and make gating decisions. This approach allows us to manage the inherent risks of a complex AI system in a principled and transparent way.

## P3 and P4 Metrics

Our risk-informed strategy is driven by a set of metrics that are calculated at different phases of the system's operation. These metrics are used to evaluate the system's performance and to make decisions about whether to proceed to the next phase.

The P3 metrics are calculated during the pre-training phase, and they measure the model's performance on a set of curated tasks. These metrics include:

*   **Δp**: A measure of the model's performance on a set of curated tasks.
*   **RSI**: A measure of the model's robustness to adversarial attacks.
*   **Ω**: A measure of the model's ability to generalize to new tasks.
*   **TDA**: A measure of the model's ability to detect and reject out-of-distribution inputs.

The P4 metrics are calculated during the post-training phase, and they measure the model's performance on a set of real-world tasks. These metrics include:

*   **Divergence**: A measure of the model's performance on a set of real-world tasks.
*   **TDA**: A measure of the model's ability to detect and reject out-of-distribution inputs.
*   **GovernanceSignals**: A set of metrics that are used to monitor the model's compliance with a set of predefined policies.

## Metric-Risk-Gate Mappings

The following table lists the metrics that are used in our risk-informed strategy, their risk interpretation, and the gate that they influence.

| Metric | Risk interpretation | Gate |
| --- | --- | --- |
| Δp | The model's performance on a set of curated tasks | P3 |
| RSI | The model's robustness to adversarial attacks | P3 |
| Ω | The model's ability to generalize to new tasks | P3 |
| TDA | The model's ability to detect and reject out-of-distribution inputs | P3, P4 |
| Divergence | The model's performance on a set of real-world tasks | P4 |
| GovernanceSignals | The model's compliance with a set of predefined policies | Deployment, defense compliance |

## Architect's Note

Our risk-informed strategy is not "policy on top of black-box ML". Instead, it is a form of quantitative control theory that is applied to cognition. This approach allows us to manage the inherent risks of a complex AI system in a principled and transparent way.
