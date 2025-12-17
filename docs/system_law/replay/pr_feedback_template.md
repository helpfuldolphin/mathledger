### Replay Governance Gate: Changes Blocked

The automated replay governance gate has detected issues that prevent this PR from being merged.

- **Radar Status**: `UNSTABLE`
- **Determinism Rate**: `{{ determinism_rate }}`%
- **Top Reasons for Block**:
    1. `{{ promotion_eval.reasons[0] }}`
    2. `{{ promotion_eval.reasons[1] }}`
    3. `{{ promotion_eval.reasons[2] }}`

#### Next Steps

- **Review the full Replay Governance Snapshot**: The detailed `replay_governance_snapshot.json` artifact from this run contains a breakdown of all component-level metrics. This is the best place to identify the root cause.
- **Consult the Debugging Guide**: See `docs/replay_debugging_guide.md` for common causes of non-determinism and drift.
- **Request Assistance**: If you are unable to resolve the issue, please ping the `@mathledger/core-infra` team in this PR for assistance.
