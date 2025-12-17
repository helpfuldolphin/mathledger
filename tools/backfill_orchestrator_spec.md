# Backfill Orchestrator Spec (Design Stub)

## State Machine
- `GAP_DETECTED` → `QUEUED` → `RUNNING` → `PACK_REBUILT`
- Transitions:
  - `GAP_DETECTED` when evidence gap report finds missing artifacts.
  - `QUEUED` when a backfill job plan is materialized (P3/P4/plots/proof).
  - `RUNNING` while generators execute.
  - `PACK_REBUILT` when all required artifacts are regenerated and validated.

## Missing → Action Mapping
- P3 artifacts (`p3_synthetic/*`): run P3 harness (synthetic).
- P4 artifacts (`p4_shadow/*`): run P4 harness (shadow).
- Visualization artifacts (`visualizations/*`): run plot renderer.
- Proof snapshot (`proof_log_snapshot.json` if adopted): run proof hash snapshotper.
- Compliance docs remain manual (flag only).

## CI / Automation Plan (Non-Gating)
- Nightly job only (non-gating) runs gap reporter across `results/first_light/evidence_pack_first_light/`.
- Emits plan JSON for manual review; does not auto-run backfill.
- Alerts on missing artifacts but does not fail PRs/CI.

## CLI Sketch (future)
- `python -m tools.evidence_gap_report --root <pack> --status-json gap_status.json --plan-output gap_plan.json`
  - `--plan-output` would emit a job plan JSON derived from missing artifacts:
    - `actions: ["run_p3_harness", "run_p4_harness", "render_plots", "write_proof_snapshot"]`
    - `state: "QUEUED"`
    - `missing: [...]`
  - No execution; orchestration layer consumes the plan to schedule runs.
