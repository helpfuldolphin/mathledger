# Claude Agent Prompts for Phase II Implementation

These prompts are ready to paste into Claude agents for Phase II coding tasks.

---

## Prompt for CLAUDE E + CLAUDE G (Curriculum & Derivation Design)

```
Role: Phase II Uplift Slice Designer (PL / Derivation / Curriculum)

I have completed Phase I: RFL infrastructure + negative control (no uplift). I now want
four Phase II slices where policy must matter: slice_uplift_goal, slice_uplift_sparse,
slice_uplift_tree, slice_uplift_dependency.

Tasks:

1. Review config/curriculum_uplift_phase2.yaml containing the four slice definitions.

2. For each slice, identify specific target formulas that should serve as goals:
   - slice_uplift_goal: Find 2-3 interesting tautologies at depth 4-5 with atoms {p,q,r,s}
   - slice_uplift_tree: Find a theorem that requires a 3-step proof chain
   - slice_uplift_dependency: Find 3 formulas that are independently provable but rarely
     all proved in the same cycle under budget constraints

3. Compute and populate the target_hashes fields with actual SHA256 hashes of the
   normalized target formulas.

4. Ensure slice parameters maintain monotonicity with existing Phase I slices.

5. Add comments marking these slices as Phase II – not used in Phase I.

Slice definitions are in: config/curriculum_uplift_phase2.yaml
```

---

## Prompt for CLAUDE D + CLAUDE C (Runners, MDAP, Orchestration)

```
Role: Phase II Uplift Runner & Orchestrator Engineer

Phase I RFL infrastructure is complete and tested as a negative control (no uplift on
symmetric slices). I now have four Phase II uplift slices in config/curriculum_uplift_phase2.yaml.

Goal: Implement a generalized uplift runner and manifest flow (U2), but do not run
experiments yet — this is design & code only.

Tasks:

1. Create experiments/run_uplift_u2.py:
   - CLI args: --slice-name, --cycles, --seed, --mode {baseline,rfl}, --out
   - For each run:
     - Initialize deterministic seeds from MDAP seed + slice + cycle
     - Load slice config from curriculum_uplift_phase2.yaml
     - Call the derivation pipeline with appropriate mode and policy state
     - Write results/<slice>_<mode>.jsonl
   - Write experiment_manifest.json with:
     - slice config hash
     - prereg link (if any)
     - H_t sequence hash
     - success metric type

2. Implement slice-specific success metrics:
   - goal_hit: Check if any target hash was proved
   - density: Count verified under candidate budget
   - chain_length: Track proof depth per statement
   - multi_goal: Check all required goals are satisfied

3. Ensure RFL mode:
   - Uses verifiable reward signal only (no RLHF/RLPF)
   - Uses policy features and update logic from Phase I
   - Extends features when goal-related features are needed

4. Update MDAP / VSD / PREREG docs to reference U2 as Phase II future work;
   maintain all "NOT YET RUN" banners.

Important: Do not reinterpret Phase I logs as uplift. U2 is a new experiment family,
governed by the existing uplift gate.

Reference files:
- experiments/run_fo_cycles.py (Phase I runner)
- config/curriculum_uplift_phase2.yaml (slice definitions)
- experiments/prereg/PREREG_UPLIFT_U2.yaml (preregistration template)
```

---

## Prompt for CLAUDE H + CLAUDE K + CLAUDE I (Telemetry, Metrics, Security)

```
Role: Phase II Uplift Telemetry & Security Designer

I now have four Phase II uplift slices and will have a new runner run_uplift_u2.py.
I need telemetry and metrics planning, but no production endpoints yet.

Tasks:

1. Telemetry plan
   Extend first_organism_telemetry_plan_v2.md with a new "Phase II Uplift Telemetry
   (Design Only)" section:
   - Define metrics:
     - p_base, p_rfl, Δp per slice, per experiment
     - Optional: verified_per_candidate, goal_hit_rate, chain_success_rate,
       joint_goal_rate matching the four slices
   - Define labels:
     - {slice_name, mode ∈ {baseline,rfl}, seed, experiment_id, metric_name}
   - Mark all of this as:
     - STATUS: NOT IMPLEMENTED
     - Internal observability only, not public product metrics
   - Include a small capability matrix contrasting:
     - Phase I: no uplift metrics
     - Phase II: uplift metrics available but internal-only

2. Analytics stubs
   In backend/metrics/fo_analytics.py:
   - Add a clearly marked comment block (no live code) that sketches:
     - How uplift metrics could be ingested (JSONL → pandas → DB or in-memory)
     - What a future RFLUpliftMetricsReader might look like
   - Make sure:
     - Nothing is imported or called at runtime in Phase I
     - The block is explicitly labeled: # PHASE II UPLIFT METRICS (DESIGN ONLY)

3. Security posture
   In FIRST_ORGANISM_SECURITY_SUMMARY.md:
   - Add a small subsection: "Phase II Uplift Experiments & Security", stating:
     - Phase II uplift runs are hermetic / batch and do not introduce new external
       attack surface by default
     - Any DB-backed uplift metrics must use dedicated RFL DB roles/URLs and
       respect existing hardening (no new privileges)
     - Phase I security guarantees remain unchanged; uplift is opt-in and gated
   - Explicitly note:
     - "No Phase II uplift telemetry is currently wired into production API or dashboards"
     - "All uplift-related configs are disabled by default in Phase I environments"

Guardrail: Do not add any new live endpoints or background jobs. Everything should be
plans, comments, and doc sections only.
```

---

## Prompt for CLAUDE A + CLAUDE B + CLAUDE M + CLAUDE N + CLAUDE O (Governance, Prereg, Narrative)

```
Role: Governance / Preregistration / Narrative for Phase II Uplift

Phase I is sealed as "RFL Infrastructure + Negative Control (no uplift)". Phase II will
introduce real uplift experiments (U2 family) on four new slices.

Tasks:

1. Phase II uplift plan doc
   Review and enhance docs/PHASE2_RFL_UPLIFT_PLAN.md:
   - Verify Phase I status summary is accurate
   - Ensure all four Phase II uplift slices are correctly described
   - Verify the Uplift Evidence Gate criteria are complete:
     - Eligibility: non-degenerate slice, prereg, paired baseline/RFL, deterministic protocol
     - Required outputs: logs, manifest, statistical_summary.json
     - Statistical criteria: Δp, CIs, basic hypothesis test
   - Confirm the explicit banner is present:
     - STATUS: PHASE II — NOT YET RUN. NO UPLIFT CLAIMS MAY BE MADE.

2. Prereg stub for U2
   Review experiments/prereg/PREREG_UPLIFT_U2.yaml:
   - Verify all required fields are present
   - Add any missing governance constraints
   - Ensure comments clearly state "Preregistered – not yet executed"

3. Wire into governance docs
   In VSD_PHASE_2.md, DECOMPOSITION_PHASE_PLAN.md, and WAVE1_PROMOTION_BLUEPRINT.md:
   - Add U2 as a Phase II item in the uplift sections
   - Explicitly state:
     - "No existing logs (fo_rfl_*, fo_baseline_*) satisfy uplift gate"
     - "Any uplift claim must reference a completed U2 experiment that passes the gate"
   - Keep all "NOT YET RUN / DESIGN ONLY" banners intact

4. Narrative hook to emphasize:
   "Industry is moving from RLHF → RLPF → RL with Verifiable Feedback.
   Phase I showed that the RFL infrastructure behaves correctly on a symmetric negative control.
   Phase II U2 will test whether policies can actually lower epistemic risk on non-degenerate
   slices, under a preregistered, statistically sound protocol."
```

---

## Usage

1. Copy the relevant prompt above
2. Paste into the appropriate Claude agent's context
3. Ensure they have access to the referenced files
4. Review their output before committing

