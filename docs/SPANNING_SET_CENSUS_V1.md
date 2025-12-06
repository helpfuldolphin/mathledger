# Spanning Set Census V1
**Generated:** 2025-11-25T23:37:11.035314

## 1. Topline Statistics
| Classification | Files | LOC | Size (Bytes) |
|---|---|---|---|
| **CORE** | 189 | 24,697 | 946,946 |
| **SUPPORTING** | 10775 | 3,241,833 | 352,160,972 |
| **EXPERIMENTAL** | 93 | 8,751 | 1,334,096 |
| **ARCHIVE-CANDIDATE** | 57 | 11,085 | 418,873 |
| **TOTAL** | 11114 | 3,286,366 | - |

## 2. Classification Map
### Core (Minimal Basis Candidate)
> **Definition:** Must survive into `basis/` (e.g., canonical hashing, dual_root, RFL core).

| Module/Path | LOC | Justification |
|---|---|---|
| `attestation/` | 348 | Whitepaper: Dual Attestation Root |
| `backend/` | 15,919 | Backend core module: axiom_engine (Logic/Ledger/Mechanism) |
| `basis/` | 833 | Target destination for minimal basis |
| `curriculum/` | 1,038 | Whitepaper: Curriculum/Learning Schedule |
| `derivation/` | 856 | Whitepaper: Derivation Engine |
| `ledger/` | 1,102 | Ledger components (if distinct from backend) |
| `normalization/` | 625 | Logic normalization components |
| `rfl/` | 2,482 | Reinforced Feedback Loop (Root) |
| `substrate/` | 1,494 | Whitepaper: Substrate (Lean/Formal Verification) |

### Supporting (Infrastructure)
> **Definition:** Ops, UI, tests, and config that support the organism but are not the organism itself.

| Area | Est. LOC | Notes |
|---|---|---|
| `.claude` | 17 | Unclassified, defaulted to supporting |
| `.coverage` | 0 | Unclassified, defaulted to supporting |
| `.coveragerc` | 0 | Unclassified, defaulted to supporting |
| `.editorconfig` | 0 | Unclassified, defaulted to supporting |
| `.env.first_organism` | 0 | Unclassified, defaulted to supporting |
| `.gitattributes` | 0 | Unclassified, defaulted to supporting |
| `.github` | 3,838 | Unclassified, defaulted to supporting |
| `.gitignore` | 0 | Unclassified, defaulted to supporting |
| `.grok` | 3 | Unclassified, defaulted to supporting |
| `.pre-commit-config.yaml` | 45 | Unclassified, defaulted to supporting |
| `.quarantine` | 5,757 | Unclassified, defaulted to supporting |
| `.venv` | 3,004,524 | Unclassified, defaulted to supporting |
| `AGENTS.md` | 26 | Agent context |
| `CLAUDE.md` | 319 | Agent context |
| `Makefile` | 0 | Project configuration and build tools |
| `README.md` | 134 | Documentation |
| `README_HARMONY_V1_1.md` | 358 | Documentation |
| `README_ops.md` | 510 | Documentation |
| `apps` | 11,151 | Frontend/UI |
| `backend` | 12,851 | Backend module: __init__.py |
| `basis_promotion_candidates.json` | 224 | Unclassified, defaulted to supporting |
| `ci_verification` | 43 | Unclassified, defaulted to supporting |
| `cli` | 403 | Operational Infrastructure |
| `config` | 522 | Operational Infrastructure |
| `docker-compose.yml` | 65 | Project configuration and build tools |
| `docs` | 30,352 | Project Documentation |
| `infra` | 50 | Operational Infrastructure |
| `interface` | 1,354 | Unclassified, defaulted to supporting |
| `interop_results_2025_11_04.json` | 209 | Unclassified, defaulted to supporting |
| `mathledger_basis_repo` | 2,246 | Unclassified, defaulted to supporting |
| `metrics` | 12 | Metrics output/config |
| `migrations` | 1,784 | Operational Infrastructure |
| `monitor.py` | 85 | Root operational script |
| `nul` | 0 | Unclassified, defaulted to supporting |
| `ops` | 95,525 | Unclassified, defaulted to supporting |
| `perf_sanity.json` | 150 | Unclassified, defaulted to supporting |
| `perf_sanity_import.json` | 33 | Unclassified, defaulted to supporting |
| `performance_passport.json` | 289 | Unclassified, defaulted to supporting |
| `pyproject.toml` | 0 | Project configuration and build tools |
| `pytest.ini` | 0 | Project configuration and build tools |
| `reports` | 152 | Unclassified, defaulted to supporting |
| `schema_actual.json` | 5 | Unclassified, defaulted to supporting |
| `scripts` | 15,401 | Operational Infrastructure |
| `services` | 636 | Unclassified, defaulted to supporting |
| `spanning_set_manifest.json` | 6,634 | Unclassified, defaulted to supporting |
| `tapi import FastAPI, HTTPException` | 0 | Unclassified, defaulted to supporting |
| `tatus` | 0 | Unclassified, defaulted to supporting |
| `templates` | 372 | Unclassified, defaulted to supporting |
| `tests` | 21,320 | Test Suite |
| `tools` | 24,192 | Operational Infrastructure |
| `ui` | 242 | Frontend/UI |
| `uv.lock` | 0 | Project configuration and build tools |
| ` grep postgres` | 0 | Unclassified, defaulted to supporting |

### Experimental (Quarantine Zone)
> **Definition:** Proto-scripts, scratchpads, and patch files. High risk of entropy.

- `CI_FIX_PATCH.diff` (0 lines): Patch file
- `artifacts\continuity\lineage_delta.json` (160 lines): Ephemeral or output directory
- `artifacts\guidance\train.csv` (0 lines): Ephemeral or output directory
- `artifacts\guidance\val.csv` (0 lines): Ephemeral or output directory
- `artifacts\keys\rotation_log.jsonl` (0 lines): Ephemeral or output directory
- `artifacts\metrics\history.json` (33 lines): Ephemeral or output directory
- `artifacts\metrics\latest.json` (135 lines): Ephemeral or output directory
- `artifacts\metrics\latest_report.txt` (0 lines): Ephemeral or output directory
- `artifacts\metrics\schema_v1.json` (242 lines): Ephemeral or output directory
- `artifacts\metrics\session_metrics-cartographer-6ec1fb1ec537.json` (135 lines): Ephemeral or output directory
- `artifacts\metrics\trends.json` (32 lines): Ephemeral or output directory
- `artifacts\perf\baseline.csv` (0 lines): Ephemeral or output directory
- `artifacts\policy\policy.json` (7 lines): Ephemeral or output directory
- `artifacts\repro\attestation_history\attestation_20251101_035113.json` (32 lines): Ephemeral or output directory
- `artifacts\repro\autofix_manifest.json` (244 lines): Ephemeral or output directory
- `artifacts\repro\determinism_attestation.json` (32 lines): Ephemeral or output directory
- `artifacts\repro\determinism_report.json` (385 lines): Ephemeral or output directory
- `artifacts\repro\drift_report.json` (24 lines): Ephemeral or output directory
- `artifacts\repro\drift_whitelist.json` (79 lines): Ephemeral or output directory
- `artifacts\repro\entropy_free_process.md` (437 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\README.md` (155 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_benchmark.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_bridge_v2.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_generate_report.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_latency_profiler.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_metrics.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_integration_middleware.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_orchestrator_app.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_testing_hermetic.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_testing_hermetic_v2.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_testing_no_network.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\backend_worker.py.patch` (0 lines): Ephemeral or output directory
- `artifacts\repro\manual_review\owner_map.json` (68 lines): Ephemeral or output directory
- `artifacts\rfl\hypotheses.json` (267 lines): Ephemeral or output directory
- `artifacts\rfl\results.json` (277 lines): Ephemeral or output directory
- `artifacts\rfl\validation_summary.md` (216 lines): Ephemeral or output directory
- `artifacts\ship\gate_smoke_pr_body.md` (260 lines): Ephemeral or output directory
- `artifacts\wpv5\EVIDENCE.md` (71 lines): Ephemeral or output directory
- `artifacts\wpv5\Email_Kalai.md` (22 lines): Ephemeral or output directory
- `artifacts\wpv5\README_EVAL.md` (15 lines): Ephemeral or output directory
- `artifacts\wpv5\ablation_rows.tex` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\fol_ab.csv` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\fol_stats.json` (16 lines): Ephemeral or output directory
- `artifacts\wpv5\throughput.json` (18 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.aux` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.fdb_latexmk` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.fls` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.out` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.pdf` (0 lines): Ephemeral or output directory
- `artifacts\wpv5\whitepaper.tex` (0 lines): Ephemeral or output directory
- `backend\phase_ix\__init__.py` (19 lines): Phase IX specific work
- `backend\phase_ix\attestation.py` (143 lines): Phase IX specific work
- `backend\phase_ix\dossier.py` (156 lines): Phase IX specific work
- `backend\phase_ix\harness.py` (369 lines): Phase IX specific work
- `bootstrap_metabolism.py` (369 lines): Root script, likely ad-hoc
- `bootstrap_output\bootstrap_curves.png` (0 lines): Ephemeral or output directory
- `bootstrap_output\coverage_results.json` (551 lines): Ephemeral or output directory
- `bridge.py` (98 lines): Root script, likely ad-hoc
- `canon_patch.diff` (0 lines): Patch file
- `derive_docs_diff.patch` (0 lines): Patch file
- `derive_patch.diff` (0 lines): Patch file
- `exports\basis_20251126_032101_86a2112e.tar.gz` (0 lines): Ephemeral or output directory
- `fix_test.py` (20 lines): Root script, likely ad-hoc
- `get_schema.py` (73 lines): Root script, likely ad-hoc
- `ledgerctl.py` (309 lines): Root script, likely ad-hoc
- `logs\nightly-20251125-231828\nightly-run-summary-20251125-231840.json` (18 lines): Ephemeral or output directory
- `logs\sanity-20250908-000836.log` (0 lines): Ephemeral or output directory
- `phase_ix_attestation.py` (308 lines): Root script, likely ad-hoc
- `rfl_gate.py` (575 lines): Root script, likely ad-hoc
- `run-nightly.ps1` (68 lines): Root script, likely ad-hoc
- `run_all_migrations.py` (54 lines): Root script, likely ad-hoc
- `run_migration.py` (34 lines): Root script, likely ad-hoc
- `run_migration_simple.py` (96 lines): Root script, likely ad-hoc
- `run_tests.py` (51 lines): Root script, likely ad-hoc
- `sanity.ps1` (32 lines): Root script, likely ad-hoc
- `start_api_server.py` (60 lines): Root script, likely ad-hoc
- `test_dual_attestation.py` (34 lines): Root script, likely ad-hoc
- `test_integration_v05.py` (376 lines): Root script, likely ad-hoc
- `test_integrity_audit.py` (397 lines): Root script, likely ad-hoc
- `test_migration_validation.py` (188 lines): Root script, likely ad-hoc
- `test_runner.py` (52 lines): Root script, likely ad-hoc
- `test_v05_integration.py` (415 lines): Root script, likely ad-hoc
- `tmp\claude.patch` (0 lines): Ephemeral or output directory
- `tmp\claude_clean.patch` (0 lines): Ephemeral or output directory
- `tmp\gemini_derive.patch` (0 lines): Ephemeral or output directory
- `tmp\job.json` (1 lines): Ephemeral or output directory
- `tmp\job1.json` (1 lines): Ephemeral or output directory
- `tmp\job2.json` (1 lines): Ephemeral or output directory
- `unified_diff.patch` (0 lines): Patch file
- `unified_patch.diff` (0 lines): Patch file
- `verify_dual_root.py` (312 lines): Root script, likely ad-hoc
- `verify_local_schema.py` (189 lines): Root script, likely ad-hoc
- `verify_slice.py` (20 lines): Root script, likely ad-hoc

### Archive Candidates
> **Definition:** Obsolete sludge to be moved to `archive/` or deleted.

- `ALLBLUE_GATE_TRIGGER_V2.md`: Loose root report/doc
- `API_README.md`: Loose root report/doc
- `CI_FIX_INSTRUCTIONS.md`: Loose root report/doc
- `EFFICIENCY_REPORT.md`: Loose root report/doc
- `ENHANCED_API_README.md`: Loose root report/doc
- `HANDOFF_RFL_GATE.md`: Loose root report/doc
- `INTEGRITY_SENTINEL_AUDIT_REPORT.md`: Loose root report/doc
- `LAWKEEPER_HANDOFF_MIRROR_AUDITOR.md`: Loose root report/doc
- `MANUS_D_COORDINATION.md`: Loose root report/doc
- `MANUS_D_FINAL_REPORT.md`: Loose root report/doc
- `MIGRATIONS.md`: Loose root report/doc
- `MIRROR_AUDITOR_HANDOFF.md`: Loose root report/doc
- `MIRROR_AUDITOR_IMPLEMENTATION.md`: Loose root report/doc
- `MIRROR_AUDITOR_OPERATIONAL_REPORT.txt`: Loose root report/doc
- `PERFORMANCE_CARTOGRAPHER_REPORT.md`: Loose root report/doc
- `PERFORMANCE_MONITORING_README.md`: Loose root report/doc
- `PHASE_A_SUMMARY.md`: Loose root report/doc
- `PHASE_B_SUMMARY.md`: Loose root report/doc
- `PHASE_C_CI_AUDIT.md`: Loose root report/doc
- `PHASE_D_MONITORING.md`: Loose root report/doc
- `PHASE_III_COMPLETE.md`: Loose root report/doc
- `PHASE_IX_FILES.md`: Loose root report/doc
- `PHASE_IX_STATUS.txt`: Loose root report/doc
- `PHASE_IX_SUMMARY.md`: Loose root report/doc
- `PR_BODY.md`: Loose root report/doc
- `PR_BODY.txt`: Loose root report/doc
- `PR_BODY_FLEET_READINESS.md`: Loose root report/doc
- `PR_BODY_PHASE_X_FLEET_READINESS.md`: Loose root report/doc
- `PR_DESCRIPTION.md`: Loose root report/doc
- `PR_MIGRATION_REPAIR.md`: Loose root report/doc
- `SPRINT_STATUS.md`: Loose root report/doc
- `STRATEGIC_PR_ANNOUNCEMENT.md`: Loose root report/doc
- `TEST_EXPORT.md`: Loose root report/doc
- `UI_IMPLEMENTATION_README.md`: Loose root report/doc
- `V03_UI_IMPLEMENTATION_README.md`: Loose root report/doc
- `V04_UI_IMPLEMENTATION_README.md`: Loose root report/doc
- `V05_UI_IMPLEMENTATION_README.md`: Loose root report/doc
- `VERSION_LINEAGE_LEDGER.md`: Loose root report/doc
- `_fail_db.txt`: Loose root report/doc
- `_fail_full.txt`: Loose root report/doc
- `_fail_unit.txt`: Loose root report/doc
- `allblue_archive\fleet_state.json`: Explicit archive directory
- `allblue_archive\fleet_state_readable.json`: Explicit archive directory
- `archive\README.md`: Explicit archive directory
- `backend\axiom_engine\derive.py.bak`: Backup file
- `backend\logic\canon.py.bak`: Backup file
- `backend\orchestrator\app.py.bak`: Backup file
- `basis_promotion_report.md`: Loose root report/doc
- `config\curriculum.yaml.bak`: Backup file
- `drift_table.md`: Loose root report/doc
- `drift_table_summary.md`: Loose root report/doc
- `governance_verdict.md`: Loose root report/doc
- `mirror_auditor_summary.md`: Loose root report/doc
- `progress.md`: Loose root report/doc
- `tests\integration\conftest.py.bak`: Backup file
- `tests\qa\test_exporter_v1.py.bak`: Backup file
- `tests\test_mp_derivation.py.bak`: Backup file

## 3. Basis Nucleus Proposal
The following modules constitute the minimal viable organism:
1. **`basis/`**: The existing formalized core.
2. **`backend/crypto`**: Canonical hashing and signatures.
3. **`backend/ledger`**: Immutable ledger structures.
4. **`backend/logic`**: Normalization and canonicalization.
5. **`backend/rfl`**: The Reinforced Feedback Loop logic.
6. **`attestation/`**: Dual root attestation (Law).
7. **`substrate/`**: The Lean/formal verification substrate.
8. **`curriculum/`**: The learning schedule.

**Action:** These should be consolidated into the `basis/` namespace to enforce the separation between 'The Organism' and 'The Lab'.