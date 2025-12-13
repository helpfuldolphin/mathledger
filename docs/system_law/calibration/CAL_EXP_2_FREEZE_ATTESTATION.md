# CAL-EXP-2 Freeze Attestation

**Experiment:** CAL-EXP-2 (P4 Divergence Minimization)
**Date:** 2025-12-13
**Status:** FREEZE COMPATIBLE

---

## Frozen Surfaces (NOT TOUCHED)

- `scripts/run_shadow_audit.py` — v0.1 implementation frozen
- CLI contract: `--input`, `--output`, `--seed`, `--verbose`, `--dry-run`
- Exit codes: 0=OK, 1=FATAL, 2=RESERVED
- Schema version: `1.0.0`
- Shadow mode markers: `mode="SHADOW"`, `shadow_mode_compliance.no_enforcement=true`

---

## Allowed Surfaces

- `scripts/usla_first_light_p4_harness.py` — P4 harness (not frozen)
- `scripts/cal_exp_2_*.py` — experiment-specific scripts (new)
- `results/cal_exp_2/*` — experiment output directory (new)
- `experiments/` — experiment tooling (not frozen)

---

## Enforcement

Any modification to frozen surfaces requires a new contract version.

---

*Attested by: CLAUDE S (Freeze Authority)*
