# SHADOW Gate Ratchet Policy

**Version**: 1.0.0
**Status**: ACTIVE
**Effective Date**: 2025-12-17
**Authority**: SMGA (Shadow Mode Governance Authority)
**Contract Reference**: `SHADOW_MODE_CONTRACT.md` v1.0.0

---

## 1. Purpose

This document defines the **forward-only ratchet policy** for reducing legacy SHADOW MODE warnings. The policy ensures that:

1. Legacy documentation is not disrupted during migration
2. New documentation uses explicit SHADOW-OBSERVE or SHADOW-GATED qualifiers
3. The warning budget can only **decrease** over time (ratchet behavior)
4. Full enforcement is achieved through a phased approach

---

## 2. Background

Per `SHADOW_MODE_CONTRACT.md` §1.1:

> "Any reference to 'SHADOW MODE' without explicit sub-mode qualification SHALL be interpreted as SHADOW-OBSERVE."

This means unqualified "SHADOW MODE" usage in legacy documents is **technically valid** but creates ambiguity. The ratchet policy provides a structured path to eliminate this ambiguity.

---

## 3. Ratchet Configuration

Configuration is stored in `config/shadow_gate_registry.yaml`:

```yaml
ratchet:
  legacy_warn_budget: 500    # Maximum allowed suppressed warnings
  enforce_budget: false      # Whether to fail when budget exceeded
  ratchet_phase: 1           # Current phase (1, 2, or 3)
  last_updated: "2025-12-17"
  next_review: "2026-03-31"
```

### 3.1 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `legacy_warn_budget` | int | Maximum allowed suppressed warnings before failure |
| `enforce_budget` | bool | If true, gate fails when budget exceeded |
| `ratchet_phase` | int | Current policy phase (1-3) |
| `last_updated` | date | Last configuration change |
| `next_review` | date | Scheduled review date |

---

## 4. Three-Step Ratchet Policy

### Phase 1: Observation (Current)
**Budget**: 500
**Enforcement**: Disabled
**Timeline**: Q4 2025 - Q1 2026

**Actions**:
- Monitor legacy warning count in CI reports
- Identify high-volume files for remediation
- Track trend: warnings should decrease as new docs use explicit qualifiers
- No CI failures for budget exceeded

**Success Criteria**:
- Legacy warnings stable or decreasing
- All new docs use explicit SHADOW-OBSERVE or SHADOW-GATED

---

### Phase 2: Reduction
**Budget**: 200
**Enforcement**: Enabled
**Timeline**: Q2 2026

**Actions**:
- Reduce budget from 500 to 200
- Enable enforcement (`enforce_budget: true`)
- Fix high-volume legacy files to reduce warning count
- CI fails if legacy warnings exceed 200

**Success Criteria**:
- Legacy warnings below 200
- No regressions in warning count

---

### Phase 3: Freeze
**Budget**: 0
**Enforcement**: Enabled
**Timeline**: Q3 2026

**Actions**:
- Reduce budget to 0
- All documents must use explicit qualifiers
- Legacy allowlist paths enforced strictly
- No new unqualified SHADOW MODE usage permitted

**Success Criteria**:
- Zero legacy warnings suppressed
- All SHADOW MODE references are qualified

---

## 5. CLI Options

The shadow release gate CLI supports ratchet configuration:

```bash
# Default scan (uses config values)
python -m backend.health.shadow_release_gate --scan-dir docs/

# Override budget for testing
python -m backend.health.shadow_release_gate --scan-dir docs/ --warn-budget 100

# Enable enforcement (fail if budget exceeded)
python -m backend.health.shadow_release_gate --scan-dir docs/ --enforce-budget

# Combined (test with low budget + enforcement)
python -m backend.health.shadow_release_gate --scan-dir docs/ --warn-budget 100 --enforce-budget

# Strict mode (disable legacy allowlist entirely)
python -m backend.health.shadow_release_gate --scan-dir docs/ --strict
```

---

## 6. Forward-Only Constraint

**CRITICAL**: The `legacy_warn_budget` value can ONLY be reduced over time.

| Transition | Allowed |
|------------|---------|
| 500 -> 200 | YES |
| 200 -> 0   | YES |
| 0 -> 100   | **NO** |
| 200 -> 300 | **NO** |

Any attempt to increase the budget requires STRATCOM authorization and formal exception process.

---

## 7. Legacy Allowlist Paths

The following paths have legacy allowlist treatment:

- `docs/system_law/`
- `docs/calibration/`
- `docs/governance/`

Files in these paths have UNQUALIFIED_SHADOW_MODE warnings **suppressed** (not emitted), counting against the budget instead.

**Important**: ERROR-level violations (prohibited phrases, missing gate registry) are **never** suppressed, regardless of path.

---

## 8. Transition Checklist

### Phase 1 -> Phase 2 Checklist:
- [ ] Legacy warnings stable below 200 for 30 days
- [ ] High-volume files remediated
- [ ] Team notified of enforcement activation
- [ ] `legacy_warn_budget` reduced to 200
- [ ] `enforce_budget` set to true
- [ ] `ratchet_phase` updated to 2

### Phase 2 -> Phase 3 Checklist:
- [ ] Legacy warnings at zero for 30 days
- [ ] All legacy files remediated
- [ ] Team notified of full enforcement
- [ ] `legacy_warn_budget` reduced to 0
- [ ] `ratchet_phase` updated to 3

---

## 9. Exception Process

If budget increase is required (emergency only):

1. Submit exception request to SMGA
2. Document justification and root cause
3. Provide remediation plan with timeline
4. Obtain STRATCOM approval
5. Update this document with exception record
6. Re-reduce budget within 30 days

---

## 10. Monitoring

Track these metrics:

| Metric | Location | Target |
|--------|----------|--------|
| `legacy_warnings_suppressed` | CI report | Decreasing |
| `budget_within_limit` | CI report | `true` |
| `violations` | CI report | 0 |
| `ratchet_phase` | Config | Advancing |

---

## 11. Changelog

| Date | Phase | Budget | Change |
|------|-------|--------|--------|
| 2025-12-17 | 1 | 500 | Initial policy; observation mode |

---

**SHADOW MODE**: SHADOW-GATED
**Gate Registry**: SRG-002 (shadow_qualification_check)
