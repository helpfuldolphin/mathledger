# External Audit Artifacts

This folder contains all formal external audits produced by hostile/cold-start evaluator roles.

## Contents

### Closure Matrices (Authoritative Summaries)
- `CLOSURE_MATRIX_v*.md` — Authoritative records of all BLOCKING, MAJOR, and MINOR findings with evidence of resolution

### Gate 2 Audits (Manus Cold-Start)
- `manus_gate2_cold_start_audit_*.md` — Formal cold-start evaluations
- Both PASS and FAIL outcomes preserved

### Gate 3 Audits (Claude Chrome Runtime)
- `claude_chrome_gate3_runtime_audit_*.md` — Formal runtime verification audits
- Both PASS and FAIL outcomes preserved

### Other Formal External Audits
- `manus_hostile_audit_*.md` — Hostile audit evaluations
- `manus_epistemic_*.md` — Epistemic integrity/coherence audits
- `manus_site_audit_*.md` — Site-level audits
- `claude_extension_hostile_audit_*.md` — Extension hostile audits

## Classification Rule

Per the [Audit Artifact Taxonomy](../AUDIT_TAXONOMY.md):

A file belongs in `external_audits/` if:
1. It was produced by a hostile/cold-start/external evaluator role
2. It is a formal Gate 2 or Gate 3 audit
3. It is a closure matrix

**Key invariant:** External FAILs are never hidden. All FAIL audits remain here permanently.

## Evaluator Guidance

**Primary evaluator path:** Use `CLOSURE_MATRIX_*.md` for authoritative status.

**Full audit trail:** Individual audit files provide complete provenance, including:
- Multiple FAILs before eventual PASS
- Transient failures (cached content, CDN issues)
- The full path to closure

## Naming Convention

```
{auditor}_{gate}_audit_{date}_{version}[_{outcome}].md
```

Examples:
- `manus_gate2_cold_start_audit_2026-01-04_v0.2.9_FAIL.md`
- `claude_chrome_gate3_runtime_audit_2026-01-05_v0.2.11_PASS.md`
