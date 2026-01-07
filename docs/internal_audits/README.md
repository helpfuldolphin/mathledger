# Internal Audit Artifacts

This folder contains operator response documents and investigation notes that support the external audit trail.

**Authoritative audit conclusions are recorded in `docs/external_audits/CLOSURE_MATRIX_*.md`.**

## Contents

- Response documents (`*_response.md`) explaining how audit findings were addressed
- Investigation notes and operator diagnostics

## Classification Rule

Per the [Audit Artifact Taxonomy](../AUDIT_TAXONOMY.md):

- **Internal audits**: Operator artifacts, response documents, scratch work
- **External audits**: Formal Gate 2/Gate 3 audits, hostile audits, closure matrices

Response documents belong here because they are operator artifacts explaining remediation, not formal external audits themselves.

## Usage

These files:

- **Support** closure matrices with remediation context
- **Are preserved** in the public repository for completeness
- **Are not** part of the primary evaluator path
- **Are not** authoritative (closure matrices are authoritative)

## Evaluator Guidance

Evaluators should reference `docs/external_audits/` for:
- Formal Gate 2 and Gate 3 audits (both PASS and FAIL)
- Closure matrices (authoritative summaries)

Response documents here provide remediation context but do not override closure matrix status.
