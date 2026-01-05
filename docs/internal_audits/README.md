# Raw Audit Artifacts

This folder contains raw audit transcripts, intermediate FAILs, and debugging artifacts preserved for completeness.

**Authoritative audit conclusions are recorded in `docs/external_audits/CLOSURE_MATRIX_*.md`.**

## Contents

- Gate 2 (Manus Cold-Start) raw audits and transient FAILs
- Gate 3 (Claude Chrome Runtime) raw audits and transient FAILs
- Response documents and investigation notes
- Epistemic coherence audits and hostile audit transcripts

## Usage

These files exist to support closure matrices, not replace them. They are:

- **Preserved**: Full audit history maintained in the public repository
- **Immutable**: Contents not modified after creation
- **Referenceable**: Can be cited by closure matrices
- **Non-primary**: Not linked from the evaluator path or website navigation

## Evaluator Guidance

Evaluators should use closure matrices (`CLOSURE_MATRIX_*.md`) for authoritative status. Raw transcripts here provide full provenance but contain transient failures, debug notes, and process artifacts that do not reflect final closed state.
