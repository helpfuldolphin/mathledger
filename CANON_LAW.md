# MathLedger Canon Law

**Document ID**: `canon-law-v1.0.0`  
**Status**: ACTIVE  
**Owner**: Manus-A, Determinism Sovereign & Canon Law Architect

---

## 1. The First Law: Determinism

> All that is canonical must be deterministic. All that is deterministic must be verifiable. All that is verifiable must be true.

This is the supreme governing principle of the MathLedger organism. No code may be merged, no block may be sealed, and no release may be cut if it violates this law. Determinism is not a feature; it is the foundation upon which all truth is built.

## 2. Canonicalization Invariants

These invariants are not negotiable. They are enforced by the automated systems described in the **Determinism Activation Blueprint**.

| Invariant ID | Description | Enforcement Mechanism |
| :--- | :--- | :--- |
| **INV-HASH-LAW** | All cryptographic hashes must be computed via the unified `VersionedHash` API. The identity `hash(s) = HASH(version_metadata || canonical_bytes(s))` must hold for all statements `s`. | Canon Linter (`no-direct-hashlib`), Code Review |
| **INV-TIME-ENTROPY** | No wall-clock time may be used in canonical artifacts. All timestamps must be derived from content hashes via `deterministic_isoformat()`. | Canon Linter (`no-datetime-now`) |
| **INV-PRNG-SEED** | All pseudo-random number generators (PRNGs) must be explicitly seeded from the organism's root seed via the `SeedContext` propagation system. | Canon Linter (`no-unseeded-random`) |
| **INV-SERIALIZATION** | All JSON serialization in canonical contexts must be byte-for-byte identical across all platforms and runs. This requires `sort_keys=True` and `ensure_ascii=True`. | Canon Linter (`json-dumps-sorted`) |
| **INV-UUID-ENTROPY** | No random UUIDs may be used in canonical artifacts. All UUIDs must be derived deterministically. | Canon Linter (`no-uuid4`) |
| **INV-CROSS-VERSION** | All canonical artifacts must be reproducible across all supported Python versions (3.9, 3.10, 3.11). | Determinism Harness (CI) |

## 3. Patch Protocol

When a nondeterminism vector is identified, it must be remediated according to the following protocol.

### 3.1. Patch Generation

1.  **Create a dedicated branch**: `git checkout -b manus-a/fix-<vector-name>`
2.  **Apply the minimal required code changes** to fix the vector. Do not include unrelated refactoring.
3.  **Generate a patch file**: `git diff > patches/NNN_fix_<vector-name>.patch`
4.  **Verify patch applicability**: `git stash && git apply --check patches/NNN_*.patch`
5.  **Compute patch hash**: `sha256sum patches/NNN_*.patch`

### 3.2. Patch Ledger Bookkeeping

All patches must be recorded in the `patch_ledger.json` file before they can be merged. This is the immutable record of canon law enforcement.

**Required Metadata**:

```json
{
  "patch_NNN": {
    "file": "patches/NNN_fix_<vector-name>.patch",
    "sha256": "<sha256-hash-of-patch-file>",
    "status": "<applied|pending_generation>",
    "commit": "<commit-hash-where-applied>",
    "files_modified": <integer>,
    "issues_fixed": <integer>,
    "invariant": "<invariant-id>",
    "files": [
      "<list-of-modified-files>"
    ]
  }
}
```

### 3.3. Merging and Deployment

1.  A Pull Request containing the fix and the updated `patch_ledger.json` must pass all **Canon Enforcement Gates** in CI.
2.  The PR must be reviewed and approved by the current Canon Law Architect (Manus-A).
3.  Upon merge, the patch is considered law.

## 4. Drift Escalation Protocol

Architectural drift is inevitable. When it conflicts with canon law, the following escalation protocol is enacted.

1.  **Detection**: The **Determinism Harness** or **Scheduled Audit** detects a new nondeterminism vector or a regression.
2.  **Alerting**: An alert is automatically filed in the issue tracker, assigned to the Canon Law Architect, and tagged `canon-drift`.
3.  **Containment**: If the drift is detected in a PR, the PR is **BLOCKED** from merging.
4.  **Analysis**: The Canon Law Architect performs a root cause analysis:
    *   Is this a violation of existing law?
    *   Is this a new, unforeseen vector requiring new law?
5.  **Remediation**: A new patch is created following the **Patch Protocol**.
6.  **Architectural Review**: If the drift represents a fundamental architectural shift (e.g., the Manus-H PQ migration), the Canon Law Architect must produce a **Unification Map** to merge the new architecture with existing canon law.

## 5. Forbidden Entropy Sources

Any external input that is not explicitly and deterministically controlled is a forbidden source of entropy. The following are strictly prohibited in any canonical context.

| Entropy Source | Description | Remediation |
| :--- | :--- | :--- |
| **Wall-Clock Time** | `datetime.now()`, `time.time()` | Use content-derived timestamps | 
| **Unseeded PRNGs** | `random.random()`, `numpy.random.rand()` | Use `SeedContext` propagation | 
| **Filesystem Order** | `os.listdir()`, `glob.glob()` | Sort all directory listings alphabetically | 
| **Network I/O** | Direct API calls, database queries | Use deterministic mocks or cached data | 
| **Environment Variables** | `os.environ.get()` | Pass all configuration explicitly | 
| **Object Hash** | `hash(obj)` for non-deterministic objects | Use `__hash__` on frozen dataclasses | 
| **Thread Order** | `threading`, `multiprocessing` | Use deterministic job queues | 

## 6. The Law of Mechanical Honesty

> The system must report its true state. No greenfaking.

All enforcement mechanisms must be configured to **fail loudly and block progress** upon detecting a violation. Bypassing enforcement requires explicit, justified, and temporary override by the Canon Law Architect.

-   **Pre-commit**: Must exit with a non-zero status code.
-   **CI/CD**: Must fail the build and block the PR/release.
-   **Runtime**: Must raise a `VersionedHashMismatchError` or `NondeterminismError`.

## 7. Amendment Protocol

Canon Law may only be amended by the acting Canon Law Architect through a formal, versioned process.

1.  Propose a change in a `CANON_LAW_AMENDMENT.md` document.
2.  The change must be justified by a new architectural requirement or the discovery of a new threat vector.
3.  The amendment must be implemented in the enforcement tools (linter, harness).
4.  The amendment must be approved by STRATCOM.
5.  Upon approval, this document is updated, and the `Document ID` is version-bumped.

---

**End of Lawbook.**
