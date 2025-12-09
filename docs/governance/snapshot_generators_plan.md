# Snapshot Generator Suite: Architecture & Implementation Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **DRAFT**

## 1. Overview

This document specifies the architecture and implementation plan for the **Snapshot Generator Suite**. This suite is a critical piece of infrastructure required to power the Drift Radar system. Its purpose is to produce deterministic, versioned, and canonicalized JSON snapshots of key governance domains from their respective data sources.

This plan defines four new generator scripts, their data sources, extraction logic, and CI integration points. The successful implementation of this suite will unblock the full activation of the drift detection gates.

## 2. Universal Design Principles

All four generator scripts will adhere to the following principles:

- **Location**: All scripts will reside in a new `scripts/generators/` directory.
- **Execution**: Scripts are designed to be executed from the repository root.
- **Output**: All scripts will write the canonicalized JSON snapshot to `stdout`. This allows for easy redirection to files in CI workflows (e.g., `python3 script.py > snapshot.json`).
- **Determinism**: All scripts MUST use the project's official RFC 8785 canonicalization library to ensure byte-for-byte reproducible output.
- **Schema Compliance**: The output of each generator MUST be 100% compliant with its corresponding JSON schema defined in the `/schemas` directory.
- **Error Handling**: Scripts will exit with a non-zero status code if data sources are missing or corrupted.

---

## 3. Generator Specifications

This section details the data sources and extraction logic for each of the four snapshot generators.

### 3.1. Curriculum Snapshot Generator

- **Script**: `scripts/generators/generate_curriculum_snapshot.py`
- **Output Schema**: `schemas/curriculum_snapshot.schema.json`

#### Data Sources

1.  **Problem Definitions**: Markdown files located in `curriculum/problems/**/*.md`.
2.  **Problem Metadata**: YAML frontmatter within each Markdown file.
3.  **Topic Taxonomy**: A YAML file located at `curriculum/topics.yml`.

**Assumed `curriculum/problems/algebra/ALG-001.md` Structure**:

```markdown
---
id: ALG-001
title: "Solving Linear Equations"
topic: algebra
difficulty_score: 0.2
---

## Problem Statement

Solve for x in the equation `2x + 5 = 15`.
```

**Assumed `curriculum/topics.yml` Structure**:

```yaml
algebra:
  - linear-equations
  - quadratic-equations
calculus:
  - differentiation
  - integration
```

#### Extraction Logic

1.  The script will use `glob` to find all `.md` files within `curriculum/problems/`.
2.  For each file, it will parse the YAML frontmatter to extract metadata (`id`, `title`, `topic`, `difficulty_score`).
3.  It will compute the SHA-256 hash of the Markdown content *below* the frontmatter to generate the `content_hash`.
4.  It will parse `curriculum/topics.yml` to build the `topic_taxonomy` object.
5.  Finally, it will assemble the complete snapshot object, validate it against the schema, and print the canonicalized JSON to `stdout`.

### 3.2. Telemetry Schema Snapshot Generator

- **Script**: `scripts/generators/generate_telemetry_snapshot.py`
- **Output Schema**: `schemas/telemetry_schema_snapshot.schema.json`

#### Data Source

- **Event Definitions**: Python classes within the `backend/telemetry/events.py` module that are decorated with a hypothetical `@register_event` decorator.

**Assumed `backend/telemetry/events.py` Structure**:

```python
from some_schema_library import register_event, String, Integer, Boolean

@register_event("user_login_success")
class UserLoginSuccess:
    """Fires when a user successfully logs in."""
    user_id: Integer(description="The ID of the user.")
    session_duration: Integer(description="The session duration in seconds.")

@register_event("problem_attempted")
class ProblemAttempted:
    """Fires when a user attempts a problem."""
    problem_id: String(description="The ID of the problem.")
    correct: Boolean(description="Whether the attempt was correct.")
```

#### Extraction Logic

1.  The script will use `importlib` to dynamically import the `backend.telemetry.events` module.
2.  It will use `inspect` to iterate through the module's members and find classes decorated with `@register_event`.
3.  For each decorated class, it will extract the event name, docstring (as `description`), and property definitions (name, type, description) to construct a JSON Schema object for that event.
4.  It will assemble the complete snapshot of all event schemas and print the canonicalized JSON to `stdout`.

### 3.3. Ledger Snapshot Generator

- **Script**: `scripts/generators/generate_ledger_snapshot.py`
- **Output Schema**: `schemas/ledger_snapshot.schema.json`

#### Data Source

- **Ledger Database**: A SQLite database file located at `artifacts/ledger/mathledger.db`.

**Assumed `blocks` Table Schema**:

| Column      | Type    | Description                               |
|-------------|---------|-------------------------------------------|
| `height`    | INTEGER | Block height (0-indexed), Primary Key     |
| `hash`      | TEXT    | SHA-256 hash of this block                |
| `prev_hash` | TEXT    | SHA-256 hash of the previous block        |
| `merkle_root`| TEXT    | Merkle root of statements in this block   |
| `timestamp` | TEXT    | ISO 8601 timestamp of block creation      |

#### Extraction Logic

1.  The script will use the `sqlite3` module to connect to `artifacts/ledger/mathledger.db`.
2.  It will execute `SELECT * FROM blocks ORDER BY height ASC` to fetch all block data.
3.  It will query for the max height and the hash of the last block.
4.  It will assemble the complete ledger snapshot, including `chain_id`, `height`, `last_block_hash`, and the full list of `blocks`.
5.  The canonicalized JSON will be printed to `stdout`.

### 3.4. Attestation Snapshot Generator

- **Script**: `scripts/generators/generate_attestation_snapshot.py`
- **Output Schema**: `schemas/attestation_snapshot.schema.json`

#### Data Source

- **Attestation History**: A JSONL file located at `artifacts/governance/attestation_history.jsonl`.

**Assumed `attestation_history.jsonl` Structure**:

```json
{"id": "attest-001", "H_t": "...", "R_t": "...", "U_t": "..."}
{"id": "attest-002", "H_t": "...", "R_t": "...", "U_t": "..."}
```

#### Extraction Logic

1.  The script will read the `attestation_history.jsonl` file line by line.
2.  Each line will be parsed as a JSON object and added to the `attestations` list.
3.  The script will assemble the complete snapshot object and print the canonicalized JSON to `stdout`.

---

## 4. CI Integration

These generator scripts are the key to enabling the drift radar CI gates. In each gate workflow, the following steps will occur:

1.  **Generate Current Snapshot**: The appropriate generator script is run, and its output is redirected to a temporary file (e.g., `current_snapshot.json`).

    ```yaml
    - name: Generate Current Curriculum Snapshot
      run: |
        python3 scripts/generators/generate_curriculum_snapshot.py > artifacts/governance/current_curriculum_snapshot.json
    ```

2.  **Fetch Baseline Snapshot**: The baseline version of the snapshot is checked out from the target branch (e.g., `main`).

    ```yaml
    - name: Get Baseline Snapshot
      run: |
        git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/curriculum_snapshot.json > artifacts/governance/baseline_curriculum_snapshot.json
    ```

3.  **Run Drift Radar**: The drift radar script is then invoked with both the baseline and current snapshots to detect changes.

This process ensures that every pull request is compared against the established ground truth from the main branch, fully automated branch, branch, providing robust and automated drift detection.
