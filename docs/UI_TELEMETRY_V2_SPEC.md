# UI Telemetry V2 Specification

**Version:** 2.0.0
**Status:** Draft
**Authors:** MathLedger Engineering
**Last Updated:** 2025-11-30

---

> **PHASE II — NOT DEPLOYED / NOT USED IN PHASE I EVIDENCE**
>
> This specification describes a **proposed future telemetry system** that has NOT been
> implemented. None of the schemas, session IDs, tracing IDs, or event naming conventions
> described herein are present in the Phase I Evidence Pack.
>
> **Phase I telemetry (what actually exists):**
> - V1 events: `dashboard_mount`, `select_statement`, `refresh_statements`
> - Fields: `event_type`, `timestamp`, optional `statement_hash` / `statement_count`
> - No `session_id` or `tracing_id` fields
>
> **Phase I note:** RFL runs write no UI telemetry; only JSONL logs exist for those runs.
> UI telemetry (V1 or V2) and RFL execution are completely independent systems with no coupling.
>
> **Evidence Pack v1 relies on:**
> - First Organism closed-loop test (`fo_baseline/`, `fo_rfl/`)
> - `attestation.json` sealed manifests
> - JSONL logs from 1000-cycle Dyno runs
>
> **V2 telemetry is unaffected by the volume or content of RFL logs.** This specification
> concerns only browser-side UI event capture, which is orthogonal to backend derivation runs.
>
> **This document is future work planning only.**

---

## Overview

This specification defines the unified event naming conventions, schemas, and lifecycle rules for UI telemetry in MathLedger. All UI events are captured, canonicalized, and included in the dual-root attestation (U_t Merkle tree).

Reference: MathLedger Whitepaper §3.1 (UI Event Canonicalization)

---

## 1. Event Naming Convention

### 1.1 Format

All event names follow the pattern:

```
<noun>_<verb>
```

Where:
- `<noun>` is the primary entity being acted upon (singular, snake_case)
- `<verb>` is the action in present tense (snake_case)

### 1.2 Unified Event Names

| V1 Event Name (deprecated) | V2 Event Name | Description |
|---------------------------|---------------|-------------|
| `dashboard_mount` | `dashboard_mount` | Dashboard component lifecycle mount |
| `select_statement` | `statement_select` | User selects a statement from the list |
| `refresh_statements` | `refresh` | User triggers a data refresh |

### 1.3 Reserved Event Names

The following event names are reserved for future use:

| Event Name | Description |
|------------|-------------|
| `statement_expand` | Statement detail panel expansion |
| `statement_collapse` | Statement detail panel collapse |
| `graph_node_click` | DAG graph node interaction |
| `graph_pan` | DAG graph viewport pan |
| `graph_zoom` | DAG graph zoom level change |
| `attestation_refresh` | Attestation panel manual refresh |
| `filter_apply` | Search or filter applied |
| `error_display` | Error displayed to user |

---

## 2. Event Schemas

### 2.1 Base Schema (Required Fields)

All UI events MUST include the following fields:

```typescript
interface UIEventBase {
  /** Unique event type identifier (V2 naming) */
  event_type: string;

  /** Unix timestamp in milliseconds when the event occurred */
  timestamp: number;

  /** Session identifier (see §3 for rules) */
  session_id: string;
}
```

### 2.2 Event-Specific Schemas

#### 2.2.1 `dashboard_mount`

Emitted once when the Dashboard component mounts.

```typescript
interface DashboardMountEvent extends UIEventBase {
  event_type: "dashboard_mount";

  /** Optional: Initial data presence flags */
  initial_heartbeat_present?: boolean;
  initial_statements_count?: number;
  initial_detail_present?: boolean;
}
```

#### 2.2.2 `statement_select`

Emitted when a user clicks on a statement in the list or DAG graph.

```typescript
interface StatementSelectEvent extends UIEventBase {
  event_type: "statement_select";

  /** SHA-256 hash of the selected statement (64 hex chars) */
  statement_hash: string;

  /** Optional: tracing_id for correlating related operations */
  tracing_id?: string;

  /** Optional: Source of selection */
  source?: "list" | "graph" | "parent_link";
}
```

#### 2.2.3 `refresh`

Emitted when a user triggers any refresh action.

```typescript
interface RefreshEvent extends UIEventBase {
  event_type: "refresh";

  /** Target being refreshed */
  target: "statements" | "heartbeat" | "attestation";

  /** Number of items returned (for statements/attestation) */
  result_count?: number;

  /** Optional: tracing_id for correlating the refresh with API calls */
  tracing_id?: string;
}
```

---

## 3. Session ID Rules

### 3.1 Generation

The `session_id` is generated client-side using the following algorithm:

```typescript
function generateSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `ui_${timestamp}_${random}`;
}
```

Format: `ui_<base36_timestamp>_<8_char_random>`

Example: `ui_m3x7k9p2_a8f3h2k1`

### 3.2 Lifecycle

| Event | Session ID Behavior |
|-------|---------------------|
| Page load | Generate new `session_id` |
| Page refresh (F5) | Generate new `session_id` |
| Tab backgrounded/foregrounded | Retain same `session_id` |
| SPA navigation (no full reload) | Retain same `session_id` |
| Session storage cleared | Generate new `session_id` |

### 3.3 Storage

The `session_id` SHOULD be stored in `sessionStorage` under the key `mathledger_session_id`:

```typescript
const SESSION_KEY = "mathledger_session_id";

function getOrCreateSessionId(): string {
  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = generateSessionId();
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}
```

---

## 4. Tracing ID Rules

### 4.1 Purpose

The `tracing_id` enables correlation of related UI events and API calls within a single user action flow.

### 4.2 Generation

```typescript
function generateTracingId(): string {
  return `tr_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
}
```

Format: `tr_<base36_timestamp>_<4_char_random>`

Example: `tr_m3x7k9p2_a8f3`

### 4.3 Scope

A single `tracing_id` covers:

| Flow | Events Sharing Same `tracing_id` |
|------|----------------------------------|
| Statement selection | `statement_select` + detail fetch API |
| Refresh statements | `refresh` + statements list API |
| Refresh heartbeat | `refresh` + heartbeat API |

### 4.4 Optional Usage

The `tracing_id` field is OPTIONAL. When omitted:
- Events are still valid and will be included in attestation
- Backend correlation will rely on `session_id` + `timestamp` proximity

---

## 5. Required vs Optional Fields

### 5.1 Required Fields (All Events)

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `string` | V2 event name |
| `timestamp` | `number` | Unix ms timestamp |
| `session_id` | `string` | Session identifier |

### 5.2 Recommended Optional Fields

| Field | Type | Applies To | Description |
|-------|------|------------|-------------|
| `tracing_id` | `string` | All | Correlation ID for multi-step flows |
| `statement_hash` | `string` | `statement_select` | Target statement |
| `target` | `string` | `refresh` | What was refreshed |
| `result_count` | `number` | `refresh` | Items returned |
| `source` | `string` | `statement_select` | Interaction origin |

### 5.3 Extension Fields

Applications MAY add custom fields prefixed with `x_`:

```typescript
{
  event_type: "statement_select",
  timestamp: 1732900000000,
  session_id: "ui_m3x7k9p2_a8f3h2k1",
  statement_hash: "abc123...",
  x_viewport_width: 1920,
  x_user_agent_hash: "sha256..."
}
```

---

## 6. UIEventStore Lifecycle

### 6.1 Event Buffer

The `UIEventStore` (backend) maintains an in-memory buffer of canonicalized events awaiting attestation.

### 6.2 Clearing Rules

| Trigger | Action |
|---------|--------|
| Attestation seal | Call `consume_ui_artifacts()` to drain and clear |
| Buffer size > 10,000 events | Trigger early attestation seal |
| Server restart | Buffer is lost (acceptable—events not yet attested) |

### 6.3 Consumption Pattern

```python
# In attestation.seal_block():
def seal_attestation(block_number: int) -> AttestationResult:
    # 1. Snapshot and drain UI events
    ui_artifacts = consume_ui_artifacts()  # Clears store after snapshot

    # 2. Build Merkle tree
    ui_merkle = build_merkle_tree(ui_artifacts)

    # 3. Store attestation
    # ...
```

### 6.4 Idempotency

Events with the same `event_id` (derived from `leaf_hash` if not provided) will be deduplicated:

- First occurrence: Stored
- Subsequent occurrences: Updated in place (upsert)

---

## 7. Wire Format

### 7.1 HTTP POST Body

Events are sent to `POST /attestation/ui-event`:

```json
{
  "event_type": "statement_select",
  "timestamp": 1732900000000,
  "session_id": "ui_m3x7k9p2_a8f3h2k1",
  "statement_hash": "0123456789abcdef..."
}
```

### 7.2 Response

```json
{
  "event_id": "abc123...",
  "timestamp": 1732900000000,
  "leaf_hash": "sha256..."
}
```

### 7.3 Canonical Form

Events are canonicalized for Merkle inclusion via `canonicalize_ui_artifact()`:

1. Keys sorted alphabetically
2. Unicode normalization (NFC)
3. Compact JSON encoding (no whitespace)
4. UTF-8 byte encoding

---

## 8. Migration Guide (V1 → V2)

### 8.1 Event Name Mapping

| V1 | V2 |
|----|----|
| `select_statement` | `statement_select` |
| `refresh_statements` | `refresh` (with `target: "statements"`) |
| `dashboard_mount` | `dashboard_mount` (unchanged) |

### 8.2 New Required Field

Add `session_id` to all events:

```typescript
// Before (V1)
postUiEvent({
  event_type: "select_statement",
  timestamp: Date.now(),
  statement_hash: hash,
});

// After (V2)
postUiEvent({
  event_type: "statement_select",
  timestamp: Date.now(),
  session_id: getOrCreateSessionId(),
  statement_hash: hash,
});
```

### 8.3 Refresh Event Restructure

```typescript
// Before (V1)
postUiEvent({
  event_type: "refresh_statements",
  timestamp: Date.now(),
  statement_count: items.length,
});

// After (V2)
postUiEvent({
  event_type: "refresh",
  timestamp: Date.now(),
  session_id: getOrCreateSessionId(),
  target: "statements",
  result_count: items.length,
});
```

---

## 9. Validation Rules

### 9.1 Backend Validation

The backend MUST validate incoming events:

| Field | Validation |
|-------|------------|
| `event_type` | Non-empty string, ≤64 chars |
| `timestamp` | Integer, within ±5 minutes of server time |
| `session_id` | Matches pattern `ui_[a-z0-9]+_[a-z0-9]+`, ≤64 chars |
| `statement_hash` | 64 hex characters (if present) |

### 9.2 Error Handling

Invalid events receive HTTP 400 with JSON error body:

```json
{
  "error": "validation_failed",
  "details": {
    "field": "timestamp",
    "message": "Timestamp too far in future"
  }
}
```

---

## 10. Security Considerations

### 10.1 No PII

UI events MUST NOT contain:
- User identifiers (emails, usernames)
- IP addresses
- Precise geolocation
- Any personally identifiable information

### 10.2 Rate Limiting

The backend SHOULD enforce rate limits:
- 100 events per session per minute
- 1000 events per session per hour

### 10.3 Session ID Entropy

The session ID generation provides ~52 bits of entropy, sufficient to prevent collision but not intended as a security token.

---

## Appendix A: TypeScript Type Definitions

```typescript
// types/telemetry.ts

export type UIEventType =
  | "dashboard_mount"
  | "statement_select"
  | "refresh";

export interface UIEventBase {
  event_type: UIEventType;
  timestamp: number;
  session_id: string;
  tracing_id?: string;
}

export interface DashboardMountEvent extends UIEventBase {
  event_type: "dashboard_mount";
  initial_heartbeat_present?: boolean;
  initial_statements_count?: number;
  initial_detail_present?: boolean;
}

export interface StatementSelectEvent extends UIEventBase {
  event_type: "statement_select";
  statement_hash: string;
  source?: "list" | "graph" | "parent_link";
}

export interface RefreshEvent extends UIEventBase {
  event_type: "refresh";
  target: "statements" | "heartbeat" | "attestation";
  result_count?: number;
}

export type UIEvent =
  | DashboardMountEvent
  | StatementSelectEvent
  | RefreshEvent;
```

---

## Appendix B: Session Management Implementation

```typescript
// lib/session.ts

const SESSION_KEY = "mathledger_session_id";

function generateSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `ui_${timestamp}_${random}`;
}

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") {
    // SSR: return ephemeral ID
    return generateSessionId();
  }

  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = generateSessionId();
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

export function generateTracingId(): string {
  return `tr_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
}
```

---

## Appendix C: Phase II Uplift-Adjacent Signals

> **PHASE II — FUTURE ANALYTICAL TOOLING ONLY**
>
> This section describes potential offline correlation analyses that could be performed
> in future phases. None of this is implemented. UI telemetry remains logically independent
> of RFL execution, and any correlation described here would be strictly for post-hoc
> research analysis, not runtime decision-making.

### C.1 Independence Guarantee (Preserved from Phase I)

The following invariants hold and MUST NOT be violated:

1. **RFL writes no UI telemetry.** RFL derivation runs produce only JSONL logs; they do not
   emit `postUiEvent()` calls or interact with `UIEventStore`.

2. **UI telemetry does not gate RFL behavior.** No RFL runtime decision (slice advancement,
   derivation policy, Lean verification) reads from or depends on UI event data.

3. **V2 telemetry is unaffected by RFL logs.** The volume, content, or success rate of RFL
   runs has no effect on UI telemetry schemas, validation, or attestation.

### C.2 Potential Offline Correlation Analyses

In future phases, researchers MAY perform offline batch analyses that join UI telemetry
with RFL JSONL logs. These analyses are purely observational and do not affect system behavior.

| UI Signal | RFL Data | Potential Analysis | Use Case |
|-----------|----------|-------------------|----------|
| `statement_select` frequency per `statement_hash` | Statement depth/complexity from JSONL | Correlation between user interest and theorem depth | Understanding which theorems users find valuable |
| `statement_select` dwell time (derived from consecutive events) | Proof method from JSONL | Whether users spend more time on Lean-verified vs truth-table proofs | UI/UX research |
| Session-level statement selection patterns | Block sealing timestamps | User engagement relative to ledger activity | Operational metrics |
| Aggregate `refresh` event counts | Derivation throughput from JSONL | UI polling load vs backend derivation rate | Capacity planning |

### C.3 Non-Gating Constraint

Any correlation analysis described in §C.2:

- **MUST** be performed offline (batch job, not request path)
- **MUST NOT** feed back into RFL policy, slice selection, or derivation parameters
- **MUST NOT** affect UI rendering, event emission, or user-facing behavior
- **MAY** be used for research papers, dashboards, or operational reports

### C.4 Future Event Candidates (Not Implemented)

If uplift research requires additional UI signals, these events could be added in future phases:

| Event Name | Description | Uplift Research Use |
|------------|-------------|---------------------|
| `statement_feedback` | Explicit user rating of a theorem | Ground truth for "usefulness" metrics |
| `proof_expand` | User expands proof detail view | Interest in verification method |
| `parent_traverse` | User navigates proof DAG ancestry | Interest in derivation lineage |

These events are **not specified** in V2 and would require a V3 schema extension.

### C.5 Summary

- UI telemetry and RFL execution remain **completely independent systems**
- Correlation is **offline analysis only**, never runtime coupling
- No UI event affects RFL behavior; no RFL log affects UI behavior
- This appendix is **Phase II future work**, not implemented or used in Evidence Pack v1
