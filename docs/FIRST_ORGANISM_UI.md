# FIRST ORGANISM UI–Attestation Handshake

## Overview
`Dashboard.tsx` is the observatory for Operation: FIRST ORGANISM. It renders heartbeat metrics, recent statements, proof attempts, DAG neighbors, and the dual-attestation bundle `(Rₜ, Uₜ, Hₜ)` that the backend seals once a derivation run completes. The UI is intentionally deterministic: every color, tooltip, and truncation is specified so the console stays reproducible across reloads.

## UI Event Recording
- UI interactions are emitted through `postUiEvent()` (`apps/ui/src/lib/api.ts`) and posted to `POST /attestation/ui-event`.
- Each payload includes `event_type`, `timestamp`, and optional contextual metadata (`statement_hash`, `selection`, etc.).
- The backend writes every payload via `backend.ledger.ui_events`, making the event leaves part of the `Uₜ` root for dual attestation.
- The dashboard emits:
  1. `dashboard_mount` during `useEffect`.
  2. `refresh_statements` when the statements list is refreshed.
  3. `select_statement` when a user taps a statement or parent link.
 These discrete events are the front-line senses for the organism.

## Dual-Root Display Contracts
- The UI expects the backend JSON endpoints (`/attestation/latest`, `/ui/statement/{hash}.json`, `/ui/parents/{hash}.json`, `/ui/proofs/{hash}.json`) to obey the Pydantic models in `backend/api/schemas.py`. Our client-side types mirror those models exactly (proof atoms include `method`, `status`, `success`, `prover`, `duration_ms`, `created_at`).
- The attestation panel renders:
  - **Rₜ** (`reasoningMerkleRoot`): computed over proof attempts in the latest block (abstentions + successes).
  - **Uₜ** (`uiMerkleRoot`): computed over recorded UI event leaves (including `dashboard_mount`, `select_statement`, etc.).
  - **Hₜ** (`composite_attestation_root`): SHA256 of `Rₜ ∥ Uₜ`.
  Each value is truncated via the `shortHash(..., 12, 8)` helper to maintain readability while reflecting the exact root provided by the backend.
- Additional metadata such as `attestation_version`, event counts, and sample leaves are used to surface determinism metrics and symbolic pain mass counts.

## Integration Footprint
1. The integration test (`tests/integration/test_first_organism_closed_loop`) posts a synthetic UI event → runs the derivation pipeline → causes a Lean abstention.
2. `LedgerIngestor` seals a block that includes both the event root and the reasoning root.
3. `/attestation/latest` serves the sealed triple alongside sample leaves; the dashboard auto-refreshes via `fetchLatestAttestation()` and displays the same Hₜ/Rₜ/Uₜ tuple to the operator.
4. The UI test fixture `apps/ui/test/fixtures/first_organism_attestation.json` mirrors the canonical bundle for front-end regression tests.

## Observability Notes
- The console logs each cached statement detail and proof attempt against `StatementDetailResponse`.
- Because the UI events are logged as first-class leaves, any discrepancy between the frontend `postUiEvent` payloads and the recorded `Uₜ` root will surface immediately when the attestation panel re-renders.

---

## Keeping UI Fixtures in Sync with Backend Attestation

The fixture at `apps/ui/test/fixtures/first_organism_attestation.json` should reflect the exact attestation artifact produced by the integration test. After a successful `test_first_organism_closed_loop` run:

1. Locate `artifacts/first_organism/attestation.json` (written by `_write_attestation_artifact`).
2. Copy or transform the artifact into the UI fixture format:
   - `blockNumber` → `block_id`
   - `blockHash` → `timestamp` (block hash string)
   - `reasoningMerkleRoot` → `reasoning_root`
   - `uiMerkleRoot` → `ui_root`
   - `compositeAttestationRoot` → `composite_root`
   - Leaf samples can be extracted from the metadata or from the integration test's sealed block.
3. Alternatively, run the sync script (if present):

   ```bash
   npm run sync:attestation-fixture
   ```

   This script reads the latest artifact and regenerates the fixture.

When the fixture is updated, the dashboard tests (`Dashboard.test.tsx`) will assert:
- The truncated `Rₜ`, `Uₜ`, and `Hₜ` values rendered in the attestation panel match the fixture.
- Every UI event (`dashboard_mount`, `select_statement`, `refresh_statements`) triggers `postUiEvent` with payloads compatible with `backend/ledger/ui_events`.

## Running UI Tests as Part of FO Verification

After a backend integration run, verify the UI layer:

```bash
cd apps/ui
npm install          # if not already installed
npm run test:ui      # runs vitest against Dashboard.test.tsx
```

The test suite:
- Mocks API calls and injects the attestation fixture.
- Asserts that the dashboard renders the correct `Rₜ`, `Uₜ`, `Hₜ` (truncated via `shortHash(12, 8)`).
- Validates that `postUiEvent` is called with backend-compatible payloads on:
  - `dashboard_mount` (initial render)
  - `select_statement` (statement click, includes `statement_hash`)
  - `refresh_statements` (refresh button, includes `statement_count`)
- Confirms payload structure matches `backend/ledger/ui_events` expectations (`event_type`, `timestamp`, optional `statement_hash`).

**Note**: The DAG rendering test (`constructs DAG nodes matching statement and parent hashes`) is skipped in the vitest suite because `GraphCanvas` uses dynamic import (`await import("cytoscape")`) which is difficult to mock in JSDOM. DAG rendering is validated via Playwright E2E tests instead.

## E2E Tests (Playwright)

For full browser testing including DAG rendering and MDAP loop validation:

```bash
cd apps/ui
npm run test:e2e        # Run headless
npm run test:e2e:ui     # Run with Playwright UI
```

The E2E suite includes:
- `dag-rendering.spec.ts`: Validates Cytoscape graph container, node clicks, resize handling
- `attestation-panel.spec.ts`: Validates R_t/U_t/H_t display, truncation format, determinism
- `mdap-loop.spec.ts`: Validates UI event → backend → attestation → UI display loop

Include both `npm run test:ui` and `npm run test:e2e` in the CI pipeline after the backend integration tests pass to ensure end-to-end reflexivity.

---

In short, the UI is a deterministic mirror of the backend's dual-root sealing, ensuring the organism's heartbeat and symbolic pain mass are verifiable across both layers.
